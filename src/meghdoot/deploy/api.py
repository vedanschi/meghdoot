"""
api.py – FastAPI Inference Server for Meghdoot-AI
=================================================

Exposes a REST API that accepts a sequence of 3 satellite frames
(as latent tensors or raw images) and returns a 0–3 hour prediction.

Endpoints
---------
    POST /predict          – run inference on uploaded frames
    GET  /health           – health check
    GET  /model/info       – model metadata

Usage
-----
    uvicorn meghdoot.deploy.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import torch
import asyncio
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from meghdoot.models.diffusion import MeghdootDiffusion
from meghdoot.models.vae import SatelliteVAE
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import get_device
from meghdoot.utils.logging import get_logger
from meghdoot.deploy.runtime import (
    ensure_vae_checkpoint,
    ensure_diffusion_checkpoint,
    load_backend_environment,
    prepare_backend_config,
)
from meghdoot.deploy.pipeline import (
    download_forecast_data,
    generate_forecast_sequence,
    prepare_history_frames,
    preprocess_frames,
    publish_to_bucket,
)

log = get_logger(__name__)

# ── App & Global State ─────────────────────────────
app = FastAPI(
    title="Meghdoot-AI",
    description="Latent Diffusion Model for weather nowcasting over the Indian subcontinent",
    version="0.1.0",
)

# Lazy-loaded models (populated on startup)
_state: dict = {}


def _serve_forecast_blob(blob_name: str) -> Response:
    cfg = _state.get("cfg", {})
    bucket_name = cfg.get("deployment", {}).get("gcs_bucket")
    if not bucket_name:
        raise HTTPException(500, "GCS bucket not configured")

    try:
        from google.cloud import storage
    except ImportError as exc:
        raise HTTPException(500, "google-cloud-storage is required for forecast proxy") from exc

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    if not blob.exists():
        raise HTTPException(404, f"Forecast artifact not found: {blob_name}")

    content_type = blob.content_type or (
        "application/json" if blob_name.endswith(".json") else "image/png"
    )
    return Response(
        content=blob.download_as_bytes(),
        media_type=content_type,
    )


# ── Lifespan Events ───────────────────────────────
@app.on_event("startup")
async def startup() -> None:
    """Load config + models into GPU memory once at server start."""
    load_backend_environment()
    _state.pop("startup_error", None)

    try:
        cfg = prepare_backend_config(load_config())
        device = get_device(cfg["project"].get("device", "cuda"))

        log.info("Loading VAE …")
        vae = SatelliteVAE(cfg).to(device).eval()
        try:
            vae_ckpt_path = ensure_vae_checkpoint(cfg)
        except Exception:
            log.exception("Failed to resolve VAE checkpoint; starting with base VAE weights")
            vae_ckpt_path = None

        if vae_ckpt_path is not None:
            vae.load(vae_ckpt_path)
            log.info(f"Loaded VAE checkpoint: {Path(vae_ckpt_path).name}")
        else:
            log.warning("No VAE checkpoint found; backend will use the base VAE weights")

        log.info("Loading Diffusion model …")
        diffusion = MeghdootDiffusion(cfg).to(device).eval()

        try:
            ckpt_path = ensure_diffusion_checkpoint(cfg)
        except Exception:
            log.exception("Failed to resolve diffusion checkpoint; starting with random weights")
            ckpt_path = None

        if ckpt_path is not None:
            diffusion.load(ckpt_path)
            log.info(f"Loaded checkpoint: {Path(ckpt_path).name}")
        else:
            log.warning("No diffusion checkpoint found; backend will start with random weights")

        _state["cfg"] = cfg
        _state["vae"] = vae
        _state["diffusion"] = diffusion
        _state["device"] = device
        log.info("Meghdoot-AI API ready ✓")
    except Exception as exc:
        # Keep the process alive so Cloud Run can expose logs and health endpoints.
        _state["startup_error"] = str(exc)
        log.exception("Startup failed; API will run in degraded mode")


# ── Health Check ──────────────────────────────────
@app.get("/health")
async def health():
    startup_error = _state.get("startup_error")
    return {
        "status": "degraded" if startup_error else "healthy",
        "gpu": torch.cuda.is_available(),
        "startup_error": startup_error,
    }


@app.get("/model/info")
async def model_info():
    cfg = _state.get("cfg", {})
    diff_cfg = cfg.get("diffusion", {})
    return {
        "project": cfg.get("project", {}).get("name"),
        "unet_params_M": round(
            sum(p.numel() for p in _state["diffusion"].unet.parameters()) / 1e6, 1
        ) if "diffusion" in _state else None,
        "num_inference_steps": diff_cfg.get("inference", {}).get("num_inference_steps"),
        "prediction_horizon_hours": diff_cfg.get("inference", {}).get("prediction_horizon_hours"),
        "conditioning_frames": diff_cfg.get("conditioning", {}).get("num_history_frames"),
    }


# ── Prediction Endpoint ──────────────────────────
@app.post("/predict")
async def predict(
    request: Request,
    num_steps: int = 50,
):
    """Run diffusion inference on 3 uploaded history frames.

    Each file should be a NumPy ``.npy`` array of shape ``[H, W]``
    (normalised brightness temperatures in [-1, 1]) or a pre-computed
    latent of shape ``[4, 64, 64]``.

    Returns the predicted frame as a downloadable ``.npy`` file.
    """
    if "diffusion" not in _state:
        raise HTTPException(503, "Model not loaded")

    vae: SatelliteVAE = _state["vae"]
    diffusion: MeghdootDiffusion = _state["diffusion"]
    device = _state["device"]

    try:
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(
                503,
                "Multipart form parsing is unavailable. Ensure python-multipart is installed in the runtime image.",
            ) from exc

        uploads = [form.get("frame1"), form.get("frame2"), form.get("frame3")]
        if any(upload is None for upload in uploads):
            raise HTTPException(400, "Missing required form files: frame1, frame2, frame3")
        typed_uploads: list[StarletteUploadFile] = []
        for upload in uploads:
            if not isinstance(upload, StarletteUploadFile):
                raise HTTPException(400, "frame1/frame2/frame3 must be uploaded files")
            typed_uploads.append(upload)

        frames = []
        for f in typed_uploads:
            content = await f.read()
            arr = np.load(io.BytesIO(content)).astype(np.float32)
            frames.append(arr)

        # Determine if inputs are pixel-space or latent-space
        is_latent = frames[0].ndim == 3 and frames[0].shape[0] == 4

        if is_latent:
            # Already latent: stack directly
            history = torch.from_numpy(np.stack(frames)).unsqueeze(0).to(device)
        else:
            # Pixel-space → encode through VAE
            latents = []
            for arr in frames:
                t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
                z = vae.encode(t)
                latents.append(z.squeeze(0))
            history = torch.stack(latents).unsqueeze(0)  # [1, 3, 4, 64, 64]

        # Run diffusion sampling
        t0 = time.time()
        pred_latent = diffusion.sample(history, num_inference_steps=num_steps)
        elapsed = time.time() - t0

        # Decode to pixel space
        pred_pixel = vae.decode(pred_latent)
        pred_np = pred_pixel[0, 0].cpu().numpy()  # [H, W]

        # Return as .npy download
        buf = io.BytesIO()
        np.save(buf, pred_np)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=prediction.npy",
                "X-Inference-Time-Sec": f"{elapsed:.2f}",
            },
        )

    except Exception as e:
        log.error(f"Prediction failed: {e}")
        raise HTTPException(500, f"Prediction error: {e}")


@app.post("/predict/json")
async def predict_json(
    request: Request,
    num_steps: int = 50,
):
    """Same as /predict but returns summary statistics as JSON
    (useful for the Streamlit dashboard)."""
    if "diffusion" not in _state:
        raise HTTPException(503, "Model not loaded")

    vae: SatelliteVAE = _state["vae"]
    diffusion: MeghdootDiffusion = _state["diffusion"]
    device = _state["device"]

    try:
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(
                503,
                "Multipart form parsing is unavailable. Ensure python-multipart is installed in the runtime image.",
            ) from exc

        uploads = [form.get("frame1"), form.get("frame2"), form.get("frame3")]
        if any(upload is None for upload in uploads):
            raise HTTPException(400, "Missing required form files: frame1, frame2, frame3")
        typed_uploads: list[StarletteUploadFile] = []
        for upload in uploads:
            if not isinstance(upload, StarletteUploadFile):
                raise HTTPException(400, "frame1/frame2/frame3 must be uploaded files")
            typed_uploads.append(upload)

        frames = []
        for f in typed_uploads:
            content = await f.read()
            arr = np.load(io.BytesIO(content)).astype(np.float32)
            frames.append(arr)

        is_latent = frames[0].ndim == 3 and frames[0].shape[0] == 4

        if is_latent:
            history = torch.from_numpy(np.stack(frames)).unsqueeze(0).to(device)
        else:
            latents = []
            for arr in frames:
                t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
                z = vae.encode(t)
                latents.append(z.squeeze(0))
            history = torch.stack(latents).unsqueeze(0)

        t0 = time.time()
        pred_latent = diffusion.sample(history, num_inference_steps=num_steps)
        elapsed = time.time() - t0

        pred_pixel = vae.decode(pred_latent)
        pred_np = pred_pixel[0, 0].cpu().numpy()

        return JSONResponse({
            "shape": list(pred_np.shape),
            "min": float(pred_np.min()),
            "max": float(pred_np.max()),
            "mean": float(pred_np.mean()),
            "std": float(pred_np.std()),
            "inference_time_sec": round(elapsed, 2),
        })

    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")


@app.post("/forecast/nowcast")
async def forecast_nowcast(num_steps: int = 6):
    """Cloud Scheduler endpoint: run full nowcasting pipeline.
    
    This endpoint orchestrates the complete forecast cycle:
    1. Download latest 3 INSAT frames from MOSDAC
    2. Preprocess to tensors
    3. Generate multi-step forecast via diffusion
    4. Publish results to GCS bucket
    
    **Security Note**: Only call this from Cloud Scheduler with authentication.
    In production, add API key validation or restrict to internal GCP networks.
    
    Parameters
    ----------
    num_steps : int
        Number of forecast steps to generate (default 6 = 3 hours)
    
    Returns
    -------
    dict
        Status of forecast cycle with timestamps and bucket location
    """
    if "diffusion" not in _state:
        raise HTTPException(503, "Models not loaded")
    
    # Run the heavy pipeline asynchronously and return immediately.
    # Scheduler has a short HTTP timeout; run the pipeline in background
    # so the scheduler receives a 202 quickly.
    try:
        run_id = uuid.uuid4().hex
        log.info(f"Enqueuing background forecast run {run_id} (num_steps={num_steps})")

        def _run_sync():
            try:
                cfg = _state["cfg"]
                vae = _state["vae"]
                diffusion = _state["diffusion"]
                device = _state["device"]

                min_steps = int(os.environ.get("MEGHDOOT_MIN_FORECAST_STEPS", "6"))
                effective_steps = max(num_steps, min_steps)
                if effective_steps != num_steps:
                    log.warning(
                        "Background run %s: requested num_steps=%s; enforcing min=%s",
                        run_id,
                        num_steps,
                        min_steps,
                    )

                t0 = time.time()

                # Step 1: Download fresh satellite data (MOSDAC-first, cache fallback)
                log.info("Background run %s: Step 1 - Acquiring forecast data...", run_id)
                download_result = download_forecast_data(cfg, n_frames=3)
                if not download_result:
                    log.error("Background run %s: failed to acquire satellite data", run_id)
                    return
                raw_files, observation_time = download_result

                # Step 2: Preprocess raw files to tensors (with cache fallback)
                log.info("Background run %s: Step 2 - Preprocessing to tensors...", run_id)
                processed_files = preprocess_frames(cfg, raw_files)
                if not processed_files:
                    log.warning("Background run %s: preprocessing failed; attempting cached tensors...", run_id)
                    processed_files = prepare_history_frames(cfg, n_frames=3)
                    if not processed_files:
                        log.error("Background run %s: failed to obtain processed tensors", run_id)
                        return

                # Step 3: Generate forecast
                log.info("Background run %s: Step 3 - Generating forecast (%d steps)...", run_id, effective_steps)
                forecast_result = generate_forecast_sequence(
                    cfg, vae, diffusion, processed_files, num_steps=effective_steps, device=device
                )
                if forecast_result is None:
                    log.error("Background run %s: inference failed", run_id)
                    return
                forecast_frames, current_observation, forecast_latents = forecast_result

                # Step 4: Publish to bucket
                log.info("Background run %s: Step 4 - Publishing forecast...", run_id)
                if not publish_to_bucket(
                    cfg,
                    forecast_frames,
                    observation_time=observation_time,
                    current_observation=current_observation,
                    forecast_latents=forecast_latents,
                ):
                    log.error("Background run %s: publishing failed", run_id)
                    return

                total_time = time.time() - t0
                log.info(
                    "Background run %s completed: %d steps in %.2fs ✓",
                    run_id,
                    len(forecast_frames),
                    total_time,
                )
            except Exception:
                log.exception("Background run %s failed", run_id)

        # schedule the synchronous runner in a background thread
        asyncio.create_task(asyncio.to_thread(_run_sync))

        return JSONResponse({"status": "accepted", "run_id": run_id}, status_code=202)

    except Exception:
        log.exception("Failed to enqueue background forecast")
        raise HTTPException(500, "Failed to start forecast job")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500,
        )


@app.get("/forecast/latest/metadata.json")
async def forecast_latest_metadata() -> Response:
    return _serve_forecast_blob("forecasts/latest/metadata.json")


@app.get("/forecast/latest/{filename}")
async def forecast_latest_file(filename: str) -> Response:
    if not filename.startswith("forecast_step_") or not filename.endswith(".png"):
        raise HTTPException(404, "Unknown forecast artifact")
    return _serve_forecast_blob(f"forecasts/latest/{filename}")
