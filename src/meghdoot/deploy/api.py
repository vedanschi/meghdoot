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
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from meghdoot.models.diffusion import MeghdootDiffusion
from meghdoot.models.vae import SatelliteVAE
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import get_device
from meghdoot.utils.logging import get_logger

log = get_logger(__name__)

# ── App & Global State ─────────────────────────────
app = FastAPI(
    title="Meghdoot-AI",
    description="Latent Diffusion Model for weather nowcasting over the Indian subcontinent",
    version="0.1.0",
)

# Lazy-loaded models (populated on startup)
_state: dict = {}


# ── Lifespan Events ───────────────────────────────
@app.on_event("startup")
async def startup() -> None:
    """Load config + models into GPU memory once at server start."""
    cfg = load_config()
    device = get_device(cfg["project"].get("device", "cuda"))

    log.info("Loading VAE …")
    vae = SatelliteVAE(cfg)

    log.info("Loading Diffusion model …")
    diffusion = MeghdootDiffusion(cfg)

    # Try to load the latest checkpoint
    ckpt_dir = Path(cfg["diffusion"]["checkpoint_dir"])
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("diffusion_epoch*.pt"))
        if ckpts:
            diffusion.load(ckpts[-1])
            log.info(f"Loaded checkpoint: {ckpts[-1].name}")

    _state["cfg"] = cfg
    _state["vae"] = vae
    _state["diffusion"] = diffusion
    _state["device"] = device
    log.info("Meghdoot-AI API ready ✓")


# ── Health Check ──────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "gpu": torch.cuda.is_available()}


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
    frame1: UploadFile = File(..., description="History frame 1 (.npy)"),
    frame2: UploadFile = File(..., description="History frame 2 (.npy)"),
    frame3: UploadFile = File(..., description="History frame 3 (.npy)"),
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
        frames = []
        for f in [frame1, frame2, frame3]:
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
    frame1: UploadFile = File(...),
    frame2: UploadFile = File(...),
    frame3: UploadFile = File(...),
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
        frames = []
        for f in [frame1, frame2, frame3]:
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
