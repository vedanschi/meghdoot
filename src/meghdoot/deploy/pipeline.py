"""
pipeline.py – Nowcasting Pipeline Runner
=========================================

Orchestrates the full inference pipeline:
1. Download latest INSAT 3DS frames from MOSDAC
2. Preprocess to tensors
3. Generate multi-step forecast via recursive diffusion rollout
4. Publish results to GCS bucket

Usage
-----
    python -m meghdoot.deploy.pipeline --config configs/default.yaml

Scheduled via cron every 30 minutes:
    0,30 * * * * cd /path/to/meghdoot && python -m meghdoot.deploy.pipeline
"""

from __future__ import annotations

import argparse
import io
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from meghdoot.data.mdapi import MOSDACClient
from meghdoot.data.preprocessing import process_single_file
from meghdoot.models.diffusion import MeghdootDiffusion
from meghdoot.models.vae import SatelliteVAE
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import ensure_dir, get_device
from meghdoot.utils.logging import get_logger
from meghdoot.deploy.runtime import (
    ensure_diffusion_checkpoint,
    ensure_vae_checkpoint,
    load_backend_environment,
    prepare_backend_config,
)

log = get_logger(__name__)


def normalize_to_image(arr: np.ndarray) -> np.ndarray:
    """Convert [-1, 1] normalized array to [0, 255] uint8 for PNG.
    
    Parameters
    ----------
    arr : np.ndarray
        Normalized array in [-1, 1] range, shape [H, W]
    
    Returns
    -------
    np.ndarray
        uint8 array in [0, 255], shape [H, W]
    """
    arr_clipped = np.clip(arr, -1, 1)
    arr_scaled = ((arr_clipped + 1) / 2 * 255).astype(np.uint8)
    return arr_scaled


def array_to_png_bytes(arr: np.ndarray) -> io.BytesIO:
    """Convert array to PNG bytes for upload to GCS.
    
    Parameters
    ----------
    arr : np.ndarray
        Normalized array in [-1, 1]
    
    Returns
    -------
    io.BytesIO
        PNG-encoded bytes
    """
    img_array = normalize_to_image(arr)
    img = Image.fromarray(img_array, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def download_latest_frames(
    cfg: dict,
    n_frames: int = 3,
) -> Optional[list[Path]]:
    """Download latest INSAT frames from MOSDAC.
    
    Parameters
    ----------
    cfg : dict
        Meghdoot config
    n_frames : int
        Number of recent frames to download
    
    Returns
    -------
    list[Path] or None
        Paths to downloaded files, or None if download failed
    """
    try:
        log.info(f"Downloading latest {n_frames} INSAT frames from MOSDAC...")
        
        client = MOSDACClient(cfg)
        if not client.authenticate():
            log.error("MOSDAC authentication failed")
            return None
        
        try:
            data_cfg = cfg.get("data", {})
            dataset_id = (data_cfg.get("dataset_ids") or ["3SIMG_L1C_SGP"])[0]
            date_range = data_cfg.get("date_range", {})
            start_date = date_range.get("start")
            end_date = date_range.get("end")

            log.info(
                "MOSDAC fetch window: dataset_id=%s, start=%s, end=%s",
                dataset_id,
                start_date or "<default>",
                end_date or "<default>",
            )

            client.bulk_download(
                dataset_id=dataset_id,
                start_date=start_date,
                end_date=end_date,
            )
        finally:
            client.logout()
        
        # Find the latest n_frames files
        raw_dir = Path(cfg["data"]["paths"]["raw"])
        files = sorted(
            list(raw_dir.rglob("*.h*5"))
            + list(raw_dir.rglob("*.nc"))
            + list(raw_dir.rglob("*.nc4")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:n_frames]
        
        if len(files) < n_frames:
            log.warning(f"Expected {n_frames} files, found only {len(files)}")
        
        return files if files else None
        
    except Exception as e:
        log.error(f"Download failed: {e}")
        return None


def preprocess_frames(
    cfg: dict,
    raw_files: list[Path],
) -> Optional[list[Path]]:
    """Preprocess raw files to tensors.
    
    Parameters
    ----------
    cfg : dict
        Meghdoot config
    raw_files : list[Path]
        Raw file paths
    
    Returns
    -------
    list[Path] or None
        Paths to processed tensors, or None if failed
    """
    try:
        log.info(f"Preprocessing {len(raw_files)} files...")
        
        region = cfg["data"]["region"]
        target_size = tuple(cfg["data"]["crop_size"])
        processed_dir = Path(cfg["data"]["paths"]["processed"])
        
        processed_files = []
        for fp in raw_files:
            try:
                out_path = process_single_file(fp, region, target_size, processed_dir)
                if out_path:
                    processed_files.append(out_path)
            except Exception as e:
                log.warning(f"Failed to process {fp.name}: {e}")
        
        if not processed_files:
            log.error("No files were successfully preprocessed")
            return None
        
        return sorted(processed_files)
        
    except Exception as e:
        log.error(f"Preprocessing failed: {e}")
        return None


def generate_forecast_sequence(
    cfg: dict,
    vae: SatelliteVAE,
    diffusion: MeghdootDiffusion,
    processed_files: list[Path],
    num_steps: int = 6,
    device: Optional[torch.device] = None,
) -> Optional[list[np.ndarray]]:
    """Generate multi-step forecast via recursive rollout.
    
    Parameters
    ----------
    cfg : dict
        Meghdoot config
    vae : SatelliteVAE
        VAE encoder/decoder
    diffusion : MeghdootDiffusion
        Diffusion model
    processed_files : list[Path]
        Latest 3 preprocessed tensor files
    num_steps : int
        Number of forecast steps to generate
    device : torch.device
        Device for inference
    
    Returns
    -------
    list[np.ndarray] or None
        List of predicted frames (pixel space), or None if failed
    """
    if device is None:
        device = get_device(cfg["project"].get("device", "cuda"))
    
    try:
        log.info(f"Generating {num_steps}-step forecast...")
        
        # Load latest 3 frames
        history_tensors = []
        for fp in sorted(processed_files)[-3:]:
            t = torch.load(fp, map_location=device)  # [2, H, W]
            history_tensors.append(t)
        
        if len(history_tensors) < 3:
            log.error(f"Need 3 history frames, got {len(history_tensors)}")
            return None
        
        # Stack into [1, 3, 2, H, W]
        history_pixel = torch.stack(history_tensors).unsqueeze(0).to(device)  # [1, 3, 2, H, W]
        
        # Encode to latent [1, 3, 4, 64, 64]
        with torch.no_grad():
            history_latent_list = []
            for i in range(3):
                frame = history_pixel[:, i]  # [1, 2, H, W]
                z = vae.encode(frame)  # [1, 4, 64, 64]
                history_latent_list.append(z)
            history_latent = torch.cat(history_latent_list, dim=0).unsqueeze(0)  # [1, 3, 4, 64, 64]
        
        # Generate forecast sequence
        forecast_frames = []
        current_history = history_latent.clone()
        
        num_inference_steps = cfg["diffusion"]["inference"].get("num_inference_steps", 50)
        
        for step in range(num_steps):
            log.info(f"  Step {step + 1}/{num_steps}...")
            
            with torch.no_grad():
                # Sample next frame in latent space
                pred_latent = diffusion.sample(
                    current_history,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=1.0,
                )  # [1, 1, 4, 64, 64]
                
                # Decode to pixel space
                pred_pixel = vae.decode(pred_latent)  # [1, 1, 2, H, W]
                pred_np = pred_pixel[0, 0].cpu().numpy()  # [2, H, W]
                
                # Take TIR1 channel (first channel)
                forecast_frames.append(pred_np[0])  # [H, W]
                
                # Update history: remove oldest, add new prediction
                # Shift: keep frames 1, 2 and add prediction as new frame 2
                new_history = torch.cat([
                    current_history[:, 1:, :, :, :],  # [1, 2, 4, 64, 64]
                    pred_latent,  # [1, 1, 4, 64, 64]
                ], dim=1)  # [1, 3, 4, 64, 64]
                current_history = new_history
        
        log.info(f"Generated {len(forecast_frames)} forecast frames ✓")
        return forecast_frames
        
    except Exception as e:
        log.error(f"Forecast generation failed: {e}")
        return None


def publish_to_bucket(
    cfg: dict,
    forecast_frames: list[np.ndarray],
    observation_time: Optional[datetime] = None,
) -> bool:
    """Publish forecast results to GCS bucket.
    
    Parameters
    ----------
    cfg : dict
        Meghdoot config
    forecast_frames : list[np.ndarray]
        List of predicted frames (pixel space)
    observation_time : datetime
        Time of observation; defaults to now
    
    Returns
    -------
    bool
        True if successful
    """
    if observation_time is None:
        observation_time = datetime.utcnow()
    
    try:
        log.info("Publishing forecast to GCS bucket...")
        
        # Lazy import to avoid requiring google-cloud-storage if not uploading
        try:
            from google.cloud import storage
        except ImportError:
            log.error("google-cloud-storage not installed; skipping bucket upload")
            return False
        
        bucket_name = cfg.get("deployment", {}).get("gcs_bucket")
        if not bucket_name:
            log.error("GCS bucket not configured in config")
            return False
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Metadata
        lead_times_minutes = [30 * (i + 1) for i in range(len(forecast_frames))]
        valid_times = [
            (observation_time + timedelta(minutes=m)).isoformat()
            for m in lead_times_minutes
        ]
        
        metadata = {
            "observation_time": observation_time.isoformat(),
            "forecast_generated_at": datetime.utcnow().isoformat(),
            "num_steps": len(forecast_frames),
            "lead_times_minutes": lead_times_minutes,
            "valid_times": valid_times,
            "model": "meghdoot-ai-diffusion",
            "inference_steps": cfg["diffusion"]["inference"].get("num_inference_steps", 50),
        }
        
        # Upload metadata
        metadata_blob = bucket.blob("forecasts/latest/metadata.json")
        metadata_blob.upload_from_string(
            json.dumps(metadata, indent=2),
            content_type="application/json",
        )
        log.info(f"  Uploaded metadata to gs://{bucket_name}/forecasts/latest/metadata.json")
        
        # Upload forecast images
        for i, frame in enumerate(forecast_frames):
            png_blob = bucket.blob(f"forecasts/latest/forecast_step_{i}.png")
            png_bytes = array_to_png_bytes(frame)
            png_blob.upload_from_file(png_bytes, content_type="image/png")
            log.info(f"  Uploaded forecast_step_{i}.png")
        
        log.info("Forecast published to bucket ✓")
        return True
        
    except Exception as e:
        log.error(f"Publishing to bucket failed: {e}")
        return False


def main() -> int:
    """Main pipeline entry point."""
    parser = argparse.ArgumentParser(description="Meghdoot Nowcasting Pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num-steps", type=int, default=6, help="Forecast steps (6 = 3 hours)")
    args = parser.parse_args()
    
    load_backend_environment()
    cfg = prepare_backend_config(load_config(args.config))
    device = get_device(cfg["project"].get("device", "cuda"))
    
    log.info("="*80)
    log.info("Meghdoot Nowcasting Pipeline Started")
    log.info("="*80)
    
    t_start = time.time()
    
    # Step 1: Download
    raw_files = download_latest_frames(cfg)
    if not raw_files:
        log.error("Pipeline failed at download step")
        return 1
    
    # Step 2: Preprocess
    processed_files = preprocess_frames(cfg, raw_files)
    if not processed_files:
        log.error("Pipeline failed at preprocessing step")
        return 1
    
    # Step 3: Load models
    try:
        log.info("Loading VAE and Diffusion models...")
        vae = SatelliteVAE(cfg).to(device).eval()
        vae_ckpt_path = ensure_vae_checkpoint(cfg)
        if vae_ckpt_path is not None:
            vae.load(vae_ckpt_path)
            log.info(f"Loaded VAE checkpoint: {vae_ckpt_path.name}")

        diffusion = MeghdootDiffusion(cfg).to(device).eval()
        ckpt_path = ensure_diffusion_checkpoint(cfg)
        if ckpt_path is not None:
            diffusion.load(ckpt_path)
            log.info(f"Loaded checkpoint: {ckpt_path.name}")
    except Exception as e:
        log.error(f"Failed to load models: {e}")
        return 1
    
    # Step 4: Generate forecast
    forecast_frames = generate_forecast_sequence(
        cfg,
        vae,
        diffusion,
        processed_files,
        num_steps=args.num_steps,
        device=device,
    )
    if forecast_frames is None:
        log.error("Pipeline failed at forecast generation step")
        return 1
    
    # Step 5: Publish
    if not publish_to_bucket(cfg, forecast_frames):
        log.error("Pipeline failed at publishing step")
        return 1
    
    elapsed = time.time() - t_start
    log.info("="*80)
    log.info(f"Pipeline completed successfully in {elapsed:.1f}s")
    log.info("="*80)
    return 0


if __name__ == "__main__":
    exit(main())
