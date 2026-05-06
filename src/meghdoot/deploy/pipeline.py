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
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

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


def get_cached_raw_frames(
    cfg: dict,
    n_frames: int = 3,
) -> list[Path]:
    """Fetch the latest raw satellite files from local cache."""
    raw_dir = Path(cfg["data"]["paths"]["raw"])
    files = sorted(
        list(raw_dir.rglob("*.h*5"))
        + list(raw_dir.rglob("*.nc"))
        + list(raw_dir.rglob("*.nc4")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:n_frames]

    if len(files) < n_frames:
        log.warning(
            "Raw cache has only %s file(s); requested %s",
            len(files),
            n_frames,
        )

    return sorted(files)


def upload_raw_frames_to_gcs(
    cfg: dict,
    raw_files: list[Path],
) -> None:
    """Mirror freshly downloaded raw files to GCS for reuse on later runs."""
    bucket_name = cfg.get("deployment", {}).get("gcs_bucket")
    if not bucket_name:
        log.warning("No deployment.gcs_bucket configured; cannot upload raw cache to GCS")
        return

    try:
        from google.cloud import storage
    except ImportError:
        log.warning("google-cloud-storage not installed; skipping raw cache upload")
        return

    raw_dir = Path(cfg["data"]["paths"]["raw"])
    gcs_prefix = os.environ.get("MEGHDOOT_RAW_GCS_PREFIX", "raw/").strip().strip("/")
    if not gcs_prefix:
        gcs_prefix = "raw"

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    uploaded = 0
    for fp in raw_files:
        try:
            rel_path = fp.relative_to(raw_dir).as_posix()
        except ValueError:
            rel_path = fp.name

        blob = bucket.blob(f"{gcs_prefix}/{rel_path}")
        if blob.exists():
            continue

        blob.upload_from_filename(str(fp))
        uploaded += 1

    if uploaded:
        log.info(
            "Uploaded %s raw frame(s) to gs://%s/%s",
            uploaded,
            bucket_name,
            gcs_prefix,
        )


def get_raw_frames_from_gcs(
    cfg: dict,
    n_frames: int = 3,
) -> list[Path]:
    """Download latest raw satellite files from GCS into local raw cache."""
    bucket_name = cfg.get("deployment", {}).get("gcs_bucket")
    if not bucket_name:
        log.warning("No deployment.gcs_bucket configured; cannot fetch raw cache from GCS")
        return []

    try:
        from google.cloud import storage
    except ImportError:
        log.warning("google-cloud-storage not installed; cannot fetch raw cache from GCS")
        return []

    preferred_prefix = os.environ.get("MEGHDOOT_RAW_GCS_PREFIX", "raw/").strip().strip("/")
    candidate_prefixes = [preferred_prefix] if preferred_prefix else []
    for fallback_prefix in ("raw", "data/raw", "meghdoot/raw"):
        if fallback_prefix not in candidate_prefixes:
            candidate_prefixes.append(fallback_prefix)
    candidate_prefixes = [f"{prefix.rstrip('/')}/" for prefix in candidate_prefixes]

    raw_dir = Path(cfg["data"]["paths"]["raw"])
    ensure_dir(raw_dir)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs: list[Any] = []
    used_prefix = ""
    for prefix in candidate_prefixes:
        prefix_blobs = [
            blob
            for blob in bucket.list_blobs(prefix=prefix)
            if not blob.name.endswith("/") and Path(blob.name).suffix.lower() in {".h5", ".hdf5", ".nc", ".nc4"}
        ]
        if prefix_blobs:
            blobs = prefix_blobs
            used_prefix = prefix
            break

    if not blobs:
        log.warning(
            "No raw files found in gs://%s under prefixes: %s",
            bucket_name,
            ", ".join(candidate_prefixes),
        )
        return []

    def blob_sort_key(blob: Any) -> tuple[float, str]:
        timestamp = getattr(blob, "updated", None) or getattr(blob, "time_created", None)
        created_value = timestamp.timestamp() if timestamp is not None else 0.0
        return created_value, blob.name

    latest_blobs = sorted(blobs, key=blob_sort_key, reverse=True)[:n_frames]
    downloaded: list[Path] = []
    for blob in latest_blobs:
        rel_name = blob.name.removeprefix(used_prefix).lstrip("/")
        destination = raw_dir / rel_name
        ensure_dir(destination.parent)
        blob.download_to_filename(str(destination))
        downloaded.append(destination)

    log.info(
        "Loaded %s raw frame(s) from gs://%s/%s for fallback",
        len(downloaded),
        bucket_name,
        used_prefix,
    )
    return sorted(downloaded)


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


def get_cached_processed_frames(
    cfg: dict,
    n_frames: int = 3,
) -> list[Path]:
    """Fetch latest processed tensors from local cache.

    Parameters
    ----------
    cfg : dict
        Meghdoot config
    n_frames : int
        Number of frames required

    Returns
    -------
    list[Path]
        Latest cached processed tensors, newest-first slicing then sorted ascending
    """
    processed_dir = Path(cfg["data"]["paths"]["processed"])
    files = sorted(
        processed_dir.rglob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:n_frames]

    if len(files) < n_frames:
        log.warning(
            "Cache has only %s processed frame(s); requested %s",
            len(files),
            n_frames,
        )

    return sorted(files)


def get_processed_frames_from_gcs(
    cfg: dict,
    n_frames: int = 3,
) -> list[Path]:
    """Download latest processed tensors from GCS into local processed cache.

    Controlled by:
    - ``MEGHDOOT_PROCESSED_GCS_PREFIX``: preferred prefix in bucket
      (default: ``processed/``)
    """
    bucket_name = cfg.get("deployment", {}).get("gcs_bucket")
    if not bucket_name:
        log.warning("No deployment.gcs_bucket configured; cannot fetch processed cache from GCS")
        return []

    try:
        from google.cloud import storage
    except ImportError:
        log.warning("google-cloud-storage not installed; cannot fetch processed cache from GCS")
        return []

    preferred_prefix = os.environ.get("MEGHDOOT_PROCESSED_GCS_PREFIX", "processed/").strip()
    candidate_prefixes = [preferred_prefix] if preferred_prefix else []
    for fallback_prefix in ("processed/", "data/processed/", "meghdoot/processed/"):
        if fallback_prefix not in candidate_prefixes:
            candidate_prefixes.append(fallback_prefix)

    processed_dir = Path(cfg["data"]["paths"]["processed"])
    ensure_dir(processed_dir)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs: list[Any] = []
    used_prefix = ""
    for prefix in candidate_prefixes:
        prefix_blobs = [blob for blob in bucket.list_blobs(prefix=prefix) if blob.name.endswith(".pt")]
        if prefix_blobs:
            blobs = prefix_blobs
            used_prefix = prefix
            break

    if not blobs:
        log.warning(
            "No processed .pt tensors found in gs://%s under prefixes: %s",
            bucket_name,
            ", ".join(candidate_prefixes),
        )
        return []

    def blob_sort_key(blob: Any) -> tuple[float, str]:
        timestamp = getattr(blob, "updated", None) or getattr(blob, "time_created", None)
        created_value = timestamp.timestamp() if timestamp is not None else 0.0
        return created_value, blob.name

    latest_blobs = sorted(blobs, key=blob_sort_key, reverse=True)[:n_frames]
    downloaded: list[Path] = []
    for blob in latest_blobs:
        destination = processed_dir / Path(blob.name).name
        blob.download_to_filename(str(destination))
        downloaded.append(destination)

    log.info(
        "Loaded %s processed frame(s) from gs://%s/%s for fallback",
        len(downloaded),
        bucket_name,
        used_prefix,
    )
    return sorted(downloaded)


def download_forecast_data(
    cfg: dict,
    n_frames: int = 3,
) -> Optional[list[Path]]:
    """Download fresh satellite data for forecast.
    
    Prioritization order (MOSDAC first, cache fallback only):
    1. Fresh MOSDAC download (if not in cache-only mode)
    2. GCS raw cache (if MOSDAC fails)
    3. Local raw cache (if GCS fails)
    
    Returns
    -------
    list[Path] or None
        Raw file paths from whichever source succeeded
    """
    cache_only_mode = os.environ.get("MEGHDOOT_USE_CACHED_PROCESSED_ONLY", "0") == "1"
    
    # Try fresh MOSDAC download FIRST unless cache-only mode
    if not cache_only_mode:
        fresh_raw_files = download_latest_frames(cfg, n_frames=n_frames)
        if fresh_raw_files:
            log.info(
                "✓ Download successful: acquired %s fresh frame(s) from MOSDAC",
                len(fresh_raw_files),
            )
            # Mirror to GCS for future cache fallback
            upload_raw_frames_to_gcs(cfg, fresh_raw_files)
            return fresh_raw_files
        log.warning("MOSDAC download failed; attempting cache fallback...")
    else:
        log.info("Cache-only mode enabled: skipping MOSDAC download")
    
    # Fallback 1: Try GCS cache
    gcs_raw_files = get_raw_frames_from_gcs(cfg, n_frames=n_frames)
    if gcs_raw_files:
        log.info(
            "⊝ Download fallback (GCS): using %s cached raw frame(s) from GCS",
            len(gcs_raw_files),
        )
        return gcs_raw_files
    
    # Fallback 2: Try local cache
    local_raw_files = get_cached_raw_frames(cfg, n_frames=n_frames)
    if local_raw_files:
        log.warning(
            "⊝ Download fallback (local): using %s cached raw frame(s) from local disk",
            len(local_raw_files),
        )
        return local_raw_files
    
    log.error("All download attempts failed (MOSDAC, GCS, local cache)")
    return None


def prepare_history_frames(
    cfg: dict,
    n_frames: int = 3,
) -> Optional[list[Path]]:
    """Preprocess raw files or use cached processed tensors for inference.
    
    This function handles preprocessing raw files to tensors, with fallback
    to cached processed tensors if preprocessing fails. Raw data download
    is handled by download_forecast_data() separately.
    
    Parameters
    ----------
    cfg : dict
        Meghdoot config
    n_frames : int
        Number of frames to use
    
    Returns
    -------
    list[Path] or None
        Paths to processed tensor files, or None if all methods fail
    """
    # Raw files should be provided from download_forecast_data()
    raw_files = None
    
    # Try preprocessing: first check if any raw files exist locally
    cached_raw = get_cached_raw_frames(cfg, n_frames=n_frames)
    if cached_raw:
        log.info("Attempting to preprocess %s local raw file(s)...", len(cached_raw))
        processed = preprocess_frames(cfg, cached_raw)
        if processed:
            log.info("✓ Preprocessing successful: %s tensor(s)", len(processed))
            return sorted(processed)
        log.warning("Preprocessing failed; attempting cached processed tensors...")
    
    # Fallback to cached processed tensors (local)
    cached_processed = get_cached_processed_frames(cfg, n_frames=n_frames)
    if cached_processed:
        log.warning(
            "⊝ Using %s cached processed frame(s) from local disk (stale data)",
            len(cached_processed),
        )
        return sorted(cached_processed)
    
    # Final fallback: GCS processed cache
    gcs_processed = get_processed_frames_from_gcs(cfg, n_frames=n_frames)
    if gcs_processed:
        log.warning(
            "⊝ Using %s cached processed frame(s) from GCS (stale data)",
            len(gcs_processed),
        )
        return sorted(gcs_processed)
    
    log.error("No processed frames available (local or GCS cache)")
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
        Latest preprocessed tensor files, ideally 3 or more
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
        
        # Load the newest available frames.
        history_tensors = []
        for fp in sorted(processed_files)[-3:]:
            t = torch.load(fp, map_location=device)  # Expected [2, H, W]
            log.debug(f"Loaded tensor from {fp.name}: shape={t.shape}, dtype={t.dtype}, device={t.device}")
            
            # Normalize: if tensor is latent space [1, 4, 64, 64], decode it back to pixel space
            if t.ndim == 4 and t.shape[1] == 4:
                log.warning(f"Loaded tensor from {fp.name} appears to be in latent space {t.shape}; decoding to pixel space")
                try:
                    t = vae.decode(t)  # [1, 1, 2, H, W] or [1, 2, H, W]
                    if t.ndim == 5:
                        t = t.squeeze(1)  # [1, 2, H, W] -> [2, H, W]? No, this removes wrong dim
                    if t.ndim == 5 and t.shape[1] == 2:  # [1, 2, H, W] expected but got [1, 2, H, W]
                        t = t.squeeze(0)  # [2, H, W]
                    elif t.ndim == 5 and t.shape[0] == 1:  # [1, ?, ?, ?, ?]
                        t = t.squeeze(0)  # [?, ?, ?, ?]
                    log.debug(f"Decoded latent tensor to pixel space: {t.shape}")
                except Exception as e:
                    log.error(f"Failed to decode latent tensor from {fp.name}: {e}")
                    raise
            elif t.ndim == 5:
                log.warning(f"Loaded tensor from {fp.name} has unexpected 5 dimensions {t.shape}; squeezing batch dim")
                t = t.squeeze(0)
            
            if t.ndim != 3 or t.shape[0] != 2:
                log.error(f"Normalized tensor from {fp.name} has unexpected shape {t.shape}; expected [2, H, W]")
                raise ValueError(f"Invalid tensor shape: {t.shape}")
            
            history_tensors.append(t)
        
        if not history_tensors:
            log.error("Need at least 1 history frame, got 0")
            return None

        log.debug(f"Loaded {len(history_tensors)} history tensors. All normalized to pixel space. Shapes: {[t.shape for t in history_tensors]}")

        if len(history_tensors) < 3:
            missing = 3 - len(history_tensors)
            log.warning(
                "Only %s history frame(s) available; padding with the latest frame to reach 3.",
                len(history_tensors),
            )
            history_tensors.extend(history_tensors[-1].clone() for _ in range(missing))
        
        # Stack into [1, 3, 2, H, W]
        history_pixel = torch.stack(history_tensors).unsqueeze(0).to(device)  # [1, 3, 2, H, W]
        log.debug(f"Stacked history_pixel shape: {history_pixel.shape}")
        
        # Encode to latent [1, 3, 4, 64, 64]
        with torch.no_grad():
            history_latent_list = []
            for i in range(3):
                frame = history_pixel[:, i]  # [1, 2, H, W]
                log.debug(f"Encoding frame {i}: shape={frame.shape}, dtype={frame.dtype}, device={frame.device}")
                try:
                    z = vae.encode(frame)  # [1, 4, 64, 64]
                    if z.ndim != 4 or z.shape != torch.Size([1, 4, 64, 64]):
                        log.warning(f"Frame {i} encoded to unexpected shape {z.shape}; expected [1, 4, 64, 64]")
                    log.debug(f"Encoded frame {i}: shape={z.shape}, dtype={z.dtype}")
                    history_latent_list.append(z)
                except Exception as e:
                    log.error(f"Failed to encode frame {i} (shape={frame.shape}): {e}")
                    raise
            
            log.debug(f"Before cat: {len(history_latent_list)} latents, shapes: {[z.shape for z in history_latent_list]}, dtypes: {[z.dtype for z in history_latent_list]}")
            
            # Validate all latents have same shape
            if history_latent_list:
                first_shape = history_latent_list[0].shape
                for i, z in enumerate(history_latent_list):
                    if z.shape != first_shape:
                        log.error(f"Shape mismatch: latent {i} has shape {z.shape}, expected {first_shape}")
                        raise ValueError(f"Inconsistent latent shapes: {[z.shape for z in history_latent_list]}")
            
            try:
                history_latent = torch.cat(history_latent_list, dim=0).unsqueeze(0)  # [1, 3, 4, 64, 64]
                log.debug(f"history_latent after cat+unsqueeze: shape={history_latent.shape}, dtype={history_latent.dtype}")
            except RuntimeError as e:
                log.error(f"torch.cat failed. Latent list: shapes={[z.shape for z in history_latent_list]}, ndims={[z.ndim for z in history_latent_list]}, dtypes={[z.dtype for z in history_latent_list]}, devices={[z.device for z in history_latent_list]}")
                raise
        
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
                )  # [1, 4, 64, 64]
                log.debug(f"Sampled latent for step {step + 1}: shape={pred_latent.shape}, dtype={pred_latent.dtype}")
                
                if pred_latent.ndim != 4:
                    raise ValueError(f"Expected sampled latent to be 4D [B, C, H, W], got {pred_latent.shape}")
                
                # Decode to pixel space
                pred_pixel = vae.decode(pred_latent)  # [1, 2, H, W]
                pred_np = pred_pixel[0].cpu().numpy()  # [2, H, W]
                
                # Take TIR1 channel (first channel)
                forecast_frames.append(pred_np[0])  # [H, W]
                
                # Update history: remove oldest, add new prediction
                # Shift: keep frames 1, 2 and add prediction as new frame 2
                # pred_latent is [1, 4, 64, 64]; unsqueeze(1) → [1, 1, 4, 64, 64]
                # Concatenate along dim=1 with history[:, 1:] [1, 2, 4, 64, 64]
                # Result: [1, 3, 4, 64, 64]
                try:
                    new_history = torch.cat(
                        [
                            current_history[:, 1:, :, :, :],  # [1, 2, 4, 64, 64]
                            pred_latent.unsqueeze(1),  # [1, 1, 4, 64, 64]
                        ],
                        dim=1,
                    )  # [1, 3, 4, 64, 64]
                    log.debug(f"Updated history for step {step + 1}: shape={new_history.shape}")
                except RuntimeError as e:
                    log.error(
                        f"Failed to concatenate history tensors at step {step + 1}: "
                        f"current_history[:, 1:] shape={current_history[:, 1:, :, :, :].shape}, "
                        f"pred_latent.unsqueeze(1) shape={pred_latent.unsqueeze(1).shape}, error={e}"
                    )
                    raise
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
    
    # Step 1: Download fresh satellite data (MOSDAC-first, cache fallback)
    log.info("Step 1: Acquiring forecast data...")
    raw_files = download_forecast_data(cfg, n_frames=3)
    if not raw_files:
        log.error("Pipeline failed to acquire any satellite data")
        return 1
    
    # Step 1b: Preprocess raw files to tensors (with cache fallback if preprocessing fails)
    log.info("Step 2: Preprocessing to tensors...")
    processed_files = preprocess_frames(cfg, raw_files)
    if not processed_files:
        log.warning("Preprocessing failed; attempting cached processed frames...")
        processed_files = prepare_history_frames(cfg, n_frames=3)
        if not processed_files:
            log.error("Pipeline failed to obtain processed tensor frames")
            return 1
    
    # Step 3: Load models
    try:
        log.info("Step 3: Loading VAE and Diffusion models...")
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
    log.info("Step 4: Generating forecast sequence...")
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
    log.info("Step 5: Publishing forecast to GCS bucket...")
    if not publish_to_bucket(cfg, forecast_frames):
        log.error("Pipeline failed at publishing step")
        return 1
    
    elapsed = time.time() - t_start
    log.info("="*80)
    log.info(f"✓ Pipeline completed successfully in {elapsed:.1f}s")
    log.info(f"  Data source: {raw_files[0].parent.parent if raw_files else 'unknown'}")
    log.info("="*80)
    return 0


if __name__ == "__main__":
    exit(main())
