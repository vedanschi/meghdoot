"""Runtime helpers for deployment targets.

These helpers keep the API and batch pipeline usable in both the current
VM-based setup and a Cloud Run GPU deployment.
"""

from __future__ import annotations

import copy
import re
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from meghdoot.utils.helpers import ensure_dir
from meghdoot.utils.logging import get_logger

log = get_logger(__name__)


def load_backend_environment() -> None:
    """Load backend environment files if present.

    Order is intentional:
    - .env.backend for deployment-specific overrides
    - .env for any generic local values
    - existing environment variables still win because override=False
    """
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env.backend", override=False)
    load_dotenv(project_root / ".env", override=False)


def is_cloud_run() -> bool:
    """Return ``True`` when running in Cloud Run or Cloud Run jobs."""
    return bool(os.environ.get("K_SERVICE") or os.environ.get("CLOUD_RUN_JOB"))


def _runtime_root() -> Path:
    """Return the writable runtime root for temporary data."""
    return Path(os.environ.get("MEGHDOOT_RUNTIME_DIR", "/tmp/meghdoot"))


def _apply_path_override(cfg: dict[str, Any], section: str, key: str, value: Path) -> None:
    cfg.setdefault(section, {}).setdefault("paths", {})[key] = str(value)


def prepare_backend_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply deployment-time overrides to the loaded config.

    This makes the same config work on the local VM and on Cloud Run by
    redirecting writable directories to /tmp when needed and honoring
    environment overrides for ports, devices, and bucket names.
    """
    runtime_cfg = copy.deepcopy(cfg)

    project_cfg = runtime_cfg.setdefault("project", {})
    deployment_cfg = runtime_cfg.setdefault("deployment", {})
    data_cfg = runtime_cfg.setdefault("data", {})
    paths_cfg = data_cfg.setdefault("paths", {})

    requested_device = os.environ.get("MEGHDOOT_DEVICE")
    if requested_device:
        project_cfg["device"] = requested_device

    gcs_bucket = os.environ.get("MEGHDOOT_GCS_BUCKET")
    if gcs_bucket:
        deployment_cfg["gcs_bucket"] = gcs_bucket

    api_port = os.environ.get("PORT")
    if api_port:
        deployment_cfg.setdefault("api", {})["port"] = int(api_port)

    if is_cloud_run():
        runtime_root = _runtime_root()
        _apply_path_override(runtime_cfg, "data", "raw", runtime_root / "raw")
        _apply_path_override(runtime_cfg, "data", "processed", runtime_root / "processed")
        _apply_path_override(runtime_cfg, "data", "metadata", runtime_root / "metadata")
        _apply_path_override(runtime_cfg, "data", "latents", runtime_root / "latents")
        ensure_dir(runtime_root)

    # Keep any explicit path overrides from env vars if present.
    raw_dir = os.environ.get("MEGHDOOT_RAW_DIR")
    processed_dir = os.environ.get("MEGHDOOT_PROCESSED_DIR")
    metadata_dir = os.environ.get("MEGHDOOT_METADATA_DIR")
    latents_dir = os.environ.get("MEGHDOOT_LATENTS_DIR")

    if raw_dir:
        paths_cfg["raw"] = raw_dir
    if processed_dir:
        paths_cfg["processed"] = processed_dir
    if metadata_dir:
        paths_cfg["metadata"] = metadata_dir
    if latents_dir:
        paths_cfg["latents"] = latents_dir

    for path_key in ("raw", "processed", "metadata", "latents"):
        if path_key in paths_cfg and paths_cfg[path_key]:
            ensure_dir(paths_cfg[path_key])

    return runtime_cfg


def _parse_gs_uri(gs_uri: str) -> tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gs_uri}")
    bucket_and_blob = gs_uri[5:]
    if "/" not in bucket_and_blob:
        return bucket_and_blob, ""
    bucket_name, blob_name = bucket_and_blob.split("/", 1)
    return bucket_name, blob_name


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"diffusion_epoch(\d+)", path.stem)
    epoch = int(match.group(1)) if match else -1
    return epoch, path.name


def download_gcs_blob(gs_uri: str, destination: Path) -> Path:
    """Download a GCS object to a local path."""
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise RuntimeError("google-cloud-storage is required for GCS downloads") from exc

    bucket_name, blob_name = _parse_gs_uri(gs_uri)
    if not blob_name:
        raise ValueError(f"GCS URI must point to an object, not a bucket: {gs_uri}")

    ensure_dir(destination.parent)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: {gs_uri}")

    blob.download_to_filename(str(destination))
    return destination


def ensure_diffusion_checkpoint(cfg: dict[str, Any]) -> Path | None:
    """Ensure the diffusion checkpoint is available locally.

    The function supports three modes:
    - Use an already-present local checkpoint directory.
    - Download a single checkpoint from ``MEGHDOOT_DIFFUSION_CHECKPOINT_GCS_URI``.
    - Download the newest matching checkpoint from a GCS bucket/prefix.
    """
    checkpoint_dir = ensure_dir(cfg["diffusion"]["checkpoint_dir"])
    local_ckpts = sorted(checkpoint_dir.glob("diffusion_epoch*.pt"), key=_checkpoint_sort_key)
    if local_ckpts and os.environ.get("MEGHDOOT_FORCE_CHECKPOINT_SYNC") != "1":
        return local_ckpts[-1]

    single_checkpoint = os.environ.get("MEGHDOOT_DIFFUSION_CHECKPOINT_GCS_URI")
    if single_checkpoint:
        local_name = Path(_parse_gs_uri(single_checkpoint)[1]).name
        destination = checkpoint_dir / local_name
        return download_gcs_blob(single_checkpoint, destination)

    bucket_name = os.environ.get("MEGHDOOT_DIFFUSION_CHECKPOINT_BUCKET")
    checkpoint_prefix = os.environ.get("MEGHDOOT_DIFFUSION_CHECKPOINT_PREFIX", "")
    if bucket_name:
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError("google-cloud-storage is required for GCS downloads") from exc

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = [
            blob
            for blob in bucket.list_blobs(prefix=checkpoint_prefix)
            if blob.name.endswith(".pt") and "diffusion_epoch" in blob.name
        ]
        if not blobs:
            log.warning("No diffusion checkpoints found in GCS bucket %s", bucket_name)
            return None

        def blob_sort_key(blob: Any) -> tuple[float, str]:
            timestamp = getattr(blob, "time_created", None) or getattr(blob, "updated", None)
            created_value = timestamp.timestamp() if timestamp is not None else 0.0
            return created_value, blob.name

        newest = sorted(blobs, key=blob_sort_key)[-1]
        destination = checkpoint_dir / Path(newest.name).name
        newest.download_to_filename(str(destination))
        return destination

    return local_ckpts[-1] if local_ckpts else None


def ensure_vae_checkpoint(cfg: dict[str, Any]) -> Path | None:
    """Ensure the VAE checkpoint is available locally when one is configured."""
    checkpoint_dir = ensure_dir(cfg["vae"]["checkpoint_dir"])
    local_ckpts = sorted(checkpoint_dir.glob("vae*.pt"))
    if local_ckpts and os.environ.get("MEGHDOOT_FORCE_CHECKPOINT_SYNC") != "1":
        return local_ckpts[-1]

    single_checkpoint = os.environ.get("MEGHDOOT_VAE_CHECKPOINT_GCS_URI")
    if single_checkpoint:
        local_name = Path(_parse_gs_uri(single_checkpoint)[1]).name
        destination = checkpoint_dir / local_name
        return download_gcs_blob(single_checkpoint, destination)

    bucket_name = os.environ.get("MEGHDOOT_VAE_CHECKPOINT_BUCKET")
    checkpoint_prefix = os.environ.get("MEGHDOOT_VAE_CHECKPOINT_PREFIX", "")
    if bucket_name:
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError("google-cloud-storage is required for GCS downloads") from exc

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = [
            blob
            for blob in bucket.list_blobs(prefix=checkpoint_prefix)
            if blob.name.endswith(".pt") and "vae" in blob.name
        ]
        if not blobs:
            log.warning("No VAE checkpoints found in GCS bucket %s", bucket_name)
            return None

        def blob_sort_key(blob: Any) -> tuple[float, str]:
            timestamp = getattr(blob, "time_created", None) or getattr(blob, "updated", None)
            created_value = timestamp.timestamp() if timestamp is not None else 0.0
            return created_value, blob.name

        newest = sorted(blobs, key=blob_sort_key)[-1]
        destination = checkpoint_dir / Path(newest.name).name
        newest.download_to_filename(str(destination))
        return destination

    return local_ckpts[-1] if local_ckpts else None