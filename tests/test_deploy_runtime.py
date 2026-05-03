"""Tests for deployment runtime helpers."""

from __future__ import annotations

import os
from pathlib import Path


def test_prepare_backend_config_applies_cloud_run_overrides(monkeypatch):
    from meghdoot.deploy.runtime import is_cloud_run, prepare_backend_config

    monkeypatch.setenv("K_SERVICE", "meghdoot-backend")
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("MEGHDOOT_DEVICE", "cuda")
    monkeypatch.setenv("MEGHDOOT_GCS_BUCKET", "meghdoot-bucket")

    cfg = {
        "project": {"device": "cpu"},
        "data": {
            "paths": {
                "raw": "/home/jupyter/raw",
                "processed": "/home/jupyter/processed",
                "metadata": "/home/jupyter/metadata",
                "latents": "/home/jupyter/latents",
            }
        },
        "deployment": {"api": {"port": 8000}},
    }

    runtime_cfg = prepare_backend_config(cfg)

    assert is_cloud_run() is True
    assert runtime_cfg["project"]["device"] == "cuda"
    assert runtime_cfg["deployment"]["gcs_bucket"] == "meghdoot-bucket"
    assert runtime_cfg["deployment"]["api"]["port"] == 8080
    assert runtime_cfg["data"]["paths"]["raw"] == "/tmp/meghdoot/raw"
    assert runtime_cfg["data"]["paths"]["processed"] == "/tmp/meghdoot/processed"
    assert runtime_cfg["data"]["paths"]["metadata"] == "/tmp/meghdoot/metadata"
    assert runtime_cfg["data"]["paths"]["latents"] == "/tmp/meghdoot/latents"


def test_load_backend_environment_reads_env_file(monkeypatch):
    from meghdoot.deploy.runtime import load_backend_environment

    for key in [
        "MEGHDOOT_GCS_BUCKET",
        "MEGHDOOT_VAE_CHECKPOINT_GCS_URI",
        "MEGHDOOT_DIFFUSION_CHECKPOINT_GCS_URI",
    ]:
        monkeypatch.delenv(key, raising=False)

    load_backend_environment()

    assert os.environ["MEGHDOOT_GCS_BUCKET"] == "meghdoot-satellite-data"
    assert (
        os.environ["MEGHDOOT_VAE_CHECKPOINT_GCS_URI"]
        == "gs://meghdoot-satellite-data/vae_final/vae_final.pt"
    )
    assert (
        os.environ["MEGHDOOT_DIFFUSION_CHECKPOINT_GCS_URI"]
        == "gs://meghdoot-satellite-data/production/meghdoot_epoch290_prod.pt"
    )


def test_ensure_diffusion_checkpoint_prefers_latest_epoch(tmp_path, monkeypatch):
    from meghdoot.deploy.runtime import ensure_diffusion_checkpoint

    ckpt_dir = tmp_path / "checkpoints" / "diffusion"
    ckpt_dir.mkdir(parents=True)
    older = ckpt_dir / "diffusion_epoch12.pt"
    newer = ckpt_dir / "diffusion_epoch290.pt"
    older.write_text("old")
    newer.write_text("new")

    cfg = {"diffusion": {"checkpoint_dir": str(ckpt_dir)}}
    monkeypatch.delenv("MEGHDOOT_FORCE_CHECKPOINT_SYNC", raising=False)
    monkeypatch.delenv("MEGHDOOT_DIFFUSION_CHECKPOINT_GCS_URI", raising=False)
    monkeypatch.delenv("MEGHDOOT_DIFFUSION_CHECKPOINT_BUCKET", raising=False)

    selected = ensure_diffusion_checkpoint(cfg)

    assert selected is not None
    assert selected == newer
    assert selected.read_text() == "new"
