"""Tests for utility modules."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


class TestConfig:
    def test_load_default_config(self):
        from meghdoot.utils.config import load_config

        cfg = load_config()
        assert "data" in cfg
        assert "vae" in cfg
        assert "diffusion" in cfg

    def test_save_and_load(self, tmp_path):
        from meghdoot.utils.config import load_config, save_config

        cfg = load_config()
        out = tmp_path / "test_cfg.yaml"
        save_config(cfg, out)

        reloaded = load_config(out)
        assert reloaded["data"]["region"] == cfg["data"]["region"]


class TestHelpers:
    def test_seed_everything_deterministic(self):
        import torch
        from meghdoot.utils.helpers import seed_everything

        seed_everything(42)
        a = torch.randn(10)
        seed_everything(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_get_device(self):
        from meghdoot.utils.helpers import get_device

        device = get_device()
        assert device in ("cuda", "cpu")

    def test_ensure_dir(self, tmp_path):
        from meghdoot.utils.helpers import ensure_dir

        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(target)
        assert result.exists()
        assert result.is_dir()


class TestLogger:
    def test_get_logger(self):
        from meghdoot.utils.logging import get_logger

        logger = get_logger("test")
        assert logger is not None
