"""Tests for the evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from meghdoot.evaluation.metrics import csi, compute_all_metrics, psnr, rmse, ssim


class TestSSIM:
    def test_identical_images(self):
        img = np.random.rand(64, 64).astype(np.float32) * 2 - 1
        assert ssim(img, img) > 0.99

    def test_different_images(self):
        img1 = np.ones((64, 64), dtype=np.float32)
        img2 = -np.ones((64, 64), dtype=np.float32)
        assert ssim(img1, img2) < 0.1


class TestRMSE:
    def test_identical(self):
        img = np.random.rand(64, 64).astype(np.float32)
        assert rmse(img, img) == 0.0

    def test_known_value(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.ones((10, 10), dtype=np.float32)
        assert np.isclose(rmse(a, b), 1.0)


class TestPSNR:
    def test_identical(self):
        img = np.random.rand(64, 64).astype(np.float32)
        assert psnr(img, img) == float("inf")


class TestCSI:
    def test_perfect_prediction(self):
        img = np.random.rand(64, 64).astype(np.float32) * 2 - 1
        score = csi(img, img, threshold=240.0)
        assert score == 1.0

    def test_no_events(self):
        # All values above threshold → no events → CSI = 1.0
        img = np.ones((64, 64), dtype=np.float32)
        score = csi(img, img, threshold=150.0)
        assert score == 1.0


class TestComputeAll:
    def test_returns_all_keys(self):
        img = np.random.rand(64, 64).astype(np.float32) * 2 - 1
        metrics = compute_all_metrics(img, img, csi_thresholds=[220, 240])
        assert "ssim" in metrics
        assert "rmse" in metrics
        assert "psnr" in metrics
        assert "csi_220" in metrics
        assert "csi_240" in metrics
