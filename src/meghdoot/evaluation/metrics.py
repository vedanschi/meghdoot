"""
metrics.py – Evaluation Metrics for Weather Nowcasting
======================================================

Implements:
  • SSIM  – Structural Similarity (cloud boundary sharpness)
  • RMSE  – Root Mean Squared Error (overall accuracy)
  • CSI   – Critical Success Index (convective event reliability)
  • PSNR  – Peak Signal-to-Noise Ratio
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as sk_ssim


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 2.0) -> float:
    """Structural Similarity Index."""
    return float(sk_ssim(pred, target, data_range=data_range))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 2.0) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10 * np.log10(data_range ** 2 / mse))


def csi(
    pred: np.ndarray,
    target: np.ndarray,
    threshold_raw: float = 800.0, 
) -> float:
    """Critical Success Index (Threat Score) for intensity events.

    Binarises predictions based on 10-bit raw counts.
    (Note: Adjust threshold_raw based on what corresponds to heavy convection 
    in your specific 10-bit INSAT calibration).

    Parameters
    ----------
    pred, target : ndarray [H, W]
        Normalised images in [-1, 1].
    threshold_raw : float
        10-bit Raw count threshold (0-1023) for defining an event.
    """
    # De-normalise from [-1,1] back to 10-bit raw counts (0-1023)
    pred_raw = (pred + 1.0) * 511.5
    target_raw = (target + 1.0) * 511.5

    pred_event = pred_raw > threshold_raw
    tgt_event = target_raw > threshold_raw

    hits = np.sum(pred_event & tgt_event)
    misses = np.sum(~pred_event & tgt_event)
    false_alarms = np.sum(pred_event & ~tgt_event)

    denom = hits + misses + false_alarms
    if denom == 0:
        return 1.0  # no events → perfect score
    return float(hits / denom)


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    csi_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Compute all metrics in one call."""
    results = {
        "ssim": ssim(pred, target),
        "rmse": rmse(pred, target),
        "psnr": psnr(pred, target),
    }

    if csi_thresholds:
        for t in csi_thresholds:
            results[f"csi_{int(t)}"] = csi(pred, target, threshold_raw=t)

    return results