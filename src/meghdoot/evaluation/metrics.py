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
    """Structural Similarity Index.

    Parameters
    ----------
    pred, target : ndarray [H, W]
        Images in [-1, 1].
    data_range : float
        Dynamic range (2.0 for [-1, 1] normalised data).
    """
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
    threshold: float = 240.0,
    denorm_range: tuple[float, float] = (180.0, 320.0),
) -> float:
    """Critical Success Index (Threat Score) for intensity events.

    Binarises predictions: "event" = brightness temp **below** threshold
    (lower BT → deeper convection → more intense weather).

    Parameters
    ----------
    pred, target : ndarray [H, W]
        Normalised images in [-1, 1].
    threshold : float
        Brightness temperature threshold (Kelvin) for defining an event.
    denorm_range : tuple
        (T_min, T_max) used during normalisation, to convert back to Kelvin.
    """
    # De-normalise from [-1,1] to Kelvin
    t_min, t_max = denorm_range
    pred_k = (pred + 1) / 2 * (t_max - t_min) + t_min
    target_k = (target + 1) / 2 * (t_max - t_min) + t_min

    pred_event = pred_k < threshold
    tgt_event = target_k < threshold

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
    """Compute all metrics in one call.

    Returns
    -------
    dict
        Keys: ``ssim``, ``rmse``, ``psnr``, ``csi_<threshold>`` for each threshold.
    """
    results = {
        "ssim": ssim(pred, target),
        "rmse": rmse(pred, target),
        "psnr": psnr(pred, target),
    }

    if csi_thresholds:
        for t in csi_thresholds:
            results[f"csi_{int(t)}"] = csi(pred, target, threshold=t)

    return results
