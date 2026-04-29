"""
benchmark.py – Run Comparative Evaluation
==========================================

Evaluates Meghdoot-AI against ConvLSTM and PySTEPS baselines,
computes all metrics, and generates comparison visualisations.

Usage
-----
    python -m meghdoot.evaluation.benchmark --config configs/default.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from meghdoot.data.dataset import INSATSequenceDataset, LatentSequenceDataset
from meghdoot.evaluation.baselines import ConvLSTMPredictor, pysteps_forecast
from meghdoot.evaluation.metrics import compute_all_metrics
from meghdoot.models.diffusion import MeghdootDiffusion
from meghdoot.models.vae import SatelliteVAE
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import ensure_dir, get_device, seed_everything
from meghdoot.utils.logging import get_logger

log = get_logger(__name__)


def evaluate_meghdoot(
    cfg: dict,
    diffusion: MeghdootDiffusion,
    vae: SatelliteVAE,
    dataset: LatentSequenceDataset,
    pixel_dataset: INSATSequenceDataset,
    device: torch.device,
    n_samples: int = 50,
) -> dict[str, float]:
    """Evaluate the Meghdoot-AI diffusion model on the primary TIR1 channel."""
    metrics_list = []
    # Note: These should now represent 10-bit raw thresholds, not Kelvin
    csi_thresholds = cfg["evaluation"].get("csi_thresholds", [600, 700, 800])

    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        pixel_sample = pixel_dataset[i]

        history = sample["history"].unsqueeze(0).to(device)
        pred_latent = diffusion.sample(history, num_inference_steps=50)

        # Decode to pixel space [B, 2, H, W]
        pred_pixel = vae.decode(pred_latent)
        
        # Extract Channel 0 (TIR1) for metrics
        pred_np = pred_pixel[0, 0].cpu().numpy()
        
        # Ground truth (pixel) [2, H, W] -> Extract Channel 0 (TIR1)
        gt_np = pixel_sample["target"][0].numpy()

        metrics = compute_all_metrics(pred_np, gt_np, csi_thresholds=csi_thresholds)
        metrics_list.append(metrics)

    # Average metrics
    avg = {}
    for key in metrics_list[0]:
        avg[key] = float(np.mean([m[key] for m in metrics_list]))

    return avg


def evaluate_convlstm(
    cfg: dict,
    dataset: INSATSequenceDataset,
    device: torch.device,
    ckpt_path: str | None = None,
    n_samples: int = 50,
) -> dict[str, float]:
    """Evaluate the ConvLSTM baseline on the primary TIR1 channel."""
    eval_cfg = cfg["evaluation"]["baselines"]["convlstm"]
    model = ConvLSTMPredictor(
        in_channels=2, # Updated for 2-channel tensors
        hidden_dims=eval_cfg["hidden_dims"],
        kernel_size=eval_cfg["kernel_size"],
    ).to(device)

    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    model.eval()
    metrics_list = []
    csi_thresholds = cfg["evaluation"].get("csi_thresholds", [600, 700, 800])

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]
            history = sample["history"].unsqueeze(0).to(device)
            
            # Predict [1, 2, H, W]
            pred = model(history)
            
            # Extract Channel 0 (TIR1)
            pred_np = pred[0, 0].cpu().numpy()
            gt_np = sample["target"][0].numpy()

            metrics = compute_all_metrics(pred_np, gt_np, csi_thresholds=csi_thresholds)
            metrics_list.append(metrics)

    avg = {}
    for key in metrics_list[0]:
        avg[key] = float(np.mean([m[key] for m in metrics_list]))
    return avg


def generate_comparison_video(
    meghdoot_preds: list[np.ndarray],
    convlstm_preds: list[np.ndarray],
    ground_truths: list[np.ndarray],
    output_path: str | Path,
) -> None:
    """Generate a side-by-side comparison grid saved as individual PNG frames."""
    output_path = ensure_dir(Path(output_path))

    for i, (mg, cl, gt) in enumerate(zip(meghdoot_preds, convlstm_preds, ground_truths)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(gt, cmap="inferno", vmin=-1, vmax=1)
        axes[0].set_title("Ground Truth", fontsize=14)
        axes[1].imshow(cl, cmap="inferno", vmin=-1, vmax=1)
        axes[1].set_title("ConvLSTM (Baseline)", fontsize=14)
        axes[2].imshow(mg, cmap="inferno", vmin=-1, vmax=1)
        axes[2].set_title("Meghdoot-AI (Ours)", fontsize=14)

        for ax in axes:
            ax.axis("off")

        plt.suptitle(f"Frame {i + 1}", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path / f"frame_{i:04d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    log.info(f"Saved {len(meghdoot_preds)} comparison frames → {output_path}")
    
def evaluate_pysteps(
    cfg: dict,
    dataset: INSATSequenceDataset,
    n_samples: int = 50,
) -> dict[str, float]:
    """Evaluate the PySTEPS optical flow baseline."""
    metrics_list = []
    csi_thresholds = cfg["evaluation"]["csi_thresholds"]

    for idx in range(min(n_samples, len(dataset))):
        sample = dataset[idx]
        history = sample["history"].numpy()  # [T_past, C, H, W]
        target = sample["target"].numpy()    # [T_future, C, H, W]

        # Generate prediction
        pred = pysteps_forecast(history, n_leadtimes=target.shape[0])

        # Evaluate strictly on TIR1 channel (Index 0) to match Meghdoot-AI
        pred_tir1 = pred[:, 0, :, :]
        target_tir1 = target[:, 0, :, :]

        metrics_list.append(compute_all_metrics(pred_tir1, target_tir1, csi_thresholds))

    # Average metrics over all samples
    avg_metrics = {}
    for key in metrics_list[0].keys():
        avg_metrics[key] = float(np.mean([m[key] for m in metrics_list]))

    return avg_metrics

def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run comparative evaluation")
    parser.add_argument("--config", default="configs/test.yaml")
    parser.add_argument("--diffusion-ckpt", required=True, help="Diffusion model checkpoint")
    parser.add_argument("--convlstm-ckpt", default=None, help="ConvLSTM checkpoint")
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg["project"].get("device", "cuda"))
    
    # Ensure out_dir is a Path object for the / operator
    out_dir = Path(cfg["evaluation"]["output_dir"])
    ensure_dir(out_dir)

    # FIX 1: Updated parameters to match dataset.py signature
    past_len = cfg["diffusion"]["conditioning"]["num_history_frames"]
    future_len = cfg.get("diffusion", {}).get("inference", {}).get("prediction_horizon_frames", 6)

    pixel_dataset = INSATSequenceDataset(
        data_dir=cfg["data"]["paths"]["processed"],
        seq_len_past=past_len,
        seq_len_future=future_len
    )
    latent_dataset = LatentSequenceDataset(
        latent_dir=cfg["data"]["paths"]["latents"],
        seq_len_past=past_len,
        seq_len_future=future_len
    )

    # FIX 2: Explicitly load fine-tuned VAE weights from bucket path
    vae = SatelliteVAE(cfg).to(device)
    vae_path = cfg["vae"]["pretrained"]
    if vae_path.endswith(".pt") and os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        log.info(f"Loaded fine-tuned VAE weights from {vae_path}")
    
    vae.eval()

    diffusion = MeghdootDiffusion(cfg).to(device)
    diffusion.load(args.diffusion_ckpt)
    diffusion.eval()

    # ── Evaluate ──────────────────────────────────
    log.info("═══ Evaluating Meghdoot-AI ═══")
    meghdoot_metrics = evaluate_meghdoot(
        cfg, diffusion, vae, latent_dataset, pixel_dataset, device, args.n_samples
    )
    
    results = {"meghdoot_ai": meghdoot_metrics}
    convlstm_metrics = {} # Initialize to prevent NameError

    if cfg["evaluation"]["baselines"]["convlstm"]["enabled"]:
        log.info("═══ Evaluating ConvLSTM Baseline ═══")
        convlstm_metrics = evaluate_convlstm(
            cfg, pixel_dataset, device, args.convlstm_ckpt, args.n_samples
        )
        results["convlstm"] = convlstm_metrics

    if cfg["evaluation"]["baselines"]["pysteps"]["enabled"]:
        log.info("═══ Evaluating PySTEPS Baseline ═══")
        pysteps_metrics = evaluate_pysteps(cfg, pixel_dataset, args.n_samples)
        results["pysteps"] = pysteps_metrics

    # Save results
    results_path = out_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {results_path}")

    # FIX 3: Robust comparison table
    log.info("\n" + "=" * 60)
    log.info(f"{'Metric':<20} {'Meghdoot-AI':>15} {'ConvLSTM':>15}")
    log.info("-" * 60)
    for key in meghdoot_metrics:
        m_val = meghdoot_metrics[key]
        c_val = convlstm_metrics.get(key, float("nan"))
        
        # Comparison logic (higher is better for SSIM/CSI, lower for RMSE)
        better = ""
        if not np.isnan(c_val):
            if key == "rmse":
                better = "✓" if m_val < c_val else ""
            else:
                better = "✓" if m_val > c_val else ""
                
        log.info(f"{key:<20} {m_val:>15.4f} {c_val:>15.4f}  {better}")
    log.info("=" * 60)
