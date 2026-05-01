#!/usr/bin/env python3
"""Evaluate only the Hybrid (ConvLSTM + Diffusion) model on pixel test data.

Usage:
  python scripts/eval_hybrid.py --config configs/default.yaml \
      --diffusion checkpoints/diffusion/diffusion_epoch5.pt \
      --convlstm checkpoints/baselines/convlstm.pt \
      --processed-dir /home/jupyter/local_data --n-samples 100
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from meghdoot.data.dataset import INSATSequenceDataset
from meghdoot.evaluation.metrics import compute_all_metrics
from meghdoot.models.vae import SatelliteVAE
from meghdoot.models.hybrid import ConvLSTMDiffusionHybrid
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import get_device
from meghdoot.utils.logging import get_logger

log = get_logger(__name__)


def safe_channel_adjust(x: torch.Tensor, expected_in: int) -> torch.Tensor:
    """Ensure `x` has `expected_in` channels by slicing or zero-padding.
    x: [B, C, H, W]
    """
    C = x.shape[1]
    if C == expected_in:
        return x
    if C > expected_in:
        return x[:, :expected_in]
    pad = torch.zeros((x.shape[0], expected_in - C, *x.shape[2:]), device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--diffusion", required=True)
    parser.add_argument("--convlstm", required=True)
    parser.add_argument("--processed-dir", default="/home/jupyter/local_data")
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["project"].get("device", "cuda"))

    past_len = cfg["diffusion"]["conditioning"]["num_history_frames"]
    dataset = INSATSequenceDataset(data_dir=Path(args.processed_dir), num_history=past_len, prefer_local_cache=False)

    vae = SatelliteVAE(cfg).to(device)
    # load optional pretrained VAE if configured
    try:
        if cfg["vae"].get("pretrained") and str(cfg["vae"]["pretrained"]).endswith('.pt'):
            vae.load(cfg["vae"]["pretrained"])
    except Exception:
        log.info("VAE pretrained load skipped or failed; continuing.")
    vae.eval()

    hybrid = ConvLSTMDiffusionHybrid(cfg, convlstm_ckpt=args.convlstm, freeze_convlstm=True).to(device)
    # load diffusion weights into hybrid if provided
    try:
        hybrid.diffusion.load(args.diffusion)
    except Exception:
        log.info("Could not load diffusion checkpoint into hybrid; ensure path is correct.")
    hybrid.eval()

    # Expected pixel channels for VAE input
    expected_in = vae.vae.encoder.conv_in.in_channels

    n = min(args.n_samples, len(dataset))
    log.info(f"Evaluating {n} samples from {dataset.target_dir}")

    metrics_hybrid = []

    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            history = sample["history"]  # [T, C, H, W]
            target = sample["target"]    # [C, H, W]

            # Encode history frames into latents
            latents = []
            for t in range(history.shape[0]):
                frame = history[t:t+1].to(device)
                frame = safe_channel_adjust(frame, expected_in)
                z = vae.encode(frame)
                latents.append(z)
            history_latent = torch.cat(latents, dim=0).unsqueeze(0)  # [1, T, C_lat, H, W]

            pred_latent = hybrid.sample(history_latent)
            pred_pixel = vae.decode(pred_latent)[0, 0].cpu().numpy()

            tgt = target[0].cpu().numpy()
            metrics_hybrid.append(compute_all_metrics(pred_pixel, tgt, csi_thresholds=cfg.get("evaluation", {}).get("csi_thresholds", None)))

    # Average and print
    keys = metrics_hybrid[0].keys() if metrics_hybrid else []
    avg = {k: float(np.mean([m[k] for m in metrics_hybrid])) for k in keys}
    print("Hybrid model metrics (averaged):")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
