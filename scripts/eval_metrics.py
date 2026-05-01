#!/usr/bin/env python3
"""Quick evaluation script: compute RMSE/SSIM/PSNR (and CSI thresholds) for
diffusion-only, hybrid (ConvLSTM+Diffusion) and ConvLSTM baseline.

Usage:
  python scripts/eval_metrics.py --config configs/default.yaml --diffusion checkpoints/diffusion/diffusion_epoch5.pt --convlstm checkpoints/baselines/convlstm.pt --processed-dir /home/jupyter/local_data --n-samples 100
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from meghdoot.data.dataset import INSATSequenceDataset
from meghdoot.evaluation.baselines import ConvLSTMPredictor
from meghdoot.evaluation.metrics import compute_all_metrics
from meghdoot.models.vae import SatelliteVAE
from meghdoot.models.diffusion import MeghdootDiffusion
from meghdoot.models.hybrid import ConvLSTMDiffusionHybrid
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import get_device
from meghdoot.utils.logging import get_logger

log = get_logger(__name__)


def safe_channel_adjust(x: torch.Tensor, expected_in: int) -> torch.Tensor:
    """Slice or pad tensor `x` to have channel dim == expected_in.

    x: Tensor [B, C, H, W]
    """
    C = x.shape[1]
    if C == expected_in:
        return x
    if C > expected_in:
        return x[:, :expected_in]
    # C < expected_in -> pad with zeros
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
    dataset = INSATSequenceDataset(data_dir=Path(args.processed_dir), num_history=past_len, prefer_local_cache=True)

    vae = SatelliteVAE(cfg).to(device)
    # load weights if specified in config (optional)
    if cfg["vae"].get("pretrained") and str(cfg["vae"]["pretrained"]).endswith('.pt'):
        try:
            vae.load(cfg["vae"]["pretrained"])
        except Exception:
            log.info("No pretrained VAE weights loaded (optional).")
    vae.eval()

    diffusion = MeghdootDiffusion(cfg).to(device)
    diffusion.load(args.diffusion)
    diffusion.eval()

    convlstm = ConvLSTMPredictor(in_channels=cfg["vae"].get("latent_channels", 4),
                                 hidden_dims=cfg["evaluation"]["baselines"]["convlstm"]["hidden_dims"],
                                 kernel_size=cfg["evaluation"]["baselines"]["convlstm"]["kernel_size"]).to(device)
    try:
        state = torch.load(args.convlstm, map_location=device)
        convlstm.load_state_dict(state if isinstance(state, dict) else state)
    except Exception:
        log.warning("Could not load ConvLSTM weights fully; continuing with partial weights.")
    convlstm.eval()

    hybrid = ConvLSTMDiffusionHybrid(cfg, convlstm_ckpt=args.convlstm, freeze_convlstm=True).to(device)
    hybrid.eval()

    expected_in = vae.vae.encoder.conv_in.in_channels

    n = min(args.n_samples, len(dataset))
    log.info(f"Evaluating {n} samples from {dataset.target_dir}")

    metrics_diff = []
    metrics_hybrid = []
    metrics_conv = []

    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            history = sample["history"]  # [T, C, H, W]
            target = sample["target"]    # [C, H, W]

            # Ensure channel dims match VAE input
            # Encode each history frame separately and stack latents
            latents = []
            for t in range(history.shape[0]):
                frame = history[t:t+1].to(device)
                frame = safe_channel_adjust(frame, expected_in)
                z = vae.encode(frame)  # [1, latent_c, h, w]
                latents.append(z)
            history_latent = torch.cat(latents, dim=0).unsqueeze(0)  # [1, T, C_lat, H, W]

            # Diffusion-only sample
            pred_latent = diffusion.sample(history_latent, num_inference_steps=50, guidance_scale=1.0)
            pred_pixel = vae.decode(pred_latent)[0, 0].cpu().numpy()

            # Hybrid sample
            pred_hybrid_latent = hybrid.sample(history_latent)
            pred_hybrid_pixel = vae.decode(pred_hybrid_latent)[0, 0].cpu().numpy()

            # ConvLSTM baseline: predict base latent then decode
            base = convlstm(history_latent)
            base_pixel = vae.decode(base)[0, 0].cpu().numpy()

            tgt = target[0].cpu().numpy()

            metrics_diff.append(compute_all_metrics(pred_pixel, tgt, csi_thresholds=cfg.get("evaluation", {}).get("csi_thresholds", None)))
            metrics_hybrid.append(compute_all_metrics(pred_hybrid_pixel, tgt, csi_thresholds=cfg.get("evaluation", {}).get("csi_thresholds", None)))
            metrics_conv.append(compute_all_metrics(base_pixel, tgt, csi_thresholds=cfg.get("evaluation", {}).get("csi_thresholds", None)))

    def avg(list_of_dicts):
        keys = list_of_dicts[0].keys()
        out = {k: float(np.mean([d[k] for d in list_of_dicts])) for k in keys}
        return out

    print("Diffusion-only metrics:", avg(metrics_diff))
    print("Hybrid metrics:", avg(metrics_hybrid))
    print("ConvLSTM metrics:", avg(metrics_conv))


if __name__ == "__main__":
    main()
