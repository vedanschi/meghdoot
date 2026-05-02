#!/usr/bin/env python3
"""Diagnostic: compare reference latent stats for history vs ConvLSTM base

Saves per-sample mean/std/max/min for the last-history latent and the
ConvLSTM forecast latent so we can inspect amplitude mismatches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from meghdoot.data.dataset import INSATSequenceDataset
from meghdoot.models.hybrid import ConvLSTMDiffusionHybrid
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import get_device, seed_everything
from meghdoot.utils.logging import get_logger

log = get_logger(__name__)


def tensor_stats(x: torch.Tensor) -> dict:
    # x: [B, C, H, W]
    x_cpu = x.detach().cpu()
    stats = {}
    stats["mean"] = float(x_cpu.mean().item())
    stats["std"] = float(x_cpu.std().item())
    stats["min"] = float(x_cpu.min().item())
    stats["max"] = float(x_cpu.max().item())
    # per-channel means
    stats["per_channel_mean"] = [float(v) for v in x_cpu.flatten(2).mean(-1).mean(0).tolist()]
    stats["per_channel_std"] = [float(v) for v in x_cpu.flatten(2).std(-1).mean(0).tolist()]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--convlstm-local", default="checkpoints/baselines/convlstm.pt")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--output", default="results/latent_stats.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(args.seed)
    device = get_device(cfg["project"].get("device", "cuda"))
    log.info(f"Device: {device}")

    processed_dir = Path(args.processed_dir or cfg["data"]["paths"]["processed"])

    # Load hybrid wrapper (gives us a VAE and the ConvLSTM configured for latents)
    hybrid = ConvLSTMDiffusionHybrid(cfg, convlstm_ckpt=Path(args.convlstm_local), freeze_convlstm=True).to(device)
    hybrid.eval()

    # Dataset (prefer top-level pixel .pt files)
    past_len = cfg["diffusion"]["conditioning"]["num_history_frames"]
    dataset = INSATSequenceDataset(data_dir=processed_dir, num_history=past_len, prefer_local_cache=False)
    top_level_files = list(processed_dir.glob("*.pt"))
    if top_level_files:
        dataset.target_dir = processed_dir
        dataset.files = sorted(dataset.target_dir.glob("*.pt"))
        log.info(f"Using top-level test files: {len(dataset.files)} sequences")

    n = min(args.n_samples, len(dataset))
    results = []
    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            history_pixel = sample["history"]  # [T, C, H, W]

            # Encode history frames -> latent (treat T as batch)
            history_latent = hybrid.vae.encode(history_pixel.to(device)).unsqueeze(0)  # [1, T, C_z, H_z, W_z]

            # Reference latent = last history frame
            ref_latent = history_latent[:, -1]

            # ConvLSTM forecast (expects latent input)
            conv_base = hybrid.predict_base(history_latent)  # [B, C_z, H_z, W_z]

            # Stats
            entry = {
                "idx": i,
                "ref_stats": tensor_stats(ref_latent[0]),
                "conv_stats": tensor_stats(conv_base[0]),
            }
            # additional ratio metrics
            entry["mean_ratio_conv_ref"] = entry["conv_stats"]["mean"] / (entry["ref_stats"]["mean"] + 1e-8)
            entry["max_ratio_conv_ref"] = entry["conv_stats"]["max"] / (entry["ref_stats"]["max"] + 1e-8)

            results.append(entry)

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump({"cfg": args.config, "n": n, "results": results}, f, indent=2)

    print(f"Wrote {len(results)} entries to {outp}")


if __name__ == "__main__":
    main()
