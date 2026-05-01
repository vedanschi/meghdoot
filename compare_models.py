#!/usr/bin/env python3
"""Compare Meghdoot diffusion against the ConvLSTM baseline.

Loads the epoch-130 diffusion checkpoint, the fine-tuned VAE, and the
ConvLSTM baseline checkpoint from GCS, then evaluates both models on the
same matched samples and saves a few side-by-side panels.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import cast

from meghdoot.data.dataset import INSATSequenceDataset, LatentSequenceDataset
from meghdoot.evaluation.baselines import ConvLSTMPredictor
from meghdoot.evaluation.metrics import compute_all_metrics
from meghdoot.models.diffusion import MeghdootDiffusion
from meghdoot.models.vae import SatelliteVAE
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import get_device, seed_everything
from meghdoot.utils.logging import get_logger


log = get_logger(__name__)


def download_from_gcs(gcs_uri: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        log.info(f"Using cached file: {local_path}")
        return local_path

    log.info(f"Downloading {gcs_uri} -> {local_path}")
    subprocess.run(["gsutil", "cp", gcs_uri, str(local_path)], check=True)
    return local_path


def load_checkpoint_any(module: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)

    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "ema_state_dict", "model", "weights"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not isinstance(state, dict):
        module.load_state_dict(state)
        return

    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value

    try:
        module.load_state_dict(cleaned, strict=True)
    except RuntimeError:
        module.load_state_dict(cleaned, strict=False)


def save_panels(out_dir: Path, target: np.ndarray, meghdoot: np.ndarray, convlstm: np.ndarray, idx: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [("Target", target), ("Meghdoot", meghdoot), ("ConvLSTM", convlstm)]

    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"Sample {idx}")
    fig.tight_layout()
    fig.savefig(out_dir / f"sample_{idx:03d}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Meghdoot diffusion vs ConvLSTM baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--diffusion-ckpt", default="checkpoints/diffusion/diffusion_epoch130.pt")
    parser.add_argument(
        "--convlstm-gcs-uri",
        default="gs://meghdoot-satellite-data/weights/baselines/convlstm.pt",
    )
    parser.add_argument("--convlstm-local", default="checkpoints/baselines/convlstm.pt")
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Directory containing the processed test tensors (.pt files)",
    )
    parser.add_argument(
        "--latent-dir",
        default=None,
        help="Directory containing the cached VAE latents for the same test split",
    )
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--output-dir", default="results/model_compare")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg["project"].get("device", "cuda"))
    log.info(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    convlstm_ckpt = download_from_gcs(args.convlstm_gcs_uri, Path(args.convlstm_local))

    past_len = cfg["diffusion"]["conditioning"]["num_history_frames"]
    csi_thresholds = cfg.get("evaluation", {}).get("csi_thresholds", [600, 700, 800])

    processed_dir = Path(args.processed_dir or cfg["data"]["paths"]["processed"])
    latent_dir = Path(args.latent_dir or cfg["data"]["paths"]["latents"])
    log.info(f"Using processed test data: {processed_dir}")
    log.info(f"Using latent test data: {latent_dir}")

    pixel_dataset = INSATSequenceDataset(data_dir=processed_dir, num_history=past_len)
    latent_dataset = LatentSequenceDataset(
        latent_dir=latent_dir,
        num_history=past_len,
        cache_in_memory=True,
    )

    vae = SatelliteVAE(cfg).to(device)
    vae.load(cfg["vae"]["pretrained"])
    vae.eval()

    diffusion = MeghdootDiffusion(cfg).to(device)
    diffusion.load(args.diffusion_ckpt)
    diffusion.eval()
    log.info(f"Loaded diffusion checkpoint: {args.diffusion_ckpt}")

    convlstm_cfg = cfg["evaluation"]["baselines"]["convlstm"]
    convlstm = ConvLSTMPredictor(
        in_channels=2,
        hidden_dims=convlstm_cfg["hidden_dims"],
        kernel_size=convlstm_cfg["kernel_size"],
    ).to(device)
    load_checkpoint_any(convlstm, convlstm_ckpt, device)
    convlstm.eval()
    log.info(f"Loaded ConvLSTM checkpoint: {convlstm_ckpt}")

    meghdoot_metrics: list[dict[str, float]] = []
    convlstm_metrics: list[dict[str, float]] = []

    sample_count = min(args.n_samples, len(pixel_dataset), len(latent_dataset))
    log.info(f"Evaluating {sample_count} matched samples")

    with torch.no_grad():
        for idx in range(sample_count):
            pixel_sample = pixel_dataset[idx]
            latent_sample = latent_dataset[idx]

            target_tensor = cast(torch.Tensor, pixel_sample["target"])
            history_pixel_tensor = cast(torch.Tensor, pixel_sample["history"])
            history_latent_tensor = cast(torch.Tensor, latent_sample["history"])

            target = target_tensor[0].cpu().numpy()
            history_pixel = history_pixel_tensor.to(device).unsqueeze(0)
            history_latent = history_latent_tensor.to(device).unsqueeze(0)

            pred_latent = diffusion.sample(history_latent, num_inference_steps=args.num_inference_steps)
            pred_pixel = vae.decode(pred_latent)[0, 0].detach().cpu().numpy()

            pred_conv = convlstm(history_pixel)[0, 0].detach().cpu().numpy()

            m_metrics = compute_all_metrics(pred_pixel, target, csi_thresholds=csi_thresholds)
            c_metrics = compute_all_metrics(pred_conv, target, csi_thresholds=csi_thresholds)

            meghdoot_metrics.append(m_metrics)
            convlstm_metrics.append(c_metrics)

            if idx < 5:
                save_panels(out_dir, target, pred_pixel, pred_conv, idx)

            log.info(
                f"[{idx + 1:03d}/{sample_count:03d}] "
                f"Meghdoot SSIM={m_metrics['ssim']:.4f} RMSE={m_metrics['rmse']:.4f} PSNR={m_metrics['psnr']:.2f} | "
                f"ConvLSTM SSIM={c_metrics['ssim']:.4f} RMSE={c_metrics['rmse']:.4f} PSNR={c_metrics['psnr']:.2f}"
            )

    summary = {"meghdoot": {}, "convlstm": {}}
    for key in meghdoot_metrics[0].keys():
        summary["meghdoot"][key] = float(np.mean([m[key] for m in meghdoot_metrics]))
        summary["convlstm"][key] = float(np.mean([m[key] for m in convlstm_metrics]))

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("FINAL COMPARISON")
    print("=" * 78)
    print(f"{'Metric':<16} {'Meghdoot':<16} {'ConvLSTM':<16} {'Winner'}")
    print("-" * 78)

    meghdoot_wins = 0
    convlstm_wins = 0

    for metric_name in ["ssim", "rmse", "psnr", "csi_600", "csi_700", "csi_800"]:
        if metric_name not in summary["meghdoot"]:
            continue

        m_val = summary["meghdoot"][metric_name]
        c_val = summary["convlstm"][metric_name]
        higher_is_better = metric_name != "rmse"

        if higher_is_better:
            winner = "Meghdoot" if m_val > c_val else "ConvLSTM"
        else:
            winner = "Meghdoot" if m_val < c_val else "ConvLSTM"

        if winner == "Meghdoot":
            meghdoot_wins += 1
        else:
            convlstm_wins += 1

        print(f"{metric_name:<16} {m_val:<16.4f} {c_val:<16.4f} {winner}")

    print("-" * 78)
    print(f"Overall: Meghdoot {meghdoot_wins}  |  ConvLSTM {convlstm_wins}")
    if meghdoot_wins > convlstm_wins:
        print("Result: Meghdoot is better on the numeric metrics.")
    elif meghdoot_wins < convlstm_wins:
        print("Result: ConvLSTM is better on the numeric metrics.")
    else:
        print("Result: Tie on the numeric metrics.")
    print("=" * 78)
    print(f"Saved metrics to: {out_dir / 'metrics.json'}")
    print(f"Saved sample panels to: {out_dir}")


if __name__ == "__main__":
    main()