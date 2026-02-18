"""
train_vae.py – Fine-tune the VAE on INSAT-3DR frames
=====================================================

Usage
-----
    python -m meghdoot.training.train_vae --config configs/default.yaml
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from meghdoot.data.dataset import INSATSequenceDataset
from meghdoot.models.vae import SatelliteVAE
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import seed_everything
from meghdoot.utils.logging import get_logger, setup_wandb

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune VAE on satellite data")
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache-only", action="store_true",
                        help="Skip fine-tuning, only cache latents")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    setup_wandb(cfg)

    # Dataset (pixel-space)
    dataset = INSATSequenceDataset(
        data_dir=cfg["data"]["paths"]["processed"],
        channel=cfg["data"]["channels"][0],
        num_history=cfg["diffusion"]["conditioning"]["num_history_frames"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["vae"]["fine_tune"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    vae = SatelliteVAE(cfg)

    # Fine-tune
    if not args.cache_only:
        log.info("═══ Starting VAE Fine-Tuning ═══")
        history = vae.fine_tune(dataloader)

        try:
            import wandb
            for i, (l, s, m, v) in enumerate(
                zip(history["loss"], history["ssim"], history["mae"], history["vgg"])
            ):
                wandb.log({
                    "vae/loss": l, "vae/ssim_loss": s,
                    "vae/mae": m, "vae/vgg": v, "vae/epoch": i + 1,
                })
        except Exception:
            pass

    # Cache latents
    log.info("═══ Caching Latent Vectors ═══")
    cache_loader = DataLoader(
        dataset,
        batch_size=cfg["vae"]["fine_tune"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    vae.cache_latents(
        cache_loader,
        out_dir=cfg["data"]["paths"]["latents"],
        channel=cfg["data"]["channels"][0],
    )

    log.info("VAE pipeline complete ✓")


if __name__ == "__main__":
    main()
