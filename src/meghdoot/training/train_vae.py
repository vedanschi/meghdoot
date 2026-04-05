"""
train_vae.py – Fine-tune the VAE on INSAT-3DR/3DS frames
=========================================================

Usage
-----
    python -m meghdoot.training.train_vae --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from meghdoot.models.vae import SatelliteVAE
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import seed_everything
from meghdoot.utils.logging import get_logger, setup_wandb

log = get_logger(__name__)


class VAESingleFrameDataset(Dataset):
    """
    Loads individual .pt tensors for VAE training.
    Unlike the Diffusion model, the VAE compresses single frames.
    This dataset ensures 100% of the downloaded data is utilized.
    """
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        
        target_dir = self.data_dir / "vae_tensors"
        if not target_dir.exists():
            target_dir = self.data_dir
            
        self.files = sorted(target_dir.glob("*.pt"))
        if not self.files:
            log.warning(f"No .pt files found in {target_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        fp = self.files[idx]
        tensor = torch.load(fp, weights_only=True)
        return {
            "target": tensor,         # [2, 512, 512] Tensor
            "timestamps": [fp.stem]   # Filename for latent caching
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune VAE on satellite data")
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache-only", action="store_true",
                        help="Skip fine-tuning, only cache latents")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    setup_wandb(cfg)

    # Use the VAE-specific single frame dataset
    dataset = VAESingleFrameDataset(
        data_dir=cfg["data"]["paths"]["processed"]
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
        log.info(f"Training on {len(dataset)} individual satellite frames.")
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
    
    # Removed obsolete 'channel' argument
    vae.cache_latents(
        cache_loader,
        out_dir=cfg["data"]["paths"]["latents"],
    )

    log.info("VAE pipeline complete ✓")


if __name__ == "__main__":
    main()
