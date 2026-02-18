"""
train_diffusion.py – Train the Latent Diffusion Model
=====================================================

Trains the conditional UNet on pre-cached VAE latents with:
  • MSE noise-prediction loss
  • Physics-aware mass-conservation penalty
  • EMA weight averaging
  • W&B logging

Usage
-----
    python -m meghdoot.training.train_diffusion --config configs/default.yaml
    accelerate launch -m meghdoot.training.train_diffusion --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import math

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from meghdoot.data.dataset import LatentSequenceDataset
from meghdoot.models.diffusion import MeghdootDiffusion
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import seed_everything, get_device
from meghdoot.utils.logging import get_logger, setup_wandb

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Latent Diffusion Model")
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    setup_wandb(cfg)

    t_cfg = cfg["diffusion"]["training"]
    device = get_device(cfg["project"].get("device", "cuda"))

    # ── Data ──────────────────────────────────────
    dataset = LatentSequenceDataset(
        latent_dir=cfg["data"]["paths"]["latents"],
        channel=cfg["data"]["channels"][0],
        num_history=cfg["diffusion"]["conditioning"]["num_history_frames"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # ── Model ─────────────────────────────────────
    model = MeghdootDiffusion(cfg)
    start_epoch = 0
    if args.resume:
        start_epoch = model.load(args.resume)

    # ── Optimiser ─────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.unet.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=1e-4,
    )

    total_steps = t_cfg["epochs"] * len(dataloader) // t_cfg["gradient_accumulation_steps"]
    warmup_steps = t_cfg["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    use_amp = t_cfg.get("mixed_precision", "fp16") == "fp16" and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # ── Training Loop ─────────────────────────────
    log.info("═══ Starting Diffusion Training ═══")
    log.info(f"  Epochs: {t_cfg['epochs']}  |  Batch: {t_cfg['batch_size']}  "
             f"|  Accum: {t_cfg['gradient_accumulation_steps']}  |  AMP: {use_amp}")

    global_step = 0

    for epoch in range(start_epoch + 1, t_cfg["epochs"] + 1):
        model.unet.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_phys = 0.0
        epoch_ssim = 0.0
        epoch_mae = 0.0

        optimizer.zero_grad()

        for step, batch in enumerate(dataloader, 1):
            history = batch["history"].to(device)   # [B, 3, 4, 64, 64]
            target = batch["target"].to(device)      # [B, 4, 64, 64]

            with autocast(enabled=use_amp):
                losses = model.training_step(history, target)
                loss = losses["loss"] / t_cfg["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            if step % t_cfg["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.unet.parameters(),
                    t_cfg["max_grad_norm"],
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                model.ema.update(model.unet)
                global_step += 1

            epoch_loss += losses["loss"].item()
            epoch_mse += losses["mse_loss"].item()
            epoch_phys += losses["physics_loss"].item()
            epoch_ssim += losses["ssim_loss"].item()
            epoch_mae += losses["mae_loss"].item()

        n = len(dataloader)
        avg_loss = epoch_loss / n
        avg_mse = epoch_mse / n
        avg_phys = epoch_phys / n
        avg_ssim = epoch_ssim / n
        avg_mae = epoch_mae / n

        log.info(
            f"Epoch {epoch:3d}/{t_cfg['epochs']} │ "
            f"loss={avg_loss:.5f}  mse={avg_mse:.5f}  phys={avg_phys:.5f}  "
            f"ssim={avg_ssim:.5f}  mae={avg_mae:.5f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # W&B logging
        try:
            import wandb
            wandb.log({
                "diffusion/loss": avg_loss,
                "diffusion/mse": avg_mse,
                "diffusion/physics": avg_phys,
                "diffusion/ssim": avg_ssim,
                "diffusion/mae": avg_mae,
                "diffusion/lr": scheduler.get_last_lr()[0],
                "diffusion/epoch": epoch,
            })
        except Exception:
            pass

        # Checkpoint
        if epoch % 10 == 0 or epoch == t_cfg["epochs"]:
            model.save(cfg["diffusion"]["checkpoint_dir"], epoch)

        # Sample visualization every N epochs
        log_img_every = cfg.get("logging", {}).get("wandb", {}).get(
            "log_images_every_n_epochs", 10
        )
        if epoch % log_img_every == 0:
            _log_sample(model, dataset, device, epoch)

    log.info("Diffusion training complete ✓")


def _log_sample(model, dataset, device, epoch):
    """Generate a sample prediction and log to W&B."""
    try:
        import wandb
        import matplotlib.pyplot as plt

        sample = dataset[0]
        history = sample["history"].unsqueeze(0).to(device)
        pred = model.sample(history, num_inference_steps=20)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(sample["target"][0].numpy(), cmap="gray")
        axes[0].set_title("Ground Truth (latent ch-0)")
        axes[1].imshow(pred[0, 0].cpu().numpy(), cmap="gray")
        axes[1].set_title("Predicted (latent ch-0)")
        for ax in axes:
            ax.axis("off")
        plt.suptitle(f"Epoch {epoch}")
        plt.tight_layout()

        wandb.log({f"diffusion/sample_epoch{epoch}": wandb.Image(fig)})
        plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":
    main()
