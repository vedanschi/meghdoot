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
import os 
import subprocess
import math
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from meghdoot.data.dataset import LatentSequenceDataset
from meghdoot.models.diffusion import MeghdootDiffusion
from meghdoot.models.vae import SatelliteVAE
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import seed_everything, get_device
from meghdoot.utils.logging import get_logger, setup_wandb

log = get_logger(__name__)


def main() -> None:
    # --- AUTO-SYNC DATA FROM GCS TO NVME ---
    #local_data_path = "/home/jupyter/local_data"
    #if not os.path.exists(local_data_path):
        #log.info("Syncing latents from GCS to local NVMe...")
       # subprocess.run(["gsutil", "-m", "cp", "-r", "gs://meghdoot-satellite-data/latents/stacked_tensors", local_data_path], check=True)
    # ----------------------------------------
    parser = argparse.ArgumentParser(description="Train Latent Diffusion Model")
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    setup_wandb(cfg)

    t_cfg = cfg["diffusion"]["training"]
    requested_device = cfg["project"].get("device", "cuda")
    device = get_device(requested_device)
    if requested_device == "cuda" and device.type != "cuda":
        raise RuntimeError(
            "CUDA was requested but is not available in the current Python environment. "
            "The training loop would otherwise silently run on CPU."
        )
    log.info(f"Using device: {device}")

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
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

# ── Model & Optimiser ─────────────────────────
    model = MeghdootDiffusion(cfg)
    # Ensure the wrapped model is moved to the training device
    try:
        model = model.to(device)
    except Exception:
        # Some environments may not support direct .to on wrappers; try moving internal UNet
        try:
            model.unet.to(device)
        except Exception:
            pass
    
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

    # Now load the checkpoint and apply states to the optimizer/scheduler
    start_epoch = 0
    if args.resume:
        start_epoch = model.load(args.resume, optimizer=optimizer, lr_scheduler=scheduler)

    # Mixed precision
    use_amp = t_cfg.get("mixed_precision", "fp16") == "fp16" and device.type == "cuda"
    # Create a GradScaler in a way that's compatible across torch versions
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
        epoch_latent_l1 = 0.0
        epoch_temporal = 0.0
        epoch_start = time.perf_counter()

        optimizer.zero_grad()

        for step, batch in enumerate(dataloader, 1):
            step_start = time.perf_counter()
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
            epoch_latent_l1 += losses["latent_l1_loss"].item()
            epoch_temporal += losses["temporal_loss"].item()

            if step == 1 or step % 25 == 0:
                log.info(
                    f"Epoch {epoch:3d} step {step:4d}/{len(dataloader)} │ "
                    f"loss={losses['loss'].item():.5f} │ "
                    f"step_time={time.perf_counter() - step_start:.2f}s"
                )

        n = len(dataloader)
        avg_loss = epoch_loss / n
        
        # 1. FATAL CRASH CHECK: Stop immediately if loss becomes NaN
        if not math.isfinite(avg_loss):
            log.error(f"Loss exploded (NaN) at epoch {epoch}. Stopping to protect weights.")
            break

        avg_mse = epoch_mse / n
        avg_phys = epoch_phys / n
        avg_latent_l1 = epoch_latent_l1 / n
        avg_temporal = epoch_temporal / n
        epoch_time = time.perf_counter() - epoch_start

        log.info(
            f"Epoch {epoch:3d}/{t_cfg['epochs']} │ "
            f"loss={avg_loss:.5f}  mse={avg_mse:.5f}  phys={avg_phys:.5f}  "
            f"latent_l1={avg_latent_l1:.5f}  temporal={avg_temporal:.5f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  epoch_time={epoch_time:.1f}s"
        )

        # W&B logging
        try:
            import wandb
            wandb.log({
                "diffusion/loss": avg_loss,
                "diffusion/mse": avg_mse,
                "diffusion/physics": avg_phys,
                "diffusion/latent_l1": avg_latent_l1,
                "diffusion/temporal": avg_temporal,
                "diffusion/lr": scheduler.get_last_lr()[0],
                "diffusion/epoch": epoch,
            })
        except Exception:
            pass

        # 2. CHECKPOINT EVERY EPOCH: Pass optimizer and scheduler states
        save_freq = t_cfg.get("save_every_n_epochs", 1)
        if epoch % save_freq == 0 or epoch == t_cfg["epochs"]:
            model.save(
                cfg["diffusion"]["checkpoint_dir"], 
                epoch, 
                optimizer=optimizer, 
                lr_scheduler=scheduler
            )

        # Sample visualization every N epochs
        log_img_every = cfg.get("logging", {}).get("wandb", {}).get(
            "log_images_every_n_epochs", 10
        )
        if epoch % log_img_every == 0:
            _log_sample(model, dataset, device, epoch)

    log.info("Diffusion training complete ✓")

def _log_sample(model, dataset, device, epoch):
    """Generate a sample prediction, decode it, and log directly to W&B."""
    try:
        import wandb
        import numpy as np

        model.unet.eval()
        with torch.no_grad():
            sample = dataset[0]
            history = sample["history"].unsqueeze(0).to(device)
            # Use 20 steps for faster logging during training
            latent_pred = model.sample(history, num_inference_steps=20)

            # Decode using project SatelliteVAE to preserve 2-channel adaptation + latent scaling
            vae = SatelliteVAE(model.cfg).to(device).eval()
            pred_img = vae.decode(latent_pred)
            target_latent = sample["target"].unsqueeze(0).to(device)
            target_img = vae.decode(target_latent)

            # Normalize to [0, 255] for grayscale W&B logging
            def to_uint8(t):
                t = t.detach().cpu()[0, 0].numpy()
                t = (t - t.min()) / (t.max() - t.min() + 1e-6)
                return (t * 255).astype(np.uint8)

            # Log side-by-side grayscale images directly to W&B
            wandb.log({
                f"diffusion/epoch_{epoch}_target": wandb.Image(to_uint8(target_img), caption=f"Epoch {epoch} Ground Truth"),
                f"diffusion/epoch_{epoch}_pred": wandb.Image(to_uint8(pred_img), caption=f"Epoch {epoch} Prediction")
            })
            
            # Free up GPU memory so the UNet can continue training safely
            del vae
            torch.cuda.empty_cache()

        model.unet.train()
    except Exception as e:
        log.error(f"Visualization failed: {e}")

if __name__ == "__main__":
    main()
