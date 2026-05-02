"""Train the ConvLSTM + diffusion hybrid refiner."""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from meghdoot.data.dataset import LatentSequenceDataset
from meghdoot.models.hybrid import ConvLSTMDiffusionHybrid
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import get_device, seed_everything
from meghdoot.utils.logging import get_logger, setup_wandb


log = get_logger(__name__)


def prune_old_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    if keep_last <= 0 or not checkpoint_dir.exists():
        return

    pattern = re.compile(r"diffusion_epoch(\d+)\.pt$")
    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("diffusion_epoch*.pt"):
        match = pattern.search(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))

    checkpoints.sort(key=lambda item: item[0])
    for _, path in checkpoints[:-keep_last]:
        try:
            path.unlink()
            log.info(f"Pruned old checkpoint: {path.name}")
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ConvLSTM + diffusion hybrid")
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", default=None, help="Optional diffusion checkpoint to resume from")
    parser.add_argument("--convlstm-ckpt", default=None, help="ConvLSTM checkpoint used as motion prior")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs for this run")
    parser.add_argument("--log-images-every", type=int, default=None, help="Image logging cadence")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    setup_wandb(cfg)

    t_cfg = cfg["diffusion"]["training"]
    run_epochs = args.epochs if args.epochs is not None else t_cfg["epochs"]
    log_images_every = (
        args.log_images_every
        if args.log_images_every is not None
        else cfg.get("logging", {}).get("wandb", {}).get("log_images_every_n_epochs", 10)
    )

    requested_device = cfg["project"].get("device", "cuda")
    device = get_device(requested_device)
    if requested_device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not available in the current environment.")
    log.info(f"Using device: {device}")

    dataset = LatentSequenceDataset(
        latent_dir=cfg["data"]["paths"]["latents"],
        num_history=cfg["diffusion"]["conditioning"]["num_history_frames"],
        cache_in_memory=True,
    )
    num_workers = 0 if device.type == "cuda" else cfg["data"]["num_workers"]
    dataloader = DataLoader(
        dataset,
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=cfg["data"].get("prefetch_factor", 2) if num_workers > 0 else None,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    convlstm_ckpt = args.convlstm_ckpt or cfg.get("hybrid", {}).get("convlstm_ckpt")
    freeze_convlstm = cfg.get("hybrid", {}).get("freeze_convlstm", True)
    model = ConvLSTMDiffusionHybrid(
        cfg,
        convlstm_ckpt=convlstm_ckpt,
        freeze_convlstm=freeze_convlstm,
    ).to(device)

    # Build optimizer params from explicitly named leaf Parameters.
    # This prevents "can't optimize a non-leaf Tensor" errors from silent tensor moves.
    named_opt_params: list[tuple[str, torch.nn.Parameter]] = []
    named_opt_params.extend(
        (f"diffusion.unet.{name}", p)
        for name, p in model.diffusion.unet.named_parameters()
        if p.requires_grad
    )
    if not freeze_convlstm:
        named_opt_params.extend(
            (f"convlstm.{name}", p)
            for name, p in model.convlstm.named_parameters()
            if p.requires_grad
        )
    if model.use_affine_calibration and model.affine_scale is not None:
        named_opt_params.append(("hybrid.affine_scale", model.affine_scale))
        named_opt_params.append(("hybrid.affine_bias", model.affine_bias))

    non_leaf = [name for name, p in named_opt_params if not p.is_leaf]
    if non_leaf:
        bad = ", ".join(non_leaf)
        raise RuntimeError(
            "Optimizer received non-leaf parameters: "
            f"{bad}. Ensure parameters are not reassigned via tensor .to() results."
        )

    opt_params = [p for _, p in named_opt_params]

    # Gradient clipping params for monitoring
    clip_params = [p for p in opt_params if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        opt_params,
        lr=t_cfg["learning_rate"],
        weight_decay=1e-4,
    )

    total_steps = run_epochs * len(dataloader) // t_cfg["gradient_accumulation_steps"]
    warmup_steps = t_cfg["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    if args.resume:
        start_epoch = model.diffusion.load(args.resume, optimizer=optimizer, lr_scheduler=scheduler)
    final_epoch = start_epoch + run_epochs

    use_amp = t_cfg.get("mixed_precision", "fp16") == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    checkpoint_every = cfg.get("hybrid", {}).get("save_every_n_epochs", 10)
    keep_last_checkpoints = cfg.get("hybrid", {}).get("keep_last_checkpoints", 3)
    checkpoint_dir = Path(cfg.get("hybrid", {}).get("checkpoint_dir", cfg["diffusion"]["checkpoint_dir"].replace("diffusion", "hybrid")))

    log.info("═══ Starting Hybrid ConvLSTM + Diffusion Training ═══")
    log.info(
        f"  Run epochs: {run_epochs}  |  Total target epoch: {final_epoch}  |  Batch: {t_cfg['batch_size']}"
        f"  |  Accum: {t_cfg['gradient_accumulation_steps']}  |  AMP: {use_amp}"
    )

    global_step = 0

    for epoch in range(start_epoch + 1, final_epoch + 1):
        run_epoch = epoch - start_epoch
        model.diffusion.unet.train()
        # Train ConvLSTM only if unfrozen; otherwise keep in eval mode
        model.convlstm.train(mode=not freeze_convlstm)

        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_x0_recon = 0.0
        epoch_edge = 0.0
        epoch_contrast = 0.0
        epoch_start = time.perf_counter()

        optimizer.zero_grad()

        pbar = tqdm(
            enumerate(dataloader, 1),
            total=len(dataloader),
            desc=f"Epoch {run_epoch:3d}/{run_epochs} (abs {epoch})",
            unit="batch",
            leave=True,
            colour="cyan",
        )

        batch_wait_start = time.perf_counter()
        for step, batch in pbar:
            batch_ready = time.perf_counter()
            data_time = batch_ready - batch_wait_start
            step_start = batch_ready

            history = batch["history"].to(device, non_blocking=torch.cuda.is_available())
            target = batch["target"].to(device, non_blocking=torch.cuda.is_available())

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                losses = model.training_step(history, target)
                loss = losses["loss"] / t_cfg["gradient_accumulation_steps"]

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % t_cfg["gradient_accumulation_steps"] == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, t_cfg["max_grad_norm"])
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                model.diffusion.ema.update(model.diffusion.unet)
                global_step += 1

            epoch_loss += losses["loss"].item()
            epoch_mse += losses["mse_loss"].item()
            epoch_x0_recon += losses["x0_recon_loss"].item()
            epoch_edge += losses["edge_loss"].item()
            epoch_contrast += losses["contrast_loss"].item()

            pbar.set_postfix(
                {
                    "loss": f"{losses['loss'].item():.4f}",
                    "mse": f"{losses['mse_loss'].item():.4f}",
                    "data": f"{data_time:.2f}s",
                    "step": f"{time.perf_counter() - step_start:.2f}s",
                }
            )

            batch_wait_start = time.perf_counter()

        n = len(dataloader)
        avg_loss = epoch_loss / n
        avg_mse = epoch_mse / n
        avg_x0_recon = epoch_x0_recon / n
        avg_edge = epoch_edge / n
        avg_contrast = epoch_contrast / n
        epoch_time = time.perf_counter() - epoch_start

        if not math.isfinite(avg_loss):
            log.error(f"Loss exploded (NaN) at epoch {epoch}. Stopping to protect weights.")
            break

        log.info(
            f"Epoch {epoch:3d}/{final_epoch} │ loss={avg_loss:.5f}  mse={avg_mse:.5f}  "
            f"x0_recon={avg_x0_recon:.5f}  edge={avg_edge:.5f}  contrast={avg_contrast:.5f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  epoch_time={epoch_time:.1f}s"
        )

        try:
            import wandb

            if getattr(wandb, "run", None) is not None:
                wandb.log(
                    {
                        "hybrid/loss": avg_loss,
                        "hybrid/mse_loss": avg_mse,
                        "hybrid/x0_recon_loss": avg_x0_recon,
                        "hybrid/edge_loss": avg_edge,
                        "hybrid/contrast_loss": avg_contrast,
                        "hybrid/lr": scheduler.get_last_lr()[0],
                        "hybrid/epoch": epoch,
                    },
                    step=global_step,
                )
        except Exception:
            pass

        if epoch % log_images_every == 0:
            try:
                sample = dataset[0]
                preview_history = sample["history"].unsqueeze(0).to(device)
                preview_target = sample["target"].unsqueeze(0).to(device)
                preview_pred = model.sample(
                    preview_history,
                    num_inference_steps=cfg["diffusion"]["inference"]["num_inference_steps"],
                    guidance_scale=cfg["diffusion"]["inference"].get("guidance_scale", 1.0),
                )
                import wandb

                if getattr(wandb, "run", None) is not None:
                    wandb.log(
                        {
                            "hybrid/sample_target": wandb.Image(model.vae.decode(preview_target)[0, 0].detach().cpu().numpy()),
                            "hybrid/sample_pred": wandb.Image(model.vae.decode(preview_pred)[0, 0].detach().cpu().numpy()),
                        },
                        step=global_step,
                    )
            except Exception as exc:
                log.warning(f"Could not log hybrid samples: {exc}")

        if epoch % checkpoint_every == 0 or epoch == final_epoch:
            model.diffusion.save(checkpoint_dir, epoch, optimizer=optimizer, lr_scheduler=scheduler)
            prune_old_checkpoints(checkpoint_dir, keep_last_checkpoints)


if __name__ == "__main__":
    main()
