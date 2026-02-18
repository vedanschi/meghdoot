"""
diffusion.py – Latent Diffusion Model for Weather Nowcasting
============================================================

Implements the conditional UNet that operates in VAE latent space:
  • Input : 3 historical latent frames  (channel-concatenated)
  • Output: denoised prediction of the 4th frame's latent

Includes:
  - Standard MSE noise-prediction loss
  - Physics-aware mass-conservation penalty
  - EMA model averaging
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm

from meghdoot.utils.helpers import ensure_dir, get_device
from meghdoot.utils.logging import get_logger
from meghdoot.models.temporal_loss import TemporalConsistencyLoss

log = get_logger(__name__)


# ── Physics-Aware Loss ─────────────────────────────
class MassConservationLoss(nn.Module):
    """Penalises sudden creation / disappearance of "cloud mass"
    between the last conditioning frame and the predicted frame.

    Cloud mass is approximated as the spatial integral (sum) of
    brightness-temperature deviations from a reference value.
    """

    def forward(
        self,
        predicted: torch.Tensor,
        last_condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predicted : Tensor [B, C, H, W]
            Predicted latent (denoised).
        last_condition : Tensor [B, C, H, W]
            Last historical latent frame.

        Returns
        -------
        Tensor (scalar)
            Mean absolute difference in per-channel spatial sums.
        """
        mass_pred = predicted.sum(dim=(-2, -1))      # [B, C]
        mass_cond = last_condition.sum(dim=(-2, -1))  # [B, C]
        return (mass_pred - mass_cond).abs().mean()


# ── EMA Helper ─────────────────────────────────────
class EMAModel:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)

    def state_dict(self) -> dict:
        return self.shadow


# ── Meghdoot Diffusion Model ──────────────────────
class MeghdootDiffusion:
    """End-to-end wrapper for the conditional latent diffusion model."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.diff_cfg = cfg["diffusion"]
        self.device = get_device(cfg["project"].get("device", "cuda"))

        # Build UNet
        unet_cfg = self.diff_cfg["unet"]
        self.unet = UNet2DModel(
            sample_size=64,
            in_channels=unet_cfg["in_channels"],    # 4×(3+1) = 16
            out_channels=unet_cfg["out_channels"],   # 4
            block_out_channels=tuple(unet_cfg["block_out_channels"]),
            layers_per_block=unet_cfg["layers_per_block"],
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        ).to(self.device)

        # Noise scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.diff_cfg["num_train_timesteps"],
            beta_start=self.diff_cfg["beta_start"],
            beta_end=self.diff_cfg["beta_end"],
            beta_schedule=self.diff_cfg["beta_schedule"],
            prediction_type="epsilon",
        )

        # Physics loss
        self.mass_loss = MassConservationLoss()
        self.physics_weight = self.diff_cfg["physics_loss"].get("mass_conservation_weight", 0.1)
        self.grad_penalty_weight = self.diff_cfg["physics_loss"].get("gradient_penalty_weight", 0.05)

        # SSIM + MAE on latent x0 predictions (hybrid loss per tech spec)
        self.ssim_loss = SSIMLoss(channels=unet_cfg["out_channels"]).to(self.device)
        self.ssim_weight = self.diff_cfg.get("training", {}).get("ssim_weight", 0.1)
        self.mae_weight = self.diff_cfg.get("training", {}).get("mae_weight", 0.1)

        # Temporal consistency loss (optical-flow-based)
        temp_cfg = cfg.get("temporal_loss", {})
        self.temporal_loss_enabled = temp_cfg.get("enabled", False)
        if self.temporal_loss_enabled:
            self.temporal_loss = TemporalConsistencyLoss(
                warp_weight=temp_cfg.get("warp_weight", 1.0),
                flow_smooth_weight=temp_cfg.get("flow_smooth_weight", 0.1),
                flow_mag_weight=temp_cfg.get("flow_mag_weight", 0.01),
            )
            self.temporal_weight = temp_cfg.get("temporal_weight", 0.05)
            log.info("Temporal consistency loss ENABLED")

        # EMA
        self.ema = EMAModel(self.unet, decay=self.diff_cfg["training"]["ema_decay"])

        log.info(
            f"MeghdootDiffusion initialised: "
            f"{sum(p.numel() for p in self.unet.parameters())/1e6:.1f}M params"
        )

    # ── Training Step ──────────────────────────────
    def training_step(
        self,
        history_latents: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """One forward + loss computation.

        Parameters
        ----------
        history_latents : Tensor [B, N_hist, C, H, W]
            E.g. [B, 3, 4, 64, 64] – three historical latent frames.
        target_latent : Tensor [B, C, H, W]
            Ground-truth latent for the next frame.

        Returns
        -------
        dict  with keys ``"loss"``, ``"mse_loss"``, ``"physics_loss"``
        """
        B = target_latent.size(0)

        # Flatten history: [B, 3, 4, 64, 64] → [B, 12, 64, 64]
        cond = history_latents.view(B, -1, *history_latents.shape[-2:])

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        # Add noise to target
        noise = torch.randn_like(target_latent)
        noisy_target = self.scheduler.add_noise(target_latent, noise, timesteps)

        # Concatenate condition + noisy target along channel dim
        # [B, 12+4, 64, 64] = [B, 16, 64, 64]
        model_input = torch.cat([cond, noisy_target], dim=1)

        # Predict noise
        noise_pred = self.unet(model_input, timesteps).sample

        # MSE loss on noise
        mse_loss = F.mse_loss(noise_pred, noise)

        # Physics-aware loss: approximate denoised output
        # Use x0 prediction formula: x0 ≈ (x_t - sqrt(1-α̅) * ε) / sqrt(α̅)
        alpha_bar = self.scheduler.alphas_cumprod[timesteps].view(B, 1, 1, 1).to(self.device)
        predicted_x0 = (noisy_target - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()

        last_cond = history_latents[:, -1]  # [B, 4, 64, 64]
        phys_loss = self.mass_loss(predicted_x0, last_cond)

        # Gradient smoothness penalty (discourage sharp artefacts)
        dx = torch.diff(predicted_x0, dim=-1)
        dy = torch.diff(predicted_x0, dim=-2)
        grad_loss = (dx.abs().mean() + dy.abs().mean()) * self.grad_penalty_weight

        # SSIM + MAE on latent x0 vs ground-truth (hybrid fidelity loss)
        ssim_loss = self.ssim_loss(predicted_x0, target_latent)
        mae_loss = F.l1_loss(predicted_x0, target_latent)

        # Temporal consistency loss (optical-flow warping between last cond & prediction)
        temporal_loss = torch.tensor(0.0, device=self.device)
        if self.temporal_loss_enabled:
            temp_result = self.temporal_loss(
                predicted=predicted_x0,
                previous=last_cond,
            )
            temporal_loss = temp_result["temporal_loss"]

        total_loss = (
            mse_loss
            + self.physics_weight * phys_loss
            + grad_loss
            + self.ssim_weight * ssim_loss
            + self.mae_weight * mae_loss
            + (self.temporal_weight * temporal_loss if self.temporal_loss_enabled else 0.0)
        )

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "physics_loss": phys_loss,
            "grad_loss": grad_loss,
            "ssim_loss": ssim_loss,
            "mae_loss": mae_loss,
            "temporal_loss": temporal_loss,
        }

    # ── Inference (sampling) ───────────────────────
    @torch.no_grad()
    def sample(
        self,
        history_latents: torch.Tensor,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate the next latent frame given history.

        Parameters
        ----------
        history_latents : Tensor [B, N_hist, C, H, W]
        num_inference_steps : int, optional

        Returns
        -------
        Tensor [B, C, H, W]  – denoised predicted latent
        """
        self.unet.eval()
        steps = num_inference_steps or self.diff_cfg["inference"]["num_inference_steps"]
        self.scheduler.set_timesteps(steps, device=self.device)

        B = history_latents.size(0)
        cond = history_latents.view(B, -1, *history_latents.shape[-2:])

        # Start from pure noise
        C_out = self.diff_cfg["unet"]["out_channels"]
        H, W = history_latents.shape[-2:]
        x_t = torch.randn(B, C_out, H, W, device=self.device)

        for t in tqdm(self.scheduler.timesteps, desc="Sampling", leave=False):
            model_input = torch.cat([cond, x_t], dim=1)
            t_batch = t.expand(B).to(self.device)
            noise_pred = self.unet(model_input, t_batch).sample
            x_t = self.scheduler.step(noise_pred, t, x_t).prev_sample

        return x_t

    # ── Checkpointing ──────────────────────────────
    def save(self, path: str | Path, epoch: int) -> None:
        path = ensure_dir(Path(path))
        ckpt = {
            "epoch": epoch,
            "unet": self.unet.state_dict(),
            "ema": self.ema.state_dict(),
            "scheduler": self.scheduler.config,
        }
        torch.save(ckpt, path / f"diffusion_epoch{epoch}.pt")
        log.info(f"Saved diffusion checkpoint → epoch {epoch}")

    def load(self, ckpt_path: str | Path) -> int:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.unet.load_state_dict(ckpt["unet"])
        self.ema = EMAModel(self.unet)
        self.ema.shadow = ckpt["ema"]
        log.info(f"Loaded diffusion checkpoint (epoch {ckpt['epoch']})")
        return ckpt["epoch"]
