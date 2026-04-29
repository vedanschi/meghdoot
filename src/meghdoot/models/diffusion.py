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
from diffusers.configuration_utils import FrozenDict
from tqdm import tqdm

from meghdoot.utils.helpers import ensure_dir, get_device
from meghdoot.utils.logging import get_logger
from meghdoot.models.temporal_loss import TemporalConsistencyLoss

log = get_logger(__name__)
torch.serialization.add_safe_globals([FrozenDict])


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

        # Optional latent L1 regularizer (keep small; avoid pixel-structure losses in latent space)
        self.latent_l1_weight = self.diff_cfg.get("training", {}).get("latent_l1_weight", 0.0)

        # Conditioning dropout for CFG-style training (prevents conditional neglect)
        cond_cfg = self.diff_cfg.get("conditioning", {})
        self.cond_dropout_prob = cond_cfg.get("cond_dropout_prob", 0.1)

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

        # Conditioning dropout (classifier-free guidance training)
        if self.cond_dropout_prob > 0:
            keep_mask = (
                torch.rand(B, 1, 1, 1, device=self.device) >= self.cond_dropout_prob
            ).float()
            cond = cond * keep_mask

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

        # Keep latent-space regularization simple and low-weight
        latent_l1_loss = F.l1_loss(predicted_x0, target_latent)

        # Temporal consistency loss (optical-flow warping between last cond & prediction)
        temporal_loss = torch.tensor(0.0, device=self.device)
        if self.temporal_loss_enabled:
            # Extract t-1 and t from the history sequence
            frame_t_minus_1 = history_latents[:, -2]
            frame_t = history_latents[:, -1]
            
            # Pass both frames so Farneback can calculate motion vectors
            temporal_loss = self.temporal_loss(
                predicted=predicted_x0,
                frame_t_minus_1=frame_t_minus_1,
                frame_t=frame_t,
            )

        total_loss = (
            mse_loss
            + self.physics_weight * phys_loss
            + grad_loss
            + self.latent_l1_weight * latent_l1_loss
            + (self.temporal_weight * temporal_loss if self.temporal_loss_enabled else 0.0)
        )

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "physics_loss": phys_loss,
            "grad_loss": grad_loss,
            "latent_l1_loss": latent_l1_loss,
            "temporal_loss": temporal_loss,
        }

    # ── Inference (sampling) ───────────────────────
    @torch.no_grad()
    def sample(
        self,
        history_latents: torch.Tensor,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
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
        cfg_scale = guidance_scale
        if cfg_scale is None:
            cfg_scale = self.diff_cfg.get("inference", {}).get("guidance_scale", 1.0)
        self.scheduler.set_timesteps(steps, device=self.device)

        B = history_latents.size(0)
        cond = history_latents.view(B, -1, *history_latents.shape[-2:])

        # Start from pure noise
        C_out = self.diff_cfg["unet"]["out_channels"]
        H, W = history_latents.shape[-2:]
        x_t = torch.randn(B, C_out, H, W, device=self.device)

        for t in tqdm(self.scheduler.timesteps, desc="Sampling", leave=False):
            t_batch = t.expand(B).to(self.device)

            if cfg_scale is not None and cfg_scale > 1.0:
                cond_input = torch.cat([cond, x_t], dim=1)
                uncond_input = torch.cat([torch.zeros_like(cond), x_t], dim=1)
                noise_pred_cond = self.unet(cond_input, t_batch).sample
                noise_pred_uncond = self.unet(uncond_input, t_batch).sample
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                model_input = torch.cat([cond, x_t], dim=1)
                noise_pred = self.unet(model_input, t_batch).sample

            x_t = self.scheduler.step(noise_pred, t, x_t).prev_sample

        return x_t

    # ── Checkpointing ──────────────────────────────
    def save(self, path: str | Path, epoch: int, optimizer=None, lr_scheduler=None) -> None:
        path = ensure_dir(Path(path))
        ckpt = {
            "epoch": epoch,
            "unet": self.unet.state_dict(),
            "ema": self.ema.state_dict(),
            "scheduler": self.scheduler.config, # Noise scheduler config
        }
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None:
            ckpt["lr_scheduler"] = lr_scheduler.state_dict()
            
        torch.save(ckpt, path / f"diffusion_epoch{epoch}.pt")
        log.info(f"Saved diffusion checkpoint → epoch {epoch}")

    def load(self, ckpt_path: str | Path, optimizer=None, lr_scheduler=None) -> int:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            log.warning(f"Checkpoint {ckpt_path} not found. Starting from scratch.")
            return 0
            
        # CHANGED: Added weights_only=False
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        self.unet.load_state_dict(ckpt["unet"])
        
        self.ema = EMAModel(self.unet)
        if "ema" in ckpt:
            self.ema.shadow = ckpt["ema"]
            
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if lr_scheduler is not None and "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            
        log.info(f"Loaded diffusion checkpoint (epoch {ckpt['epoch']})")
        return ckpt["epoch"]
