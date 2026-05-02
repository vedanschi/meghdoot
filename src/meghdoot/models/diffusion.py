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
    
    Normalized by L2 norm to prevent huge raw values destabilizing training.
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
            Normalized mean absolute difference in per-channel spatial sums.
        """
        mass_pred = predicted.sum(dim=(-2, -1))      # [B, C]
        mass_cond = last_condition.sum(dim=(-2, -1))  # [B, C]
        
        # Normalize by L2 norm to keep loss in reasonable range (~0.01-0.1)
        # Prevents raw latent magnitude from dominating the loss
        norm_factor = (mass_cond.abs().mean() + 1e-8)  # avoid division by zero
        return ((mass_pred - mass_cond).abs() / norm_factor).mean()


# ── EMA Helper ─────────────────────────────────────
class EMAModel:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        # Store shadow weights on the same device as the model parameters
        try:
            param_device = next(model.parameters()).device
        except StopIteration:
            param_device = torch.device("cpu")
        self.shadow = {k: v.clone().to(param_device) for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            # Ensure shadow and current param are on the same device
            if self.shadow[k].device != v.device:
                self.shadow[k] = self.shadow[k].to(v.device)
            # Ensure dtypes match (avoid unexpected type promotion)
            if self.shadow[k].dtype != v.dtype:
                self.shadow[k] = self.shadow[k].to(v.dtype)
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
        self.mass_loss = MassConservationLoss().to(self.device)
        self.physics_weight = self.diff_cfg["physics_loss"].get("mass_conservation_weight", 0.1)
        self.grad_penalty_weight = self.diff_cfg["physics_loss"].get("gradient_penalty_weight", 0.05)

        # Optional latent L1 regularizer (keep small; avoid pixel-structure losses in latent space)
        self.latent_l1_weight = self.diff_cfg.get("training", {}).get("latent_l1_weight", 0.0)

        # Forecasting-focused options
        train_cfg = self.diff_cfg.get("training", {})
        # Residual mode predicts (x_t - x_{t-1}) in latent space, then reconstructs x_t.
        self.use_residual_prediction = train_cfg.get("residual_prediction", False)
        # Direct x0 reconstruction supervision prevents drift toward texture-like mean outputs.
        self.x0_recon_weight = train_cfg.get("x0_recon_weight", 0.0)
        # Event focus up-weights high-magnitude latent regions to improve extremes/contrast.
        self.event_focus_weight = train_cfg.get("event_focus_weight", 0.0)
        self.event_focus_threshold = train_cfg.get("event_focus_threshold", 0.35)
        # Morphology/texture losses in latent space.
        self.edge_loss_weight = train_cfg.get("edge_loss_weight", 0.0)
        # Contrast matching aligns per-channel latent standard deviation.
        self.contrast_loss_weight = train_cfg.get("contrast_loss_weight", 0.0)
        # Extreme value loss: explicitly penalize not matching bright/high-magnitude pixels
        # This prevents mean-regression and encourages capturing storms/clouds with white peaks.
        self.extreme_value_loss_weight = train_cfg.get("extreme_value_loss_weight", 0.0)
        self.extreme_value_threshold = train_cfg.get("extreme_value_threshold", 0.5)

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
            ).to(self.device)
            self.temporal_weight = temp_cfg.get("temporal_weight", 0.05)
            log.info("Temporal consistency loss ENABLED")

        # EMA
        self.ema = EMAModel(self.unet, decay=self.diff_cfg["training"]["ema_decay"])

        log.info(
            f"MeghdootDiffusion initialised: "
            f"{sum(p.numel() for p in self.unet.parameters())/1e6:.1f}M params"
        )

    def _prepare_conditioning(
        self,
        history_latents: torch.Tensor,
        base_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the conditioning tensor and residual reference latent.

        When ``base_latent`` is provided, the last history frame is replaced by the
        ConvLSTM forecast so the UNet still sees a 3-frame conditioning stack while
        the residual target is anchored to the forecast instead of the raw last input.
        """
        B = history_latents.size(0)

        if base_latent is not None:
            if base_latent.dim() == 3:
                base_latent = base_latent.unsqueeze(0)
            if base_latent.dim() != 4:
                raise ValueError("base_latent must have shape [B, C, H, W] or [C, H, W]")
            if base_latent.size(0) != B:
                if base_latent.size(0) == 1:
                    base_latent = base_latent.expand(B, -1, -1, -1)
                else:
                    raise ValueError("base_latent batch size must match history_latents")

            if history_latents.size(1) > 1:
                cond_frames = torch.cat([history_latents[:, :-1], base_latent.unsqueeze(1)], dim=1)
            else:
                cond_frames = base_latent.unsqueeze(1)

            cond = cond_frames.view(B, -1, *cond_frames.shape[-2:])
            return cond, base_latent

        cond = history_latents.view(B, -1, *history_latents.shape[-2:])
        return cond, history_latents[:, -1]

    # ── Training Step ──────────────────────────────
    def training_step(
        self,
        history_latents: torch.Tensor,
        target_latent: torch.Tensor,
        base_latent: torch.Tensor | None = None,
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

        # Ensure inputs live on the model device to avoid mixed-device ops
        if history_latents.device != self.device:
            history_latents = history_latents.to(self.device)
        if target_latent.device != self.device:
            target_latent = target_latent.to(self.device)
        if base_latent is not None and base_latent.device != self.device:
            base_latent = base_latent.to(self.device)

        # Flatten history: [B, 3, 4, 64, 64] → [B, 12, 64, 64]
        cond, reference_latent = self._prepare_conditioning(history_latents, base_latent)

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

        # Add noise to either absolute target or residual target.
        if self.use_residual_prediction:
            diffusion_target = target_latent - reference_latent
        else:
            diffusion_target = target_latent

        # Add noise to diffusion target
        noise = torch.randn_like(target_latent)
        noisy_target = self.scheduler.add_noise(diffusion_target, noise, timesteps)

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
        predicted_x0_base = (noisy_target - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()

        if self.use_residual_prediction:
            predicted_x0 = predicted_x0_base + reference_latent
        else:
            predicted_x0 = predicted_x0_base

        # Clamp predicted_x0 tightly: latents are [-1, 1], so allow [-3, 3] conservatively
        # (NOT [-10, 10] which allows 10x larger values and destabilizes losses)
        predicted_x0 = torch.clamp(predicted_x0, -3.0, 3.0)
        
        # PHYSICS LOSS SCHEDULING: Only apply at low timesteps (high alpha_bar)
        # where x0 reconstruction is meaningful and stable.
        # At t=0 (alpha_bar≈1), x0 is nearly the true image. At t=1000 (alpha_bar≈0), it's mostly noise.
        # Threshold: only apply physics when alpha_bar > 0.1 (very low noise, where x0 is reliable)
        physics_mask = (alpha_bar > 0.1).float()  # shape [B, 1, 1, 1]
        if physics_mask.mean() > 0:
            phys_loss = self.mass_loss(predicted_x0, reference_latent)
            phys_loss = phys_loss * physics_mask.mean()  # scale by proportion of valid timesteps
        else:
            phys_loss = torch.tensor(0.0, device=self.device)  # no valid timesteps, zero out

        # Gradient smoothness penalty (discourage sharp artefacts)
        # Divided by 2.0 to normalize gradient components
        dx = torch.diff(predicted_x0, dim=-1)
        dy = torch.diff(predicted_x0, dim=-2)
        grad_loss = ((dx.abs().mean() + dy.abs().mean()) / 2.0) * self.grad_penalty_weight

        # Keep latent-space regularization simple and low-weight
        # Only apply when significantly off-target (not during high-noise timesteps where x0 is garbage)
        latent_l1_loss = F.l1_loss(predicted_x0, target_latent) * physics_mask.mean()

        # Direct x0 reconstruction objective with optional event up-weighting.
        x0_recon_loss = torch.tensor(0.0, device=self.device)
        if self.x0_recon_weight > 0:
            if self.event_focus_weight > 0:
                # Weight map in latent space: emphasize larger-magnitude structures.
                event_map = (target_latent.abs() > self.event_focus_threshold).float()
                weight_map = 1.0 + self.event_focus_weight * event_map
                x0_recon_loss = ((predicted_x0 - target_latent).abs() * weight_map).mean()
            else:
                x0_recon_loss = F.l1_loss(predicted_x0, target_latent)
            x0_recon_loss = x0_recon_loss * physics_mask.mean()

        # Edge-consistency loss helps preserve morphology and sharp cloud boundaries.
        edge_loss = torch.tensor(0.0, device=self.device)
        if self.edge_loss_weight > 0:
            # Sobel filters applied per channel.
            sobel_x = torch.tensor(
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                device=self.device,
                dtype=predicted_x0.dtype,
            ).unsqueeze(0)
            sobel_y = torch.tensor(
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
                device=self.device,
                dtype=predicted_x0.dtype,
            ).unsqueeze(0)
            c = predicted_x0.shape[1]
            sobel_x = sobel_x.expand(c, 1, 3, 3)
            sobel_y = sobel_y.expand(c, 1, 3, 3)

            pred_gx = F.conv2d(predicted_x0, sobel_x, padding=1, groups=c)
            pred_gy = F.conv2d(predicted_x0, sobel_y, padding=1, groups=c)
            tgt_gx = F.conv2d(target_latent, sobel_x, padding=1, groups=c)
            tgt_gy = F.conv2d(target_latent, sobel_y, padding=1, groups=c)
            edge_loss = (F.l1_loss(pred_gx, tgt_gx) + F.l1_loss(pred_gy, tgt_gy)) * 0.5
            edge_loss = edge_loss * physics_mask.mean()

        # Contrast loss prevents gray/low-dynamic-range collapse.
        contrast_loss = torch.tensor(0.0, device=self.device)
        if self.contrast_loss_weight > 0:
            # Compute per-sample, per-channel std over spatial dimensions.
            pred_std = predicted_x0.flatten(2).std(dim=-1)
            tgt_std = target_latent.flatten(2).std(dim=-1)
            contrast_loss = F.l1_loss(pred_std, tgt_std)
            contrast_loss = contrast_loss * physics_mask.mean()

        # Extreme value loss: explicitly penalize missing bright/high-magnitude pixels.
        # This directly targets the "gray/washed-out" problem by forcing the model to
        # match extreme values (white clouds, storm peaks) instead of converging to smooth means.
        extreme_value_loss = torch.tensor(0.0, device=self.device)
        if self.extreme_value_loss_weight > 0:
            # Create mask for high-magnitude regions in target
            extreme_mask = (target_latent.abs() > self.extreme_value_threshold).float()
            if extreme_mask.sum() > 0:
                # Only penalize extreme regions where they exist in target
                extreme_loss_val = (F.l1_loss(predicted_x0 * extreme_mask, target_latent * extreme_mask)
                                    / extreme_mask.sum().clamp(min=1e-6))
                extreme_value_loss = extreme_loss_val * physics_mask.mean()
            else:
                extreme_value_loss = torch.tensor(0.0, device=self.device)

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
            + self.x0_recon_weight * x0_recon_loss
            + self.edge_loss_weight * edge_loss
            + self.contrast_loss_weight * contrast_loss
            + self.extreme_value_loss_weight * extreme_value_loss
            + (self.temporal_weight * temporal_loss if self.temporal_loss_enabled else 0.0)
        )

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "physics_loss": phys_loss,
            "grad_loss": grad_loss,
            "latent_l1_loss": latent_l1_loss,
            "x0_recon_loss": x0_recon_loss,
            "edge_loss": edge_loss,
            "contrast_loss": contrast_loss,
            "extreme_value_loss": extreme_value_loss,
            "temporal_loss": temporal_loss,
        }

    # ── Inference (sampling) ───────────────────────
    @torch.no_grad()
    def sample(
        self,
        history_latents: torch.Tensor,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        base_latent: torch.Tensor | None = None,
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

        history_latents = history_latents.to(self.device)
        self.scheduler.set_timesteps(steps, device=self.device)

        B = history_latents.size(0)
        cond, reference_latent = self._prepare_conditioning(history_latents, base_latent)

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

        if self.use_residual_prediction:
            return x_t + reference_latent
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
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as exc:
                log.warning(
                    "Skipping optimizer state load due to parameter-group mismatch: %s", exc
                )
        if lr_scheduler is not None and "lr_scheduler" in ckpt:
            try:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            except Exception as exc:
                log.warning(
                    "Skipping LR scheduler state load due to incompatibility: %s", exc
                )
            
        log.info(f"Loaded diffusion checkpoint (epoch {ckpt['epoch']})")
        return ckpt["epoch"]

    # ── Device / Mode Helpers ─────────────────────
    def to(self, device: str | torch.device):
        """Move the internal modules and loss helpers to `device`.

        Returns self for chaining (e.g., `model.to(device)`).
        """
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = get_device(device)
        self.unet.to(self.device)
        try:
            self.mass_loss.to(self.device)
        except Exception:
            # mass_loss may be stateless; ignore if cannot move
            pass
        if getattr(self, "temporal_loss_enabled", False):
            try:
                self.temporal_loss.to(self.device)
            except Exception:
                pass
        # Scheduler/EMA live on CPU by design; return self for convenience
        return self

    def train(self, mode: bool = True):
        """Set training/eval mode on the wrapped UNet."""
        self.unet.train(mode)
        return self

    def eval(self):
        """Shortcut to set the UNet to evaluation mode."""
        self.unet.eval()
        return self
