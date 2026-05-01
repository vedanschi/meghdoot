"""Hybrid ConvLSTM + diffusion refinement models."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from meghdoot.evaluation.baselines import ConvLSTMPredictor
from meghdoot.models.diffusion import MeghdootDiffusion
from meghdoot.models.vae import SatelliteVAE
from meghdoot.utils.helpers import get_device
from meghdoot.utils.logging import get_logger


log = get_logger(__name__)


def load_checkpoint_any(module: torch.nn.Module, ckpt_path: str | Path, device: torch.device) -> None:
    state = torch.load(Path(ckpt_path), map_location=device, weights_only=True)

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


def _inflate_conv_input_weights(weight: torch.Tensor, target_in_channels: int) -> torch.Tensor:
    """Expand a conv kernel from a smaller input channel count to a larger one."""
    out_channels, in_channels, kh, kw = weight.shape
    if in_channels == target_in_channels:
        return weight

    if in_channels > target_in_channels:
        return weight[:, :target_in_channels].contiguous()

    expanded = torch.zeros(out_channels, target_in_channels, kh, kw, device=weight.device, dtype=weight.dtype)
    expanded[:, :in_channels] = weight
    if in_channels > 0:
        repeat_slice = weight.mean(dim=1, keepdim=True)
        for channel in range(in_channels, target_in_channels):
            expanded[:, channel : channel + 1] = repeat_slice
    return expanded


def _adapt_convlstm_state_dict(state_dict: dict[str, torch.Tensor], target_in_channels: int, target_out_channels: int) -> dict[str, torch.Tensor]:
    """Adapt a 2-channel ConvLSTM checkpoint to a wider latent-space model."""
    adapted = copy.deepcopy(state_dict)

    first_conv_key = "encoder_cells.0.conv.weight"
    if first_conv_key in adapted:
        adapted[first_conv_key] = _inflate_conv_input_weights(adapted[first_conv_key], target_in_channels)

    final_weight_key = "decoder.2.weight"
    final_bias_key = "decoder.2.bias"
    if final_weight_key in adapted:
        old_weight = adapted[final_weight_key]
        out_channels, in_channels, kh, kw = old_weight.shape
        if out_channels != target_out_channels:
            new_weight = torch.zeros(target_out_channels, in_channels, kh, kw, device=old_weight.device, dtype=old_weight.dtype)
            copy_count = min(out_channels, target_out_channels)
            new_weight[:copy_count] = old_weight[:copy_count]
            if target_out_channels > out_channels:
                fill = old_weight.mean(dim=0, keepdim=True)
                for channel in range(out_channels, target_out_channels):
                    new_weight[channel : channel + 1] = fill
            adapted[final_weight_key] = new_weight

    if final_bias_key in adapted:
        old_bias = adapted[final_bias_key]
        if old_bias.shape[0] != target_out_channels:
            new_bias = torch.zeros(target_out_channels, device=old_bias.device, dtype=old_bias.dtype)
            copy_count = min(old_bias.shape[0], target_out_channels)
            new_bias[:copy_count] = old_bias[:copy_count]
            if target_out_channels > old_bias.shape[0]:
                new_bias[old_bias.shape[0] :] = old_bias.mean()
            adapted[final_bias_key] = new_bias

    return adapted


class ConvLSTMDiffusionHybrid:
    """ConvLSTM motion prior followed by DDPM refinement in latent space."""

    def __init__(
        self,
        cfg: dict,
        convlstm_ckpt: str | Path | None = None,
        freeze_convlstm: bool = True,
    ) -> None:
        self.cfg = cfg
        self.device = get_device(cfg["project"].get("device", "cuda"))
        self.freeze_convlstm = freeze_convlstm

        self.vae = SatelliteVAE(cfg).to(self.device)
        self.diffusion = MeghdootDiffusion(cfg).to(self.device)

        latent_channels = cfg["vae"].get("latent_channels", cfg["diffusion"]["unet"]["out_channels"])
        conv_cfg = cfg["evaluation"]["baselines"]["convlstm"]
        self.convlstm = ConvLSTMPredictor(
            in_channels=latent_channels,
            hidden_dims=conv_cfg["hidden_dims"],
            kernel_size=conv_cfg["kernel_size"],
        ).to(self.device)

        if convlstm_ckpt is not None:
            state = torch.load(Path(convlstm_ckpt), map_location=self.device, weights_only=True)
            if isinstance(state, dict):
                for key in ("state_dict", "model_state_dict", "ema_state_dict", "model", "weights"):
                    if key in state and isinstance(state[key], dict):
                        state = state[key]
                        break
            if isinstance(state, dict):
                state = _adapt_convlstm_state_dict(state, latent_channels, latent_channels)
                try:
                    self.convlstm.load_state_dict(state, strict=True)
                except RuntimeError:
                    self.convlstm.load_state_dict(state, strict=False)
            else:
                load_checkpoint_any(self.convlstm, convlstm_ckpt, self.device)
            log.info(f"Loaded ConvLSTM checkpoint: {convlstm_ckpt}")

        if self.freeze_convlstm:
            for parameter in self.convlstm.parameters():
                parameter.requires_grad = False

    def to(self, device: str | torch.device):
        self.device = get_device(device)
        self.vae.to(self.device)
        self.diffusion.to(self.device)
        self.convlstm.to(self.device)
        return self

    def train(self, mode: bool = True):
        self.diffusion.train(mode)
        self.convlstm.train(mode and not self.freeze_convlstm)
        return self

    def eval(self):
        self.diffusion.eval()
        self.convlstm.eval()
        self.vae.eval()
        return self

    def _ensure_batch(self, history_latents: torch.Tensor) -> torch.Tensor:
        if history_latents.dim() == 4:
            return history_latents.unsqueeze(0)
        return history_latents

    @torch.no_grad()
    def predict_base(self, history_latents: torch.Tensor) -> torch.Tensor:
        history_latents = self._ensure_batch(history_latents).to(self.device)
        return self.convlstm(history_latents)

    def training_step(
        self,
        history_latents: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        history_latents = self._ensure_batch(history_latents).to(self.device)
        target_latent = target_latent.to(self.device)

        with torch.no_grad() if self.freeze_convlstm else torch.enable_grad():
            base_latent = self.convlstm(history_latents)

        return self.diffusion.training_step(
            history_latents=history_latents,
            target_latent=target_latent,
            base_latent=base_latent,
        )

    @torch.no_grad()
    def sample(
        self,
        history_latents: torch.Tensor,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        history_latents = self._ensure_batch(history_latents).to(self.device)
        base_latent = self.convlstm(history_latents)

        pred_latent = self.diffusion.sample(
            history_latents=history_latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            base_latent=base_latent,
        )
        return pred_latent
