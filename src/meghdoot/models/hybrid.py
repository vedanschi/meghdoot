"""Hybrid ConvLSTM + diffusion refinement models."""

from __future__ import annotations

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

        conv_cfg = cfg["evaluation"]["baselines"]["convlstm"]
        self.convlstm = ConvLSTMPredictor(
            in_channels=len(cfg["data"]["channels"]),
            hidden_dims=conv_cfg["hidden_dims"],
            kernel_size=conv_cfg["kernel_size"],
        ).to(self.device)

        if convlstm_ckpt is not None:
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

    def _ensure_batch(self, history_pixels: torch.Tensor) -> torch.Tensor:
        if history_pixels.dim() == 4:
            return history_pixels.unsqueeze(0)
        return history_pixels

    def _encode_history(self, history_pixels: torch.Tensor) -> torch.Tensor:
        history_pixels = self._ensure_batch(history_pixels).to(self.device)
        B, T, C, H, W = history_pixels.shape
        flat_history = history_pixels.reshape(B * T, C, H, W)
        flat_latents = self.vae.encode(flat_history)
        return flat_latents.view(B, T, *flat_latents.shape[1:])

    def _encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.to(self.device)
        return self.vae.encode(pixels)

    @torch.no_grad()
    def predict_base(self, history_pixels: torch.Tensor) -> torch.Tensor:
        history_pixels = self._ensure_batch(history_pixels).to(self.device)
        return self.convlstm(history_pixels)

    def training_step(
        self,
        history_pixels: torch.Tensor,
        target_pixels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        history_pixels = self._ensure_batch(history_pixels).to(self.device)
        target_pixels = target_pixels.to(self.device)

        with torch.no_grad() if self.freeze_convlstm else torch.enable_grad():
            base_pixels = self.convlstm(history_pixels)

        history_latents = self._encode_history(history_pixels)
        target_latent = self._encode_pixels(target_pixels)
        base_latent = self._encode_pixels(base_pixels)

        return self.diffusion.training_step(
            history_latents=history_latents,
            target_latent=target_latent,
            base_latent=base_latent,
        )

    @torch.no_grad()
    def sample(
        self,
        history_pixels: torch.Tensor,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        history_pixels = self._ensure_batch(history_pixels).to(self.device)
        history_latents = self._encode_history(history_pixels)
        base_pixels = self.convlstm(history_pixels)
        base_latent = self._encode_pixels(base_pixels)

        pred_latent = self.diffusion.sample(
            history_latents=history_latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            base_latent=base_latent,
        )
        return self.vae.decode(pred_latent)
