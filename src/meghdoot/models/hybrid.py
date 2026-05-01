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
        current_state = module.state_dict()
        filtered = {
            key: value
            for key, value in cleaned.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        missing = sorted(set(current_state) - set(filtered))
        skipped = sorted(set(cleaned) - set(filtered))
        if skipped:
            log.warning(
                f"Skipping {len(skipped)} ConvLSTM checkpoint tensors with incompatible shapes; "
                f"loading {len(filtered)} matching tensors instead."
            )
        module.load_state_dict(filtered, strict=False)


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
        
        # Manually move ConvLSTM parameters/buffers since it may not support .to() properly
        for param in self.convlstm.parameters():
            param.data = param.data.to(self.device)
        for buf in self.convlstm.buffers():
            buf.data = buf.data.to(self.device)

        if convlstm_ckpt is not None:
            load_checkpoint_any(self.convlstm, convlstm_ckpt, self.device)
            log.info(f"Loaded ConvLSTM checkpoint: {convlstm_ckpt}")

        if self.freeze_convlstm:
            for parameter in self.convlstm.parameters():
                parameter.requires_grad = False

    def to(self, device: str | torch.device):
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = get_device(device)
        self.vae.to(self.device)
        self.diffusion.to(self.device)
        self.convlstm.to(self.device)
        
        # Manually move ConvLSTM parameters/buffers since it may not support .to() properly
        for param in self.convlstm.parameters():
            param.data = param.data.to(self.device)
        for buf in self.convlstm.buffers():
            buf.data = buf.data.to(self.device)
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
