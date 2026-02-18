"""
vae.py – VAE Fine-tuning & Latent Caching
==========================================

Uses HuggingFace ``diffusers.AutoencoderKL`` as the latent encoder.

Workflow
--------
1. Load pre-trained VAE  (``stabilityai/sd-vae-ft-mse``)
2. Fine-tune on INSAT-3DR/3DS frames with a hybrid loss:
   SSIM + MAE + VGG perceptual loss  (as per technical requirements)
3. Cache all frames as 64×64 latent vectors for fast diffusion training
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision import models as tv_models
from tqdm import tqdm

from meghdoot.utils.helpers import ensure_dir, get_device, seed_everything
from meghdoot.utils.logging import get_logger

log = get_logger(__name__)


# ── SSIM Loss (differentiable) ─────────────────────
class SSIMLoss(nn.Module):
    """Differentiable SSIM loss using a Gaussian window."""

    def __init__(self, window_size: int = 11, channels: int = 1) -> None:
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.window = self._create_window(window_size, channels)

    @staticmethod
    def _gaussian(window_size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size: int, channels: int) -> torch.Tensor:
        _1d = self._gaussian(window_size).unsqueeze(1)
        _2d = _1d @ _1d.t()
        window = _2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()
        return window

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        channels = x.size(1)
        window = self.window.to(x.device, x.dtype)
        if channels != self.channels:
            window = self._create_window(self.window_size, channels).to(x.device, x.dtype)

        pad = self.window_size // 2

        mu_x = F.conv2d(x, window, padding=pad, groups=channels)
        mu_y = F.conv2d(y, window, padding=pad, groups=channels)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )
        return 1.0 - ssim_map.mean()  # 0 = perfect


# ── VGG Perceptual Loss ───────────────────────────
class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss for visual realism.

    Extracts features from intermediate VGG-19 layers and computes
    L1 distance between reconstructed and original feature maps.
    This encourages the VAE to preserve high-frequency cloud textures
    that pixel-wise losses (MSE/MAE) tend to blur away.
    """

    def __init__(self, layer_ids: tuple[int, ...] = (3, 8, 17, 26)) -> None:
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.DEFAULT).features
        self.blocks = nn.ModuleList()
        prev = 0
        for lid in layer_ids:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:lid + 1]))
            prev = lid + 1

        # Freeze VGG entirely
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalisation constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Shift from [-1,1] to ImageNet range."""
        return (x * 0.5 + 0.5 - self.mean) / self.std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        y = self._normalize(y)
        loss = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + F.l1_loss(x, y)
        return loss


# ── VAE Wrapper ────────────────────────────────────
class SatelliteVAE:
    """Manages VAE loading, fine-tuning, encoding, and decoding."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.vae_cfg = cfg["vae"]
        self.device = get_device(cfg["project"].get("device", "cuda"))

        # Load pre-trained VAE
        log.info(f"Loading VAE: {self.vae_cfg['pretrained']}")
        self.vae = AutoencoderKL.from_pretrained(
            self.vae_cfg["pretrained"],
            torch_dtype=torch.float32,
        ).to(self.device)

        # Hybrid loss components: SSIM + MAE + VGG perceptual
        self.ssim_loss = SSIMLoss(channels=1).to(self.device)
        self.vgg_loss = VGGPerceptualLoss().to(self.device).eval()

    # ── Encoding / Decoding ────────────────────────
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space image to latent.

        Parameters
        ----------
        x : Tensor [B, C, H, W]
            Pixel images in [-1, 1].  For single-channel data,
            the channel is repeated 3× to match the RGB VAE.

        Returns
        -------
        Tensor [B, 4, H//8, W//8]
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        posterior = self.vae.encode(x).latent_dist
        return posterior.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent back to pixel space.

        Returns
        -------
        Tensor [B, 3, H, W]   (take channel 0 for greyscale)
        """
        z = z / self.vae.config.scaling_factor
        return self.vae.decode(z).sample

    # ── Fine-tune ──────────────────────────────────
    def fine_tune(self, dataloader: DataLoader) -> dict[str, list[float]]:
        """Fine-tune the VAE decoder on INSAT satellite frames.

        Uses a **hybrid loss** as specified in the technical requirements:
          - SSIM  (Structural Similarity Index)
          - MAE   (Mean Absolute Error)
          - VGG   (Perceptual loss for visual realism)

        The encoder is frozen; only the decoder is updated so that
        cloud textures survive the 8× spatial compression.

        Returns
        -------
        dict
            Training history with ``"loss"``, ``"ssim"``, ``"mae"``, ``"vgg"`` lists.
        """
        ft_cfg = self.vae_cfg["fine_tune"]
        epochs = ft_cfg["epochs"]
        lr = ft_cfg["learning_rate"]
        ssim_w = ft_cfg.get("ssim_weight", 0.5)
        mae_w = ft_cfg.get("mae_weight", 0.3)
        vgg_w = ft_cfg.get("vgg_weight", 0.2)

        # Freeze encoder
        for p in self.vae.encoder.parameters():
            p.requires_grad = False
        for p in self.vae.quant_conv.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            [p for p in self.vae.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=ft_cfg.get("weight_decay", 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history: dict[str, list[float]] = {"loss": [], "ssim": [], "mae": [], "vgg": []}
        self.vae.train()

        for epoch in range(1, epochs + 1):
            epoch_loss, epoch_ssim, epoch_mae, epoch_vgg = 0.0, 0.0, 0.0, 0.0

            for batch in tqdm(dataloader, desc=f"VAE Epoch {epoch}/{epochs}", leave=False):
                # batch["target"] is [B, H, W]; add channel dim
                x = batch["target"].unsqueeze(1).to(self.device)  # [B, 1, H, W]
                x_rgb = x.repeat(1, 3, 1, 1)                      # [B, 3, H, W]

                # Forward
                posterior = self.vae.encode(x_rgb).latent_dist
                z = posterior.sample()
                recon = self.vae.decode(z).sample  # [B, 3, H, W]

                # Greyscale proxy for SSIM/MAE (channel-0)
                recon_grey = recon[:, :1, :, :]  # [B, 1, H, W]

                # ── Hybrid loss: SSIM + MAE + VGG perceptual ──
                loss_ssim = self.ssim_loss(recon_grey, x)
                loss_mae = F.l1_loss(recon_grey, x)
                loss_vgg = self.vgg_loss(recon, x_rgb)

                loss = ssim_w * loss_ssim + mae_w * loss_mae + vgg_w * loss_vgg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_ssim += loss_ssim.item()
                epoch_mae += loss_mae.item()
                epoch_vgg += loss_vgg.item()

            scheduler.step()
            n = len(dataloader)
            history["loss"].append(epoch_loss / n)
            history["ssim"].append(epoch_ssim / n)
            history["mae"].append(epoch_mae / n)
            history["vgg"].append(epoch_vgg / n)

            log.info(
                f"Epoch {epoch:3d} │ loss={epoch_loss/n:.5f}  "
                f"SSIM={epoch_ssim/n:.5f}  MAE={epoch_mae/n:.5f}  VGG={epoch_vgg/n:.5f}"
            )

            # Checkpoint every 5 epochs
            if epoch % 5 == 0:
                self._save_checkpoint(epoch)

        self._save_checkpoint(epochs, tag="final")
        return history

    def _save_checkpoint(self, epoch: int, tag: str | None = None) -> None:
        ckpt_dir = ensure_dir(self.vae_cfg["checkpoint_dir"])
        name = f"vae_epoch{epoch}.pt" if tag is None else f"vae_{tag}.pt"
        self.vae.save_pretrained(ckpt_dir / name)
        log.info(f"  ↳ saved checkpoint → {ckpt_dir / name}")

    # ── Latent Caching ─────────────────────────────
    @torch.no_grad()
    def cache_latents(
        self,
        dataloader: DataLoader,
        out_dir: str | Path,
        channel: str = "TIR1",
    ) -> int:
        """Encode the entire dataset and save latent vectors to disk.

        This is the **critical speed optimisation**: once cached, the
        diffusion training loop skips the VAE encoder entirely.

        Parameters
        ----------
        dataloader : DataLoader
            Yields dicts with ``"target"`` (pixel frames) and ``"timestamps"``.
        out_dir : str or Path
            Root directory for saved latents.
        channel : str
            Sub-directory name.

        Returns
        -------
        int
            Number of latent files written.
        """
        self.vae.eval()
        save_dir = ensure_dir(Path(out_dir) / channel)
        count = 0

        for batch in tqdm(dataloader, desc="Caching latents"):
            x = batch["target"].unsqueeze(1).to(self.device)
            x_rgb = x.repeat(1, 3, 1, 1)

            posterior = self.vae.encode(x_rgb).latent_dist
            z = posterior.sample() * self.vae.config.scaling_factor  # [B, 4, 64, 64]

            timestamps = batch["timestamps"]
            # timestamps is list of lists; take the last (target) name
            for i in range(z.size(0)):
                name = timestamps[-1][i] if isinstance(timestamps[-1], list) else timestamps[-1]
                np.save(save_dir / f"{name}.npy", z[i].cpu().numpy())
                count += 1

        log.info(f"Cached {count} latent tensors → {save_dir}")
        return count
