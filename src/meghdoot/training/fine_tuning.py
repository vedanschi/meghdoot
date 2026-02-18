"""
fine_tuning.py – 3-Stage Fine-Tuning Pipeline
===============================================

Implements the 3-stage transfer learning strategy from the
Meghdoot architecture diagram:

    Stage 1: Domain Adaptation
        Fine-tune the pretrained Stable Diffusion VAE on general
        satellite imagery.  Encoder frozen, decoder trained with
        SSIM + MAE + VGG loss.  This teaches the VAE to reconstruct
        cloud textures instead of natural images.

    Stage 2: INSAT Fine-Tuning
        Continue fine-tuning on specifically INSAT-3DR/3DS data with
        a lower learning rate.  The Channel Integration Layer is
        trained jointly.  This adapts to the specific spectral
        characteristics and noise patterns of INSAT sensors.

    Stage 3: Regional Specialisation
        Final fine-tuning on India-region-cropped data with the
        smallest learning rate.  Uses physics-aware augmentations
        (monsoon-season weighting, day/night split, land-sea mask).
        This specialises the model for Indian subcontinent weather.

Each stage uses progressively lower learning rates and can optionally
unfreeze more VAE layers for deeper adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from meghdoot.models.vae import SatelliteVAE, SSIMLoss, VGGPerceptualLoss
from meghdoot.models.channel_fusion import ChannelIntegrationLayer
from meghdoot.utils.helpers import ensure_dir, seed_everything
from meghdoot.utils.logging import get_logger, setup_wandb

log = get_logger(__name__)


@dataclass
class StageConfig:
    """Configuration for a single fine-tuning stage."""

    name: str
    epochs: int
    learning_rate: float
    weight_decay: float = 1e-4
    unfreeze_encoder: bool = False
    ssim_weight: float = 0.5
    mae_weight: float = 0.3
    vgg_weight: float = 0.2
    description: str = ""


# ── Default 3-stage schedule ───────────────────────
DEFAULT_STAGES = [
    StageConfig(
        name="domain_adaptation",
        epochs=15,
        learning_rate=1e-4,
        unfreeze_encoder=False,
        description=(
            "Stage 1: Adapt pretrained SD-VAE from natural images → "
            "satellite imagery.  Encoder frozen, decoder learns cloud textures."
        ),
    ),
    StageConfig(
        name="insat_fine_tuning",
        epochs=10,
        learning_rate=1e-5,
        unfreeze_encoder=False,
        description=(
            "Stage 2: Specialise on INSAT-3DR/3DS sensor characteristics. "
            "Channel Integration Layer trained jointly."
        ),
    ),
    StageConfig(
        name="regional_specialisation",
        epochs=5,
        learning_rate=1e-6,
        unfreeze_encoder=True,  # shallow unfreeze of last encoder blocks
        description=(
            "Stage 3: Final adaptation to India-region cropped data. "
            "Encoder partially unfrozen for deepest adaptation."
        ),
    ),
]


class ThreeStageFineTuner:
    """Orchestrates the 3-stage VAE fine-tuning pipeline.

    Parameters
    ----------
    cfg : dict
        Full Meghdoot config.
    vae : SatelliteVAE
        The VAE wrapper (encoder + decoder).
    channel_layer : ChannelIntegrationLayer, optional
        The multi-channel fusion module.  If provided, it is
        trained jointly from Stage 2 onward.
    stages : list[StageConfig], optional
        Custom stage definitions.  Defaults to 3 stages.
    """

    def __init__(
        self,
        cfg: dict,
        vae: SatelliteVAE,
        channel_layer: ChannelIntegrationLayer | None = None,
        stages: list[StageConfig] | None = None,
    ) -> None:
        self.cfg = cfg
        self.vae = vae
        self.channel_layer = channel_layer
        self.stages = stages or DEFAULT_STAGES
        self.device = vae.device
        self.checkpoint_dir = ensure_dir(cfg["vae"]["checkpoint_dir"])

        # Loss functions
        self.ssim_loss = SSIMLoss(channels=1).to(self.device)
        self.vgg_loss = VGGPerceptualLoss().to(self.device).eval()

    def _get_trainable_params(
        self,
        stage: StageConfig,
    ) -> list[torch.nn.Parameter]:
        """Collect parameters to optimise for this stage."""
        params = []

        # Decoder is always trained
        params.extend(
            p for p in self.vae.vae.decoder.parameters() if p.requires_grad
        )
        params.extend(
            p for p in self.vae.vae.post_quant_conv.parameters()
        )

        # Channel Integration Layer (Stage 2+)
        if self.channel_layer is not None and stage.name != "domain_adaptation":
            params.extend(self.channel_layer.parameters())

        # Partial encoder unfreeze (Stage 3)
        if stage.unfreeze_encoder:
            # Unfreeze only the last 2 up/mid blocks of the encoder
            encoder = self.vae.vae.encoder
            # Unfreeze mid_block and the last down_block
            for p in encoder.mid_block.parameters():
                p.requires_grad = True
                params.append(p)
            if hasattr(encoder, "down_blocks") and len(encoder.down_blocks) > 0:
                for p in encoder.down_blocks[-1].parameters():
                    p.requires_grad = True
                    params.append(p)

        return params

    def _freeze_all(self) -> None:
        """Freeze everything before selectively unfreezing."""
        for p in self.vae.vae.parameters():
            p.requires_grad = False
        if self.channel_layer is not None:
            for p in self.channel_layer.parameters():
                p.requires_grad = False

    def _unfreeze_decoder(self) -> None:
        """Unfreeze decoder + post_quant_conv."""
        for p in self.vae.vae.decoder.parameters():
            p.requires_grad = True
        for p in self.vae.vae.post_quant_conv.parameters():
            p.requires_grad = True

    def run_stage(
        self,
        stage: StageConfig,
        dataloader: DataLoader,
        stage_num: int,
    ) -> dict[str, list[float]]:
        """Execute a single fine-tuning stage.

        Parameters
        ----------
        stage : StageConfig
        dataloader : DataLoader
        stage_num : int
            1-indexed stage number (for logging).

        Returns
        -------
        dict
            Per-epoch loss history.
        """
        log.info(f"\n{'═' * 60}")
        log.info(f"Stage {stage_num}/3: {stage.name}")
        log.info(f"  {stage.description}")
        log.info(f"  epochs={stage.epochs}  lr={stage.learning_rate}")
        log.info(f"{'═' * 60}")

        # Freeze/unfreeze
        self._freeze_all()
        self._unfreeze_decoder()

        if self.channel_layer is not None and stage.name != "domain_adaptation":
            for p in self.channel_layer.parameters():
                p.requires_grad = True

        if stage.unfreeze_encoder:
            encoder = self.vae.vae.encoder
            for p in encoder.mid_block.parameters():
                p.requires_grad = True
            if hasattr(encoder, "down_blocks") and len(encoder.down_blocks) > 0:
                for p in encoder.down_blocks[-1].parameters():
                    p.requires_grad = True

        # Collect trainable params
        params = [p for p in self.vae.vae.parameters() if p.requires_grad]
        if self.channel_layer is not None:
            params.extend(p for p in self.channel_layer.parameters() if p.requires_grad)

        n_trainable = sum(p.numel() for p in params)
        log.info(f"  Trainable parameters: {n_trainable:,}")

        optimizer = torch.optim.AdamW(
            params,
            lr=stage.learning_rate,
            weight_decay=stage.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage.epochs
        )

        history: dict[str, list[float]] = {
            "loss": [], "ssim": [], "mae": [], "vgg": [],
        }

        self.vae.vae.train()
        if self.channel_layer is not None:
            self.channel_layer.train()

        from tqdm import tqdm

        for epoch in range(1, stage.epochs + 1):
            epoch_loss = epoch_ssim = epoch_mae = epoch_vgg = 0.0

            for batch in tqdm(
                dataloader,
                desc=f"  [{stage.name}] Epoch {epoch}/{stage.epochs}",
                leave=False,
            ):
                target = batch["target"].to(self.device)

                # Handle multi-channel vs single-channel input
                if target.dim() == 3:
                    # [B, H, W] → [B, 1, H, W] → [B, 3, H, W]
                    x = target.unsqueeze(1).repeat(1, 3, 1, 1)
                elif target.dim() == 4 and target.shape[1] == 1:
                    x = target.repeat(1, 3, 1, 1)
                elif target.dim() == 4 and target.shape[1] > 3:
                    # Multi-channel: use Channel Integration Layer
                    if self.channel_layer is not None:
                        x = self.channel_layer(target)  # [B, N, H, W] → [B, 3, H, W]
                    else:
                        x = target[:, :3]  # fallback: take first 3
                else:
                    x = target

                # VAE forward
                posterior = self.vae.vae.encode(x).latent_dist
                z = posterior.sample()
                recon = self.vae.vae.decode(z).sample

                # Greyscale proxy for SSIM/MAE
                recon_grey = recon[:, :1]
                x_grey = x[:, :1]

                # Hybrid loss
                loss_ssim = self.ssim_loss(recon_grey, x_grey)
                loss_mae = F.l1_loss(recon_grey, x_grey)
                loss_vgg = self.vgg_loss(recon, x)

                loss = (
                    stage.ssim_weight * loss_ssim
                    + stage.mae_weight * loss_mae
                    + stage.vgg_weight * loss_vgg
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_ssim += loss_ssim.item()
                epoch_mae += loss_mae.item()
                epoch_vgg += loss_vgg.item()

            scheduler.step()
            n = max(len(dataloader), 1)
            history["loss"].append(epoch_loss / n)
            history["ssim"].append(epoch_ssim / n)
            history["mae"].append(epoch_mae / n)
            history["vgg"].append(epoch_vgg / n)

            log.info(
                f"  Epoch {epoch:3d} │ loss={epoch_loss/n:.5f}  "
                f"SSIM={epoch_ssim/n:.5f}  MAE={epoch_mae/n:.5f}  VGG={epoch_vgg/n:.5f}"
            )

        # Save stage checkpoint
        tag = f"stage{stage_num}_{stage.name}"
        ckpt_path = self.checkpoint_dir / f"vae_{tag}.pt"
        self.vae.vae.save_pretrained(str(ckpt_path))
        if self.channel_layer is not None:
            torch.save(
                self.channel_layer.state_dict(),
                self.checkpoint_dir / f"channel_fusion_{tag}.pt",
            )
        log.info(f"  ↳ Checkpoint saved → {ckpt_path}")

        return history

    def run_all(
        self,
        dataloaders: dict[str, DataLoader],
    ) -> dict[str, dict[str, list[float]]]:
        """Run all 3 stages sequentially.

        Parameters
        ----------
        dataloaders : dict
            Keys should include ``"domain_adaptation"``,
            ``"insat_fine_tuning"``, ``"regional_specialisation"``.
            If a key is missing, that stage's dataloader falls back
            to ``"default"`` or the first available.

        Returns
        -------
        dict
            Nested history: ``{stage_name: {metric: [values]}}``.
        """
        all_history: dict[str, dict[str, list[float]]] = {}

        for i, stage in enumerate(self.stages, 1):
            # Pick the right dataloader for this stage
            dl = dataloaders.get(
                stage.name,
                dataloaders.get("default", list(dataloaders.values())[0]),
            )
            history = self.run_stage(stage, dl, stage_num=i)
            all_history[stage.name] = history

        log.info("\n✓ All 3 fine-tuning stages complete!")
        return all_history
