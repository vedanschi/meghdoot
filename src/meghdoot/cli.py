"""
cli.py – Meghdoot-AI Command-Line Interface
============================================

Unified entry point for all project stages.

Usage
-----
    meghdoot download   --config configs/default.yaml
    meghdoot preprocess --config configs/default.yaml
    meghdoot train-vae  --config configs/default.yaml
    meghdoot train      --config configs/default.yaml
    meghdoot evaluate   --config configs/default.yaml --diffusion-ckpt ...
    meghdoot serve      --config configs/default.yaml
"""

from __future__ import annotations

import click

from meghdoot.utils.config import load_config
from meghdoot.utils.logging import get_logger

log = get_logger(__name__)


@click.group()
@click.option("--config", default=None, help="Path to YAML config file")
@click.pass_context
def cli(ctx, config):
    """Meghdoot-AI: Latent Diffusion Weather Nowcasting"""
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config(config)


@cli.command()
@click.option(
    "--mosdac-config",
    default=None,
    help="Path to official MOSDAC config.json (default: config.json at project root)",
)
@click.option(
    "--official",
    is_flag=True,
    help="Run the unmodified official MOSDAC mdapi.py script directly",
)
@click.option("--dataset-id", default=None, help="Override datasetId")
@click.pass_context
def download(ctx, mosdac_config, official, dataset_id):
    """Phase 1 – Download INSAT-3DR/3DS data from MOSDAC.

    Two modes:

    \b
      --official   Run the unmodified MOSDAC mdapi.py + config.json
                   (interactive Y/N prompt, exactly as MOSDAC intended).
    \b
      (default)    Use the programmatic MOSDACClient with auto-pagination,
                   token refresh, and retry logic.

    \b
    Examples:
      meghdoot download --official
      meghdoot download --mosdac-config config.json
      meghdoot download --dataset-id 3RIMG_L1B_STD
    """
    from meghdoot.data.mdapi import MOSDACClient, run_official_mdapi

    cfg = ctx.obj["cfg"]

    if official:
        mc = mosdac_config or cfg["data"].get("mosdac_config", "config.json")
        rc = run_official_mdapi(config_json=mc)
        raise SystemExit(rc)

    # Programmatic mode
    if mosdac_config:
        client = MOSDACClient.from_mosdac_json(mosdac_config)
    else:
        client = MOSDACClient(cfg, mosdac_config=cfg["data"].get("mosdac_config"))

    if not client.authenticate():
        log.error("Cannot proceed without authentication. Exiting.")
        raise SystemExit(1)

    try:
        client.bulk_download(dataset_id=dataset_id)
    finally:
        client.logout()


@cli.command()
@click.pass_context
def preprocess(ctx):
    """Phase 1 – Pre-process raw satellite files to normalised tensors."""
    from meghdoot.data.preprocessing import run_preprocessing

    run_preprocessing(ctx.obj["cfg"])


@cli.command("train-vae")
@click.option("--cache-only", is_flag=True, help="Skip fine-tuning, only cache latents")
@click.pass_context
def train_vae(ctx, cache_only):
    """Phase 2 – Fine-tune VAE and cache latent vectors."""
    from torch.utils.data import DataLoader

    from meghdoot.data.dataset import INSATSequenceDataset
    from meghdoot.models.vae import SatelliteVAE
    from meghdoot.utils.helpers import seed_everything
    from meghdoot.utils.logging import setup_wandb

    cfg = ctx.obj["cfg"]
    seed_everything(cfg["project"]["seed"])
    setup_wandb(cfg)

    dataset = INSATSequenceDataset(
        data_dir=cfg["data"]["paths"]["processed"],
        channel=cfg["data"]["channels"][0],
        num_history=cfg["diffusion"]["conditioning"]["num_history_frames"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["vae"]["fine_tune"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    vae = SatelliteVAE(cfg)

    if not cache_only:
        log.info("═══ Starting VAE Fine-Tuning (SSIM + MAE + VGG) ═══")
        vae.fine_tune(dataloader)

    log.info("═══ Caching Latent Vectors ═══")
    cache_loader = DataLoader(
        dataset,
        batch_size=cfg["vae"]["fine_tune"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    vae.cache_latents(
        cache_loader,
        out_dir=cfg["data"]["paths"]["latents"],
        channel=cfg["data"]["channels"][0],
    )


@cli.command()
@click.option("--resume", default=None, help="Checkpoint path to resume from")
@click.pass_context
def train(ctx, resume):
    """Phase 3 – Train the Latent Diffusion Model."""
    import sys
    sys.argv = ["train_diffusion"]
    if resume:
        sys.argv += ["--resume", resume]

    from meghdoot.training.train_diffusion import main as train_main
    train_main()


@cli.command()
@click.option("--diffusion-ckpt", required=True, help="Diffusion model checkpoint")
@click.option("--convlstm-ckpt", default=None, help="ConvLSTM baseline checkpoint")
@click.option("--n-samples", default=50, type=int)
@click.pass_context
def evaluate(ctx, diffusion_ckpt, convlstm_ckpt, n_samples):
    """Phase 4 – Run comparative evaluation & benchmarking."""
    import sys
    sys.argv = ["benchmark", "--diffusion-ckpt", diffusion_ckpt]
    if convlstm_ckpt:
        sys.argv += ["--convlstm-ckpt", convlstm_ckpt]
    sys.argv += ["--n-samples", str(n_samples)]

    from meghdoot.evaluation.benchmark import main as bench_main
    bench_main()


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000, type=int)
@click.pass_context
def serve(ctx, host, port):
    """Phase 5 – Launch the FastAPI inference server."""
    import uvicorn

    log.info(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "meghdoot.deploy.api:app",
        host=host,
        port=port,
        workers=ctx.obj["cfg"]["deployment"]["api"].get("workers", 2),
    )


@cli.command("train-finetune")
@click.option("--stage", type=click.Choice(["1", "2", "3", "all"]), default="all",
              help="Which fine-tuning stage to run (1/2/3/all)")
@click.option("--resume-stage", default=None, type=int,
              help="Resume from a specific stage (1, 2, or 3)")
@click.pass_context
def train_finetune(ctx, stage, resume_stage):
    """Phase 2b – Run 3-stage VAE fine-tuning pipeline.

    \b
    Stages:
      1  Domain Adaptation   – decoder only, generic satellite data
      2  INSAT Fine-Tuning   – unfreeze encoder, add temporal loss
      3  Regional Specialist – India crops, add physics loss

    \b
    Examples:
      meghdoot train-finetune                     # run all 3 stages
      meghdoot train-finetune --stage 2           # run stage 2 only
      meghdoot train-finetune --resume-stage 2    # resume from stage 2
    """
    from torch.utils.data import DataLoader

    from meghdoot.data.dataset import INSATSequenceDataset
    from meghdoot.training.fine_tuning import ThreeStageFineTuner
    from meghdoot.utils.helpers import seed_everything
    from meghdoot.utils.logging import setup_wandb

    cfg = ctx.obj["cfg"]
    seed_everything(cfg["project"]["seed"])
    setup_wandb(cfg)

    dataset = INSATSequenceDataset(
        data_dir=cfg["data"]["paths"]["processed"],
        channel=cfg["data"]["channels"][0],
        num_history=cfg["diffusion"]["conditioning"]["num_history_frames"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["vae"]["fine_tune"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    finetuner = ThreeStageFineTuner(cfg)

    if stage == "all":
        log.info("═══ Running 3-Stage VAE Fine-Tuning Pipeline ═══")
        finetuner.run_all_stages(dataloader, resume_from_stage=resume_stage)
    else:
        stage_num = int(stage)
        log.info(f"═══ Running Fine-Tuning Stage {stage_num} ═══")
        finetuner.run_stage(stage_num, dataloader)

    log.info("═══ 3-Stage Fine-Tuning Complete ═══")


def main():
    cli()


if __name__ == "__main__":
    main()
