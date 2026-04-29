#!/usr/bin/env python3
"""
Preflight validation script for Meghdoot-AI training.
Run this once before starting long training runs to catch setup issues.

Usage:
    python preflight.py
"""

import sys
import os
from pathlib import Path

def check(condition, name, success_msg="✓", fail_msg=""):
    """Print a check result."""
    if condition:
        print(f"  ✓ {name}")
        return True
    else:
        status = fail_msg if fail_msg else "✗ FAILED"
        print(f"  ✗ {name} — {status}")
        return False

def main():
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    print("\n" + "="*70)
    print("MEGHDOOT-AI PREFLIGHT CHECKS")
    print("="*70 + "\n")
    
    all_pass = True
    
    # ── 1. Config ──────────────────────────────────────────
    print("1. Configuration")
    try:
        from meghdoot.utils.config import load_config
        cfg = load_config('configs/default.yaml')
        check(True, "Config loads", success_msg="✓")
    except Exception as e:
        all_pass &= check(False, "Config loads", fail_msg=str(e))
        return 1
    
    # ── 2. Paths ───────────────────────────────────────────
    print("\n2. Data Paths")
    
    vae_ckpt = cfg['vae']['pretrained']
    all_pass &= check(
        os.path.exists(vae_ckpt),
        f"VAE checkpoint exists",
        fail_msg=f"Not found: {vae_ckpt}"
    )
    
    latent_dir = cfg['data']['paths']['latents']
    latent_files = list(Path(latent_dir).glob("*.pt"))
    all_pass &= check(
        len(latent_files) > 100,
        f"Latent tensors ({len(latent_files)} files)",
        fail_msg=f"Only {len(latent_files)} files" if len(latent_files) > 0 else f"No files in {latent_dir}"
    )
    
    proc_dir = cfg['data']['paths']['processed']
    proc_files = list(Path(proc_dir).glob("*.pt"))
    all_pass &= check(
        len(proc_files) > 100,
        f"Processed tensors ({len(proc_files)} files)",
        fail_msg=f"Only {len(proc_files)} files" if len(proc_files) > 0 else f"No files in {proc_dir}"
    )
    
    # ── 3. Imports ─────────────────────────────────────────
    print("\n3. Python Imports")
    
    try:
        from meghdoot.data.dataset import LatentSequenceDataset, INSATSequenceDataset
        check(True, "Dataset classes")
    except Exception as e:
        all_pass &= check(False, "Dataset classes", fail_msg=str(e))
    
    try:
        from meghdoot.models.vae import SatelliteVAE
        check(True, "VAE model")
    except Exception as e:
        all_pass &= check(False, "VAE model", fail_msg=str(e))
    
    try:
        from meghdoot.models.diffusion import MeghdootDiffusion
        check(True, "Diffusion model")
    except Exception as e:
        all_pass &= check(False, "Diffusion model", fail_msg=str(e))
    
    try:
        from meghdoot.evaluation.metrics import compute_all_metrics
        check(True, "Evaluation metrics")
    except Exception as e:
        all_pass &= check(False, "Evaluation metrics", fail_msg=str(e))
    
    # ── 4. Model Instantiation ──────────────────────────────
    print("\n4. Model Initialization")
    
    try:
        import torch
        from meghdoot.models.vae import SatelliteVAE
        vae = SatelliteVAE(cfg)
        n_params = sum(p.numel() for p in vae.vae.parameters())
        check(True, f"VAE instantiation ({n_params/1e6:.1f}M params)")
    except Exception as e:
        all_pass &= check(False, "VAE instantiation", fail_msg=str(e))
    
    try:
        from meghdoot.models.diffusion import MeghdootDiffusion
        diffusion = MeghdootDiffusion(cfg)
        n_params = sum(p.numel() for p in diffusion.unet.parameters())
        check(True, f"Diffusion instantiation ({n_params/1e6:.1f}M params)")
    except Exception as e:
        all_pass &= check(False, "Diffusion instantiation", fail_msg=str(e))
    
    # ── 5. Dataset ──────────────────────────────────────────
    print("\n5. Data Loading")
    
    try:
        from meghdoot.data.dataset import LatentSequenceDataset
        ds = LatentSequenceDataset(
            latent_dir=cfg['data']['paths']['latents'],
            num_history=cfg['diffusion']['conditioning']['num_history_frames']
        )
        check(
            len(ds) > 0,
            f"LatentSequenceDataset ({len(ds)} sequences)",
            fail_msg=f"0 sequences found"
        )
        
        # Try loading a sample
        sample = ds[0]
        history_shape = sample['history'].shape
        target_shape = sample['target'].shape
        check(
            history_shape[0] == 3 and target_shape[0] == 4,
            f"Sample shapes (history={history_shape}, target={target_shape})",
        )
    except Exception as e:
        all_pass &= check(False, "LatentSequenceDataset", fail_msg=str(e))
    
    # ── 6. Forward Pass ─────────────────────────────────────
    print("\n6. Forward Pass (Sanity Check)")
    
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from meghdoot.data.dataset import LatentSequenceDataset
        from meghdoot.models.diffusion import MeghdootDiffusion
        
        ds = LatentSequenceDataset(
            latent_dir=cfg['data']['paths']['latents'],
            num_history=3
        )
        diffusion = MeghdootDiffusion(cfg).to(device)
        diffusion.unet.eval()
        
        # Load one sample
        sample = ds[0]
        history = sample['history'].unsqueeze(0).to(device)  # [1, 3, 4, 64, 64]
        target = sample['target'].unsqueeze(0).to(device)    # [1, 4, 64, 64]
        
        # Run one training step
        with torch.no_grad():
            losses = diffusion.training_step(history, target)
        
        loss_val = losses['loss'].item()
        is_finite = torch.isfinite(torch.tensor(loss_val)).item()
        
        check(
            is_finite and loss_val > 0,
            f"Training step (loss={loss_val:.5f})",
            fail_msg=f"Loss is NaN or infinite"
        )
    except Exception as e:
        all_pass &= check(False, "Forward pass", fail_msg=str(e))
    
    # ── 7. GPU Check ────────────────────────────────────────
    print("\n7. Hardware")
    
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            check(True, f"GPU available ({gpu_name})")
        else:
            check(False, "GPU available", fail_msg="Using CPU (training will be slow!)")
    except Exception as e:
        check(False, "GPU check", fail_msg=str(e))
    
    # ── Final Result ────────────────────────────────────────
    print("\n" + "="*70)
    if all_pass:
        print("🟢 GO: All checks passed! Ready to train.")
        print("="*70 + "\n")
        return 0
    else:
        print("🔴 NO-GO: Fix issues above before training.")
        print("="*70 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
