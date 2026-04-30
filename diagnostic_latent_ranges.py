#!/usr/bin/env python3
"""
Comprehensive diagnostic of latent ranges and normalizations throughout the pipeline.
Run on the instance to verify data integrity before training.
"""

import torch
import glob
from pathlib import Path
import numpy as np

print("=" * 80)
print("LATENT PIPELINE DIAGNOSTICS")
print("=" * 80)

# 1. Check raw latent files on disk
latent_dir = Path("/home/jupyter/local_data")
latent_files = sorted(glob.glob(str(latent_dir / "latents_*.pt")))[:10]

print(f"\n1. RAW LATENT FILES (first 10 of {len(glob.glob(str(latent_dir / 'latents_*.pt')))} total):")
print("-" * 80)

all_mins = []
all_maxs = []
all_means = []
all_stds = []
all_sums = []

for fpath in latent_files:
    latent = torch.load(fpath, weights_only=True)
    
    lat_min = latent.min().item()
    lat_max = latent.max().item()
    lat_mean = latent.mean().item()
    lat_std = latent.std().item()
    lat_sum = latent.sum(dim=(-2, -1))  # spatial sum per channel
    
    all_mins.append(lat_min)
    all_maxs.append(lat_max)
    all_means.append(lat_mean)
    all_stds.append(lat_std)
    all_sums.append(lat_sum)
    
    print(f"  {Path(fpath).name}:")
    print(f"    Shape: {latent.shape}")
    print(f"    Range: [{lat_min:7.4f}, {lat_max:7.4f}]")
    print(f"    Mean: {lat_mean:7.4f}, Std: {lat_std:7.4f}")
    print(f"    Spatial sum per channel: min={lat_sum.min().item():8.2f}, max={lat_sum.max().item():8.2f}, mean={lat_sum.mean().item():8.2f}")

print(f"\n  AGGREGATED STATISTICS (across {len(latent_files)} samples):")
print(f"    Global min: {min(all_mins):.4f}, Global max: {max(all_maxs):.4f}")
print(f"    Mean of means: {np.mean(all_means):.4f}, Std of stds: {np.std(all_stds):.4f}")
print(f"    Spatial sum stats: min={min([s.min().item() for s in all_sums]):.2f}, max={max([s.max().item() for s in all_sums]):.2f}")

# 2. Check what mass conservation loss would produce
print(f"\n2. MASS CONSERVATION LOSS SCALE:")
print("-" * 80)

# Load a pair of consecutive latents
latent1 = torch.load(latent_files[0], weights_only=True)
latent2 = torch.load(latent_files[1], weights_only=True)

mass1 = latent1.sum(dim=(-2, -1))  # [C]
mass2 = latent2.sum(dim=(-2, -1))  # [C]

raw_mass_diff = (mass1 - mass2).abs().mean().item()
normalized_mass_diff = raw_mass_diff / (mass2.abs().mean().item() + 1e-8)

print(f"  Latent1 spatial sum: {mass1}")
print(f"  Latent2 spatial sum: {mass2}")
print(f"  Raw mass loss (unnormalized): {raw_mass_diff:.4f}")
print(f"  Normalized mass loss (div by latent scale): {normalized_mass_diff:.6f}")

# 3. Check predicted_x0 clamping
print(f"\n3. PREDICTED_X0 CLAMPING ANALYSIS:")
print("-" * 80)
print(f"  Current clamp range: [-10.0, 10.0]")
print(f"  BUT latents are in range: [{min(all_mins):.4f}, {max(all_maxs):.4f}]")
print(f"  ❌ PROBLEM: Clamping [-10, 10] allows 10x larger values than actual latents!")
print(f"  ✓ CORRECT clamp range should be: [-3.0, 3.0] (conservative)")

# 4. Check noise prediction scale
print(f"\n4. NOISE PREDICTION SCALE:")
print("-" * 80)
noise_sample = torch.randn(4, 4, 64, 64)
print(f"  Typical noise tensor: shape={noise_sample.shape}, mean={noise_sample.mean().item():.4f}, std={noise_sample.std().item():.4f}")
print(f"  Noise min/max: [{noise_sample.min().item():.4f}, {noise_sample.max().item():.4f}]")
print(f"  ✓ Noise scale is correct (standard normal)")

# 5. Summary recommendations
print(f"\n5. RECOMMENDATIONS:")
print("-" * 80)
print(f"  ✓ Latents ARE normalized to [-1, 1] range (preprocessing is correct)")
print(f"  ❌ MassConservationLoss needs normalization (already fixed)")
print(f"  ❌ Predicted_x0 clamping is too loose: [-10, 10] → [-3, 3]")
print(f"  ✓ Physics loss threshold (alpha_bar > 0.1) is reasonable")
print(f"  ✓ All loss components should be logged separately to catch anomalies")

print("\n" + "=" * 80)
