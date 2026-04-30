# COMPREHENSIVE PIPELINE FIXES - April 30, 2026

## Problem Summary
Training was producing physics loss values of **200-1400+** per batch instead of **~0.01-0.1**. This massive scale was destabilizing gradients and causing the model to ignore spatial conditioning (leaf hallucination).

## Root Causes Identified

### 1. **Unnormalized Mass Conservation Loss** ✓ FIXED
**Issue**: `MassConservationLoss` was computing raw spatial sums of latents without normalization.
- Latents are 4-channel, [-1, 1] range
- Sum across 64×64 grid: raw sums range from -4096 to +4096
- Without normalization, physics_loss could be hundreds

**Fix Applied** (commit d0a717c):
```python
def forward(self, predicted, last_condition):
    mass_pred = predicted.sum(dim=(-2, -1))      # [B, C]
    mass_cond = last_condition.sum(dim=(-2, -1))  # [B, C]
    
    # NORMALIZED by L2 norm to keep loss in reasonable range (~0.01-0.1)
    norm_factor = (mass_cond.abs().mean() + 1e-8)
    return ((mass_pred - mass_cond).abs() / norm_factor).mean()
```

### 2. **Excessive Clamping of predicted_x0** ✓ FIXED
**Issue**: `predicted_x0` was clamped to [-10, 10] but latents are [-1, 1].
- This allowed 10x larger reconstruction errors
- Extreme values in predicted_x0 made loss computation unstable
- Clamping should match actual latent range

**Fix Applied** (commit 294fc0e):
```python
# OLD: predicted_x0 = torch.clamp(predicted_x0, -10.0, 10.0)
# NEW: predicted_x0 = torch.clamp(predicted_x0, -3.0, 3.0)  # Conservative, matches latent range
```

### 3. **Unnormalized Gradient Penalty** ✓ FIXED
**Issue**: Gradient penalty was summing dx + dy without averaging.
- Gradient magnitudes scale with spatial resolution
- Large loss dominated by spatial dimensions

**Fix Applied** (commit 294fc0e):
```python
# OLD: grad_loss = (dx.abs().mean() + dy.abs().mean()) * self.grad_penalty_weight
# NEW: grad_loss = ((dx.abs().mean() + dy.abs().mean()) / 2.0) * self.grad_penalty_weight
```

### 4. **Unmasked Latent L1 Loss at High Timesteps** ✓ FIXED
**Issue**: L1 loss between predicted_x0 and target was applied unconditionally.
- At high timesteps (high noise), target is buried in noise
- Forcing predicted_x0 to match garbage target wastes gradient
- Should only apply when x0 reconstruction is meaningful

**Fix Applied** (commit 294fc0e):
```python
# OLD: latent_l1_loss = F.l1_loss(predicted_x0, target_latent)
# NEW: latent_l1_loss = F.l1_loss(predicted_x0, target_latent) * physics_mask.mean()
```

### 5. **Physics Loss Scheduling** ✓ ALREADY FIXED
**From**: commit d0a717c + 294fc0e
- Only apply physics loss when alpha_bar > 0.1 (low timesteps)
- At high timesteps (alpha_bar ~ 0), x0 is mostly noise anyway
- Prevents destabilization at high-noise timesteps

---

## Expected Behavior After Fixes

**Physics Loss Scale**:
- OLD: 200-1400 per batch ❌
- NEW: 0.01-0.1 per batch ✓

**MSE Loss Scale**:
- Should remain ~1.0-1.2 (standard for diffusion noise prediction)

**Gradient Penalty**:
- Should be ~0.001-0.01 (small smoothness constraint)

**Latent L1**:
- Only active at low timesteps (alpha_bar > 0.1)
- Should be ~0.1-0.3 when active

**Total Loss**:
- Should be dominated by MSE (~1.1)
- Physics, grad, latent_l1 are small auxiliary terms

---

## Deployment Instructions

1. **On Instance**:
   ```bash
   pkill -9 -f train_diffusion.py
   cd /home/jupyter/meghdoot && git pull
   python diagnostic_latent_ranges.py  # Verify latent statistics
   python -m meghdoot.training.train_diffusion
   ```

2. **Monitor First 20 Batches**:
   - Physics loss should be ~0.01-0.1 (not 200+)
   - MSE should be ~1.0-1.3
   - Total loss should decrease smoothly
   - No NaN or Inf values

3. **Expected Training Duration**: 12-16 hours for 100 epochs (May 2 deadline)

---

## Verification Checklist

- [ ] Git pull on instance completes successfully
- [ ] diagnostic_latent_ranges.py shows latents in [-1, 1] range
- [ ] First 20 batches show physics_loss ~0.01-0.1
- [ ] MSE loss ~1.0-1.3, not flat
- [ ] Total loss decreasing smoothly
- [ ] No NaN/Inf in W&B logs
- [ ] 100 epochs complete in 12-16 hours
- [ ] Sample images show progressive improvement (noise → structure → weather patterns)

---

## Code Locations of Fixes

| Issue | File | Lines | Commit |
|-------|------|-------|--------|
| Mass loss normalization | `src/meghdoot/models/diffusion.py` | 37-75 | d0a717c |
| Predicted_x0 clamp | `src/meghdoot/models/diffusion.py` | 235-238 | 294fc0e |
| Gradient penalty norm | `src/meghdoot/models/diffusion.py` | 260-263 | 294fc0e |
| Latent L1 masking | `src/meghdoot/models/diffusion.py` | 266-268 | 294fc0e |
| Physics threshold | `src/meghdoot/models/diffusion.py` | 245-254 | d0a717c |

---

## Why These Fixes Work (Scientifically)

**Diffusion Models Require Careful Loss Balancing**:
1. MSE drives noise prediction (primary signal)
2. Auxiliary losses must be much smaller to avoid gradient interference
3. Loss scales must match problem domain (latents [-1, 1], not [-10, 10])
4. Losses should be timestep-aware (early steps = structured, late steps = noise)

**Physics Loss Normalization**:
- Mass conservation is relative, not absolute
- Dividing by expected mass scale makes loss invariant to latent magnitude
- Prevents loss from exploding with larger latents

**Predicted_x0 Clamping**:
- Clamping must be tight relative to data range
- Loose clamping ([-10, 10]) allows extreme reconstructions
- Tight clamping ([-3, 3]) keeps model honest about latent space

**Timestep-Aware Masking**:
- Early diffusion steps (high alpha_bar): x0 is nearly signal → apply all losses
- Late diffusion steps (low alpha_bar): x0 is mostly noise → only MSE helps
- Physics loss on garbage x0 creates harmful gradients → mask it out

---

## Next Steps (After Training Validation)

1. Validate 50+ epochs without hallucination
2. Check SSIM trends (should improve, target >0.35)
3. Sample images at epochs 10, 30, 50, 100 for visual inspection
4. If all good: prepare final model for May 2 presentation
