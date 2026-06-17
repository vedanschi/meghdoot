# ☁️ Meghdoot-AI

**High-Resolution Weather Nowcasting over India using Latent Diffusion Models**

> *Leveraging INSAT-3DR/3DS satellite imagery and physics-aware latent diffusion for short-range (0–6 hr) precipitation and cloud-top forecasting over the Indian subcontinent.*

**Current Status:** Research-phase model focusing on latent diffusion with physics-aware losses, 3-stage VAE fine-tuning, and automated GCS pipeline deployment.

---

## 🏗 Architecture Overview

The pipeline consists of **five integrated phases**:

```
┌─────────────────────────────────────────────────────────────┐
│                     MEGHDOOT-AI PIPELINE                     │
└─────────────────────────────────────────────────────────────┘

INSAT-3DR/3DS HDF5/NetCDF
         │
         └─→ [PHASE 1: Data Acquisition & Preprocessing]
         │
         ├─→ Download from MOSDAC (6°N–38°N, 66°E–100°E)
         ├─→ Crop Indian subcontinent region
         ├─→ Reproject to WGS-84 + resize to 512×512
         └─→ Normalize to [-1, 1] range
                │
                └─→ [PHASE 2: VAE Fine-Tuning (3-stage)]
                │
                ├─ Stage 1: Domain Adaptation
                │  └─ Pretrained SD-VAE (decoder-only, frozen encoder)
                │     Loss: SSIM(0.5) + MAE(0.3) + VGG(0.2)
                │
                ├─ Stage 2: INSAT Fine-Tuning  
                │  └─ Unfreeze encoder, add Channel Integration Layer
                │     Loss: SSIM(0.4) + MAE(0.3) + VGG(0.2) + Temporal(0.1)
                │
                └─ Stage 3: Regional Specialization
                   └─ India-region crops + physics-aware loss
                      Loss: SSIM(0.4) + MAE(0.2) + VGG(0.2) + Temporal(0.1) + Mass(0.1)
                      │
                      └─→ Cache VAE latents: 4 channels × 64 × 64
                           │
                           └─→ [PHASE 3: Latent Diffusion Training]
                           │
                           ├─ Condition: 3 historical latent frames
                           ├─ Predict: denoised 4th frame latent
                           ├─ Architecture: UNet2D + DDPM (1000 steps)
                           ├─ Loss: MSE + Mass-Conservation + Grad Penalty + SSIM + MAE
                           └─ Optimization: AdamW + gradient accumulation + EMA
                                │
                                └─→ [PHASE 4: Evaluation]
                                │
                                ├─ Metrics: SSIM, RMSE, PSNR, CSI
                                ├─ Baselines: ConvLSTM, PySTEPS (optional)
                                └─ W&B logging
                                     │
                                     └─→ [PHASE 5: Deployment]
                                        │
                                        ├─ FastAPI inference server
                                        ├─ GCS artifact storage
                                        └─ Cloud Run / Docker containerization
```

**Data Flow:**
- Input: TIR1 (Thermal) + WV (Water Vapor) satellite channels
- Latent Space: 4 channels × 64 × 64 (8× spatial compression)
- Output: 512 × 512 predicted satellite frame (0–3 hour horizon)

## 📂 Project Structure

```
meghdoot/
├── mdapi.py                      # Official MOSDAC API script (unmodified)
├── config.json                   # Official MOSDAC credentials (edit here)
├── configs/
│   ├── default.yaml              # Master config: data, VAE, diffusion, deployment
│   └── mosdac_config.json        # MOSDAC config template
├── src/meghdoot/
│   ├── cli.py                    # Unified Click CLI entry point
│   ├── data/
│   │   ├── mdapi.py              # MOSDAC client wrapper + official downloader
│   │   ├── preprocessing.py      # HDF5/NetCDF → normalized .npy/.pt pipeline
│   │   ├── dataset.py            # PyTorch Dataset (image & latent sequences)
│   │   └── __init__.py
│   ├── models/
│   │   ├── vae.py                # VAE fine-tuning (3 loss types) + latent caching
│   │   ├── diffusion.py          # Latent Diffusion UNet + EMA + mass conservation loss
│   │   ├── channel_fusion.py     # Multi-spectral channel integration layer
│   │   ├── temporal_loss.py      # Optical-flow warped temporal consistency
│   │   └── __init__.py
│   ├── training/
│   │   ├── train_vae.py          # Simple VAE fine-tuning loop
│   │   ├── fine_tuning.py        # 3-stage transfer learning pipeline
│   │   ├── train_diffusion.py    # Latent diffusion training (Accelerate support)
│   │   ├── train_convlstm.py     # ConvLSTM baseline (optional)
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── metrics.py            # SSIM, RMSE, PSNR, CSI, MAE
│   │   ├── baselines.py          # ConvLSTM & PySTEPS baseline implementations
│   │   ├── benchmark.py          # Comparative evaluation framework
│   │   ├── visualize_predictions.py  # Matplotlib/Plotly visualizations
│   │   └── __init__.py
│   ├── deploy/
│   │   ├── api.py                # FastAPI inference server (async endpoints)
│   │   ├── pipeline.py           # End-to-end forecast pipeline
│   │   ├── runtime.py            # GCS integration, model loading
│   │   └── __init__.py
│   ├── utils/
│   │   ├── config.py             # YAML config loader + type casting
│   │   ├── logging.py            # Rich console + Weights & Biases integration
│   │   ├── helpers.py            # Seeds, device detection, path utilities
│   │   └── __init__.py
│   └── __init__.py
├── tests/
│   ├── conftest.py               # pytest fixtures
│   ├── test_api.py               # FastAPI endpoint tests
│   ├── test_dataset.py           # Dataset loading & caching tests
│   ├── test_models.py            # VAE & diffusion model tests
│   ├── test_preprocessing.py     # Satellite data preprocessing tests
│   ├── test_mdapi.py             # MOSDAC downloader tests
│   ├── test_deploy_runtime.py    # Deployment utilities tests
│   ├── test_metrics.py           # Metric calculation tests
│   └── test_utils.py             # Utility function tests
├── scripts/
│   ├── setup_gcp_stack.sh        # GCP infrastructure setup (Cloud Run, Cloud Scheduler)
│   ├── deploy_cloud_run_backend.sh   # Deploy backend to Cloud Run
│   ├── deploy_cloud_run_frontend.sh  # Deploy frontend to Cloud Run
│   ├── run_forecast_cycle.sh     # Orchestrate download → preprocess → predict
│   ├── sync_gcs.py               # Sync latents/artifacts with GCS
│   ├── eval_metrics.py           # Evaluate model on test set
│   ├── compare_models.py         # Compare multiple checkpoints
│   ├── diagnose_latent_stats.py  # Analyze VAE latent distributions
│   └── delete_gpu_instance.sh    # Clean up GCP resources
├── frontend/                     # Next.js dashboard (TypeScript)
│   ├── app/
│   │   ├── page.tsx              # Home page with forecast map
│   │   ├── layout.tsx            # Root layout
│   │   ├── api/                  # API route handlers
│   │   │   ├── forecast/latest/  # Fetch latest prediction
│   │   │   ├── weather/summary/  # Weather summary endpoint
│   │   │   └── metrics/          # Model metrics endpoint
│   │   └── globals.css
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.mjs
│   └── public/
├── docs/                         # API documentation & guides
├── pyproject.toml                # Project metadata + dependencies
├── requirements-pinned.txt       # Pinned versions (production)
├── Dockerfile                    # NVIDIA CUDA 12 + GDAL base image
├── docker-compose.yml            # Local dev compose (backend + frontend)
├── cloudbuild.yaml               # GCP Cloud Build configuration
├── diagnostic_latent_ranges.py   # Analyze/visualize VAE latent statistics
├── preflight.py                  # Pre-training diagnostics & validation
└── CLOUD_SCHEDULER_SETUP.md      # Cloud Scheduler configuration guide
```

## 🚀 Usage

All operations are available through the unified `meghdoot` CLI. The typical workflow follows **five phases**:

### Phase 1 – Data Acquisition & Preprocessing

Download INSAT-3DR/3DS satellite imagery from [MOSDAC](https://mosdac.gov.in/) and preprocess to normalized tensors.

**Step 1: Configure MOSDAC credentials**

Edit `config.json` at the project root:

```json
{
  "user_credentials": {
    "username/email": "your_mosdac_username",
    "password": "your_mosdac_password"
  },
  "search_parameters": {
    "datasetId": "3SIMG_L1C_SGP",
    "startTime": "2026-01-01",
    "endTime": "2026-03-31",
    "boundingBox": "66.0,6.0,100.0,38.0"
  }
}
```

> **Dataset IDs:** Browse at https://mosdac.gov.in/catalog/satellite.php  
> **Available:** 3SIMG_L1C_SGP (INSAT-3DS), 3RIMG_L1C_SGP (INSAT-3DR)

**Step 2: Download satellite data**

Choose one of two modes:

```bash
# Option A: Run official MOSDAC script (interactive mode)
meghdoot download --official

# Option B: Programmatic download (auto-retry, pagination, GCS sync)
meghdoot download --dataset-id 3SIMG_L1C_SGP

# Or run the official script standalone
python mdapi.py
```

**Step 3: Preprocess to normalized tensors**

```bash
meghdoot preprocess
```

Converts raw HDF5/NetCDF files to 512 × 512 normalized [-1, 1] tensors, cropped to the Indian subcontinent (6°N–38°N, 66°E–100°E).

---

### Phase 2 – VAE Fine-Tuning (3-Stage Pipeline)

The VAE encoder learns to compress satellite imagery into efficient latent representations (4 ch × 64 × 64).

**Option A: Quick VAE Cache (skip fine-tuning)**

```bash
meghdoot train-vae --cache-only
```

Uses the pre-trained Stable Diffusion VAE without fine-tuning. Fast, but less adapted to satellite data.

**Option B: 3-Stage Fine-Tuning Pipeline (recommended)**

```bash
# Run all 3 stages sequentially
meghdoot train-finetune --stage all

# Or run individual stages
meghdoot train-finetune --stage 1   # Domain Adaptation (generic satellite)
meghdoot train-finetune --stage 2   # INSAT Fine-Tuning (INSAT-specific)
meghdoot train-finetune --stage 3   # Regional Specialization (India crops)

# Resume from a specific stage
meghdoot train-finetune --resume-stage 2
```

**3-Stage Details:**

| Stage | Focus | Encoder | Loss Weights | Duration |
|-------|-------|---------|--------------|----------|
| **1: Domain Adaptation** | Adapt to satellite clouds | Frozen | SSIM(0.5) + MAE(0.3) + VGG(0.2) | ~10 epochs |
| **2: INSAT Fine-Tuning** | Learn INSAT-specific features | Partially unfrozen | + Temporal(0.1) | ~15 epochs |
| **3: Regional Specialist** | India region expertise | Unfrozen | + Mass Conservation(0.1) | ~10 epochs |

**Loss Functions:**

- **SSIM** (Structural Similarity): Preserves cloud boundary sharpness
- **MAE** (Mean Absolute Error): Pixel-level accuracy
- **VGG Perceptual**: High-level feature consistency  
- **Temporal Consistency** (Stages 2–3): Optical-flow smoothness between frames
- **Mass Conservation** (Stage 3): Prevents spurious cloud creation/destruction

After fine-tuning, latent vectors are cached to disk for fast diffusion training.

**Option C: Simple VAE Fine-Tuning (legacy)**

```bash
meghdoot train-vae --epochs 20 --batch-size 8
```

Single-stage VAE fine-tuning with hybrid loss. Latents are cached automatically.

---

### Phase 3 – Latent Diffusion Training

Train the conditional UNet to predict future weather from latent representations.

```bash
# Standard training
meghdoot train --epochs 100

# Resume from checkpoint
meghdoot train --resume checkpoints/diffusion/latest.pt

# Resume with modified epoch count
meghdoot train --resume checkpoints/diffusion/latest.pt --epochs 50
```

**Architecture:**

- **Input:** 3 historical latent frames (channel-concatenated, 16 channels total)
- **Output:** Predicted 4th frame latent (4 channels)
- **Backbone:** UNet2D + DDPM scheduler (1000 noise levels)
- **Optimization:** AdamW + gradient accumulation + EMA weight averaging

**Training Loss:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \alpha \cdot \mathcal{L}_{\text{mass}} + \beta \cdot \mathcal{L}_{\text{grad}} + \gamma \cdot \mathcal{L}_{\text{SSIM}} + \delta \cdot \mathcal{L}_{\text{MAE}}$$

where:
- $\mathcal{L}_{\text{MSE}}$: Noise prediction loss (core diffusion objective)
- $\mathcal{L}_{\text{mass}}$: Mass conservation penalty (physics)
- $\mathcal{L}_{\text{grad}}$: Gradient penalty (spatial smoothness)
- $\mathcal{L}_{\text{SSIM}} + \mathcal{L}_{\text{MAE}}$: Image quality terms

---

### Phase 4 – Evaluation

Run benchmarks against baselines (ConvLSTM, PySTEPS) and log to W&B.

```bash
# Evaluate on test set
meghdoot evaluate --diffusion-ckpt checkpoints/diffusion/latest.pt --n-samples 100

# Include ConvLSTM baseline
meghdoot evaluate --diffusion-ckpt checkpoints/diffusion/latest.pt \
                   --convlstm-ckpt checkpoints/baselines/convlstm.pt \
                   --n-samples 100
```

**Metrics:**

- **SSIM** (Structural Similarity): Cloud boundary precision [0–1]
- **RMSE** (Root Mean Squared Error): Pixel-level error [lower is better]
- **PSNR** (Peak Signal-to-Noise Ratio): Image fidelity in dB [higher is better]
- **CSI** (Critical Success Index): Convective event detection [0–1]
- **MAE** (Mean Absolute Error): Per-pixel error

Results are logged to [Weights & Biases](https://wandb.ai).

---

### Phase 5 – Deployment

Launch the inference API and optionally deploy to Cloud Run.

**Local API Server:**

```bash
# Start FastAPI server
meghdoot serve --host 0.0.0.0 --port 8000

# With custom workers (default: 2)
meghdoot serve --host 0.0.0.0 --port 8000
```

**API Endpoints:**

| Method | Path | Input | Output |
|--------|------|-------|--------|
| `POST` | `/predict` | 3 satellite frames (HxWx2) | `.npy` predictions |
| `GET` | `/health` | — | JSON status |
| `GET` | `/model/info` | — | Model metadata |

**Docker Deployment:**

```bash
# Build Docker image (NVIDIA CUDA 12 + GDAL)
docker build -t meghdoot-ai:latest .

# Run locally
docker run --gpus all -p 8000:8000 meghdoot-ai:latest

# Deploy to Cloud Run
./scripts/deploy_cloud_run_backend.sh
```

**Frontend Dashboard (optional):**

```bash
# Deploy Next.js frontend
cd frontend
npm install
npm run build
./scripts/deploy_cloud_run_frontend.sh
```

---

## 🧪 Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src/meghdoot --cov-report=html

# Run specific test module
pytest tests/test_models.py -v

# Run tests matching a pattern
pytest tests/ -k "vae or diffusion" -v
```

**Test Coverage:**

- `test_models.py` — VAE encoding/decoding, diffusion noise scheduling
- `test_dataset.py` — Image & latent sequence loading, normalization
- `test_preprocessing.py` — HDF5/NetCDF parsing, geospatial cropping, resizing
- `test_mdapi.py` — MOSDAC download authentication, pagination
- `test_metrics.py` — SSIM, RMSE, PSNR, CSI calculations
- `test_api.py` — FastAPI endpoint validation, latency
- `test_deploy_runtime.py` — GCS checkpoint loading, model initialization
- `test_utils.py` — Config parsing, seed management, device detection

---

## 🔧 Configuration

All hyperparameters are controlled via `configs/default.yaml`. **Key sections:**

| Section | Controls |
|---------|----------|
| `project` | Random seed, device (cuda/cpu/mps) |
| `data` | MOSDAC credentials, region bounds, channels, normalization |
| `vae` | Pre-trained model ID, fine-tuning parameters, loss weights |
| `fine_tuning` | 3-stage pipeline: learning rates, epochs, loss weights per stage |
| `channel_fusion` | Multi-spectral integration layer dimensions |
| `temporal_loss` | Optical-flow warping, smoothness penalties |
| `diffusion` | Scheduler, UNet architecture, noise steps, conditioning |
| `training` | Batch size, accumulation, AMP precision, checkpoint frequency |
| `evaluation` | Metrics, baselines, CSI thresholds |
| `logging` | W&B project, experiment tags, artifact logging |
| `deployment` | API host/port, GCS bucket, Cloud Run settings |

**Quick Config Customization:**

```yaml
# Example: Reduce training epochs for experimentation
diffusion:
  training:
    epochs: 20  # instead of 100

# Example: Use CPU instead of GPU
project:
  device: cpu

# Example: Change batch size
vae:
  fine_tune:
    batch_size: 8  # instead of 4

# Example: Adjust physics loss weight
fine_tuning:
  stages:
    regional:
      physics_weight: 0.2  # increase from 0.1
```

---

## 📡 Supported Data

| Satellite | Agency | Format | Channels | Coverage | Resolution |
|-----------|--------|--------|----------|----------|------------|
| **INSAT-3DR** | ISRO | HDF5 (`.h5`) | TIR1, TIR2, MIR, WV, VIS, SWIR | 6°N–38°N, 66°E–100°E | 1 km / 4 km |
| **INSAT-3DS** | ISRO | HDF5 / NetCDF (`.nc4`) | TIR1, TIR2, MIR, WV, VIS, SWIR | 6°N–38°N, 66°E–100°E | 0.5 km / 2 km |

**Default channels used:** TIR1 (Thermal) + WV (Water Vapor)

**Download from:** [MOSDAC](https://mosdac.gov.in/) – Indian meteorological satellite data portal

---

## 🌐 Deployment

### Local Development

```bash
# Use docker-compose for full stack (backend + frontend)
docker compose up --build -d

# Access:
#   Backend API: http://localhost:8000
#   Frontend:    http://localhost:3000
#   API Docs:    http://localhost:8000/docs
```

### Cloud Deployment (GCP)

Meghdoot integrates with Google Cloud for automated data syncing, training, and inference.

**Setup:**

```bash
# 1. Create GCP project and enable APIs
gcloud projects create meghdoot-project
gcloud config set project meghdoot-project

gcloud services enable \
  compute.googleapis.com \
  run.googleapis.com \
  storage-api.googleapis.com \
  cloudscheduler.googleapis.com \
  cloudbuild.googleapis.com

# 2. Create GCS buckets
gsutil mb gs://meghdoot-satellite-data
gsutil mb gs://meghdoot-forecasts

# 3. Deploy infrastructure
./scripts/setup_gcp_stack.sh
```

**Automated Forecast Cycle:**

Cloud Scheduler triggers periodic (e.g., hourly) forecast cycles:

```
INSAT-3DR/3DS → Download → Preprocess → Diffusion → GCS Upload → Dashboard
    (MOSDAC)        ↓          ↓            ↓           ↓
              Cloud Function  Cloud Run   Cloud Run  Frontend
```

**Deploy Backend to Cloud Run:**

```bash
./scripts/deploy_cloud_run_backend.sh
```

**Deploy Frontend to Cloud Run:**

```bash
./scripts/deploy_cloud_run_frontend.sh
```

---

## 📊 Experiment Tracking

Training logs go to [Weights & Biases](https://wandb.ai):

```bash
# Login to W&B
wandb login

# Start training (logs automatically)
meghdoot train --epochs 100

# View dashboard
# https://wandb.ai/<username>/<project>/
```

**Logged Metrics:**

- Training loss components (MSE, mass, gradient, SSIM, MAE)
- Validation loss & metrics (SSIM, RMSE, PSNR, CSI)
- Learning rate schedules
- Prediction samples (every N epochs)
- Model checkpoints

---

## ⚙️ Advanced Features

### Physics-Aware Losses

**Mass Conservation Loss:**  
Penalizes spurious creation/destruction of "cloud mass" between frames. Computed as:

$$\mathcal{L}_{\text{mass}} = \frac{1}{\text{norm}} \sum_c \left| \sum_{h,w} \text{pred}_{c,h,w} - \sum_{h,w} \text{cond}_{c,h,w} \right|$$

where norm prevents raw latent magnitudes from dominating.

**Temporal Consistency Loss:**  
Enforces smooth motion between frames using optical flow (Farneback algorithm). Penalizes:
- Warped-frame L1 error
- Flow spatial smoothness
- Unrealistic flow magnitudes

### Multi-Spectral Channel Fusion

Combines TIR1 (thermal) and WV (water vapor) channels via learned integration layer before encoding:

```
TIR1 [512×512] ─┐
                ├─→ ChannelIntegrationLayer ─→ Fusion [512×512]
WV   [512×512] ─┘
                    (3 fused channels for VAE encoder)
```

### Distributed Training

Use `accelerate` for multi-GPU/multi-node training:

```bash
accelerate config
accelerate launch -m meghdoot.training.train_diffusion --epochs 100
```

---

## 🛠️ Troubleshooting

### MOSDAC Download Issues

**Problem:** Authentication fails  
**Solution:** Verify credentials in `config.json` and ensure MOSDAC account is active.

```bash
meghdoot download --official  # test with official script first
```

### Out-of-Memory (OOM) Errors

**Problem:** CUDA OOM during training  
**Solutions:**
- Reduce batch size in `configs/default.yaml`
- Enable gradient accumulation: `training.gradient_accumulation_steps: 4`
- Use mixed precision: `training.mixed_precision: "fp16"`

### Slow Data Loading

**Problem:** Training is I/O-bound  
**Solutions:**
- Increase `data.num_workers` (e.g., 4–8)
- Enable `cache_in_memory: true` for diffusion (requires 3–5 GB RAM)
- Use NVMe SSD for latent caching

### Missing VAE Latents

**Problem:** Diffusion training fails – can't find cached latents  
**Solutions:**
```bash
# Ensure VAE fine-tuning completed
meghdoot train-vae --epochs 20

# Verify latent files exist
ls -la /home/jupyter/local_data/stacked_tensors/TIR1/
```

### GPU Device Issues

**Problem:** "CUDA is not available"  
**Solutions:**
```bash
# Verify NVIDIA drivers
nvidia-smi

# Install pytorch-cuda
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Force CPU (for debugging)
python -c "from meghdoot.utils.helpers import get_device; print(get_device('cpu'))"
```

---

## 📄 License

MIT — See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **[ISRO](https://www.isro.gov.in/) / [MOSDAC](https://mosdac.gov.in/)** — INSAT-3DR/3DS satellite imagery and geospatial data infrastructure
- **[Stability AI](https://stability.ai/)** — Pre-trained VAE encoder (`stabilityai/sd-vae-ft-mse`) from Stable Diffusion
- **[HuggingFace Diffusers](https://github.com/huggingface/diffusers)** — DDPM scheduler, UNet2D backbone, and diffusion utilities
- **[PyTorch](https://pytorch.org/)** — Deep learning framework
- **[Weights & Biases](https://wandb.ai/)** — Experiment tracking and model management
- **[PySTEPS](https://pysteps.github.io/)** — Optical-flow nowcasting baseline
- **[OpenCV](https://opencv.org/)** — Image processing and geometric transformations
- **[Rasterio](https://rasterio.readthedocs.io/)** — Geospatial data I/O
- **[xarray](http://xarray.pydata.org/)** — Labeled multi-dimensional array handling

---

## 📖 Citation

If you use Meghdoot-AI in your research, please cite:

```bibtex
@software{meghdoot2026,
  title={Meghdoot-AI: Latent Diffusion for Weather Nowcasting},
  author={Vedanschi},
  year={2026},
  url={https://github.com/vedanschi/meghdoot}
}
```

---

## 📞 Support & Contributions

**Issues & Bug Reports:** [GitHub Issues](https://github.com/vedanschi/meghdoot/issues)  
**Discussions:** [GitHub Discussions](https://github.com/vedanschi/meghdoot/discussions)  
**Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
