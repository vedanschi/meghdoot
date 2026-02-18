# ☁️ Meghdoot-AI

**High-Resolution Weather Nowcasting over India using Latent Diffusion Models**

> *Leveraging INSAT-3DR/3DS satellite imagery and physics-aware latent diffusion for short-range (0–6 hr) precipitation and cloud-top forecasting.*

---

## 🏗 Architecture Overview

```
INSAT-3DR/3DS HDF5/NetCDF
         │
    ┌────▼─────┐
    │ Preprocess│  Crop ➜ Resize ➜ Reproject ➜ Normalize
    └────┬─────┘
         │  (512 × 512 pixel-space)
    ┌────▼─────┐
    │   VAE    │  StabilityAI sd-vae-ft-mse (fine-tuned decoder)
    │ Encoder  │  Hybrid loss: SSIM + MAE + VGG Perceptual
    └────┬─────┘
         │  (4 × 64 × 64 latent-space)
    ┌────▼─────┐
    │ Diffusion│  UNet2D + DDPM (1000 steps)
    │  (cond.) │  Physics loss: Mass Conservation + Gradient Penalty
    └────┬─────┘
         │  (predicted latent)
    ┌────▼─────┐
    │   VAE    │
    │ Decoder  │
    └────┬─────┘
         │  (512 × 512 predicted frame)
    ┌────▼─────┐
    │ Evaluate │  SSIM · RMSE · PSNR · CSI
    └──────────┘
```

## 📂 Project Structure

```
meghdoot/
├── mdapi.py                   # Official MOSDAC download script (unmodified)
├── config.json                # Official MOSDAC config (edit credentials here)
├── configs/
│   ├── default.yaml           # Master configuration
│   └── mosdac_config.json     # Alternate MOSDAC config template
├── src/meghdoot/
│   ├── data/
│   │   ├── mdapi.py           # MOSDAC integration wrapper
│   │   ├── preprocessing.py   # HDF5/NetCDF → .npy pipeline
│   │   └── dataset.py         # PyTorch datasets
│   ├── models/
│   │   ├── vae.py             # VAE fine-tuning + latent caching
│   │   └── diffusion.py       # Conditional latent diffusion
│   ├── training/
│   │   ├── train_vae.py       # VAE fine-tuning loop
│   │   └── train_diffusion.py # Diffusion training loop
│   ├── evaluation/
│   │   ├── metrics.py         # SSIM, RMSE, PSNR, CSI
│   │   ├── baselines.py       # ConvLSTM, PySTEPS
│   │   └── benchmark.py       # Comparative evaluation
│   ├── deploy/
│   │   └── api.py             # FastAPI inference server
│   ├── utils/
│   │   ├── config.py          # YAML config loader
│   │   ├── logging.py         # Rich + W&B logging
│   │   └── helpers.py         # Seeds, device, paths
│   └── cli.py                 # Click CLI entry point
├── tests/                     # pytest test suite
├── Dockerfile                 # NVIDIA CUDA + GDAL container
├── docker-compose.yml
└── pyproject.toml
```

## ⚙️ Installation

### Prerequisites

| Requirement | Minimum |
|---|---|
| Python | 3.10+ |
| CUDA | 12.x (for GPU training) |
| GDAL | 3.6+ (system-level) |
| RAM | 32 GB recommended |
| VRAM | 16 GB (A100/V100 recommended) |

### System Dependencies (Ubuntu/Debian)

```bash
sudo apt-get update && sudo apt-get install -y \
    libgdal-dev gdal-bin libhdf5-dev libnetcdf-dev \
    libgl1-mesa-glx libglib2.0-0
```

### Python Setup

```bash
# Clone the repository
git clone https://github.com/vedanschi/meghdoot.git
cd meghdoot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"

# (Optional) Install baselines
pip install -e ".[baselines]"
```

### Docker (Recommended for Deployment)

```bash
docker compose up --build -d
```

---

## 🚀 Usage

All commands are available through the unified `meghdoot` CLI:

### Phase 1 — Data Acquisition

Data comes from [MOSDAC](https://mosdac.gov.in/) via the official Download API.

**Step 1: Configure credentials** — edit `config.json` at the project root:
```json
{
  "user_credentials": {
    "username/email": "your_mosdac_username",
    "password": "your_mosdac_password"
  },
  "search_parameters": {
    "datasetId": "3RIMG_L1B_STD",
    "startTime": "2024-06-01",
    "endTime": "2024-08-31",
    "boundingBox": "66.0,6.0,100.0,38.0"
  }
}
```
> Browse dataset IDs at: https://mosdac.gov.in/catalog/satellite.php

**Step 2: Download** — choose either mode:
```bash
# Option A: Run official MOSDAC script directly (interactive Y/N prompt)
meghdoot download --official

# Option B: Programmatic download with auto-retry & pagination
meghdoot download --dataset-id 3RIMG_L1B_STD

# Or run the official script standalone (no meghdoot CLI needed)
python mdapi.py
```

**Step 3: Preprocess** — convert raw HDF5/NetCDF to normalized .npy:
```bash
meghdoot preprocess
```

### Phase 2 — VAE Fine-Tuning

```bash
# Fine-tune the VAE decoder with hybrid loss
meghdoot train-vae --epochs 20 --batch-size 8
```

Loss function: $\mathcal{L}_{\text{VAE}} = \lambda_1 \cdot \mathcal{L}_{\text{SSIM}} + \lambda_2 \cdot \mathcal{L}_{\text{MAE}} + \lambda_3 \cdot \mathcal{L}_{\text{VGG}}$

where $\lambda_1 = 0.5$, $\lambda_2 = 0.3$, $\lambda_3 = 0.2$.

### Phase 3 — Diffusion Training

```bash
# Train the latent diffusion model
meghdoot train --epochs 100 --batch-size 4
```

Training loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \alpha \cdot \mathcal{L}_{\text{physics}} + \beta \cdot \mathcal{L}_{\text{grad}} + \gamma \cdot \mathcal{L}_{\text{SSIM}} + \delta \cdot \mathcal{L}_{\text{MAE}}$$

### Phase 4 — Evaluation

```bash
# Run comparative evaluation against baselines
meghdoot evaluate --checkpoint checkpoints/latest.pt
```

### Phase 5 — Deployment

```bash
# Start the inference API
meghdoot serve --host 0.0.0.0 --port 8000
```

**API Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Returns `.npy` prediction |
| `POST` | `/predict/json` | Returns JSON with stats |
| `GET` | `/health` | Health check |
| `GET` | `/model/info` | Model metadata |

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 🔧 Configuration

All parameters are controlled via `configs/default.yaml`. Key sections:

| Section | Controls |
|---|---|
| `data` | MOSDAC credentials, region bounds, resolution, satellites |
| `vae` | Pre-trained model ID, loss weights, learning rate |
| `diffusion` | UNet architecture, scheduler, physics loss weights |
| `training` | Epochs, batch size, AMP, gradient accumulation |
| `evaluation` | Metrics, CSI thresholds, baseline configs |
| `deployment` | API host/port, checkpoint path |

---

## 📡 Supported Data

| Satellite | Format | Channels |
|---|---|---|
| **INSAT-3DR** | HDF5 (`.h5`) | TIR1, TIR2, MIR, WV, VIS, SWIR |
| **INSAT-3DS** | HDF5 / NetCDF (`.nc`, `.nc4`) | TIR1, TIR2, MIR, WV, VIS, SWIR |

Coverage: Indian subcontinent (6°N–38°N, 66°E–100°E)

---

## 📊 Experiment Tracking

Training metrics are logged to [Weights & Biases](https://wandb.ai):

```bash
wandb login
meghdoot train --epochs 100
```

---

## 📄 License

MIT

---

## 🙏 Acknowledgements

- **ISRO / MOSDAC** — INSAT-3DR/3DS satellite imagery
- **Stability AI** — Pre-trained VAE (`stabilityai/sd-vae-ft-mse`)
- **HuggingFace Diffusers** — Diffusion model primitives
- **PySTEPS** — Optical-flow nowcasting baseline
