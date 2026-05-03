# Meghdoot Nowcasting Pipeline – Setup & Deployment Guide

This guide explains how to set up and run the automated nowcasting pipeline that generates 3-hour forecasts every 30 minutes.

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│ Cron / Systemd Timer (every 30 minutes)                    │
└────────────────────┬─────────────────────────────────────┘
                     │
                     v
     ┌───────────────────────────────┐
     │ run_forecast_cycle.sh         │
     │ (wrapper script)              │
     └───────────┬─────────────────┘
                 │
     ┌───────────v──────────────────────┐
     │ meghdoot.deploy.pipeline         │
     │ (Python main entrypoint)         │
     └───────────┬──────────────────────┘
                 │
        ┌────────┴────────┬─────────┬──────────┐
        v                 v         v          v
   ┌────────┐      ┌──────────┐ ┌─────────┐ ┌──────────┐
   │MOSDAC  │      │Preprocess│ │Diffusion│ │GCS Bucket│
   │Download│      │   →VAE   │ │ Infer   │ │ Upload   │
   │(3 frames)    │ Latents  │ │(6 steps)│ │(metadata)│
   └────────┘      └──────────┘ └─────────┘ └──────────┘
```

## Prerequisites

### 1. System Requirements
- **Python**: 3.10+
- **GPU**: CUDA 12.1+ (for inference acceleration)
- **Storage**: ~50GB for raw satellite data cache
- **GCS**: Credentials configured (`~/.config/gcloud/` or `GOOGLE_APPLICATION_CREDENTIALS`)

### 2. Install Dependencies

Install google-cloud-storage for GCS bucket access:
```bash
cd /home/vedanschi/meghdoot
source .venv/bin/activate
pip install google-cloud-storage
```

Or update requirements:
```bash
pip install -r requirements-pinned.txt google-cloud-storage
```

### 3. Configure MOSDAC Credentials

The pipeline expects MOSDAC credentials in `config.json`:
```bash
# Create the config file (you'll be prompted for username/password)
python -m meghdoot.data.mdapi --login
```

This creates `config.json` with your encrypted MOSDAC credentials.

## Configuration

### 1. Update `configs/default.yaml`

Verify the following sections:

**Data paths** (must exist and be writable):
```yaml
data:
  paths:
    raw: "/home/jupyter/meghdoot_data/raw"
    processed: "/home/jupyter/meghdoot_data/processed"
```

**GCS bucket** (for forecast output):
```yaml
deployment:
  gcs_bucket: "megdhoot-satellite-data"  # Your bucket name
```

**Model checkpoint** (should exist):
```yaml
diffusion:
  checkpoint_dir: "checkpoints/diffusion"
  # Will auto-load the latest epoch*.pt file
```

### 2. Create Required Directories

```bash
mkdir -p /home/jupyter/meghdoot_data/{raw,processed,metadata}
mkdir -p /home/vedanschi/meghdoot/logs
```

### 3. Verify GCS Credentials

```bash
# Test GCS access
python -c "from google.cloud import storage; \
    client = storage.Client(); \
    bucket = client.bucket('megdhoot-satellite-data'); \
    print(f'Bucket exists: {bucket.exists()}')"
```

## Running the Pipeline

### Manual Test Run

```bash
cd /home/vedanschi/meghdoot
source .venv/bin/activate
python -m meghdoot.deploy.pipeline --config configs/default.yaml
```

Expected output:
```
================================================================================
Meghdoot Nowcasting Pipeline Started
================================================================================
Downloading latest 3 INSAT frames from MOSDAC...
  (fetching...) 
Preprocessing 3 files...
  (converting to tensors...)
Loading VAE and Diffusion models...
  Loaded checkpoint: diffusion_epoch290.pt
Generating 6-step forecast...
  Step 1/6...
  Step 2/6...
  ...
Publishing forecast to GCS bucket...
  Uploaded metadata to gs://megdhoot-satellite-data/forecasts/latest/metadata.json
  Uploaded forecast_step_0.png
  ...
================================================================================
Pipeline completed successfully in 45.2s
================================================================================
```

### Options

```bash
# Generate different number of forecast steps
python -m meghdoot.deploy.pipeline --config configs/default.yaml --num-steps 8

# Use alternative config
python -m meghdoot.deploy.pipeline --config configs/custom.yaml
```

## Scheduling

### Option 1: Cron Job

Edit your crontab:
```bash
crontab -e
```

Add this line to run every 30 minutes:
```cron
0,30 * * * * /home/vedanschi/scripts/run_forecast_cycle.sh >> /home/vedanschi/meghdoot/logs/cron.log 2>&1
```

Or every 30 minutes starting at 00:00:
```cron
*/30 * * * * /home/vedanschi/scripts/run_forecast_cycle.sh
```

### Option 2: Systemd Timer (Recommended)

Create a systemd service file:
```bash
sudo tee /etc/systemd/system/meghdoot-pipeline.service > /dev/null << 'EOF'
[Unit]
Description=Meghdoot Nowcasting Pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=vedanschi
WorkingDirectory=/home/vedanschi/meghdoot
Environment="PATH=/home/vedanschi/meghdoot/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
ExecStart=/home/vedanschi/meghdoot/.venv/bin/python -m meghdoot.deploy.pipeline --config configs/default.yaml
StandardOutput=journal
StandardError=journal
EOF
```

Create the timer:
```bash
sudo tee /etc/systemd/system/meghdoot-pipeline.timer > /dev/null << 'EOF'
[Unit]
Description=Meghdoot Pipeline Timer (every 30 minutes)
Requires=megdhoot-pipeline.service

[Timer]
OnBootSec=1min
OnUnitActiveSec=30min
Persistent=true
Unit=megdhoot-pipeline.service

[Install]
WantedBy=timers.target
EOF
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable megdhoot-pipeline.timer
sudo systemctl start megdhoot-pipeline.timer
```

Check status:
```bash
sudo systemctl list-timers
sudo systemctl status megdhoot-pipeline.service
```

View logs:
```bash
sudo journalctl -u megdhoot-pipeline.service -f
```

## Monitoring & Logs

### Log Locations

- **Cron runs**: `/home/vedanschi/meghdoot/logs/pipeline_YYYYMMDD_HHMMSS.log`
- **Systemd timer**: `journalctl -u megdhoot-pipeline.service`
- **Application logs**: Created by `meghdoot.utils.logging`

### Check Latest Forecast

```bash
# View metadata
gsutil cat gs://megdhoot-satellite-data/forecasts/latest/metadata.json

# List forecast files
gsutil ls gs://megdhoot-satellite-data/forecasts/latest/

# Download latest forecast image
gsutil cp gs://megdhoot-satellite-data/forecasts/latest/forecast_step_0.png ./
```

### Common Issues

**1. MOSDAC Authentication Failed**
- Verify `config.json` exists and is valid
- Check if MOSDAC credentials have expired (need re-authentication)
- Solution: Re-run `python -m meghdoot.data.mdapi --login`

**2. "No files were successfully preprocessed"**
- Verify raw satellite files were downloaded (check `/home/jupyter/meghdoot_data/raw/`)
- Check if preprocessing paths are writable
- May indicate corrupted MOSDAC file; will auto-retry on next run

**3. GCS Upload Failed**
- Verify `gcs_bucket` is set in `configs/default.yaml`
- Check GCS credentials: `gcloud auth list`
- Ensure bucket exists and is writable
- Check GCS permissions: `gsutil iam ch user:your-email@example.com:objectAdmin gs://your-bucket`

**4. Pipeline Times Out**
- Diffusion inference can take 30-60s (6 steps × 50 inference steps per step)
- Verify GPU is available: `nvidia-smi`
- Check if other jobs are using GPU: `nvidia-smi pmon -c 1`

## Pipeline Output Format

The GCS bucket will contain:

```
gs://megdhoot-satellite-data/forecasts/latest/
├── metadata.json           # Forecast metadata
├── forecast_step_0.png     # +30 minutes
├── forecast_step_1.png     # +60 minutes
├── forecast_step_2.png     # +90 minutes
├── forecast_step_3.png     # +120 minutes
├── forecast_step_4.png     # +150 minutes
└── forecast_step_5.png     # +180 minutes
```

### Metadata Format

```json
{
  "observation_time": "2026-05-03T12:00:00",
  "forecast_generated_at": "2026-05-03T12:03:45",
  "num_steps": 6,
  "lead_times_minutes": [30, 60, 90, 120, 150, 180],
  "valid_times": [
    "2026-05-03T12:30:00",
    "2026-05-03T13:00:00",
    "2026-05-03T13:30:00",
    "2026-05-03T14:00:00",
    "2026-05-03T14:30:00",
    "2026-05-03T15:00:00"
  ],
  "model": "meghdoot-ai-diffusion",
  "inference_steps": 50
}
```

## Performance Tuning

### GPU Memory

If running out of memory, reduce batch size in diffusion inference:
- Edit `configs/default.yaml` → `diffusion.inference.guidance_scale`
- Or modify `generate_forecast_sequence()` to sample fewer steps

### Inference Speed

- **Current**: ~50-60 seconds for 6 steps (50 inference steps each)
- **Faster**: Reduce `num_inference_steps` (trade quality for speed)
- **Baseline**: 30 steps = ~30-40 seconds (slightly lower quality)
- **Fastest**: 20 steps = ~20-30 seconds (noticeably degraded forecasts)

Adjust in `configs/default.yaml`:
```yaml
diffusion:
  inference:
    num_inference_steps: 50  # ← reduce this
```

## Next Steps

1. **Frontend**: Deploy React/Next.js dashboard that reads forecasts from GCS bucket (see FRONTEND_SETUP.md)
2. **Alerts**: Add email/Slack notifications for pipeline failures
3. **Multi-Region**: Extend pipeline to forecast multiple regions simultaneously
4. **Ensembles**: Run multiple model checkpoints and average predictions

---

**Questions?** Check logs at `/home/vedanschi/meghdoot/logs/` or review pipeline code at `src/meghdoot/deploy/pipeline.py`.
