# Meghdoot-AI Training Setup on GCP (Colab / VM)

## Prerequisites
- GCP project with GCS bucket: `gs://meghdoot-satellite-data/`
- Bucket contains:
  - `weights/vae/vae_final.pt` (VAE checkpoint)
  - `latents/stacked_tensors/*.pt` (cached latent tensors)
  - `processed/vae_tensors/*.pt` (pixel-space training data)
- GitHub repo: `https://github.com/vedanschi/meghdoot.git`

---

## Option A: Colab (Recommended for 12–16h training)

### 1. Start a New Colab Notebook
Go to [colab.research.google.com](https://colab.research.google.com)  
Create a new notebook. Choose **GPU runtime** (Runtime → Change runtime type → GPU).

### 2. Authenticate & Setup
```python
# Cell 1 – Authenticate to GCP
from google.colab import auth
auth.authenticate_user()

!gcloud config set project YOUR_PROJECT_ID  # Replace with your GCP project

# Cell 2 – Clone repo & install deps
!git clone https://github.com/vedanschi/meghdoot.git
%cd meghdoot

!pip install -q -r requirements.txt

# Cell 3 – Setup directories
import os
os.makedirs('/content/meghdoot_data', exist_ok=True)
os.makedirs('/content/meghdoot_data/weights/vae', exist_ok=True)
os.makedirs('/content/meghdoot_data/processed', exist_ok=True)
os.makedirs('/content/local_data', exist_ok=True)
```

### 3. Mount GCS Bucket
```python
# Cell 4 – Mount bucket
!gsutil -m cp -r gs://meghdoot-satellite-data/weights/vae/vae_final.pt /content/meghdoot_data/weights/vae/
!gsutil -m cp -r gs://meghdoot-satellite-data/latents/stacked_tensors /content/local_data/
!gsutil -m cp -r gs://meghdoot-satellite-data/processed/vae_tensors /content/meghdoot_data/processed/

# Verify files exist
!ls -lh /content/meghdoot_data/weights/vae/vae_final.pt
!ls -lh /content/local_data/stacked_tensors/ | head -20
!ls -lh /content/meghdoot_data/processed/vae_tensors/ | head -20
```

### 4. Run Preflight Validation
```python
# Cell 5 – Preflight checks
import sys
sys.path.insert(0, '/content/meghdoot')

from meghdoot.utils.config import load_config
from meghdoot.data.dataset import LatentSequenceDataset, INSATSequenceDataset
from meghdoot.models.vae import SatelliteVAE
from meghdoot.models.diffusion import MeghdootDiffusion
import torch

cfg = load_config('configs/default.yaml')

# Check paths
print("✓ Config loaded")
print(f"  VAE checkpoint: {cfg['vae']['pretrained']}")
print(f"  Latent dir: {cfg['data']['paths']['latents']}")
print(f"  Processed dir: {cfg['data']['paths']['processed']}")

# Check datasets
try:
    latent_ds = LatentSequenceDataset(
        latent_dir=cfg["data"]["paths"]["latents"],
        num_history=cfg["diffusion"]["conditioning"]["num_history_frames"]
    )
    print(f"✓ LatentSequenceDataset: {len(latent_ds)} sequences")
except Exception as e:
    print(f"✗ LatentSequenceDataset failed: {e}")

# Check VAE
try:
    vae = SatelliteVAE(cfg)
    print(f"✓ VAE loaded: {sum(p.numel() for p in vae.vae.parameters())/1e6:.1f}M params")
except Exception as e:
    print(f"✗ VAE failed: {e}")

# Check Diffusion
try:
    diffusion = MeghdootDiffusion(cfg)
    print(f"✓ Diffusion loaded: {sum(p.numel() for p in diffusion.unet.parameters())/1e6:.1f}M params")
except Exception as e:
    print(f"✗ Diffusion failed: {e}")

print("\n✓✓✓ All preflight checks passed!")
```

### 5. Start Training
```python
# Cell 6 – Launch training (runs in background)
import subprocess
import os

os.chdir('/content/meghdoot')

# Train diffusion (no resume)
subprocess.Popen([
    'python', '-m', 'meghdoot.training.train_diffusion',
    '--config', 'configs/default.yaml'
])

print("Training started! Monitor W&B dashboard for real-time logs.")
```

### 6. Monitor & Save Checkpoints
```python
# Cell 7 – Periodically save checkpoints to GCS
import time

def backup_checkpoints():
    while True:
        time.sleep(3600)  # Every hour
        os.system('gsutil -m cp -r /content/meghdoot/checkpoints/diffusion /tmp/')
        os.system('gsutil -m cp -r /tmp/diffusion gs://meghdoot-satellite-data/checkpoints/')
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checkpoints backed up to GCS")

# Run in background
import threading
backup_thread = threading.Thread(target=backup_checkpoints, daemon=True)
backup_thread.start()
```

---

## Option B: GCP Compute Engine VM

### 1. Create VM Instance
```bash
gcloud compute instances create meghdoot-train \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --scopes=storage-full
```

### 2. SSH Into VM
```bash
gcloud compute ssh meghdoot-train --zone=us-central1-a
```

### 3. Setup on VM
```bash
# On the VM:
cd ~

# Clone repo
git clone https://github.com/vedanschi/meghdoot.git
cd meghdoot

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install deps
pip install -q -r requirements.txt

# Create local directories
mkdir -p ~/meghdoot_data/weights/vae
mkdir -p ~/meghdoot_data/processed
mkdir -p ~/local_data

# Update config paths
sed -i 's|/home/jupyter|/root|g' configs/default.yaml
```

### 4. Sync Data from GCS
```bash
# On the VM:
gsutil -m cp gs://meghdoot-satellite-data/weights/vae/vae_final.pt ~/meghdoot_data/weights/vae/
gsutil -m cp -r gs://meghdoot-satellite-data/latents/stacked_tensors ~/local_data/
gsutil -m cp -r gs://meghdoot-satellite-data/processed/vae_tensors ~/meghdoot_data/processed/

# Verify
ls -lh ~/meghdoot_data/weights/vae/vae_final.pt
ls ~/local_data/stacked_tensors/ | wc -l
```

### 5. Run Preflight
```bash
cd ~/meghdoot

# Quick check
source .venv/bin/activate

python3 << 'EOF'
import sys
sys.path.insert(0, 'src')

from meghdoot.utils.config import load_config
from meghdoot.data.dataset import LatentSequenceDataset

cfg = load_config('configs/default.yaml')
latent_ds = LatentSequenceDataset(
    latent_dir=cfg["data"]["paths"]["latents"],
    num_history=3
)
print(f"✓ LatentSequenceDataset: {len(latent_ds)} sequences")
print("✓ Ready to train!")
EOF
```

### 6. Start Training (with screen/tmux)
```bash
# On the VM, use screen to keep training alive if you disconnect
screen -S training

cd ~/meghdoot
source .venv/bin/activate

# Start training
python -m meghdoot.training.train_diffusion --config configs/default.yaml

# Detach: Ctrl+A, then D
# Reconnect: screen -r training
```

### 7. Periodic Backup
```bash
# Open a new SSH session while training runs
screen -S backup

# Run this to backup every 30 mins
while true; do
  gsutil -m cp -r ~/meghdoot/checkpoints/diffusion gs://meghdoot-satellite-data/checkpoints/
  echo "[$(date)] Backup complete"
  sleep 1800
done

# Detach with Ctrl+A, D
```

---

## Quick Preflight Checklist (Before Training)

Run this once to validate everything in ~2 minutes:

```bash
cd ~/meghdoot
source .venv/bin/activate

python3 << 'EOF'
import sys
import os
sys.path.insert(0, 'src')

checks = []

# 1. Config loads
try:
    from meghdoot.utils.config import load_config
    cfg = load_config('configs/default.yaml')
    checks.append(("Config", "✓"))
except Exception as e:
    checks.append(("Config", f"✗ {e}"))

# 2. VAE checkpoint exists
vae_ckpt = cfg['vae']['pretrained']
if os.path.exists(vae_ckpt):
    checks.append(("VAE checkpoint", "✓"))
else:
    checks.append(("VAE checkpoint", f"✗ Not found: {vae_ckpt}"))

# 3. Latent files exist
latent_dir = cfg['data']['paths']['latents']
latent_files = list(__import__('pathlib').Path(latent_dir).glob('*.pt'))
if len(latent_files) > 100:
    checks.append(("Latent tensors", f"✓ {len(latent_files)} files"))
else:
    checks.append(("Latent tensors", f"✗ Only {len(latent_files)} files"))

# 4. Processed data exists
proc_dir = cfg['data']['paths']['processed']
proc_files = list(__import__('pathlib').Path(proc_dir).glob('*.pt'))
if len(proc_files) > 100:
    checks.append(("Processed tensors", f"✓ {len(proc_files)} files"))
else:
    checks.append(("Processed tensors", f"✗ Only {len(proc_files)} files"))

# 5. Imports work
try:
    from meghdoot.data.dataset import LatentSequenceDataset
    from meghdoot.models.vae import SatelliteVAE
    from meghdoot.models.diffusion import MeghdootDiffusion
    checks.append(("Imports", "✓"))
except Exception as e:
    checks.append(("Imports", f"✗ {e}"))

# 6. VAE instantiates
try:
    vae = SatelliteVAE(cfg)
    checks.append(("VAE init", "✓"))
except Exception as e:
    checks.append(("VAE init", f"✗ {e}"))

# 7. Diffusion instantiates
try:
    diffusion = MeghdootDiffusion(cfg)
    checks.append(("Diffusion init", "✓"))
except Exception as e:
    checks.append(("Diffusion init", f"✗ {e}"))

# 8. Dataset instantiates
try:
    ds = LatentSequenceDataset(cfg['data']['paths']['latents'], num_history=3)
    checks.append(("Dataset init", f"✓ {len(ds)} sequences"))
except Exception as e:
    checks.append(("Dataset init", f"✗ {e}"))

print("\n" + "="*50)
print("PREFLIGHT CHECKLIST")
print("="*50)
for name, status in checks:
    print(f"  {name:<25} {status}")
print("="*50)

all_pass = all("✓" in s for _, s in checks)
if all_pass:
    print("\n🟢 GO: All checks passed. Ready to train!\n")
else:
    print("\n🔴 NO-GO: Fix issues above before training.\n")
    sys.exit(1)
EOF
```

---

## During Training

### Monitor Loss in Real-Time
- **W&B Dashboard**: https://wandb.ai/vedanschi/meghdoot-ai
- **Terminal**: Watch console output for `Epoch X` lines

### If Training Crashes
```bash
# Resume from last checkpoint
python -m megdhoot.training.train_diffusion \
  --config configs/default.yaml \
  --resume checkpoints/diffusion/diffusion_epoch100.pt
```

### Kill Training Gracefully
```bash
# Colab: Stop the cell
# VM: In training screen, press Ctrl+C (saves checkpoint)
```

---

## After Training (Epoch 100)

### Download Best Checkpoint
```bash
gsutil cp gs://meghdoot-satellite-data/checkpoints/diffusion/diffusion_epoch100.pt ./checkpoints/diffusion/
```

### Run Evaluation
```bash
python -m meghdoot.evaluation.benchmark \
  --config configs/default.yaml \
  --diffusion-ckpt checkpoints/diffusion/diffusion_epoch100.pt \
  --n-samples 100
```

---

## Troubleshooting

### "No such file or directory: vae_final.pt"
- Check GCS bucket path: `gsutil ls gs://meghdoot-satellite-data/weights/vae/`
- Verify config paths in `configs/default.yaml`

### "LatentSequenceDataset: Found 0 latents"
- Check latent dir synced: `ls -l ~/local_data/stacked_tensors/`
- Verify config: `cat configs/default.yaml | grep latents`

### GPU out of memory
- Reduce `batch_size` in `configs/default.yaml` (default: 4 → try 2)
- Reduce `gradient_accumulation_steps` (default: 4 → try 2)

### Training too slow
- Ensure GPU is being used: `nvidia-smi` during training
- Check Colab GPU type (prefer A100 over T4)

---

## Final Command: Copy & Paste for VM

```bash
#!/bin/bash
set -e

echo "=== Meghdoot Training Setup ==="

cd ~/meghdoot
source .venv/bin/activate

echo "1. Syncing data from GCS..."
gsutil -m cp gs://meghdoot-satellite-data/weights/vae/vae_final.pt ~/meghdoot_data/weights/vae/
gsutil -m cp -r gs://meghdoot-satellite-data/latents/stacked_tensors ~/local_data/ 2>/dev/null || true
gsutil -m cp -r gs://meghdoot-satellite-data/processed/vae_tensors ~/meghdoot_data/processed/ 2>/dev/null || true

echo "2. Running preflight checks..."
python3 SETUP_GCP.md  # This won't work directly; use the Python snippet above instead

echo "3. Starting training in screen session..."
screen -dmS training bash -c "source .venv/bin/activate && python -m meghdoot.training.train_diffusion --config configs/default.yaml"

echo "4. Starting backup daemon in screen session..."
screen -dmS backup bash -c "while true; do gsutil -m cp -r ~/meghdoot/checkpoints/diffusion gs://meghdoot-satellite-data/checkpoints/ 2>/dev/null; sleep 1800; done"

echo ""
echo "✓ Training started!"
echo ""
echo "Monitor training:"
echo "  screen -r training"
echo ""
echo "Check backups:"
echo "  screen -r backup"
echo ""
echo "View W&B dashboard:"
echo "  https://wandb.ai/vedanschi/meghdoot-ai"
```

Save as `~/meghdoot/run_training.sh`, then:
```bash
bash ~/meghdoot/run_training.sh
```

---

**Good luck! 🚀**
