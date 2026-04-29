"""
visualize_predictions.py – Generate Comparative Visual Grids
============================================================
Produces high-resolution PNGs comparing Meghdoot-AI, ConvLSTM,
and PySTEPS against Ground Truth.
"""
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from meghdoot.data.dataset import INSATSequenceDataset, LatentSequenceDataset
from meghdoot.evaluation.baselines import ConvLSTMPredictor, pysteps_forecast
from meghdoot.models.diffusion import MeghdootDiffusion
from meghdoot.models.vae import SatelliteVAE
from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import get_device, seed_everything
from meghdoot.utils.logging import get_logger

log = get_logger(__name__)

def plot_and_save_grid(
    target: np.ndarray, 
    meghdoot_pred: np.ndarray, 
    convlstm_pred: np.ndarray, 
    pysteps_pred: np.ndarray, 
    sample_idx: int, 
    out_dir: str
):
    """Generates a clean, high-contrast visual grid for the TIR1 channel."""
    T = target.shape[0]  
    
    # Sleek, functional dark mode aesthetic
    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=T, ncols=4, figsize=(16, 4 * T))
    
    col_titles = ["Ground Truth", "Meghdoot-AI", "ConvLSTM", "PySTEPS (Optical Flow)"]
    
    for t in range(T):
        frames = [
            target[t, 0],         # Index 0 is TIR1
            meghdoot_pred[t, 0],
            convlstm_pred[t, 0],
            pysteps_pred[t, 0]
        ]
        
        for col, frame in enumerate(frames):
            ax = axes[t, col]
            # 'bone' colormap provides sharp, professional contrast for satellite infrared
            im = ax.imshow(frame, cmap='bone', vmin=-1.0, vmax=1.0)
            ax.axis('off')
            
            if t == 0:
                ax.set_title(col_titles[col], fontsize=16, fontweight='bold', pad=15)
            if col == 0:
                ax.text(-0.1, 0.5, f"T + {t+1}", transform=ax.transAxes, 
                        fontsize=14, fontweight='bold', va='center', ha='right')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"comparison_sample_{sample_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved visual grid to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test.yaml")
    parser.add_argument("--diffusion-ckpt", type=str, required=True)
    parser.add_argument("--convlstm-ckpt", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])
    
    out_dir = "results/visualizations"
    os.makedirs(out_dir, exist_ok=True)

    log.info("Loading Datasets...")
    pixel_dataset = INSATSequenceDataset(cfg["data"]["paths"]["processed"], seq_len_past=3, seq_len_future=6)
    latent_dataset = LatentSequenceDataset(cfg["data"]["paths"]["latents"], seq_len_past=3, seq_len_future=6)

    log.info("Loading Models...")
    vae = SatelliteVAE(cfg).to(device)
    vae.eval()
    
    diffusion = MeghdootDiffusion(cfg).to(device)
    diffusion.load(args.diffusion_ckpt)
    diffusion.eval()
    
    convlstm = ConvLSTMPredictor(
        in_channels=len(cfg["data"]["channels"]),
        hidden_dims=cfg["evaluation"]["baselines"]["convlstm"]["hidden_dims"],
        kernel_size=cfg["evaluation"]["baselines"]["convlstm"]["kernel_size"]
    ).to(device)
    convlstm.load_state_dict(torch.load(args.convlstm_ckpt, map_location=device))
    convlstm.eval()

    log.info(f"Generating visualizations for {args.n_samples} samples...")
    
    with torch.no_grad():
        for i in range(min(args.n_samples, len(pixel_dataset))):
            pixel_sample = pixel_dataset[i]
            latent_sample = latent_dataset[i]
            
            # 1. Ground Truth
            target_pixels = pixel_sample["target"].numpy()  # [T_future, C, H, W]
            
            # 2. Meghdoot-AI Prediction
            history_latents = latent_sample["history"].unsqueeze(0).to(device)
            pred_latents = diffusion.sample(history_latents, num_inference_steps=cfg["diffusion"]["inference"]["num_inference_steps"])
            meghdoot_pred = vae.decode(pred_latents).squeeze(0).cpu().numpy()
            
            # 3. ConvLSTM Prediction
            history_pixels = pixel_sample["history"].unsqueeze(0).to(device)
            current_input = history_pixels
            convlstm_preds = []
            for _ in range(target_pixels.shape[0]):
                pred = convlstm(current_input)
                convlstm_preds.append(pred.unsqueeze(1))
                current_input = torch.cat([current_input[:, 1:], pred.unsqueeze(1)], dim=1)
            convlstm_pred = torch.cat(convlstm_preds, dim=1).squeeze(0).cpu().numpy()
            
            # 4. PySTEPS Prediction
            pysteps_pred = pysteps_forecast(pixel_sample["history"].numpy(), n_leadtimes=target_pixels.shape[0])
            
            # Plot and save
            plot_and_save_grid(target_pixels, meghdoot_pred, convlstm_pred, pysteps_pred, i, out_dir)

    log.info("Visualizations complete.")

if __name__ == "__main__":
    main()
