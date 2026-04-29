"""
train_convlstm.py – Train the ConvLSTM Baseline
===============================================
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from meghdoot.data.dataset import INSATSequenceDataset
from meghdoot.evaluation.baselines import ConvLSTMPredictor
from meghdoot.utils.config import load_config
from meghdoot.utils.logging import get_logger

log = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["project"]["device"])
    
    log.info("Loading Pixel Dataset for ConvLSTM...")
    # ConvLSTM operates on raw pixels, not latents
    dataset = INSATSequenceDataset(
        data_dir=cfg["data"]["paths"]["processed"],
        seq_len_past=3,
        seq_len_future=6
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    log.info("Initializing ConvLSTM Model...")
    model = ConvLSTMPredictor(
        in_channels=len(cfg["data"]["channels"]),
        hidden_dims=cfg["evaluation"]["baselines"]["convlstm"]["hidden_dims"],
        kernel_size=cfg["evaluation"]["baselines"]["convlstm"]["kernel_size"],
        num_layers=cfg["evaluation"]["baselines"]["convlstm"]["num_layers"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    os.makedirs("checkpoints/baselines", exist_ok=True)

    log.info(f"Starting Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            # Move to GPU
            x = batch["history"].to(device)  # [B, 3, C, H, W]
            y = batch["target"].to(device)   # [B, 6, C, H, W]
            
            optimizer.zero_grad()
            
            # Predict future sequence autoregressively
            current_input = x
            predictions = []
            for t in range(y.shape[1]):
                pred = model(current_input)  # Predict next frame
                predictions.append(pred.unsqueeze(1))
                
                # Append prediction to history and drop oldest frame
                current_input = torch.cat([current_input[:, 1:], pred.unsqueeze(1)], dim=1)
                
            predictions = torch.cat(predictions, dim=1)
            
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        log.info(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

    # Save final model
    save_path = "checkpoints/baselines/convlstm.pt"
    torch.save(model.state_dict(), save_path)
    log.info(f"ConvLSTM saved successfully to {save_path}")

if __name__ == "__main__":
    main()
