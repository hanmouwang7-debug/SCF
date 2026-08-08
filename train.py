import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import json

from models.deepLabv3plus import DeepLabV3PlusSCF
from utils.data_loader import get_data_loader, load_normalization_stats
from models.losses import combined_loss
from utils.visualization import save_sample_images

def evaluate(model, loader, device, stats):
    model.eval()
    mae_sum = 0
    rmse_sum = 0
    n = 0
    with torch.no_grad():
        for spatial, scf in loader:
            spatial = spatial.to(device)
            scf = scf.to(device)
            pred = model(spatial)

            mae = F.l1_loss(pred, scf).item()
            rmse = torch.sqrt(F.mse_loss(pred, scf)).item()
            mae_sum += mae * scf.size(0)
            rmse_sum += rmse * scf.size(0)
            n += scf.size(0)
    return mae_sum / n, rmse_sum / n

def main():
    config_path = Path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for p in config['paths'].values():
        Path(p).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3PlusSCF(**config['model']['deeplabv3plus']).to(device)

    # No attention + no physical loss
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 5e-5},
        {'params': model.aspp.parameters(), 'lr': 1e-4},
        {'params': model.decoder.parameters(), 'lr': 1e-4},
        {'params': model.out.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-5)

    criterion = combined_loss
    data_dir = config['data']['dir']
    stats = load_normalization_stats(data_dir)

    train_loader = get_data_loader(config, 'train')
    val_loader = get_data_loader(config, 'val')

    writer = SummaryWriter(config['paths']['logs'])
    best_mae = float('inf')
    patience = config['training']['patience']
    counter = 0
    epochs = config['training']['epochs']
    sample_interval = config['training']['sample_interval']

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_n = 0

        for batch_idx, (spatial, scf) in enumerate(train_loader):
            spatial = spatial.to(device)
            scf = scf.to(device)
            pred = model(spatial)

            # Pure basic loss
            loss = criterion(pred, scf)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * scf.size(0)
            total_n += scf.size(0)

        avg_loss = total_loss / total_n
        val_mae, val_rmse = evaluate(model, val_loader, device, stats)

        print(f"Epoch {epoch+1:2d} | Loss {avg_loss:.4f} | Val MAE {val_mae:.4f} | RMSE {val_rmse:.4f}")

        if (epoch + 1) % sample_interval == 0:
            save_sample_images(spatial, scf, pred, None, config['paths']['samples'], epoch+1)

        try:
            writer.add_scalar('Train/loss', avg_loss, epoch)
            writer.add_scalar('Val/MAE', val_mae, epoch)
            writer.add_scalar('Val/RMSE', val_rmse, epoch)
        except:
            pass

        if val_mae < best_mae:
            best_mae = val_mae
            counter = 0
            torch.save(model.state_dict(), Path(config['paths']['checkpoints'])/'deeplab_best_mae.pth')
            print(f" Best model saved (MAE: {best_mae:.4f})")
        else:
            counter += 1
            print(f" No improvement {counter}/{patience}")
            if counter >= patience:
                print(" Early stopping triggered")
                break

    torch.save(model.state_dict(), Path(config['paths']['checkpoints'])/'deeplab_final.pth')
    writer.close()
    print(f"Training completed. Best validation MAE: {best_mae:.4f}")

if __name__ == '__main__':
    main()