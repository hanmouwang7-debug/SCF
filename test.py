import torch
import yaml
import numpy as np
from pathlib import Path
from models.deepLabv3plus import DeepLabV3PlusSCF
from utils.data_loader import get_data_loader, load_normalization_stats
import torch.nn.functional as F

def test_model(model, loader, device, stats):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3PlusSCF(**config['model']['deeplabv3plus']).to(device)
    
    model_path = Path(config['paths']['checkpoints']) / "deeplab_best_mae.pth"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    test_loader = get_data_loader(config, 'test')
    stats = load_normalization_stats(config['data']['dir'])

    test_mae, test_rmse = test_model(model, test_loader, device, stats)
    print(f"Test Set Results:")
    print(f"MAE = {test_mae:.4f}")
    print(f"RMSE = {test_rmse:.4f}")

if __name__ == "__main__":
    main()