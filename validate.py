import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from utils.data_loader import get_data_loader, load_normalization_stats
from models.deepLabv3plus import DeepLabV3PlusSCF

def evaluate_full(model, loader, device):
    model.eval()
    total_mae = 0.0
    total_rmse = 0.0
    total_samples = 0

    with torch.no_grad():
        for spatial, scf in loader:
            spatial = spatial.to(device)
            scf = scf.to(device)
            pred = model(spatial)

            mae = F.l1_loss(pred, scf)
            rmse = torch.sqrt(F.mse_loss(pred, scf))

            batch_size = scf.size(0)
            total_mae += mae.item() * batch_size
            total_rmse += rmse.item() * batch_size
            total_samples += batch_size

    avg_mae = total_mae / total_samples
    avg_rmse = total_rmse / total_samples
    return avg_mae, avg_rmse

if __name__ == "__main__":
    print("="*50)
    print(" Model Validation Script")
    print("="*50)
    
    config = yaml.safe_load(open("configs/config.yaml","r",encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepLabV3PlusSCF(**config["model"]["deeplabv3plus"]).to(device)
    ckpt_path = Path(config["paths"]["checkpoints"]) / "deeplab_best_mae.pth"

    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=True) 
        print(f"Loaded best model weights: {ckpt_path.name}")
    else:
        print("Weight file not found. Please train the model first.")
        exit()

    stats = load_normalization_stats(config["data"]["dir"])
    val_loader = get_data_loader(config, mode="val")
    
    print(f"Starting validation ({len(val_loader.dataset)} samples)...")
    mae, rmse = evaluate_full(model, val_loader, device)
    
    print("-" * 30)
    print(f"Validation Results:")
    print(f"   MAE  = {mae:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    print("="*50)