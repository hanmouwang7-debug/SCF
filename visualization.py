import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

def save_sample_images(spatial_features, real_imgs, fake_imgs, conditions, save_dir, epoch, max_display=3):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    display_samples = min(real_imgs.shape[0], max_display)

    spatial_features = spatial_features[:display_samples].detach().cpu()
    elev_imgs   = spatial_features[:, 0:1, :, :]
    slope_imgs  = spatial_features[:, 1:2, :, :]
    ndvi_imgs   = spatial_features[:, 2:3, :, :]

    real_imgs = real_imgs[:display_samples].detach().cpu().clamp(0, 1)
    fake_imgs = fake_imgs[:display_samples].detach().cpu().clamp(0, 1)

    fig, axes = plt.subplots(display_samples, 5, figsize=(25, display_samples * 5))
    if display_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(display_samples):
        axes[i,0].imshow(elev_imgs[i].squeeze(), cmap='terrain')
        axes[i,0].set_title("Elevation")
        axes[i,0].axis('off')

        axes[i,1].imshow(slope_imgs[i].squeeze(), cmap='viridis')
        axes[i,1].set_title("Slope")
        axes[i,1].axis('off')

        axes[i,2].imshow(ndvi_imgs[i].squeeze(), cmap='RdYlGn', vmin=-1, vmax=1)
        axes[i,2].set_title("NDVI")
        axes[i,2].axis('off')

        axes[i,3].imshow(real_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i,3].set_title("Real SCF")
        axes[i,3].axis('off')

        axes[i,4].imshow(fake_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i,4].set_title("Pred SCF")
        axes[i,4].axis('off')

    plt.tight_layout()
    save_path = save_dir / f"epoch_{epoch}_elev_slope_ndvi.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"3-channel sample image saved: {save_path}")