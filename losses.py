import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

def l1_loss(pred, target):
    return F.l1_loss(pred, target)

def mse_loss(pred, target):
    return F.mse_loss(pred, target)

# Huber Loss core function
def huber_loss(pred, target, delta=0.1):

    error = torch.abs(pred - target)
    loss = torch.where(error < delta,
                      0.5 * torch.pow(error, 2),
                      delta * (error - 0.5 * delta))
    return loss.mean()

# Combined loss
def combined_loss(pred, target):
    return huber_loss(pred, target)

class PhysicsInformedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        phys_cfg = config['model']['physics']

        self.max_snow_slope = phys_cfg['max_snow_slope']
        self.high_ndvi_threshold = phys_cfg['high_ndvi_threshold']
        self.melting_temp = phys_cfg['melting_temp']
        self.no_precip_threshold = phys_cfg['no_precip_threshold']
        self.physics_weight = config['training']['physics_loss_weight']

        data_dir = Path(config['data']['dir'])
        stats_path = data_dir / "normalization_stats.json"
        with open(stats_path, encoding='utf-8') as f:
            self.stats = json.load(f)

    def forward(self, pred, target, inputs):
        slope = inputs[:, 1:2]
        temp = inputs[:, 3:4]
        precip = inputs[:, 4:5]
        ndvi = inputs[:, 2:3]

        T_mean = self.stats['temperature']['mean']
        T_std = self.stats['temperature']['std']
        P_mean = self.stats['precipitation']['mean']
        P_std = self.stats['precipitation']['std']

        temp_thresh_norm = (self.melting_temp - T_mean) / T_std
        precip_thresh_norm = (self.no_precip_threshold - P_mean) / P_std

        s_min = self.stats["slope"]["min"]
        s_max = self.stats["slope"]["max"]
        slope_thresh_norm = (self.max_snow_slope - s_min) / (s_max - s_min + 1e-8)

        s_min_ndvi = self.stats["ndvi"]["min"]
        s_max_ndvi = self.stats["ndvi"]["max"]
        ndvi_thresh_norm = (self.high_ndvi_threshold - s_min_ndvi) / (s_max_ndvi - s_min_ndvi + 1e-8)

        steep_mask = (slope > slope_thresh_norm).float()
        ndvi_mask = (ndvi > ndvi_thresh_norm).float()
        too_hot_mask = (temp > temp_thresh_norm).float()
        no_precip_mask = (precip <= precip_thresh_norm).float()

        # Increase weight of temperature and precipitation penalties
        steep_pen = torch.abs(pred * steep_mask).mean()
        ndvi_pen = torch.abs(pred * ndvi_mask).mean()
        too_hot_pen = 2.0 * torch.abs(pred * too_hot_mask).mean() 
        no_precip_pen = 2.0 * torch.abs(pred * no_precip_mask).mean() 
        physics_loss = steep_pen + ndvi_pen + too_hot_pen + no_precip_pen
        base_loss = combined_loss(pred, target)
        total_loss = base_loss + self.physics_weight * physics_loss
        return total_loss