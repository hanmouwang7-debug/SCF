import torch
import torch.nn.functional as F
import json
from pathlib import Path

class SnowPhysicsModel:
    def __init__(self, config):
        self.config = config
        phys_cfg = config['model']['physics']

        self.max_snow_slope = phys_cfg['max_snow_slope']
        self.slope_factor = phys_cfg['slope_factor']
        self.high_ndvi_threshold = phys_cfg['high_ndvi_threshold']
        self.ndvi_reduction_factor = phys_cfg['ndvi_reduction_factor']
        self.melting_temp = phys_cfg['melting_temp']
        self.no_precip_threshold = phys_cfg['no_precip_threshold']

        data_dir = Path(config['data']['dir'])
        stats_path = data_dir / "normalization_stats.json"
        with open(stats_path, encoding='utf-8') as f:
            self.stats = json.load(f)

    def postprocess(self, pred_snow, inputs):
        pred = pred_snow.clone()

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

    
        mask_too_hot = temp > temp_thresh_norm
        mask_no_precip = precip <= precip_thresh_norm
        mask_steep = slope > slope_thresh_norm
        mask_high_ndvi = ndvi > ndvi_thresh_norm

        pred[mask_too_hot] *= 0.1
        pred[mask_no_precip] *= 0.1
        pred[mask_steep] *= 0.2

        slope_weight = 1.0 - slope * self.slope_factor
        slope_weight = torch.clamp(slope_weight, 0.2, 1.0)
        pred *= slope_weight

        ndvi_mask = mask_high_ndvi.float()
        ndvi_weight = 1.0 - ndvi_mask * self.ndvi_reduction_factor
        pred *= ndvi_weight

        return torch.clamp(pred, 0.0, 1.0)

    def to(self, device):
        return self