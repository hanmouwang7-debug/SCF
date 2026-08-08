import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import rasterio
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import yaml
import json

def load_config(config_path: str = "configs/config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if 'data' not in config or 'image_size' not in config['data']:
        raise ValueError("Missing 'data.image_size' in config")
    return config

CONFIG = load_config()
IMAGE_HEIGHT, IMAGE_WIDTH = CONFIG['data']['image_size']

def load_normalization_stats(data_dir):
    stats_path = Path(data_dir) / "normalization_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError("Please run preprocess_data.py first to generate normalization stats")
    with open(stats_path, encoding="utf-8") as f:
        return json.load(f)

class SnowDataset(Dataset):
    def __init__(self, data_dir, csv_path, transform=None, mode='train'):
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(self.data_dir / csv_path)
        self.transform = transform
        self.mode = mode
        self.stats = load_normalization_stats(data_dir)
        self.H, self.W = IMAGE_HEIGHT, IMAGE_WIDTH

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        def read_norm(path, key):
            with rasterio.open(self.data_dir / path) as src:
                arr = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
            s = self.stats[key]
            arr = (arr - s['min']) / (s['max'] - s['min'] + 1e-8)
            arr = np.clip(arr, 0, 1)
            return np.nan_to_num(arr, nan=0.0)

        elev = read_norm(row.elevation_path, 'elevation')
        slope = read_norm(row.slope_path, 'slope')
        ndvi = read_norm(row.ndvi_path, 'ndvi')

        temp_val = (row.temperature - self.stats['temperature']['mean']) / (self.stats['temperature']['std'] + 1e-8)
        precip_val = (row.precipitation - self.stats['precipitation']['mean']) / (self.stats['precipitation']['std'] + 1e-8)
        wspeed_val = (row.wind_speed - self.stats['wind_speed']['mean']) / (self.stats['wind_speed']['std'] + 1e-8)

        wdir_deg = row.wind_direction
        wd_sin = np.sin(np.radians(wdir_deg))
        wd_cos = np.cos(np.radians(wdir_deg))

        temp = np.full((self.H, self.W), temp_val, dtype=np.float32)
        precip = np.full((self.H, self.W), precip_val, dtype=np.float32)
        wind_spd = np.full((self.H, self.W), wspeed_val, dtype=np.float32)
        wind_dir_sin = np.full((self.H, self.W), wd_sin, dtype=np.float32)
        wind_dir_cos = np.full((self.H, self.W), wd_cos, dtype=np.float32)
        
        # Added: relative humidity processing (normalized by x/100)
        hum_val = row.relative_humidity / 100.0
        hum = np.full((self.H, self.W), hum_val, dtype=np.float32)

        # Added humidity channel, 8 channels -> 9 channels
        stacked = np.stack([
            elev, slope, ndvi,
            temp, precip, wind_spd,
            wind_dir_sin, wind_dir_cos,
            hum
        ], axis=-1)

        # Add noise regularization to terrain features during training
        if self.mode == 'train':
            noise = np.random.normal(0, 0.02, stacked.shape[:2]).astype(np.float32)
            stacked[:, :, 0] = np.clip(stacked[:, :, 0] + noise, 0, 1)
            stacked[:, :, 1] = np.clip(stacked[:, :, 1] + noise, 0, 1)

        with rasterio.open(self.data_dir / row.scf_path) as src:
            scf = src.read(1).astype(np.float32)
            if src.nodata is not None:
                scf[scf == src.nodata] = np.nan
        scf = np.nan_to_num(scf, nan=0.0)
        scf = np.clip(scf, 0, 1)

        if self.transform:
            aug = self.transform(image=stacked, mask=scf)
            stacked = aug['image']
            scf = aug['mask']

        image = torch.from_numpy(stacked).permute(2, 0, 1).float()
        scf = torch.from_numpy(scf).unsqueeze(0).float()

        return image, scf

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])

def get_val_transforms():
    return A.Compose([])

def get_data_loader(config, mode='train'):
    data_dir = Path(config['data']['dir'])
    csv = config['data'][f'{mode}_csv']
    ts = get_train_transforms() if mode == 'train' else get_val_transforms()
    dataset = SnowDataset(data_dir, csv, ts, mode=mode)
    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(mode == 'train'),
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True
    )
    return loader