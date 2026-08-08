import numpy as np
import torch
from scipy.ndimage import sobel

def wind_components_to_direction(u, v):
    if isinstance(u, np.ndarray):
        direction = np.degrees(np.arctan2(v, u)) % 360
    elif isinstance(u, torch.Tensor):
        direction = torch.rad2deg(torch.atan2(v, u)) % 360
    else:
        raise TypeError("Only numpy arrays or torch tensors are supported")
    return direction

def wind_direction_to_components(direction):
    if isinstance(direction, np.ndarray):
        radians = np.radians(direction)
        u = np.cos(radians)
        v = np.sin(radians)
    elif isinstance(direction, torch.Tensor):
        radians = torch.deg2rad(direction)
        u = torch.cos(radians)
        v = torch.sin(radians)
    else:
        raise TypeError("Only numpy arrays or torch tensors are supported")
    return u, v

def calculate_slope_from_dem(dem):
    if isinstance(dem, torch.Tensor):
        dem = dem.detach().cpu().numpy()
    
    dx = sobel(dem, axis=1)
    dy = sobel(dem, axis=0)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    return slope

def apply_wind_effect(snow_distribution, wind_u, wind_v, dem, config=None, ndvi=None):
    is_tensor = isinstance(snow_distribution, torch.Tensor)
    if is_tensor:
        snow = snow_distribution.detach().cpu().numpy()
    else:
        snow = snow_distribution.copy()

    # No forced modifications
    adjusted_snow = snow.copy()
    adjusted_snow = np.clip(adjusted_snow, 0.0, 1.0)

    if is_tensor:
        adjusted_snow = torch.tensor(adjusted_snow, dtype=torch.float32, device=snow_distribution.device)

    return adjusted_snow