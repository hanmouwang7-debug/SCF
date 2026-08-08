# Snow Distribution Prediction System

![Snow Prediction Example](outputs/samples/epoch_50_samples.png)

This project uses a deep learning model (DeepLabV3+) to predict snow distribution in specific areas based on multi-source data including terrain, meteorology, and NDVI.

## Main Features

- Input multi-channel spatial features (elevation, slope, NDVI, temperature, precipitation, wind speed, wind direction, humidity)
- Generate snow cover fraction (SCF) maps
- Use DeepLabV3+ with a ResNet-50 backbone for regression
- Visualize prediction results

## System Requirements

- Windows 10/11
- NVIDIA GPU (recommended) or CPU
- Anaconda/Miniconda

## Installation Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/hanmouwang7-debug/SCF
   cd SCF
