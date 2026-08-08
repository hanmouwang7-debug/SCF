import os
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Dict
import yaml
import json
import numpy as np
from osgeo import gdal

class SnowDataPreprocessor:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.raw = self.root / "data" / "raw"
        self.processed = self.root / "data" / "processed"
        self.config_path = self.root / "configs" / "config.yaml"
        self.norm_stats_path = self.processed / "normalization_stats.json"

        self.raw_terrain = self.raw / "terrain"
        self.raw_meteorology = self.raw / "meteorology" / "meteorology_metadata.csv"
        self.raw_snow_obs = self.raw / "scf"
        self.raw_ndvi = self.raw / "ndvi"

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.config = self._load_config()
        self.target_size = tuple(self.config["data"]["image_size"])

        self.out_terrain = self.processed / "terrain"
        self.out_scf = self.processed / "scf"
        self.out_ndvi = self.processed / "ndvi"
        [dir_path.mkdir(parents=True, exist_ok=True) for dir_path in [self.out_terrain, self.out_scf, self.out_ndvi]]

        print(f"Initializing preprocessing tool")
        print(f"Target size: {self.target_size[0]}x{self.target_size[1]}")
        print(f" Raw data path: {self.raw}")
        print(f" Output path: {self.processed}")

    def _load_config(self) -> Dict:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if "data" not in config or "image_size" not in config["data"]:
                raise KeyError("Missing 'data.image_size' in config.yaml")
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config file: {str(e)}")

    def _resample_geotiff(self, input_path: Path, output_path: Path, is_mask: bool = False) -> bool:
        try:
            with rasterio.open(input_path) as src:
                resampling_method = Resampling.nearest if is_mask else Resampling.bilinear
                data = src.read(
                    out_shape=(src.count, self.target_size[0], self.target_size[1]),
                    resampling=resampling_method
                )
                transform = src.transform * src.transform.scale(
                    (src.width / data.shape[-1]),
                    (src.height / data.shape[-2])
                )
                with rasterio.open(
                    output_path, 'w', driver='GTiff', height=self.target_size[0],
                    width=self.target_size[1], count=src.count, dtype=data.dtype,
                    crs=src.crs, transform=transform
                ) as dst:
                    dst.write(data)
            return True
        except Exception as e:
            print(f"Resampling failed: {input_path.name} ({str(e)})")
            return False

    def process_terrain(self) -> List[str]:
        print(f"\n Start processing terrain data (elevation + slope)...")
        terrain_files = list(self.raw_terrain.glob("area_*_dem.tif"))
        if not terrain_files:
            raise FileNotFoundError(f"Terrain data not found: {self.raw_terrain}")

        success_regions = []
        for file in terrain_files:
            rid = file.stem.split("_dem")[0]
            print(f"  Processing region: {rid}")

            dem_resampled = self.out_terrain / f"{rid}_dem.tif"
            self._resample_geotiff(file, dem_resampled, False)

            slope_path = self.out_terrain / f"{rid}_slope.tif"
            self._compute_slope(dem_resampled, slope_path)

            elevation_path = self.out_terrain / f"{rid}_elevation.tif"
            dem_resampled.rename(elevation_path)

            print(f"  Generated: elevation + slope")
            success_regions.append(rid)

        print(f"Terrain processing complete: {len(success_regions)} regions")
        return success_regions

    def _compute_slope(self, dem_path: Path, slope_path: Path):
        try:
            dem_ds = gdal.Open(str(dem_path))
            gdal.DEMProcessing(
                str(slope_path),
                dem_ds,
                "slope",
                slopeFormat="degree"
            )
            dem_ds = None
        except Exception as e:
            print(f"Slope calculation failed: {str(e)}")

    def process_scf(self, valid_terrain_regions: List[str]) -> List[str]:
        print(f"\nStart processing snow cover fraction data...")
        scf_files = list(self.raw_snow_obs.glob("area_*_scf.tif"))
        if not scf_files:
            raise FileNotFoundError(f"SCF data not found: {self.raw_snow_obs}")

        scf_file_info = [(file, file.stem.split("_scf")[0]) for file in scf_files
                         if file.stem.split("_scf")[0] in valid_terrain_regions]
        process_results = [(rid, self._resample_geotiff(file, self.out_scf / file.name, is_mask=False))
                           for file, rid in scf_file_info]
        success_regions = [rid for rid, success in process_results if success]

        for (file, rid), (_, success) in zip(scf_file_info, process_results):
            if success:
                print(f"  Done: {file.name}")
        print(f" SCF processing complete: {len(success_regions)}/{len(scf_files)}")
        return success_regions

    def process_ndvi(self, valid_terrain_regions: List[str]) -> List[str]:
        print(f"\nStart processing NDVI data...")
        ndvi_files = list(self.raw_ndvi.glob("area_*_ndvi.tif"))
        if not ndvi_files:
            raise FileNotFoundError(f"NDVI data not found: {self.raw_ndvi}")

        ndvi_file_info = [(file, file.stem.split("_ndvi")[0]) for file in ndvi_files
                          if file.stem.split("_ndvi")[0] in valid_terrain_regions]
        process_results = [(rid, self._resample_geotiff(file, self.out_ndvi / f"{rid}_ndvi.tif", False))
                           for file, rid in ndvi_file_info]
        success_regions = [rid for rid, success in process_results if success]

        for (file, rid), (_, success) in zip(ndvi_file_info, process_results):
            if success:
                print(f"  Done: {file.name}")
        print(f"NDVI processing complete: {len(success_regions)}/{len(ndvi_files)}")
        return success_regions

    def load_and_filter_meteorology(self, valid_regions: List[str]) -> pd.DataFrame:
        print("\nLoading meteorological data...")
        if not self.raw_meteorology.exists():
            raise FileNotFoundError(f"Meteorological file not found: {self.raw_meteorology}")

        meteor_df = pd.read_csv(self.raw_meteorology)
        # Added relative_humidity as required field
        required_cols = ["region_id", "temperature", "precipitation", "wind_speed", "wind_direction", "relative_humidity"]
        if not set(required_cols).issubset(meteor_df.columns):
            raise ValueError(f"Meteorological data missing fields: {required_cols}")

        meteor_df["region_id"] = meteor_df["region_id"].astype(str).str.replace(".", "_", regex=False)
        valid_meteor = meteor_df[meteor_df["region_id"].isin(valid_regions)].copy()
        print(f"Meteorological data loaded: {len(valid_meteor)} records")
        return valid_meteor

    def split_and_save_metadata(self, meteor_df: pd.DataFrame, valid_ndvi_regions: List[str]) -> None:
        print("\nSplitting train/val/test sets...")
        valid_meteor = meteor_df[meteor_df["region_id"].isin(valid_ndvi_regions)].copy()
        unique_regions = valid_meteor["region_id"].unique()

        if len(unique_regions) < 3:
            raise ValueError("Not enough valid regions")

        train_regions, temp = train_test_split(unique_regions, test_size=0.3, random_state=42)
        val_regions, test_regions = train_test_split(temp, test_size=0.5, random_state=42)

        def create_meta(regions, name):
            df = pd.DataFrame({
                "region_id": regions,
                "elevation_path": [f"terrain/{r}_elevation.tif" for r in regions],
                "slope_path": [f"terrain/{r}_slope.tif" for r in regions],
                "scf_path": [f"scf/{r}_scf.tif" for r in regions],
                "ndvi_path": [f"ndvi/{r}_ndvi.tif" for r in regions]
            })
            meta = pd.merge(valid_meteor, df, on="region_id")
            meta.to_csv(self.processed / f"{name}_metadata.csv", index=False)
            print(f"  {name}: {len(meta)} entries")

        create_meta(train_regions, "train")
        create_meta(val_regions, "val")
        create_meta(test_regions, "test")

    def compute_normalization_stats(self):
        print("\nComputing global normalization statistics (elevation + slope + NDVI + meteorology)...")
        train_csv = self.processed / "train_metadata.csv"
        df = pd.read_csv(train_csv)

        def get_values(paths):
            arr = []
            for p in paths:
                with rasterio.open(self.processed / p) as src:
                    d = src.read(1).astype(np.float32)
                    if src.nodata is not None:
                        d = d[d != src.nodata]
                    arr.append(d[~np.isnan(d)].flatten())
            return np.concatenate(arr)

        elev = get_values(df["elevation_path"])
        slope = get_values(df["slope_path"])
        ndvi = get_values(df["ndvi_path"])

        stats = {
            "elevation": {"min": float(elev.min()), "max": float(elev.max())},
            "slope": {"min": float(slope.min()), "max": float(slope.max())},
            "ndvi": {"min": float(ndvi.min()), "max": float(ndvi.max())},
            "temperature": {"mean": float(df["temperature"].mean()), "std": float(df["temperature"].std())},
            "precipitation": {"mean": float(df["precipitation"].mean()), "std": float(df["precipitation"].std())},
            "wind_speed": {"mean": float(df["wind_speed"].mean()), "std": float(df["wind_speed"].std())},
            "wind_direction": {"mean": float(df["wind_direction"].mean()), "std": float(df["wind_direction"].std())},
        }

        with open(self.norm_stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        print(f"Normalization statistics saved:")
        print(f"  Elevation: min={elev.min():.2f}, max={elev.max():.2f}")
        print(f"  Slope: min={slope.min():.2f}, max={slope.max():.2f}")
        print(f"  NDVI: min={ndvi.min():.2f}, max={ndvi.max():.2f}")
        return stats

    def run_pipeline(self):
        print("Starting full preprocessing pipeline")
        t = self.process_terrain()
        s = self.process_scf(t)
        n = self.process_ndvi(t)
        valid = list(set(s) & set(n))
        met = self.load_and_filter_meteorology(valid)
        self.split_and_save_metadata(met, n)
        self.compute_normalization_stats()
        print("\nAll done!")

if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        p = SnowDataPreprocessor(root)
        p.run_pipeline()
    except Exception as e:
        print(f"Failed: {e}")