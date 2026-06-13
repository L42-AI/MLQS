#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from data.loader import load_single_sensor_csv
from preprocessing.missing import interpolate_linear
from preprocessing.noise import apply_filter_to_columns
from preprocessing.resample import resample_to_uniform_grid
from schema import SENSOR_MAP
from utils.viz import plot_sensor_time_series

experiment_config = Config()

raw_data_directory = experiment_config.raw_dir
target_csv = None
for csv_file in sorted(raw_data_directory.rglob("Accelerometer.csv")):
    if csv_file.stat().st_size > 0:
        target_csv = csv_file
        break

if target_csv is None:
    print("No Accelerometer.csv found in experiment data directory")
    exit(1)

print(f"Processing: {target_csv.name}")
sensor_schema = SENSOR_MAP.get("Accelerometer")
sensor_data = load_single_sensor_csv(target_csv, schema=sensor_schema)

sensor_columns = [c for c in sensor_data.columns if c not in ("time", "seconds_elapsed")]

sensor_data = resample_to_uniform_grid(
    sensor_data,
    resample_rule=experiment_config.preprocessing.resample_rule,
    time_column="seconds_elapsed",
)
print(f"Resampled to {experiment_config.preprocessing.resample_rule}: {len(sensor_data)} samples")

preprocessing_config = experiment_config.preprocessing
sensor_data = apply_filter_to_columns(
    sensor_data,
    sensor_columns,
    filter_method=preprocessing_config.filter_method,
    cutoff_frequency=preprocessing_config.filter_cutoff,
    sample_rate_hz=10.0,
    filter_order=preprocessing_config.filter_order,
    filter_type=preprocessing_config.filter_type,
)
print(f"Applied {preprocessing_config.filter_method} filter (cutoff={preprocessing_config.filter_cutoff} Hz)")

sensor_data = interpolate_linear(
    sensor_data,
    max_gap=preprocessing_config.imputation_max_gap,
    columns=sensor_columns,
)
print(f"Missing after: {sensor_data.isnull().sum().sum()} cells")

output_path = experiment_config.processed_dir / f"{target_csv.stem}_processed.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
sensor_data.to_csv(output_path)
print(f"Saved to {output_path}")
