#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from data.loader import compute_sensor_summary, detect_data_quality_issues, load_single_sensor_csv
from schema import detect_schema

experiment_config = Config()
raw_data_directory = experiment_config.raw_dir

for csv_file in sorted(raw_data_directory.rglob("*.csv")):
    if csv_file.stat().st_size == 0:
        continue

    sensor_schema = detect_schema(csv_file.stem)
    sensor_data = load_single_sensor_csv(csv_file, schema=sensor_schema)
    data_summary = compute_sensor_summary(sensor_data)
    quality_report = detect_data_quality_issues(sensor_data)

    print(f"{'='*60}")
    print(f"{csv_file.stem:40s}  {sensor_data.shape}")
    if sensor_schema:
        print(f"  Device: {sensor_schema.device}  Sample rate: {sensor_schema.sample_rate} Hz")
    print(f"  Columns: {list(sensor_data.columns)}")
    print(f"  Missing: {data_summary['missing']['total']} cells ({data_summary['missing']['pct']}%)")
    if quality_report["constant_columns"]:
        print(f"  Constants: {quality_report['constant_columns']}")
    if quality_report["outliers"]:
        print(f"  Outliers: {quality_report['outliers']}")
    print()
