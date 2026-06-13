"""Load phyphox sensor CSVs — all at once, merged by time, with labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from consts import SRC
from labels import Label
from preprocessing.resample import synchronise_sensor_frames
from schema import SensorSchema, detect_schema


def load_single_sensor_csv(
    file_path: str | Path, schema: SensorSchema | None = None
) -> pd.DataFrame:
    file_path = Path(file_path)
    sensor_data = pd.read_csv(file_path, encoding="utf-8-sig")
    sensor_data.columns = sensor_data.columns.str.strip()
    if schema is None:
        schema = detect_schema(file_path.stem)
    if schema is not None:
        schema.validate(sensor_data)
    for time_column in ("time", "seconds_elapsed"):
        if time_column in sensor_data.columns:
            sensor_data[time_column] = pd.to_numeric(sensor_data[time_column], errors="coerce")
    return sensor_data


SENSOR_NAMES_TO_SKIP = {"Manifest", "Metadata", "Annotation"}


def _detect_label_from_directory_name(directory_name: str) -> Label | None:
    for activity_label in Label:
        if activity_label.value in directory_name.upper():
            return activity_label
    return None


def _load_sensor_csvs_in_directory(
    experiment_directory: Path,
) -> dict[str, pd.DataFrame]:
    loaded_sensors: dict[str, pd.DataFrame] = {}
    for csv_file in sorted(experiment_directory.rglob("*.csv")):
        if csv_file.stat().st_size == 0:
            continue
        sensor_schema = detect_schema(csv_file.stem)
        if sensor_schema is None or sensor_schema.name in SENSOR_NAMES_TO_SKIP:
            continue
        sensor_data = load_single_sensor_csv(csv_file, schema=sensor_schema)
        sensor_data = sensor_data.drop(columns=["time"], errors="ignore")
        loaded_sensors[sensor_schema.name] = sensor_data
    return loaded_sensors


def _flatten_multiindex_columns(sensor_data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(sensor_data.columns, pd.MultiIndex):
        return sensor_data
    sensor_data.columns = [
        f"{sensor_name}_{measurement}"
        for sensor_name, measurement in sensor_data.columns
    ]
    return sensor_data


def load_all_experiment_sensors(
    raw_data_directory: str | Path = SRC / "data" / "raw",
    resample_rule: str = "100ms",
) -> pd.DataFrame:
    raw_data_directory = Path(raw_data_directory)
    if not raw_data_directory.is_dir():
        raise ValueError(f"raw_data_directory not found: {raw_data_directory}")

    experiment_dataframes: list[pd.DataFrame] = []

    for experiment_directory in sorted(raw_data_directory.iterdir()):
        if not experiment_directory.is_dir():
            continue

        detected_label = _detect_label_from_directory_name(experiment_directory.name)
        loaded_sensors = _load_sensor_csvs_in_directory(experiment_directory)
        if not loaded_sensors:
            continue

        merged_sensor_data = synchronise_sensor_frames(
            loaded_sensors,
            resample_rule=resample_rule,
            time_column="seconds_elapsed",
        )
        merged_sensor_data = _flatten_multiindex_columns(merged_sensor_data)

        if detected_label is not None:
            merged_sensor_data["label"] = detected_label.value

        experiment_dataframes.append(merged_sensor_data)

    if not experiment_dataframes:
        return pd.DataFrame()

    combined_experiments = pd.concat(experiment_dataframes)
    return combined_experiments.sort_index()


def compute_sensor_summary(sensor_data: pd.DataFrame) -> dict:
    numeric_data = sensor_data.select_dtypes(include=[np.number])
    num_rows, num_cols = len(sensor_data), len(sensor_data.columns)
    missing_counts = sensor_data.isnull().sum()
    total_cells = num_rows * num_cols

    return {
        "shape": sensor_data.shape,
        "columns": list(sensor_data.columns),
        "missing": {
            "total": int(missing_counts.sum()),
            "pct": round(float(missing_counts.sum() / total_cells * 100), 2),
            "per_column": (missing_counts / num_rows * 100).round(2).to_dict(),
        },
        "stats": numeric_data.describe().to_dict() if not numeric_data.empty else {},
    }


def detect_data_quality_issues(sensor_data: pd.DataFrame) -> dict:
    quality_report: dict = {"outliers": {}, "constant_columns": []}
    for column in sensor_data.select_dtypes(include=[np.number]):
        column_data = sensor_data[column].dropna()
        if column_data.nunique() == 1:
            quality_report["constant_columns"].append(column)
            continue
        first_quartile, third_quartile = column_data.quantile(0.25), column_data.quantile(0.75)
        interquartile_range_val = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * interquartile_range_val
        upper_bound = third_quartile + 1.5 * interquartile_range_val
        outlier_count = int(((column_data < lower_bound) | (column_data > upper_bound)).sum())
        if outlier_count:
            quality_report["outliers"][column] = outlier_count
    return quality_report
