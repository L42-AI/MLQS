"""Feature engineering orchestrator — window → extract → merge.

Takes the merged multi-sensor DataFrame, windows it, and extracts
all registered features from each sensor column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config
from features import frequency_domain, statistical, time_domain
from features.windowing import create_sliding_windows
from utils.convert import resample_rule_to_frequency_hz


def _extract_features_for_column(
    sensor_readings: np.ndarray,
    column_prefix: str,
    feature_config,
    sample_rate_hz: float,
) -> dict[str, float]:
    extracted: dict[str, float] = {}

    if feature_config.time_domain:
        for feature_name, extractor_fn in time_domain.FEATURE_REGISTRY.items():
            extracted[column_prefix + feature_name] = extractor_fn(sensor_readings)

    if feature_config.frequency_domain:
        for feature_name, extractor_fn in frequency_domain.FEATURE_REGISTRY.items():
            extracted[column_prefix + feature_name] = extractor_fn(
                sensor_readings, fs=sample_rate_hz
            )
        for band_name, band_value in frequency_domain.compute_band_energy_ratios(
            sensor_readings, fs=sample_rate_hz
        ).items():
            extracted[column_prefix + band_name] = band_value

    if feature_config.statistical:
        for feature_name, extractor_fn in statistical.FEATURE_REGISTRY.items():
            extracted[column_prefix + feature_name] = extractor_fn(sensor_readings)

    return extracted


def extract_features_from_windows(
    merged_sensor_data: pd.DataFrame,
    sensor_columns: list[str],
    experiment_config: Config,
) -> pd.DataFrame:
    feature_config = experiment_config.features
    sample_rate_hz = resample_rule_to_frequency_hz(
        experiment_config.preprocessing.resample_rule
    )

    windows = create_sliding_windows(
        merged_sensor_data,
        window_size_seconds=feature_config.window_size,
        sampling_rate_hz=sample_rate_hz,
        overlap_fraction=feature_config.window_overlap,
        time_column=None,
    )
    if windows.empty:
        return pd.DataFrame()

    window_ids = windows.index.get_level_values("window").unique()

    feature_rows: list[dict[str, float]] = []
    window_labels: list[str] = []
    window_experiment_ids: list[int] = []

    for window_id in window_ids:
        window_data = windows.loc[window_id]
        feature_row: dict[str, float] = {}

        for sensor_column in sensor_columns:
            if sensor_column not in window_data.columns:
                continue

            sensor_readings = window_data[sensor_column].dropna().values.astype(float)
            if len(sensor_readings) < 2:
                continue

            feature_row.update(
                _extract_features_for_column(
                    sensor_readings,
                    f"{sensor_column}__",
                    feature_config,
                    sample_rate_hz,
                )
            )

        feature_rows.append(feature_row)

        if "label" in window_data.columns:
            window_labels.append(str(window_data["label"].iloc[0]))

        if "experiment_id" in window_data.columns:
            window_experiment_ids.append(int(window_data["experiment_id"].iloc[0]))

    if not feature_rows:
        return pd.DataFrame()

    extracted_features = pd.DataFrame(feature_rows)
    if window_labels:
        extracted_features["label"] = window_labels
    if window_experiment_ids:
        extracted_features["experiment_id"] = window_experiment_ids
    return extracted_features
