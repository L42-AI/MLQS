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


def compute_context_window_indices(
    window_id: int,
    window_size_seconds: float,
    overlap_fraction: float,
    sampling_rate_hz: float,
    freq_window_size_seconds: float,
    total_samples: int,
) -> tuple[int, int]:
    """Compute freq-domain context window indices centered on a base window.

    Result is clamped to [0, total_samples).  When *freq_window_size_seconds*
    equals *window_size_seconds*, returns the base window bounds exactly.
    """
    win_len = int(round(window_size_seconds * sampling_rate_hz))
    slide = int(round(win_len * (1.0 - overlap_fraction)))
    start = window_id * slide

    if freq_window_size_seconds == window_size_seconds:
        return (start, start + win_len)

    center = start + win_len // 2
    freq_len = int(round(freq_window_size_seconds * sampling_rate_hz))
    ctx_start = center - freq_len // 2
    ctx_end = ctx_start + freq_len

    if ctx_start < 0:
        shift = -ctx_start
        ctx_start += shift
        ctx_end += shift

    if ctx_end > total_samples:
        shift = total_samples - ctx_end
        ctx_start += shift
        ctx_end += shift

    if ctx_start < 0:
        ctx_start = 0

    return (ctx_start, min(ctx_end, total_samples))


def _extract_features_for_column(
    sensor_readings: np.ndarray,
    column_prefix: str,
    feature_config,
    sample_rate_hz: float,
    freq_readings: np.ndarray | None = None,
) -> dict[str, float]:
    extracted: dict[str, float] = {}

    if feature_config.time_domain:
        for feature_name, extractor_fn in time_domain.FEATURE_REGISTRY.items():
            extracted[column_prefix + feature_name] = extractor_fn(sensor_readings)

    if feature_config.frequency_domain:
        freq_data = freq_readings if freq_readings is not None else sensor_readings
        for feature_name, extractor_fn in frequency_domain.FEATURE_REGISTRY.items():
            extracted[column_prefix + feature_name] = extractor_fn(
                freq_data, fs=sample_rate_hz
            )
        for band_name, band_value in frequency_domain.compute_band_energy_ratios(
            freq_data, fs=sample_rate_hz
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
    total_samples = len(merged_sensor_data)

    freq_window_size = feature_config.frequency_window_size
    use_multi_resolution = (
        freq_window_size is not None
        and freq_window_size != feature_config.window_size
    )

    feature_rows: list[dict[str, float]] = []
    window_labels: list[str] = []
    window_experiment_ids: list[int] = []

    for window_id in window_ids:
        window_data = windows.loc[window_id]
        feature_row: dict[str, float] = {}

        if use_multi_resolution:
            ctx_start, ctx_end = compute_context_window_indices(
                window_id,
                window_size_seconds=feature_config.window_size,
                overlap_fraction=feature_config.window_overlap,
                sampling_rate_hz=sample_rate_hz,
                freq_window_size_seconds=freq_window_size,
                total_samples=total_samples,
            )
            freq_context = merged_sensor_data.iloc[ctx_start:ctx_end]

        for sensor_column in sensor_columns:
            if sensor_column not in merged_sensor_data.columns:
                continue

            base_readings = (
                window_data[sensor_column].dropna().values.astype(float)
                if sensor_column in window_data.columns
                else np.array([], dtype=float)
            )
            if len(base_readings) < 2:
                continue

            freq_readings: np.ndarray | None = None
            if use_multi_resolution and sensor_column in freq_context.columns:
                freq_readings = (
                    freq_context[sensor_column].dropna().values.astype(float)
                )
                if len(freq_readings) < 2:
                    freq_readings = None

            feature_row.update(
                _extract_features_for_column(
                    base_readings,
                    f"{sensor_column}__",
                    feature_config,
                    sample_rate_hz,
                    freq_readings=freq_readings,
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
