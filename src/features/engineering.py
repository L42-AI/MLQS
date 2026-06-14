"""Feature engineering orchestrator — window → extract → merge.

Takes the merged multi-sensor DataFrame, windows it, and extracts
all registered features from each sensor column.

Supports **per-sensor context windows** so that slowly-varying signals
(e.g. heart rate, 30 s context) can use larger extraction windows than
fast motion sensors (e.g. gyroscope, 2 s), while keeping a single
anchor grid for temporal alignment across all sensors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config, SensorWindowConfig
from features import frequency_domain, statistical, time_domain
from features.cross_sensor import (
    DEFAULT_CROSS_SENSOR_PAIRS,
    compute_cross_sensor_features,
)
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


def _parse_sensor_name(column_name: str) -> str:
    """Extract sensor name from a flattened column like ``HeartRate_bpm``.

    The merged DataFrame uses ``{sensor_name}_{axis}`` naming (e.g.
    ``Accelerometer_x``, ``HeartRate_bpm``, ``WatchOrientation_qx``).
    """
    return column_name.rsplit("_", 1)[0]


def _resolve_window_config(
    sensor_name: str,
    feature_config,
) -> tuple[float, float | None]:
    """Return ``(base_window, freq_window)`` for *sensor_name*.

    Falls back to the global ``FeatureConfig.window_size`` /
    ``FeatureConfig.frequency_window_size`` when no per-sensor override
    exists in ``FeatureConfig.sensor_windows``.
    """
    override: SensorWindowConfig | None = feature_config.sensor_windows.get(
        sensor_name
    )
    if override is not None:
        return override.base_window_seconds, override.freq_window_seconds
    return feature_config.window_size, feature_config.frequency_window_size


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

    # ── Pre-compute effective window sizes per sensor column ──────────
    # Maps column name → (base_window, freq_window) via sensor name lookup.
    col_window_config: dict[str, tuple[float, float | None]] = {}
    for col in sensor_columns:
        sensor_name = _parse_sensor_name(col)
        col_window_config[col] = _resolve_window_config(sensor_name, feature_config)

    # ── Per-window, per-sensor feature extraction ─────────────────────
    feature_rows: list[dict[str, float]] = []
    window_labels: list[str] = []
    window_experiment_ids: list[int] = []

    for window_id in window_ids:
        window_data = windows.loc[window_id]
        feature_row: dict[str, float] = {}

        for sensor_column in sensor_columns:
            if sensor_column not in merged_sensor_data.columns:
                continue

            base_win, freq_win = col_window_config[sensor_column]
            effective_win = freq_win if freq_win is not None else base_win

            # Sensor-specific context window (centred on anchor position)
            ctx_start, ctx_end = compute_context_window_indices(
                window_id,
                window_size_seconds=feature_config.window_size,
                overlap_fraction=feature_config.window_overlap,
                sampling_rate_hz=sample_rate_hz,
                freq_window_size_seconds=effective_win,
                total_samples=total_samples,
            )
            context_data = merged_sensor_data.iloc[ctx_start:ctx_end]

            sensor_readings = (
                context_data[sensor_column].dropna().values.astype(float)
                if sensor_column in context_data.columns
                else np.array([], dtype=float)
            )
            if len(sensor_readings) < 2:
                continue

            # All feature types (time, freq, statistical) are computed
            # from the same sensor-specific context.
            feature_row.update(
                _extract_features_for_column(
                    sensor_readings,
                    f"{sensor_column}__",
                    feature_config,
                    sample_rate_hz,
                )
            )

        # ── Cross-sensor relationship features ─────────────────────────
        # Uses magnitude columns (added by augmentation) within the anchor
        # window to compare pocket vs wrist movement synchronisation.
        if feature_config.cross_sensor_features:
            feature_row.update(
                compute_cross_sensor_features(
                    window_data, pairs=DEFAULT_CROSS_SENSOR_PAIRS
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
