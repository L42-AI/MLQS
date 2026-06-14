"""Feature engineering orchestrator — window → extract → merge.

Windows the merged sensor DataFrame and extracts features from every
sensor column.  Supports **per-sensor context windows** so that slowly-
varying signals (e.g. heart rate) can use a larger extraction window
than fast motion sensors, while keeping a single anchor grid so all
windows produce aligned feature vectors.
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


def context_bounds(
    window_id: int,
    anchor_size: float,
    overlap: float,
    sample_rate: float,
    context_size: float,
    total_samples: int,
) -> tuple[int, int]:
    """Return (start, end) of a context window centred on the anchor.

    When *context_size* equals *anchor_size* the context is the anchor
    window itself.  Result is clamped to the data bounds.
    """
    win_len = int(round(anchor_size * sample_rate))
    slide = int(round(win_len * (1.0 - overlap)))
    start = window_id * slide

    if context_size == anchor_size:
        return (start, start + win_len)

    centre = start + win_len // 2
    ctx_len = int(round(context_size * sample_rate))
    ctx_start = centre - ctx_len // 2
    ctx_end = ctx_start + ctx_len

    if ctx_start < 0:
        shift = -ctx_start
        ctx_start += shift
        ctx_end += shift

    if ctx_end > total_samples:
        shift = total_samples - ctx_end
        ctx_start += shift
        ctx_end += shift

    return (max(ctx_start, 0), min(ctx_end, total_samples))


def _sensor_name(column: str) -> str:
    """Extract ``HeartRate`` from ``HeartRate_bpm``."""
    return column.rsplit("_", 1)[0]


def _sensor_window_sizes(
    sensor_name: str,
    feature_config,
) -> tuple[float, float | None]:
    """Return (base_window, freq_window) for a sensor.

    Checks for a per-sensor override in ``sensor_windows``, otherwise
    falls back to the global defaults.
    """
    override: SensorWindowConfig | None = feature_config.sensor_windows.get(sensor_name)
    if override is not None:
        return override.base_window_seconds, override.freq_window_seconds
    return feature_config.window_size, feature_config.frequency_window_size


def _features_for_column(
    readings: np.ndarray,
    prefix: str,
    feature_config,
    sample_rate_hz: float,
) -> dict[str, float]:
    """Run all enabled feature registries on *readings*."""
    extracted: dict[str, float] = {}

    if feature_config.time_domain:
        for name, fn in time_domain.FEATURE_REGISTRY.items():
            extracted[prefix + name] = fn(readings)

    if feature_config.frequency_domain:
        for name, fn in frequency_domain.FEATURE_REGISTRY.items():
            extracted[prefix + name] = fn(readings, fs=sample_rate_hz)
        for band, val in frequency_domain.compute_band_energy_ratios(
            readings, fs=sample_rate_hz
        ).items():
            extracted[prefix + band] = val

    if feature_config.statistical:
        for name, fn in statistical.FEATURE_REGISTRY.items():
            extracted[prefix + name] = fn(readings)

    return extracted


def _readings_for_column(
    sensor_data: pd.DataFrame,
    column: str,
    window_id: int,
    anchor_size: float,
    overlap: float,
    sample_rate: float,
    context_size: float,
    total_samples: int,
) -> np.ndarray:
    """Pull the sensor values for *column* from a context window.

    The context is centred on the anchor window position, giving slower
    signals (e.g. heart rate) more data without changing the grid.
    """
    start, end = context_bounds(
        window_id, anchor_size, overlap, sample_rate, context_size, total_samples,
    )
    segment = sensor_data.iloc[start:end]
    if column not in segment.columns:
        return np.array([], dtype=float)
    return segment[column].dropna().to_numpy(dtype=float)


def _window_labels_and_ids(window_data: pd.DataFrame):
    """Return (label, experiment_id) from the first row of *window_data*."""
    label = (
        str(window_data["label"].iloc[0])
        if "label" in window_data.columns
        else None
    )
    exp_id = (
        int(window_data["experiment_id"].iloc[0])
        if "experiment_id" in window_data.columns
        else None
    )
    return label, exp_id


def extract_features_from_windows(
    sensor_data: pd.DataFrame,
    sensor_columns: list[str],
    config: Config,
) -> pd.DataFrame:
    feature_config = config.features
    sample_rate = resample_rule_to_frequency_hz(
        config.preprocessing.resample_rule
    )
    total_samples = len(sensor_data)

    windows = create_sliding_windows(
        sensor_data,
        window_size_seconds=feature_config.window_size,
        sampling_rate_hz=sample_rate,
        overlap_fraction=feature_config.window_overlap,
        time_column=None,
    )
    if windows.empty:
        return pd.DataFrame()

    # Pre-compute effective (base, freq) sizes for every sensor column
    sensor_sizes: dict[str, tuple[float, float | None]] = {}
    for col in sensor_columns:
        name = _sensor_name(col)
        sensor_sizes[col] = _sensor_window_sizes(name, feature_config)

    # ── Per-window extraction ────────────────────────────────────────
    window_ids = windows.index.get_level_values("window").unique()
    rows: list[dict[str, float]] = []
    labels: list[str] = []
    exp_ids: list[int] = []

    for window_id in window_ids:
        window_slice = windows.loc[window_id]
        row: dict[str, float] = {}

        # Per-column features (each with its own context size)
        for col in sensor_columns:
            if col not in sensor_data.columns:
                continue

            base_size, freq_size = sensor_sizes[col]
            effective_size = freq_size if freq_size is not None else base_size

            readings = _readings_for_column(
                sensor_data, col, window_id,
                anchor_size=feature_config.window_size,
                overlap=feature_config.window_overlap,
                sample_rate=sample_rate,
                context_size=effective_size,
                total_samples=total_samples,
            )
            if len(readings) < 2:
                continue

            row.update(
                _features_for_column(readings, f"{col}__", feature_config, sample_rate)
            )

        # Cross-sensor features (from the anchor window's magnitude columns)
        if feature_config.cross_sensor_features:
            row.update(compute_cross_sensor_features(window_slice))

        rows.append(row)

        label, exp_id = _window_labels_and_ids(window_slice)
        if label is not None:
            labels.append(label)
        if exp_id is not None:
            exp_ids.append(exp_id)

    if not rows:
        return pd.DataFrame()

    features = pd.DataFrame(rows)
    if labels:
        features["label"] = labels
    if exp_ids:
        features["experiment_id"] = exp_ids
    return features
