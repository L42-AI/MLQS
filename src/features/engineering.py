"""Feature engineering — window the sensor DataFrame and extract features.

Supports **per-sensor context windows** so that slowly-varying signals
(e.g. heart rate) use a larger extraction window than fast motion sensors,
while keeping a single anchor grid so all windows produce aligned vectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from config import SensorWindowConfig
from features import frequency_domain, statistical, time_domain
from features.cross_sensor import compute_cross_sensor_features


# ── Context window geometry ─────────────────────────────────────────────────


@njit
def context_bounds(
    window_id: int,
    anchor_size: float,
    overlap: float,
    sample_rate: float,
    context_size: float,
    total_samples: int,
) -> tuple[int, int]:
    """Return (start, end) of a context window centred on the anchor window.

    When *context_size* equals *anchor_size* the context is the anchor window
    itself.  The result is clamped to ``[0, total_samples)``.

    Compiled with ``@njit`` since this is called O(unique_sizes × windows)
    times — every nanosecond saved compounds across thousands of calls.
    """
    win_len = int(round(anchor_size * sample_rate))
    slide = int(round(win_len * (1.0 - overlap)))
    start = window_id * slide

    if context_size == anchor_size:
        return (start, start + win_len)

    centre = start + win_len // 2
    ctx_len = int(round(context_size * sample_rate))
    ctx_start = centre - ctx_len // 2
    if ctx_start < 0:
        ctx_start = 0
    if ctx_start + ctx_len > total_samples:
        ctx_start = max(0, total_samples - ctx_len)

    return (ctx_start, min(ctx_start + ctx_len, total_samples))


# ── Helpers ─────────────────────────────────────────────────────────────────


def _effective_window_seconds(sensor_name: str, feature_config) -> float:
    """Return the context-window size (seconds) for *sensor_name*.

    Resolution priority (first match wins)::

        1. Per-sensor ``freq_window_seconds`` override
        2. Per-sensor ``base_window_seconds`` override
        3. Global ``frequency_window_size``
        4. Global ``window_size`` (fallback)
    """
    override: SensorWindowConfig | None = feature_config.sensor_windows.get(sensor_name)
    if override is not None:
        return override.freq_window_seconds or override.base_window_seconds
    return feature_config.frequency_window_size or feature_config.window_size


def _extract_features(
    readings: np.ndarray,
    prefix: str,
    feature_config,
    sample_rate_hz: float,
) -> dict[str, float]:
    """Run all enabled feature extractors on *readings*.

    Frequency-domain features share a single PSD computation via
    ``compute_all_frequency_features``.  Time-domain features fuse all 12
    individual functions into a single pass via ``compute_all_time_domain_features``
    to share intermediate results (``np.mean``, ``np.diff``, etc.).
    """
    extracted: dict[str, float] = {}

    if feature_config.time_domain:
        extracted.update(
            (prefix + k, v)
            for k, v in time_domain.compute_all_time_domain_features(readings).items()
        )

    if feature_config.frequency_domain:
        for name, val in frequency_domain.compute_all_frequency_features(
            readings, fs=sample_rate_hz,
        ).items():
            extracted[prefix + name] = val

    if feature_config.statistical:
        for name, fn in statistical.FEATURE_REGISTRY.items():
            extracted[prefix + name] = fn(readings)

    return extracted


# ── Public API ──────────────────────────────────────────────────────────────


def extract_features_from_windows(
    sensor_data: pd.DataFrame,
    sensor_columns: list[str],
    feature_config,
    sample_rate: float,
    block_info: str = "",
) -> pd.DataFrame:
    """Extract features from sliding windows over *sensor_data*.

    Parameters
    ----------
    block_info : str
        When called from a block-split pipeline, pass ``"3/10"`` so the
        progress bar shows which block is being processed (e.g. ``"Block 3/10
        | 45 windows"``).  Empty string omits the block prefix.

    Steps
    -----
    1. Compute anchor-window positions from window size / overlap.
    2. Group columns by their effective context-window size.
    3. Pre-compute (start, end) bounds for every (context_size, window) pair.
    4. Convert each sensor column to a NumPy array.
    5. Extract features per window (sequentially).
    6. Attach cross-sensor features.
    7. Attach ``label`` and ``experiment_id`` from the anchor window.

    Returns
    -------
    pd.DataFrame
        One row per window with ``label`` and ``experiment_id`` columns
        appended when present in the input.
    """
    total_samples = len(sensor_data)

    # ── 1. Window geometry ───────────────────────────────────────────────
    anchor_size = feature_config.window_size
    overlap = feature_config.window_overlap
    win_len = int(round(anchor_size * sample_rate))
    slide = int(round(win_len * (1.0 - overlap)))
    start_indices = np.arange(0, total_samples - win_len + 1, slide)
    num_windows = len(start_indices)

    if num_windows == 0:
        return pd.DataFrame()

    # ── 2. Map each column → its context-window size (seconds) ───────────
    col_context: dict[str, float] = {}
    for col in sensor_columns:
        if col not in sensor_data.columns:
            continue
        sensor_name = col.rsplit("_", 1)[0]
        col_context[col] = _effective_window_seconds(sensor_name, feature_config)

    # ── 3. Pre-compute context bounds for every (size, window) pair ──────
    bounds: dict[tuple[float, int], tuple[int, int]] = {}
    for size in set(col_context.values()):
        for w_id in range(num_windows):
            bounds[(size, w_id)] = context_bounds(
                w_id, anchor_size, overlap, sample_rate, size, total_samples,
            )

    # ── 4. Convert every sensor column to a NumPy array ──────────────────
    arrays: dict[str, np.ndarray] = {
        col: sensor_data[col].to_numpy(dtype=float)
        for col in col_context
    }
    columns: tuple[str, ...] = tuple(col_context.keys())

    # ── 5. Per-window feature extraction ─────────────────────────────────
    desc = f"Block {block_info}  |  {num_windows} windows" if block_info else f"{num_windows} windows"
    rows: list[dict[str, float]] = []
    for w_id in tqdm(range(num_windows), desc=desc, leave=False):
        row: dict[str, float] = {}
        for col in columns:
            s, e = bounds[(col_context[col], w_id)]
            readings = arrays[col][s:e]
            readings = readings[~np.isnan(readings)]
            if len(readings) < 2:
                continue
            row.update(_extract_features(readings, f"{col}__", feature_config, sample_rate))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    features = pd.DataFrame(rows)

    # ── 6. Cross-sensor features ─────────────────────────────────────────
    if feature_config.cross_sensor_features:
        cross_rows = [
            compute_cross_sensor_features(sensor_data.iloc[s:s + win_len])
            for s in start_indices
        ]
        if cross_rows:
            features = pd.concat([features, pd.DataFrame(cross_rows)], axis=1)

    # ── 7. Labels and experiment IDs ─────────────────────────────────────
    if "label" in sensor_data.columns:
        features["label"] = [
            str(v) if pd.notna(v) else None
            for v in sensor_data["label"].iloc[start_indices]
        ]
    if "experiment_id" in sensor_data.columns:
        features["experiment_id"] = [
            int(v) if pd.notna(v) else None
            for v in sensor_data["experiment_id"].iloc[start_indices]
        ]

    return features
