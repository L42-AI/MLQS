"""Feature engineering — window the sensor DataFrame and extract features.

Supports **per-sensor context windows** so that slowly-varying signals
(e.g. heart rate) use a larger extraction window than fast motion sensors,
while keeping a single anchor grid so all windows produce aligned vectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import SensorWindowConfig
from features import frequency_domain, statistical, time_domain
from features.cross_sensor import compute_batch_cross_sensor_features


# ── Context window geometry ─────────────────────────────────────────────────


def _vectorised_bounds(
    num_windows: int,
    anchor_size: float,
    overlap: float,
    sample_rate: float,
    context_size: float,
    total_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (starts, ends) arrays for all windows at a given context size.

    Vectorised replacement for the per-window ``context_bounds`` loop.
    When *context_size* == *anchor_size* the output is trivial (anchor
    windows).  For larger / smaller contexts each window is centred on
    the corresponding anchor window and clamped to ``[0, total_samples)``.
    """
    win_len = int(round(anchor_size * sample_rate))
    slide = int(round(win_len * (1.0 - overlap)))
    starts = np.arange(num_windows, dtype=np.intp) * slide

    if context_size == anchor_size:
        return starts, starts + win_len

    centre = starts + win_len // 2
    ctx_len = int(round(context_size * sample_rate))
    ctx_starts = centre - ctx_len // 2
    ctx_starts = np.clip(ctx_starts, 0, total_samples - ctx_len)
    ends = ctx_starts + ctx_len
    return ctx_starts, np.minimum(ends, total_samples)


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
    individual functions into a single pass via ``compute_all_time_domain_features``.
    Kept for windows that require per-window NaN handling.
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
            frequency_bands=feature_config.frequency_bands,
            rolloff_fraction=feature_config.rolloff_fraction,
        ).items():
            extracted[prefix + name] = val

    if feature_config.statistical:
        for name, fn in statistical.FEATURE_REGISTRY.items():
            extracted[prefix + name] = fn(readings)

    return extracted


# ── Column-major batch helpers ──────────────────────────────────────────────


def _extract_windows_1d(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Extract windows from a 1-D array into a 2-D ``(num_windows, win_len)`` array.

    All windows must have the same length (``ends - starts`` is constant).
    Uses ``np.lib.stride_tricks.sliding_window_view`` when windows are
    contiguous and regularly-spaced (the common case), falling back to
    index-trickery otherwise.
    """
    win_len = int(ends[0] - starts[0])
    # Fast path: contiguous regularly-spaced windows
    slide = int(starts[1] - starts[0]) if len(starts) > 1 else win_len
    if np.all(starts[1:] - starts[:-1] == slide) and ends[-1] <= len(arr):
        return np.lib.stride_tricks.sliding_window_view(arr, win_len)[::slide][:len(starts)]
    # General path
    n = len(starts)
    idx = starts[:, None] + np.arange(win_len)
    return arr[idx]


def _batch_extract_and_compute(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    prefix: str,
    feature_config,
    sample_rate: float,
) -> dict[str, np.ndarray]:
    """Extract all windows from *arr* and compute every enabled feature.

    Time-domain and statistical features are computed in fully vectorised
    batch form (one call per column).  Frequency features use the original
    per-window Welch PSD to stay consistent with the established feature
    definitions (Welch smooths the spectrum differently from a raw FFT).

    Returns a dict ``{feature_name: ndarray of shape (num_windows,)}``.
    """
    windows = _extract_windows_1d(arr, starts, ends)
    out: dict[str, np.ndarray] = {}
    n_windows = windows.shape[0]

    if feature_config.time_domain:
        for name, vals in time_domain.compute_batch_time_domain_features(windows).items():
            out[prefix + name] = vals

    if feature_config.frequency_domain:
        # Vectorised batch — one np.fft.rfft call + numpy ops for all windows.
        for name, vals in frequency_domain.compute_batch_frequency_features(
            windows, fs=sample_rate,
            frequency_bands=feature_config.frequency_bands,
            rolloff_fraction=feature_config.rolloff_fraction,
        ).items():
            out[prefix + name] = vals

    if feature_config.statistical:
        for name, vals in statistical.compute_batch_statistical_features(windows).items():
            out[prefix + name] = vals

    return out


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
    3. Pre-compute (start, end) bounds vectorised per context size.
    4. For each column, extract *all* windows at once and compute features
       in batch using vectorised ``compute_batch_*`` functions.
    5. Columns with NaN values fall back to per-window extraction.
    6. Attach cross-sensor features, labels, and ``experiment_id``.

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

    if not col_context:
        return pd.DataFrame()

    # ── 3. Vectorised bounds per context size ────────────────────────────
    size_bounds: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for size in set(col_context.values()):
        size_bounds[size] = _vectorised_bounds(
            num_windows, anchor_size, overlap, sample_rate, size, total_samples,
        )

    # ── 4. Convert sensor columns to arrays + check for NaN ──────────────
    arrays: dict[str, np.ndarray] = {}
    col_has_nan: dict[str, bool] = {}
    for col in col_context:
        arr = sensor_data[col].to_numpy(dtype=float)
        arrays[col] = arr
        col_has_nan[col] = bool(np.isnan(arr).any())

    columns: tuple[str, ...] = tuple(col_context.keys())

    # ── 5. Column-major batch feature extraction ─────────────────────────
    desc = f"Block {block_info}  |  {len(columns)} cols" if block_info else f"{len(columns)} cols"
    feature_data: dict[str, np.ndarray] = {}

    for idx, col in enumerate(tqdm(columns, desc=desc)):
        starts, ends = size_bounds[col_context[col]]

        if col_has_nan[col]:
            # Fallback: per-window with NaN masking (rare after imputation)
            rows: list[dict[str, float]] = []
            for s, e in zip(starts, ends):
                segment = arrays[col][s:e]
                segment = segment[~np.isnan(segment)]
                if len(segment) < 2:
                    rows.append({})
                    continue
                rows.append(_extract_features(segment, f"{col}__", feature_config, sample_rate))

            # Merge row-oriented dicts into column-oriented arrays
            if rows:
                keys = rows[0].keys()
                for key in keys:
                    vals = np.array([r.get(key, np.nan) for r in rows])
                    feature_data[key] = vals
        else:
            # Fast batch path
            result = _batch_extract_and_compute(
                arrays[col], starts, ends, f"{col}__", feature_config, sample_rate,
            )
            feature_data.update(result)

    # ── 6. Build DataFrame ───────────────────────────────────────────────
    if not feature_data:
        return pd.DataFrame()

    features = pd.DataFrame(feature_data)

    # ── 7. Cross-sensor features (batch — no per-window iloc) ────────────
    if feature_config.cross_sensor_features:
        cross_feats = compute_batch_cross_sensor_features(
            arrays, start_indices, win_len,
        )
        if cross_feats:
            cross_df = pd.DataFrame(cross_feats)
            features = pd.concat([features, cross_df], axis=1)

    # ── 8. Labels, participant, and experiment IDs ────────────────────────
    for column_name in ["label", "participant", "experiment_id"]:
        if column_name in sensor_data.columns:
            features[column_name] = [
                str(v) if pd.notna(v) else None
                for v in sensor_data[column_name].iloc[start_indices]
            ]

    return features
