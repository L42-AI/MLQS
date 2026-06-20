"""Pocket-vs-wrist cross-sensor features per window.

For each (pocket, wrist) pair with magnitude columns, computes:
  mag_diff_mean   — mean(wrist) − mean(pocket)
  mag_diff_std    — std(wrist)  − std(pocket)
  mag_ratio       — mean(wrist) / mean(pocket)
  mag_corr        — Pearson correlation between the two magnitude series

High correlation = synchronised movement; low = decoupled body parts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CROSS_SENSOR_PAIRS: tuple[tuple[str, str], ...] = (
    ("Accelerometer", "WatchAccelerometer"),
    ("Gyroscope", "WatchGyroscope"),
    ("TotalAcceleration", "WatchTotalAcceleration"),
)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation.  Returns 0 when either signal is constant."""
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    denom = np.sqrt(np.sum(x_c**2) * np.sum(y_c**2))
    return float(np.sum(x_c * y_c) / denom) if denom > 1e-12 else 0.0


def compute_cross_sensor_features(
    window: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...] = DEFAULT_CROSS_SENSOR_PAIRS,
) -> dict[str, float]:
    """Per-window cross-sensor features (kept for backward compatibility)."""
    features: dict[str, float] = {}

    for pocket_name, wrist_name in pairs:
        pocket_col = f"{pocket_name}_magnitude"
        wrist_col = f"{wrist_name}_magnitude"

        if pocket_col not in window.columns or wrist_col not in window.columns:
            continue

        # Only use timestamps where both signals are valid
        valid = window[pocket_col].notna() & window[wrist_col].notna()
        if valid.sum() < 2:
            continue

        p = window[pocket_col][valid].to_numpy(dtype=float)
        w = window[wrist_col][valid].to_numpy(dtype=float)

        prefix = f"{pocket_name}_vs_{wrist_name}__"

        p_mean = float(np.mean(p))
        w_mean = float(np.mean(w))
        features[prefix + "mag_diff_mean"] = w_mean - p_mean
        features[prefix + "mag_diff_std"] = float(np.std(w, ddof=1)) - float(
            np.std(p, ddof=1)
        )
        features[prefix + "mag_ratio"] = (
            w_mean / p_mean if abs(p_mean) > 1e-12 else 0.0
        )
        features[prefix + "mag_corr"] = _correlation(p, w)

    return features


def compute_batch_cross_sensor_features(
    arrays: dict[str, np.ndarray],
    start_indices: np.ndarray,
    win_len: int,
    pairs: tuple[tuple[str, str], ...] = DEFAULT_CROSS_SENSOR_PAIRS,
) -> dict[str, np.ndarray]:
    """Compute cross-sensor features for ALL windows at once.

    Uses pre-extracted numpy arrays (from the ``arrays`` dict built by
    ``extract_features_from_windows``) to eliminate ~44 K per-window
    ``iloc`` slices.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Column-name → 1-D signal array (same dict used by ``_batch_extract_and_compute``).
    start_indices : np.ndarray
        1-D array of window-start sample indices, shape ``(num_windows,)``.
    win_len : int
        Anchor window length in samples (constant for all windows).
    pairs : tuple[tuple[str, str], …]
        (pocket_name, wrist_name) pairs — see ``DEFAULT_CROSS_SENSOR_PAIRS``.

    Returns
    -------
    dict[str, np.ndarray]
        Feature-name → array of shape ``(num_windows,)``.
        Windows with fewer than 2 jointly-valid samples receive ``NaN``.
    """
    n_windows = len(start_indices)
    if n_windows == 0:
        return {}

    # Pre-compute window indices once (reused for every sensor pair).
    idx = start_indices[:, None] + np.arange(win_len)  # (N, win_len)

    out: dict[str, np.ndarray] = {}

    for pocket_name, wrist_name in pairs:
        pocket_col = f"{pocket_name}_magnitude"
        wrist_col = f"{wrist_name}_magnitude"

        if pocket_col not in arrays or wrist_col not in arrays:
            continue

        # ── Extract all windows at once (no per-window iloc) ──────────────
        p = arrays[pocket_col][idx]  # (N, win_len)
        w = arrays[wrist_col][idx]   # (N, win_len)

        # Validity mask: True where BOTH are finite
        valid = ~(np.isnan(p) | np.isnan(w))    # (N, win_len)
        n_valid = np.sum(valid, axis=1)           # (N,)

        # Windows with < 2 valid pairs → NaN
        too_short = n_valid < 2

        # Zero out invalid positions for safe aggregation
        p_clean = np.where(valid, p, 0.0)        # (N, win_len)
        w_clean = np.where(valid, w, 0.0)

        # ── Means ────────────────────────────────────────────────────────
        denom = np.maximum(n_valid.astype(np.float64), 1.0)
        p_mean = np.sum(p_clean, axis=1) / denom
        w_mean = np.sum(w_clean, axis=1) / denom

        prefix = f"{pocket_name}_vs_{wrist_name}__"

        diff_mean = w_mean - p_mean
        ratio = np.where(np.abs(p_mean) > 1e-12, w_mean / p_mean, 0.0)

        # ── Sample std (ddof=1) via E[X²] − E[X]² ────────────────────────
        sum_sq_p = np.sum(p_clean ** 2, axis=1)
        sum_sq_w = np.sum(w_clean ** 2, axis=1)

        var_p = np.maximum(0.0, sum_sq_p / denom - p_mean ** 2)
        var_w = np.maximum(0.0, sum_sq_w / denom - w_mean ** 2)

        # ddof=1 correction:  s² = σ²_pop * n / (n − 1)
        ddof_factor = np.where(n_valid > 1, n_valid / (n_valid - 1).astype(np.float64), 1.0)
        var_p = var_p * ddof_factor
        var_w = var_w * ddof_factor
        diff_std = np.sqrt(var_w) - np.sqrt(var_p)

        # ── Pearson correlation ──────────────────────────────────────────
        sum_xy = np.sum(p_clean * w_clean, axis=1)
        cov = (sum_xy / denom - p_mean * w_mean) * ddof_factor
        denom_corr = np.sqrt(var_p * var_w)
        corr = np.where(denom_corr > 1e-12, cov / denom_corr, 0.0)

        # ── Apply NaN for insufficient-data windows ──────────────────────
        diff_mean[too_short] = np.nan
        diff_std[too_short] = np.nan
        ratio[too_short] = np.nan
        corr[too_short] = np.nan

        out[prefix + "mag_diff_mean"] = diff_mean
        out[prefix + "mag_diff_std"] = diff_std
        out[prefix + "mag_ratio"] = ratio
        out[prefix + "mag_corr"] = corr

    return out
