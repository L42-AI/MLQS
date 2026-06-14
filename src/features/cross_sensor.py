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
