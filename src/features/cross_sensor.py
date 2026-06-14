"""Cross-sensor relationship features — pocket vs wrist comparisons.

Why these features
------------------
You have motion sensors in two locations (pocket and wrist) plus heart rate.
While per-sensor features capture each signal independently, explicit
cross-sensor relationships encode how *synchronised* or *decoupled* the
two body parts move under different music conditions — e.g. sitting-and-nodding
vs full-body dancing.

Features (computed per window)
------------------------------
For each ``(pocket_sensor, wrist_sensor)`` pair with available magnitude
columns:

  *mag_diff_mean* — mean(wrist) − mean(pocket)
    Positive → wrist moves more intensely than pocket.
  *mag_diff_std* — std(wrist) − std(pocket)
    Which location has more variable movement.
  *mag_ratio* — mean(wrist) / mean(pocket)
    Relative intensity (ratio, unitless).
  *mag_corr* — Pearson correlation between the two magnitude series
    High = synchronised movement; low/negative = independent movement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default pairs to compare.
# The convention is (pocket_sensor_name, wrist_sensor_name) and the
# code looks for ``{name}_magnitude`` columns in the window data.
DEFAULT_CROSS_SENSOR_PAIRS: tuple[tuple[str, str], ...] = (
    ("Accelerometer", "WatchAccelerometer"),
    ("Gyroscope", "WatchGyroscope"),
    ("TotalAcceleration", "WatchTotalAcceleration"),
)


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient, guarding against constant signals."""
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    numerator = float(np.sum(x_centered * y_centered))
    denom = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    return numerator / denom if denom > 1e-12 else 0.0


def compute_cross_sensor_features(
    window_data: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...] = DEFAULT_CROSS_SENSOR_PAIRS,
) -> dict[str, float]:
    """Compute cross-sensor relationship features for a single window.

    Parameters
    ----------
    window_data:
        A single window's worth of sensor data (the slice returned by
        ``windows.loc[window_id]``).  Must contain ``{sensor}_magnitude``
        columns for the sensors listed in *pairs*.
    pairs:
        Sequence of ``(pocket_sensor, wrist_sensor)`` tuples to compare.
        Defaults to :obj:`DEFAULT_CROSS_SENSOR_PAIRS`.

    Returns
    -------
    Dict keyed by ``{pocket}_vs_{wrist}__{feature}``, or empty if no
    magnitude columns are available for the requested pairs.
    """
    features: dict[str, float] = {}

    for pocket_sensor, wrist_sensor in pairs:
        pocket_col = f"{pocket_sensor}_magnitude"
        wrist_col = f"{wrist_sensor}_magnitude"

        if pocket_col not in window_data.columns or wrist_col not in window_data.columns:
            continue

        # Aligned values — only keep timestamps where both are valid
        pocket_series = window_data[pocket_col]
        wrist_series = window_data[wrist_col]
        common_mask = pocket_series.notna() & wrist_series.notna()
        if common_mask.sum() < 2:
            continue

        pocket_vals = pocket_series[common_mask].to_numpy(dtype=float)
        wrist_vals = wrist_series[common_mask].to_numpy(dtype=float)

        prefix = f"{pocket_sensor}_vs_{wrist_sensor}__"

        # ── Difference of means ────────────────────────────────────────
        pocket_mean = float(np.mean(pocket_vals))
        wrist_mean = float(np.mean(wrist_vals))
        features[prefix + "mag_diff_mean"] = wrist_mean - pocket_mean

        # ── Difference of standard deviations ──────────────────────────
        pocket_std = float(np.std(pocket_vals, ddof=1))
        wrist_std = float(np.std(wrist_vals, ddof=1))
        features[prefix + "mag_diff_std"] = wrist_std - pocket_std

        # ── Ratio of means ─────────────────────────────────────────────
        if abs(pocket_mean) > 1e-12:
            features[prefix + "mag_ratio"] = wrist_mean / pocket_mean
        else:
            features[prefix + "mag_ratio"] = 0.0

        # ── Correlation ────────────────────────────────────────────────
        features[prefix + "mag_corr"] = _pearson_correlation(pocket_vals, wrist_vals)

    return features
