"""Data augmentation — add derived channels before feature extraction.

Currently provides orientation-robust magnitude channels for 3-axis
sensors, making features less sensitive to device rotation and more
directly reflective of overall movement intensity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 3-axis sensors whose _x,_y,_z columns we can combine into a magnitude.
# Excludes HeartRate (1-channel bpm) and WatchOrientation (angles/quaternions).
THREE_AXIS_SENSORS: tuple[str, ...] = (
    "Accelerometer",
    "AccelerometerUncalibrated",
    "Gyroscope",
    "GyroscopeUncalibrated",
    "TotalAcceleration",
    "WatchAccelerometer",
    "WatchGravity",
    "WatchGyroscope",
    "WatchTotalAcceleration",
)


def _collect_existing_axis_sensors(
    sensor_data: pd.DataFrame,
    candidates: tuple[str, ...] = THREE_AXIS_SENSORS,
) -> list[str]:
    """Return sensors from *candidates* that have all three axis columns present."""
    existing: list[str] = []
    for sensor in candidates:
        if all(f"{sensor}_{axis}" in sensor_data.columns for axis in ("x", "y", "z")):
            existing.append(sensor)
    return existing


def add_magnitude_channels(
    sensor_data: pd.DataFrame,
    sensors: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Add ``sqrt(x² + y² + z²)`` magnitude columns for each 3-axis sensor.

    Parameters
    ----------
    sensor_data:
        Merged sensor DataFrame with ``{sensor}_{axis}`` column naming
        (e.g. ``Accelerometer_x``, ``Gyroscope_y``).
    sensors:
        Subset of 3-axis sensors to augment.  ``None`` → all known
        3-axis sensors that have columns in the data.

    Returns
    -------
    A new DataFrame with additional ``{sensor}_magnitude`` columns.
    The original data is not mutated.

    Notes
    -----
    Magnitude channels are rotation-invariant — they capture the total
    signal intensity regardless of device orientation.  This is directly
    relevant for distinguishing activity levels (no/soft/hard movement).
    """
    if sensors is None:
        sensors_to_process = _collect_existing_axis_sensors(sensor_data)
    else:
        sensors_to_process = [s for s in sensors if s in _collect_existing_axis_sensors(sensor_data)]

    if not sensors_to_process:
        return sensor_data.copy()

    result = sensor_data.copy()
    for sensor in sensors_to_process:
        x = result[f"{sensor}_x"].values
        y = result[f"{sensor}_y"].values
        z = result[f"{sensor}_z"].values
        magnitude = np.sqrt(x * x + y * y + z * z)
        result[f"{sensor}_magnitude"] = magnitude

    return result
