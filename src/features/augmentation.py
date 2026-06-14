"""Add sqrt(x²+y²+z²) magnitude channels for every 3-axis sensor.

Magnitude is rotation-invariant — it captures total movement intensity
regardless of device orientation.  Features derived from magnitude are
more directly comparable across differently-worn devices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sensors whose _x,_y,_z columns can be combined into a magnitude.
# HeartRate (1-channel bpm) and WatchOrientation (angles/quaternions) are excluded.
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


def _sensors_with_all_axes(sensor_data: pd.DataFrame) -> list[str]:
    """Return sensors that have _x, _y, _z columns in the data."""
    found = []
    for sensor in THREE_AXIS_SENSORS:
        if all(f"{sensor}_{axis}" in sensor_data.columns for axis in ("x", "y", "z")):
            found.append(sensor)
    return found


def add_magnitude_channels(sensor_data: pd.DataFrame) -> pd.DataFrame:
    """Add ``sqrt(x² + y² + z²)`` columns for every 3-axis sensor present.

    Returns a new DataFrame (the original is not mutated).
    """
    sensors = _sensors_with_all_axes(sensor_data)
    if not sensors:
        return sensor_data.copy()

    result = sensor_data.copy()
    for sensor in sensors:
        x = result[f"{sensor}_x"].values
        y = result[f"{sensor}_y"].values
        z = result[f"{sensor}_z"].values
        result[f"{sensor}_magnitude"] = np.sqrt(x * x + y * y + z * z)

    return result
