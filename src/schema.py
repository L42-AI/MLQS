"""Sensor schemas — describes the actual columns in each phyphox CSV export.

Based on the real export structure:
  - Phone sensors (Accelerometer, Gyroscope, etc): time, seconds_elapsed, z, y, x
  - Watch sensors (same 3-axis pattern, different sampling rate)
  - WatchOrientation: time, seconds_elapsed, yaw, pitch, roll, qx, qy, qz, qw
  - HeartRate: time, seconds_elapsed, bpm
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SensorSchema:
    """Describes a single phyphox sensor CSV file."""

    name: str                     # e.g. "Accelerometer", "Gyroscope"
    columns: tuple[str, ...]      # actual column names in the CSV
    time_col: str = "seconds_elapsed"  # use seconds not nanosecond epoch
    sample_rate: float | None = None   # Hz, if known
    device: str = "phone"         # "phone" or "watch"

    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"Schema '{self.name}' missing columns: {missing}")


# ── Real phyphox sensor schemas ─────────────────────────────────────────────

THREE_AXIS_SENSOR_COLUMNS = ("time", "seconds_elapsed", "z", "y", "x")  # phyphox uses z,y,x order
PHONE_SAMPLE_RATE_HZ = 50.0   # ~50 Hz (20ms intervals)
WATCH_SAMPLE_RATE_HZ = 10.0   # ~10 Hz (100ms intervals)

ACCELEROMETER = SensorSchema("Accelerometer", THREE_AXIS_SENSOR_COLUMNS, sample_rate=PHONE_SAMPLE_RATE_HZ, device="phone")
GYROSCOPE = SensorSchema("Gyroscope", THREE_AXIS_SENSOR_COLUMNS, sample_rate=PHONE_SAMPLE_RATE_HZ, device="phone")
TOTAL_ACCELERATION = SensorSchema("TotalAcceleration", THREE_AXIS_SENSOR_COLUMNS, sample_rate=PHONE_SAMPLE_RATE_HZ, device="phone")
ACCELEROMETER_UNCALIBRATED = SensorSchema("AccelerometerUncalibrated", THREE_AXIS_SENSOR_COLUMNS, sample_rate=PHONE_SAMPLE_RATE_HZ, device="phone")
GYROSCOPE_UNCALIBRATED = SensorSchema("GyroscopeUncalibrated", THREE_AXIS_SENSOR_COLUMNS, sample_rate=PHONE_SAMPLE_RATE_HZ, device="phone")

WATCH_ACCELEROMETER = SensorSchema("WatchAccelerometer", THREE_AXIS_SENSOR_COLUMNS, sample_rate=WATCH_SAMPLE_RATE_HZ, device="watch")
WATCH_GYROSCOPE = SensorSchema("WatchGyroscope", THREE_AXIS_SENSOR_COLUMNS, sample_rate=WATCH_SAMPLE_RATE_HZ, device="watch")
WATCH_GRAVITY = SensorSchema("WatchGravity", THREE_AXIS_SENSOR_COLUMNS, sample_rate=WATCH_SAMPLE_RATE_HZ, device="watch")
WATCH_TOTAL_ACCELERATION = SensorSchema("WatchTotalAcceleration", THREE_AXIS_SENSOR_COLUMNS, sample_rate=WATCH_SAMPLE_RATE_HZ, device="watch")

WATCH_ORIENTATION = SensorSchema(
    "WatchOrientation",
    ("time", "seconds_elapsed", "yaw", "pitch", "roll", "qx", "qy", "qz", "qw"),
    sample_rate=WATCH_SAMPLE_RATE_HZ,
    device="watch",
)

HEART_RATE = SensorSchema(
    "HeartRate",
    ("time", "seconds_elapsed", "bpm"),
    sample_rate=10.0,
    device="phone",
)  # Note: HeartRate has negative seconds_elapsed (starts before main recording)

# ── Lookup by filename ──────────────────────────────────────────────────────

SENSOR_MAP: dict[str, SensorSchema] = {
    "Accelerometer": ACCELEROMETER,
    "Gyroscope": GYROSCOPE,
    "TotalAcceleration": TOTAL_ACCELERATION,
    "AccelerometerUncalibrated": ACCELEROMETER_UNCALIBRATED,
    "GyroscopeUncalibrated": GYROSCOPE_UNCALIBRATED,
    "WatchAccelerometer": WATCH_ACCELEROMETER,
    "WatchGyroscope": WATCH_GYROSCOPE,
    "WatchGravity": WATCH_GRAVITY,
    "WatchTotalAcceleration": WATCH_TOTAL_ACCELERATION,
    "WatchOrientation": WATCH_ORIENTATION,
    "HeartRate": HEART_RATE,
}


def detect_schema(filename: str) -> SensorSchema | None:
    """Detect schema from a CSV filename (stem only, no extension)."""
    for key, schema in SENSOR_MAP.items():
        if key in filename:
            return schema
    return None
