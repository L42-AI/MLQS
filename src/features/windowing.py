"""Window segmentation — slice sensor streams into analysis windows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_sliding_windows(
    sensor_data: pd.DataFrame,
    window_size_seconds: float,
    sampling_rate_hz: float = 100.0,
    overlap_fraction: float = 0.5,
    time_column: str | None = "time",
) -> pd.DataFrame:
    sensor_readings = (
        sensor_data.drop(columns=[time_column])
        if time_column and time_column in sensor_data.columns
        else sensor_data.copy()
    )
    window_length = int(round(window_size_seconds * sampling_rate_hz))
    slide_step = max(1, int(round(window_length * (1.0 - overlap_fraction))))
    start_indices = np.arange(0, len(sensor_readings) - window_length + 1, slide_step)

    window_dataframes = []
    for window_index, start_index in enumerate(start_indices):
        window_data = sensor_readings.iloc[start_index : start_index + window_length].copy()
        window_data.index = pd.MultiIndex.from_product(
            [[window_index], range(window_length)], names=["window", "sample"]
        )
        window_dataframes.append(window_data)
    return pd.concat(window_dataframes)
