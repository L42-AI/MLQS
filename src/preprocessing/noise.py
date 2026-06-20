"""Noise-removal filters for sensor time series."""

import inspect

import numpy as np
import pandas as pd
from scipy import signal


def apply_butterworth_filter(
    series: pd.Series,
    cutoff_frequency: float | list[float],
    sample_rate_hz: float = 100.0,
    filter_order: int = 4,
    filter_type: str = "low",
) -> pd.Series:
    nyquist = 0.5 * sample_rate_hz
    if filter_type == "band":
        if isinstance(cutoff_frequency, (list, tuple)):
            normalised_cutoff = [c / nyquist for c in cutoff_frequency]
        else:
            # Float cutoff for band-pass: create a ±30 % band around center.
            half_bw = cutoff_frequency * 0.3
            normalised_cutoff = [
                (cutoff_frequency - half_bw) / nyquist,
                (cutoff_frequency + half_bw) / nyquist,
            ]
    else:
        normalised_cutoff = cutoff_frequency / nyquist
    # Clamp to (0, 1) so that extreme cutoff/sample-rate combos don't crash.
    _EPS = 1e-6
    if isinstance(normalised_cutoff, list):
        normalised_cutoff = [
            max(_EPS, min(1.0 - _EPS, c)) for c in normalised_cutoff
        ]
    else:
        normalised_cutoff = max(_EPS, min(1.0 - _EPS, normalised_cutoff))

    numerator, denominator = signal.butter(filter_order, normalised_cutoff, btype=filter_type, analog=False)
    filled_values = series.interpolate().bfill().fillna(0.0)
    return pd.Series(
        signal.filtfilt(numerator, denominator, filled_values.values),
        index=series.index,
        name=series.name,
    )


def apply_moving_average_filter(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def apply_savitzky_golay_filter(
    series: pd.Series, window_length: int = 11, polyorder: int = 3
) -> pd.Series:
    if len(series) < window_length:
        return series
    filled_values = series.interpolate().bfill().fillna(0.0)
    return pd.Series(
        signal.savgol_filter(filled_values.values, window_length, polyorder),
        index=series.index,
        name=series.name,
    )


FILTER_NAME_TO_FUNCTION = {
    "butterworth": apply_butterworth_filter,
    "moving_average": apply_moving_average_filter,
    "savitzky_golay": apply_savitzky_golay_filter,
}


def apply_filter_to_columns(
    sensor_data: pd.DataFrame,
    columns: list[str],
    filter_method: str = "butterworth",
    **filter_kwargs,
) -> pd.DataFrame:
    selected_filter = FILTER_NAME_TO_FUNCTION.get(filter_method)
    if selected_filter is None:
        raise ValueError(f"Unknown filter: {filter_method}")

    # Only forward keyword arguments that the specific filter function accepts.
    # (butterworth expects cutoff_frequency, sample_rate_hz, filter_order, filter_type;
    #  savitzky_golay expects window_length, polyorder;
    #  moving_average expects window.)
    sig = inspect.signature(selected_filter)
    valid_kwargs = {
        k: v for k, v in filter_kwargs.items()
        if k in sig.parameters
    }

    filtered_data = sensor_data.copy()
    for column in columns:
        if column in filtered_data.columns:
            filtered_data[column] = selected_filter(filtered_data[column], **valid_kwargs)
    return filtered_data
