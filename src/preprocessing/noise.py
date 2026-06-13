"""Noise-removal filters for sensor time series."""

import numpy as np
import pandas as pd
from scipy import signal


def apply_butterworth_filter(
    series: pd.Series,
    cutoff_frequency: float,
    sample_rate_hz: float = 100.0,
    filter_order: int = 4,
    filter_type: str = "low",
) -> pd.Series:
    nyquist = 0.5 * sample_rate_hz
    normalised_cutoff = (
        [c / nyquist for c in cutoff_frequency]
        if filter_type == "band"
        else cutoff_frequency / nyquist
    )
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
    filtered_data = sensor_data.copy()
    for column in columns:
        if column in filtered_data.columns:
            filtered_data[column] = selected_filter(filtered_data[column], **filter_kwargs)
    return filtered_data
