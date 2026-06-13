"""Temporal resampling — align irregular sensor readings to a uniform grid."""

import pandas as pd


def resample_to_uniform_grid(
    sensor_data: pd.DataFrame,
    resample_rule: str | float,
    time_column: str = "time",
    aggregation: str = "mean",
) -> pd.DataFrame:
    if sensor_data.empty:
        return sensor_data

    if isinstance(resample_rule, (int, float)):
        resolved_rule = (
            f"{int(resample_rule * 1000)}ms"
            if resample_rule < 1
            else f"{int(resample_rule)}s"
        )
    else:
        resolved_rule = resample_rule

    if time_column in sensor_data.columns:
        data_with_time_index = sensor_data.drop(columns=[time_column]).set_index(
            pd.to_timedelta(sensor_data[time_column].values, unit="s")
        )
    else:
        data_with_time_index = sensor_data.set_index(
            pd.to_timedelta(sensor_data.index, unit="s")
        )

    return data_with_time_index.resample(resolved_rule).agg(aggregation).dropna(how="all")


def synchronise_sensor_frames(
    sensor_dataframes: dict[str, pd.DataFrame],
    resample_rule: str | float = "100ms",
    time_column: str = "time",
) -> pd.DataFrame:
    resampled_frames = []
    for sensor_name, sensor_frame in sensor_dataframes.items():
        uniform_frame = resample_to_uniform_grid(
            sensor_frame, resample_rule=resample_rule, time_column=time_column
        )
        uniform_frame.columns = pd.MultiIndex.from_product([[sensor_name], uniform_frame.columns])
        resampled_frames.append(uniform_frame)
    return pd.concat(resampled_frames, axis=1).sort_index()
