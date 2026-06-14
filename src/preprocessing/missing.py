"""Missing-value handling for sensor time series."""

import numpy as np
import pandas as pd


def build_missing_value_report(sensor_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in sensor_data.columns:
        missing_count = sensor_data[column].isnull().sum()
        if missing_count:
            rows.append(
                {
                    "column": column,
                    "n_missing": missing_count,
                    "pct": round(missing_count / len(sensor_data) * 100, 2),
                }
            )
    return pd.DataFrame(rows).sort_values("n_missing", ascending=False)


def _impute_with_method(
    sensor_data: pd.DataFrame,
    method: str,
    max_gap: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    imputed_data = sensor_data.copy()
    columns_to_impute = columns or imputed_data.select_dtypes(include=[np.number]).columns.tolist()
    for column in columns_to_impute:
        if column not in imputed_data.columns:
            continue
        if max_gap is not None:
            imputed_data[column] = _fill_limited_gaps(imputed_data[column], max_gap, method)
        elif method == "interpolate":
            imputed_data[column] = imputed_data[column].interpolate()
        elif method == "ffill":
            imputed_data[column] = imputed_data[column].ffill()
    return imputed_data


def interpolate_linear(
    sensor_data: pd.DataFrame,
    max_gap: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    return _impute_with_method(sensor_data, "interpolate", max_gap, columns)


def forward_fill(
    sensor_data: pd.DataFrame,
    max_gap: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    return _impute_with_method(sensor_data, "ffill", max_gap, columns)


def knn_impute(
    sensor_data: pd.DataFrame,
    max_gap: int | None = None,      # ignored — KNN handles all gaps
    columns: list[str] | None = None,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    from sklearn.impute import KNNImputer

    columns_to_impute = columns or sensor_data.select_dtypes(include=[np.number]).columns.tolist()
    if not any(sensor_data[column].isnull().any() for column in columns_to_impute):
        return sensor_data.copy()
    imputed_data = sensor_data.copy()
    imputed_data[columns_to_impute] = KNNImputer(n_neighbors=n_neighbors).fit_transform(
        imputed_data[columns_to_impute]
    )
    return imputed_data


def _fill_limited_gaps(
    series: pd.Series, max_gap: int, method: str
) -> pd.Series:
    filled = series.copy()
    null_mask = filled.isnull()
    if not null_mask.any():
        return filled

    null_run_ids = null_mask.ne(null_mask.shift()).cumsum()
    for run_id, gap_length in null_mask.groupby(null_run_ids).sum().items():
        if gap_length <= max_gap:
            gap_mask = null_run_ids == run_id
            if method == "linear":
                filled[gap_mask] = filled.interpolate()[gap_mask]
            else:
                filled[gap_mask] = filled.ffill()[gap_mask]
    return filled
