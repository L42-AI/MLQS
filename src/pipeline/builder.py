"""Preprocessing → feature extraction → selection pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import Config
from features.engineering import extract_features_from_windows
from features.selection import run_selection_pipeline
from preprocessing.missing import forward_fill, interpolate_linear, knn_impute
from preprocessing.noise import apply_filter_to_columns
from utils.convert import resample_rule_to_frequency_hz


@dataclass
class PipelineResult:
    feature_matrix: pd.DataFrame
    labels: pd.Series | None
    feature_names: list[str]
    groups: pd.Series | None = None


IMPUTATION_METHOD_TO_FUNCTION = {
    "interpolate": lambda data, config, columns: interpolate_linear(
        data, max_gap=config.imputation_max_gap, columns=columns
    ),
    "ffill": lambda data, config, columns: forward_fill(
        data, max_gap=config.imputation_max_gap, columns=columns
    ),
    "knn": lambda data, config, columns: knn_impute(data, columns=columns),
}


def run_feature_pipeline(
    merged_sensor_data: pd.DataFrame,
    experiment_config: Config,
) -> PipelineResult:
    processed_data = merged_sensor_data.copy()
    preprocessing_config = experiment_config.preprocessing

    label_column = "label" if "label" in processed_data.columns else None
    non_sensor_columns = {"label", "experiment_id"}
    sensor_columns = [
        column
        for column in processed_data.select_dtypes(include="number").columns
        if column not in non_sensor_columns
    ]
    if not sensor_columns:
        return PipelineResult(feature_matrix=pd.DataFrame(), labels=None, feature_names=[])

    sample_rate_hz = resample_rule_to_frequency_hz(preprocessing_config.resample_rule)

    if preprocessing_config.filter_method:
        processed_data = apply_filter_to_columns(
            processed_data,
            sensor_columns,
            filter_method=preprocessing_config.filter_method,
            cutoff_frequency=preprocessing_config.filter_cutoff,
            sample_rate_hz=sample_rate_hz,
            filter_order=preprocessing_config.filter_order,
            filter_type=preprocessing_config.filter_type,
        )

    impute_fn = IMPUTATION_METHOD_TO_FUNCTION.get(
        preprocessing_config.imputation_method,
        IMPUTATION_METHOD_TO_FUNCTION["interpolate"],
    )
    processed_data = impute_fn(processed_data, preprocessing_config, sensor_columns)

    extracted_features = extract_features_from_windows(
        processed_data, sensor_columns, experiment_config
    )
    if extracted_features.empty:
        return PipelineResult(feature_matrix=pd.DataFrame(), labels=None, feature_names=[])

    extracted_labels = (
        extracted_features.pop("label")
        if "label" in extracted_features.columns
        else None
    )

    extracted_groups = (
        extracted_features.pop("experiment_id")
        if "experiment_id" in extracted_features.columns
        else None
    )

    if experiment_config.features.selection_methods and extracted_labels is not None:
        extracted_features = run_selection_pipeline(
            extracted_features,
            extracted_labels,
            selection_methods=list(experiment_config.features.selection_methods),
        )

    return PipelineResult(
        feature_matrix=extracted_features,
        labels=extracted_labels,
        feature_names=extracted_features.columns.tolist(),
        groups=extracted_groups,
    )
