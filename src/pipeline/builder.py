"""Preprocessing → feature extraction → selection pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import Config
from features.augmentation import add_magnitude_channels
from features.engineering import extract_features_from_windows
from features.selection import run_selection_pipeline
from preprocessing.missing import forward_fill, interpolate_linear, knn_impute
from preprocessing.noise import apply_filter_to_columns
from utils.convert import resample_rule_to_frequency_hz

NON_SENSOR_COLUMNS = {"label", "experiment_id"}


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


def _sensor_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric column names that aren't labels or metadata."""
    return [
        c
        for c in data.select_dtypes(include="number").columns
        if c not in NON_SENSOR_COLUMNS
    ]


def _pop_metadata(features: pd.DataFrame):
    """Extract label and experiment_id columns from *features* (if present).

    Returns (labels, groups, remaining_features).
    """
    labels = features.pop("label") if "label" in features.columns else None
    groups = (
        features.pop("experiment_id") if "experiment_id" in features.columns else None
    )
    return labels, groups, features


def run_feature_pipeline(
    merged_sensor_data: pd.DataFrame,
    experiment_config: Config,
) -> PipelineResult:
    data = merged_sensor_data.copy()
    preprocessing = experiment_config.preprocessing
    features_config = experiment_config.features

    sensor_columns = _sensor_columns(data)
    if not sensor_columns:
        return PipelineResult(feature_matrix=pd.DataFrame(), labels=None, feature_names=[])

    sample_rate = resample_rule_to_frequency_hz(preprocessing.resample_rule)

    # ── Filtering ─────────────────────────────────────────────────────
    if preprocessing.filter_method:
        data = apply_filter_to_columns(
            data,
            sensor_columns,
            filter_method=preprocessing.filter_method,
            cutoff_frequency=preprocessing.filter_cutoff,
            sample_rate_hz=sample_rate,
            filter_order=preprocessing.filter_order,
            filter_type=preprocessing.filter_type,
        )

    # ── Imputation ────────────────────────────────────────────────────
    impute = IMPUTATION_METHOD_TO_FUNCTION.get(
        preprocessing.imputation_method,
        IMPUTATION_METHOD_TO_FUNCTION["interpolate"],
    )
    data = impute(data, preprocessing, sensor_columns)

    # ── Magnitude channels (rotation-invariant) ───────────────────────
    if features_config.magnitude_channels:
        data = add_magnitude_channels(data)
        sensor_columns = _sensor_columns(data)

    # ── Feature extraction ────────────────────────────────────────────
    features = extract_features_from_windows(data, sensor_columns, experiment_config)
    if features.empty:
        return PipelineResult(feature_matrix=pd.DataFrame(), labels=None, feature_names=[])

    labels, groups, features = _pop_metadata(features)

    # ── Feature selection ─────────────────────────────────────────────
    if features_config.selection_methods and labels is not None:
        features = run_selection_pipeline(
            features,
            labels,
            selection_methods=list(features_config.selection_methods),
        )

    return PipelineResult(
        feature_matrix=features,
        labels=labels,
        feature_names=features.columns.tolist(),
        groups=groups,
    )
