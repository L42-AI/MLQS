"""Preprocessing → feature extraction → selection pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
    block_ids: pd.Series | None = None


IMPUTATION_METHOD_TO_FUNCTION = {
    "interpolate": lambda data, max_gap, columns: interpolate_linear(
        data, max_gap=max_gap, columns=columns
    ),
    "ffill": lambda data, max_gap, columns: forward_fill(
        data, max_gap=max_gap, columns=columns
    ),
    "knn": lambda data, max_gap, columns: knn_impute(data, columns=columns),
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
    data = impute(data, max_gap=preprocessing.imputation_max_gap, columns=sensor_columns)

    # ── Magnitude channels (rotation-invariant) ───────────────────────
    if features_config.magnitude_channels:
        data = add_magnitude_channels(data)
        sensor_columns = _sensor_columns(data)

    # ── Feature extraction ────────────────────────────────────────────
    features = extract_features_from_windows(data, sensor_columns, features_config, sample_rate)
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


# ── Block-based train/test split (no overlapping windows) ──────────────────


def _split_blocks(data: pd.DataFrame, n_blocks: int) -> list[pd.DataFrame]:
    """Split a time series into *n_blocks* contiguous blocks."""
    total = len(data)
    block_size = total // n_blocks
    blocks: list[pd.DataFrame] = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else total
        blocks.append(data.iloc[start:end].copy())
    return blocks


def _preprocess(
    data: pd.DataFrame,
    config: Config,
    sensor_columns: list[str],
    sample_rate: float,
) -> pd.DataFrame:
    """Apply filtering, imputation, and magnitude augmentation."""
    pp = config.preprocessing
    fc = config.features

    if pp.filter_method:
        data = apply_filter_to_columns(
            data, sensor_columns,
            filter_method=pp.filter_method,
            cutoff_frequency=pp.filter_cutoff,
            sample_rate_hz=sample_rate,
            filter_order=pp.filter_order,
            filter_type=pp.filter_type,
        )

    impute = IMPUTATION_METHOD_TO_FUNCTION.get(
        pp.imputation_method,
        IMPUTATION_METHOD_TO_FUNCTION["interpolate"],
    )
    data = impute(data, max_gap=pp.imputation_max_gap, columns=sensor_columns)

    if fc.magnitude_channels:
        data = add_magnitude_channels(data)

    return data


def run_train_test_pipeline(
    merged_sensor_data: pd.DataFrame,
    experiment_config: Config,
    test_fraction: float = 0.2,
    n_blocks: int = 10,
    random_seed: int = 42,
) -> tuple[PipelineResult, PipelineResult]:
    """Preprocess → split into blocks → window each block → train/test.

    The raw time series is divided into *n_blocks* contiguous blocks,
    shuffled, and assigned to train/test.  Each block is **windowed
    independently**, guaranteeing **zero overlapping samples** between
    train and test windows (no data leakage).

    Returns
    -------
    (train_result, test_result)
        Each ``PipelineResult`` contains the feature matrix, labels,
        groups, and feature names for its respective split.
    """
    data = merged_sensor_data.copy()
    preprocessing = experiment_config.preprocessing
    features_config = experiment_config.features

    sensor_columns = _sensor_columns(data)
    if not sensor_columns:
        empty = PipelineResult(feature_matrix=pd.DataFrame(), labels=None, feature_names=[])
        return empty, empty

    sample_rate = resample_rule_to_frequency_hz(preprocessing.resample_rule)

    # ── Preprocess (filter, impute, augment) ────────────────────────────
    data = _preprocess(data, experiment_config, sensor_columns, sample_rate)
    sensor_columns = _sensor_columns(data)

    # ── Split into blocks, shuffle, window independently ───────────────
    blocks = _split_blocks(data, n_blocks)
    rng = np.random.RandomState(random_seed)
    rng.shuffle(blocks)

    block_results: list[pd.DataFrame] = []
    for block_id, block in enumerate(blocks):
        feats = extract_features_from_windows(
            block, sensor_columns, features_config, sample_rate,
            block_info=f"{block_id + 1}/{n_blocks}",
        )
        if not feats.empty:
            feats["_block_id"] = block_id
            block_results.append(feats)

    if not block_results:
        empty = PipelineResult(feature_matrix=pd.DataFrame(), labels=None, feature_names=[])
        return empty, empty

    # ── Assign blocks to train/test ─────────────────────────────────────
    split_idx = max(1, int(len(block_results) * (1.0 - test_fraction)))
    train_blocks = block_results[:split_idx]
    test_blocks = block_results[split_idx:]

    train_df = pd.concat(train_blocks, ignore_index=True)
    test_df = pd.concat(test_blocks, ignore_index=True)

    # ── Pop metadata + block_ids ────────────────────────────────────────
    train_labels, train_groups, train_features = _pop_metadata(train_df)
    train_block_ids = train_features.pop("_block_id") if "_block_id" in train_features.columns else None

    test_labels, test_groups, test_features = _pop_metadata(test_df)
    test_block_ids = test_features.pop("_block_id") if "_block_id" in test_features.columns else None

    # ── Feature selection (fit on train, transform both) ────────────────
    if features_config.selection_methods and train_labels is not None:
        train_features = run_selection_pipeline(
            train_features, train_labels,
            selection_methods=list(features_config.selection_methods),
        )
        # Apply same column selection to test (columns that survived on train)
        common_cols = [c for c in train_features.columns if c in test_features.columns]
        test_features = test_features[common_cols]

    return (
        PipelineResult(
            feature_matrix=train_features,
            labels=train_labels,
            feature_names=train_features.columns.tolist(),
            groups=train_groups,
            block_ids=train_block_ids,
        ),
        PipelineResult(
            feature_matrix=test_features,
            labels=test_labels,
            feature_names=test_features.columns.tolist(),
            groups=test_groups,
            block_ids=test_block_ids,
        ),
    )
