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

NON_SENSOR_COLUMNS = {"label", "experiment_id", "participant"}


@dataclass
class PipelineResult:
    feature_matrix: pd.DataFrame
    labels: pd.Series | None
    feature_names: list[str]
    groups: pd.Series | None = None
    block_ids: pd.Series | None = None
    participant: pd.Series | None = None


IMPUTATION_METHOD_TO_FUNCTION = {
    "interpolate": interpolate_linear,
    "ffill": forward_fill,
    "knn": knn_impute,
}


def _sensor_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric column names that aren't labels or metadata."""
    return [
        c
        for c in data.select_dtypes(include="number").columns
        if c not in NON_SENSOR_COLUMNS
    ]


def _pop_metadata(features: pd.DataFrame):
    """Extract label, participant, and experiment_id from *features* (if present).

    Returns (labels, groups, participant, remaining_features).
    """
    labels = features.pop("label") if "label" in features.columns else None
    groups = (
        features.pop("experiment_id") if "experiment_id" in features.columns else None
    )
    participant = (
        features.pop("participant") if "participant" in features.columns else None
    )
    return labels, groups, participant, features


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

    labels, groups, participant, features = _pop_metadata(features)

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
        participant=participant,
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
    train_labels, train_groups, _, train_features = _pop_metadata(train_df)
    train_block_ids = train_features.pop("_block_id") if "_block_id" in train_features.columns else None

    test_labels, test_groups, _, test_features = _pop_metadata(test_df)
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


# ── Participant-based train/test split (no participant leakage) ──────────────


def run_participant_train_test_pipeline(
    merged_sensor_data: pd.DataFrame,
    experiment_config: Config,
    oos_participant: str = "Kim",
) -> tuple[PipelineResult, PipelineResult]:
    """Preprocess → split by participant → window independently → train/OOS.

    Unlike the block-based split, this function **never** lets the same
    participant appear in both training and test sets.  All recordings
    from *oos_participant* are held out for final evaluation; all other
    participants become the training set.

    This eliminates the risk that the model is simply learning
    participant-specific gait patterns (walking style, phone placement,
    sensor biases) rather than the actual effect of music on movement.

    Parameters
    ----------
    merged_sensor_data :
        Sensor DataFrame with a ``participant`` column (added by
        :func:`~data.loader.load_all_experiment_sensors`).
    experiment_config :
        Full configuration (preprocessing, features, model).
    oos_participant :
        Participant whose recordings are never seen during training.
        Defaults to ``"Kim"``.

    Returns
    -------
    (train_result, oos_result)
        Each ``PipelineResult`` contains the feature matrix, labels,
        participant IDs, and feature names for its split.
    """
    if "participant" not in merged_sensor_data.columns:
        raise ValueError(
            "participant-based split requires a 'participant' column. "
            "Make sure load_all_experiment_sensors is configured to extract it."
        )

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

    # ── Split by participant ────────────────────────────────────────────
    train_data = data[data["participant"] != oos_participant].copy()
    oos_data = data[data["participant"] == oos_participant].copy()

    if train_data.empty:
        print(f"  WARNING: no training data (all participants are '{oos_participant}')")
        empty = PipelineResult(feature_matrix=pd.DataFrame(), labels=None, feature_names=[])
        return empty, empty

    # ── Helper: window group data WITHOUT feature selection ─────────────
    # Feature selection is applied ONCE on the combined training set
    # (see step 3 below) so that all participants use the same features.
    def _window_group(group_data: pd.DataFrame, label: str) -> PipelineResult:
        """Window a single participant's data — no feature selection yet."""
        feats = extract_features_from_windows(
            group_data, sensor_columns, features_config, sample_rate,
            block_info=label,
        )
        if feats.empty:
            return PipelineResult(feature_matrix=pd.DataFrame(), labels=None, feature_names=[])

        labs, grps, parts, feat_matrix = _pop_metadata(feats)

        return PipelineResult(
            feature_matrix=feat_matrix,
            labels=labs,
            feature_names=feat_matrix.columns.tolist() if not feat_matrix.empty else [],
            groups=grps,
            participant=parts,
        )

    # ── 1. Window each training participant independently ───────────────
    train_results: list[PipelineResult] = []
    for participant_name, group_df in train_data.groupby("participant"):
        print(f"  Windowing participant '{participant_name}' ...", flush=True)
        result = _window_group(group_df, participant_name)
        if not result.feature_matrix.empty:
            train_results.append(result)

    if not train_results:
        empty = PipelineResult(feature_matrix=pd.DataFrame(), labels=None, feature_names=[])
        return empty, empty

    # ── 2. Concatenate all training participant features ────────────────
    train_feats = pd.concat([r.feature_matrix for r in train_results], ignore_index=True)
    train_labels = (
        pd.concat([r.labels for r in train_results], ignore_index=True)
        if train_results[0].labels is not None else None
    )
    train_parts = (
        pd.concat([r.participant for r in train_results], ignore_index=True)
        if train_results[0].participant is not None else None
    )
    train_groups = (
        pd.concat([r.groups for r in train_results], ignore_index=True)
        if train_results[0].groups is not None else None
    )

    # ── 3. Feature selection ONCE on combined training data ─────────────
    # This ensures every participant's features go through identical
    # selection — no column mismatch at concatenation time.
    if features_config.selection_methods and train_labels is not None:
        train_feats = run_selection_pipeline(
            train_feats, train_labels,
            selection_methods=list(features_config.selection_methods),
        )

    train_result = PipelineResult(
        feature_matrix=train_feats,
        labels=train_labels,
        feature_names=train_feats.columns.tolist(),
        groups=train_groups,
        participant=train_parts,
    )

    # ── 4. Process OOS participant (same windowing, no selection) ───────
    if oos_data.empty:
        return train_result, PipelineResult(feature_matrix=pd.DataFrame(), labels=None, feature_names=[])

    print(f"  Windowing OOS participant '{oos_participant}' ...", flush=True)
    oos_result = _window_group(oos_data, oos_participant)

    # Apply same feature columns as training (features that survived selection)
    if not oos_result.feature_matrix.empty and not train_result.feature_matrix.empty:
        common_cols = [c for c in train_result.feature_matrix.columns if c in oos_result.feature_matrix.columns]
        oos_result.feature_matrix = oos_result.feature_matrix[common_cols]
        oos_result.feature_names = common_cols

    return train_result, oos_result


# ── Within-subject train/test split (same participant in both) ─────────────


def run_within_subject_train_test_pipeline(
    merged_sensor_data: pd.DataFrame,
    experiment_config: Config,
    test_fraction: float = 0.2,
    n_blocks: int = 10,
    random_seed: int = 42,
) -> tuple[PipelineResult, PipelineResult]:
    """Preprocess → split blocks per participant → window → train/test.

    Unlike the participant-based split, **every** participant contributes
    to both training and test sets, with their individual time series split
    into contiguous blocks, shuffled, and windowed independently per block.

    This tells you whether the model can learn anything from the data at
    all: if performance is good here but poor in the participant-based
    LOPO split, the signal is real but person-specific (not generalisable
    to unseen participants).

    Parameters
    ----------
    test_fraction :
        Fraction of each participant's blocks held out for testing.
    n_blocks :
        Number of contiguous blocks to split each participant's data into.
    random_seed :
        Seed for shuffling blocks (per-participant).

    Returns
    -------
    (train_result, test_result)
        Each ``PipelineResult`` contains the feature matrix, labels,
        participant IDs, and feature names.
        Both splits contain all participants.
    """
    if "participant" not in merged_sensor_data.columns:
        raise ValueError(
            "within-subject split requires a 'participant' column. "
            "Make sure load_all_experiment_sensors is configured to extract it."
        )

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

    # ── Helper: window a single block (pre-allocated slice) ─────────────
    def _window_block(block: pd.DataFrame, block_id: int, label: str) -> PipelineResult:
        feats = extract_features_from_windows(
            block, sensor_columns, features_config, sample_rate,
            block_info=label,
        )
        if feats.empty:
            return PipelineResult(feature_matrix=pd.DataFrame(), labels=None, feature_names=[])

        labs, grps, parts, feat_matrix = _pop_metadata(feats)
        feat_matrix["_block_id"] = block_id
        return PipelineResult(
            feature_matrix=feat_matrix,
            labels=labs,
            feature_names=feat_matrix.columns.tolist() if not feat_matrix.empty else [],
            groups=grps,
            participant=parts,
        )

    train_blocks: list[PipelineResult] = []
    test_blocks: list[PipelineResult] = []

    for participant_name, group_df in data.groupby("participant"):
        total = len(group_df)
        if total < n_blocks * 2:
            print(f"  Participant '{participant_name}' too short ({total} rows, need ≥{n_blocks * 2}) — skipping", flush=True)
            continue

        block_size = total // n_blocks
        raw_blocks: list[pd.DataFrame] = []
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size if i < n_blocks - 1 else total
            raw_blocks.append(group_df.iloc[start:end].copy())

        rng = np.random.RandomState(random_seed)
        rng.shuffle(raw_blocks)

        split_idx = max(1, int(len(raw_blocks) * (1.0 - test_fraction)))
        participant_train = raw_blocks[:split_idx]
        participant_test = raw_blocks[split_idx:]

        for bid, block in enumerate(participant_train):
            result = _window_block(block, bid, f"{participant_name}_train")
            if not result.feature_matrix.empty:
                train_blocks.append(result)

        for bid, block in enumerate(participant_test):
            result = _window_block(block, bid, f"{participant_name}_test")
            if not result.feature_matrix.empty:
                test_blocks.append(result)

        print(f"  Participant '{participant_name}': {len(participant_train)} train blocks, {len(participant_test)} test blocks", flush=True)

    if not train_blocks or not test_blocks:
        print("  WARNING: no data for train or test — aborting", flush=True)
        empty = PipelineResult(feature_matrix=pd.DataFrame(), labels=None, feature_names=[])
        return empty, empty

    # ── Concatenate ─────────────────────────────────────────────────────
    def _concat(results: list[PipelineResult]) -> PipelineResult:
        feats = pd.concat([r.feature_matrix for r in results], ignore_index=True)
        labels = pd.concat([r.labels for r in results], ignore_index=True) if results[0].labels is not None else None
        parts = pd.concat([r.participant for r in results], ignore_index=True) if results[0].participant is not None else None
        groups = pd.concat([r.groups for r in results], ignore_index=True) if results[0].groups is not None else None
        block_ids = feats.pop("_block_id") if "_block_id" in feats.columns else None
        return PipelineResult(
            feature_matrix=feats, labels=labels,
            feature_names=feats.columns.tolist(),
            groups=groups, participant=parts, block_ids=block_ids,
        )

    train_result = _concat(train_blocks)
    test_result = _concat(test_blocks)

    # ── Feature selection ONCE on training set ──────────────────────────
    if features_config.selection_methods and train_result.labels is not None:
        train_result.feature_matrix = run_selection_pipeline(
            train_result.feature_matrix, train_result.labels,
            selection_methods=list(features_config.selection_methods),
        )
        train_result.feature_names = train_result.feature_matrix.columns.tolist()

    # Apply same columns to test
    if not test_result.feature_matrix.empty and not train_result.feature_matrix.empty:
        common_cols = [c for c in train_result.feature_matrix.columns if c in test_result.feature_matrix.columns]
        test_result.feature_matrix = test_result.feature_matrix[common_cols]
        test_result.feature_names = common_cols

    return train_result, test_result
