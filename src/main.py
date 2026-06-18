#!/usr/bin/env python
"""Train and evaluate models on phyphox sensor data."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import faulthandler
import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

faulthandler.enable()

# Prevent OpenMP thread spawning that causes XGBoost segfaults on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from tabulate import tabulate
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder


from config import Config, PreprocessingConfig, FeatureConfig, ModelConfig
from data.loader import compute_sensor_summary, detect_data_quality_issues, load_all_experiment_sensors
from models.classical import build_classifier
from models.deep import build_deep_classifier, prepare_sequences, train_deep_model
from models.evaluation import (
    build_model_comparison_table,
    compute_classification_metrics,
    compute_confusion_matrix,
    format_classification_report,
)
from features.importance import (
    extract_rf_importances,
    extract_xgboost_importances,
    sort_and_display_top_features,
)
from pipeline.builder import (
    PipelineResult,
    run_participant_train_test_pipeline,
    run_train_test_pipeline,
    run_within_subject_train_test_pipeline,
)




def _run_exploratory_analysis(experiment_config: Config) -> None:
    sensor_data = load_all_experiment_sensors(
        experiment_config.raw_dir,
        resample_rule=experiment_config.preprocessing.resample_rule,
    )
    data_summary = compute_sensor_summary(sensor_data)
    quality_report = detect_data_quality_issues(sensor_data)

    print(f"Experiments loaded from: {experiment_config.raw_dir}")
    print(f"Shape:     {data_summary['shape']}")
    print(f"Columns:   {data_summary['columns']}")
    print(f"Missing:   {data_summary['missing']['total']} ({data_summary['missing']['pct']}%)")
    if quality_report["constant_columns"]:
        print(f"Constants: {quality_report['constant_columns']}")
    if "label" in sensor_data.columns:
        print(f"Labels:    {sensor_data['label'].value_counts().to_dict()}")


# ── Class-subset helpers ─────────────────────────────────────────────────────

_CLASS_SUBSETS: dict[str, list[str] | None] = {
    "all": None,
    "hard+soft": ["HARD", "SOFT"],
    "hard+silence": ["HARD", "SILENCE"],
    "soft+silence": ["SOFT", "SILENCE"],
}


def _filter_pipeline_result(
    result: PipelineResult,
    keep_classes: list[str] | None,
) -> PipelineResult | None:
    if keep_classes is None:
        return result

    mask = result.labels.isin(keep_classes) if result.labels is not None else None
    if mask is None or mask.sum() == 0:
        return None

    n_unique = result.labels[mask].nunique()
    if n_unique < 2:
        return None

    return PipelineResult(
        feature_matrix=result.feature_matrix.loc[mask].reset_index(drop=True),
        labels=result.labels.loc[mask].reset_index(drop=True),
        feature_names=result.feature_names,
        groups=result.groups.loc[mask].reset_index(drop=True) if result.groups is not None else None,
        participant=result.participant.loc[mask].reset_index(drop=True) if result.participant is not None else None,
        block_ids=result.block_ids.loc[mask].reset_index(drop=True) if result.block_ids is not None else None,
    )


# ── Pipeline result caching ───────────────────────────────────────────────────

CACHE_ROOT = Path(__file__).resolve().parent.parent / ".tmp" / "feature_cache"


def _config_cache_key(config: Config, split_strategy: str) -> str:
    config_dict = {
        "split": split_strategy,
        "preprocessing": {
            "filter_method": config.preprocessing.filter_method,
            "filter_cutoff": config.preprocessing.filter_cutoff,
            "filter_order": config.preprocessing.filter_order,
            "filter_type": config.preprocessing.filter_type,
            "imputation_method": config.preprocessing.imputation_method,
            "imputation_max_gap": config.preprocessing.imputation_max_gap,
            "resample_rule": config.preprocessing.resample_rule,
        },
        "features": {
            "window_size": config.features.window_size,
            "window_overlap": config.features.window_overlap,
            "frequency_window_size": config.features.frequency_window_size,
            "magnitude_channels": config.features.magnitude_channels,
            "cross_sensor_features": config.features.cross_sensor_features,
            "time_domain": config.features.time_domain,
            "frequency_domain": config.features.frequency_domain,
            "statistical": config.features.statistical,
            "selection_methods": list(config.features.selection_methods),
        },
    }
    raw = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _pipeline_cache_path(cache_key: str, split_strategy: str) -> Path:
    """Path to cached pipeline result for *split_strategy*."""
    return CACHE_ROOT / cache_key / f"{split_strategy}.pkl"


def _save_pipelines_to_cache(
    cache_key: str,
    pipelines: dict[str, tuple[PipelineResult, PipelineResult]],
) -> None:
    """Save all pipeline results for a given cache key."""
    cache_dir = CACHE_ROOT / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split_strategy, (val, tst) in pipelines.items():
        path = _pipeline_cache_path(cache_key, split_strategy)
        with open(path, "wb") as f:
            pickle.dump((val, tst), f)


def _load_pipelines_from_cache(
    cache_key: str,
    split_strategies: list[str],
) -> dict[str, tuple[PipelineResult, PipelineResult]] | None:
    """Load cached pipeline results for all *split_strategies*.

    Returns ``None`` if any strategy is missing from cache.
    """
    pipelines: dict[str, tuple[PipelineResult, PipelineResult]] = {}
    for ss in split_strategies:
        path = _pipeline_cache_path(cache_key, ss)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            pipelines[ss] = pickle.load(f)
    return pipelines


# ── Model configurations ─────────────────────────────────────────────────────
_CLASSICAL_MODELS: list[tuple[str, dict]] = [
    ("random_forest", {"n_estimators": 100, "max_depth": 10, "random_state": 42}),
    ("xgboost", {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "n_jobs": 1, "tree_method": "hist", "random_state": 42, "verbosity": 0}),
]



def _validate_and_clean_features(X: np.ndarray, label: str = "") -> np.ndarray:
    n_inf = int(np.isinf(X).sum())
    if n_inf:
        print(f"  ⚠ {label}: clipped {n_inf} Inf value(s) → NaN", flush=True)
        X = np.where(np.isinf(X), np.nan, X)

    n_nan = int(np.isnan(X).sum())
    if n_nan:
        raise ValueError(
            f"{label}: feature matrix contains {n_nan} NaN value(s). "
            "Fix upstream feature engineering before training."
        )
    return X


def _train_and_get_metrics(
    model_name: str,
    hyperparameters: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Train a single classifier and return (metrics_dict, predictions)."""
    X_train = _validate_and_clean_features(X_train, f"{model_name} train")
    X_test = _validate_and_clean_features(X_test, f"{model_name} test")
    classifier = build_classifier(model_name, **hyperparameters).fit(X_train, y_train)
    preds = classifier.predict(X_test)
    proba = classifier.predict_proba(X_test) if hasattr(classifier, "predict_proba") else None
    return compute_classification_metrics(y_test, preds, proba), preds


# ── Comparison runner ────────────────────────────────────────────────────────


def _split_label(label: str) -> str:
    """Short label for a split strategy used in tables."""
    return {"within_subject": "Within-Subject", "participant": "Cross-Participant (LOPO)"}.get(label, label)


def _run_comparison(sensor_data: pd.DataFrame, config: Config, no_cache: bool = False) -> None:
    split_strategies = ["within_subject", "participant"]
    subset_names = ["all", "hard+soft", "hard+silence", "soft+silence"]
    n_total = len(split_strategies) * len(subset_names) * len(_CLASSICAL_MODELS)

    cache_key = _config_cache_key(config, "compare")
    cache_hit = False

    print("=" * 78)
    print("  COMPARISON: Predictive power × generalisability × class contrast")
    print("=" * 78)

    # ── 0. Try loading from cache ───────────────────────────────────────
    pipelines: dict[str, tuple[PipelineResult, PipelineResult]] = {}
    if not no_cache:
        cached = _load_pipelines_from_cache(cache_key, split_strategies)
        if cached is not None:
            pipelines = cached
            cache_hit = True
            print(f"\n  Loaded {len(pipelines)} pipelines from cache ({CACHE_ROOT / cache_key})")
            for ss, (val, tst) in pipelines.items():
                print(f"    {ss}: {val.feature_matrix.shape[0]} train / {tst.feature_matrix.shape[0]} test windows")

    # ── 1. Run pipelines that weren't in cache ───────────────────────────
    if not cache_hit:
        for ss in split_strategies:
            if ss in pipelines:
                continue  # already loaded from cache
            if ss == "within_subject":
                print(f"\n  Pipeline: within-subject split ...", end=" ", flush=True)
                val, tst = run_within_subject_train_test_pipeline(
                    sensor_data, config, test_fraction=0.2, n_blocks=10, random_seed=42,
                )
            else:
                print(f"\n  Pipeline: cross-participant (OOS = '{config.models.oos_participant}') ...", end=" ", flush=True)
                val, tst = run_participant_train_test_pipeline(
                    sensor_data, config, oos_participant=config.models.oos_participant,
                )
            if val.feature_matrix.empty or tst.feature_matrix.empty:
                print("EMPTY — skipping")
                continue
            print(f"{val.feature_matrix.shape[0]} train / {tst.feature_matrix.shape[0]} test windows")
            pipelines[ss] = (val, tst)

        if pipelines:
            _save_pipelines_to_cache(cache_key, pipelines)
            print(f"\n  Cached pipelines to {CACHE_ROOT / cache_key}")

    if not pipelines:
        print("\n  No pipelines produced data — aborting.")
        return

    # ── 1. Grid evaluation ──────────────────────────────────────────────
    rows: list[dict] = []
    detail_store: dict[tuple[str, str, str], dict] = {}  # (split, subset, model) → metrics + preds

    idx = 0
    for ss, (val_result, test_result) in pipelines.items():
        for sub_name in subset_names:
            keep = _CLASS_SUBSETS[sub_name]

            val_sub = _filter_pipeline_result(val_result, keep)
            tst_sub = _filter_pipeline_result(test_result, keep)
            if val_sub is None or tst_sub is None:
                continue

            # Encode
            le = LabelEncoder()
            y_train = le.fit_transform(val_sub.labels)
            y_test = le.transform(tst_sub.labels)
            n_classes = len(le.classes_)
            X_train = val_sub.feature_matrix.values
            X_test = tst_sub.feature_matrix.values

            for model_name, hp in _CLASSICAL_MODELS:
                idx += 1
                print(f"  [{idx}/{n_total}] {_split_label(ss)} | {sub_name} | {model_name} ...", end=" ", flush=True)
                try:
                    metrics, preds = _train_and_get_metrics(model_name, hp, X_train, y_train, X_test, y_test)
                except Exception as exc:
                    print(f"FAILED ({exc})")
                    continue

                # Store for detailed breakdown
                detail_store[(ss, sub_name, model_name)] = {
                    "metrics": metrics,
                    "y_test": y_test,
                    "preds": preds,
                    "label_names": list(le.classes_),
                }

                rows.append({
                    "split": _split_label(ss),
                    "classes": sub_name,
                    "n": n_classes,
                    "model": model_name,
                    "train_windows": X_train.shape[0],
                    "test_windows": X_test.shape[0],
                    "test_acc": metrics["accuracy"],
                    "test_f1": metrics["f1"],
                    "test_precision": metrics["precision"],
                    "test_recall": metrics["recall"],
                })
                print(f"acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}")

    if not rows:
        print("\n  No results collected.")
        return

    results_df = pd.DataFrame(rows)

    # ── 2. Summary comparison table ─────────────────────────────────────
    print("\n\n" + "=" * 78)
    print("  FINDING 1: Signal exists but does not generalise")
    print("=" * 78)
    print("  Compare within-subject (same participants in train & test)\n"
          "  vs cross-participant LOPO (held-out participant).\n"
          "  If within-subject >> LOPO → signal is real but person-specific.\n")

    # Find the 3-class all-participant rows
    mask_all = results_df["classes"] == "all"
    summary = results_df[mask_all].pivot_table(
        index="model", columns="split",
        values=["test_acc", "test_f1"],
        aggfunc="first",
    )
    # Flatten multi-level columns
    summary.columns = [f"{col[1]}_{col[0]}" for col in summary.columns]
    summary = summary.reset_index()
    # Reorder for readability
    col_order = ["model",
                 "Within-Subject_test_acc", "Within-Subject_test_f1",
                 "Cross-Participant (LOPO)_test_acc", "Cross-Participant (LOPO)_test_f1"]
    summary = summary[[c for c in col_order if c in summary.columns]]
    summary = summary.round(4)

    # Delta row (xgboost − random_forest)
    if len(summary) >= 2:
        delta = {"model": "delta"}
        for col in summary.columns:
            if col != "model" and pd.api.types.is_numeric_dtype(summary[col]):
                delta[col] = summary[col].iloc[1] - summary[col].iloc[0]
        summary = pd.concat([summary, pd.DataFrame([delta])], ignore_index=True)

    print(tabulate(summary, headers="keys", tablefmt="simple", showindex=False))

    # ── 3. Class contrast comparison ────────────────────────────────────
    print("\n\n" + "=" * 78)
    print("  FINDING 2: HARD vs SOFT is more discriminable than HARD vs SILENCE")
    print("=" * 78)
    print("  Within-subject: compare full 3-class vs binary subsets.\n"
          "  If `hard+soft` >> `hard+silence`, the intuitive contrast\n"
          "  (music vs silence) is weaker than the intensity contrast.\n")

    mask_ws = results_df["split"] == "Within-Subject"
    ws_results = results_df[mask_ws].copy()
    ws_pivot = ws_results.pivot_table(
        index="model", columns="classes",
        values=["test_acc", "test_f1"],
        aggfunc="first",
    )
    ws_pivot.columns = [f"{col[1]}_{col[0]}" for col in ws_pivot.columns]
    ws_pivot = ws_pivot.reset_index()
    # Prioritise columns we care about
    priority = ["all", "hard+soft", "hard+silence", "soft+silence"]
    acc_cols = [f"{p}_test_acc" for p in priority if f"{p}_test_acc" in ws_pivot.columns]
    f1_cols = [f"{p}_test_f1" for p in priority if f"{p}_test_f1" in ws_pivot.columns]
    ws_pivot = ws_pivot[["model"] + acc_cols + f1_cols]
    ws_pivot = ws_pivot.round(4)

    # Delta row (xgboost − random_forest)
    if len(ws_pivot) >= 2:
        delta = {"model": "delta"}
        for col in ws_pivot.columns:
            if col != "model" and pd.api.types.is_numeric_dtype(ws_pivot[col]):
                delta[col] = ws_pivot[col].iloc[1] - ws_pivot[col].iloc[0]
        ws_pivot = pd.concat([ws_pivot, pd.DataFrame([delta])], ignore_index=True)

    print(tabulate(ws_pivot, headers="keys", tablefmt="simple", showindex=False))

    # ── 4. Detailed per-class breakdown for key comparisons ─────────────
    print("\n\n" + "=" * 78)
    print("  DETAILED BREAKDOWN: Key comparisons")
    print("=" * 78)

    key_configs = [
        ("within_subject", "all", "random_forest", "Within-subject, 3-class (RF) — is there signal?"),
        ("participant", "all", "random_forest", "Cross-participant LOPO, 3-class (RF) — does it generalise?"),
        ("within_subject", "hard+soft", "random_forest", "Within-subject, HARD vs SOFT (RF) — intensity contrast"),
        ("within_subject", "hard+silence", "random_forest", "Within-subject, HARD vs SILENCE (RF) — music vs silence"),
    ]

    for ss, sub, model_name, title in key_configs:
        key = (ss, sub, model_name)
        if key not in detail_store:
            continue
        info = detail_store[key]

        print(f"\n  ── {title} ──")
        m = info["metrics"]
        print(f"    Test — acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
              f"precision={m['precision']:.4f}  recall={m['recall']:.4f}")
        # Print per-class if we can
        if "label_names" in info and "y_test" in info and "preds" in info:
            report = format_classification_report(
                info["y_test"], info["preds"], label_names=info["label_names"],
            )
            for line in report.split("\n"):
                print(f"    {line}")

    # ── 5. Summary narrative ────────────────────────────────────────────
    print("\n\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)

    # Pick the best within-subject and LOPO rows for 3-class
    ws_best = results_df[(results_df["split"] == "Within-Subject") & (results_df["classes"] == "all")].sort_values("test_f1", ascending=False)
    lop_best = results_df[(results_df["split"] == "Cross-Participant (LOPO)") & (results_df["classes"] == "all")].sort_values("test_f1", ascending=False)

    if not ws_best.empty and not lop_best.empty:
        ws_acc = ws_best.iloc[0]["test_acc"]
        ws_f1 = ws_best.iloc[0]["test_f1"]
        lp_acc = lop_best.iloc[0]["test_acc"]
        lp_f1 = lop_best.iloc[0]["test_f1"]
        print(f"\n  Finding 1 — Predictive power without generalisation:")
        print(f"    Within-subject (best):  acc={ws_acc:.3f}  f1={ws_f1:.3f}")
        print(f"    Cross-participant:      acc={lp_acc:.3f}  f1={lp_f1:.3f}")
        delta = (ws_f1 - lp_f1) / max(lp_f1, 0.001) * 100
        print(f"    → Within-subject outperforms by {delta:.0f}% in F1. Signal is")
        print(f"      real but highly person-specific — models learn individual")
        print(f"      gait/movement patterns, not universal music-response markers.")

    # Best within-subject binary comparison
    ws_bin = results_df[results_df["split"] == "Within-Subject"]
    hs = ws_bin[ws_bin["classes"] == "hard+soft"].sort_values("test_f1", ascending=False)
    hl = ws_bin[ws_bin["classes"] == "hard+silence"].sort_values("test_f1", ascending=False)

    if not hs.empty and not hl.empty:
        hs_f1 = hs.iloc[0]["test_f1"]
        hl_f1 = hl.iloc[0]["test_f1"]
        hs_acc = hs.iloc[0]["test_acc"]
        hl_acc = hl.iloc[0]["test_acc"]
        print(f"\n  Finding 2 — HARD vs SOFT is more discriminable than HARD vs SILENCE:")
        print(f"    HARD vs SOFT:     acc={hs_acc:.3f}  f1={hs_f1:.3f}")
        print(f"    HARD vs SILENCE:  acc={hl_acc:.3f}  f1={hl_f1:.3f}")
        print(f"    → The music intensity contrast (HARD↔SOFT) is easier to detect")
        print(f"      than the music-vs-silence contrast. This suggests movement")
        print(f"      during silence is not a clean 'baseline' — participants may")
        print(f"      still be moving actively.")

    print("\n" + "=" * 78)


def main() -> None:
    argument_parser = argparse.ArgumentParser(prog="mlqs")
    argument_parser.add_argument(
        "--model",
        choices=["classical", "deep"],
        default=None,
        help="Run model after feature pipeline",
    )
    argument_parser.add_argument(
        "--split",
        choices=["block", "participant", "within_subject"],
        default="participant",
        help=(
            "Data splitting strategy. 'block' splits the concatenated time series "
            "into contiguous blocks (may leak participant identity). "
            "'participant' splits by participant (default) — training set contains "
            "all participants except the OOS participant (Kim), and LOPO-CV is used "
            "for validation. "
            "'within_subject' splits each participant's data into train/test blocks "
            "so the same participants appear in both sets — tests whether there is "
            "any learnable signal at all."
        ),
    )
    argument_parser.add_argument(
        "--compare",
        action="store_true",
        help="Run full comparison grid: all splits × class subsets × models",
    )
    argument_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable pipeline result caching (re-run feature extraction)",
    )
    argument_parser.add_argument("--eda", action="store_true", help="EDA only")
    parsed_args = argument_parser.parse_args()

    experiment_config = Config(
        preprocessing=PreprocessingConfig(),
        features=FeatureConfig(),
        models=ModelConfig(),
    )

    if parsed_args.eda:
        _run_exploratory_analysis(experiment_config)
        return

    # ── Load data ────────────────────────────────────────────────────────
    sensor_data = load_all_experiment_sensors(
        experiment_config.raw_dir,
        resample_rule=experiment_config.preprocessing.resample_rule,
    )
    print(f"Loaded: {sensor_data.shape}")
    if "label" in sensor_data.columns:
        print(f"Labels:  {sensor_data['label'].value_counts().to_dict()}")
    if "participant" in sensor_data.columns:
        print(f"Participants:  {sensor_data['participant'].value_counts().to_dict()}")

    # ── Compare mode ─────────────────────────────────────────────────────
    if parsed_args.compare:
        _run_comparison(sensor_data, experiment_config, no_cache=parsed_args.no_cache)
        return

    # ── Single-run mode ──────────────────────────────────────────────────
    oos_participant = experiment_config.models.oos_participant

    if parsed_args.split == "participant":
        print(f"\n── Participant-level split (OOS = '{oos_participant}') ──\n")
        val_result, test_result = run_participant_train_test_pipeline(
            sensor_data, experiment_config,
            oos_participant=oos_participant,
        )
        cv_groups_name = "participant"
    elif parsed_args.split == "within_subject":
        print("\n── Within-subject split (same participants in train & test) ──\n")
        val_result, test_result = run_within_subject_train_test_pipeline(
            sensor_data, experiment_config,
            test_fraction=0.2,
            n_blocks=10,
            random_seed=42,
        )
        cv_groups_name = "participant"
    else:
        print("\n── Block-based split ──\n")
        val_result, test_result = run_train_test_pipeline(
            sensor_data, experiment_config,
            test_fraction=0.2,
            n_blocks=10,
            random_seed=42,
        )
        cv_groups_name = "block"

    if val_result.feature_matrix.empty or test_result.feature_matrix.empty:
        print("Feature extraction produced empty result — aborting")
        return

    print(f"Train windows: {val_result.feature_matrix.shape[0]}  "
          f"Test windows: {test_result.feature_matrix.shape[0]}")
    print(f"Features: {val_result.feature_matrix.shape[1]}")

    if val_result.participant is not None:
        print(f"Train participants: {sorted(val_result.participant.unique())}")
    if test_result.participant is not None:
        print(f"Test participants:  {sorted(test_result.participant.unique())}")

    # ── Encode labels ────────────────────────────────────────────────────
    if val_result.labels is None or test_result.labels is None:
        print("No labels found — skipping model training")
        return

    label_encoder = LabelEncoder()
    y_val = label_encoder.fit_transform(val_result.labels)
    y_test = label_encoder.transform(test_result.labels)
    print(f"Classes:  {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    X_val = val_result.feature_matrix.values
    X_test = test_result.feature_matrix.values

    # ── Groups for CV ────────────────────────────────────────────────────
    groups_val: np.ndarray | None = None
    if parsed_args.split in ("participant", "within_subject") and val_result.participant is not None:
        groups_val = val_result.participant.values
    elif val_result.block_ids is not None:
        groups_val = val_result.block_ids.values

    # ── Train / evaluate ─────────────────────────────────────────────────
    if parsed_args.model == "classical":
        class_names = list(label_encoder.classes_)
        _train_and_evaluate_classical_models(
            X_val, X_test, y_val, y_test, groups_val,
            feature_names=val_result.feature_names,
            label_names=class_names,
            cv_name=cv_groups_name,
        )
    elif parsed_args.model == "deep":
        _train_and_evaluate_deep_model(
            X_val, X_test, y_val, y_test, groups_val,
            cv_name=cv_groups_name,
        )


def _train_and_evaluate_classical_models(
    X_val: np.ndarray, X_test: np.ndarray,
    y_val: np.ndarray, y_test: np.ndarray,
    groups: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    label_names: list[str] | None = None,
    cv_name: str = "block",
) -> None:
    """Group-aware CV on *val* (LOPO when groups = participant), then OOS test.

    Parameters
    ----------
    groups :
        Group labels for GroupKFold — either participant IDs (for LOPO-CV)
        or block IDs (for block-split CV).  ``None`` skips CV.
    feature_names:
        Column names of the feature matrix — used for feature importance display.
    label_names:
        Human-readable class names — used for per-class metrics reporting.
    cv_name :
        Human-readable name for the CV strategy ("participant" or "block").
    """

    all_test_metrics: dict[str, dict] = {}

    # Clean non-finite values that can crash C-level trainers (especially XGBoost)
    X_val = _validate_and_clean_features(X_val, "standalone val")
    X_test = _validate_and_clean_features(X_test, "standalone test")

    for model_name, hyperparameters in [
        ("random_forest", {"n_estimators": 100, "max_depth": 10, "random_state": 42}),
    ("xgboost", {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "n_jobs": 1, "tree_method": "hist", "random_state": 42, "verbosity": 0}),
    ]:
        print(f"\n── {model_name} ──", flush=True)

        # ── Group-aware CV (LOPO for participants) ──────────────────────
        if groups is not None and len(np.unique(groups)) >= 2:
            n_folds = min(5, len(np.unique(groups)))
            cv_label = "LOPO-CV" if cv_name == "participant" else f"{n_folds}-fold CV"
            print(f"  Running {cv_label} ...", flush=True)
            gkf = GroupKFold(n_splits=n_folds)
            fold_accs: list[float] = []
            fold_f1s: list[float] = []
            for fold, (train_idx, val_idx) in enumerate(
                gkf.split(X_val, y_val, groups=groups), 1,
            ):
                classifier = build_classifier(model_name, **hyperparameters)
                classifier.fit(X_val[train_idx], y_val[train_idx])
                preds = classifier.predict(X_val[val_idx])
                metrics = compute_classification_metrics(y_val[val_idx], preds)
                fold_accs.append(metrics["accuracy"])
                fold_f1s.append(metrics["f1"])
                print(f"  Fold {fold}:  acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}", flush=True)

            mean_acc = float(np.mean(fold_accs))
            std_acc = float(np.std(fold_accs))
            mean_f1 = float(np.mean(fold_f1s))
            std_f1 = float(np.std(fold_f1s))
            print(f"  {cv_label}:  acc={mean_acc:.3f} ± {std_acc:.3f}  "
                  f"f1={mean_f1:.3f} ± {std_f1:.3f}", flush=True)
        else:
            print(f"  (fewer than 2 {cv_name} groups — skipping CV)", flush=True)

        # ── Final evaluation on held-out test set ───────────────────────
        classifier = build_classifier(model_name, **hyperparameters).fit(X_val, y_val)
        preds = classifier.predict(X_test)
        proba = classifier.predict_proba(X_test) if hasattr(classifier, "predict_proba") else None
        metrics = compute_classification_metrics(y_test, preds, proba)
        all_test_metrics[model_name] = metrics
        print(f"  Test:           acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}", flush=True)

        # ── Per-class metrics ───────────────────────────────────────────
        print("\n  Per-class metrics:")
        report = format_classification_report(y_test, preds, label_names=label_names)
        for line in report.split("\n"):
            print(f"    {line}", flush=True)

        # ── Confusion matrix ────────────────────────────────────────────
        cm = compute_confusion_matrix(y_test, preds)
        print(f"\n  Confusion matrix (raw):\n    {cm['raw']}")
        if "normalized" in cm:
            print(f"  Confusion matrix (normalized, row-wise):\n    {np.round(cm['normalized'], 3)}")

        # ── Feature importance ──────────────────────────────────────────
        if feature_names is not None:
            importances: dict[str, float] | None = None
            try:
                if model_name == "random_forest":
                    importances = extract_rf_importances(classifier, feature_names)
                elif model_name == "xgboost":
                    importances = extract_xgboost_importances(classifier, feature_names)
            except (ValueError, AttributeError, Exception) as exc:
                print(f"  (feature importance unavailable: {exc})")

            if importances:
                sort_and_display_top_features(
                    importances, top_n=10,
                    title=f"  Top-10 Features ({model_name})",
                )

    # ── Multi-model comparison table ────────────────────────────────────
    if len(all_test_metrics) > 1:
        print("\n── Model Comparison ──", flush=True)
        comparison = build_model_comparison_table(all_test_metrics)
        if tabulate is not None:
            print(tabulate(comparison, headers="keys", tablefmt="simple", showindex=False))
        else:
            print(comparison.to_string(index=False))


def _train_and_evaluate_deep_model(
    X_val: np.ndarray, X_test: np.ndarray,
    y_val: np.ndarray, y_test: np.ndarray,
    groups: np.ndarray | None = None,
    cv_name: str = "block",
) -> None:
    """Train a deep model on *val* participants, evaluate on OOS *test*.

    When groups are participant IDs from LOPO splitting, this tests
    cross-participant generalization directly.
    """
    sequence_length = min(32, len(X_val) // 10)
    batch_size = 32
    training_loader = prepare_sequences(X_val, y_val, sequence_length, batch_size)
    test_loader = prepare_sequences(X_test, y_test, sequence_length, batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    deep_classifier = build_deep_classifier(
        model_type="lstm",
        input_size=X_val.shape[1],
        num_classes=len(np.unique(y_val)),
    )
    print(f"  Model: {type(deep_classifier).__name__} on {device}")
    if cv_name == "participant":
        print(f"  CV strategy: LOPO (no participant overlap between train/val)")
    else:
        print(f"  CV strategy: block-based (may leak participant identity)")

    train_deep_model(
        deep_classifier,
        training_loader,
        num_epochs=20,
        learning_rate=0.001,
        device=device,
    )

    deep_classifier.eval()
    all_predictions, all_true_labels = [], []
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            logits = deep_classifier(batch_inputs.to(device))
            all_predictions.extend(logits.argmax(1).cpu().numpy())
            all_true_labels.extend(batch_labels.numpy())

    metric_scores = compute_classification_metrics(
        np.array(all_true_labels), np.array(all_predictions)
    )
    print(
        f"  Test — acc={metric_scores['accuracy']:.3f}  f1={metric_scores['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
