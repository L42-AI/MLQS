#!/usr/bin/env python
"""Train and evaluate models on phyphox sensor data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

import numpy as np
import torch
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
    compute_permutation_importance,
    extract_rf_importances,
    extract_xgboost_importances,
    sort_and_display_top_features,
)
from pipeline.builder import run_train_test_pipeline




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


def main() -> None:
    argument_parser = argparse.ArgumentParser(prog="mlqs")
    argument_parser.add_argument(
        "--model",
        choices=["classical", "deep"],
        default=None,
        help="Run model after feature pipeline",
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

    # ── Block-split pipeline ─────────────────────────────────────────────
    # No overlapping windows — each block is windowed independently.
    # test blocks (20%) are held out for final evaluation only.
    val_result, test_result = run_train_test_pipeline(
        sensor_data, experiment_config,
        test_fraction=0.2,
        n_blocks=10,
        random_seed=42,
    )

    if val_result.feature_matrix.empty or test_result.feature_matrix.empty:
        print("Feature extraction produced empty result — aborting")
        return

    print(f"CV windows: {val_result.feature_matrix.shape[0]}  "
          f"Test windows: {test_result.feature_matrix.shape[0]}")
    print(f"Features: {val_result.feature_matrix.shape[1]}")

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
    block_ids_val = val_result.block_ids.values if val_result.block_ids is not None else None

    # ── Train / evaluate ─────────────────────────────────────────────────
    if parsed_args.model == "classical":
        class_names = list(label_encoder.classes_)
        _train_and_evaluate_classical_models(
            X_val, X_test, y_val, y_test, block_ids_val,
            feature_names=val_result.feature_names,
            label_names=class_names,
        )
    elif parsed_args.model == "deep":
        _train_and_evaluate_deep_model(
            X_val, X_test, y_val, y_test, block_ids_val,
        )


def _train_and_evaluate_classical_models(
    X_val: np.ndarray, X_test: np.ndarray,
    y_val: np.ndarray, y_test: np.ndarray,
    block_ids: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    label_names: list[str] | None = None,
) -> None:
    """k-fold block CV on *val* blocks, then final eval on held-out *test*.

    Parameters
    ----------
    feature_names:
        Column names of the feature matrix — used for feature importance display.
    label_names:
        Human-readable class names — used for per-class metrics reporting.
    """

    all_test_metrics: dict[str, dict] = {}

    for model_name, hyperparameters in [
        ("random_forest", {"n_estimators": 100, "max_depth": 10, "random_state": 42}),
        ("svm", {"kernel": "rbf", "C": 1.0, "gamma": "scale", "random_state": 42}),
        ("xgboost", {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "nthread": 1, "random_state": 42, "verbosity": 0}),
    ]:
        print(f"\n── {model_name} ──", flush=True)

        # ── k-fold block CV ─────────────────────────────────────────────
        if block_ids is not None and len(np.unique(block_ids)) >= 2:
            n_folds = min(5, len(np.unique(block_ids)))
            print(f"  Running {n_folds}-fold CV ...", flush=True)
            gkf = GroupKFold(n_splits=n_folds)
            fold_accs: list[float] = []
            fold_f1s: list[float] = []
            for fold, (train_idx, val_idx) in enumerate(
                gkf.split(X_val, y_val, groups=block_ids), 1,
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
            print(f"  CV ({n_folds}-fold):  acc={mean_acc:.3f} ± {std_acc:.3f}  "
                  f"f1={mean_f1:.3f} ± {std_f1:.3f}", flush=True)
        else:
            print("  (fewer than 2 blocks — skipping CV)", flush=True)

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
                else:
                    # SVM via permutation importance
                    imp = compute_permutation_importance(
                        classifier, X_test, y_test, feature_names, n_repeats=5,
                    )
                    importances = {k: v[0] for k, v in imp.items()}
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
    block_ids: np.ndarray | None = None,
) -> None:
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
