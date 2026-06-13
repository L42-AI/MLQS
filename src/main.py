#!/usr/bin/env python
"""Train and evaluate models on phyphox sensor data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import Config
from data.loader import compute_sensor_summary, detect_data_quality_issues, load_all_experiment_sensors
from models.classical import build_classifier, split_train_test_data
from models.evaluation import compute_classification_metrics, rank_model_performances
from pipeline.builder import run_feature_pipeline


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


def _run_feature_extraction_pipeline(
    experiment_config: Config,
) -> tuple[np.ndarray, np.ndarray | None]:
    sensor_data = load_all_experiment_sensors(
        experiment_config.raw_dir,
        resample_rule=experiment_config.preprocessing.resample_rule,
    )
    print(f"Loaded: {sensor_data.shape}")
    if "label" in sensor_data.columns:
        print(f"Labels:  {sensor_data['label'].value_counts().to_dict()}")

    pipeline_result = run_feature_pipeline(sensor_data, experiment_config)
    print(f"Windows:  {pipeline_result.feature_matrix.shape[0]}")
    print(f"Features: {pipeline_result.feature_matrix.shape[1]}")

    extracted_labels = pipeline_result.labels
    if extracted_labels is not None:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(extracted_labels)
        print(
            f"Classes:  {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
        )
        return pipeline_result.feature_matrix.values, encoded_labels
    return pipeline_result.feature_matrix.values, None


def _train_and_evaluate_classical_models(
    feature_matrix: np.ndarray, encoded_labels: np.ndarray
) -> None:
    X_train, X_test, y_train, y_test = split_train_test_data(
        feature_matrix, encoded_labels, test_fraction=0.2
    )
    all_model_results = {}

    for model_name, hyperparameters in [
        ("random_forest", {"n_estimators": 100, "max_depth": 10, "random_state": 42}),
        ("svm", {"kernel": "rbf", "C": 1.0, "gamma": "scale", "random_state": 42}),
    ]:
        classifier = build_classifier(model_name, **hyperparameters).fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        prediction_probabilities = (
            classifier.predict_proba(X_test)
            if hasattr(classifier, "predict_proba")
            else None
        )
        all_model_results[model_name] = compute_classification_metrics(
            y_test, predictions, prediction_probabilities
        )
        print(
            f"  {model_name:15s}  acc={all_model_results[model_name]['accuracy']:.3f}  "
            f"f1={all_model_results[model_name]['f1']:.3f}"
        )

    print("\nComparison:")
    print(rank_model_performances(all_model_results).to_string())


def _train_and_evaluate_deep_model(
    feature_matrix: np.ndarray, encoded_labels: np.ndarray
) -> None:
    try:
        import torch
    except ImportError:
        print("Install torch: pip install torch")
        return

    from models.deep import build_deep_classifier, prepare_sequences, train_deep_model

    X_train, X_test, y_train, y_test = split_train_test_data(
        feature_matrix, encoded_labels, test_fraction=0.2
    )

    sequence_length = min(32, len(X_train) // 10)
    batch_size = 32
    training_loader = prepare_sequences(X_train, y_train, sequence_length, batch_size)
    test_loader = prepare_sequences(X_test, y_test, sequence_length, batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    deep_classifier = build_deep_classifier(
        model_type="lstm",
        input_size=feature_matrix.shape[1],
        num_classes=len(np.unique(encoded_labels)),
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

    experiment_config = Config()

    if parsed_args.eda:
        _run_exploratory_analysis(experiment_config)
        return

    feature_matrix, encoded_labels = _run_feature_extraction_pipeline(experiment_config)
    if encoded_labels is None:
        print("No labels found — skipping model training")
        return

    if parsed_args.model == "classical":
        _train_and_evaluate_classical_models(feature_matrix, encoded_labels)
    elif parsed_args.model == "deep":
        _train_and_evaluate_deep_model(feature_matrix, encoded_labels)


if __name__ == "__main__":
    main()
