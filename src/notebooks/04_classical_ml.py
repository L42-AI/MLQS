#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import Config
from models.classical import build_classifier, split_train_test_data
from models.evaluation import compute_classification_metrics, rank_model_performances

experiment_config = Config()

feature_data = pd.read_csv(experiment_config.features_dir / "features.csv")
print(f"Loaded: {feature_data.shape}")

encoded_labels = (
    feature_data.pop("label") if "label" in feature_data.columns else feature_data.pop(feature_data.columns[-1])
)
feature_matrix = feature_data

if encoded_labels.dtype == "object":
    encoded_labels = LabelEncoder().fit_transform(encoded_labels)

X_train, X_test, y_train, y_test = split_train_test_data(
    feature_matrix, encoded_labels, test_fraction=experiment_config.models.test_size
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

all_model_results = {}
for model_name, hyperparameters in [
    ("random_forest", {"n_estimators": 100, "max_depth": 10, "random_state": 42}),
    ("xgboost", {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42}),
]:
    print(f"\nTraining: {model_name}")
    try:
        classifier = build_classifier(model_name, **hyperparameters).fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        prediction_probabilities = (
            classifier.predict_proba(X_test) if hasattr(classifier, "predict_proba") else None
        )
        all_model_results[model_name] = compute_classification_metrics(
            y_test, predictions, prediction_probabilities
        )
        print(f"  Acc: {all_model_results[model_name]['accuracy']:.4f}  F1: {all_model_results[model_name]['f1']:.4f}")
    except Exception as error:
        print(f"  Failed: {error}")

print("\n" + rank_model_performances(all_model_results).to_string())
