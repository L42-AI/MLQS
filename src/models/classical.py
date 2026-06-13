"""Train and evaluate classical ML models (RF, SVM, XGBoost)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier


def build_classifier(model_name: str, **hyperparameters):
    match model_name:
        case "random_forest":
            return RandomForestClassifier(**hyperparameters)
        case "svm":
            return SVC(**hyperparameters, probability=True)
        case "xgboost":
            return XGBClassifier(**hyperparameters)
        case _:
            raise ValueError(f"Unknown model: {model_name}")


def split_train_test_data(features, labels, test_fraction=0.2, stratify=True):
    stratify_labels = (
        labels
        if stratify and labels.dtype in ("object", "category")
        else None
    )
    return train_test_split(
        features, labels, test_size=test_fraction, random_state=42, stratify=stratify_labels
    )
