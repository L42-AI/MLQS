"""Train and evaluate classical ML models (RF, SVM, XGBoost)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
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


def split_train_test_data(features, labels, test_fraction=0.2, stratify=True, groups=None):
    if groups is not None and len(np.unique(groups)) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=42)
        train_idx, test_idx = next(splitter.split(features, labels, groups=groups))
        return features[train_idx], features[test_idx], labels[train_idx], labels[test_idx]

    stratify_labels = (
        labels
        if stratify and labels.dtype in ("object", "category")
        else None
    )
    return train_test_split(
        features, labels, test_size=test_fraction, random_state=42, stratify=stratify_labels
    )
