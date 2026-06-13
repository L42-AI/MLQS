"""Evaluation metrics for model comparison."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    true_labels, predicted_labels, prediction_probabilities=None
) -> dict[str, float]:
    metric_scores = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels),
        "f1": f1_score(true_labels, predicted_labels, average="weighted", zero_division=0),
        "precision": precision_score(true_labels, predicted_labels, average="weighted", zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(true_labels, predicted_labels),
    }
    if prediction_probabilities is not None and len(np.unique(true_labels)) == 2:
        try:
            metric_scores["roc_auc"] = roc_auc_score(
                true_labels, prediction_probabilities[:, 1]
            )
        except Exception:
            pass
    return metric_scores


def rank_model_performances(model_results: dict[str, dict]) -> pd.DataFrame:
    results_dataframe = pd.DataFrame(model_results).T
    if "f1" in results_dataframe.columns:
        results_dataframe = results_dataframe.sort_values("f1", ascending=False)
    return results_dataframe.round(4)
