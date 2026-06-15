"""Evaluation metrics for model comparison."""

import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
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


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    """Compute per-class precision, recall, f1, and support.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        label_names: Optional list of class names for labeling the output dicts.

    Returns:
        dict with keys "precision", "recall", "f1", "support", each containing
        a list of per-class values. If label_names is provided, keys are class
        names instead of integer indices.
    """
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Per-class support (number of true instances per class)
    classes = np.unique(y_true)
    support = np.array([np.sum(y_true == c) for c in classes], dtype=int)

    if label_names is not None:
        precision = {name: float(precision[i]) for i, name in enumerate(label_names)}
        recall = {name: float(recall[i]) for i, name in enumerate(label_names)}
        f1 = {name: float(f1[i]) for i, name in enumerate(label_names)}
        support = {name: int(support[i]) for i, name in enumerate(label_names)}
    else:
        precision = [float(v) for v in precision]
        recall = [float(v) for v in recall]
        f1 = [float(v) for v in f1]
        support = [int(v) for v in support]

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
) -> dict:
    """Compute raw and optionally normalized confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        normalize: If True, include row-wise normalized matrix.

    Returns:
        dict with keys "raw" (ndarray) and optionally "normalized" (ndarray).
    """
    cm = confusion_matrix(y_true, y_pred)
    result: dict = {"raw": cm}
    if normalize:
        with np.errstate(all="ignore"):
            cm_normalized = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
            cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)
        result["normalized"] = cm_normalized
    return result


def format_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> str:
    """Format a classification report string similar to sklearn's output.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        label_names: Optional list of class names.

    Returns:
        Formatted classification report string.
    """
    per_class = compute_per_class_metrics(y_true, y_pred, label_names=None)
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)

    if label_names is not None and len(label_names) == n_classes:
        class_labels = label_names
    else:
        class_labels = [str(c) for c in unique_classes]

    precisions = per_class["precision"]
    recalls = per_class["recall"]
    f1_scores = per_class["f1"]
    supports = per_class["support"]

    total_support = sum(supports)

    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Compute macro averages
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)

    # Compute weighted averages
    weighted_precision = np.average(precisions, weights=supports)
    weighted_recall = np.average(recalls, weights=supports)
    weighted_f1 = np.average(f1_scores, weights=supports)

    headers = ["class", "precision", "recall", "f1", "support"]

    if tabulate is not None:
        rows = []
        for i in range(n_classes):
            rows.append([
                class_labels[i],
                f"{precisions[i]:.4f}",
                f"{recalls[i]:.4f}",
                f"{f1_scores[i]:.4f}",
                int(supports[i]),
            ])
        rows.append(["accuracy", "", "", f"{accuracy:.4f}", int(total_support)])
        rows.append(["macro avg", f"{macro_precision:.4f}", f"{macro_recall:.4f}", f"{macro_f1:.4f}", int(total_support)])
        rows.append(["weighted avg", f"{weighted_precision:.4f}", f"{weighted_recall:.4f}", f"{weighted_f1:.4f}", int(total_support)])

        table = tabulate(rows, headers=headers, tablefmt="simple")
        return "Classification Report\n" + table
    else:
        # Manual formatting fallback
        col_widths = {
            "class": max(len("class"), max(len(str(l)) for l in class_labels)),
            "precision": 11,
            "recall": 10,
            "f1": 9,
            "support": 9,
        }
        col_widths["class"] = max(col_widths["class"], 14)

        header = (
            f"{'':>{col_widths['class']}}  "
            f"{'precision':>{col_widths['precision']}}  "
            f"{'recall':>{col_widths['recall']}}  "
            f"{'f1':>{col_widths['f1']}}  "
            f"{'support':>{col_widths['support']}}"
        )
        sep = "-" * len(header)
        lines = ["", "Classification Report", sep, header, sep]

        for i in range(n_classes):
            lines.append(
                f"{class_labels[i]:>{col_widths['class']}}  "
                f"{precisions[i]:>{col_widths['precision']}.4f}  "
                f"{recalls[i]:>{col_widths['recall']}.4f}  "
                f"{f1_scores[i]:>{col_widths['f1']}.4f}  "
                f"{supports[i]:>{col_widths['support']}}"
            )

        lines.append(sep)
        lines.append(
            f"{'accuracy':>{col_widths['class']}}  "
            f"{'':>{col_widths['precision']}}  "
            f"{'':>{col_widths['recall']}}  "
            f"{accuracy:>{col_widths['f1']}.4f}  "
            f"{total_support:>{col_widths['support']}}"
        )
        lines.append(
            f"{'macro avg':>{col_widths['class']}}  "
            f"{macro_precision:>{col_widths['precision']}.4f}  "
            f"{macro_recall:>{col_widths['recall']}.4f}  "
            f"{macro_f1:>{col_widths['f1']}.4f}  "
            f"{total_support:>{col_widths['support']}}"
        )
        lines.append(
            f"{'weighted avg':>{col_widths['class']}}  "
            f"{weighted_precision:>{col_widths['precision']}.4f}  "
            f"{weighted_recall:>{col_widths['recall']}.4f}  "
            f"{weighted_f1:>{col_widths['f1']}.4f}  "
            f"{total_support:>{col_widths['support']}}"
        )
        lines.append(sep)
        return "\n".join(lines)


def build_model_comparison_table(
    model_results: dict[str, dict],
) -> pd.DataFrame:
    """Build a comparison DataFrame from multiple model results.

    Args:
        model_results: dict of {model_name: {metric: value, ...}}.
            Expected metric keys: accuracy, balanced_accuracy, f1, precision,
            recall, mcc, roc_auc.

    Returns:
        pd.DataFrame with one row per model, sorted by f1 descending,
        values rounded to 4 decimal places.
    """
    ordered_columns = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "mcc",
        "roc_auc",
    ]
    records = []
    for model_name, metrics in model_results.items():
        record = {"model": model_name}
        for col in ordered_columns:
            record[col] = metrics.get(col, None)
        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values("f1", ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    df = df.round(4)
    return df
