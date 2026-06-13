"""Plotting helpers for EDA, features, and model evaluation."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from preprocessing.missing import build_missing_value_report


def plot_sensor_time_series(
    sensor_data: pd.DataFrame,
    columns: list[str] | None = None,
    time_column: str = "time",
    title: str = "Sensor Time Series",
    axes: Axes | None = None,
) -> Axes:
    if axes is None:
        _, axes = plt.subplots(figsize=(14, 6))
    time_values = sensor_data[time_column] if time_column in sensor_data.columns else sensor_data.index
    for column in columns or [c for c in sensor_data.columns if c != time_column]:
        if column in sensor_data.columns:
            axes.plot(time_values, sensor_data[column], label=column, alpha=0.8)
    axes.set(xlabel="Time (s)", ylabel="Amplitude", title=title)
    axes.legend()
    axes.grid(True, alpha=0.3)
    return axes


def plot_missing_value_barchart(sensor_data: pd.DataFrame, axes: Axes | None = None) -> Axes:
    if axes is None:
        _, axes = plt.subplots(figsize=(12, 4))
    missing_report_data = build_missing_value_report(sensor_data)
    if missing_report_data.empty:
        axes.text(
            0.5,
            0.5,
            "No missing values",
            ha="center",
            va="center",
            transform=axes.transAxes,
        )
        return axes
    axes.bar(
        missing_report_data["column"],
        missing_report_data["n_missing"],
        color="coral",
        edgecolor="black",
    )
    axes.set(xlabel="Column", ylabel="Missing count", title="Missing Values")
    axes.tick_params(axis="x", rotation=45)
    axes.grid(True, alpha=0.3, axis="y")
    return axes


def plot_correlation_heatmap(feature_matrix: pd.DataFrame, axes: Axes | None = None) -> Axes:
    correlation_matrix = feature_matrix.select_dtypes(include=[np.number]).corr()
    if axes is None:
        _, axes = plt.subplots(figsize=(14, 12))
    heatmap_image = axes.imshow(
        correlation_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
    )
    axes.set(
        xticks=range(len(correlation_matrix.columns)),
        yticks=range(len(correlation_matrix.columns)),
        xticklabels=correlation_matrix.columns,
        yticklabels=correlation_matrix.columns,
        title="Feature Correlation Matrix",
    )
    plt.setp(axes.get_xticklabels(), fontsize=6, rotation=90)
    plt.setp(axes.get_yticklabels(), fontsize=6)
    plt.colorbar(heatmap_image, ax=axes, shrink=0.8)
    return axes


def plot_confusion_matrix(
    confusion_matrix_values: pd.DataFrame | np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    axes: Axes | None = None,
) -> Axes:
    if axes is None:
        _, axes = plt.subplots(figsize=(6, 5))
    matrix_values = (
        confusion_matrix_values.values
        if isinstance(confusion_matrix_values, pd.DataFrame)
        else confusion_matrix_values
    )
    class_labels = class_names or [str(i) for i in range(len(matrix_values))]
    heatmap_image = axes.imshow(matrix_values, cmap="Blues", aspect="auto")
    plt.colorbar(heatmap_image, ax=axes)
    for row_index in range(len(matrix_values)):
        for col_index in range(len(matrix_values)):
            axes.text(
                col_index,
                row_index,
                str(matrix_values[row_index, col_index]),
                ha="center",
                va="center",
                color="white" if matrix_values[row_index, col_index] > matrix_values.max() / 2 else "black",
            )
    axes.set(
        xticks=range(len(class_labels)),
        yticks=range(len(class_labels)),
        xticklabels=class_labels,
        yticklabels=class_labels,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    return axes


def plot_feature_importance(
    importance_scores: dict[str, float],
    top_k: int = 20,
    title: str = "Feature Importance",
    axes: Axes | None = None,
) -> Axes:
    if axes is None:
        _, axes = plt.subplots(figsize=(10, 6))
    ranked_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    feature_names, feature_scores = zip(*ranked_features) if ranked_features else ([], [])
    axes.barh(range(len(feature_names)), feature_scores, color="teal", edgecolor="black")
    axes.set(
        yticks=range(len(feature_names)),
        yticklabels=feature_names,
        xlabel="Importance",
        title=title,
    )
    axes.invert_yaxis()
    axes.grid(True, alpha=0.3, axis="x")
    return axes


def plot_training_history(training_history, figure_size=(12, 4)):
    figure, axes_list = plt.subplots(1, 2, figsize=figure_size)
    axes_list[0].plot(
        training_history.training_losses, label="Train Loss", color="steelblue"
    )
    if hasattr(training_history, "validation_losses") and training_history.validation_losses:
        axes_list[0].plot(
            training_history.validation_losses, label="Val Loss", color="coral"
        )
    axes_list[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes_list[0].legend()
    axes_list[0].grid(True, alpha=0.3)

    if (
        hasattr(training_history, "validation_accuracies")
        and training_history.validation_accuracies
    ):
        axes_list[1].plot(
            training_history.validation_accuracies, label="Val Accuracy", color="green"
        )
        axes_list[1].set(
            xlabel="Epoch", ylabel="Accuracy", title="Validation Accuracy"
        )
        axes_list[1].legend()
        axes_list[1].grid(True, alpha=0.3)
    else:
        axes_list[1].set_visible(False)

    figure.tight_layout()
    return figure, axes_list
