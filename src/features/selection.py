"""Feature selection — reduce dimensionality before modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    SelectFromModel,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def drop_low_variance_features(
    feature_matrix: pd.DataFrame,
    labels: pd.Series | None = None,
    variance_threshold: float = 0.01,
) -> pd.DataFrame:
    selector = VarianceThreshold(threshold=variance_threshold)
    kept_indices = selector.fit(
        feature_matrix.select_dtypes(include=[np.number])
    ).get_support(indices=True)
    return feature_matrix.iloc[:, kept_indices]


def drop_highly_correlated_features(
    feature_matrix: pd.DataFrame,
    labels: pd.Series | None = None,
    correlation_threshold: float = 0.95,
) -> pd.DataFrame:
    numeric_features = feature_matrix.select_dtypes(include=[np.number])
    correlation_matrix = numeric_features.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    features_to_drop: set[str] = set()
    for column in upper_triangle.columns:
        correlated_partners = upper_triangle[column][
            upper_triangle[column] >= correlation_threshold
        ]
        for partner in correlated_partners.index:
            column_has_lower_variance = (
                numeric_features[column].var() < numeric_features[partner].var()
            )
            # Drop the higher-variance feature — keep the more stable one
            feature_to_drop = column if column_has_lower_variance else partner
            features_to_drop.add(feature_to_drop)

    return feature_matrix.drop(columns=list(features_to_drop), errors="ignore")


def select_top_by_mutual_information(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    top_k: int = 20,
) -> pd.DataFrame:
    numeric_features = feature_matrix.select_dtypes(include=[np.number])
    encoded_labels = (
        LabelEncoder().fit_transform(labels) if labels.dtype == "object" else labels.values
    )
    top_k = min(top_k, numeric_features.shape[1])
    selector = SelectKBest(mutual_info_classif, k=top_k).fit(numeric_features, encoded_labels)
    return feature_matrix.iloc[:, selector.get_support(indices=True)]


def select_by_recursive_elimination(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    top_k: int = 10,
) -> pd.DataFrame:
    numeric_features = feature_matrix.select_dtypes(include=[np.number])
    top_k = min(top_k, numeric_features.shape[1])
    selector = RFE(
        RandomForestClassifier(n_estimators=50, random_state=42),
        n_features_to_select=top_k,
    )
    selector.fit(numeric_features, labels)
    return feature_matrix.iloc[:, selector.get_support(indices=True)]


def select_by_l1_regularization(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    regularisation_strength: float = 0.1,
) -> pd.DataFrame:
    numeric_features = feature_matrix.select_dtypes(include=[np.number])
    encoded_labels = (
        LabelEncoder().fit_transform(labels) if labels.dtype == "object" else labels.values
    )
    estimator = LogisticRegression(
        penalty="l1",
        C=regularisation_strength,
        solver="saga",
        max_iter=5000,
        random_state=42,
    )
    selector = SelectFromModel(estimator.fit(numeric_features, encoded_labels), prefit=True)
    return feature_matrix.iloc[:, selector.get_support(indices=True)]


SELECTION_METHOD_NAME_TO_FUNCTION = {
    "variance": drop_low_variance_features,
    "correlation": drop_highly_correlated_features,
    "mutual_information": select_top_by_mutual_information,
    "rfe": select_by_recursive_elimination,
    "l1": select_by_l1_regularization,
}


def run_selection_pipeline(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    selection_methods: list[str],
) -> pd.DataFrame:
    reduced_features = feature_matrix.copy()
    for method_name in selection_methods:
        selector_fn = SELECTION_METHOD_NAME_TO_FUNCTION.get(method_name)
        if selector_fn is not None:
            reduced_features = selector_fn(reduced_features, labels)
    return reduced_features
