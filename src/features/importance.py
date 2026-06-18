from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


def extract_rf_importances(
    classifier,
    feature_names: list[str],
) -> dict[str, float]:
    """Extract feature importances from a trained RandomForestClassifier.

    Parameters
    ----------
    classifier : RandomForestClassifier
        A trained RandomForestClassifier (or any model with ``feature_importances_``
        or ``coef_``).
    feature_names : list[str]
        List of feature names corresponding to the model's features.

    Returns
    -------
    dict[str, float]
        Mapping of feature name to importance score, sorted by importance descending.

    Raises
    ------
    ValueError
        If the classifier has neither ``feature_importances_`` nor ``coef_``.
    """
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_).flatten()
    else:
        raise ValueError(
            "The classifier does not have 'feature_importances_' or 'coef_' "
            "attributes. Ensure the model is trained and supports feature "
            "importance extraction."
        )

    result = dict(zip(feature_names, importances))
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


def extract_xgboost_importances(
    classifier,
    feature_names: list[str],
    importance_type: str = "gain",
) -> dict[str, float]:
    """Extract feature importances from a trained XGBClassifier.

    Parameters
    ----------
    classifier : XGBClassifier
        A trained XGBClassifier instance.
    feature_names : list[str]
        List of feature names corresponding to the model's features.
    importance_type : str, default="gain"
        The type of importance to use when falling back to
        ``get_booster().get_score()``. Common values are ``"gain"``,
        ``"weight"``, ``"cover"``, ``"total_gain"``, and ``"total_cover"``.

    Returns
    -------
    dict[str, float]
        Mapping of feature name to importance score, sorted by importance descending.
    """
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    else:
        booster = classifier.get_booster()
        score_dict = booster.get_score(importance_type=importance_type)
        # Map f0, f1, ... indices to provided feature names
        importances = np.array([
            score_dict.get(f"f{i}", 0.0) for i in range(len(feature_names))
        ])

    result = dict(zip(feature_names, importances))
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


def sort_and_display_top_features(
    importances: dict[str, float | tuple[float, float]],
    top_n: int = 10,
    title: str = "Top Features",
) -> pd.DataFrame:
    """Sort, display, and return the top-N most important features.

    Parameters
    ----------
    importances : dict[str, float | tuple[float, float]]
        Dictionary mapping feature names to either a single importance score
        (from RF/XGBoost) or a ``(mean, std)`` tuple (from permutation
        importance).
    top_n : int, default=10
        Number of top features to keep.
    title : str, default="Top Features"
        Title printed above the table.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``feature`` and ``importance``, sorted by
        importance descending and limited to ``top_n`` rows.
    """
    # Normalise: if the value is a tuple, use the first element (mean).
    records = []
    for name, value in importances.items():
        if isinstance(value, (tuple, list)):
            score = float(value[0])
        else:
            score = float(value)
        records.append((name, score))

    df = pd.DataFrame(records, columns=["feature", "importance"])
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df = df.head(top_n)

    # Pretty-print to console.
    print(f"\n{title}")
    print("=" * len(title))
    if tabulate is not None:
        print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))
    else:
        print(df.to_string(index=False))

    return df
