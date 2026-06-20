"""Feature selection — reduce dimensionality before modelling."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server/CLI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from boruta import BorutaPy
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
from tqdm import tqdm

_BORUTA_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / ".tmp" / "boruta"


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
    n_features = numeric_features.shape[1]
    step = max(1, n_features // 20)  # eliminate ~5% per iteration
    selector = RFE(
        RandomForestClassifier(n_estimators=50, random_state=42),
        n_features_to_select=top_k,
        step=step,
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
        l1_ratio=1.0,
        C=regularisation_strength,
        solver="saga",
        max_iter=5000,
        random_state=42,
    )
    selector = SelectFromModel(estimator.fit(numeric_features, encoded_labels), prefit=True)
    return feature_matrix.iloc[:, selector.get_support(indices=True)]


# ── Tracking Boruta wrapper ───────────────────────────────────────────────────


class _TrackingBoruta(BorutaPy):
    """BorutaPy subclass that records per-iteration state for live progress +
    after-training visualisation.

    Three methods are overridden:

    * ``_add_shadows_get_imps`` — captures per-feature importance values.
    * ``_print_results`` — records decision counts and drives the tqdm bar.
    * ``_fit`` — ensures the parent's ``verbose`` threshold is met so
      ``_print_results`` is actually called each iteration.
    """

    def __init__(self, *args, **kwargs):
        kwargs["verbose"] = 2  # always ≥ 1 so _print_results fires
        super().__init__(*args, **kwargs)

    def _add_shadows_get_imps(self, X, y, dec_reg):
        result = super()._add_shadows_get_imps(X, y, dec_reg)
        # result = (real_importances, shadow_importances)
        if not hasattr(self, "_imp_history_list"):
            self._imp_history_list: list[np.ndarray] = []
            self._sha_max_list: list[float] = []
        self._imp_history_list.append(result[0].copy())
        imp_sha_max = float(np.percentile(result[1], self.perc))
        self._sha_max_list.append(imp_sha_max)
        return result

    def _print_results(self, dec_reg, _iter, flag):
        # Record iteration state
        if not hasattr(self, "_iteration_records"):
            self._iteration_records: list[dict] = []
        n_confirmed = int(np.sum(dec_reg == 1))
        n_rejected = int(np.sum(dec_reg == -1))
        n_tentative = int(np.sum(dec_reg == 0))
        sha = self._sha_max_list[-1] if self._sha_max_list else 0.0
        self._iteration_records.append({
            "iter": _iter,
            "confirmed": n_confirmed,
            "tentative": n_tentative,
            "rejected": n_rejected,
            "max_shadow": sha,
        })
        # Live progress bar — only advance during iterations (flag=0),
        # not on the final summary call (flag=1).
        if flag == 0 and hasattr(self, "_pbar") and self._pbar is not None:
            self._pbar.set_postfix(
                confirmed=n_confirmed,
                tentative=n_tentative,
                rejected=n_rejected,
                shadow=f"{sha:.4f}",
            )
            self._pbar.update(1)


def _save_boruta_plot(
    boruta: _TrackingBoruta,
    feature_names: list[str],
    output_dir: Path,
) -> Path | None:
    """Create and save a Boruta visualisation.

    The plot shows each feature's median importance across iterations,
    colour-coded by decision (confirmed / tentative / rejected), with the
    shadow importance threshold overlaid.

    Returns the path to the saved PNG, or ``None`` if there is nothing to
    plot (e.g. Boruta converged without entering the main loop).
    """
    if not boruta._imp_history_list:
        return None  # no data to plot
    records = getattr(boruta, "_iteration_records", None)
    if not records:
        return None

    # ── Build decision lookup ────────────────────────────────────────────
    decision: dict[int, str] = {}  # feature_idx → label
    for idx in np.where(boruta.support_)[0]:
        decision[int(idx)] = "confirmed"
    for idx in np.where(boruta.support_weak_)[0]:
        decision[int(idx)] = "tentative"
    for idx in np.where(boruta.ranking_ >= 3)[0]:
        decision[int(idx)] = "rejected"

    # median importance across all iterations (skip the initial zero row
    # that Boruta seeds imp_history with internally).
    imp_matrix = np.array(boruta._imp_history_list)  # [n_iter, n_feat]
    median_imp = np.nanmedian(imp_matrix, axis=0)     # [n_feat]
    shadow_threshold = float(np.median(boruta._sha_max_list))

    # ── Sort features by median importance (descending) ──────────────────
    order = np.argsort(median_imp)[::-1]
    labels = [f"{i}: {feature_names[i]}" for i in order]
    vals = median_imp[order]
    colors = []
    for i in order:
        d = decision.get(int(i), "rejected")
        if d == "confirmed":
            colors.append("#2ecc71")  # green
        elif d == "tentative":
            colors.append("#3498db")  # blue
        else:
            colors.append("#e74c3c")  # red
    # Shorten labels that are too long
    labels_short = []
    for lab in labels:
        name_part = lab.split(": ", 1)[1] if ": " in lab else lab
        if len(name_part) > 55:
            name_part = name_part[:25] + "…" + name_part[-27:]
            labels_short.append(f"{lab.split(':')[0]}: {name_part}")
        else:
            labels_short.append(lab)

    # ── Plot ─────────────────────────────────────────────────────────────
    n_feat = len(vals)
    height = max(6, n_feat * 0.22)
    fig, ax = plt.subplots(figsize=(10, height))
    bars = ax.barh(range(n_feat), vals, color=colors, edgecolor="none", height=0.7)

    ax.axvline(shadow_threshold, color="grey", ls="--", lw=1.2,
               label=f"Shadow threshold (median) = {shadow_threshold:.4f}")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Confirmed"),
        Patch(facecolor="#3498db", label="Tentative"),
        Patch(facecolor="#e74c3c", label="Rejected"),
        plt.Line2D([0], [0], color="grey", ls="--", label="Shadow threshold"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Iteration summary in the top-left
    n_iter = len(records)
    n_conf = int(boruta.support_.sum())
    n_tent = int(boruta.support_weak_.sum())
    n_rej = n_feat - n_conf - n_tent
    summary = (
        f"Boruta: {n_iter} iters  |  "
        f"[+] {n_conf} confirmed  |  "
        f"[~] {n_tent} tentative  |  "
        f"[-] {n_rej} rejected"
    )
    ax.text(0.99, 0.98, summary, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            fontfamily="monospace")

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(labels_short, fontsize=7)
    ax.set_xlabel("Median feature importance across Boruta iterations", fontsize=10)
    ax.set_title("Boruta Feature Selection Results", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.margins(y=0.01)
    fig.tight_layout()

    out_path = output_dir / "boruta_feature_importance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def select_by_boruta(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    **kwargs,
) -> pd.DataFrame:
    """Select features via Boruta — compares real vs. randomised shadow features.

    Boruta fits a Random Forest repeatedly, comparing each feature's importance
    against its own shuffled copy. Features that consistently outperform their
    shadow are kept; the rest are dropped — **no threshold tuning needed**.

    During training a tqdm progress bar shows the iteration count and the number
    of confirmed / tentative / rejected features in real time.  After training a
    visualisation is saved to ``.tmp/boruta/boruta_feature_importance.png``.

    Parameters
    ----------
    feature_matrix :
        Full feature matrix.
    labels :
        Target labels.
    **kwargs :
        Passed to :class:`boruta.BorutaPy` (e.g. ``n_estimators``, ``max_iter``).

    Returns
    -------
    pd.DataFrame
        Reduced feature matrix (only Boruta-confirmed features).
    """
    numeric_features = feature_matrix.select_dtypes(include=[np.number])
    n_total = numeric_features.shape[1]

    # ── Pre-filter: remove near-constant and highly-correlated features ──
    # This reduces noise dilution that slows down Boruta's convergence.
    n_before_pre = numeric_features.shape[1]
    numeric_features = drop_low_variance_features(
        numeric_features, variance_threshold=kwargs.get("variance_threshold", 0.01),
    )
    numeric_features = drop_highly_correlated_features(
        numeric_features, correlation_threshold=kwargs.get("correlation_threshold", 0.95),
    )
    n_after_pre = numeric_features.shape[1]
    if n_after_pre < n_before_pre:
        print(f"  Pre-filter removed {n_before_pre - n_after_pre}/{n_before_pre} "
              f"features ({n_after_pre} remaining)", flush=True)
    pre_filtered_columns = list(numeric_features.columns)

    encoded_labels = LabelEncoder().fit_transform(labels.astype(str))

    # ── Run Boruta ──────────────────────────────────────────────────────
    # Deeper trees produce wider importance spread → higher shadow threshold
    # → faster rejection of irrelevant features.  perc=85 means a feature
    # must beat the 85th percentile of shadow importances (not the max),
    # which is a much stronger test.
    max_depth = kwargs.pop("max_depth", 20)
    perc = kwargs.pop("perc", 85)

    estimator = RandomForestClassifier(
        n_jobs=-1, class_weight="balanced", max_depth=max_depth, random_state=42,
    )
    boruta = _TrackingBoruta(
        estimator,
        n_estimators="auto",
        random_state=42,
        perc=perc,
        verbose=0,
        **kwargs,
    )

    feature_names = list(numeric_features.columns)
    max_iter = kwargs.get("max_iter", 200)

    with tqdm(
        total=max_iter,
        desc="  Boruta",
        unit="iter",
        bar_format=(
            "{desc}: {n_fmt}/{total_fmt} [{postfix}]"
            "  |{bar}|  {elapsed}<{remaining}"
        ),
    ) as pbar:
        boruta._pbar = pbar
        boruta.fit(numeric_features.values, encoded_labels)
        boruta._pbar = None

    kept_mask = boruta.support_
    kept_columns = [pre_filtered_columns[i] for i in np.where(kept_mask)[0]]
    n_kept = len(kept_columns)

    print(f"  [+] Boruta selected {n_kept}/{n_total} features "
          f"(from {n_total} total, {n_after_pre} after pre-filter)", flush=True)

    # ── Save visualisation ──────────────────────────────────────────────
    _BORUTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = _save_boruta_plot(
        boruta, feature_names, _BORUTA_OUTPUT_DIR,
    )
    if plot_path is not None:
        print(f"  [plot] Boruta visualisation saved → {plot_path}", flush=True)

    return feature_matrix[kept_columns]


SELECTION_METHOD_NAME_TO_FUNCTION = {
    "variance": drop_low_variance_features,
    "correlation": drop_highly_correlated_features,
    "mutual_information": select_top_by_mutual_information,
    "rfe": select_by_recursive_elimination,
    "l1": select_by_l1_regularization,
    "boruta": select_by_boruta,
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
