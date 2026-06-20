"""CLI handler — creates an Optuna study and runs tuning trials.

The main entry point is :func:`run_tuning`, which wires together the
:class:`TuningConfig`, the per-category search spaces, the appropriate
objective function, and parallelisation settings.

Execution strategies
--------------------
* **Cached pipeline** (``classical_models`` ± ``feature_selection``, no
  ``preprocessing``/``windowing``/``sensor_windows``/``features``):
  runs the full pipeline **once**, then optimises model hyperparameters
  (and optionally feature-selection params) in parallel against the
  cached feature matrix (``n_jobs=os.cpu_count()``).

* **Full pipeline** (any of ``preprocessing``/``windowing``/
  ``sensor_windows``/``features``): each trial re-runs the pipeline
  end-to-end, sequential only (``n_jobs=1``).
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler
from sklearn.preprocessing import LabelEncoder

from config import Config
from data.loader import load_all_experiment_sensors
from features.selection import run_selection_pipeline
from pipeline.builder import run_participant_train_test_pipeline

from .config import TuningCategory, TuningConfig
from .objectives import PipelineObjective, classical_trial
from .search_spaces import (
    suggest_feature_selection_params,
)

TUNING_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / ".tmp" / "tuning"

# Pipeline categories whose active presence forces full per-trial pipeline runs.
_PIPELINE_CATEGORIES = {"preprocessing", "windowing", "sensor_windows", "features"}

# ── Progress callback ────────────────────────────────────────────────────────


class _ProgressCallback:
    """Optuna callback that prints per-trial timing, rolling ETA, and best-so-far."""

    def __init__(self, n_trials: int) -> None:
        self.n_trials = n_trials
        self.trial_times: list[float] = []
        self.start_time: float | None = None
        self.best_so_far: float | None = None

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self.start_time is None:
            self.start_time = time.monotonic()

        now = time.monotonic()

        # Per-trial wall time
        if len(self.trial_times) > 0:
            trial_dt = now - self.trial_times[-1]
        else:
            trial_dt = 0.0
        self.trial_times.append(now)

        n_done = trial.number + 1
        rolling_window = min(10, n_done)
        avg_trial_time = (
            (self.trial_times[-1] - self.trial_times[-rolling_window]) / rolling_window
            if rolling_window > 1
            else trial_dt
        )

        remaining = self.n_trials - n_done
        eta = avg_trial_time * remaining if remaining > 0 else 0.0

        best_value = study.best_value if study.best_trial is not None else None
        if best_value is not None and (self.best_so_far is None or best_value != self.best_so_far):
            self.best_so_far = best_value

        trials_per_sec = 1.0 / avg_trial_time if avg_trial_time > 0 else float("inf")
        eta_str = (
            f"{eta / 60:.0f}m {eta % 60:.0f}s" if eta >= 60
            else f"{eta:.0f}s" if eta > 0
            else "—"
        )
        best_str = f"{best_value:.4f}" if best_value is not None else "—"

        print(
            f"  [{n_done:>4}/{self.n_trials}]  "
            f"trial #{trial.number:>3}  "
            f"value={trial.value or 0:.4f}  "
            f"best={best_str}  "
            f"trial/s={trials_per_sec:.1f}  "
            f"ETA={eta_str}",
            flush=True,
        )


# ── Study name ───────────────────────────────────────────────────────────────


def _auto_study_name(categories: list[TuningCategory]) -> str:
    cat_str = "_".join(c.value for c in categories)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"tune_{cat_str}_{timestamp}"


# ── Post-tuning visualisation plots ──────────────────────────────────────────


def _save_study_plots(study: optuna.Study, output_dir: Path) -> None:
    """Generate interactive HTML plots from a completed study.

    Saves parameter importance, optimisation history, parallel-coordinate,
    and slice plots to *output_dir*.
    """
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_slice,
        )
    except ImportError:
        print("  ⚠  optuna.visualization not available (install plotly) — skipping plots.")
        return

    if len(study.trials) < 2:
        return  # not enough data for meaningful plots

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("param_importances", plot_param_importances),
        ("optimization_history", plot_optimization_history),
        ("parallel_coordinate", plot_parallel_coordinate),
        ("slice", plot_slice),
    ]

    for name, fn in generators:
        try:
            fig = fn(study)
            path = plots_dir / f"{name}.html"
            fig.write_html(str(path))
            print(f"  📊  Saved {path.name}")
        except Exception as exc:
            print(f"  ⚠  Could not generate {name}: {exc}")


# ── Cached sub-pipeline tuning (picklable trial functions) ───────────────────


def _model_trial(
    trial: optuna.Trial,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    n_folds: int,
) -> float:
    """Module-level trial function: tune a classical model on pre-computed features."""
    return classical_trial(trial, model_name, X, y, groups, n_folds)


def _selection_model_trial(
    trial: optuna.Trial,
    model_name: str,
    feature_df_array: np.ndarray,
    feature_df_columns: list[str],
    y: np.ndarray,
    groups: np.ndarray | None,
    n_folds: int,
) -> float:
    """Module-level trial function: sample feature-selection + tune model.

    Uses cached feature matrix, applies selection per trial, then trains
    the model.  Picklable (module-level function) for parallel execution.
    """
    import pandas as pd

    # Reconstruct DataFrame to use run_selection_pipeline
    X_df = pd.DataFrame(feature_df_array, columns=feature_df_columns)
    y_series = pd.Series(y)
    s = suggest_feature_selection_params(trial)
    methods = s.get("selection_methods", ["variance"])
    X_selected = run_selection_pipeline(X_df.copy(), y_series, selection_methods=methods)
    return classical_trial(trial, model_name, X_selected.values, y, groups, n_folds)


def _run_cached_tuning(
    base_config: Config,
    tuning_config: TuningConfig,
    study: optuna.Study,
    cat_labels: set[str],
) -> None:
    """Run feature pipeline once, then tune remaining categories with cached data.

    Handles any combination of ``classical_models`` ± ``feature_selection``
    (without pipeline categories).  Uses ``n_jobs=os.cpu_count()``.
    """
    has_selection = "feature_selection" in cat_labels

    print("\n  Running pipeline once to cache features …", end=" ", flush=True)
    sensor_data = load_all_experiment_sensors(
        base_config.raw_dir,
        resample_rule=base_config.preprocessing.resample_rule,
    )

    val_result, test_result = run_participant_train_test_pipeline(
        sensor_data, base_config, oos_participant=base_config.models.oos_participant,
    )

    if val_result.feature_matrix.empty or test_result.feature_matrix.empty:
        print("EMPTY — aborting.")
        return

    print(f"done  ({val_result.feature_matrix.shape[0]} windows, "
          f"{val_result.feature_matrix.shape[1]} features).")

    X_train = val_result.feature_matrix.values
    y_train = LabelEncoder().fit_transform(val_result.labels.values)
    groups = val_result.participant.values if val_result.participant is not None else None
    feature_columns = val_result.feature_names

    n_jobs = os.cpu_count() or 1
    print(f"  Spawning {n_jobs} worker processes for parallel trials.")

    # Split trials between RF and XGBoost
    per_model = tuning_config.n_trials // 2

    for model_name in ("random_forest", "xgboost"):
        print(f"\n  ── Tuning {model_name} ──")

        if has_selection:
            obj = partial(
                _selection_model_trial,
                model_name=model_name,
                feature_df_array=X_train,
                feature_df_columns=feature_columns,
                y=y_train,
                groups=groups,
                n_folds=base_config.models.cv_folds,
            )
        else:
            obj = partial(
                _model_trial,
                model_name=model_name,
                X=X_train,
                y=y_train,
                groups=groups,
                n_folds=base_config.models.cv_folds,
            )

        study.optimize(
            obj,
            n_trials=per_model,
            timeout=tuning_config.timeout,
            n_jobs=n_jobs,
            callbacks=[_ProgressCallback(per_model)],
        )


# ── Pruner selection ─────────────────────────────────────────────────────────


def _select_pruner(cat_labels: set[str]) -> optuna.pruners.BasePruner:
    """Choose a pruner based on the active tuning categories.

    * ``deep_models`` → :class:`~optuna.pruners.HyperbandPruner` (best for
      epoch-based iterative training).
    * Everything else → :class:`~optuna.pruners.MedianPruner` (good general
      purpose for folds / single-report objectives).
    """
    if "deep_models" in cat_labels:
        return HyperbandPruner(
            min_resource=5,       # earliest epoch at which pruning may happen
            max_resource=200,     # upper bound (matches search_spaces max_epochs)
            reduction_factor=3,   # aggressive: keeps best 1/3 each bracket
        )
    return MedianPruner(n_startup_trials=5, n_warmup_steps=10)


# ── Main entry point ─────────────────────────────────────────────────────────


def run_tuning(
    base_config: Config,
    tuning_config: TuningConfig,
) -> None:
    """Run hyperparameter tuning for the selected categories."""
    categories = tuning_config.categories
    study_name = tuning_config.study_name or _auto_study_name(categories)
    cat_labels = {c.value for c in categories}

    TUNING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study_dir = TUNING_OUTPUT_DIR / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    # ── Decide execution strategy ──────────────────────────────────────────
    # "Cached" = no pipeline categories → run pipeline once, parallel trials.
    # "Full"   = any pipeline category → per-trial pipeline, sequential.
    has_pipeline = bool(cat_labels & _PIPELINE_CATEGORIES)
    use_cached = not has_pipeline and ("classical_models" in cat_labels or "feature_selection" in cat_labels)

    # ── Build objective (full pipeline path) ───────────────────────────────
    if not use_cached:
        _objective = PipelineObjective(
            categories=list(cat_labels),
            base_config=base_config,
            oos_participant=base_config.models.oos_participant,
            n_cv_folds=base_config.models.cv_folds,
        )

    # ── Pruner (Hyperband for deep models, MedianPruner otherwise) ────────
    pruner = _select_pruner(cat_labels)

    # ── SQLite storage ─────────────────────────────────────────────────────
    storage = tuning_config.storage_url
    if storage is not None and storage.startswith("sqlite:///"):
        db_path = storage.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(**(tuning_config.sampler_config or {})),
        pruner=pruner,
        direction=tuning_config.direction,
    )

    # ── Ensure joblib workers can import our modules ───────────────────────
    _ensure_worker_path()

    # ── Print header ───────────────────────────────────────────────────────
    n_jobs = os.cpu_count() if use_cached else 1
    print(f"\n{'=' * 70}")
    print(f"  OPTUNA TUNING")
    print(f"{'=' * 70}")
    print(f"  Study:        {study_name}")
    print(f"  Categories:   {', '.join(sorted(cat_labels))}")
    print(f"  Trials:       {tuning_config.n_trials}")
    print(f"  Workers:      {n_jobs} ({'parallel' if n_jobs > 1 else 'sequential'})")
    print(f"  Pruner:       {type(pruner).__name__}")
    print(f"  Timeout:      {tuning_config.timeout or 'unlimited'}")
    print(f"  Storage:      {storage or 'in-memory'}")
    print(f"  Direction:    {tuning_config.direction}")
    print(f"{'=' * 70}\n")

    # ── Run optimisation ───────────────────────────────────────────────────
    start_time = time.monotonic()

    if use_cached:
        _run_cached_tuning(base_config, tuning_config, study, cat_labels)
    else:
        study.optimize(
            _objective,
            n_trials=tuning_config.n_trials,
            timeout=tuning_config.timeout,
            n_jobs=1,
            callbacks=[_ProgressCallback(tuning_config.n_trials)],
        )

    elapsed = time.monotonic() - start_time

    # ── Report results ─────────────────────────────────────────────────────
    best_trial = study.best_trial

    print(f"\n{'=' * 70}")
    print(f"  TUNING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Duration:     {elapsed:.1f}s  ({elapsed / 60:.1f}m)")
    print(f"  Trials run:   {len(study.trials)}")
    print(f"  Best trial:   #{best_trial.number}")
    print(f"  Best value:   {best_trial.value:.6f}")
    print(f"\n  Best parameters:")
    for key, value in sorted(best_trial.params.items()):
        print(f"    {key}: {value}")

    # ── Persist best params ────────────────────────────────────────────────
    best_path = study_dir / "best.json"
    best_data = {
        "study_name": study_name,
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "categories": sorted(cat_labels),
        "n_trials": tuning_config.n_trials,
        "n_jobs": n_jobs,
        "pruner": type(pruner).__name__,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(best_path, "w") as f:
        json.dump(best_data, f, indent=2, default=str)
    print(f"\n  Best params saved to: {best_path}")

    # ── Post-tuning plots ──────────────────────────────────────────────────
    _save_study_plots(study, study_dir)

    print(f"{'=' * 70}\n")


def _ensure_worker_path() -> None:
    """Ensure joblib worker subprocesses can import ``src/`` modules via
    ``PYTHONPATH`` (child processes inherit this environment variable)."""
    project_src = str(Path(__file__).resolve().parent.parent)  # src/
    current = os.environ.get("PYTHONPATH", "")
    if project_src not in current:
        os.environ["PYTHONPATH"] = f"{project_src}:{current}" if current else project_src
