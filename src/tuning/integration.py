"""CLI handler — creates an Optuna study and runs tuning trials.

The main entry point is :func:`run_tuning`, which wires together the
:class:`TuningConfig`, the per-category search spaces, the appropriate
objective function, and parallelisation settings.

Execution strategies
--------------------
* **Cached pipeline** (one of ``random_forest``/``xgboost``/``lstm``/``tcn``,
  no ``preprocessing``/``windowing``/``sensor_windows``/``features``):
  runs the full pipeline **once**, then tunes the requested model's
  hyperparameters against the cached feature matrix.

* **Full pipeline** (any of ``preprocessing``/``windowing``/
  ``sensor_windows``/``features``): each trial re-runs the pipeline
  end-to-end, sequential only (``n_jobs=1``).
"""

from __future__ import annotations

import json
import math
import os
import signal
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from config import Config
from data.loader import load_all_experiment_sensors
from features.selection import select_by_boruta
from pipeline.builder import run_participant_train_test_pipeline

from .config import TuningCategory, TuningConfig
from .objectives import PipelineObjective, _build_and_train_deep, classical_trial
from .search_spaces import suggest_lstm_params, suggest_rf_params, suggest_tcn_params, suggest_xgb_params

TUNING_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / ".tmp" / "tuning"

# Pipeline categories whose active presence forces full per-trial pipeline runs.
_PIPELINE_CATEGORIES = {"preprocessing", "windowing", "sensor_windows", "features"}

# ── Progress callback ────────────────────────────────────────────────────────


class _ProgressCallback:
    """Optuna callback that updates a tqdm progress bar and prints trial results."""

    def __init__(self, n_trials: int, desc: str = "Tuning") -> None:
        self.pbar = tqdm(
            total=n_trials,
            desc=desc,
            unit="trial",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        self.best_so_far: float | None = None
        self.n_pruned = 0
        self.n_failed = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Track pruned / failed
        if trial.state == optuna.trial.TrialState.PRUNED:
            self.n_pruned += 1
        elif trial.state == optuna.trial.TrialState.FAIL:
            self.n_failed += 1

        # Print trial result line
        duration = trial.duration.total_seconds() if trial.duration else 0.0
        if trial.value is not None:
            print(f"    Trial #{trial.number:>3d}  F1={trial.value:.4f}  ({duration:.1f}s)", flush=True)
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"    Trial #{trial.number:>3d}  PRUNED  ({duration:.1f}s)", flush=True)
        elif trial.state == optuna.trial.TrialState.FAIL:
            print(f"    Trial #{trial.number:>3d}  FAILED  ({duration:.1f}s)", flush=True)

        # Update progress bar
        try:
            best_value = study.best_value
        except ValueError:
            best_value = None
        if best_value is not None and best_value != self.best_so_far:
            self.best_so_far = best_value
            postfix = {"best": f"{self.best_so_far:.4f}"}
            if self.n_pruned:
                postfix["pruned"] = self.n_pruned
            if self.n_failed:
                postfix["failed"] = self.n_failed
            self.pbar.set_postfix(postfix)
        self.pbar.update(1)

    def close(self) -> None:
        self.pbar.close()


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
    suggest_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    n_folds: int,
    trial_timeout: int | None = None,
) -> float:
    """Module-level trial function: tune a classical model on pre-computed features.

    Supports per-trial timeout via ``signal.SIGALRM``.
    """
    if trial_timeout is not None:
        old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(trial_timeout)
        try:
            return classical_trial(trial, model_name, suggest_fn, X, y, groups, n_folds)
        except TimeoutError:
            print(f"  ⏱  Trial #{trial.number} timed out "
                  f"(> {trial_timeout}s) → scoring 0.0", flush=True)
            return 0.0
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    return classical_trial(trial, model_name, suggest_fn, X, y, groups, n_folds)


def _raise_timeout(signum: int, frame: object) -> None:
    """SIGALRM handler — raises TimeoutError for per-trial timeout."""
    raise TimeoutError("Trial timed out")


BORUTA_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / ".tmp" / "boruta"
KEPT_FEATURES_PATH = BORUTA_OUTPUT_DIR / "kept_features.csv"


def _run_cached_tuning(
    base_config: Config,
    tuning_config: TuningConfig,
    study: optuna.Study,
    cat_labels: set[str],
    boruta_features: bool = False,
) -> None:
    """Run feature pipeline once, then tune requested models with cached data.

    Each selected model gets ``n_trials`` (``tuning_config.n_trials``) allocated
    independently.  Pipeline categories must NOT be present (they force per-trial
    re-runs).  Sequential execution only (``n_jobs=1``).

    Boruta feature selection is NOT part of tuning — run separately with
    ``--boruta``, then use ``--boruta-features`` when tuning.
    """
    n_trials = tuning_config.n_trials

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
    feature_columns = list(val_result.feature_matrix.columns)

    # ── Apply Boruta feature mask ─────────────────────────────────────────
    if boruta_features:
        if KEPT_FEATURES_PATH.exists():
            kept_names = [
                line.strip()
                for line in KEPT_FEATURES_PATH.read_text().splitlines()
                if line.strip()
            ]
            keep_idx = [i for i, col in enumerate(feature_columns) if col in kept_names]
            X_train = X_train[:, keep_idx]
            feature_columns = [feature_columns[i] for i in keep_idx]
            print(f"  [boruta] Filtered to {len(feature_columns)} Boruta-confirmed features.", flush=True)
        else:
            print(f"  ⚠  No Boruta feature list found at {KEPT_FEATURES_PATH}. "
                  f"Run `--boruta` first.", flush=True)

    print(f"\n  Per-trial timeout: {tuning_config.trial_timeout_seconds or 'unlimited'}s")

    # ── Random Forest ──────────────────────────────────────────────────────
    if "random_forest" in cat_labels:
        print(f"\n{'─' * 60}")
        print(f"  Random Forest ({n_trials} trials)")
        print(f"{'─' * 60}")
        obj = partial(
            _model_trial,
            model_name="random_forest",
            suggest_fn=suggest_rf_params,
            X=X_train, y=y_train, groups=groups,
            n_folds=base_config.models.cv_folds,
            trial_timeout=tuning_config.trial_timeout_seconds,
        )
        cb = _ProgressCallback(n_trials, desc="  RandomForest")
        study.optimize(obj, n_trials=n_trials, timeout=tuning_config.timeout,
                       n_jobs=1, callbacks=[cb])
        cb.close()

    # ── XGBoost ────────────────────────────────────────────────────────────
    if "xgboost" in cat_labels:
        print(f"\n{'─' * 60}")
        print(f"  XGBoost ({n_trials} trials)")
        print(f"{'─' * 60}")
        obj = partial(
            _model_trial,
            model_name="xgboost",
            suggest_fn=suggest_xgb_params,
            X=X_train, y=y_train, groups=groups,
            n_folds=base_config.models.cv_folds,
            trial_timeout=tuning_config.trial_timeout_seconds,
        )
        cb = _ProgressCallback(n_trials, desc="  XGBoost")
        study.optimize(obj, n_trials=n_trials, timeout=tuning_config.timeout,
                       n_jobs=1, callbacks=[cb])
        cb.close()

    # ── LSTM ───────────────────────────────────────────────────────────────
    if "lstm" in cat_labels:
        from models.deep import LSTMClassifier
        print(f"\n{'─' * 60}")
        print(f"  LSTM ({n_trials} trials)")
        print(f"{'─' * 60}")
        obj = partial(
            _build_and_train_deep,
            suggest_fn=suggest_lstm_params,
            model_builder=lambda input_size, num_classes, **hp: LSTMClassifier(
                input_size=input_size,
                hidden_size=hp.get("hidden_size", 128),
                num_layers=hp.get("num_layers", 2),
                num_classes=num_classes,
                dropout_probability=hp.get("dropout", 0.2),
            ),
            X=X_train, y=y_train,
        )
        cb = _ProgressCallback(n_trials, desc="  LSTM")
        study.optimize(obj, n_trials=n_trials, timeout=tuning_config.timeout,
                       n_jobs=1, callbacks=[cb])
        cb.close()

    # ── TCN ────────────────────────────────────────────────────────────────
    if "tcn" in cat_labels:
        from models.deep import TCNClassifier
        print(f"\n{'─' * 60}")
        print(f"  TCN ({n_trials} trials)")
        print(f"{'─' * 60}")
        obj = partial(
            _build_and_train_deep,
            suggest_fn=suggest_tcn_params,
            model_builder=lambda input_size, num_classes, **hp: TCNClassifier(
                input_size=input_size,
                channel_sizes=hp.get("channel_sizes", [16, 32, 32]),
                num_classes=num_classes,
                kernel_size=hp.get("kernel_size", 5),
                dropout_probability=hp.get("dropout", 0.2),
            ),
            X=X_train, y=y_train,
        )
        cb = _ProgressCallback(n_trials, desc="  TCN")
        study.optimize(obj, n_trials=n_trials, timeout=tuning_config.timeout,
                       n_jobs=1, callbacks=[cb])
        cb.close()


# ── Pruner selection ─────────────────────────────────────────────────────────


def _select_pruner(cat_labels: set[str]) -> optuna.pruners.BasePruner:
    """Choose a pruner based on the active tuning categories.

    * ``lstm`` / ``tcn`` → :class:`~optuna.pruners.HyperbandPruner` (best for
      epoch-based iterative training).
    * Everything else → :class:`~optuna.pruners.MedianPruner` (good general
      purpose for folds / single-report objectives).
    """
    if {"lstm", "tcn"} & cat_labels:
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
    boruta_features: bool = False,
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
    non_pipeline_cats = {"random_forest", "xgboost", "lstm", "tcn"}
    use_cached = not has_pipeline and bool(cat_labels & non_pipeline_cats)

    # Warn if pipeline + non-pipeline categories are mixed — the non-pipeline
    # ones will be silently ignored by PipelineObjective.
    if has_pipeline:
        ignored = cat_labels - _PIPELINE_CATEGORIES
        if ignored:
            print(f"  ⚠  Categories {sorted(ignored)} are ignored when "
                  f"pipeline categories are active.", flush=True)

    # ── Build objective (full pipeline path) ───────────────────────────────
    if not use_cached:
        _objective = PipelineObjective(
            categories=list(cat_labels),
            base_config=base_config,
            oos_participant=base_config.models.oos_participant,
            n_cv_folds=base_config.models.cv_folds,
            trial_timeout=tuning_config.trial_timeout_seconds,
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
    n_jobs = 1
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
        _run_cached_tuning(base_config, tuning_config, study, cat_labels,
                           boruta_features=boruta_features)
    else:
        cb = _ProgressCallback(tuning_config.n_trials, desc="  Tuning")
        study.optimize(
            _objective,
            n_trials=tuning_config.n_trials,
            timeout=tuning_config.timeout,
            n_jobs=1,
            callbacks=[cb],
        )
        cb.close()

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


# ── Standalone Boruta runner (no Optuna) ──────────────────────────────────


def _run_boruta_once(base_config: Config) -> None:
    """Run pipeline once + Boruta feature selection (no model tuning)."""
    print(f"\n{'=' * 70}")
    print(f"  BORUTA FEATURE SELECTION")
    print(f"{'=' * 70}")

    print("\n  Running pipeline to cache features …", end=" ", flush=True)
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

    X_full = val_result.feature_matrix
    y_full = val_result.labels
    feature_names = val_result.feature_names

    print(f"\n  Running Boruta on {X_full.shape[1]} features …", flush=True)
    X_selected = select_by_boruta(X_full, y_full)
    kept = list(X_selected.columns)
    dropped = [c for c in feature_names if c not in kept]

    print(f"\n  {'=' * 50}")
    print(f"  Boruta Results")
    print(f"  {'=' * 50}")
    print(f"  Features kept:   {len(kept)}/{X_full.shape[1]}")
    print(f"  Features dropped: {len(dropped)}")
    if kept:
        print(f"\n  Kept features:")
        for name in kept:
            print(f"    [+] {name}")
    if dropped:
        print(f"\n  Dropped features:")
        for name in dropped:
            print(f"    [-] {name}")

    # Save the result
    output_dir = Path(__file__).resolve().parent.parent.parent / ".tmp" / "boruta"
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_path = output_dir / "kept_features.csv"
    with open(kept_path, "w") as f:
        f.write("\n".join(kept))
    report_path = output_dir / "boruta_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Boruta Feature Selection Report\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"Total features: {X_full.shape[1]}\n")
        f.write(f"Kept:           {len(kept)}\n")
        f.write(f"Dropped:        {len(dropped)}\n\n")
        f.write("Kept features:\n")
        for n in kept:
            f.write(f"  {n}\n")
        f.write("\nDropped features:\n")
        for n in dropped:
            f.write(f"  {n}\n")
    print(f"\n  Saved: {kept_path}")
    print(f"  Saved: {report_path}")
    print(f"{'=' * 70}\n")


def _ensure_worker_path() -> None:
    """Ensure joblib worker subprocesses can import ``src/`` modules via
    ``PYTHONPATH`` (child processes inherit this environment variable)."""
    project_src = str(Path(__file__).resolve().parent.parent)  # src/
    current = os.environ.get("PYTHONPATH", "")
    if project_src not in current:
        os.environ["PYTHONPATH"] = f"{project_src}:{current}" if current else project_src
