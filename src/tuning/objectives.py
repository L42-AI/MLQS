"""Optuna objective functions for pipeline-level and model-level tuning.

Three objectives are provided at different levels of granularity:

* ``PipelineObjective`` — tunes preprocessing + windowing + features end-to-end.
* ``ClassicalModelObjective`` — tunes RF / XGBoost on pre-computed features.
* ``DeepModelObjective`` — tunes LSTM / TCN on pre-computed sequences.

All objectives support Optuna's pruning API via
:meth:`optuna.Trial.report` + :meth:`optuna.Trial.should_prune`.
"""

from __future__ import annotations

import signal
from copy import deepcopy
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from config import Config, FeatureConfig, ModelConfig, PreprocessingConfig, SensorWindowConfig
from data.loader import load_all_experiment_sensors
from models.classical import build_classifier
from models.deep import build_deep_classifier, prepare_sequences, train_deep_model
from models.evaluation import compute_classification_metrics
from pipeline.builder import run_participant_train_test_pipeline

from .search_spaces import (
    suggest_classical_model_params,
    suggest_deep_model_params,
    suggest_feature_params,
    suggest_feature_selection_params,
    suggest_preprocessing_params,
    suggest_sensor_window_params,
    suggest_windowing_params,
    _MOTION_SENSORS,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_config_from_trial(trial: optuna.Trial, base_config: Config, categories: list[str]) -> Config:
    """Deep-copy *base_config* and override fields sampled by *trial*."""
    cfg = deepcopy(base_config)

    if "preprocessing" in categories:
        p = suggest_preprocessing_params(trial)
        cfg.preprocessing.filter_method = p["filter_method"]
        cfg.preprocessing.filter_cutoff = p["filter_cutoff"]
        cfg.preprocessing.filter_type = p["filter_type"]
        if p["filter_method"] == "butterworth":
            cfg.preprocessing.filter_order = p["filter_order"]
        elif p["filter_method"] == "savitzky_golay":
            cfg.preprocessing.savitzky_golay_window_length = p.get("filter_window_length", 11)
            cfg.preprocessing.savitzky_golay_polyorder = p.get("filter_polyorder", 3)
        cfg.preprocessing.imputation_method = p["imputation_method"]
        cfg.preprocessing.imputation_max_gap = p["imputation_max_gap"]

    if "windowing" in categories:
        w = suggest_windowing_params(trial)
        cfg.features.window_size = w["window_size"]
        cfg.features.window_overlap = w["window_overlap"]
        cfg.features.frequency_window_size = w["frequency_window_size"]

    if "sensor_windows" in categories:
        sw = suggest_sensor_window_params(trial)
        if sw.get("use_sensor_windows"):
            overrides: dict[str, SensorWindowConfig] = {}
            # HeartRate — slow physiological signal, sampled at 10 Hz
            if "sensor_window_HeartRate" in sw:
                overrides["HeartRate"] = SensorWindowConfig(
                    base_window_seconds=sw["sensor_window_HeartRate"]
                )
            # Motion sensors — fast 3-axis movement, sampled at 50 Hz
            if "sensor_window_motion" in sw:
                motion_win = sw["sensor_window_motion"]
                for sensor in _MOTION_SENSORS:
                    overrides[sensor] = SensorWindowConfig(base_window_seconds=motion_win)
            cfg.features.sensor_windows = overrides

    if "features" in categories:
        f = suggest_feature_params(trial)
        cfg.features.time_domain = f["time_domain"]
        cfg.features.frequency_domain = f["frequency_domain"]
        cfg.features.statistical = f["statistical"]
        cfg.features.magnitude_channels = f["magnitude_channels"]
        cfg.features.cross_sensor_features = f["cross_sensor_features"]

    if "feature_selection" in categories:
        s = suggest_feature_selection_params(trial)
        cfg.features.selection_methods = tuple(s["selection_methods"])

    return cfg


# ── Pipeline-level objective ─────────────────────────────────────────────────


class _TrialTimeout:
    """Per-trial timeout via ``signal.SIGALRM`` (Unix-only)."""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._old_handler: Any = None

    def __enter__(self) -> None:
        self._old_handler = signal.signal(signal.SIGALRM, self._raise_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, *args: Any) -> None:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old_handler)

    @staticmethod
    def _raise_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError("Trial timed out")


class PipelineObjective:
    """End-to-end pipeline objective: preprocessing → features → model.

    This is the most comprehensive objective — it tunes every stage of the
    pipeline and is therefore the most expensive per trial.

    Parameters
    ----------
    categories :
        Which tuning categories to include in the search.
    base_config :
        Starting :class:`Config` whose non-tuned fields are kept as-is.
    oos_participant :
        Held-out participant for the LOPO evaluation.
    n_cv_folds :
        Number of GroupKFold folds for validation.
    trial_timeout :
        Maximum seconds for a single trial (``None`` = no limit).
    """

    def __init__(
        self,
        categories: list[str],
        base_config: Config,
        oos_participant: str = "Kim",
        n_cv_folds: int = 5,
        trial_timeout: int | None = None,
    ) -> None:
        self.categories = categories
        self.base_config = base_config
        self.oos_participant = oos_participant
        self.n_cv_folds = n_cv_folds
        self.trial_timeout = trial_timeout
        self._data: any = None

    def _ensure_data(self) -> None:
        if self._data is None:
            self._data = load_all_experiment_sensors(
                self.base_config.raw_dir,
                resample_rule=self.base_config.preprocessing.resample_rule,
            )

    def __call__(self, trial: optuna.Trial) -> float:
        # ── Per-trial timeout ────────────────────────────────────────────
        timeout_ctx = (
            _TrialTimeout(self.trial_timeout)
            if self.trial_timeout is not None
            else None
        )
        if timeout_ctx is not None:
            timeout_ctx.__enter__()

        try:
            return self._run_trial(trial)
        except TimeoutError:
            # Return a terrible score so TPE learns to avoid this region.
            print(f"  ⏱  Trial #{trial.number} timed out (> {self.trial_timeout}s) → scoring 0.0", flush=True)
            return 0.0
        finally:
            if timeout_ctx is not None:
                timeout_ctx.__exit__(None, None, None)

    def _run_trial(self, trial: optuna.Trial) -> float:
        self._ensure_data()

        # Build config from trial params
        cfg = _build_config_from_trial(trial, self.base_config, self.categories)

        # Run pipeline with participant-based split
        val_result, test_result = run_participant_train_test_pipeline(
            self._data, cfg, oos_participant=self.oos_participant,
        )

        if val_result.feature_matrix.empty or test_result.feature_matrix.empty:
            raise optuna.TrialPruned("Empty pipeline result")

        X_train = val_result.feature_matrix.values
        y_train = val_result.labels.values
        X_test = test_result.feature_matrix.values
        y_test = test_result.labels.values

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        # Quick evaluation with default XGBoost (tune the model separately)
        n_feats = X_train.shape[1]
        n_train = len(X_train)
        print(f"    Features: {n_feats}  |  Train samples: {n_train}  |  Test samples: {len(X_test)}", flush=True)
        print(f"    Training XGBoost …", flush=True)
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            n_jobs=1, tree_method="hist", random_state=42, verbosity=0,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        metrics = compute_classification_metrics(y_test, preds)

        f1 = metrics["f1"]

        # Report intermediate value for pruning
        trial.report(f1, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return f1


# ── Classical model objective ────────────────────────────────────────────────


def classical_trial(
    trial: optuna.Trial,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_folds: int = 5,
) -> float:
    """Module-level trial function for classical model tuning (picklable).

    Parameters
    ----------
    model_name :
        ``"random_forest"`` or ``"xgboost"``.
    X :
        Feature matrix.
    y :
        Labels.
    groups :
        Group labels for GroupKFold (e.g. participant IDs).
    n_folds :
        Number of CV folds.

    Returns
    -------
    Mean CV F1 score across folds.
    """
    params = suggest_classical_model_params(trial, model_name)
    model_name = params.pop("model_name")
    hp = _rename_params_for_model(model_name, params)

    from sklearn.model_selection import GroupKFold, KFold

    if groups is not None and len(np.unique(groups)) >= 2:
        cv = GroupKFold(n_splits=n_folds)
    else:
        cv = KFold(n_splits=min(5, len(X) // 10))

    fold_f1s: list[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        clf = build_classifier(model_name, **hp)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[val_idx])
        metrics = compute_classification_metrics(y[val_idx], preds)
        f1 = metrics["f1"]
        fold_f1s.append(f1)

        trial.report(f1, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_f1s))


class ClassicalModelObjective:
    """Tune Random Forest or XGBoost on **pre-computed** features.

    .. deprecated::
        Use :func:`classical_trial` with ``functools.partial`` instead,
        which is picklable and supports ``study.optimize(n_jobs=...)``.

    Parameters
    ----------
    model_name :
        ``"random_forest"`` or ``"xgboost"``.
    X :
        Feature matrix.
    y :
        Labels.
    groups :
        Group labels for GroupKFold (e.g. participant IDs).
    n_folds :
        Number of CV folds.
    """

    def __init__(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        n_folds: int = 5,
    ) -> None:
        self.model_name = model_name
        self.X = X
        self.y = y
        self.groups = groups
        self.n_folds = min(n_folds, len(np.unique(groups)) if groups is not None else 5)

    def __call__(self, trial: optuna.Trial) -> float:
        return classical_trial(trial, self.model_name, self.X, self.y, self.groups, self.n_folds)


def _rename_params_for_model(model_name: str, raw: dict) -> dict:
    """Strip per-model prefixes from Optuna-suggested param names.

    ``rf_n_estimators`` → ``n_estimators``, ``xgb_n_estimators`` → ``n_estimators``.
    """
    prefix = "rf_" if model_name == "random_forest" else "xgb_"
    renamed: dict = {}
    for k, v in raw.items():
        if k.startswith(prefix):
            renamed[k[len(prefix):]] = v
        else:
            renamed[k] = v
    # Remove meta keys
    renamed.pop("model_name", None)
    return renamed


# ── Deep model objective ─────────────────────────────────────────────────────


class DeepModelObjective:
    """Tune LSTM or TCN hyperparameters on pre-computed sequences.

    Parameters
    ----------
    X :
        Feature matrix (2-D windows × features).
    y :
        Labels.
    sequence_length :
        Number of time-steps per sequence for the deep model.
    n_trials_for_pruning :
        How many intermediate evaluation steps to report during training
        (spaced evenly across epochs) for the pruner.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 32,
        n_trials_for_pruning: int = 5,
    ) -> None:
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.n_trials_for_pruning = n_trials_for_pruning

    def __call__(self, trial: optuna.Trial) -> float:
        params = suggest_deep_model_params(trial)
        model_type = params.pop("model_type")
        batch_size = params.pop("batch_size")
        num_epochs = params.pop("num_epochs")
        learning_rate = params.pop("learning_rate")
        channel_sizes = params.pop("channel_sizes", None)

        # Prepare sequences
        train_loader = prepare_sequences(
            self.X, self.y, self.sequence_length, batch_size=batch_size
        )
        # Use a subset for validation during tuning
        split = int(0.8 * len(self.X) // self.sequence_length * self.sequence_length)
        X_val = self.X[split:]
        y_val = self.y[split:]
        X_tr = self.X[:split]
        y_tr = self.y[:split]

        if len(X_tr) < self.sequence_length:
            raise optuna.TrialPruned("Not enough data for sequence length")

        val_loader = prepare_sequences(
            X_val, y_val, self.sequence_length, batch_size=batch_size
        )
        tr_loader = prepare_sequences(
            X_tr, y_tr, self.sequence_length, batch_size=batch_size
        )

        # Build model
        input_size = self.X.shape[1]
        num_classes = len(np.unique(self.y))

        if model_type == "lstm":
            from models.deep import LSTMClassifier
            model = LSTMClassifier(
                input_size=input_size,
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                num_classes=num_classes,
                dropout_probability=params["dropout_probability"],
            )
        else:
            from models.deep import TCNClassifier
            model = TCNClassifier(
                input_size=input_size,
                channel_sizes=channel_sizes or [32, 64, 64],
                num_classes=num_classes,
                kernel_size=params.get("kernel_size", 3),
                dropout_probability=params["dropout_probability"],
            )

        # Training with intermediate reporting for pruning
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        report_interval = max(1, num_epochs // self.n_trials_for_pruning)
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            model.train()
            for batch_inputs, batch_labels in tr_loader:
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(batch_inputs), batch_labels)
                loss.backward()
                optimizer.step()

            # Validation
            if (epoch + 1) % report_interval == 0 or epoch == num_epochs - 1:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_inputs, batch_labels in val_loader:
                        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                        logits = model(batch_inputs)
                        correct += (logits.argmax(1) == batch_labels).sum().item()
                        total += batch_labels.size(0)
                val_acc = correct / total if total > 0 else 0.0
                best_val_acc = max(best_val_acc, val_acc)

                trial.report(val_acc, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return best_val_acc
