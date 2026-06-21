"""Configuration dataclass and enum for Optuna hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class TuningCategory(StrEnum):
    """Categories of hyperparameters that can be tuned independently."""

    PREPROCESSING = "preprocessing"
    WINDOWING = "windowing"
    SENSOR_WINDOWS = "sensor_windows"
    FEATURES = "features"
    FEATURE_SELECTION = "feature_selection"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    TCN = "tcn"


# Convenience: all categories in a single tuple
ALL_TUNING_CATEGORIES: tuple[TuningCategory, ...] = tuple(TuningCategory)


@dataclass
class TuningConfig:
    """Configuration for an Optuna tuning run.

    Attributes
    ----------
    categories :
        Subset of tuning categories to optimise in this run.  Pass an empty
        list (default) to mean **all** categories.
    storage_url :
        Optuna storage URL for study persistence.
        ``None`` → in-memory (no persistence).
        ``"sqlite:///.tmp/tuning/studies.db"`` → local SQLite persistence.
    n_trials :
        Number of Optuna trials to run.
    timeout :
        Maximum wall-clock seconds for the whole optimisation (``None`` = no limit).
    trial_timeout_seconds :
        Maximum seconds for a **single trial**.  When a trial exceeds this
        limit it is interrupted and reported as a failure (pruned).
        ``None`` = no per-trial limit.
    pruner_config :
        Optional dict of keyword arguments forwarded to the ``MedianPruner``
        constructor (e.g. ``{"n_startup_trials": 10, "n_warmup_steps": 5}``).
    sampler_config :
        Optional dict of keyword arguments forwarded to the ``TPESampler``
        constructor (e.g. ``{"n_startup_trials": 10}``).
    study_name :
        Optional explicit study name.  When ``None``, a name is auto-generated
        from the selected categories and a timestamp.
    direction :
        Optimisation direction — ``"maximize"`` (default) or ``"minimize"``.
    """

    categories: list[TuningCategory] = field(default_factory=list)
    storage_url: str | None = "sqlite:///.tmp/tuning/studies.db"
    n_trials: int = 50
    timeout: int | None = None
    trial_timeout_seconds: int | None = 180  # 3 min per trial
    pruner_config: dict | None = field(default_factory=lambda: {"n_startup_trials": 5, "n_warmup_steps": 10})
    sampler_config: dict | None = field(default_factory=lambda: {"n_startup_trials": 5})
    study_name: str | None = None
    direction: Literal["maximize", "minimize"] = "maximize"

    def __post_init__(self) -> None:
        if not self.categories:
            self.categories = list(ALL_TUNING_CATEGORIES)
