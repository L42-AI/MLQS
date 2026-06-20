"""Per-category Optuna search-space definitions.

Each ``suggest_*`` function accepts an ``optuna.Trial`` and returns a flat
dictionary of sampled hyperparameters for its category.
"""

from __future__ import annotations

import optuna


def suggest_preprocessing_params(trial: optuna.Trial) -> dict:
    """Sample preprocessing hyperparameters.

    Conditional branches:
    - ``savitzky_golay`` filter → also tunes ``filter_window_length`` & ``filter_polyorder``.
    - ``knn`` imputation → also tunes ``imputation_n_neighbors``.
    """
    params: dict = {}

    params["filter_method"] = trial.suggest_categorical(
        "filter_method", ["butterworth", "moving_average", "savitzky_golay"]
    )
    # Max cutoff must stay below nyquist (sample_rate / 2).
    # With default 100 ms resample → 10 Hz → nyquist = 5 Hz.
    # Higher-order filters need extra headroom → cap at 4.5.
    params["filter_cutoff"] = trial.suggest_float("filter_cutoff", 0.5, 4.5)
    params["filter_type"] = trial.suggest_categorical(
        "filter_type", ["low", "high", "band"]
    )

    if params["filter_method"] == "butterworth":
        params["filter_order"] = trial.suggest_int("filter_order", 1, 8)
    elif params["filter_method"] == "savitzky_golay":
        params["filter_window_length"] = trial.suggest_int(
            "filter_window_length", 5, 25, step=2
        )
        params["filter_polyorder"] = trial.suggest_int("filter_polyorder", 1, 5)

    params["imputation_method"] = trial.suggest_categorical(
        "imputation_method", ["interpolate", "ffill", "knn"]
    )
    params["imputation_max_gap"] = trial.suggest_int("imputation_max_gap", 1, 20)

    if params["imputation_method"] == "knn":
        params["imputation_n_neighbors"] = trial.suggest_int(
            "imputation_n_neighbors", 2, 15
        )

    return params


def suggest_windowing_params(trial: optuna.Trial) -> dict:
    """Sample windowing / feature-extraction hyperparameters."""
    params: dict = {}

    params["window_size"] = trial.suggest_float("window_size", 0.5, 5.0)
    params["window_overlap"] = trial.suggest_float("window_overlap", 0.0, 0.9)

    # Frequency window size — None means "use window_size"
    use_freq_window = trial.suggest_categorical("use_frequency_window", [False, True])
    if use_freq_window:
        params["frequency_window_size"] = trial.suggest_float(
            "frequency_window_size", 0.5, 5.0
        )
    else:
        params["frequency_window_size"] = None

    return params


# ── Known sensor names in the dataset ─────────────────────────────────────────
# HeartRate (1-channel bpm, 10 Hz) — slow physiological signal.
# Motion sensors (3-axis xyz, 50 Hz) — fast movement.
_MOTION_SENSORS: tuple[str, ...] = (
    "Accelerometer",
    "AccelerometerUncalibrated",
    "Gyroscope",
    "GyroscopeUncalibrated",
    "TotalAcceleration",
    "WatchAccelerometer",
    "WatchGravity",
    "WatchGyroscope",
    "WatchTotalAcceleration",
)
_ALL_SENSORS: tuple[str, ...] = ("HeartRate",) + _MOTION_SENSORS


def suggest_sensor_window_params(trial: optuna.Trial) -> dict:
    """Sample per-sensor window-size overrides.

    Different sensors operate at very different timescales so they benefit
    from independent context-window sizes:

    * **HeartRate** (10 Hz, bpm) — slow physiological response.
      Window range: **5–30 seconds**.
    * **Motion sensors** (50 Hz, xyz) — fast movement intensity.
      Window range: **0.5–5 seconds** (independent of the global
      ``window_size`` when ``sensor_windows`` is active).

    The returned dict contains a boolean flag ``use_sensor_windows`` plus
    float window sizes for each overridden sensor.  When a sensor is not
    present in the output, the global ``window_size`` fallback applies.
    """
    params: dict = {}

    params["use_sensor_windows"] = trial.suggest_categorical(
        "use_sensor_windows", [False, True]
    )

    if params["use_sensor_windows"]:
        # ── HeartRate (slow: 5–30 s) ────────────────────────────────────
        params["sensor_window_HeartRate"] = trial.suggest_float(
            "sensor_window_HeartRate", 5.0, 30.0
        )

        # ── Motion sensors (fast: 0.5–5 s) ──────────────────────────────
        # Offered as a single shared value so the search stays efficient.
        params["sensor_window_motion"] = trial.suggest_float(
            "sensor_window_motion", 0.5, 5.0
        )

    return params


def suggest_feature_params(trial: optuna.Trial) -> dict:
    """Sample boolean feature-domain toggles."""
    return {
        "time_domain": trial.suggest_categorical("time_domain", [True, False]),
        "frequency_domain": trial.suggest_categorical("frequency_domain", [True, False]),
        "statistical": trial.suggest_categorical("statistical", [True, False]),
        "magnitude_channels": trial.suggest_categorical("magnitude_channels", [True, False]),
        "cross_sensor_features": trial.suggest_categorical("cross_sensor_features", [True, False]),
    }


def suggest_feature_selection_params(trial: optuna.Trial) -> dict:
    """Sample feature-selection hyperparameters."""
    params: dict = {}

    # Choose a subset of selection methods
    all_methods = ["variance", "correlation", "mutual_information", "rfe", "l1"]
    n_methods = trial.suggest_int("n_selection_methods", 1, len(all_methods))
    chosen = trial.suggest_categorical("selection_methods", all_methods)
    # Optuna's categorical only picks 1, so we build a variable-length set
    # via sequential int choices
    selected: list[str] = []
    for i in range(n_methods):
        method = trial.suggest_categorical(f"sel_method_{i}", all_methods)
        if method not in selected:
            selected.append(method)
    params["selection_methods"] = selected if selected else ["variance"]

    params["variance_threshold"] = trial.suggest_float(
        "variance_threshold", 0.001, 0.05, log=True
    )
    params["correlation_threshold"] = trial.suggest_float(
        "correlation_threshold", 0.80, 0.99
    )
    params["mutual_info_k"] = trial.suggest_int("mutual_info_k", 5, 50, step=5)
    params["rfe_k"] = trial.suggest_int("rfe_k", 5, 30, step=5)

    return params


def suggest_classical_model_params(
    trial: optuna.Trial, model_name: str = "random_forest"
) -> dict:
    """Sample classical ML hyperparameters.

    Parameters
    ----------
    model_name :
        ``"random_forest"`` or ``"xgboost"``.
    """
    params: dict = {"model_name": model_name}

    if model_name == "random_forest":
        params["n_estimators"] = trial.suggest_int("rf_n_estimators", 50, 500, step=25)
        params["max_depth"] = trial.suggest_int("rf_max_depth", 3, 30)
        params["min_samples_split"] = trial.suggest_int("rf_min_samples_split", 2, 20)
        params["min_samples_leaf"] = trial.suggest_int("rf_min_samples_leaf", 1, 10)
        params["max_features"] = trial.suggest_categorical(
            "rf_max_features", ["sqrt", "log2", None]
        )
        # Bootstrap choice
        params["bootstrap"] = trial.suggest_categorical("rf_bootstrap", [True, False])

    elif model_name == "xgboost":
        params["n_estimators"] = trial.suggest_int("xgb_n_estimators", 50, 500, step=25)
        params["learning_rate"] = trial.suggest_float(
            "xgb_learning_rate", 0.01, 0.3, log=True
        )
        params["max_depth"] = trial.suggest_int("xgb_max_depth", 3, 12)
        params["subsample"] = trial.suggest_float("xgb_subsample", 0.6, 1.0)
        params["colsample_bytree"] = trial.suggest_float(
            "xgb_colsample_bytree", 0.6, 1.0
        )
        params["gamma"] = trial.suggest_float("xgb_gamma", 0.0, 5.0)
        params["reg_alpha"] = trial.suggest_float("xgb_reg_alpha", 0.0, 2.0, log=True)
        params["reg_lambda"] = trial.suggest_float(
            "xgb_reg_lambda", 0.0, 2.0, log=True
        )
        params["min_child_weight"] = trial.suggest_int("xgb_min_child_weight", 1, 10)

    return params


def suggest_deep_model_params(trial: optuna.Trial) -> dict:
    """Sample deep-learning hyperparameters (LSTM / TCN)."""
    params: dict = {}

    params["model_type"] = trial.suggest_categorical("model_type", ["lstm", "tcn"])

    # Shared params
    params["hidden_size"] = trial.suggest_int("hidden_size", 32, 256, step=16)
    params["num_layers"] = trial.suggest_int("num_layers", 1, 4)
    params["dropout_probability"] = trial.suggest_float(
        "dropout_probability", 0.0, 0.5
    )
    params["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 1e-2, log=True
    )
    params["num_epochs"] = trial.suggest_int("num_epochs", 20, 200, step=10)
    params["batch_size"] = trial.suggest_categorical(
        "batch_size", [16, 32, 64, 128]
    )

    # TCN-specific
    if params["model_type"] == "tcn":
        params["kernel_size"] = trial.suggest_int("kernel_size", 2, 5)
        channel_config = trial.suggest_categorical(
            "channel_config", ["small", "medium", "large"]
        )
        channel_map = {
            "small": [16, 32, 32],
            "medium": [32, 64, 64],
            "large": [64, 128, 128],
        }
        params["channel_sizes"] = channel_map[channel_config]

    return params
