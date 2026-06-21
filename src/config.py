"""Configuration — plain Python dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from consts import SRC


# ── Known motion sensors ──────────────────────────────────────────────────────
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


@dataclass
class PreprocessingConfig:
    filter_method: Literal["butterworth", "moving_average", "savitzky_golay"] = "savitzky_golay"
    filter_cutoff: float = 3.408
    filter_order: int = 4
    filter_type: Literal["low", "high", "band"] = "band"
    savitzky_golay_window_length: int = 17
    savitzky_golay_polyorder: int = 1
    imputation_method: Literal["interpolate", "ffill", "knn"] = "interpolate"
    imputation_max_gap: int = 14
    resample_rule: str = "100ms"


@dataclass
class SensorWindowConfig:
    base_window_seconds: float
    freq_window_seconds: float | None = None


@dataclass
class FeatureConfig:
    window_size: float = 2.315
    window_overlap: float = 0.501
    frequency_window_size: float | None = 3.574
    frequency_bands: tuple[tuple[float, float], ...] = (
        (0.5, 1.282),
        (1.282, 18.077),
        (18.077, 25.837),
        (25.837, 30.0),
    )
    rolloff_fraction: float = 0.85
    sensor_windows: dict[str, SensorWindowConfig] = field(default_factory=lambda: {
        "HeartRate": SensorWindowConfig(base_window_seconds=25.073),
        **{s: SensorWindowConfig(base_window_seconds=2.737) for s in _MOTION_SENSORS},
    })
    magnitude_channels: bool = False
    cross_sensor_features: bool = False
    time_domain: bool = True
    frequency_domain: bool = True
    statistical: bool = True
    selection_methods: tuple[str, ...] = ("variance", "correlation")


@dataclass
class RandomForestConfig:
    """Random Forest hyperparameters (defaults = scikit-learn defaults)."""
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | None = "sqrt"
    bootstrap: bool = True


@dataclass
class XGBoostConfig:
    """XGBoost hyperparameters (defaults = library defaults)."""
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 6
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    min_child_weight: int = 1


@dataclass
class LSTMConfig:
    """LSTM hyperparameters (from best tuning trial)."""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    num_epochs: int = 100
    batch_size: int = 32


@dataclass
class TCNConfig:
    """TCN hyperparameters (from best tuning trial)."""
    hidden_size: int = 176
    num_layers: int = 4
    dropout: float = 0.414
    kernel_size: int = 5
    channel_config: Literal["small", "medium", "large"] = "small"
    learning_rate: float = 0.0035
    num_epochs: int = 110
    batch_size: int = 32


@dataclass
class ModelConfig:
    test_size: float = 0.2
    cv_folds: int = 5
    oos_participant: str = "Kim"

    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    tcn: TCNConfig = field(default_factory=TCNConfig)


@dataclass
class Config:
    data_root: Path = SRC / "data"
    raw_dir: Path = data_root / "raw"
    processed_dir: Path = data_root / "processed"
    features_dir: Path = data_root / "features"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
