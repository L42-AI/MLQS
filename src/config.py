"""Configuration — plain Python dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from consts import SRC


@dataclass
class PreprocessingConfig:
    filter_method: Literal["butterworth", "moving_average", "savitzky_golay"] = "butterworth"
    filter_cutoff: float = 4.0
    filter_order: int = 4
    filter_type: Literal["low", "high", "band"] = "low"
    imputation_method: Literal["interpolate", "ffill", "knn"] = "interpolate"
    imputation_max_gap: int = 5
    resample_rule: str = "100ms"


@dataclass
class SensorWindowConfig:
    """Per-sensor override for feature extraction window sizes.

    Attributes:
        base_window_seconds: Context window length for time-domain and
            statistical features.  Larger values (e.g. 30 s for heart rate)
            capture slower dynamics that would be invisible in a short window.
        freq_window_seconds: Separate (usually larger) context for
            frequency-domain features.  ``None`` → same as *base_window_seconds*.
    """
    base_window_seconds: float
    freq_window_seconds: float | None = None


@dataclass
class FeatureConfig:
    window_size: float = 2.0
    window_overlap: float = 0.5
    frequency_window_size: float | None = field(default=None)
    sensor_windows: dict[str, SensorWindowConfig] = field(default_factory=dict)
    magnitude_channels: bool = True
    cross_sensor_features: bool = True
    time_domain: bool = True
    frequency_domain: bool = True
    statistical: bool = True
    selection_methods: tuple[str, ...] = ("variance", "correlation")


@dataclass
class ModelConfig:
    test_size: float = 0.2
    cv_folds: int = 5
    deep_model: Literal["lstm", "tcn"] = "lstm"
    deep_epochs: int = 100
    deep_batch_size: int = 32
    deep_learning_rate: float = 0.001


@dataclass
class Config:
    data_root: Path = SRC / "data"
    raw_dir: Path = data_root / "raw"
    processed_dir: Path = data_root / "processed"
    features_dir: Path = data_root / "features"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
