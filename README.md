# MLQS — Machine Learning for the Quantified Self

End-to-end ML pipeline for sensor data collected via the [phyphox](https://phyphox.org/) mobile app. Processes phone and watch sensor CSV exports through preprocessing, feature engineering, feature selection, and model evaluation.

## Project structure

```
MLQS/
├── pyproject.toml          # dependencies, Python 3.12+
├── data/
│   ├── raw/                # phyphox CSV exports (gitignored)
│   ├── processed/          # cleaned & imputed data (gitignored)
│   └── features/           # engineered feature matrices (gitignored)
└── src/
    ├── consts.py           # SRC = absolute path to src/
    ├── config.py           # Config, PreprocessingConfig, FeatureConfig, ModelConfig
    ├── labels.py           # Label enum (SILENCE / SOFT / HARD)
    ├── schema.py           # SensorSchema — column names, sample rates per sensor
    ├── main.py             # CLI entry point: pipeline, models, EDA
    ├── data/
    │   └── loader.py       # load_all_experiment_sensors, load_single_sensor_csv
    ├── preprocessing/
    │   ├── noise.py        # Butterworth / moving average / Savitzky–Golay filters
    │   ├── missing.py      # interpolate, forward-fill, KNN imputation
    │   └── resample.py     # synchronise_sensor_frames, resample_to_uniform_grid
    ├── features/
    │   ├── windowing.py    # sliding-window segmentation
    │   ├── time_domain.py  # mean, variance, skew, kurtosis, zero-crossing-rate …
    │   ├── frequency_domain.py  # FFT features + band-energy ratios
    │   ├── statistical.py       # entropy, iqr, mad, percentile range …
    │   ├── engineering.py       # orchestrator: window → extract → merge
    │   └── selection.py         # variance threshold + correlation filter
    ├── models/
    │   ├── classical.py    # Random Forest, SVM, XGBoost
    │   ├── deep.py         # LSTM / TCN (PyTorch)
    │   └── evaluation.py   # accuracy, F1, confusion matrix, model ranking
    ├── pipeline/
    │   └── builder.py      # run_feature_pipeline — full preprocess+features flow
    ├── utils/
    │   ├── convert.py      # time-string ↔ frequency conversion helpers
    │   └── viz.py          # plotting helpers for EDA, features, evaluation
    └── notebooks/
        ├── 01_eda.py              # exploratory data analysis
        ├── 02_preprocessing.py    # noise filter + imputation walkthrough
        ├── 03_feature_engineering.py  # full feature pipeline run
        ├── 04_classical_ml.py     # RF / SVM / XGBoost train + compare
        └── 05_deep_learning.py    # LSTM / TCN train + evaluate
```

## Setup

```bash
uv sync                    # install dependencies (see pyproject.toml)
```

Requires Python ≥ 3.12. Dependencies: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `torch`, `xgboost`.

## How to run

All paths are absolute, rooted at `src/consts.SRC`. The `raw_dir`, `processed_dir`, and `features_dir` default to `src/data/{raw,processed,features}` and can be overridden on `Config()`.

### Entry-point

```bash
cd src

# Run as script (bootstrap adds src/ to sys.path automatically)
uv run main.py
uv run main.py --model classical
uv run main.py --model deep

# Run as module (equivalent — Python adds src/ to sys.path via -m)
python -m main --model classical
python -m main --eda
```

### Notebooks

Each notebook in `src/notebooks/` is a standalone Python script (not a Jupyter `.ipynb`):

```bash
cd src

# Run as script
uv run notebooks/01_eda.py
uv run notebooks/02_preprocessing.py

# Run as module
python -m notebooks.03_feature_engineering
python -m notebooks.04_classical_ml
python -m notebooks.05_deep_learning
```

## Configuration

Plain Python dataclasses — no YAML. All config lives in `src/config.py`:

```python
from config import Config

cfg = Config()
cfg.raw_dir                         # pathlib.Path → src/data/raw
cfg.preprocessing.filter_method     # "butterworth" | "moving_average" | "savitzky_golay"
cfg.preprocessing.imputation_method # "interpolate" | "ffill" | "knn"
cfg.features.window_size            # seconds (default 2.0)
cfg.features.selection_methods      # ("variance", "correlation")
cfg.models.deep_model               # "lstm" | "tcn"
```

Override any field:

```python
cfg = Config(
    raw_dir=Path("/custom/path"),
    models=ModelConfig(deep_epochs=50),
)
```

## Pipeline flow

```
phyphox CSVs
    │
    ▼
load_all_experiment_sensors()     ← merges phone + watch by time, detects labels
    │
    ▼
resample_to_uniform_grid()        ← 100 ms grid (configurable)
    │
    ▼
apply_filter_to_columns()         ← noise removal
    │
    ▼
impute (interpolate / ffill / knn)  ← missing values
    │
    ▼
create_sliding_windows()          ← 2 s windows, 50 % overlap
    │
    ▼
extract_features_from_windows()   ← time-domain + frequency-domain + statistical
    │
    ▼
run_selection_pipeline()          ← variance threshold → correlation filter
    │
    ▼
train model (RF / SVM / XGBoost / LSTM / TCN)
```

The entire preprocess → feature pipeline is a single call:

```python
from config import Config
from data.loader import load_all_experiment_sensors
from pipeline.builder import run_feature_pipeline

cfg = Config()
data = load_all_experiment_sensors(cfg.raw_dir)
result = run_feature_pipeline(data, cfg)
# result.feature_matrix  → pd.DataFrame
# result.labels          → pd.Series | None
```

## Data format

Sensor CSV exports from phyphox share a common structure:

| Column | Description |
|---|---|
| `time` | wall-clock time (string) |
| `seconds_elapsed` | elapsed seconds (float) |
| `x`, `y`, `z` | 3-axis sensor readings (Accelerometer, Gyroscope, etc.) |
| `yaw`, `pitch`, `roll` | orientation (WatchOrientation) |
| `qx`, `qy`, `qz`, `qw` | quaternion orientation (WatchOrientation) |
| `bpm` | heart rate (HeartRate) |

Activity labels are inferred from directory names: directories containing `SILENCE`, `SOFT`, or `HARD` map to `Label.{SILENCE,SOFT,HARD}`.

## Design notes

- **Flat functional style** — no classes (except dataclasses), no closures, no inheritance. Functions take data and config, return results.
- **Dispatch dicts** instead of match/case for imputation method lookup.
- **Feature registries** — `time_domain.FEATURE_REGISTRY`, `frequency_domain.FEATURE_REGISTRY`, `statistical.FEATURE_REGISTRY` are plain dicts mapping name → function. Adding a feature means adding one entry.
- **Single-source path** — `consts.SRC` is the absolute `src/` directory. All data directory defaults derive from it.
- **No YAML/JSON config** — one `config.py`, importable, composable, overridable.
