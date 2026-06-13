#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from data.loader import load_all_experiment_sensors
from features.selection import run_selection_pipeline
from pipeline.builder import run_feature_pipeline

experiment_config = Config()

sensor_data = load_all_experiment_sensors(
    experiment_config.raw_dir,
    resample_rule=experiment_config.preprocessing.resample_rule,
)
print(f"Loaded: {sensor_data.shape}")

pipeline_result = run_feature_pipeline(sensor_data, experiment_config)
print(f"Windows:  {pipeline_result.feature_matrix.shape[0]}")
print(f"Features: {pipeline_result.feature_matrix.shape[1]}")

extracted_labels = pipeline_result.labels
if extracted_labels is not None and experiment_config.features.selection_methods:
    selected_features = run_selection_pipeline(
        pipeline_result.feature_matrix,
        extracted_labels,
        selection_methods=list(experiment_config.features.selection_methods),
    )
    print(f"After selection: {selected_features.shape}")
else:
    selected_features = pipeline_result.feature_matrix

output_path = experiment_config.features_dir / "features.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
selected_features.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
print(f"Feature names ({len(selected_features.columns)}):")
for feature_name in selected_features.columns[:12]:
    print(f"  - {feature_name}")
if len(selected_features.columns) > 12:
    print(f"  ... and {len(selected_features.columns) - 12} more")
