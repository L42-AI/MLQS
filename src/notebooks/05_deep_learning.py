#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from config import Config
from models.classical import split_train_test_data
from models.deep import build_deep_classifier, prepare_sequences, train_deep_model
from models.evaluation import compute_classification_metrics

experiment_config = Config()

feature_data = pd.read_csv(experiment_config.features_dir / "features.csv")
encoded_labels = feature_data.pop("label") if "label" in feature_data.columns else feature_data.pop(feature_data.columns[-1])
feature_matrix = feature_data

if encoded_labels.dtype == "object":
    encoded_labels = LabelEncoder().fit_transform(encoded_labels)

X_train, X_test, y_train, y_test = split_train_test_data(
    feature_matrix, encoded_labels, test_fraction=experiment_config.models.test_size
)

sequence_length = min(32, len(X_train) // 10)
training_loader = prepare_sequences(
    X_train, y_train, sequence_length, batch_size=experiment_config.models.deep_batch_size
)
test_loader = prepare_sequences(
    X_test, y_test, sequence_length, batch_size=experiment_config.models.deep_batch_size
)

device = "cuda" if torch.cuda.is_available() else "cpu"
deep_classifier = build_deep_classifier(
    model_type=experiment_config.models.deep_model,
    input_size=feature_matrix.shape[1],
    num_classes=int(encoded_labels.nunique()),
)
print(f"Model: {type(deep_classifier).__name__} on {device}")

train_deep_model(
    deep_classifier,
    training_loader,
    num_epochs=experiment_config.models.deep_epochs,
    learning_rate=experiment_config.models.deep_learning_rate,
    device=device,
)

deep_classifier.eval()
all_predictions, all_true_labels = [], []
with torch.no_grad():
    for batch_inputs, batch_labels in test_loader:
        logits = deep_classifier(batch_inputs.to(device))
        all_predictions.extend(logits.argmax(1).cpu().numpy())
        all_true_labels.extend(batch_labels.numpy())

metric_scores = compute_classification_metrics(
    np.array(all_true_labels), np.array(all_predictions)
)
print(f"Test — Acc: {metric_scores['accuracy']:.4f}  F1: {metric_scores['f1']:.4f}")
