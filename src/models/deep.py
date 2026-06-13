"""Deep learning models (LSTM / TCN) in PyTorch."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class LSTMClassifier(nn.Module):
    """Multi-layer LSTM with a linear classification head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout_probability: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_probability if num_layers > 1 else 0.0,
        )
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout_probability),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_sequence):
        _, (hidden_state, _) = self.lstm(input_sequence)
        return self.classification_head(hidden_state[-1])


class _TCNResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dilation_rate,
        kernel_size=3,
        dropout_probability=0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation_rate
        self.convolution_1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation_rate
        )
        self.convolution_2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation_rate
        )
        self.dropout_layer = nn.Dropout(dropout_probability)
        self.residual_connection = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, input_tensor):
        residual = self.residual_connection(input_tensor)
        output = F.relu(self.convolution_1(input_tensor))
        output = self.dropout_layer(output)
        output = F.relu(self.convolution_2(output))
        output = self.dropout_layer(output)
        size_diff = output.size(2) - residual.size(2)
        if size_diff > 0:
            output = output[:, :, :-size_diff]
        return F.relu(output + residual)


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network with dilated convolutions."""

    def __init__(
        self,
        input_size: int,
        channel_sizes=None,
        num_classes: int = 2,
        kernel_size: int = 3,
        dropout_probability: float = 0.2,
    ):
        super().__init__()
        channel_sizes = channel_sizes or [32, 64, 64]
        tcn_layers = []
        current_channels = input_size
        for layer_index, out_channels in enumerate(channel_sizes):
            tcn_layers.append(
                _TCNResidualBlock(
                    current_channels,
                    out_channels,
                    2 ** layer_index,
                    kernel_size,
                    dropout_probability,
                )
            )
            current_channels = out_channels
        self.tcn_blocks = nn.Sequential(*tcn_layers)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.classification_head = nn.Linear(channel_sizes[-1], num_classes)

    def forward(self, input_tensor):
        tcn_output = self.tcn_blocks(input_tensor.permute(0, 2, 1))
        pooled = self.global_pooling(tcn_output).squeeze(-1)
        return self.classification_head(pooled)


@dataclass
class TrainingHistory:
    training_losses: list[float]
    validation_losses: list[float] = None
    validation_accuracies: list[float] = None


def prepare_sequences(
    feature_matrix: np.ndarray | pd.DataFrame,
    label_array: np.ndarray | pd.Series,
    sequence_length: int,
    batch_size: int = 32,
):
    feature_array = np.asarray(feature_matrix)
    label_array = np.asarray(label_array)
    num_sequences = len(feature_array) // sequence_length
    usable_samples = num_sequences * sequence_length

    feature_array = feature_array[:usable_samples].reshape(
        num_sequences, sequence_length, feature_array.shape[1]
    )
    label_array = label_array[:usable_samples]

    sequence_labels = np.array(
        [
            np.bincount(
                label_array[i * sequence_length : (i + 1) * sequence_length].astype(int)
            ).argmax()
            for i in range(num_sequences)
        ]
    )

    dataset = TensorDataset(
        torch.tensor(feature_array, dtype=torch.float32),
        torch.tensor(sequence_labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_deep_model(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader | None = None,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cpu",
) -> TrainingHistory:
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = TrainingHistory(training_losses=[])
    has_validation = validation_loader is not None
    if has_validation:
        history.validation_losses = []
        history.validation_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_training_loss = 0.0

        for batch_inputs, batch_labels in training_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch_inputs), batch_labels)
            loss.backward()
            optimizer.step()
            total_training_loss += loss.item()

        history.training_losses.append(total_training_loss / len(training_loader))

        if has_validation:
            model.eval()
            validation_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            with torch.no_grad():
                for batch_inputs, batch_labels in validation_loader:
                    batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                    logits = model(batch_inputs)
                    validation_loss += F.cross_entropy(logits, batch_labels).item()
                    correct_predictions += (logits.argmax(1) == batch_labels).sum().item()
                    total_samples += batch_labels.size(0)

            history.validation_losses.append(validation_loss / len(validation_loader))
            history.validation_accuracies.append(correct_predictions / total_samples)

    return history


def build_deep_classifier(
    model_type: str = "lstm",
    input_size: int = 1,
    num_classes: int = 2,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout_probability: float = 0.2,
) -> nn.Module:
    if model_type == "lstm":
        return LSTMClassifier(input_size, hidden_size, num_layers, num_classes, dropout_probability)
    elif model_type == "tcn":
        return TCNClassifier(input_size, num_classes=num_classes, dropout_probability=dropout_probability)
    raise ValueError(f"Unknown deep model type: {model_type}")
