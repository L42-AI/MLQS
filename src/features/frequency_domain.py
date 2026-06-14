"""Frequency-domain feature extractors — FFT-based spectral features."""

from typing import Callable

import numpy as np
from scipy import signal

# Default frequency bands for band-energy ratios.
_DEFAULT_FREQUENCY_BANDS: list[tuple[float, float]] = [
    (0.5, 4),
    (4, 8),
    (8, 13),
    (13, 30),
]

# Small constant to avoid division-by-zero in spectral computations.
_EPSILON: float = 1e-12


def _compute_power_spectral_density(values: np.ndarray, sample_rate_hz: float):
    """Power spectral density via Welch's method, fallback to FFT for short segments."""
    segments = min(256, len(values) // 2)
    if segments < 8:
        frequencies = np.fft.rfftfreq(len(values), d=1.0 / sample_rate_hz)
        psd_values = np.abs(np.fft.rfft(values)) ** 2
        return frequencies, psd_values
    return signal.welch(values, fs=sample_rate_hz, nperseg=segments)


# ---------------------------------------------------------------------------
# Individual feature functions (each computes PSD independently).
# Kept for backward compatibility and direct use.
# ---------------------------------------------------------------------------

def dominant_frequency(values, fs=100.0):
    frequencies, psd_values = _compute_power_spectral_density(values, fs)
    return float(frequencies[np.argmax(psd_values)])


def spectral_centroid(values, fs=100.0):
    frequencies, psd_values = _compute_power_spectral_density(values, fs)
    return float(np.sum(frequencies * psd_values) / (np.sum(psd_values) + _EPSILON))


def spectral_rolloff(values, fs=100.0, rolloff_fraction=0.85):
    frequencies, psd_values = _compute_power_spectral_density(values, fs)
    cumulative_energy = np.cumsum(psd_values)
    rolloff_index = np.searchsorted(cumulative_energy, rolloff_fraction * cumulative_energy[-1])
    return float(frequencies[min(rolloff_index, len(frequencies) - 1)])


def spectral_entropy(values, fs=100.0):
    _, psd_values = _compute_power_spectral_density(values, fs)
    probability_distribution = psd_values / (np.sum(psd_values) + _EPSILON)
    return float(-np.sum(probability_distribution * np.log2(probability_distribution + _EPSILON)))


def compute_band_energy_ratios(values, frequency_bands=None, fs=100.0):
    if frequency_bands is None:
        frequency_bands = _DEFAULT_FREQUENCY_BANDS
    frequencies, psd_values = _compute_power_spectral_density(values, fs)
    total_energy = np.sum(psd_values) + _EPSILON
    return {
        f"{low_freq}-{high_freq}hz": float(
            np.sum(psd_values[(frequencies >= low_freq) & (frequencies <= high_freq)]) / total_energy
        )
        for low_freq, high_freq in frequency_bands
    }


FEATURE_REGISTRY: dict[str, Callable] = {
    "dominant_frequency": dominant_frequency,
    "spectral_centroid": spectral_centroid,
    "spectral_rolloff": spectral_rolloff,
    "spectral_entropy": spectral_entropy,
}


# ---------------------------------------------------------------------------
# Batched computation — single PSD pass for all frequency features.
# ---------------------------------------------------------------------------

def compute_all_frequency_features(
    values: np.ndarray,
    fs: float = 100.0,
    rolloff_fraction: float = 0.85,
    frequency_bands: list[tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Compute all frequency-domain features from a single PSD pass.

    Calling this once instead of iterating ``FEATURE_REGISTRY`` + ``compute_band_energy_ratios``
    avoids 4 redundant PSD (Welch/FFT) computations per invocation.

    Returns a dict with the same keys the individual functions produce:
      ``dominant_frequency``, ``spectral_centroid``, ``spectral_rolloff``,
      ``spectral_entropy``, plus band-energy keys like ``"0.5-4hz"``.
    """
    if frequency_bands is None:
        frequency_bands = _DEFAULT_FREQUENCY_BANDS

    frequencies, psd_values = _compute_power_spectral_density(values, fs)

    total_power = np.sum(psd_values) + _EPSILON

    # --- Dominant frequency ---
    features: dict[str, float] = {
        "dominant_frequency": float(frequencies[np.argmax(psd_values)]),
    }

    # --- Spectral centroid ---
    # Weighted mean of the frequency spectrum.
    features["spectral_centroid"] = float(
        np.sum(frequencies * psd_values) / total_power,
    )

    # --- Spectral rolloff ---
    # Frequency below which *rolloff_fraction* of the total energy lies.
    cumulative_energy = np.cumsum(psd_values)
    rolloff_energy = rolloff_fraction * cumulative_energy[-1]
    rolloff_index = np.searchsorted(cumulative_energy, rolloff_energy)
    features["spectral_rolloff"] = float(
        frequencies[min(rolloff_index, len(frequencies) - 1)],
    )

    # --- Spectral entropy ---
    # Shannon entropy of the normalised PSD (measure of spectral flatness).
    normalised_psd = psd_values / total_power
    features["spectral_entropy"] = float(
        -np.sum(normalised_psd * np.log2(normalised_psd + _EPSILON)),
    )

    # --- Band energy ratios ---
    # Fraction of total spectral energy within each predefined band.
    for low_freq, high_freq in frequency_bands:
        band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        features[f"{low_freq}-{high_freq}hz"] = float(
            np.sum(psd_values[band_mask]) / total_power,
        )

    return features
