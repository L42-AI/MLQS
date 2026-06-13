"""Frequency-domain feature extractors — FFT-based spectral features."""

from typing import Callable

import numpy as np
from scipy import signal


def _compute_power_spectral_density(values: np.ndarray, sample_rate_hz: float):
    """Power spectral density via Welch's method, fallback to FFT for short segments."""
    segments = min(256, len(values) // 2)
    if segments < 8:
        frequencies = np.fft.rfftfreq(len(values), d=1.0 / sample_rate_hz)
        psd_values = np.abs(np.fft.rfft(values)) ** 2
        return frequencies, psd_values
    return signal.welch(values, fs=sample_rate_hz, nperseg=segments)


def dominant_frequency(values, fs=100.0):
    frequencies, psd_values = _compute_power_spectral_density(values, fs)
    return float(frequencies[np.argmax(psd_values)])


def spectral_centroid(values, fs=100.0):
    frequencies, psd_values = _compute_power_spectral_density(values, fs)
    return float(np.sum(frequencies * psd_values) / (np.sum(psd_values) + 1e-12))


def spectral_rolloff(values, fs=100.0, rolloff_fraction=0.85):
    frequencies, psd_values = _compute_power_spectral_density(values, fs)
    cumulative_energy = np.cumsum(psd_values)
    rolloff_index = np.searchsorted(cumulative_energy, rolloff_fraction * cumulative_energy[-1])
    return float(frequencies[min(rolloff_index, len(frequencies) - 1)])


def spectral_entropy(values, fs=100.0):
    _, psd_values = _compute_power_spectral_density(values, fs)
    probability_distribution = psd_values / (np.sum(psd_values) + 1e-12)
    return float(-np.sum(probability_distribution * np.log2(probability_distribution + 1e-12)))


def compute_band_energy_ratios(values, frequency_bands=None, fs=100.0):
    if frequency_bands is None:
        frequency_bands = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
    frequencies, psd_values = _compute_power_spectral_density(values, fs)
    total_energy = np.sum(psd_values) + 1e-12
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
