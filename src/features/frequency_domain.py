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


# ── Fully vectorised batch variant (2D) ──────────────────────────────────────
# Instead of calling compute_all_frequency_features once per window, we
# compute the FFT of ALL windows in a single np.fft.rfft() call over
# axis=1, then compute every feature from the resulting 2-D PSD matrix
# using pure numpy vectorised operations.  This eliminates 1M+ Python
# loop iterations and 1M+ calls to signal.welch / np.fft.rfft.


def compute_batch_frequency_features(
    windows: np.ndarray,
    fs: float = 100.0,
    rolloff_fraction: float = 0.85,
    frequency_bands: list[tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Compute all frequency-domain features for *all* windows at once.

    Parameters
    ----------
    windows : np.ndarray, shape ``(num_windows, window_len)``
        Stack of windowed signal segments.
    fs :
        Sampling rate in Hz.
    rolloff_fraction :
        Fraction of total energy for the roll-off point.
    frequency_bands :
        List of ``(low, high)`` tuples for band-energy ratios.
        Defaults to ``_DEFAULT_FREQUENCY_BANDS``.

    Returns
    -------
    dict[str, np.ndarray]
        Each value has shape ``(num_windows,)`` with the same keys as
        :func:`compute_all_frequency_features`:
        ``dominant_frequency``, ``spectral_centroid``, ``spectral_rolloff``,
        ``spectral_entropy``, plus band-energy keys like ``"0.5-4hz"``.
    """
    if frequency_bands is None:
        frequency_bands = _DEFAULT_FREQUENCY_BANDS

    n_windows, win_len = windows.shape

    # ── Batch power-spectral-density ───────────────────────────────────────
    # One call to signal.welch (or np.fft.rfft for very short windows)
    # instead of N individual calls.  The results are bit-identical.
    segments = min(256, win_len // 2)
    if segments < 8:
        # Fallback: raw FFT (same as _compute_power_spectral_density)
        fft_vals = np.fft.rfft(windows, axis=1)          # (N, n_freqs)
        psd = np.abs(fft_vals, dtype=np.float64) ** 2
        freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)
    else:
        # Welch's method — vectorised across all windows via axis=-1
        freqs, psd = signal.welch(windows, fs=fs, nperseg=segments, axis=-1)

    # Avoid division by zero
    total_power = np.sum(psd, axis=1, keepdims=True) + _EPSILON  # (N, 1)
    n_freqs = psd.shape[1]

    features: dict[str, np.ndarray] = {}

    # ── Dominant frequency ────────────────────────────────────────────────
    max_idx = np.argmax(psd, axis=1)                     # (N,)
    features["dominant_frequency"] = freqs[max_idx]

    # ── Spectral centroid ─────────────────────────────────────────────────
    # Weighted mean of the spectrum, per window.
    features["spectral_centroid"] = (
        np.sum(freqs * psd, axis=1) / total_power[:, 0]
    )

    # ── Spectral rolloff ──────────────────────────────────────────────────
    # Frequency below which *rolloff_fraction* of energy lies.
    # Vectorised: the rolloff index is the first position where cumulative
    # energy >= target.  ``np.sum(cum < target, axis=1)`` counts how many
    # positions precede it — equivalent to ``searchsorted`` but 2-D safe.
    cum_energy = np.cumsum(psd, axis=1)                  # (N, n_freqs)
    rolloff_target = rolloff_fraction * cum_energy[:, -1:]  # (N, 1)
    rolloff_idx = np.sum(cum_energy < rolloff_target, axis=1)  # (N,)
    rolloff_idx = np.clip(rolloff_idx, 0, n_freqs - 1)
    features["spectral_rolloff"] = freqs[rolloff_idx]

    # ── Spectral entropy ──────────────────────────────────────────────────
    # Shannon entropy of the normalised PSD.
    norm_psd = psd / total_power                          # (N, n_freqs)
    features["spectral_entropy"] = -np.sum(
        norm_psd * np.log2(norm_psd + _EPSILON), axis=1
    )

    # ── Band energy ratios ────────────────────────────────────────────────
    for low_freq, high_freq in frequency_bands:
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)  # (n_freqs,)
        features[f"{low_freq}-{high_freq}hz"] = (
            np.sum(psd[:, band_mask], axis=1) / total_power[:, 0]
        )

    return features
