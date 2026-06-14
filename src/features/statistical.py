"""Statistical feature extractors — distribution shape, complexity, entropy."""

from typing import Callable

import numpy as np
from scipy import stats


def skewness(values):
    if np.std(values) < 1e-12:
        return 0.0
    return float(stats.skew(values, bias=False))


def kurtosis(values):
    if np.std(values) < 1e-12:
        return -3.0
    return float(stats.kurtosis(values, bias=False))


def shannon_entropy(values, num_bins=10):
    bin_counts, _ = np.histogram(values, bins=num_bins, density=True)
    probability_distribution = bin_counts / (np.sum(bin_counts) + 1e-12)
    return float(-np.sum(probability_distribution * np.log2(probability_distribution + 1e-12)))


def hjorth_mobility(values):
    if len(values) < 3:
        return 0.0
    variance_signal = np.var(values, ddof=1)
    variance_derivative = np.var(np.diff(values), ddof=1)
    return float(np.sqrt(variance_derivative / variance_signal)) if variance_signal > 0 else 0.0


def hjorth_complexity(values):
    if len(values) < 4:
        return 0.0
    mobility = hjorth_mobility(values)
    mobility_of_derivative = hjorth_mobility(np.diff(values))
    return float(mobility_of_derivative / mobility) if mobility > 0 else 0.0


def interquartile_range(values):
    return float(np.percentile(values, 75) - np.percentile(values, 25))


FEATURE_REGISTRY: dict[str, Callable] = {
    "skewness": skewness,
    "kurtosis": kurtosis,
    "shannon_entropy": shannon_entropy,
    "hjorth_mobility": hjorth_mobility,
    "hjorth_complexity": hjorth_complexity,
    "interquartile_range": interquartile_range,
}


# ── Batched variants — operate on (num_windows, window_len) arrays ──────────


def compute_batch_statistical_features(
    windows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute all statistical features for *all* windows at once.

    Parameters
    ----------
    windows : np.ndarray, shape ``(num_windows, window_len)``
        Stack of windowed signal segments.

    Returns
    -------
    dict[str, np.ndarray]
        Each value has shape ``(num_windows,)``.
    """
    n_windows, n = windows.shape
    out: dict[str, np.ndarray] = {}

    # Skewness and kurtosis (vectorized via scipy)
    std_ = np.std(windows, axis=1, ddof=1)
    mask = std_ > 1e-12

    skew = np.full(n_windows, 0.0)
    if mask.any():
        skew[mask] = stats.skew(windows[mask], axis=1, bias=False)
    out["skewness"] = skew

    kurt = np.full(n_windows, -3.0)
    if mask.any():
        kurt[mask] = stats.kurtosis(windows[mask], axis=1, bias=False)
    out["kurtosis"] = kurt

    # Hjorth mobility & complexity — vectorized via diff
    if n >= 3:
        diff1 = np.diff(windows, axis=1)           # shape (N, n-1)
        var_signal = np.var(windows, axis=1, ddof=1)
        var_deriv = np.var(diff1, axis=1, ddof=1)
        mobility = np.where(var_signal > 0, np.sqrt(var_deriv / var_signal), 0.0)
        out["hjorth_mobility"] = mobility

        if n >= 4:
            diff2 = np.diff(diff1, axis=1)
            var_deriv2 = np.var(diff2, axis=1, ddof=1)
            mob_deriv = np.where(
                (var_deriv2 > 0) & (var_deriv > 0),
                np.sqrt(var_deriv2 / var_deriv),
                0.0,
            )
            out["hjorth_complexity"] = np.where(mobility > 0, mob_deriv / mobility, 0.0)
        else:
            out["hjorth_complexity"] = np.zeros(n_windows)
    else:
        out["hjorth_mobility"] = np.zeros(n_windows)
        out["hjorth_complexity"] = np.zeros(n_windows)

    # Interquartile range — vectorized percentile
    q75 = np.percentile(windows, 75, axis=1)
    q25 = np.percentile(windows, 25, axis=1)
    out["interquartile_range"] = q75 - q25

    # Shannon entropy — still per-window (histogram is hard to vectorize)
    shannon = np.zeros(n_windows)
    for i in range(n_windows):
        counts, _ = np.histogram(windows[i], bins=10, density=True)
        prob = counts / (np.sum(counts) + 1e-12)
        shannon[i] = -np.sum(prob * np.log2(prob + 1e-12))
    out["shannon_entropy"] = shannon

    return out
