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
