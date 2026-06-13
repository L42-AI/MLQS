"""Time-domain feature extractors — each takes a 1-D array, returns a float."""

import numpy as np


def signal_mean(values): return float(np.mean(values))
def signal_std(values): return float(np.std(values, ddof=1))
def signal_variance(values): return float(np.var(values, ddof=1))
def signal_rms(values): return float(np.sqrt(np.mean(values ** 2)))
def signal_minimum(values): return float(np.min(values))
def signal_maximum(values): return float(np.max(values))
def signal_median(values): return float(np.median(values))
def signal_peak_to_peak(values): return float(np.ptp(values))


def zero_crossing_rate(values):
    if len(values) < 2:
        return 0.0
    return float(np.sum(np.diff(np.signbit(values))) / (len(values) - 1))


def mean_crossing_rate(values):
    if len(values) < 2:
        return 0.0
    return float(np.sum(np.diff(np.signbit(values - np.mean(values)))) / (len(values) - 1))


def absolute_integral(values):
    return float(np.trapezoid(np.abs(values)))


def linear_slope(values):
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


FEATURE_REGISTRY: dict[str, callable] = {
    "mean": signal_mean,
    "std": signal_std,
    "variance": signal_variance,
    "rms": signal_rms,
    "min": signal_minimum,
    "max": signal_maximum,
    "median": signal_median,
    "peak_to_peak": signal_peak_to_peak,
    "zero_crossing_rate": zero_crossing_rate,
    "mean_crossing_rate": mean_crossing_rate,
    "abs_integral": absolute_integral,
    "slope": linear_slope,
}
