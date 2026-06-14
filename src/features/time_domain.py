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


FEATURE_NAMES: tuple[str, ...] = (
    "mean", "std", "variance", "rms",
    "min", "max", "median", "peak_to_peak",
    "zero_crossing_rate", "mean_crossing_rate",
    "abs_integral", "slope",
)

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


# ── Fused / batched variants ────────────────────────────────────────────────


def compute_all_time_domain_features(values: np.ndarray) -> dict[str, float]:
    """Compute all 12 time-domain features in one pass, sharing intermediates.

    Avoids redundant calls to ``np.mean``, ``np.diff``, etc. that occur when
    the individual ``FEATURE_REGISTRY`` functions are called separately.
    """
    n = len(values)
    mean = np.mean(values)
    out: dict[str, float] = {}

    out["mean"] = float(mean)
    out["std"] = float((std := np.std(values, ddof=1)))
    out["variance"] = float(std * std)
    out["rms"] = float(np.sqrt(np.mean(values ** 2)))
    out["min"] = float(np.min(values))
    out["max"] = float(np.max(values))
    out["median"] = float(np.median(values))
    out["peak_to_peak"] = float(np.ptp(values))

    if n >= 2:
        zc = np.sum(np.diff(np.signbit(values)))
        out["zero_crossing_rate"] = float(zc / (n - 1))
        mc = np.sum(np.diff(np.signbit(values - mean)))
        out["mean_crossing_rate"] = float(mc / (n - 1))

    out["abs_integral"] = float(np.trapezoid(np.abs(values)))

    if n >= 2:
        # Closed-form linear slope (avoids np.polyfit overhead for tiny arrays)
        x = np.arange(n)
        sum_x = n * (n - 1) / 2
        sum_x2 = (n - 1) * n * (2 * n - 1) / 6
        sum_y = n * mean
        sum_xy = np.sum(values * x)
        denom = n * sum_x2 - sum_x * sum_x
        out["slope"] = float((n * sum_xy - sum_x * sum_y) / denom) if abs(denom) > 1e-12 else 0.0

    return out


def compute_batch_time_domain_features(
    windows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute all time-domain features for *all* windows at once.

    Parameters
    ----------
    windows : np.ndarray, shape ``(num_windows, window_length)``
        Stack of windowed signal segments (e.g. from ``sliding_window_view``).

    Returns
    -------
    dict[str, np.ndarray]
        Each value has shape ``(num_windows,)`` — one scalar per window.
    """
    n = windows.shape[1]
    mean = np.mean(windows, axis=1)

    out: dict[str, np.ndarray] = {}

    out["mean"] = mean
    out["std"] = (std := np.std(windows, axis=1, ddof=1))
    out["variance"] = std ** 2
    out["rms"] = np.sqrt(np.mean(windows ** 2, axis=1))
    out["min"] = np.min(windows, axis=1)
    out["max"] = np.max(windows, axis=1)
    out["median"] = np.median(windows, axis=1)
    out["peak_to_peak"] = np.ptp(windows, axis=1)

    if n >= 2:
        zc = np.sum(np.diff(np.signbit(windows), axis=1), axis=1)
        out["zero_crossing_rate"] = zc / (n - 1)

        mc = np.sum(
            np.diff(np.signbit(windows - mean[:, np.newaxis]), axis=1),
            axis=1,
        )
        out["mean_crossing_rate"] = mc / (n - 1)

    out["abs_integral"] = np.trapezoid(np.abs(windows), axis=1)

    if n >= 2:
        x = np.arange(n)
        sum_x = n * (n - 1) / 2
        sum_x2 = (n - 1) * n * (2 * n - 1) / 6
        sum_y = np.sum(windows, axis=1)
        sum_xy = np.sum(windows * x, axis=1)
        denom = n * sum_x2 - sum_x * sum_x
        out["slope"] = np.where(
            np.abs(denom) > 1e-12,
            (n * sum_xy - sum_x * sum_y) / denom,
            0.0,
        )

    return out
