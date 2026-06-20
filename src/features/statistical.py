"""Statistical feature extractors — distribution shape, complexity, entropy."""

from typing import Callable

import numpy as np
from numba import njit
from scipy import stats


# ── Numba-JIT kernels (fused per-window loops) ──────────────────────────────


@njit
def _njit_skew_kurtosis(
    windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute adjusted Fisher-Pearson skewness & excess kurtosis.

    Matches ``scipy.stats.skew(w, bias=False)`` and
    ``scipy.stats.kurtosis(w, bias=False)`` to machine precision but runs
    ~25× faster by fusing the per-window loop.
    """
    n_windows, n = windows.shape
    skew = np.zeros(n_windows)
    kurt = np.full(n_windows, -3.0)

    if n < 3:
        return skew, kurt

    for i in range(n_windows):
        mean = 0.0
        for j in range(n):
            mean += windows[i, j]
        mean /= n

        m2 = 0.0
        m3 = 0.0
        m4 = 0.0
        for j in range(n):
            d = windows[i, j] - mean
            d2 = d * d
            m2 += d2
            m3 += d2 * d
            m4 += d2 * d2

        m2 /= n
        m3 /= n
        m4 /= n

        std = np.sqrt(m2)
        if std > 1e-12:
            # bias=False adjustments:
            #   skew:  g1 = sqrt(n(n-1))/(n-2) · m₃ / σ³
            sqrt_nn1 = np.sqrt(n * (n - 1))
            skew[i] = sqrt_nn1 / (n - 2) * m3 / (std * std * std)

            #   kurt:  g2 = (n-1)/((n-2)(n-3)) · ((n+1)·m₄/m₂² − 3(n-1))
            kurt[i] = (n - 1) / ((n - 2) * (n - 3)) * (
                (n + 1) * m4 / (m2 * m2) - 3 * (n - 1)
            )

    return skew, kurt


@njit
def _njit_hjorth(
    windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hjorth mobility & complexity in a single fused loop.

    Matches ``np.diff`` + ``np.var`` approach to machine precision but
    avoids intermediate 2-D diff arrays.
    """
    n_windows, n = windows.shape
    mobility = np.zeros(n_windows)
    complexity = np.zeros(n_windows)

    if n < 3:
        return mobility, complexity

    for i in range(n_windows):
        # ── signal mean & variance ──────────────────────────────────────
        mean = 0.0
        for j in range(n):
            mean += windows[i, j]
        mean /= n

        var_sig = 0.0
        for j in range(n):
            d = windows[i, j] - mean
            var_sig += d * d
        var_sig /= n - 1

        # ── first difference ────────────────────────────────────────────
        nd1 = n - 1
        d1 = np.empty(nd1)
        for j in range(nd1):
            d1[j] = windows[i, j + 1] - windows[i, j]

        mean_d1 = 0.0
        for j in range(nd1):
            mean_d1 += d1[j]
        mean_d1 /= nd1

        var_d1 = 0.0
        for j in range(nd1):
            d = d1[j] - mean_d1
            var_d1 += d * d
        var_d1 /= nd1 - 1

        if var_sig > 0 and var_d1 >= 0:
            mobility[i] = np.sqrt(var_d1 / var_sig)

        # ── second difference (complexity) ─────────────────────────────
        if n >= 4:
            nd2 = nd1 - 1
            d2 = np.empty(nd2)
            for j in range(nd2):
                d2[j] = d1[j + 1] - d1[j]

            mean_d2 = 0.0
            for j in range(nd2):
                mean_d2 += d2[j]
            mean_d2 /= nd2

            var_d2 = 0.0
            for j in range(nd2):
                d = d2[j] - mean_d2
                var_d2 += d * d
            var_d2 /= nd2 - 1

            if var_d1 > 0 and var_d2 >= 0:
                mob_deriv = np.sqrt(var_d2 / var_d1)
                if mobility[i] > 0:
                    complexity[i] = mob_deriv / mobility[i]

    return mobility, complexity


@njit
def _njit_shannon_entropy(windows: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Shannon entropy with per-window bin edges.

    Matches the original ``np.histogram(window, bins=10, density=True)``
    approach to machine precision in a single fused pass.
    """
    n_windows, n = windows.shape
    result = np.zeros(n_windows)

    for i in range(n_windows):
        # ── per-window min / max ──────────────────────────────────────
        vmin = windows[i, 0]
        vmax = windows[i, 0]
        for j in range(1, n):
            v = windows[i, j]
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v

        bw = (vmax - vmin) / n_bins if vmax > vmin else 1.0

        # ── digitize + count in one pass ─────────────────────────────
        counts = np.zeros(n_bins)
        for j in range(n):
            idx = int((windows[i, j] - vmin) / bw)
            if idx < 0:
                idx = 0
            if idx >= n_bins:
                idx = n_bins - 1
            counts[idx] += 1.0

        total = np.sum(counts)
        if total < 2:
            result[i] = 0.0
            continue

        density = counts / (total * bw + 1e-12)
        prob = density * bw

        entropy = 0.0
        for j in range(n_bins):
            if prob[j] > 0:
                entropy -= prob[j] * np.log2(prob[j])
        result[i] = entropy

    return result


# ── Per-window helpers (kept for the rare NaN-fallback path) ────────────────


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

    # ── Skewness & kurtosis (numba) ───────────────────────────────────────
    skew, kurt = _njit_skew_kurtosis(windows)
    out["skewness"] = skew
    out["kurtosis"] = kurt

    # ── Hjorth mobility & complexity (numba) ──────────────────────────────
    mobility, complexity = _njit_hjorth(windows)
    out["hjorth_mobility"] = mobility
    out["hjorth_complexity"] = complexity

    # ── Interquartile range — vectorized percentile ───────────────────────
    q75 = np.percentile(windows, 75, axis=1)
    q25 = np.percentile(windows, 25, axis=1)
    out["interquartile_range"] = q75 - q25

    # ── Shannon entropy (numba) ───────────────────────────────────────────
    out["shannon_entropy"] = _njit_shannon_entropy(windows)

    return out
