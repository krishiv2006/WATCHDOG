"""
Core drift detection algorithms for Watchdog AI.

Implements:
- Population Stability Index (PSI)
- Jensen-Shannon Distance
- Performance degradation scoring
- Composite health score
"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from typing import Optional


EPSILON = 1e-10  # avoid log(0)


def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_buckets: int = 10,
) -> float:
    """
    Compute the Population Stability Index (PSI) between two distributions.

    Uses quantile-based bucketing derived from the baseline distribution.

    Parameters
    ----------
    baseline : array-like  – 1-D array of baseline (training) values
    current  : array-like  – 1-D array of current (production) values
    n_buckets: int          – number of quantile buckets (default 10)

    Returns
    -------
    float – PSI score.  < 0.1 stable, 0.1–0.25 warning, > 0.25 critical
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    # Build bucket edges from baseline quantiles
    quantiles = np.linspace(0, 100, n_buckets + 1)
    edges = np.percentile(baseline, quantiles)
    # Ensure unique edges to avoid zero-width buckets
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0

    # Count observations in each bucket
    baseline_counts = np.histogram(baseline, bins=edges)[0].astype(float)
    current_counts = np.histogram(current, bins=edges)[0].astype(float)

    # Convert to proportions
    baseline_pct = baseline_counts / baseline_counts.sum()
    current_pct = current_counts / current_counts.sum()

    # Replace zeros with epsilon
    baseline_pct = np.where(baseline_pct == 0, EPSILON, baseline_pct)
    current_pct = np.where(current_pct == 0, EPSILON, current_pct)

    # PSI = sum( (current% - baseline%) * ln(current% / baseline%) )
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def compute_psi_from_histograms(
    baseline_counts: list[float],
    current_counts: list[float],
) -> float:
    """
    Compute PSI from pre-computed histogram bucket counts.
    Useful when we already have stored histogram data.
    """
    b = np.asarray(baseline_counts, dtype=float)
    c = np.asarray(current_counts, dtype=float)

    if b.sum() == 0 or c.sum() == 0:
        return 0.0

    b_pct = b / b.sum()
    c_pct = c / c.sum()

    b_pct = np.where(b_pct == 0, EPSILON, b_pct)
    c_pct = np.where(c_pct == 0, EPSILON, c_pct)

    psi = np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))
    return float(psi)


def compute_js_distance(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 50,
) -> float:
    """
    Compute Jensen-Shannon distance between two distributions.

    Parameters
    ----------
    baseline : array-like – 1-D baseline values
    current  : array-like – 1-D current values
    n_bins   : int        – number of histogram bins

    Returns
    -------
    float – JS distance in [0, 1].  Higher = more diverged.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    # Build shared bin edges
    combined = np.concatenate([baseline, current])
    edges = np.histogram_bin_edges(combined, bins=n_bins)

    b_hist = np.histogram(baseline, bins=edges)[0].astype(float)
    c_hist = np.histogram(current, bins=edges)[0].astype(float)

    # Normalize to probability distributions
    b_hist = b_hist / (b_hist.sum() + EPSILON)
    c_hist = c_hist / (c_hist.sum() + EPSILON)

    # scipy.spatial.distance.jensenshannon returns the JS *distance* (sqrt of divergence)
    return float(jensenshannon(b_hist, c_hist))


def psi_to_status(psi_score: float) -> str:
    """Map a PSI score to a drift status label."""
    if psi_score > 0.25:
        return "critical"
    elif psi_score > 0.1:
        return "warning"
    return "stable"


def compute_health_score(
    avg_drift: float,
    performance_ratio: float,
    open_critical_alerts: int,
    weight_drift: float = 0.4,
    weight_perf: float = 0.4,
    weight_alerts: float = 0.2,
    alert_penalty: float = 5.0,
) -> float:
    """
    Compute a composite health score in [0, 100].

    Parameters
    ----------
    avg_drift            : average drift score across features (0–1 scale)
    performance_ratio    : current_metric / baseline_metric (e.g. 0.89/0.94 ≈ 0.947)
    open_critical_alerts : count of unresolved critical alerts
    weight_*             : weighting factors (should sum to 1.0)
    alert_penalty        : points subtracted per open critical alert

    Returns
    -------
    float – health score clamped to [0, 100]
    """
    drift_component = max(0.0, 1.0 - avg_drift) * 100
    perf_component = min(performance_ratio, 1.0) * 100
    alert_component = max(0.0, 100.0 - open_critical_alerts * alert_penalty)

    raw = (
        weight_drift * drift_component
        + weight_perf * perf_component
        + weight_alerts * alert_component
    )
    return round(max(0.0, min(100.0, raw)), 1)


def compute_performance_delta(current: float, baseline: float) -> float:
    """Return the percentage-point difference: current - baseline."""
    return round((current - baseline) * 100, 2)
