"""
Core drift detection algorithms for Watchdog AI.

Implements:
- Population Stability Index (PSI) with open-ended outer bins
- Jensen-Shannon Distance
- Combined drift_decision() cross-checking PSI + JS
- Composite health score with component breakdown
- Performance degradation scoring

Design decisions
----------------
- Open-ended outer bins (-inf / +inf): live values outside the baseline range
  are counted in the first / last bucket, preventing NaN from 0-count denominators.
- Additive smoothing (0.5 pseudo-count per bucket): avoids enormous PSI terms
  from buckets with very few live observations.  The floor is
  ``0.5 / (n + 0.5 * n_buckets)`` which is ~1e-3 for typical n, far more
  stable than the former 1e-10 epsilon.
- Noise floor: for small samples the expected PSI under the null is roughly
  ``(B-1) * (1/n_baseline + 1/n_live)``; we only escalate if PSI exceeds *both*
  the configured threshold *and* this noise floor.
- Consecutive-window persistence: callers should check the last N DriftScore
  rows before creating a critical alert (handled in the alerting service).
"""

from __future__ import annotations

import math
import logging
from typing import Optional, NamedTuple

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

# ── Default thresholds (overridden via config / AlertRule) ──────────────────
DEFAULT_PSI_WARNING: float = 0.1
DEFAULT_PSI_CRITICAL: float = 0.25
DEFAULT_JS_WARNING: float = 0.05   # JS distance [0,1]
DEFAULT_JS_CRITICAL: float = 0.15
DEFAULT_MIN_SAMPLE_SIZE: int = 250


# ── Internal helpers ────────────────────────────────────────────────────────

def _clean(arr: np.ndarray, label: str = "") -> tuple[np.ndarray, float]:
    """
    Drop NaN and ±inf values; return cleaned array and null_rate.

    Parameters
    ----------
    arr   : input array (dtype float)
    label : name for logging

    Returns
    -------
    cleaned_arr, null_rate (fraction of dropped values)
    """
    arr = np.asarray(arr, dtype=float)
    bad_mask = ~np.isfinite(arr)
    null_rate = float(bad_mask.mean()) if len(arr) > 0 else 0.0
    clean = arr[~bad_mask]
    if bad_mask.any():
        logger.debug(
            "Dropped %d non-finite values from '%s' (%.1f%%)",
            bad_mask.sum(), label, null_rate * 100,
        )
    return clean, null_rate


def _psi_smoothed(b_counts: np.ndarray, c_counts: np.ndarray) -> float:
    """
    PSI with additive smoothing (0.5 pseudo-count per bucket).

    Parameters
    ----------
    b_counts : raw integer counts in each baseline bucket
    c_counts : raw integer counts in each current bucket

    Returns
    -------
    PSI score ≥ 0  (finite for all inputs with at least 1 bucket)
    """
    b = b_counts.astype(float) + 0.5
    c = c_counts.astype(float) + 0.5

    b_pct = b / b.sum()
    c_pct = c / c.sum()

    # PSI = Σ (c% - b%) * ln(c% / b%)
    psi = float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))
    return max(0.0, psi)   # numerical guard (can be tiny negative)


def _psi_noise_floor(n_baseline: int, n_live: int, n_buckets: int) -> float:
    """Expected PSI under the null (no drift) for given sample sizes."""
    if n_baseline <= 0 or n_live <= 0:
        return 0.0
    return max(0.0, (n_buckets - 1) * (1.0 / n_baseline + 1.0 / n_live))


# ── Public PSI API ──────────────────────────────────────────────────────────

def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_buckets: int = 10,
) -> tuple[float, float, float]:
    """
    Compute PSI between two distributions using open-ended quantile bins.

    Open-ended outer bins (-inf / +inf) ensure that live values outside the
    baseline range are correctly counted in the first or last bucket instead of
    being silently discarded (which produced NaN when 0 live points fell in any
    bucket).

    Parameters
    ----------
    baseline  : 1-D baseline (training) values
    current   : 1-D current (production) values
    n_buckets : number of quantile buckets (default 10)

    Returns
    -------
    (psi, null_rate_baseline, null_rate_live)
        psi is finite and ≥ 0 for any finite input.
    """
    baseline_arr, null_b = _clean(np.asarray(baseline, dtype=float), "baseline")
    current_arr, null_c = _clean(np.asarray(current, dtype=float), "current")

    if len(baseline_arr) == 0 or len(current_arr) == 0:
        return 0.0, null_b, null_c

    # All-same baseline → no information to bin by
    if np.ptp(baseline_arr) == 0:
        logger.debug("Constant baseline column detected; PSI undefined (returning 0)")
        return 0.0, null_b, null_c

    # Build quantile edges from baseline (interior edges only)
    quantiles = np.linspace(0, 100, n_buckets + 1)[1:-1]   # exclude 0 and 100
    interior_edges = np.unique(np.percentile(baseline_arr, quantiles))

    # Open-ended outer edges
    edges = np.concatenate([[-np.inf], interior_edges, [np.inf]])

    # np.digitize maps each value to a bucket index (1-indexed)
    b_buckets = np.digitize(baseline_arr, interior_edges)   # 0 … n_buckets-1
    c_buckets = np.digitize(current_arr, interior_edges)

    n_actual_buckets = len(interior_edges) + 1  # may be < n_buckets if baseline has ties
    b_counts = np.bincount(b_buckets, minlength=n_actual_buckets)[:n_actual_buckets]
    c_counts = np.bincount(c_buckets, minlength=n_actual_buckets)[:n_actual_buckets]

    psi = _psi_smoothed(b_counts, c_counts)
    return psi, null_b, null_c


def compute_psi_from_histograms(
    baseline_counts: list[float],
    current_counts: list[float],
) -> float:
    """
    Compute PSI from pre-computed histogram bucket counts using additive smoothing.

    Useful when we already have stored histogram data (e.g. from baseline profiles).
    """
    b = np.asarray(baseline_counts, dtype=float)
    c = np.asarray(current_counts, dtype=float)

    if b.sum() == 0 or c.sum() == 0:
        return 0.0

    return _psi_smoothed(b, c)


def compute_psi_from_quantile_edges(
    live: np.ndarray,
    interior_edges: np.ndarray,
    baseline_counts: np.ndarray,
) -> float:
    """PSI of live samples against stored baseline quantile-bucket counts.

    Outer bins are open-ended via ``np.digitize`` (values below the first
    interior edge land in bucket 0; values above the last land in the last
    bucket), so out-of-range live points are counted, not dropped.
    """
    live_arr, _ = _clean(np.asarray(live, dtype=float), "live")
    if len(live_arr) == 0 or len(baseline_counts) == 0:
        return 0.0
    interior = np.asarray(interior_edges, dtype=float)
    n_actual = len(interior) + 1
    c_buckets = np.digitize(live_arr, interior)
    c_counts = np.bincount(c_buckets, minlength=n_actual)[:n_actual]
    b_counts = np.asarray(baseline_counts, dtype=float)
    if len(b_counts) != n_actual:
        # Pad or trim to match live bucket count
        padded = np.zeros(n_actual)
        n = min(len(b_counts), n_actual)
        padded[:n] = b_counts[:n]
        b_counts = padded
    return _psi_smoothed(b_counts, c_counts)


def compute_js_from_histograms(
    baseline_counts: list[float],
    current_counts: list[float],
) -> float:
    """Jensen-Shannon distance from two aligned histogram count vectors."""
    b = np.asarray(baseline_counts, dtype=float)
    c = np.asarray(current_counts, dtype=float)
    if b.sum() == 0 or c.sum() == 0:
        return 0.0
    n = max(len(b), len(c))
    b_pad = np.zeros(n)
    c_pad = np.zeros(n)
    b_pad[: len(b)] = b
    c_pad[: len(c)] = c
    b_pad = (b_pad + 0.5) / (b_pad.sum() + 0.5 * n)
    c_pad = (c_pad + 0.5) / (c_pad.sum() + 0.5 * n)
    return float(jensenshannon(b_pad, c_pad))


# ── Public JS API ───────────────────────────────────────────────────────────

def compute_js_distance(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 50,
) -> float:
    """
    Compute Jensen-Shannon distance between two distributions.

    Parameters
    ----------
    baseline : 1-D baseline values
    current  : 1-D current values
    n_bins   : number of histogram bins (shared range)

    Returns
    -------
    float in [0, 1].  Higher = more diverged.
    """
    baseline_arr, _ = _clean(np.asarray(baseline, dtype=float), "baseline")
    current_arr, _ = _clean(np.asarray(current, dtype=float), "current")

    if len(baseline_arr) == 0 or len(current_arr) == 0:
        return 0.0

    # Shared bin edges spanning both distributions
    combined = np.concatenate([baseline_arr, current_arr])
    if np.ptp(combined) == 0:
        return 0.0   # all identical values — no distance

    edges = np.histogram_bin_edges(combined, bins=n_bins)

    b_hist = np.histogram(baseline_arr, bins=edges)[0].astype(float)
    c_hist = np.histogram(current_arr, bins=edges)[0].astype(float)

    # Small additive smoothing so js is numerically stable
    b_hist = (b_hist + 0.5) / (b_hist.sum() + 0.5 * n_bins)
    c_hist = (c_hist + 0.5) / (c_hist.sum() + 0.5 * n_bins)

    return float(jensenshannon(b_hist, c_hist))


def compute_ks_test(baseline: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test.

    Returns
    -------
    (statistic, p_value)
    """
    b, _ = _clean(np.asarray(baseline, dtype=float))
    c, _ = _clean(np.asarray(current, dtype=float))
    if len(b) < 2 or len(c) < 2:
        return 0.0, 1.0
    result = ks_2samp(b, c)
    return float(result.statistic), float(result.pvalue)


# ── Drift decision ──────────────────────────────────────────────────────────

class DriftDecision(NamedTuple):
    status: str           # "insufficient_data" | "stable" | "warning" | "critical"
    confidence: str       # "high" | "medium" | "low"
    reason: str           # human-readable explanation
    psi: float
    js_distance: float
    noise_floor: float
    n_live: int


def drift_decision(
    psi: float,
    js_distance: float,
    n_live: int,
    n_baseline: int,
    n_buckets: int = 10,
    psi_warning: float = DEFAULT_PSI_WARNING,
    psi_critical: float = DEFAULT_PSI_CRITICAL,
    js_warning: float = DEFAULT_JS_WARNING,
    js_critical: float = DEFAULT_JS_CRITICAL,
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
    is_constant: bool = False,
    ks_p: Optional[float] = None,
) -> DriftDecision:
    """
    Combined drift status decision using PSI (primary) + JS (cross-check).

    Logic
    -----
    1. If n_live < min_sample_size → ``insufficient_data`` (never alert on this).
    2. If PSI ≤ noise_floor → ``stable`` (sampling noise only).
    3. PSI determines candidate status; JS cross-checks:
       - If JS agrees (within its threshold) → confirmed status.
       - If JS disagrees (below its threshold) → status is downgraded one level;
         confidence is "low".
    4. Candidate ``critical`` is confirmed only when PSI > psi_critical AND
       JS > js_critical.  Otherwise falls back to ``warning``.

    Parameters
    ----------
    psi            : computed PSI
    js_distance    : computed JS distance
    n_live         : number of live samples used
    n_baseline     : number of baseline samples
    n_buckets      : number of PSI buckets (used to compute noise floor)
    psi_warning    : PSI threshold for warning status
    psi_critical   : PSI threshold for critical status
    js_warning     : JS threshold for warning cross-check
    js_critical    : JS threshold for critical cross-check
    min_sample_size: minimum n_live to produce any actionable status
    is_constant    : True when baseline column has zero variance

    Returns
    -------
    DriftDecision named tuple
    """
    noise_floor = _psi_noise_floor(n_baseline, n_live, n_buckets)

    # ── 1. Constant column ──────────────────────────────────────────────────
    if is_constant:
        return DriftDecision(
            status="insufficient_data",
            confidence="low",
            reason="Constant baseline column — PSI undefined.",
            psi=psi, js_distance=js_distance,
            noise_floor=noise_floor, n_live=n_live,
        )

    # ── 2. Insufficient sample size ─────────────────────────────────────────
    if n_live < min_sample_size:
        return DriftDecision(
            status="insufficient_data",
            confidence="low",
            reason=(
                f"Only {n_live} live samples; {min_sample_size} required "
                "for statistically reliable drift detection."
            ),
            psi=psi, js_distance=js_distance,
            noise_floor=noise_floor, n_live=n_live,
        )

    # ── 3. Sampling noise floor ─────────────────────────────────────────────
    if psi <= noise_floor:
        return DriftDecision(
            status="stable",
            confidence="high",
            reason=f"PSI {psi:.4f} ≤ noise floor {noise_floor:.4f}; consistent with sampling variation.",
            psi=psi, js_distance=js_distance,
            noise_floor=noise_floor, n_live=n_live,
        )

    # ── 4. PSI-based candidate status ──────────────────────────────────────
    if psi > psi_critical:
        psi_status = "critical"
    elif psi > psi_warning:
        psi_status = "warning"
    else:
        psi_status = "stable"

    if psi_status == "stable":
        return DriftDecision(
            status="stable",
            confidence="high",
            reason=f"PSI {psi:.4f} < warning threshold {psi_warning:.4f}.",
            psi=psi, js_distance=js_distance,
            noise_floor=noise_floor, n_live=n_live,
        )

    # ── 5. JS (and optional KS p-value) cross-check ────────────────────────
    # KS p-value, when available, can only *downgrade*: a non-significant KS
    # test (p > 0.05) means we lack evidence the CDFs differ.
    ks_disagrees = ks_p is not None and ks_p > 0.05

    if psi_status == "critical":
        js_agrees = js_distance > js_critical and not ks_disagrees
        if js_agrees:
            status = "critical"
            confidence = "high"
            reason = (
                f"PSI {psi:.4f} > {psi_critical:.2f} (critical) "
                f"AND JS {js_distance:.4f} > {js_critical:.2f} — both metrics confirm critical drift."
            )
        else:
            # JS doesn't agree → downgrade to warning
            status = "warning"
            confidence = "medium"
            reason = (
                f"PSI {psi:.4f} indicates critical drift but JS {js_distance:.4f} "
                f"< {js_critical:.2f} — downgraded to warning pending more data."
            )
    else:  # psi_status == "warning"
        js_agrees = js_distance > js_warning and not ks_disagrees
        if js_agrees:
            status = "warning"
            confidence = "high"
            reason = (
                f"PSI {psi:.4f} > {psi_warning:.2f} (warning) "
                f"AND JS {js_distance:.4f} > {js_warning:.2f} — drift detected."
            )
        else:
            # JS doesn't agree → downgrade to stable
            status = "stable"
            confidence = "low"
            reason = (
                f"PSI {psi:.4f} > {psi_warning:.2f} but JS {js_distance:.4f} "
                f"< {js_warning:.2f} — treating as stable; monitor closely."
            )

    return DriftDecision(
        status=status,
        confidence=confidence,
        reason=reason,
        psi=psi, js_distance=js_distance,
        noise_floor=noise_floor, n_live=n_live,
    )


# ── Kept for backwards compatibility with existing callers ──────────────────
def psi_to_status(
    psi_score: float,
    psi_warning: float = DEFAULT_PSI_WARNING,
    psi_critical: float = DEFAULT_PSI_CRITICAL,
) -> str:
    """
    Simple PSI → status mapping.

    Deprecated: prefer drift_decision() which also cross-checks JS and
    applies the noise floor / sample-size guard.
    """
    if psi_score > psi_critical:
        return "critical"
    elif psi_score > psi_warning:
        return "warning"
    return "stable"


# ── Health score ────────────────────────────────────────────────────────────

class HealthScore(NamedTuple):
    score: float           # [0, 100]
    components: dict       # breakdown for UI transparency


def compute_health_score(
    avg_psi: float,
    open_critical_alerts: int,
    psi_warning: float = DEFAULT_PSI_WARNING,
    psi_critical: float = DEFAULT_PSI_CRITICAL,
    performance_ratio: Optional[float] = None,
    weight_drift: float = 0.4,
    weight_perf: float = 0.4,
    weight_alerts: float = 0.2,
    alert_penalty: float = 5.0,
) -> HealthScore:
    """
    Compute a composite health score in [0, 100] with component breakdown.

    Changes from v1
    ---------------
    - ``avg_psi`` is now the actual (unbounded) average PSI; it is normalised
      to [0, 1] via ``min(avg_psi / psi_critical, 1.0)`` before use.
    - ``performance_ratio`` is optional (None = no labels yet).  When absent,
      the drift weight absorbs the performance weight so the total is still 1.0.
    - Returns HealthScore with a ``components`` dict for frontend transparency.

    Parameters
    ----------
    avg_psi              : mean PSI across features (unbounded ≥ 0)
    open_critical_alerts : count of unresolved critical alerts
    psi_warning          : PSI threshold for warning (used for normalisation)
    psi_critical         : PSI threshold for critical (used for normalisation cap)
    performance_ratio    : current_metric / baseline_metric, or None
    weight_drift         : weight for drift component (default 0.4)
    weight_perf          : weight for performance component (default 0.4)
    weight_alerts        : weight for alerts component (default 0.2)
    alert_penalty        : points subtracted per open critical alert
    """
    # Normalise PSI: 0 PSI → 1.0 drift_ok; PSI at critical → 0.0
    normalised_psi = min(avg_psi / max(psi_critical, 1e-9), 1.0)
    drift_component = round((1.0 - normalised_psi) * 100, 1)

    alert_component = round(max(0.0, 100.0 - open_critical_alerts * alert_penalty), 1)

    if performance_ratio is not None:
        perf_component = round(min(performance_ratio, 1.0) * 100, 1)
        # All three components active
        raw = (
            weight_drift * drift_component
            + weight_perf * perf_component
            + weight_alerts * alert_component
        )
        components = {
            "drift": {"value": drift_component, "weight": weight_drift},
            "performance": {"value": perf_component, "weight": weight_perf},
            "alerts": {"value": alert_component, "weight": weight_alerts},
        }
    else:
        # No labels: redistribute performance weight to drift
        effective_drift_weight = weight_drift + weight_perf
        raw = (
            effective_drift_weight * drift_component
            + weight_alerts * alert_component
        )
        components = {
            "drift": {"value": drift_component, "weight": effective_drift_weight},
            "performance": {"value": None, "weight": 0, "reason": "awaiting_labels"},
            "alerts": {"value": alert_component, "weight": weight_alerts},
        }

    score = round(max(0.0, min(100.0, raw)), 1)
    return HealthScore(score=score, components=components)


def compute_performance_delta(current: float, baseline: float) -> float:
    """Return the percentage-point difference: current - baseline."""
    return round((current - baseline) * 100, 2)
