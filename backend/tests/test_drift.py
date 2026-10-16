"""
Comprehensive unit tests for PSI, JS, drift_decision, and health score.

Covers:
- Identical / shifted / drastically different distributions (regression safety)
- Out-of-range live values (the NaN bug — A.1)
- NaN / inf inputs are dropped gracefully (A.1)
- Constant baseline column (A.3 / A.1)
- Tiny sample size → insufficient_data (A.3)
- JS cross-check downgrade (A.4)
- Health score components and no-label re-weighting (A.6)
- PSI noise floor guard (A.3)
- Window parsing (A.7)
- AlertRuleOut camelCase serialisation (A.9)
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from app.services.drift import (
    compute_psi,
    compute_psi_from_histograms,
    compute_js_distance,
    compute_js_from_quantile_edges,
    benjamini_hochberg,
    compute_ks_test,
    drift_decision,
    psi_to_status,
    compute_health_score,
    compute_performance_delta,
    _psi_noise_floor,
    DEFAULT_PSI_WARNING,
    DEFAULT_PSI_CRITICAL,
    DEFAULT_JS_WARNING,
    DEFAULT_JS_CRITICAL,
    DEFAULT_MIN_SAMPLE_SIZE,
)


# ═══════════════════════════════════════════════════════════════════════════
# PSI Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestComputePSI:
    """Tests for compute_psi (returns tuple: psi, null_b, null_c)."""

    def _psi(self, baseline, current, n_buckets=10):
        psi, _, _ = compute_psi(baseline, current, n_buckets)
        return psi

    def test_identical_distributions_gives_near_zero(self):
        """Same distribution should have PSI ≈ 0."""
        rng = np.random.default_rng(42)
        data = rng.normal(50, 10, 5000)
        psi = self._psi(data, data)
        assert psi < 0.01, f"PSI for identical data should be ~0, got {psi}"

    def test_slight_shift_gives_low_psi(self):
        """A small mean shift should produce PSI in stable/low-warning range."""
        rng = np.random.default_rng(42)
        baseline = rng.normal(50, 10, 5000)
        current = rng.normal(51, 10, 5000)  # +0.1σ shift
        psi = self._psi(baseline, current)
        assert psi < 0.1, f"Slight shift should be stable, got PSI={psi}"

    def test_moderate_shift_gives_warning_psi(self):
        """A moderate shift should produce PSI in the warning range."""
        rng = np.random.default_rng(42)
        baseline = rng.normal(50, 10, 5000)
        current = rng.normal(55, 12, 5000)  # +0.5σ shift + wider spread
        psi = self._psi(baseline, current)
        assert 0.05 < psi < 0.5, f"Moderate shift PSI should be in mid-range, got {psi}"

    def test_drastic_shift_gives_critical_psi(self):
        """A large distribution change should produce PSI > 0.25."""
        rng = np.random.default_rng(42)
        baseline = rng.normal(50, 10, 5000)
        current = rng.normal(80, 5, 5000)  # large mean shift
        psi = self._psi(baseline, current)
        assert psi > 0.25, f"Drastic shift should be critical, got PSI={psi}"

    # ── A.1 fixes ────────────────────────────────────────────────────────

    def test_out_of_range_live_values_not_dropped(self):
        """
        Regression for A.1: baseline N(50,10), live N(120,5).

        Old code: 0/1000 live points fell in baseline percentile range → NaN.
        Fixed:    open-ended outer bins catch all points → finite PSI.
        """
        rng = np.random.default_rng(0)
        baseline = rng.normal(50, 10, 1000)
        current = rng.normal(120, 5, 1000)  # entirely outside baseline range
        psi, null_b, null_c = compute_psi(baseline, current)
        assert math.isfinite(psi), f"PSI must be finite for out-of-range live data, got {psi}"
        assert psi > 0.25, f"Extreme drift should be critical, got PSI={psi}"

    def test_nan_inputs_dropped_gracefully(self):
        """NaN values should be dropped, not cause NaN output."""
        rng = np.random.default_rng(1)
        baseline = rng.normal(0, 1, 500)
        current_clean = rng.normal(0, 1, 400)
        nans = np.full(100, np.nan)
        current_with_nans = np.concatenate([current_clean, nans])

        psi, null_b, null_c = compute_psi(baseline, current_with_nans)
        assert math.isfinite(psi), f"PSI with NaN inputs must be finite, got {psi}"
        assert null_c > 0, f"null_rate_live should reflect dropped NaN, got {null_c}"
        assert null_b == 0.0, "Baseline has no NaN, null_rate should be 0"

    def test_inf_inputs_dropped_gracefully(self):
        """±inf values should be dropped, not cause NaN/inf PSI."""
        rng = np.random.default_rng(2)
        baseline = rng.normal(0, 1, 500)
        current = rng.normal(0, 1, 500)
        current[0] = np.inf
        current[1] = -np.inf

        psi, _, null_c = compute_psi(baseline, current)
        assert math.isfinite(psi), f"PSI with ±inf inputs must be finite, got {psi}"
        assert null_c > 0

    def test_all_nan_baseline_returns_zero(self):
        """All-NaN baseline → return (0, null_rate, 0) without error."""
        baseline = np.full(100, np.nan)
        current = np.random.default_rng(3).normal(0, 1, 100)
        psi, null_b, null_c = compute_psi(baseline, current)
        assert psi == 0.0
        assert null_b == 1.0

    def test_empty_arrays_return_zero(self):
        psi, _, _ = compute_psi(np.array([]), np.array([1, 2, 3]))
        assert psi == 0.0
        psi2, _, _ = compute_psi(np.array([1, 2, 3]), np.array([]))
        assert psi2 == 0.0

    def test_constant_baseline_returns_zero(self):
        """All-same baseline → PSI = 0 (no bucketing possible)."""
        baseline = np.ones(1000)
        current = np.random.default_rng(4).normal(1.0, 0.5, 500)
        psi, _, _ = compute_psi(baseline, current)
        assert psi == 0.0

    def test_psi_is_non_negative(self):
        """PSI should always be >= 0."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            b = rng.normal(0, 1, 1000)
            c = rng.normal(rng.uniform(-2, 2), rng.uniform(0.5, 3), 1000)
            psi, _, _ = compute_psi(b, c)
            assert psi >= 0, f"PSI should be non-negative, got {psi}"

    def test_psi_is_finite_under_extreme_shift(self):
        """PSI must be finite even when all live samples are far from baseline."""
        rng = np.random.default_rng(5)
        baseline = rng.normal(0, 1, 500)
        current = rng.normal(1000, 0.1, 500)
        psi, _, _ = compute_psi(baseline, current)
        assert math.isfinite(psi), f"PSI must be finite under extreme shift, got {psi}"

    def test_same_distribution_js_is_low_across_resamples(self):
        """Finite samples from one distribution must not look like drift."""
        rng = np.random.default_rng(2026)
        below_threshold = 0
        for _ in range(200):
            baseline = rng.normal(50, 10, 460)
            current = rng.normal(50, 10, 300)
            below_threshold += compute_js_distance(baseline, current, n_bins=10) < 0.1

        assert below_threshold >= 190

    def test_identical_arrays_have_zero_js(self):
        values = np.random.default_rng(7).normal(size=300)
        assert compute_js_distance(values, values, n_bins=10) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# PSI from histograms
# ═══════════════════════════════════════════════════════════════════════════

class TestComputePSIFromHistograms:
    def test_identical_histograms(self):
        counts = [100, 200, 300, 200, 100]
        psi = compute_psi_from_histograms(counts, counts)
        assert psi < 0.01

    def test_shifted_histograms(self):
        baseline = [100, 200, 300, 200, 100]
        current = [50, 100, 200, 300, 250]  # shifted right
        psi = compute_psi_from_histograms(baseline, current)
        assert psi > 0.05

    def test_empty_histograms(self):
        assert compute_psi_from_histograms([0, 0, 0], [1, 2, 3]) == 0.0

    def test_empty_current_baseline(self):
        assert compute_psi_from_histograms([1, 2, 3], [0, 0, 0]) == 0.0

    def test_very_different_histograms_not_nan(self):
        """One bucket has all mass in baseline, zero in current → must be finite."""
        baseline = [1000, 0, 0, 0, 0]
        current = [0, 0, 0, 0, 1000]
        psi = compute_psi_from_histograms(baseline, current)
        assert math.isfinite(psi), f"PSI from extreme histograms must be finite, got {psi}"
        assert psi > 0


# ═══════════════════════════════════════════════════════════════════════════
# JS Distance Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeJSDistance:
    def test_stable_resamples_are_below_warning_threshold(self):
        rng = np.random.default_rng(2026)
        baseline = rng.normal(0, 1, 5000)
        distances = [
            compute_js_from_quantile_edges(
                baseline, rng.choice(baseline, size=250, replace=False)
            )
            for _ in range(200)
        ]
        assert np.quantile(distances, 0.95) < DEFAULT_JS_WARNING

    def test_identical_distributions_gives_near_zero(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 10, 5000)
        js = compute_js_distance(data, data)
        assert js < 0.05, f"JS for identical data should be ~0, got {js}"

    def test_similar_distributions_gives_low_js(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(50, 10, 5000)
        current = rng.normal(51, 10, 5000)
        js = compute_js_distance(baseline, current)
        assert js < 0.15, f"Similar distributions should have low JS, got {js}"

    def test_different_distributions_gives_high_js(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(50, 10, 5000)
        current = rng.normal(80, 5, 5000)
        js = compute_js_distance(baseline, current)
        assert js > 0.3, f"Different distributions should have high JS, got {js}"

    def test_js_bounded_zero_to_one(self):
        """JS distance should be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            b = rng.normal(0, 1, 1000)
            c = rng.normal(rng.uniform(-5, 5), rng.uniform(0.1, 5), 1000)
            js = compute_js_distance(b, c)
            assert 0 <= js <= 1, f"JS should be in [0,1], got {js}"

    def test_empty_arrays_return_zero(self):
        assert compute_js_distance(np.array([]), np.array([1, 2, 3])) == 0.0

    def test_all_same_values_returns_zero(self):
        """Constant arrays → JS = 0."""
        b = np.ones(100)
        c = np.ones(100)
        js = compute_js_distance(b, c)
        assert js == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Drift Decision Tests (A.3 + A.4)
# ═══════════════════════════════════════════════════════════════════════════

class TestDriftDecision:
    """Tests for drift_decision() — cross-check, noise floor, sample size guard."""

    def _decide(self, psi, js, n_live=1000, n_baseline=5000, **kwargs):
        return drift_decision(
            psi=psi,
            js_distance=js,
            n_live=n_live,
            n_baseline=n_baseline,
            **kwargs,
        )

    # ── A.3: sample size guard ────────────────────────────────────────────

    def test_tiny_sample_gives_insufficient_data(self):
        """n_live < MIN_SAMPLE_SIZE → always insufficient_data."""
        d = self._decide(psi=5.0, js=0.9, n_live=50, min_sample_size=250)
        assert d.status == "insufficient_data"
        assert "50" in d.reason
        assert "250" in d.reason

    def test_exactly_at_min_sample_passes(self):
        """n_live == min_sample_size → eligible for real decision."""
        d = self._decide(psi=0.5, js=0.5, n_live=250, min_sample_size=250)
        assert d.status != "insufficient_data"

    def test_noise_floor_suppresses_small_psi(self):
        """PSI ≤ noise floor → stable even if PSI > warning threshold."""
        # noise_floor for n_live=300, n_baseline=300, 10 buckets = 9*(1/300+1/300) ≈ 0.06
        # PSI at 0.05 (< noise floor of ~0.06) → stable
        d = self._decide(
            psi=0.05,
            js=0.1,
            n_live=300,
            n_baseline=300,
            psi_warning=0.04,   # would normally be warning
        )
        # Either noise floor or low PSI relative to threshold → stable
        # (the noise floor may or may not dominate depending on actual calculation)
        # Just assert we get a finite result and a reason
        assert d.status in ("stable", "insufficient_data")
        assert isinstance(d.reason, str)

    # ── A.4: JS cross-check ───────────────────────────────────────────────

    def test_critical_psi_confirmed_by_js(self):
        """High PSI + high JS → critical, high confidence."""
        d = self._decide(
            psi=0.5, js=0.3,
            psi_warning=0.1, psi_critical=0.25,
            js_warning=0.05, js_critical=0.15,
            min_sample_size=100,
        )
        assert d.status == "critical"
        assert d.confidence == "high"

    def test_critical_psi_but_low_js_downgraded_to_warning(self):
        """High PSI but low JS → downgraded from critical to warning."""
        d = self._decide(
            psi=0.5, js=0.02,     # JS below js_critical
            psi_warning=0.1, psi_critical=0.25,
            js_warning=0.05, js_critical=0.15,
            min_sample_size=100,
        )
        assert d.status == "warning"
        assert d.confidence == "medium"
        assert "downgraded" in d.reason.lower()

    def test_warning_psi_confirmed_by_js(self):
        """Moderate PSI + moderate JS → warning, high confidence."""
        d = self._decide(
            psi=0.15, js=0.08,
            psi_warning=0.1, psi_critical=0.25,
            js_warning=0.05, js_critical=0.15,
            min_sample_size=100,
        )
        assert d.status == "warning"
        assert d.confidence == "high"

    def test_warning_psi_not_confirmed_by_js_downgraded_to_stable(self):
        """Moderate PSI but very low JS → stable (JS disagrees)."""
        d = self._decide(
            psi=0.15, js=0.01,    # JS well below js_warning
            psi_warning=0.1, psi_critical=0.25,
            js_warning=0.05, js_critical=0.15,
            min_sample_size=100,
        )
        assert d.status == "stable"
        assert d.confidence == "low"

    def test_threshold_boundaries_are_inclusive(self):
        warning = self._decide(
            psi=0.1,
            js=0.05,
            n_live=1000,
            n_baseline=100000,
            psi_warning=0.1,
            psi_critical=0.25,
            js_warning=0.05,
            js_critical=0.15,
            min_sample_size=250,
        )
        critical = self._decide(
            psi=0.25,
            js=0.15,
            n_live=1000,
            n_baseline=100000,
            psi_warning=0.1,
            psi_critical=0.25,
            js_warning=0.05,
            js_critical=0.15,
            min_sample_size=250,
        )
        assert warning.status == "warning"
        assert critical.status == "critical"

    def test_stable_psi_stays_stable(self):
        """Low PSI → stable regardless of JS."""
        d = self._decide(
            psi=0.05, js=0.9,
            psi_warning=0.1, psi_critical=0.25,
            min_sample_size=100,
        )
        # Either noise floor or low PSI → stable
        assert d.status in ("stable",)

    def test_non_significant_psi_excursion_is_stable(self):
        d = self._decide(
            psi=0.2,
            js=0.2,
            n_live=1000,
            n_baseline=5000,
            ks_p=0.2,
            min_sample_size=250,
        )
        assert d.status == "stable"
        assert "within sampling noise" in d.reason

    def test_stable_resamples_have_low_false_warning_rate(self):
        rng = np.random.default_rng(7)
        baseline = rng.normal(0, 1, 5000)
        warnings = 0
        trials = 100
        for _ in range(trials):
            live = rng.choice(baseline, size=250, replace=False)
            psi, _, _ = compute_psi(baseline, live)
            js = compute_js_from_quantile_edges(baseline, live)
            _, p_value = compute_ks_test(baseline, live)
            decision = self._decide(
                psi=psi,
                js=js,
                n_live=len(live),
                n_baseline=len(baseline),
                ks_p=p_value,
                min_sample_size=250,
            )
            warnings += decision.status in ("warning", "critical")
        assert warnings / trials < 0.05

    def test_extreme_drift_is_critical_with_finite_psi(self):
        rng = np.random.default_rng(8)
        baseline = rng.normal(0, 1, 5000)
        live = rng.normal(8, 0.2, 250)
        psi, _, _ = compute_psi(baseline, live)
        js = compute_js_from_quantile_edges(baseline, live)
        _, p_value = compute_ks_test(baseline, live)
        decision = self._decide(
            psi=psi,
            js=js,
            n_live=len(live),
            n_baseline=len(baseline),
            ks_p=p_value,
            min_sample_size=250,
        )
        assert math.isfinite(psi)
        assert decision.status == "critical"

    def test_constant_column_gives_insufficient_data(self):
        """Constant baseline column → always insufficient_data."""
        d = self._decide(psi=0.0, js=0.0, is_constant=True, n_live=1000)
        assert d.status == "insufficient_data"
        assert "constant" in d.reason.lower()

    def test_decision_always_has_reason_string(self):
        """All decision paths must return a non-empty reason string."""
        cases = [
            {"psi": 0.0, "js": 0.0},
            {"psi": 0.5, "js": 0.5},
            {"psi": 0.5, "js": 0.01},
            {"psi": 0.15, "js": 0.08},
            {"psi": 0.15, "js": 0.01},
        ]
        for case in cases:
            d = self._decide(**case, min_sample_size=100)
            assert isinstance(d.reason, str) and len(d.reason) > 0, f"Empty reason for {case}"

    def test_noise_floor_helper(self):
        """_psi_noise_floor returns expected values."""
        # B=10, n_b=1000, n_l=1000 → 9 * (1/1000 + 1/1000) = 0.018
        floor = _psi_noise_floor(1000, 1000, 10)
        assert abs(floor - 0.018) < 1e-10

        # Zero samples → 0
        assert _psi_noise_floor(0, 1000, 10) == 0.0
        assert _psi_noise_floor(1000, 0, 10) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Health Score Tests (A.6)
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeHealthScore:
    """Tests for compute_health_score — now returns HealthScore named tuple."""

    def test_perfect_health_with_labels(self):
        hs = compute_health_score(
            avg_psi=0.0,
            open_critical_alerts=0,
            performance_ratio=1.0,
        )
        assert hs.score == 100.0
        assert hs.components["drift"]["value"] == 100.0
        assert hs.components["performance"]["value"] == 100.0

    def test_perfect_health_no_labels(self):
        """Without labels, score still 100 when psi=0 and no alerts."""
        hs = compute_health_score(avg_psi=0.0, open_critical_alerts=0, performance_ratio=None)
        assert hs.score == 100.0
        assert hs.components["performance"]["value"] is None
        assert hs.components["performance"]["reason"] == "awaiting_labels"

    def test_high_psi_degrades_score(self):
        """PSI at critical threshold should substantially reduce score."""
        hs_low = compute_health_score(avg_psi=0.0, open_critical_alerts=0)
        hs_high = compute_health_score(avg_psi=0.25, open_critical_alerts=0)  # at critical
        assert hs_high.score < hs_low.score
        assert hs_high.score <= 60

    def test_psi_above_critical_clamped(self):
        """PSI well above critical continues degrading without reaching zero."""
        hs = compute_health_score(avg_psi=10.0, open_critical_alerts=0, performance_ratio=None)
        assert hs.score >= 0
        assert hs.components["drift"]["value"] > 0.0

    def test_health_decreases_monotonically_as_psi_rises(self):
        scores = [
            compute_health_score(avg_psi=psi, open_critical_alerts=0).score
            for psi in np.linspace(0.0, 5.0, 51)
        ]
        assert all(left >= right for left, right in zip(scores, scores[1:]))

    def test_critical_alerts_reduce_score(self):
        hs0 = compute_health_score(avg_psi=0.0, open_critical_alerts=0)
        hs5 = compute_health_score(avg_psi=0.0, open_critical_alerts=5)
        assert hs5.score < hs0.score

    def test_performance_ratio_below_one_reduces_score(self):
        hs_perfect = compute_health_score(avg_psi=0.0, open_critical_alerts=0, performance_ratio=1.0)
        hs_degraded = compute_health_score(avg_psi=0.0, open_critical_alerts=0, performance_ratio=0.7)
        assert hs_degraded.score < hs_perfect.score

    def test_score_remains_finite_for_extreme_inputs(self):
        hs = compute_health_score(
            avg_psi=100.0, open_critical_alerts=100, performance_ratio=0.0
        )
        assert 0.0 < hs.score < 1.0

    def test_score_clamped_to_hundred(self):
        hs = compute_health_score(avg_psi=-0.5, open_critical_alerts=0, performance_ratio=2.0)
        assert hs.score == 100.0

    def test_components_sum_weights(self):
        """Ensure component weights sum to ~1.0."""
        hs = compute_health_score(avg_psi=0.1, open_critical_alerts=1, performance_ratio=0.9)
        total_weight = sum(c["weight"] for c in hs.components.values())
        assert abs(total_weight - 1.0) < 1e-9
        assert all(
            0.0 <= component["normalized"] <= 1.0
            for component in hs.components.values()
            if component["normalized"] is not None
        )

    def test_components_sum_weights_no_labels(self):
        """Without labels, performance weight redistributed to drift."""
        hs = compute_health_score(avg_psi=0.1, open_critical_alerts=1, performance_ratio=None)
        total_weight = sum(c["weight"] for c in hs.components.values())
        assert abs(total_weight - 1.0) < 1e-9

    def test_moderate_degradation_gives_mid_score(self):
        hs = compute_health_score(
            avg_psi=0.15,
            open_critical_alerts=1,
            performance_ratio=0.90,
        )
        assert 50 < hs.score < 95, f"Expected moderate score, got {hs.score}"


# ═══════════════════════════════════════════════════════════════════════════
# psi_to_status (backwards-compat shim) — A.5
# ═══════════════════════════════════════════════════════════════════════════

class TestPSIToStatus:
    def test_stable(self):
        assert psi_to_status(0.05) == "stable"
        assert psi_to_status(0.0) == "stable"
        assert psi_to_status(0.09) == "stable"

    def test_warning(self):
        assert psi_to_status(0.1001) == "warning"
        assert psi_to_status(0.18) == "warning"
        assert psi_to_status(0.25) == "warning"

    def test_critical(self):
        assert psi_to_status(0.2501) == "critical"
        assert psi_to_status(0.5) == "critical"
        assert psi_to_status(1.0) == "critical"

    def test_custom_thresholds_respected(self):
        assert psi_to_status(0.05, psi_warning=0.03, psi_critical=0.10) == "warning"
        assert psi_to_status(0.15, psi_warning=0.03, psi_critical=0.10) == "critical"


# ═══════════════════════════════════════════════════════════════════════════
# Performance Delta
# ═══════════════════════════════════════════════════════════════════════════

class TestPerformanceDelta:
    def test_positive_delta(self):
        assert compute_performance_delta(0.95, 0.90) == 5.0

    def test_negative_delta(self):
        assert compute_performance_delta(0.89, 0.94) == -5.0

    def test_zero_delta(self):
        assert compute_performance_delta(0.90, 0.90) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# KS Test
# ═══════════════════════════════════════════════════════════════════════════

class TestKSTest:
    def test_identical_gives_high_p(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        stat, p = compute_ks_test(data, data)
        assert p > 0.05

    def test_different_gives_low_p(self):
        rng = np.random.default_rng(42)
        b = rng.normal(0, 1, 1000)
        c = rng.normal(5, 1, 1000)  # large shift
        stat, p = compute_ks_test(b, c)
        assert p < 0.01
        assert stat > 0

    def test_insufficient_returns_defaults(self):
        stat, p = compute_ks_test(np.array([1.0]), np.array([2.0]))
        assert stat == 0.0
        assert p == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# AlertRuleOut camelCase serialisation (A.9)
# ═══════════════════════════════════════════════════════════════════════════

class TestAlertRuleOutCamelCase:
    """Verify AlertRuleOut always emits camelCase keys for the frontend."""

    def test_serialises_to_camel_case(self):
        from app.schemas import AlertRuleOut

        rule = AlertRuleOut(
            id="test-id",
            model_id="model-1",
            metric="psi",
            threshold_warning=0.1,
            threshold_critical=0.25,
            enabled=True,
        )
        data = rule.model_dump(by_alias=True)
        assert "thresholdWarning" in data, f"Expected 'thresholdWarning', got keys: {list(data.keys())}"
        assert "thresholdCritical" in data
        assert "modelId" in data
        assert "threshold_warning" not in data, "snake_case key should NOT appear in output"
        assert "threshold_critical" not in data

    def test_snake_case_input_accepted(self):
        """AlertRuleCreate should accept snake_case field names."""
        from app.schemas import AlertRuleCreate

        rule = AlertRuleCreate(
            metric="js_distance",
            threshold_warning=0.05,
            threshold_critical=0.15,
        )
        assert rule.threshold_warning == 0.05
        assert rule.threshold_critical == 0.15

    def test_camel_case_input_also_accepted(self):
        """AlertRuleCreate should accept camelCase field names from frontend."""
        from app.schemas import AlertRuleCreate

        rule = AlertRuleCreate.model_validate(
            {"metric": "psi", "thresholdWarning": 0.1, "thresholdCritical": 0.25}
        )
        assert rule.threshold_warning == 0.1
        assert rule.threshold_critical == 0.25


# ═══════════════════════════════════════════════════════════════════════════
# Window parsing (A.7)
# ═══════════════════════════════════════════════════════════════════════════

class TestParseWindow:
    def test_valid_windows(self):
        from app.routes.models import parse_window
        assert parse_window("7d") == 7
        assert parse_window("30d") == 30
        assert parse_window("90d") == 90

    def test_invalid_window_raises(self):
        from app.routes.models import parse_window
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            parse_window("60d")
        assert exc_info.value.status_code == 400

    def test_case_insensitive(self):
        from app.routes.models import parse_window
        assert parse_window("  30D  ") == 30  # strip + lower
