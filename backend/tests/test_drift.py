"""
Unit tests for PSI and JS-distance functions.

Uses known distributions where we can verify expected score ranges:
- Identical distributions → PSI ≈ 0, JS ≈ 0
- Slightly shifted distributions → moderate PSI/JS
- Drastically different distributions → high PSI/JS
"""

import numpy as np
import pytest
from app.services.drift import (
    compute_psi,
    compute_psi_from_histograms,
    compute_js_distance,
    psi_to_status,
    compute_health_score,
    compute_performance_delta,
)


# ── PSI Tests ──────────────────────────────────────


class TestComputePSI:
    def test_identical_distributions_gives_near_zero(self):
        """Same distribution should have PSI ≈ 0."""
        rng = np.random.default_rng(42)
        data = rng.normal(50, 10, 5000)
        psi = compute_psi(data, data)
        assert psi < 0.01, f"PSI for identical data should be ~0, got {psi}"

    def test_slight_shift_gives_low_psi(self):
        """A small mean shift should produce PSI in stable/low-warning range."""
        rng = np.random.default_rng(42)
        baseline = rng.normal(50, 10, 5000)
        current = rng.normal(51, 10, 5000)  # slight shift
        psi = compute_psi(baseline, current)
        assert psi < 0.1, f"Slight shift should be stable, got PSI={psi}"

    def test_moderate_shift_gives_warning_psi(self):
        """A moderate shift should produce PSI in the warning range (0.1–0.25)."""
        rng = np.random.default_rng(42)
        baseline = rng.normal(50, 10, 5000)
        current = rng.normal(55, 12, 5000)  # moderate shift + wider spread
        psi = compute_psi(baseline, current)
        assert 0.05 < psi < 0.5, f"Moderate shift PSI should be in mid-range, got {psi}"

    def test_drastic_shift_gives_critical_psi(self):
        """A large distribution change should produce PSI > 0.25."""
        rng = np.random.default_rng(42)
        baseline = rng.normal(50, 10, 5000)
        current = rng.normal(80, 5, 5000)  # completely different
        psi = compute_psi(baseline, current)
        assert psi > 0.25, f"Drastic shift should be critical, got PSI={psi}"

    def test_empty_arrays_return_zero(self):
        assert compute_psi(np.array([]), np.array([1, 2, 3])) == 0.0
        assert compute_psi(np.array([1, 2, 3]), np.array([])) == 0.0

    def test_psi_is_non_negative(self):
        """PSI should always be >= 0."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            b = rng.normal(0, 1, 1000)
            c = rng.normal(rng.uniform(-2, 2), rng.uniform(0.5, 3), 1000)
            psi = compute_psi(b, c)
            assert psi >= 0, f"PSI should be non-negative, got {psi}"


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


# ── JS-Distance Tests ──────────────────────────────


class TestComputeJSDistance:
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


# ── Status Mapping Tests ──────────────────────────


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


# ── Health Score Tests ─────────────────────────────


class TestComputeHealthScore:
    def test_perfect_health(self):
        score = compute_health_score(
            avg_drift=0.0,
            performance_ratio=1.0,
            open_critical_alerts=0,
        )
        assert score == 100.0

    def test_moderate_degradation(self):
        score = compute_health_score(
            avg_drift=0.15,
            performance_ratio=0.90,
            open_critical_alerts=1,
        )
        assert 60 < score < 95, f"Expected moderate score, got {score}"

    def test_critical_state(self):
        score = compute_health_score(
            avg_drift=0.5,
            performance_ratio=0.6,
            open_critical_alerts=5,
        )
        assert score < 60, f"Expected low score, got {score}"

    def test_clamped_to_zero(self):
        score = compute_health_score(
            avg_drift=1.0,
            performance_ratio=0.0,
            open_critical_alerts=100,
        )
        assert score == 0.0

    def test_clamped_to_hundred(self):
        score = compute_health_score(
            avg_drift=-0.5,  # impossible but test clamping
            performance_ratio=2.0,
            open_critical_alerts=0,
        )
        assert score == 100.0


# ── Performance Delta Tests ────────────────────────


class TestPerformanceDelta:
    def test_positive_delta(self):
        assert compute_performance_delta(0.95, 0.90) == 5.0

    def test_negative_delta(self):
        assert compute_performance_delta(0.89, 0.94) == -5.0

    def test_zero_delta(self):
        assert compute_performance_delta(0.90, 0.90) == 0.0
