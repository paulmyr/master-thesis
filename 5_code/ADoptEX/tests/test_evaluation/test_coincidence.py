"""Tests for evaluation.coincidence — the primary thesis evaluation metric."""

import numpy as np
import pytest

from ADoptEX.evaluation.coincidence import (
    CoincidenceResult, coincidence_factor, coincidence_factor_from_traces,
    count_coincidences, detect_spike_times,
    detect_spike_times_from_spike_trace, global_performance,
    intrinsic_reliability)

# =========================================================================
# count_coincidences
# =========================================================================


class TestCountCoincidences:
    def test_identical_trains(self):
        spikes = np.array([10.0, 50.0, 100.0])
        assert count_coincidences(spikes, spikes, delta_ms=2.0) == 3

    def test_no_overlap(self):
        data = np.array([10.0, 50.0])
        model = np.array([20.0, 60.0])
        assert count_coincidences(data, model, delta_ms=2.0) == 0

    def test_partial_overlap(self):
        data = np.array([10.0, 50.0, 100.0])
        model = np.array([11.0, 60.0, 99.0])  # 10±2 ✓, 50±2 ✗, 100±2 ✓
        assert count_coincidences(data, model, delta_ms=2.0) == 2

    def test_greedy_one_to_one(self):
        """Each data spike can only be matched once."""
        data = np.array([10.0])
        model = np.array([9.5, 10.5])  # Both within window of data[0]
        assert count_coincidences(data, model, delta_ms=2.0) == 1

    def test_empty_data(self):
        assert count_coincidences(np.array([]), np.array([10.0]), delta_ms=2.0) == 0

    def test_empty_model(self):
        assert count_coincidences(np.array([10.0]), np.array([]), delta_ms=2.0) == 0

    def test_both_empty(self):
        assert count_coincidences(np.array([]), np.array([]), delta_ms=2.0) == 0


# =========================================================================
# coincidence_factor
# =========================================================================


class TestCoincidenceFactor:
    def test_perfect_prediction(self):
        spikes = np.array([10.0, 50.0, 100.0, 150.0, 200.0])
        result = coincidence_factor(spikes, spikes, duration_ms=250.0)
        assert isinstance(result, CoincidenceResult)
        # Perfect match should give Gamma close to 1
        assert result.gamma > 0.9

    def test_shifted_within_window(self):
        data = np.array([10.0, 50.0, 100.0, 150.0, 200.0])
        model = data + 1.0  # Shift by 1 ms (within default Δ=2 ms)
        result = coincidence_factor(data, model, duration_ms=250.0)
        assert result.gamma > 0.8

    def test_shifted_outside_window(self):
        data = np.array([10.0, 50.0, 100.0, 150.0, 200.0])
        model = data + 5.0  # Shift by 5 ms (outside Δ=2 ms)
        result = coincidence_factor(data, model, duration_ms=250.0)
        assert result.gamma < 0.3

    def test_both_empty(self):
        result = coincidence_factor(np.array([]), np.array([]), duration_ms=100.0)
        assert result.gamma == 1.0

    def test_one_empty(self):
        result = coincidence_factor(
            np.array([10.0, 50.0]), np.array([]), duration_ms=100.0
        )
        assert result.gamma == 0.0

    def test_normalization_formula(self):
        """Check that the formula components are correctly computed."""
        data = np.array([50.0, 150.0])
        model = np.array([50.0, 150.0])
        result = coincidence_factor(data, model, duration_ms=200.0, delta_ms=2.0)
        # f_model = 2 / 0.2s = 10 Hz
        assert pytest.approx(result.firing_rate_model, rel=1e-3) == 10.0
        assert result.n_data == 2
        assert result.n_model == 2

    def test_clipping(self):
        """Result should always be in [-1, 1]."""
        data = np.array([10.0, 50.0, 100.0])
        model = np.array([10.0, 50.0, 100.0])
        result = coincidence_factor(data, model, duration_ms=150.0)
        assert -1.0 <= result.gamma <= 1.0


# =========================================================================
# detect_spike_times
# =========================================================================


class TestDetectSpikeTimes:
    def test_correct_times(self, voltage_with_spikes, dt_ms):
        times = detect_spike_times(voltage_with_spikes, dt=dt_ms, threshold=0.0)
        assert len(times) == 3
        np.testing.assert_allclose(times, [50.0, 150.0, 250.0], atol=dt_ms * 2)

    def test_rising_edge_only(self, dt_ms):
        """Only the first sample above threshold counts (rising edge)."""
        trace = np.array([-1.0, -1.0, 5.0, 5.0, 5.0, -1.0])
        times = detect_spike_times(trace, dt=dt_ms, threshold=0.0)
        assert len(times) == 1

    def test_no_spikes(self, voltage_no_spikes, dt_ms):
        times = detect_spike_times(voltage_no_spikes, dt=dt_ms, threshold=0.0)
        assert len(times) == 0

    def test_custom_threshold(self, dt_ms):
        trace = np.array([-80.0, -50.0, -30.0, -50.0])
        # Threshold = -40 should detect the crossing at index 2
        times = detect_spike_times(trace, dt=dt_ms, threshold=-40.0)
        assert len(times) == 1


# =========================================================================
# detect_spike_times_from_spike_trace
# =========================================================================


class TestDetectSpikeTimesFromSpikeTrace:
    def test_binary_trace(self):
        trace = np.array([0, 0, 1, 0, 0, 1, 0])
        times = detect_spike_times_from_spike_trace(trace, dt=0.1, threshold=0.5)
        np.testing.assert_allclose(times, [0.2, 0.5])

    def test_consecutive_collapse(self):
        """Consecutive above-threshold samples collapse to one spike."""
        trace = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        times = detect_spike_times_from_spike_trace(trace, dt=0.1, threshold=0.5)
        assert len(times) == 2  # indices 1 and 6

    def test_custom_threshold(self):
        trace = np.array([0.0, 0.3, 0.8, 0.1, 0.9])
        times = detect_spike_times_from_spike_trace(trace, dt=1.0, threshold=0.7)
        np.testing.assert_allclose(times, [2.0, 4.0])


# =========================================================================
# intrinsic_reliability
# =========================================================================


class TestIntrinsicReliability:
    def test_identical_trials(self):
        spikes = np.array([10.0, 50.0, 100.0])
        trials = [spikes, spikes, spikes]
        r = intrinsic_reliability(trials, duration_ms=150.0)
        assert r > 0.9

    def test_single_trial(self):
        trials = [np.array([10.0, 50.0])]
        assert intrinsic_reliability(trials, duration_ms=100.0) == 1.0

    def test_jittered_trials(self):
        base = np.array([50.0, 150.0, 250.0, 350.0])
        rng = np.random.default_rng(42)
        trials = [base + rng.normal(0, 0.5, size=4) for _ in range(5)]
        r = intrinsic_reliability(trials, duration_ms=400.0)
        # Small jitter should give high reliability
        assert r > 0.7


# =========================================================================
# global_performance
# =========================================================================


class TestGlobalPerformance:
    def test_perfect_case(self):
        gp = global_performance([1.0, 1.0], [1.0, 1.0])
        assert pytest.approx(gp, abs=1e-6) == 1.0

    def test_normalization_by_reliability(self):
        gp = global_performance([0.5, 0.5], [0.5, 0.5])
        assert pytest.approx(gp, abs=1e-6) == 1.0

    def test_low_reliability_exclusion(self):
        """Stimuli with reliability <= 0.1 are excluded."""
        gp = global_performance([0.8, 0.0], [0.9, 0.05])
        # Only first stimulus counted: 0.8/0.9
        assert pytest.approx(gp, abs=1e-3) == 0.8 / 0.9

    def test_empty_input(self):
        assert global_performance([], []) == 0.0
