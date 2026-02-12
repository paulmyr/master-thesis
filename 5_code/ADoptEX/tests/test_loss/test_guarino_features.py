"""Tests for loss.guarino — soft feature primitives and GuarinoFeatureExtractor."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ADoptEX.loss.guarino import (GuarinoFeatureExtractor, GuarinoFeatures,
                          soft_firing_frequency, soft_inverse_first_isi,
                          soft_inverse_last_isi, soft_spike_count,
                          soft_time_to_last_spike, soft_time_to_nth_spike,
                          soft_time_to_second_last_spike)

# =========================================================================
# soft_spike_count
# =========================================================================


class TestSoftSpikeCount:
    def test_zero_trace(self, empty_spike_trace):
        assert float(soft_spike_count(empty_spike_trace)) == pytest.approx(0.0)

    def test_binary_trace(self, binary_spike_trace_3):
        assert float(soft_spike_count(binary_spike_trace_3)) == pytest.approx(3.0)

    def test_soft_trace(self, soft_spike_trace_3):
        assert float(soft_spike_count(soft_spike_trace_3)) == pytest.approx(
            3.0, abs=0.1
        )


# =========================================================================
# soft_time_to_nth_spike
# =========================================================================


class TestSoftTimeToNthSpike:
    """Tests use binary_spike_trace_3: spikes at idx 100, 400, 700 with dt=0.1 ms."""

    def test_first_spike(self, binary_spike_trace_3):
        t = soft_time_to_nth_spike(binary_spike_trace_3, n=1, dt_ms=0.1)
        assert float(t) == pytest.approx(10.0, abs=1.0)  # idx 100 * 0.1 = 10 ms

    def test_second_spike(self, binary_spike_trace_3):
        t = soft_time_to_nth_spike(binary_spike_trace_3, n=2, dt_ms=0.1)
        assert float(t) == pytest.approx(40.0, abs=1.0)  # idx 400 * 0.1 = 40 ms

    def test_third_spike(self, binary_spike_trace_3):
        t = soft_time_to_nth_spike(binary_spike_trace_3, n=3, dt_ms=0.1)
        assert float(t) == pytest.approx(70.0, abs=1.0)  # idx 700 * 0.1 = 70 ms

    def test_monotonically_increasing(self, binary_spike_trace_3):
        t1 = float(soft_time_to_nth_spike(binary_spike_trace_3, n=1, dt_ms=0.1))
        t2 = float(soft_time_to_nth_spike(binary_spike_trace_3, n=2, dt_ms=0.1))
        t3 = float(soft_time_to_nth_spike(binary_spike_trace_3, n=3, dt_ms=0.1))
        assert t1 < t2 < t3

    def test_nonexistent_spike_degeneracy(self, single_spike_trace):
        """Requesting a spike beyond the count should still return a finite value."""
        t = soft_time_to_nth_spike(single_spike_trace, n=5, dt_ms=0.1)
        assert jnp.isfinite(t)


# =========================================================================
# soft_time_to_last_spike / soft_time_to_second_last_spike
# =========================================================================


class TestSoftTimeToLastSpike:
    def test_last_spike_position(self, binary_spike_trace_3):
        t = soft_time_to_last_spike(binary_spike_trace_3, dt_ms=0.1)
        assert float(t) == pytest.approx(70.0, abs=1.5)

    def test_second_last_spike_position(self, binary_spike_trace_3):
        t = soft_time_to_second_last_spike(binary_spike_trace_3, dt_ms=0.1)
        assert float(t) == pytest.approx(40.0, abs=1.5)

    def test_last_after_second_last(self, binary_spike_trace_3):
        t_last = float(soft_time_to_last_spike(binary_spike_trace_3, dt_ms=0.1))
        t_2nd = float(soft_time_to_second_last_spike(binary_spike_trace_3, dt_ms=0.1))
        assert t_last > t_2nd


# =========================================================================
# soft_inverse_first_isi / soft_inverse_last_isi
# =========================================================================


class TestSoftInverseISI:
    def test_first_isi_value(self, binary_spike_trace_3):
        """ISI between spike 1 (10 ms) and spike 2 (40 ms) → inv ≈ 1/30."""
        inv = soft_inverse_first_isi(binary_spike_trace_3, dt_ms=0.1)
        assert float(inv) == pytest.approx(1.0 / 30.0, rel=0.15)

    def test_last_isi_value(self, binary_spike_trace_3):
        """ISI between second-last (40 ms) and last (70 ms) → inv ≈ 1/30."""
        inv = soft_inverse_last_isi(binary_spike_trace_3, dt_ms=0.1)
        assert float(inv) == pytest.approx(1.0 / 30.0, rel=0.15)

    def test_epsilon_safety(self, single_spike_trace):
        """With only 1 spike, ISI degenerates but should not be NaN/Inf."""
        inv = soft_inverse_first_isi(single_spike_trace, dt_ms=0.1)
        assert jnp.isfinite(inv)


# =========================================================================
# soft_firing_frequency
# =========================================================================


class TestSoftFiringFrequency:
    def test_known_rate(self, binary_spike_trace_3):
        """3 spikes in 100 ms → 30 Hz."""
        freq = soft_firing_frequency(binary_spike_trace_3, stim_duration_ms=100.0)
        assert float(freq) == pytest.approx(30.0)

    def test_zero_spikes(self, empty_spike_trace):
        freq = soft_firing_frequency(empty_spike_trace, stim_duration_ms=100.0)
        assert float(freq) == pytest.approx(0.0)

    def test_scaling_with_count(self, binary_spike_trace_3):
        """Halving duration doubles frequency."""
        f1 = float(soft_firing_frequency(binary_spike_trace_3, stim_duration_ms=100.0))
        f2 = float(soft_firing_frequency(binary_spike_trace_3, stim_duration_ms=50.0))
        assert f2 == pytest.approx(2 * f1)


# =========================================================================
# GuarinoFeatureExtractor
# =========================================================================


class TestGuarinoFeatureExtractor:
    def _make_extractor(self, dt_ms=0.1, stim_dur=100.0, stim_end_idx=999):
        return GuarinoFeatureExtractor(
            dt_ms=dt_ms,
            stim_duration_ms=stim_dur,
            stim_end_index=stim_end_idx,
        )

    def test_returns_correct_type(self, binary_spike_trace_3):
        ext = self._make_extractor()
        voltage = jnp.full(1000, -65.0)
        feats = ext.extract(voltage, binary_spike_trace_3)
        assert isinstance(feats, GuarinoFeatures)

    def test_validity_flags_3_spikes(self, binary_spike_trace_3):
        ext = self._make_extractor()
        voltage = jnp.full(1000, -65.0)
        feats = ext.extract(voltage, binary_spike_trace_3)
        assert float(feats.has_first_spike) > 0.99
        assert float(feats.has_second_spike) > 0.99
        assert float(feats.has_third_spike) > 0.99

    def test_validity_flags_0_spikes(self, empty_spike_trace):
        ext = self._make_extractor()
        voltage = jnp.full(1000, -65.0)
        feats = ext.extract(voltage, empty_spike_trace)
        assert float(feats.has_first_spike) < 0.01
        assert float(feats.has_second_spike) < 0.01
        assert float(feats.has_third_spike) < 0.01

    def test_validity_flags_1_spike(self, single_spike_trace):
        ext = self._make_extractor()
        voltage = jnp.full(1000, -65.0)
        feats = ext.extract(voltage, single_spike_trace)
        assert float(feats.has_first_spike) > 0.99
        assert float(feats.has_second_spike) < 0.01

    def test_v_stim_end(self, binary_spike_trace_3):
        ext = self._make_extractor(stim_end_idx=500)
        voltage = jnp.arange(1000, dtype=jnp.float32)
        feats = ext.extract(voltage, binary_spike_trace_3)
        assert float(feats.v_stim_end) == pytest.approx(500.0)


# =========================================================================
# Differentiability
# =========================================================================


class TestDifferentiability:
    """jax.grad through each soft primitive should return finite values."""

    @pytest.fixture
    def spike_trace(self):
        t = jnp.zeros(500)
        t = t.at[100].set(1.0)
        t = t.at[300].set(1.0)
        return t

    def test_grad_soft_spike_count(self, spike_trace):
        g = jax.grad(lambda s: soft_spike_count(s))(spike_trace)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_soft_time_to_nth_spike(self, spike_trace):
        g = jax.grad(lambda s: soft_time_to_nth_spike(s, n=1, dt_ms=0.1))(spike_trace)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_soft_time_to_last_spike(self, spike_trace):
        g = jax.grad(lambda s: soft_time_to_last_spike(s, dt_ms=0.1))(spike_trace)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_soft_inverse_first_isi(self, spike_trace):
        g = jax.grad(lambda s: soft_inverse_first_isi(s, dt_ms=0.1))(spike_trace)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_soft_firing_frequency(self, spike_trace):
        g = jax.grad(lambda s: soft_firing_frequency(s, stim_duration_ms=50.0))(
            spike_trace
        )
        assert jnp.all(jnp.isfinite(g))
