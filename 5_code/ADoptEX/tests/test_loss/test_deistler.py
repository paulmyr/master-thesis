"""Tests for loss.deistler — Deistler summary statistics loss."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ADoptEX.loss import inject_spike_peaks
from ADoptEX.loss.deistler import (DeistlerLossConfig, _window_stats,
                                   deistler_loss)

# =========================================================================
# DeistlerLossConfig
# =========================================================================


class TestDeistlerLossConfig:
    def test_defaults(self):
        cfg = DeistlerLossConfig()
        assert cfg.mean_norm == 8.0
        assert cfg.std_norm == 4.0
        assert cfg.epsilon == 1e-6

    def test_custom_values(self):
        cfg = DeistlerLossConfig(mean_norm=10.0, std_norm=5.0, epsilon=1e-8)
        assert cfg.mean_norm == 10.0
        assert cfg.std_norm == 5.0
        assert cfg.epsilon == 1e-8


# =========================================================================
# _window_stats
# =========================================================================


class TestWindowStats:
    def test_constant_voltage_zero_std(self):
        """Constant voltage should give mean=constant, std≈sqrt(epsilon)."""
        v = jnp.full(100, -65.0)
        mean, std = _window_stats(v, 0, 100)
        assert mean.item() == pytest.approx(-65.0)
        # std = sqrt(0 + epsilon) ≈ 0.001 due to numerical stability term
        assert std.item() == pytest.approx(np.sqrt(1e-6), rel=0.01)

    def test_known_mean(self):
        """Known linear ramp should give correct mean."""
        v = jnp.arange(10, dtype=jnp.float32)  # 0, 1, ..., 9
        mean, std = _window_stats(v, 0, 10)
        assert mean.item() == pytest.approx(4.5)

    def test_known_std(self):
        """Two-valued array should give known std."""
        v = jnp.array([0.0, 10.0, 0.0, 10.0])
        mean, std = _window_stats(v, 0, 4, epsilon=0.0)
        assert mean.item() == pytest.approx(5.0)
        assert std.item() == pytest.approx(5.0)

    def test_subwindow(self):
        """Stats should only use the specified window."""
        v = jnp.array([1.0, 1.0, 1.0, 100.0, 100.0])
        mean, _ = _window_stats(v, 0, 3)
        assert mean.item() == pytest.approx(1.0)


# =========================================================================
# deistler_loss
# =========================================================================


class TestDeistlerLoss:
    def test_identical_traces_near_zero(self):
        """Identical traces should give near-zero loss."""
        v = jnp.array(np.random.randn(1000) * 5.0 - 65.0)
        loss = deistler_loss(v, v, stim_start_index=200, stim_end_index=800)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_nonnegative(self):
        """Loss should always be >= 0."""
        sim = jnp.array(np.random.randn(500) * 5.0 - 65.0)
        exp = jnp.array(np.random.randn(500) * 5.0 - 60.0)
        loss = deistler_loss(sim, exp, stim_start_index=100, stim_end_index=400)
        assert loss.item() >= 0.0

    def test_different_means_detected(self):
        """Traces with different means should produce nonzero loss."""
        sim = jnp.full(500, -65.0)
        exp = jnp.full(500, -55.0)
        loss = deistler_loss(sim, exp, stim_start_index=100, stim_end_index=400)
        assert loss.item() > 0.0

    def test_different_stds_detected(self):
        """Traces with same mean but different stds should produce nonzero loss."""
        rng = np.random.RandomState(42)
        base = rng.randn(500)
        sim = jnp.array(base * 1.0 - 65.0)
        exp = jnp.array(base * 10.0 - 65.0)
        loss = deistler_loss(sim, exp, stim_start_index=100, stim_end_index=400)
        assert loss.item() > 0.0

    def test_standardization_effect(self):
        """Changing normalization constants should change the loss value."""
        sim = jnp.full(500, -65.0)
        exp = jnp.full(500, -55.0)
        cfg1 = DeistlerLossConfig(mean_norm=8.0)
        cfg2 = DeistlerLossConfig(mean_norm=16.0)  # halves the mean contribution
        loss1 = deistler_loss(
            sim, exp, stim_start_index=100, stim_end_index=400, config=cfg1
        )
        loss2 = deistler_loss(
            sim, exp, stim_start_index=100, stim_end_index=400, config=cfg2
        )
        # Doubling mean_norm should roughly halve the mean-driven loss
        assert loss2.item() < loss1.item()

    def test_length_truncation(self):
        """Mismatched lengths should be handled via truncation."""
        sim = jnp.full(300, -65.0)
        exp = jnp.full(500, -65.0)
        loss = deistler_loss(sim, exp, stim_start_index=50, stim_end_index=250)
        assert jnp.isfinite(loss)

    def test_four_statistics(self):
        """The loss should use exactly 4 statistics (2 windows x 2 stats)."""
        # Build traces where only the stimulus window differs
        n = 500
        stim_start = 100
        stim_end = 400

        sim = jnp.full(n, -65.0)
        exp = np.full(n, -65.0, dtype=np.float64)
        # Only change the stimulus window
        exp[stim_start:stim_end] = -55.0
        exp = jnp.array(exp)

        loss = deistler_loss(
            sim, exp, stim_start_index=stim_start, stim_end_index=stim_end
        )
        # Pre-stim stats are identical, stim mean differs by 10 mV
        # Expected: 2 * abs(10/8) / 4 = 0.3125 (rough, stds also differ slightly)
        assert loss.item() > 0.0

    def test_pre_stim_only_change(self):
        """Changing only pre-stimulus voltage should affect the loss."""
        n = 500
        exp = jnp.full(n, -65.0)
        sim = np.full(n, -65.0, dtype=np.float64)
        sim[:100] = -75.0  # different pre-stim
        sim = jnp.array(sim)
        loss = deistler_loss(sim, exp, stim_start_index=100, stim_end_index=400)
        assert loss.item() > 0.0

    def test_no_pre_stim_window(self):
        """stim_start_index=0 should work (2 stats instead of 4)."""
        sim = jnp.full(500, -65.0)
        exp = jnp.full(500, -55.0)
        loss = deistler_loss(sim, exp, stim_start_index=0, stim_end_index=500)
        assert jnp.isfinite(loss)
        assert loss.item() > 0.0

    def test_no_pre_stim_identical_near_zero(self):
        """Identical traces with stim_start=0 should give near-zero loss."""
        v = jnp.array(np.random.randn(500) * 5.0 - 65.0)
        loss = deistler_loss(v, v, stim_start_index=0, stim_end_index=500)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


# =========================================================================
# Differentiability
# =========================================================================


class TestDeistlerDifferentiability:
    def test_grad_wrt_sim_voltage(self):
        """Gradient of loss w.r.t. simulated voltage should be finite."""
        exp = jnp.array(np.random.randn(500) * 5.0 - 65.0)

        def loss(sim):
            return deistler_loss(sim, exp, stim_start_index=100, stim_end_index=400)

        sim = jnp.array(np.random.randn(500) * 5.0 - 65.0)
        g = jax.grad(loss)(sim)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_nonzero_for_different_traces(self):
        """Gradient should be nonzero when traces differ."""
        exp = jnp.full(500, -65.0)

        def loss(sim):
            return deistler_loss(sim, exp, stim_start_index=100, stim_end_index=400)

        sim = jnp.full(500, -55.0)
        g = jax.grad(loss)(sim)
        assert jnp.any(g != 0)

    def test_grad_near_zero_for_identical_traces(self):
        """Gradient should be near zero when traces are identical."""
        v = jnp.array(np.random.randn(500) * 5.0 - 65.0)

        def loss(sim):
            return deistler_loss(sim, v, stim_start_index=100, stim_end_index=400)

        g = jax.grad(loss)(v)
        assert jnp.max(jnp.abs(g)).item() < 0.01

    def test_grad_finite_no_pre_stim(self):
        """Gradient should be finite when stim_start_index=0."""
        exp = jnp.array(np.random.randn(500) * 5.0 - 65.0)

        def loss(sim):
            return deistler_loss(sim, exp, stim_start_index=0, stim_end_index=500)

        sim = jnp.array(np.random.randn(500) * 5.0 - 65.0)
        g = jax.grad(loss)(sim)
        assert jnp.all(jnp.isfinite(g))


# =========================================================================
# Spike peak injection
# =========================================================================


class TestDeistlerWithSpikePeaks:
    def test_spike_peak_changes_loss(self):
        """spike_peak_mv should change the Deistler loss value."""
        sim_voltage = jnp.array([-70.0, -60.0, -55.0, -55.0, -65.0])
        sim_spikes = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0])
        exp_voltage = jnp.array([-70.0, -60.0, 35.0, -55.0, -65.0])

        loss_without = float(
            deistler_loss(
                sim_voltage, exp_voltage, stim_start_index=0, stim_end_index=5
            )
        )

        sim_with_peaks = inject_spike_peaks(sim_voltage, sim_spikes, v_peak_mv=35.0)
        loss_with = float(
            deistler_loss(
                sim_with_peaks, exp_voltage, stim_start_index=0, stim_end_index=5
            )
        )

        assert loss_with < loss_without

    def test_spike_peak_none_is_default(self):
        """spike_peak_mv=None gives same config behavior as default."""
        cfg = DeistlerLossConfig(spike_peak_mv=None)
        assert cfg.spike_peak_mv is None

    def test_spike_peak_increases_std(self):
        """Injecting spike peaks should increase the voltage std."""
        voltage = jnp.full(100, -65.0)
        spikes = jnp.zeros(100).at[50].set(1.0)

        v_with_peaks = inject_spike_peaks(voltage, spikes, v_peak_mv=35.0)

        _, std_without = _window_stats(voltage, 0, 100, epsilon=0.0)
        _, std_with = _window_stats(v_with_peaks, 0, 100, epsilon=0.0)

        assert std_with.item() > std_without.item()
