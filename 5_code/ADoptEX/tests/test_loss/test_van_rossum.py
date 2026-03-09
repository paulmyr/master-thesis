"""Tests for loss.van_rossum — Van Rossum distance loss."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ADoptEX.loss.van_rossum import (VanRossumLossConfig, _build_kernel,
                                     spike_train_from_voltage,
                                     van_rossum_distance)

# =========================================================================
# VanRossumLossConfig
# =========================================================================


class TestVanRossumLossConfig:
    def test_defaults(self):
        cfg = VanRossumLossConfig()
        assert cfg.tau_ms == 10.0
        assert cfg.kernel_n_tau == 5.0
        assert cfg.weight_van_rossum == 1.0
        assert cfg.weight_subthreshold == 0.0
        assert cfg.subthreshold_clamp_mv == -40.0

    def test_custom_values(self):
        cfg = VanRossumLossConfig(
            tau_ms=5.0,
            kernel_n_tau=3.0,
            weight_van_rossum=2.0,
            weight_subthreshold=0.5,
            subthreshold_clamp_mv=-50.0,
        )
        assert cfg.tau_ms == 5.0
        assert cfg.kernel_n_tau == 3.0
        assert cfg.weight_van_rossum == 2.0
        assert cfg.weight_subthreshold == 0.5
        assert cfg.subthreshold_clamp_mv == -50.0


# =========================================================================
# _build_kernel
# =========================================================================


class TestBuildKernel:
    def test_correct_length(self):
        # tau=10ms, n_tau=5, dt=0.1ms -> 500 samples
        kernel = _build_kernel(tau_ms=10.0, kernel_n_tau=5.0, dt_ms=0.1)
        assert len(kernel) == 500

    def test_starts_at_one(self):
        kernel = _build_kernel(tau_ms=10.0, kernel_n_tau=5.0, dt_ms=0.1)
        assert float(kernel[0]) == pytest.approx(1.0)

    def test_monotonically_decreasing(self):
        kernel = _build_kernel(tau_ms=10.0, kernel_n_tau=5.0, dt_ms=0.1)
        diffs = np.diff(np.array(kernel))
        assert np.all(diffs <= 0)

    def test_endpoint_approx_exp_neg5(self):
        kernel = _build_kernel(tau_ms=10.0, kernel_n_tau=5.0, dt_ms=0.1)
        # Last element should be close to exp(-5) ≈ 0.0067
        assert float(kernel[-1]) == pytest.approx(np.exp(-5.0), rel=0.05)

    def test_minimum_length_one(self):
        # Very large dt relative to tau should still give length >= 1
        kernel = _build_kernel(tau_ms=0.01, kernel_n_tau=1.0, dt_ms=100.0)
        assert len(kernel) >= 1


# =========================================================================
# van_rossum_distance
# =========================================================================


class TestVanRossumDistance:
    def test_identical_trains_zero(self, binary_spike_trace_3, dt_ms):
        d = van_rossum_distance(binary_spike_trace_3, binary_spike_trace_3, dt_ms=0.1)
        assert float(d) == pytest.approx(0.0, abs=1e-6)

    def test_empty_vs_spiking_positive(self, binary_spike_trace_3, empty_spike_trace):
        d = van_rossum_distance(binary_spike_trace_3, empty_spike_trace, dt_ms=0.1)
        assert float(d) > 0

    def test_symmetric(self, binary_spike_trace_3, single_spike_trace):
        d1 = van_rossum_distance(binary_spike_trace_3, single_spike_trace, dt_ms=0.1)
        d2 = van_rossum_distance(single_spike_trace, binary_spike_trace_3, dt_ms=0.1)
        assert float(d1) == pytest.approx(float(d2), rel=1e-5)

    def test_nonneg(self, binary_spike_trace_3, single_spike_trace):
        d = van_rossum_distance(binary_spike_trace_3, single_spike_trace, dt_ms=0.1)
        assert float(d) >= 0

    def test_small_tau_more_timing_sensitive(self):
        """With small tau, a small timing shift should produce a larger distance."""
        # Two single-spike trains offset by 10 samples
        a = jnp.zeros(200)
        a = a.at[50].set(1.0)
        b = jnp.zeros(200)
        b = b.at[60].set(1.0)

        cfg_small = VanRossumLossConfig(tau_ms=2.0)
        cfg_large = VanRossumLossConfig(tau_ms=50.0)

        d_small = float(van_rossum_distance(a, b, dt_ms=0.1, config=cfg_small))
        d_large = float(van_rossum_distance(a, b, dt_ms=0.1, config=cfg_large))

        # Small tau decays faster -> bigger difference for same offset
        assert d_small > d_large

    def test_length_truncation(self):
        """Mismatched lengths should be handled via truncation."""
        a = jnp.zeros(100).at[10].set(1.0)
        b = jnp.zeros(200).at[10].set(1.0)
        d = van_rossum_distance(a, b, dt_ms=0.1)
        assert float(d) == pytest.approx(0.0, abs=1e-6)

    def test_differentiable(self, binary_spike_trace_3):
        exp = binary_spike_trace_3

        def loss(sim):
            return van_rossum_distance(sim, exp, dt_ms=0.1)

        # Soft spike trace as input
        sim = jnp.zeros(1000).at[100].set(0.8).at[400].set(0.9).at[700].set(1.0)
        g = jax.grad(loss)(sim)
        assert jnp.all(jnp.isfinite(g))

    def test_finite_with_empty_sim(self, empty_spike_trace, binary_spike_trace_3):
        d = van_rossum_distance(empty_spike_trace, binary_spike_trace_3, dt_ms=0.1)
        assert jnp.isfinite(d)

    def test_finite_with_empty_exp(self, binary_spike_trace_3, empty_spike_trace):
        d = van_rossum_distance(binary_spike_trace_3, empty_spike_trace, dt_ms=0.1)
        assert jnp.isfinite(d)

    def test_both_empty_zero(self, empty_spike_trace):
        d = van_rossum_distance(empty_spike_trace, empty_spike_trace, dt_ms=0.1)
        assert float(d) == pytest.approx(0.0, abs=1e-6)


# =========================================================================
# spike_train_from_voltage
# =========================================================================


class TestSpikeTrainFromVoltage:
    def test_no_spikes_all_zeros(self):
        voltage = jnp.full(100, -65.0)
        train = spike_train_from_voltage(voltage, threshold_mv=0.0)
        assert float(jnp.sum(train)) == 0.0

    def test_correct_spike_count(self):
        """Three threshold crossings -> three spikes."""
        voltage = np.full(1000, -65.0)
        # Create three rising-edge crossings
        for idx in [100, 400, 700]:
            voltage[idx : idx + 5] = 20.0
        train = spike_train_from_voltage(jnp.array(voltage), threshold_mv=0.0)
        assert int(jnp.sum(train)) == 3

    def test_correct_indices(self):
        """Spikes should be at rising-edge indices."""
        voltage = np.full(500, -65.0)
        voltage[100:110] = 20.0
        voltage[300:310] = 20.0
        train = spike_train_from_voltage(jnp.array(voltage), threshold_mv=0.0)
        spike_indices = np.where(np.array(train) > 0)[0]
        np.testing.assert_array_equal(spike_indices, [100, 300])

    def test_same_length_as_input(self):
        voltage = jnp.full(500, -65.0)
        train = spike_train_from_voltage(voltage)
        assert len(train) == 500

    def test_returns_jax_array(self):
        voltage = jnp.full(100, -65.0)
        train = spike_train_from_voltage(voltage)
        assert isinstance(train, jnp.ndarray)


# =========================================================================
# make_van_rossum_loss_fn (ValueError check only — no Jaxley)
# =========================================================================


class TestMakeVanRossumLossFn:
    def test_raises_without_voltage_when_subthreshold(self):
        from ADoptEX.loss.van_rossum import make_van_rossum_loss_fn

        cfg = VanRossumLossConfig(weight_subthreshold=1.0)
        with pytest.raises(ValueError, match="exp_voltage must be provided"):
            make_van_rossum_loss_fn(
                cell=None,
                data_stimuli=None,
                t_max=100.0,
                dt_ms=0.025,
                exp_spike_train=jnp.zeros(100),
                stim_end_index=4000,
                loss_config=cfg,
                exp_voltage=None,
            )
