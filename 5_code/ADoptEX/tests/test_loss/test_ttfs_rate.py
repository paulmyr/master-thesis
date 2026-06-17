"""Tests for loss.ttfs_rate — time-to-first-spike + firing-rate loss."""

import jax
import jax.numpy as jnp
import pytest

from ADoptEX.loss.ttfs_rate import TTFSRateLossConfig, ttfs_rate_loss

# =========================================================================
# TTFSRateLossConfig
# =========================================================================


class TestTTFSRateLossConfig:
    def test_defaults(self):
        cfg = TTFSRateLossConfig()
        assert cfg.weight_t_first == 1.0
        assert cfg.weight_firing_freq == 1.0
        assert cfg.missing_spike_penalty == 3.0
        assert cfg.epsilon == 1e-6

    def test_custom_values(self):
        cfg = TTFSRateLossConfig(
            weight_t_first=2.0,
            weight_firing_freq=0.5,
            missing_spike_penalty=5.0,
            epsilon=1e-4,
        )
        assert cfg.weight_t_first == 2.0
        assert cfg.weight_firing_freq == 0.5
        assert cfg.missing_spike_penalty == 5.0
        assert cfg.epsilon == 1e-4


# =========================================================================
# ttfs_rate_loss
# =========================================================================


class TestTTFSRateLoss:
    def test_perfect_match_is_zero(self):
        loss = ttfs_rate_loss(
            sim_t_first_spike=jnp.array(10.0),
            sim_firing_frequency=jnp.array(30.0),
            sim_has_spike=jnp.array(1.0),
            exp_t_first_spike=jnp.array(10.0),
            exp_firing_frequency=jnp.array(30.0),
            exp_has_spike=jnp.array(1.0),
        )
        assert float(loss) == pytest.approx(0.0, abs=1e-5)

    def test_first_spike_error_counts_when_both_fire(self):
        # 20% relative error on t_first, frequency matches
        loss = ttfs_rate_loss(
            sim_t_first_spike=jnp.array(12.0),
            sim_firing_frequency=jnp.array(30.0),
            sim_has_spike=jnp.array(1.0),
            exp_t_first_spike=jnp.array(10.0),
            exp_firing_frequency=jnp.array(30.0),
            exp_has_spike=jnp.array(1.0),
        )
        assert float(loss) == pytest.approx(0.2, abs=1e-4)

    def test_frequency_error_counts(self):
        # frequency off by 50%, t_first matches
        loss = ttfs_rate_loss(
            sim_t_first_spike=jnp.array(10.0),
            sim_firing_frequency=jnp.array(15.0),
            sim_has_spike=jnp.array(1.0),
            exp_t_first_spike=jnp.array(10.0),
            exp_firing_frequency=jnp.array(30.0),
            exp_has_spike=jnp.array(1.0),
        )
        assert float(loss) == pytest.approx(0.5, abs=1e-4)

    def test_missing_spike_penalty_applied(self):
        # data fires, sim silent -> flat penalty (and freq error of 1.0)
        cfg = TTFSRateLossConfig(missing_spike_penalty=3.0)
        loss = ttfs_rate_loss(
            sim_t_first_spike=jnp.array(0.0),
            sim_firing_frequency=jnp.array(0.0),
            sim_has_spike=jnp.array(0.0),
            exp_t_first_spike=jnp.array(10.0),
            exp_firing_frequency=jnp.array(30.0),
            exp_has_spike=jnp.array(1.0),
            config=cfg,
        )
        # penalty 3.0 + frequency relative error |0-30|/30 = 1.0
        assert float(loss) == pytest.approx(4.0, abs=1e-4)

    def test_first_spike_gated_off_when_data_silent(self):
        # data silent -> t_first term excluded regardless of sim t_first.
        # Two sims matching the (silent) frequency but with different t_first
        # must give the same (zero) loss.
        kwargs = dict(
            sim_firing_frequency=jnp.array(0.0),
            sim_has_spike=jnp.array(1.0),
            exp_t_first_spike=jnp.array(0.0),
            exp_firing_frequency=jnp.array(0.0),
            exp_has_spike=jnp.array(0.0),
        )
        loss_a = ttfs_rate_loss(sim_t_first_spike=jnp.array(10.0), **kwargs)
        loss_b = ttfs_rate_loss(sim_t_first_spike=jnp.array(999.0), **kwargs)
        assert float(loss_a) == pytest.approx(0.0, abs=1e-4)
        assert float(loss_b) == pytest.approx(0.0, abs=1e-4)

    def test_weights_scale_terms(self):
        base = ttfs_rate_loss(
            sim_t_first_spike=jnp.array(12.0),
            sim_firing_frequency=jnp.array(30.0),
            sim_has_spike=jnp.array(1.0),
            exp_t_first_spike=jnp.array(10.0),
            exp_firing_frequency=jnp.array(30.0),
            exp_has_spike=jnp.array(1.0),
        )
        scaled = ttfs_rate_loss(
            sim_t_first_spike=jnp.array(12.0),
            sim_firing_frequency=jnp.array(30.0),
            sim_has_spike=jnp.array(1.0),
            exp_t_first_spike=jnp.array(10.0),
            exp_firing_frequency=jnp.array(30.0),
            exp_has_spike=jnp.array(1.0),
            config=TTFSRateLossConfig(weight_t_first=3.0),
        )
        assert float(scaled) == pytest.approx(3.0 * float(base), rel=1e-4)

    def test_differentiable(self):
        def f(t_first, freq):
            return ttfs_rate_loss(
                sim_t_first_spike=t_first,
                sim_firing_frequency=freq,
                sim_has_spike=jnp.array(1.0),
                exp_t_first_spike=jnp.array(10.0),
                exp_firing_frequency=jnp.array(30.0),
                exp_has_spike=jnp.array(1.0),
            )

        grads = jax.grad(f, argnums=(0, 1))(jnp.array(12.0), jnp.array(20.0))
        assert all(jnp.isfinite(g) for g in grads)
        # both features off-target -> non-zero gradient
        assert float(grads[0]) != 0.0
        assert float(grads[1]) != 0.0