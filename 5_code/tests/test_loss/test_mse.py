"""Tests for loss.mse — baseline MSE loss."""

import jax
import jax.numpy as jnp
import pytest

from loss.mse import MSELossConfig, mse_loss


class TestMSELoss:
    def test_identical_zero(self):
        v = jnp.array([1.0, 2.0, 3.0])
        assert float(mse_loss(v, v)) == pytest.approx(0.0)

    def test_known_value(self):
        sim = jnp.array([1.0, 2.0, 3.0])
        exp = jnp.array([2.0, 3.0, 4.0])
        # MSE = mean((1,1,1)^2) = 1.0
        assert float(mse_loss(sim, exp)) == pytest.approx(1.0)

    def test_length_truncation(self):
        sim = jnp.array([1.0, 2.0, 3.0, 999.0])
        exp = jnp.array([1.0, 2.0, 3.0])
        # Extra element in sim is ignored → MSE = 0
        assert float(mse_loss(sim, exp)) == pytest.approx(0.0)

    def test_nonnegative(self):
        rng = jax.random.PRNGKey(0)
        sim = jax.random.normal(rng, (100,))
        exp = jax.random.normal(jax.random.PRNGKey(1), (100,))
        assert float(mse_loss(sim, exp)) >= 0.0

    def test_symmetric(self):
        a = jnp.array([1.0, 5.0, -3.0])
        b = jnp.array([2.0, -1.0, 4.0])
        assert float(mse_loss(a, b)) == pytest.approx(float(mse_loss(b, a)))

    def test_normalize_by_variance(self):
        sim = jnp.array([0.0, 0.0, 0.0])
        exp = jnp.array([1.0, 2.0, 3.0])
        cfg = MSELossConfig(normalize=True)
        loss_norm = float(mse_loss(sim, exp, cfg))
        loss_raw = float(mse_loss(sim, exp))
        # Normalised should differ from raw (var(exp) ≠ 1)
        assert loss_norm != pytest.approx(loss_raw, abs=1e-3)

    def test_constant_trace_epsilon_safety(self):
        """Normalising by near-zero variance should not produce Inf."""
        sim = jnp.array([1.0, 1.0, 1.0])
        exp = jnp.array([1.0, 1.0, 1.0])  # variance ~ 0
        cfg = MSELossConfig(normalize=True)
        loss = mse_loss(sim, exp, cfg)
        assert jnp.isfinite(loss)

    def test_single_element(self):
        assert float(mse_loss(jnp.array([3.0]), jnp.array([5.0]))) == pytest.approx(4.0)

    def test_differentiable(self):
        exp = jnp.array([1.0, 2.0, 3.0])
        g = jax.grad(lambda s: mse_loss(s, exp))(jnp.array([2.0, 3.0, 4.0]))
        assert jnp.all(jnp.isfinite(g))
