"""Tests for loss.dtw — Soft-DTW + MAE loss."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ADoptEX.loss.dtw import (SoftDTWLossConfig, _downsample, mae_loss,
                              soft_dtw, soft_dtw_mae_loss)

# =========================================================================
# _downsample
# =========================================================================


class TestDownsample:
    def test_noop_below_max(self):
        x = jnp.arange(50.0)
        result = _downsample(x, max_length=100)
        assert len(result) == 50

    def test_correct_output_length(self):
        x = jnp.arange(500.0)
        result = _downsample(x, max_length=100)
        assert len(result) == 100

    def test_preserves_endpoints(self):
        x = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = _downsample(x, max_length=3)
        assert float(result[0]) == pytest.approx(10.0)
        assert float(result[-1]) == pytest.approx(50.0)

    def test_linear_signal_preserved(self):
        """A linear signal should remain linear after downsampling."""
        x = jnp.linspace(0.0, 100.0, 1000)
        result = _downsample(x, max_length=50)
        expected = jnp.linspace(0.0, 100.0, 50)
        np.testing.assert_allclose(np.array(result), np.array(expected), atol=0.5)


# =========================================================================
# soft_dtw
# =========================================================================


class TestSoftDTW:
    def test_identical_traces_minimal(self):
        v = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0])
        d = float(soft_dtw(v, v, gamma=1.0))
        # Identical traces: diagonal cost is 0 (squared Euclidean),
        # accumulated softmin bias keeps the result small.
        assert d < 5.0

    def test_shifted_less_than_different(self):
        """Shifted trace should have lower DTW distance than a very different trace."""
        base = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0])
        shifted = jnp.array([0.0, 0.0, 1.0, 0.0, 1.0])
        different = jnp.array([5.0, 5.0, 5.0, 5.0, 5.0])
        d_shifted = float(soft_dtw(base, shifted, gamma=1.0))
        d_different = float(soft_dtw(base, different, gamma=1.0))
        assert d_shifted < d_different

    def test_symmetric(self):
        a = jnp.array([1.0, 3.0, 2.0])
        b = jnp.array([2.0, 1.0, 3.0])
        assert float(soft_dtw(a, b, gamma=1.0)) == pytest.approx(
            float(soft_dtw(b, a, gamma=1.0)), abs=1e-4
        )

    def test_differentiable(self):
        exp = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0])
        g = jax.grad(lambda s: soft_dtw(s, exp, gamma=1.0))(
            jnp.array([1.5, 2.5, 3.5, 2.5, 1.5])
        )
        assert jnp.all(jnp.isfinite(g))

    def test_nonneg_for_distant_traces(self):
        """DTW distance between very different traces should be positive."""
        a = jnp.zeros(10)
        b = jnp.ones(10) * 100.0
        d = float(soft_dtw(a, b, gamma=1.0))
        assert d > 0


# =========================================================================
# mae_loss
# =========================================================================


class TestMAELoss:
    def test_identical_zero(self):
        v = jnp.array([1.0, 2.0, 3.0])
        assert float(mae_loss(v, v)) == pytest.approx(0.0)

    def test_known_value(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([2.0, 4.0, 6.0])
        # MAE = mean(|1,2,3|) = 2.0
        assert float(mae_loss(a, b)) == pytest.approx(2.0)

    def test_truncation(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([1.0, 2.0, 999.0])
        assert float(mae_loss(a, b)) == pytest.approx(0.0)

    def test_symmetric(self):
        a = jnp.array([1.0, 5.0])
        b = jnp.array([3.0, 2.0])
        assert float(mae_loss(a, b)) == pytest.approx(float(mae_loss(b, a)))


# =========================================================================
# soft_dtw_mae_loss (combined)
# =========================================================================


class TestSoftDTWMAELoss:
    def test_weight_dtw_zero(self):
        """Setting DTW weight to 0 gives pure MAE."""
        a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0])
        cfg = SoftDTWLossConfig(weight_dtw=0.0, weight_mae=1.0, max_length=None)
        loss = float(soft_dtw_mae_loss(a, b, cfg))
        expected_mae = float(mae_loss(a, b))
        assert loss == pytest.approx(expected_mae, abs=1e-4)

    def test_weight_mae_zero(self):
        """Setting MAE weight to 0 should give pure DTW loss."""
        a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0])
        cfg = SoftDTWLossConfig(
            weight_dtw=1.0, weight_mae=0.0, normalize=False, max_length=None
        )
        loss = float(soft_dtw_mae_loss(a, b, cfg))
        expected_dtw = float(soft_dtw(a, b, gamma=cfg.gamma_sdtw))
        assert loss == pytest.approx(expected_dtw, abs=1e-4)

    def test_downsampling_trigger(self):
        """Traces longer than max_length should still produce a finite result."""
        a = jnp.ones(300)
        b = jnp.ones(300) * 2.0
        cfg = SoftDTWLossConfig(max_length=50)
        loss = soft_dtw_mae_loss(a, b, cfg)
        assert jnp.isfinite(loss)

    def test_normalization(self):
        """Normalised DTW distance should differ from unnormalised."""
        a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0])
        cfg_norm = SoftDTWLossConfig(normalize=True, weight_mae=0.0, max_length=None)
        cfg_raw = SoftDTWLossConfig(normalize=False, weight_mae=0.0, max_length=None)
        l_norm = float(soft_dtw_mae_loss(a, b, cfg_norm))
        l_raw = float(soft_dtw_mae_loss(a, b, cfg_raw))
        assert l_norm != pytest.approx(l_raw, abs=1e-3)
