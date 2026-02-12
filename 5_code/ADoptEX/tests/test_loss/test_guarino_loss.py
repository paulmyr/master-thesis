"""Tests for loss.guarino — loss composition, penalties, differentiability."""

import jax
import jax.numpy as jnp
import pytest

from ADoptEX.loss.guarino import (GuarinoFeatures, GuarinoLossConfig, guarino_loss,
                                  relative_error)

# =========================================================================
# relative_error
# =========================================================================


class TestRelativeError:
    def test_identical(self):
        assert float(relative_error(jnp.array(5.0), jnp.array(5.0))) == pytest.approx(
            0.0, abs=1e-5
        )

    def test_known_value(self):
        # |10 - 5| / |5| = 1.0
        assert float(relative_error(jnp.array(10.0), jnp.array(5.0))) == pytest.approx(
            1.0, abs=1e-5
        )

    def test_epsilon_safety(self):
        """Near-zero experimental value should not cause Inf."""
        err = relative_error(jnp.array(1.0), jnp.array(0.0), epsilon=1e-6)
        assert jnp.isfinite(err)

    def test_negative_values(self):
        # |(-3) - (-5)| / |(-5)| = 2/5 = 0.4
        err = relative_error(jnp.array(-3.0), jnp.array(-5.0))
        assert float(err) == pytest.approx(0.4, abs=1e-4)


# =========================================================================
# guarino_loss
# =========================================================================


def _make_features(**overrides) -> GuarinoFeatures:
    """Helper: build GuarinoFeatures with sensible defaults, then override."""
    defaults = dict(
        t_first_spike=jnp.array(10.0),
        t_second_spike=jnp.array(40.0),
        t_third_spike=jnp.array(70.0),
        t_last_spike=jnp.array(70.0),
        inv_first_isi=jnp.array(1.0 / 30.0),
        inv_last_isi=jnp.array(1.0 / 30.0),
        firing_frequency=jnp.array(30.0),
        v_stim_end=jnp.array(-65.0),
        has_first_spike=jnp.array(1.0),
        has_second_spike=jnp.array(1.0),
        has_third_spike=jnp.array(1.0),
        n_spikes=jnp.array(3.0),
    )
    defaults.update(overrides)
    return GuarinoFeatures(**defaults)


class TestGuarinoLoss:
    def test_identical_features_near_zero(self):
        f = _make_features()
        loss = guarino_loss(f, f)
        assert float(loss) < 0.01

    def test_missing_spikes_incur_penalty(self, guarino_features_3spikes):
        sim_no_spikes = _make_features(
            has_first_spike=jnp.array(0.0),
            has_second_spike=jnp.array(0.0),
            has_third_spike=jnp.array(0.0),
            n_spikes=jnp.array(0.0),
        )
        loss = guarino_loss(sim_no_spikes, guarino_features_3spikes)
        # 3 missing features × penalty 3.0 each = 9.0 contribution
        assert float(loss) > 8.0

    def test_penalty_scaling(self, guarino_features_3spikes):
        sim_no = _make_features(
            has_first_spike=jnp.array(0.0),
            has_second_spike=jnp.array(0.0),
            has_third_spike=jnp.array(0.0),
            n_spikes=jnp.array(0.0),
        )
        cfg_low = GuarinoLossConfig(missing_feature_penalty=1.0)
        cfg_high = GuarinoLossConfig(missing_feature_penalty=10.0)
        loss_low = float(guarino_loss(sim_no, guarino_features_3spikes, cfg_low))
        loss_high = float(guarino_loss(sim_no, guarino_features_3spikes, cfg_high))
        assert loss_high > loss_low

    def test_extra_sim_spikes_no_penalty(self, guarino_features_0spikes):
        """Simulation has spikes but data doesn't → no missing-feature penalty.

        The penalty term max(0, exp_has - sim_has) = max(0, 0-1) = 0 for all spike
        features, so no penalty is applied.  However, the frequency relative error
        |30 - 0|/(|0| + eps) is very large, so the loss is dominated by that term.
        We verify the penalty contribution specifically instead of the total.
        """
        sim = _make_features()
        # With penalty=0, we isolate pure relative errors (no penalty contribution)
        cfg_no_penalty = GuarinoLossConfig(missing_feature_penalty=0.0)
        cfg_with_penalty = GuarinoLossConfig(missing_feature_penalty=10.0)
        loss_no = float(guarino_loss(sim, guarino_features_0spikes, cfg_no_penalty))
        loss_with = float(guarino_loss(sim, guarino_features_0spikes, cfg_with_penalty))
        # Penalty should not contribute (exp has no spikes) → losses should be equal
        assert loss_no == pytest.approx(loss_with, rel=1e-5)

    def test_weight_zero_disables_feature(self, guarino_features_3spikes):
        sim = _make_features(t_first_spike=jnp.array(999.0))  # Very wrong first spike
        cfg_enabled = GuarinoLossConfig(weight_t_first=1.0)
        cfg_disabled = GuarinoLossConfig(weight_t_first=0.0)
        loss_enabled = float(guarino_loss(sim, guarino_features_3spikes, cfg_enabled))
        loss_disabled = float(guarino_loss(sim, guarino_features_3spikes, cfg_disabled))
        assert loss_disabled < loss_enabled

    def test_isi_gated_by_has_second_spike(self):
        """ISI error should be zero-weighted when < 2 spikes."""
        exp = _make_features(has_second_spike=jnp.array(0.0), n_spikes=jnp.array(1.0))
        sim = _make_features(
            inv_first_isi=jnp.array(999.0),  # Very wrong
            has_second_spike=jnp.array(0.0),
            n_spikes=jnp.array(1.0),
        )
        loss = guarino_loss(sim, exp)
        # ISI term should be gated to 0 because has_second_spike is 0
        # Compare against identical features with same validity
        loss_match = guarino_loss(exp, exp)
        assert float(loss) == pytest.approx(float(loss_match), abs=0.01)

    def test_nonnegative(self, guarino_features_3spikes):
        rng = np.random.default_rng(0)
        for _ in range(5):
            sim = _make_features(
                t_first_spike=jnp.array(float(rng.uniform(5, 15))),
                firing_frequency=jnp.array(float(rng.uniform(10, 50))),
            )
            assert float(guarino_loss(sim, guarino_features_3spikes)) >= 0.0

    def test_differentiable_wrt_sim_features(self, guarino_features_3spikes):
        """guarino_loss should be differentiable w.r.t. sim feature values."""

        def loss_fn(t1_val):
            sim = _make_features(t_first_spike=t1_val)
            return guarino_loss(sim, guarino_features_3spikes)

        g = jax.grad(loss_fn)(jnp.array(15.0))
        assert jnp.isfinite(g)


# We need numpy for the random generator above
import numpy as np  # noqa: E402
