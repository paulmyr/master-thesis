"""Tests for training.trainer — config validation, optimizer creation, training loop."""

import jax.numpy as jnp
import pytest

from ADoptEX.training.trainer import (TrainingConfig, TrainingResult,
                                      _build_param_transform,
                                      _clip_trainable_params,
                                      _create_optimizer, _format_params,
                                      _nudge_from_bounds,
                                      _param_key_to_bounds_key, train)

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def simple_params():
    """Minimal Jaxley-format trainable params (list-of-dicts)."""
    return [
        {"AdEx_g_L": jnp.array([5.0])},
        {"AdEx_E_L": jnp.array([3.0])},
    ]


@pytest.fixture
def quadratic_loss_fn():
    """Simple quadratic loss: sum of squared param values. Minimum at 0."""

    def loss_fn(params):
        return sum(jnp.sum(v**2) for d in params for v in d.values())

    return loss_fn


# =========================================================================
# TrainingConfig defaults
# =========================================================================


class TestTrainingConfigDefaults:
    def test_grad_clip_norm_default_none(self):
        config = TrainingConfig()
        assert config.grad_clip_norm is None

    def test_lr_schedule_default_none(self):
        config = TrainingConfig()
        assert config.lr_schedule is None

    def test_lr_decay_rate_default(self):
        config = TrainingConfig()
        assert config.lr_decay_rate == 0.01


# =========================================================================
# TrainingConfig validation
# =========================================================================


class TestTrainingConfigValidation:
    def test_grad_clip_norm_zero_raises(self):
        with pytest.raises(ValueError, match="grad_clip_norm must be positive"):
            TrainingConfig(grad_clip_norm=0.0)

    def test_grad_clip_norm_negative_raises(self):
        with pytest.raises(ValueError, match="grad_clip_norm must be positive"):
            TrainingConfig(grad_clip_norm=-1.0)

    def test_grad_clip_norm_positive_ok(self):
        config = TrainingConfig(grad_clip_norm=1.0)
        assert config.grad_clip_norm == 1.0

    def test_lr_decay_rate_zero_with_exponential_raises(self):
        with pytest.raises(ValueError, match="lr_decay_rate must be positive"):
            TrainingConfig(lr_schedule="exponential", lr_decay_rate=0.0)

    def test_lr_decay_rate_negative_with_exponential_raises(self):
        with pytest.raises(ValueError, match="lr_decay_rate must be positive"):
            TrainingConfig(lr_schedule="exponential", lr_decay_rate=-0.5)

    def test_lr_decay_rate_zero_without_exponential_ok(self):
        # lr_decay_rate is only validated for exponential schedule
        config = TrainingConfig(lr_schedule="cosine", lr_decay_rate=0.0)
        assert config.lr_decay_rate == 0.0


# =========================================================================
# _create_optimizer
# =========================================================================


class TestCreateOptimizer:
    def test_default_config_creates_optimizer(self, simple_params):
        config = TrainingConfig()
        opt = _create_optimizer(config)
        state = opt.init(simple_params)
        assert state is not None

    def test_grad_clip_creates_optimizer(self, simple_params):
        config = TrainingConfig(grad_clip_norm=1.0)
        opt = _create_optimizer(config)
        state = opt.init(simple_params)
        assert state is not None

    def test_cosine_schedule_creates_optimizer(self, simple_params):
        config = TrainingConfig(lr_schedule="cosine", n_epochs=100)
        opt = _create_optimizer(config)
        state = opt.init(simple_params)
        assert state is not None

    def test_exponential_schedule_creates_optimizer(self, simple_params):
        config = TrainingConfig(lr_schedule="exponential", n_epochs=100)
        opt = _create_optimizer(config)
        state = opt.init(simple_params)
        assert state is not None

    def test_clip_and_schedule_together(self, simple_params):
        config = TrainingConfig(grad_clip_norm=1.0, lr_schedule="cosine", n_epochs=100)
        opt = _create_optimizer(config)
        state = opt.init(simple_params)
        assert state is not None

    def test_unknown_optimizer_raises(self):
        config = TrainingConfig()
        # Bypass __post_init__ by setting after construction
        object.__setattr__(config, "optimizer", "nonexistent")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            _create_optimizer(config)

    def test_sgd_optimizer(self, simple_params):
        config = TrainingConfig(optimizer="sgd")
        opt = _create_optimizer(config)
        state = opt.init(simple_params)
        assert state is not None

    def test_constant_schedule_uses_scalar_lr(self, simple_params):
        config = TrainingConfig(lr_schedule="constant")
        opt = _create_optimizer(config)
        state = opt.init(simple_params)
        assert state is not None


# =========================================================================
# Gradient clipping behavior
# =========================================================================


class TestGradientClipping:
    def test_clipped_training_converges(self, simple_params, quadratic_loss_fn):
        config = TrainingConfig(
            n_epochs=50,
            learning_rate=0.01,
            grad_clip_norm=1.0,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert result.loss_history[-1] < result.loss_history[0]

    def test_large_clip_matches_no_clip(self, simple_params, quadratic_loss_fn):
        """A very large clip value should produce similar behavior to no clip."""
        config_no_clip = TrainingConfig(
            n_epochs=20,
            learning_rate=0.01,
            clip_to_bounds=False,
            verbose=False,
        )
        config_large_clip = TrainingConfig(
            n_epochs=20,
            learning_rate=0.01,
            grad_clip_norm=1e6,
            clip_to_bounds=False,
            verbose=False,
        )
        result_no_clip = train(quadratic_loss_fn, simple_params, config_no_clip)
        result_large_clip = train(quadratic_loss_fn, simple_params, config_large_clip)
        # Final losses should be very close
        assert abs(result_no_clip.final_loss - result_large_clip.final_loss) < 1e-4

    def test_small_clip_slows_convergence(self, simple_params, quadratic_loss_fn):
        """A very small clip norm should slow convergence vs no clip (SGD)."""
        # Use SGD so gradient magnitude directly affects step size
        config_no_clip = TrainingConfig(
            optimizer="sgd",
            n_epochs=50,
            learning_rate=0.01,
            clip_to_bounds=False,
            verbose=False,
        )
        config_small_clip = TrainingConfig(
            optimizer="sgd",
            n_epochs=50,
            learning_rate=0.01,
            grad_clip_norm=0.001,
            clip_to_bounds=False,
            verbose=False,
        )
        result_no_clip = train(quadratic_loss_fn, simple_params, config_no_clip)
        result_small_clip = train(quadratic_loss_fn, simple_params, config_small_clip)
        # Small clip should converge slower (higher final loss)
        assert result_small_clip.final_loss > result_no_clip.final_loss


# =========================================================================
# LR schedule behavior
# =========================================================================


class TestLRSchedule:
    def test_cosine_schedule_converges(self, simple_params, quadratic_loss_fn):
        config = TrainingConfig(
            n_epochs=50,
            learning_rate=0.01,
            lr_schedule="cosine",
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert result.loss_history[-1] < result.loss_history[0]

    def test_exponential_schedule_converges(self, simple_params, quadratic_loss_fn):
        config = TrainingConfig(
            n_epochs=50,
            learning_rate=0.01,
            lr_schedule="exponential",
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert result.loss_history[-1] < result.loss_history[0]

    def test_schedule_with_sgd(self, simple_params, quadratic_loss_fn):
        config = TrainingConfig(
            optimizer="sgd",
            n_epochs=50,
            learning_rate=0.01,
            lr_schedule="cosine",
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert result.loss_history[-1] < result.loss_history[0]

    def test_lr_history_populated_with_schedule(self, simple_params, quadratic_loss_fn):
        config = TrainingConfig(
            n_epochs=20,
            learning_rate=0.01,
            lr_schedule="cosine",
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert len(result.lr_history) == 20
        # Cosine schedule: first LR should be close to init, last should be lower
        assert result.lr_history[0] > result.lr_history[-1]

    def test_lr_history_empty_without_schedule(self, simple_params, quadratic_loss_fn):
        config = TrainingConfig(
            n_epochs=10,
            learning_rate=0.01,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert result.lr_history == []


# =========================================================================
# Integration tests
# =========================================================================


class TestTrainIntegration:
    def test_default_config_backward_compat(self, simple_params, quadratic_loss_fn):
        """Default config (no clip, no schedule) still works."""
        config = TrainingConfig(
            n_epochs=10,
            learning_rate=0.01,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert len(result.loss_history) == 10
        assert len(result.grad_norms) == 10
        assert result.lr_history == []

    def test_clip_and_schedule_together(self, simple_params, quadratic_loss_fn):
        config = TrainingConfig(
            n_epochs=30,
            learning_rate=0.01,
            grad_clip_norm=1.0,
            lr_schedule="exponential",
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert result.loss_history[-1] < result.loss_history[0]
        assert len(result.lr_history) == 30
        assert len(result.grad_norms) == 30

    def test_grad_norms_still_populated(self, simple_params, quadratic_loss_fn):
        """Regression: grad_norms should still be populated with new features."""
        config = TrainingConfig(
            n_epochs=10,
            learning_rate=0.01,
            grad_clip_norm=1.0,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert len(result.grad_norms) == 10
        assert all(g >= 0 for g in result.grad_norms)

    def test_clipped_grad_norms_populated_with_clipping(
        self, simple_params, quadratic_loss_fn
    ):
        """clipped_grad_norms populated when grad_clip_norm is set."""
        config = TrainingConfig(
            n_epochs=10,
            learning_rate=0.01,
            grad_clip_norm=1.0,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert len(result.clipped_grad_norms) == 10
        assert all(g >= 0 for g in result.clipped_grad_norms)

    def test_clipped_grad_norms_empty_without_clipping(
        self, simple_params, quadratic_loss_fn
    ):
        """clipped_grad_norms empty when no clipping configured."""
        config = TrainingConfig(
            n_epochs=10,
            learning_rate=0.01,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert result.clipped_grad_norms == []

    def test_clipped_norms_bounded_by_raw(self, simple_params, quadratic_loss_fn):
        """Clipped norms should never exceed raw norms."""
        config = TrainingConfig(
            n_epochs=20,
            learning_rate=0.01,
            grad_clip_norm=0.5,
            optimizer="sgd",
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        for raw, clipped in zip(result.grad_norms, result.clipped_grad_norms):
            assert clipped <= raw + 1e-6


# =========================================================================
# Polyak gradient normalization
# =========================================================================


class TestPolyakOptimizer:
    def test_polyak_converges(self, simple_params, quadratic_loss_fn):
        """Polyak optimizer should decrease a quadratic loss."""
        config = TrainingConfig(
            optimizer="polyak",
            n_epochs=50,
            learning_rate=0.01,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert result.loss_history[-1] < result.loss_history[0]

    def test_polyak_lr_history_tracks_loss_scaling(
        self, simple_params, quadratic_loss_fn
    ):
        """lr_history should be populated and decrease as loss decreases."""
        config = TrainingConfig(
            optimizer="polyak",
            n_epochs=50,
            learning_rate=0.01,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert len(result.lr_history) == 50
        # Effective LR should decrease as loss decreases (alpha=1 default)
        assert result.lr_history[-1] < result.lr_history[0]

    def test_polyak_beta_zero_no_normalization(self, simple_params, quadratic_loss_fn):
        """With beta=0, grad norm is not divided out (loss-scaled SGD)."""
        # beta must be positive per validation, but we test beta close to 0
        # by comparing beta=1 (normalized) vs beta near 0 (not normalized)
        # Instead, bypass validation to test beta=0 behavior
        config = TrainingConfig(
            optimizer="polyak",
            n_epochs=20,
            learning_rate=0.001,
            clip_to_bounds=False,
            verbose=False,
        )
        # Manually set beta=0 after construction to bypass validation
        object.__setattr__(config, "polyak_beta", 0.0)
        result = train(quadratic_loss_fn, simple_params, config)
        # Should still converge (it's loss-scaled SGD)
        assert result.loss_history[-1] < result.loss_history[0]

    def test_polyak_alpha_zero_ignores_loss(self, simple_params, quadratic_loss_fn):
        """With alpha=0, effective LR should be constant (loss^0 = 1)."""
        config = TrainingConfig(
            optimizer="polyak",
            n_epochs=20,
            learning_rate=0.01,
            polyak_alpha=0.0,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        # All effective LRs should equal the base LR
        for lr in result.lr_history:
            assert abs(lr - config.learning_rate) < 1e-10

    def test_polyak_with_lr_schedule_raises(self):
        """Combining polyak + lr_schedule should raise ValueError."""
        with pytest.raises(ValueError, match="lr_schedule is incompatible with polyak"):
            TrainingConfig(optimizer="polyak", lr_schedule="cosine")

    def test_polyak_with_grad_clip_raises(self):
        """Combining polyak + grad_clip_norm should raise ValueError."""
        with pytest.raises(
            ValueError, match="grad_clip_norm is incompatible with polyak"
        ):
            TrainingConfig(optimizer="polyak", grad_clip_norm=1.0)

    def test_polyak_grad_norms_populated(self, simple_params, quadratic_loss_fn):
        """grad_norms should be populated, clipped_grad_norms should be empty."""
        config = TrainingConfig(
            optimizer="polyak",
            n_epochs=10,
            learning_rate=0.01,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert len(result.grad_norms) == 10
        assert all(g >= 0 for g in result.grad_norms)
        assert result.clipped_grad_norms == []

    def test_polyak_default_alpha_beta(self):
        """Default alpha=1.0 and beta=1.0."""
        config = TrainingConfig(optimizer="polyak")
        assert config.polyak_alpha == 1.0
        assert config.polyak_beta == 1.0


# =========================================================================
# Capacitance / C_m mapping
# =========================================================================


class TestCapacitanceMapping:
    @pytest.fixture
    def params_with_capacitance(self):
        """Trainable params including capacitance (as Jaxley would produce)."""
        return [
            {"capacitance": jnp.array([200.0])},
            {"AdEx_g_L": jnp.array([10.0])},
        ]

    def test_clip_trainable_params_handles_capacitance(self):
        """_clip_trainable_params should look up C_m bounds for capacitance key."""
        from ADoptEX.core.parameters import PARAM_BOUNDS

        params = [{"capacitance": jnp.array([999.0])}]
        clipped = _clip_trainable_params(params, PARAM_BOUNDS)
        assert clipped[0]["capacitance"].item() == PARAM_BOUNDS["C_m"].max

    def test_format_params_maps_capacitance_to_c_m(self):
        """_format_params should display 'C_m' not 'capacitance'."""
        params = [{"capacitance": jnp.array([200.0])}]
        formatted = _format_params(params)
        assert "C_m=" in formatted
        assert "capacitance" not in formatted

    def test_get_params_dict_maps_capacitance_to_c_m(self, params_with_capacitance):
        """TrainingResult.get_params_dict() should map capacitance -> C_m."""
        result = TrainingResult(
            trainable_params=params_with_capacitance,
            loss_history=[1.0],
            time_per_epoch=[0.1],
            config=TrainingConfig(),
            initial_params={},
        )
        params_dict = result.get_params_dict()
        assert "C_m" in params_dict
        assert "capacitance" not in params_dict
        assert params_dict["C_m"] == pytest.approx(200.0)

    def test_trainable_params_config_accepts_c_m(self):
        """TrainingConfig should accept C_m in trainable_params list."""
        config = TrainingConfig(trainable_params=["C_m", "g_L"])
        assert "C_m" in config.trainable_params


# =========================================================================
# _param_key_to_bounds_key
# =========================================================================


class TestParamKeyToBoundsKey:
    def test_strips_adex_prefix(self):
        assert _param_key_to_bounds_key("AdEx_g_L") == "g_L"

    def test_maps_capacitance_to_c_m(self):
        assert _param_key_to_bounds_key("capacitance") == "C_m"

    def test_adex_capacitance_maps_to_c_m(self):
        assert _param_key_to_bounds_key("AdEx_capacitance") == "C_m"

    def test_no_prefix_passthrough(self):
        assert _param_key_to_bounds_key("tau_w") == "tau_w"


# =========================================================================
# _nudge_from_bounds
# =========================================================================


class TestNudgeFromBounds:
    def test_nudges_at_lower_bound(self):
        from ADoptEX.core.parameters import PARAM_BOUNDS

        params = [{"AdEx_g_L": jnp.array([PARAM_BOUNDS["g_L"].min])}]
        nudged = _nudge_from_bounds(params, PARAM_BOUNDS)
        val = nudged[0]["AdEx_g_L"].item()
        assert val > PARAM_BOUNDS["g_L"].min
        assert val < PARAM_BOUNDS["g_L"].min + 0.01 * (
            PARAM_BOUNDS["g_L"].max - PARAM_BOUNDS["g_L"].min
        )

    def test_nudges_at_upper_bound(self):
        from ADoptEX.core.parameters import PARAM_BOUNDS

        params = [{"AdEx_g_L": jnp.array([PARAM_BOUNDS["g_L"].max])}]
        nudged = _nudge_from_bounds(params, PARAM_BOUNDS)
        val = nudged[0]["AdEx_g_L"].item()
        assert val < PARAM_BOUNDS["g_L"].max

    def test_noop_for_interior_values(self):
        from ADoptEX.core.parameters import PARAM_BOUNDS

        mid = (PARAM_BOUNDS["g_L"].min + PARAM_BOUNDS["g_L"].max) / 2
        params = [{"AdEx_g_L": jnp.array([mid])}]
        nudged = _nudge_from_bounds(params, PARAM_BOUNDS)
        assert nudged[0]["AdEx_g_L"].item() == pytest.approx(mid)

    def test_handles_capacitance_key(self):
        from ADoptEX.core.parameters import PARAM_BOUNDS

        params = [{"capacitance": jnp.array([PARAM_BOUNDS["C_m"].min])}]
        nudged = _nudge_from_bounds(params, PARAM_BOUNDS)
        val = nudged[0]["capacitance"].item()
        assert val > PARAM_BOUNDS["C_m"].min


# =========================================================================
# _build_param_transform
# =========================================================================


class TestBuildParamTransform:
    def test_builds_successfully(self):
        from ADoptEX.core.parameters import PARAM_BOUNDS

        params = [{"AdEx_g_L": jnp.array([10.0])}, {"AdEx_E_L": jnp.array([-65.0])}]
        transform = _build_param_transform(params, PARAM_BOUNDS)
        assert transform is not None

    def test_roundtrip_forward_inverse(self):
        from ADoptEX.core.parameters import PARAM_BOUNDS

        params = [{"AdEx_g_L": jnp.array([10.0])}, {"AdEx_E_L": jnp.array([-65.0])}]
        transform = _build_param_transform(params, PARAM_BOUNDS)

        unconstrained = transform.inverse(params)
        reconstructed = transform.forward(unconstrained)
        for orig_d, recon_d in zip(params, reconstructed):
            for key in orig_d:
                assert recon_d[key].item() == pytest.approx(
                    orig_d[key].item(), abs=1e-5
                )

    def test_forward_stays_in_bounds(self):
        from ADoptEX.core.parameters import PARAM_BOUNDS

        params = [{"AdEx_g_L": jnp.array([10.0])}, {"AdEx_a": jnp.array([2.0])}]
        transform = _build_param_transform(params, PARAM_BOUNDS)

        # Extreme unconstrained values should still map inside bounds
        extreme = [{"AdEx_g_L": jnp.array([100.0])}, {"AdEx_a": jnp.array([-100.0])}]
        constrained = transform.forward(extreme)
        g_L = constrained[0]["AdEx_g_L"].item()
        a = constrained[1]["AdEx_a"].item()
        assert PARAM_BOUNDS["g_L"].min <= g_L <= PARAM_BOUNDS["g_L"].max
        assert PARAM_BOUNDS["a"].min <= a <= PARAM_BOUNDS["a"].max

    def test_raises_for_unknown_param(self):
        from ADoptEX.core.parameters import PARAM_BOUNDS

        params = [{"AdEx_unknown_xyz": jnp.array([1.0])}]
        with pytest.raises(ValueError, match="No bounds for parameter"):
            _build_param_transform(params, PARAM_BOUNDS)


# =========================================================================
# Sigmoid reparameterization (integration tests)
# =========================================================================


class TestSigmoidReparameterization:
    def test_config_default_false(self):
        config = TrainingConfig()
        assert config.use_param_transform is False

    def test_convergence_with_transform(self, simple_params, quadratic_loss_fn):
        """Training with sigmoid transform should still converge on a quadratic."""
        config = TrainingConfig(
            n_epochs=100,
            learning_rate=0.5,
            use_param_transform=True,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert result.loss_history[-1] < result.loss_history[0]

    def test_overrides_clip_to_bounds(self, simple_params, quadratic_loss_fn):
        """When use_param_transform=True, clip_to_bounds should be skipped."""
        config = TrainingConfig(
            n_epochs=20,
            learning_rate=0.1,
            use_param_transform=True,
            clip_to_bounds=True,
            verbose=False,
        )
        # Should not raise and should still converge
        result = train(quadratic_loss_fn, simple_params, config)
        assert len(result.loss_history) == 20

    def test_final_params_in_bounds(self):
        """Output params must be within PARAM_BOUNDS."""
        from ADoptEX.core.parameters import PARAM_BOUNDS

        params = [
            {"AdEx_g_L": jnp.array([10.0])},
            {"AdEx_E_L": jnp.array([-65.0])},
        ]

        def loss_fn(p):
            return sum(jnp.sum(v**2) for d in p for v in d.values())

        config = TrainingConfig(
            n_epochs=50,
            learning_rate=1.0,
            use_param_transform=True,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(loss_fn, params, config)
        for d in result.trainable_params:
            for name, value in d.items():
                bounds_key = _param_key_to_bounds_key(name)
                bound = PARAM_BOUNDS[bounds_key]
                val = float(value.flatten()[0])
                assert (
                    bound.min <= val <= bound.max
                ), f"{name}={val} outside [{bound.min}, {bound.max}]"

    def test_return_best_with_transform(self, simple_params, quadratic_loss_fn):
        """return_best should work correctly with sigmoid transform."""
        config = TrainingConfig(
            n_epochs=50,
            learning_rate=0.5,
            use_param_transform=True,
            clip_to_bounds=False,
            return_best=True,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert result.best_epoch >= 0
        # Best loss should be <= final loss
        assert result.best_loss <= result.final_loss + 1e-6

    def test_polyak_with_transform(self, simple_params, quadratic_loss_fn):
        """Polyak + sigmoid transform should be compatible and converge."""
        config = TrainingConfig(
            optimizer="polyak",
            n_epochs=50,
            learning_rate=0.1,
            use_param_transform=True,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert result.loss_history[-1] < result.loss_history[0]
        assert len(result.lr_history) == 50

    def test_grad_norms_populated(self, simple_params, quadratic_loss_fn):
        """Gradient norms should still be tracked with transform."""
        config = TrainingConfig(
            n_epochs=10,
            learning_rate=0.1,
            use_param_transform=True,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(quadratic_loss_fn, simple_params, config)
        assert len(result.grad_norms) == 10
        assert all(g >= 0 for g in result.grad_norms)

    def test_params_at_exact_bounds_no_nan(self):
        """Parameters sitting at exact bounds should not produce NaN."""
        from ADoptEX.core.parameters import PARAM_BOUNDS

        params = [
            {"AdEx_g_L": jnp.array([PARAM_BOUNDS["g_L"].min])},
            {"AdEx_E_L": jnp.array([PARAM_BOUNDS["E_L"].max])},
        ]

        def loss_fn(p):
            return sum(jnp.sum(v**2) for d in p for v in d.values())

        config = TrainingConfig(
            n_epochs=10,
            learning_rate=0.1,
            use_param_transform=True,
            clip_to_bounds=False,
            verbose=False,
        )
        result = train(loss_fn, params, config)
        assert all(not jnp.isnan(l) for l in result.loss_history)
        # Final params should not be NaN
        for d in result.trainable_params:
            for name, value in d.items():
                assert not jnp.isnan(value).any(), f"{name} is NaN"
