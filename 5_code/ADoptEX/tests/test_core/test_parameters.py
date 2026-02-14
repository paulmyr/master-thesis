"""Tests for core.parameters — bounds, clipping, format conversion."""

import jax.numpy as jnp
import pytest

from ADoptEX.core.parameters import (DEFAULT_PARAMS, NAUD_PARAMETERS,
                                     PARAM_BOUNDS, ParamBounds, clip_params,
                                     convert_trainable_to_params,
                                     params_to_trainable_format)

# =========================================================================
# ParamBounds
# =========================================================================


class TestParamBounds:
    def test_clip_within(self):
        b = ParamBounds(0.0, 10.0)
        assert b.clip(5.0) == 5.0

    def test_clip_below(self):
        b = ParamBounds(0.0, 10.0)
        assert b.clip(-5.0) == 0.0

    def test_clip_above(self):
        b = ParamBounds(0.0, 10.0)
        assert b.clip(15.0) == 10.0

    def test_contains_within(self):
        b = ParamBounds(0.0, 10.0)
        assert 5.0 in b

    def test_contains_outside(self):
        b = ParamBounds(0.0, 10.0)
        assert 15.0 not in b

    def test_contains_boundary(self):
        b = ParamBounds(0.0, 10.0)
        assert 0.0 in b
        assert 10.0 in b


# =========================================================================
# clip_params
# =========================================================================


class TestClipParams:
    def test_within_bounds_unchanged(self, default_adex_params):
        clipped = clip_params(default_adex_params)
        for key in default_adex_params:
            bounds_key = key.replace("AdEx_", "")
            if bounds_key in PARAM_BOUNDS:
                assert clipped[key] == default_adex_params[key]

    def test_out_of_bounds_clipped(self):
        params = {"g_L": 999.0, "E_L": -200.0}
        clipped = clip_params(params)
        assert clipped["g_L"] == PARAM_BOUNDS["g_L"].max
        assert clipped["E_L"] == PARAM_BOUNDS["E_L"].min

    def test_adex_prefix(self):
        params = {"AdEx_g_L": 999.0}
        clipped = clip_params(params)
        assert clipped["AdEx_g_L"] == PARAM_BOUNDS["g_L"].max

    def test_unknown_key_passthrough(self):
        params = {"unknown_param": 42.0}
        clipped = clip_params(params)
        assert clipped["unknown_param"] == 42.0


# =========================================================================
# convert_trainable_to_params / params_to_trainable_format
# =========================================================================


class TestFormatConversion:
    def test_prefix_stripping(self, trainable_params_jaxley_format):
        result = convert_trainable_to_params(trainable_params_jaxley_format)
        assert "g_L" in result
        assert "AdEx_g_L" not in result

    def test_float_extraction(self, trainable_params_jaxley_format):
        result = convert_trainable_to_params(trainable_params_jaxley_format)
        assert isinstance(result["g_L"], float)
        assert result["g_L"] == pytest.approx(10.0)

    def test_prefix_adding(self):
        params = {"g_L": 10.0, "E_L": -70.0}
        trainable = params_to_trainable_format(params)
        assert len(trainable) == 2
        all_keys = [k for d in trainable for k in d]
        assert "AdEx_g_L" in all_keys
        assert "AdEx_E_L" in all_keys

    def test_roundtrip(self):
        original = {"g_L": 10.0, "E_L": -70.0, "a": 2.0}
        trainable = params_to_trainable_format(original)
        recovered = convert_trainable_to_params(trainable)
        for key in original:
            assert recovered[key] == pytest.approx(original[key])


# =========================================================================
# Constants sanity checks
# =========================================================================


class TestConstants:
    def test_naud_parameters_have_required_keys(self):
        required = {"C_m", "g_L", "E_L", "v_T", "delta_T", "v_reset", "tau_w", "a", "b"}
        for name, params in NAUD_PARAMETERS.items():
            assert required.issubset(params.keys()), f"Missing keys in {name}"

    def test_default_params_within_bounds(self):
        for key, value in DEFAULT_PARAMS.items():
            if key in PARAM_BOUNDS:
                assert value in PARAM_BOUNDS[key], f"{key}={value} out of bounds"

    def test_bounds_min_lt_max(self):
        for key, bounds in PARAM_BOUNDS.items():
            assert bounds.min < bounds.max, f"{key}: min >= max"
