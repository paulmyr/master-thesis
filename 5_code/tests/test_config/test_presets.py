"""Tests for config.presets — preset retrieval, deep copy, listing."""

import pytest

from config.experiment import ExperimentConfig
from config.presets import PRESETS, describe_presets, get_preset, list_presets

# =========================================================================
# get_preset
# =========================================================================


class TestGetPreset:
    def test_returns_experiment_config(self):
        cfg = get_preset("standard")
        assert isinstance(cfg, ExperimentConfig)

    def test_deep_copy(self):
        """Mutations should not affect the original preset."""
        cfg = get_preset("standard")
        cfg.n_epochs = 99999
        original = PRESETS["standard"]
        assert original.n_epochs != 99999

    def test_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent_preset")

    def test_error_message_helpful(self):
        """Error message should list available presets."""
        with pytest.raises(KeyError, match="standard"):
            get_preset("nonexistent_preset")


# =========================================================================
# list_presets
# =========================================================================


class TestListPresets:
    def test_returns_strings(self):
        names = list_presets()
        assert all(isinstance(n, str) for n in names)

    def test_contains_expected_names(self):
        names = list_presets()
        assert "standard" in names
        assert "quick_test" in names
        assert "mse_standard" in names

    def test_matches_presets_keys(self):
        assert set(list_presets()) == set(PRESETS.keys())


# =========================================================================
# describe_presets
# =========================================================================


class TestDescribePresets:
    def test_returns_string(self):
        desc = describe_presets()
        assert isinstance(desc, str)

    def test_includes_all_names(self):
        desc = describe_presets()
        for name in PRESETS:
            assert name in desc
