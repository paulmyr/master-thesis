"""Tests for config.experiment — ExperimentConfig serialization."""

import json

import pytest

from ADoptEX.config.experiment import ExperimentConfig, GuarinoWeights

# =========================================================================
# ExperimentConfig
# =========================================================================


class TestExperimentConfig:
    def test_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.loss_type == "guarino"
        assert cfg.n_epochs == 500
        assert isinstance(cfg.guarino_weights, GuarinoWeights)

    def test_to_dict_serializable(self):
        cfg = ExperimentConfig(name="test")
        d = cfg.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_roundtrip_to_dict_from_dict(self):
        cfg = ExperimentConfig(
            name="roundtrip",
            loss_type="mse",
            n_epochs=100,
            learning_rate=0.05,
            guarino_weights=GuarinoWeights(weight_t_first=2.0),
        )
        d = cfg.to_dict()
        restored = ExperimentConfig.from_dict(d)
        assert restored.name == "roundtrip"
        assert restored.loss_type == "mse"
        assert restored.n_epochs == 100
        assert restored.learning_rate == 0.05
        assert restored.guarino_weights.weight_t_first == 2.0

    def test_save_load(self, tmp_path):
        cfg = ExperimentConfig(name="save_test", n_epochs=42)
        path = tmp_path / "cfg.json"
        cfg.save(path)
        loaded = ExperimentConfig.load(path)
        assert loaded.name == "save_test"
        assert loaded.n_epochs == 42

    def test_guarino_weights_reconstruction(self):
        w = GuarinoWeights(weight_firing_freq=5.0, weight_v_stim_end=0.5)
        d = w.to_dict()
        w2 = GuarinoWeights(**d)
        assert w2.weight_firing_freq == 5.0
        assert w2.weight_v_stim_end == 0.5
