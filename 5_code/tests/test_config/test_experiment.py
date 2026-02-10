"""Tests for config.experiment — serialization, results, directory creation."""

import json
import math

import pytest

from config.experiment import (ExperimentConfig, ExperimentResult,
                               GuarinoWeights, create_result_dir)

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


# =========================================================================
# ExperimentResult
# =========================================================================


class TestExperimentResult:
    def _make_result(self, **overrides):
        defaults = dict(
            config=ExperimentConfig(name="test"),
            timestamp="2025-01-01_120000",
            loss_history=[10.0, 5.0, 1.0],
            time_per_epoch=[0.5, 0.5, 0.5],
            final_params={"g_L": 12.0},
            gamma=0.8,
            spike_count_target=5,
            spike_count_model=4,
        )
        defaults.update(overrides)
        return ExperimentResult(**defaults)

    def test_final_loss(self):
        r = self._make_result()
        assert r.final_loss == 1.0

    def test_final_loss_nan_when_empty(self):
        r = self._make_result(loss_history=[])
        assert math.isnan(r.final_loss)

    def test_total_time(self):
        r = self._make_result()
        assert r.total_time == pytest.approx(1.5)

    def test_spike_count_diff(self):
        r = self._make_result()
        assert r.spike_count_diff == -1  # 4 - 5

    def test_roundtrip_save_load(self, tmp_path):
        r = self._make_result()
        path = tmp_path / "result.json"
        r.save(path)
        loaded = ExperimentResult.load(path)
        assert loaded.gamma == 0.8
        assert loaded.config.name == "test"
        assert loaded.loss_history == [10.0, 5.0, 1.0]

    def test_summary_string(self):
        r = self._make_result()
        s = r.summary()
        assert "test" in s
        assert "0.8" in s


# =========================================================================
# create_result_dir
# =========================================================================


class TestCreateResultDir:
    def test_directory_created(self, tmp_path):
        result_dir = create_result_dir(base_dir=tmp_path / "results")
        assert result_dir.exists()
        assert result_dir.is_dir()

    def test_timestamped_name(self, tmp_path):
        result_dir = create_result_dir(base_dir=tmp_path / "results")
        # Name should contain digits (timestamp pattern)
        assert any(c.isdigit() for c in result_dir.name)
