"""Tests for benchmark result serialization."""

import json
import math
from pathlib import Path

import pytest

from ADoptEX.benchmark.results import (
    RunResult,
    ScenarioResults,
    load_all_results,
    load_results,
    save_results,
)


# ── RunResult ────────────────────────────────────────────────────────────────


class TestRunResult:
    def test_to_dict_roundtrip(self):
        """to_dict -> from_dict preserves all fields."""
        run = RunResult(
            scenario_name="tonic_15pct_full",
            method_name="grad_vanrossum",
            start_index=3,
            seed=42,
            initial_params={"g_L": 10.5, "E_L": -69.0},
            final_params={"g_L": 10.0, "E_L": -70.0},
            ground_truth_params={"g_L": 10.0, "E_L": -70.0},
            final_loss=0.0123,
            best_loss=0.0100,
            gamma=0.75,
            n_spikes_model=5,
            n_spikes_target=5,
            spike_count_error=0,
            first_spike_error_ms=1.2,
            param_recovery={"g_L": {"error_abs": 0.0, "error_pct": 0.0}},
            wall_time_s=3.5,
            n_function_evals=500,
            time_per_eval_ms=7.0,
            loss_history=[1.0, 0.5, 0.1],
            grad_norms=[10.0, 5.0, 1.0],
        )

        d = run.to_dict()
        restored = RunResult.from_dict(d)

        assert restored.scenario_name == run.scenario_name
        assert restored.method_name == run.method_name
        assert restored.start_index == run.start_index
        assert restored.seed == run.seed
        assert restored.initial_params == run.initial_params
        assert restored.final_params == run.final_params
        assert restored.ground_truth_params == run.ground_truth_params
        assert restored.final_loss == pytest.approx(run.final_loss)
        assert restored.best_loss == pytest.approx(run.best_loss)
        assert restored.gamma == pytest.approx(run.gamma)
        assert restored.n_spikes_model == run.n_spikes_model
        assert restored.n_spikes_target == run.n_spikes_target
        assert restored.loss_history == run.loss_history
        assert restored.grad_norms == run.grad_norms
        assert restored.wall_time_s == pytest.approx(run.wall_time_s)

    def test_nan_serialization(self):
        """NaN values survive JSON roundtrip as None."""
        run = RunResult(
            scenario_name="test",
            method_name="test",
            start_index=0,
            seed=42,
            initial_params={},
            final_params={},
            gamma=float("nan"),
            first_spike_error_ms=float("nan"),
        )

        d = run.to_dict()
        assert d["gamma"] is None
        assert d["first_spike_error_ms"] is None

        # JSON roundtrip
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        restored = RunResult.from_dict(d2)
        assert restored.gamma is None
        assert restored.first_spike_error_ms is None

    def test_none_ground_truth(self):
        """Experimental scenarios have None ground_truth_params."""
        run = RunResult(
            scenario_name="exp_test",
            method_name="test",
            start_index=0,
            seed=42,
            initial_params={"g_L": 10.0},
            final_params={"g_L": 10.5},
            ground_truth_params=None,
        )

        d = run.to_dict()
        assert d["ground_truth_params"] is None

        restored = RunResult.from_dict(d)
        assert restored.ground_truth_params is None

    def test_timestamp_auto_set(self):
        """Timestamp is automatically set on creation."""
        run = RunResult(
            scenario_name="test",
            method_name="test",
            start_index=0,
            seed=42,
            initial_params={},
            final_params={},
        )
        assert run.timestamp is not None
        assert len(run.timestamp) > 0


# ── ScenarioResults ──────────────────────────────────────────────────────────


class TestScenarioResults:
    def _make_run(self, gamma: float, loss: float = 0.1) -> RunResult:
        return RunResult(
            scenario_name="test",
            method_name="test",
            start_index=0,
            seed=42,
            initial_params={},
            final_params={},
            gamma=gamma,
            final_loss=loss,
        )

    def test_best_run_by_gamma(self):
        """best_run returns highest gamma."""
        sr = ScenarioResults(
            scenario_name="test",
            method_name="test",
            runs=[
                self._make_run(0.3),
                self._make_run(0.8),
                self._make_run(0.5),
            ],
        )
        assert sr.best_run.gamma == pytest.approx(0.8)

    def test_mean_gamma(self):
        """mean_gamma computes correctly."""
        sr = ScenarioResults(
            scenario_name="test",
            method_name="test",
            runs=[
                self._make_run(0.3),
                self._make_run(0.6),
                self._make_run(0.9),
            ],
        )
        assert sr.mean_gamma == pytest.approx(0.6)

    def test_best_gamma(self):
        """best_gamma returns max."""
        sr = ScenarioResults(
            scenario_name="test",
            method_name="test",
            runs=[self._make_run(0.2), self._make_run(0.9)],
        )
        assert sr.best_gamma == pytest.approx(0.9)

    def test_empty_results(self):
        """Empty results return None/NaN gracefully."""
        sr = ScenarioResults(scenario_name="test", method_name="test")
        assert sr.best_run is None
        assert math.isnan(sr.mean_gamma)
        assert math.isnan(sr.best_gamma)
        assert sr.n_completed == 0

    def test_summary_dict(self):
        """summary_dict produces flat dict with expected keys."""
        sr = ScenarioResults(
            scenario_name="tonic_15pct_full",
            method_name="grad_vanrossum",
            runs=[self._make_run(0.5, 0.02), self._make_run(0.8, 0.01)],
        )
        d = sr.summary_dict()
        assert d["scenario"] == "tonic_15pct_full"
        assert d["method"] == "grad_vanrossum"
        assert d["n_runs"] == 2
        assert d["mean_gamma"] == pytest.approx(0.65)
        assert d["best_gamma"] == pytest.approx(0.8)
        assert d["best_loss"] == pytest.approx(0.01)

    def test_serialization_roundtrip(self):
        """Full to_dict/from_dict roundtrip."""
        sr = ScenarioResults(
            scenario_name="test",
            method_name="test",
            runs=[self._make_run(0.5), self._make_run(0.7)],
        )
        d = sr.to_dict()
        restored = ScenarioResults.from_dict(d)
        assert restored.scenario_name == sr.scenario_name
        assert restored.method_name == sr.method_name
        assert len(restored.runs) == 2
        assert restored.runs[0].gamma == pytest.approx(0.5)


# ── File I/O ─────────────────────────────────────────────────────────────────


class TestFileIO:
    def test_save_and_load(self, tmp_path):
        """save_results -> load_results roundtrip."""
        sr = ScenarioResults(
            scenario_name="tonic_15pct_full",
            method_name="grad_vanrossum",
            runs=[
                RunResult(
                    scenario_name="tonic_15pct_full",
                    method_name="grad_vanrossum",
                    start_index=0,
                    seed=42,
                    initial_params={"g_L": 10.5},
                    final_params={"g_L": 10.0},
                    gamma=0.75,
                    final_loss=0.01,
                )
            ],
        )

        path = save_results(sr, tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

        loaded = load_results(path)
        assert loaded.scenario_name == sr.scenario_name
        assert loaded.method_name == sr.method_name
        assert len(loaded.runs) == 1
        assert loaded.runs[0].gamma == pytest.approx(0.75)

    def test_load_all_results(self, tmp_path):
        """load_all_results loads all JSON files from directory."""
        for i in range(3):
            sr = ScenarioResults(
                scenario_name=f"scenario_{i}",
                method_name="test",
                runs=[
                    RunResult(
                        scenario_name=f"scenario_{i}",
                        method_name="test",
                        start_index=0,
                        seed=42,
                        initial_params={},
                        final_params={},
                    )
                ],
            )
            save_results(sr, tmp_path)

        results = load_all_results(tmp_path)
        assert len(results) == 3

    def test_load_all_results_empty_dir(self, tmp_path):
        """Empty directory returns empty list."""
        results = load_all_results(tmp_path)
        assert results == []

    def test_filename_format(self, tmp_path):
        """Filenames follow {scenario}__{method}.json pattern."""
        sr = ScenarioResults(
            scenario_name="tonic_15pct_full",
            method_name="grad_vanrossum",
        )
        path = save_results(sr, tmp_path)
        assert path.name == "tonic_15pct_full__grad_vanrossum.json"
