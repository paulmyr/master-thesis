"""Tests for benchmark scenario definitions."""

import pytest

from ADoptEX.benchmark.scenarios import (
    FULL_PARAMS,
    MEMBRANE_PARAMS,
    SyntheticScenario,
    get_synthetic_scenarios,
)
from ADoptEX.core.parameters import PARAM_BOUNDS


class TestSyntheticScenario:
    def test_validation_trainable_in_bounds(self):
        """All trainable params must have bounds defined."""
        with pytest.raises(ValueError, match="No bounds defined"):
            SyntheticScenario(
                name="bad",
                ground_truth_params={"nonexistent": 1.0},
                trainable_params=["nonexistent"],
            )

    def test_validation_ground_truth_in_bounds(self):
        """Ground truth must be within parameter bounds."""
        with pytest.raises(ValueError, match="outside bounds"):
            SyntheticScenario(
                name="bad",
                ground_truth_params={"g_L": 999.0},  # way out of bounds
                trainable_params=["g_L"],
            )

    def test_valid_scenario(self):
        """Valid scenario passes validation."""
        s = SyntheticScenario(
            name="test",
            ground_truth_params={"g_L": 10.0, "E_L": -70.0},
            trainable_params=["g_L", "E_L"],
        )
        assert s.name == "test"
        assert s.perturbation == 0.15
        assert s.n_starts == 10

    def test_defaults(self):
        """Default values are sensible."""
        s = SyntheticScenario(
            name="test",
            ground_truth_params={"g_L": 10.0},
            trainable_params=["g_L"],
        )
        assert s.dt_ms == 0.025
        assert s.seed == 42
        assert s.stim_current_pA == 500.0
        assert s.stim_duration_ms == 400.0
        assert s.stim_delay_ms == 50.0

    def test_t_max_computed(self):
        """t_max_ms = delay + duration + 50."""
        s = SyntheticScenario(
            name="test",
            ground_truth_params={"g_L": 10.0},
            trainable_params=["g_L"],
            stim_delay_ms=50.0,
            stim_duration_ms=400.0,
        )
        assert s.t_max_ms == 500.0


class TestGetSyntheticScenarios:
    def test_scenario_count(self):
        """5 patterns x 2 groups x 2 perturbations = 20 scenarios."""
        scenarios = get_synthetic_scenarios()
        assert len(scenarios) == 20

    def test_all_names_unique(self):
        """All scenario names are unique."""
        scenarios = get_synthetic_scenarios()
        names = [s.name for s in scenarios]
        assert len(names) == len(set(names))

    def test_name_format(self):
        """Names follow {pattern}_{pct}pct_{group} format."""
        scenarios = get_synthetic_scenarios()
        for s in scenarios:
            assert "pct" in s.name
            assert s.name.endswith("membrane") or s.name.endswith("full")

    def test_membrane_group_params(self):
        """Membrane-only scenarios have 6 trainable params."""
        scenarios = get_synthetic_scenarios()
        membrane = [s for s in scenarios if s.name.endswith("membrane")]
        for s in membrane:
            assert len(s.trainable_params) == 6
            assert s.trainable_params == MEMBRANE_PARAMS

    def test_full_group_params(self):
        """Full scenarios have 9 trainable params."""
        scenarios = get_synthetic_scenarios()
        full = [s for s in scenarios if s.name.endswith("full")]
        for s in full:
            assert len(s.trainable_params) == 9
            assert s.trainable_params == FULL_PARAMS

    def test_ground_truth_within_bounds(self):
        """All ground truth params are within PARAM_BOUNDS."""
        scenarios = get_synthetic_scenarios()
        for s in scenarios:
            for p in s.trainable_params:
                val = s.ground_truth_params[p]
                bounds = PARAM_BOUNDS[p]
                assert bounds.min <= val <= bounds.max, (
                    f"{s.name}: {p}={val} outside [{bounds.min}, {bounds.max}]"
                )

    def test_n_starts_override(self):
        """n_starts parameter is passed through."""
        scenarios = get_synthetic_scenarios(n_starts=5)
        for s in scenarios:
            assert s.n_starts == 5

    def test_perturbation_values(self):
        """Each pattern has 15% and 25% perturbation variants."""
        pcts = sorted({s.perturbation for s in get_synthetic_scenarios()})
        assert pcts == [0.15, 0.25]

    def test_all_have_stimulus(self):
        """All scenarios have stimulus current defined."""
        for s in get_synthetic_scenarios():
            assert s.stim_current_pA > 0
            assert s.stim_duration_ms > 0
            assert s.t_max_ms > s.stim_duration_ms
