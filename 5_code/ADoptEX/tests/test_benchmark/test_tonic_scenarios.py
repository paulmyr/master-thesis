"""Tests for tonic benchmark scenario generation."""

import re

import pytest

from ADoptEX.benchmark.scenarios import (
    MEMBRANE_PARAMS,
    TONIC_PERTURBATIONS,
    get_tonic_scenarios,
)
from ADoptEX.benchmark.sampling import TONIC_FIXED_PARAMS
from ADoptEX.core.parameters import PARAM_BOUNDS


# Pre-built ground truths to avoid simulation in tests
def _make_ground_truths(n: int = 20) -> list[dict]:
    """Create mock ground truths with valid params."""
    gts = []
    for i in range(n):
        frac = i / max(n - 1, 1)
        params = dict(TONIC_FIXED_PARAMS)
        for name in MEMBRANE_PARAMS:
            bounds = PARAM_BOUNDS[name]
            params[name] = bounds.min + frac * (bounds.max - bounds.min)
        gts.append(
            {
                "params": params,
                "stim_current_pA": 300.0 + frac * 500.0,
                "n_spikes": 10,
            }
        )
    return gts


class TestGetTonicScenarios:
    def test_scenario_count(self):
        """20 GTs x 6 perturbations = 120 scenarios."""
        gts = _make_ground_truths(20)
        scenarios = get_tonic_scenarios(ground_truths=gts)
        assert len(scenarios) == 120

    def test_custom_perturbations(self):
        """Custom perturbation list changes scenario count."""
        gts = _make_ground_truths(5)
        scenarios = get_tonic_scenarios(
            ground_truths=gts, perturbations=[0.05, 0.10]
        )
        assert len(scenarios) == 10  # 5 GTs x 2 perturbations

    def test_all_names_unique(self):
        """All scenario names are unique."""
        gts = _make_ground_truths(20)
        scenarios = get_tonic_scenarios(ground_truths=gts)
        names = [s.name for s in scenarios]
        assert len(names) == len(set(names))

    def test_name_format(self):
        """Names match tonic_gtNN_Npct_membrane pattern."""
        gts = _make_ground_truths(20)
        scenarios = get_tonic_scenarios(ground_truths=gts)
        pattern = re.compile(r"^tonic_gt\d{2}_\d+pct_membrane$")
        for s in scenarios:
            assert pattern.match(s.name), f"Name '{s.name}' doesn't match pattern"

    def test_all_membrane_trainable(self):
        """All scenarios use MEMBRANE_PARAMS as trainable."""
        gts = _make_ground_truths(20)
        scenarios = get_tonic_scenarios(ground_truths=gts)
        for s in scenarios:
            assert s.trainable_params == MEMBRANE_PARAMS

    def test_perturbation_coverage(self):
        """Each GT appears with all perturbation levels."""
        gts = _make_ground_truths(5)
        scenarios = get_tonic_scenarios(ground_truths=gts)
        for i in range(5):
            gt_scenarios = [s for s in scenarios if f"gt{i:02d}" in s.name]
            pcts = sorted(s.perturbation for s in gt_scenarios)
            assert pcts == sorted(TONIC_PERTURBATIONS)

    def test_all_tonic_adaptation(self):
        """All ground truths have tonic adaptation params."""
        gts = _make_ground_truths(10)
        scenarios = get_tonic_scenarios(ground_truths=gts)
        for s in scenarios:
            for name, expected in TONIC_FIXED_PARAMS.items():
                assert s.ground_truth_params[name] == expected

    def test_stim_current_varies(self):
        """Not all scenarios use the same stimulus current."""
        gts = _make_ground_truths(20)
        scenarios = get_tonic_scenarios(ground_truths=gts)
        currents = {s.stim_current_pA for s in scenarios}
        assert len(currents) > 1

    def test_ground_truth_within_bounds(self):
        """All ground truth params are within PARAM_BOUNDS."""
        gts = _make_ground_truths(20)
        scenarios = get_tonic_scenarios(ground_truths=gts)
        for s in scenarios:
            for p in s.trainable_params:
                val = s.ground_truth_params[p]
                bounds = PARAM_BOUNDS[p]
                assert bounds.min <= val <= bounds.max

    def test_n_starts_passthrough(self):
        """n_starts parameter is passed through."""
        gts = _make_ground_truths(3)
        scenarios = get_tonic_scenarios(n_starts=5, ground_truths=gts)
        for s in scenarios:
            assert s.n_starts == 5

    def test_seed_passthrough(self):
        """seed parameter is passed through."""
        gts = _make_ground_truths(3)
        scenarios = get_tonic_scenarios(seed=999, ground_truths=gts)
        for s in scenarios:
            assert s.seed == 999
