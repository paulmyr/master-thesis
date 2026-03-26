"""Tests for ground truth parameter sampling."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ADoptEX.benchmark.sampling import (
    TONIC_FIXED_PARAMS,
    TONIC_TRAINABLE_PARAMS,
    TONIC_STIM_RANGE_PA,
    sample_tonic_ground_truths,
)
from ADoptEX.core.parameters import PARAM_BOUNDS


def _make_sim_result(n_spikes: int):
    """Create a mock SimulationResult with the given spike count."""
    mock = MagicMock()
    mock.n_spikes = n_spikes
    mock.spike_times = np.arange(n_spikes) * 10.0
    return mock


class TestSampleTonicGroundTruths:
    @patch("ADoptEX.benchmark.sampling.simulate_jaxley")
    def test_returns_correct_count(self, mock_sim):
        """Returns exactly n ground truths."""
        mock_sim.return_value = _make_sim_result(10)
        gts = sample_tonic_ground_truths(n=5, seed=42)
        assert len(gts) == 5

    @patch("ADoptEX.benchmark.sampling.simulate_jaxley")
    def test_deterministic(self, mock_sim):
        """Same seed produces identical results."""
        mock_sim.return_value = _make_sim_result(10)
        gts1 = sample_tonic_ground_truths(n=5, seed=123)
        gts2 = sample_tonic_ground_truths(n=5, seed=123)
        for a, b in zip(gts1, gts2):
            assert a["params"] == b["params"]
            assert a["stim_current_pA"] == b["stim_current_pA"]

    @patch("ADoptEX.benchmark.sampling.simulate_jaxley")
    def test_different_seeds_differ(self, mock_sim):
        """Different seeds produce different results."""
        mock_sim.return_value = _make_sim_result(10)
        gts1 = sample_tonic_ground_truths(n=5, seed=1)
        gts2 = sample_tonic_ground_truths(n=5, seed=2)
        params1 = [gt["params"]["g_L"] for gt in gts1]
        params2 = [gt["params"]["g_L"] for gt in gts2]
        assert params1 != params2

    @patch("ADoptEX.benchmark.sampling.simulate_jaxley")
    def test_membrane_params_within_bounds(self, mock_sim):
        """All sampled membrane params are within PARAM_BOUNDS."""
        mock_sim.return_value = _make_sim_result(10)
        gts = sample_tonic_ground_truths(n=20, seed=42)
        for gt in gts:
            for name in TONIC_TRAINABLE_PARAMS:
                val = gt["params"][name]
                bounds = PARAM_BOUNDS[name]
                assert bounds.min <= val <= bounds.max, (
                    f"{name}={val} outside [{bounds.min}, {bounds.max}]"
                )

    @patch("ADoptEX.benchmark.sampling.simulate_jaxley")
    def test_fixed_adaptation_params(self, mock_sim):
        """Adaptation params match TONIC_FIXED_PARAMS."""
        mock_sim.return_value = _make_sim_result(10)
        gts = sample_tonic_ground_truths(n=10, seed=42)
        for gt in gts:
            for name, expected in TONIC_FIXED_PARAMS.items():
                assert gt["params"][name] == expected, (
                    f"Expected {name}={expected}, got {gt['params'][name]}"
                )

    @patch("ADoptEX.benchmark.sampling.simulate_jaxley")
    def test_stim_current_in_range(self, mock_sim):
        """Stimulus current is within TONIC_STIM_RANGE_PA."""
        mock_sim.return_value = _make_sim_result(10)
        gts = sample_tonic_ground_truths(n=20, seed=42)
        lo, hi = TONIC_STIM_RANGE_PA
        for gt in gts:
            assert lo <= gt["stim_current_pA"] <= hi

    @patch("ADoptEX.benchmark.sampling.simulate_jaxley")
    def test_n_spikes_recorded(self, mock_sim):
        """Each ground truth records its spike count."""
        mock_sim.return_value = _make_sim_result(7)
        gts = sample_tonic_ground_truths(n=3, seed=42)
        for gt in gts:
            assert gt["n_spikes"] == 7

    @patch("ADoptEX.benchmark.sampling.simulate_jaxley")
    def test_rejection_filters_non_spiking(self, mock_sim):
        """Candidates with too few spikes are rejected."""
        # Alternate: 0 spikes (rejected), 5 spikes (accepted)
        call_count = 0

        def alternating_sim(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:
                return _make_sim_result(0)  # rejected
            return _make_sim_result(5)  # accepted

        mock_sim.side_effect = alternating_sim
        gts = sample_tonic_ground_truths(n=3, seed=42)
        assert len(gts) == 3
        for gt in gts:
            assert gt["n_spikes"] >= 3

    @patch("ADoptEX.benchmark.sampling.simulate_jaxley")
    def test_raises_if_too_few_accepted(self, mock_sim):
        """Raises RuntimeError if not enough candidates pass."""
        mock_sim.return_value = _make_sim_result(0)  # all rejected
        with pytest.raises(RuntimeError, match="Only 0/5"):
            sample_tonic_ground_truths(n=5, seed=42)

    @patch("ADoptEX.benchmark.sampling.simulate_jaxley")
    def test_space_filling(self, mock_sim):
        """Parameters cover diverse regions (not clustered)."""
        mock_sim.return_value = _make_sim_result(10)
        gts = sample_tonic_ground_truths(n=20, seed=42)

        # For each param, check that the sampled range covers >40% of bounds
        for name in TONIC_TRAINABLE_PARAMS:
            vals = [gt["params"][name] for gt in gts]
            bounds = PARAM_BOUNDS[name]
            bounds_range = bounds.max - bounds.min
            sampled_range = max(vals) - min(vals)
            coverage = sampled_range / bounds_range
            assert coverage > 0.4, (
                f"{name}: coverage={coverage:.2f}, "
                f"range=[{min(vals):.2f}, {max(vals):.2f}], "
                f"bounds=[{bounds.min}, {bounds.max}]"
            )

    @patch("ADoptEX.benchmark.sampling.simulate_jaxley")
    def test_dict_structure(self, mock_sim):
        """Each ground truth has the expected keys."""
        mock_sim.return_value = _make_sim_result(10)
        gts = sample_tonic_ground_truths(n=3, seed=42)
        for gt in gts:
            assert "params" in gt
            assert "stim_current_pA" in gt
            assert "n_spikes" in gt
            assert isinstance(gt["params"], dict)
            assert isinstance(gt["stim_current_pA"], float)
            assert isinstance(gt["n_spikes"], int)
