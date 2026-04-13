"""Tests for benchmark runner.

Unit tests for generate_initial_params and metric helpers.
Integration tests (marked @slow) require Jaxley.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from ADoptEX.benchmark.runner import _compute_nm_loss_from_traces, generate_initial_params
from ADoptEX.benchmark.scenarios import SyntheticScenario
from ADoptEX.core.parameters import PARAM_BOUNDS


class TestGenerateInitialParams:
    @pytest.fixture
    def tonic_scenario(self):
        return SyntheticScenario(
            name="tonic_15pct_full",
            ground_truth_params={
                "C_m": 200.0,
                "g_L": 10.0,
                "E_L": -70.0,
                "v_T": -50.0,
                "delta_T": 2.0,
                "v_reset": -58.0,
                "v_threshold": 0.0,
                "tau_w": 30.0,
                "a": 2.0,
                "b": 0.0,
            },
            trainable_params=["C_m", "g_L", "E_L", "v_T", "delta_T", "v_reset", "a", "b", "tau_w"],
            perturbation=0.15,
            n_starts=10,
            seed=42,
        )

    def test_returns_dict(self, tonic_scenario):
        """Returns a dict with all trainable params."""
        params = generate_initial_params(tonic_scenario, 0)
        assert isinstance(params, dict)
        for p in tonic_scenario.trainable_params:
            assert p in params

    def test_within_bounds(self, tonic_scenario):
        """All generated params are within PARAM_BOUNDS."""
        for start_idx in range(10):
            params = generate_initial_params(tonic_scenario, start_idx)
            for p in tonic_scenario.trainable_params:
                bounds = PARAM_BOUNDS[p]
                assert bounds.min <= params[p] <= bounds.max, (
                    f"start={start_idx} {p}={params[p]} outside [{bounds.min}, {bounds.max}]"
                )

    def test_reproducible(self, tonic_scenario):
        """Same seed + start_index gives same params."""
        params1 = generate_initial_params(tonic_scenario, 5)
        params2 = generate_initial_params(tonic_scenario, 5)
        for p in tonic_scenario.trainable_params:
            assert params1[p] == params2[p]

    def test_different_starts_differ(self, tonic_scenario):
        """Different start indices give different params."""
        params0 = generate_initial_params(tonic_scenario, 0)
        params1 = generate_initial_params(tonic_scenario, 1)
        # At least some params should differ
        diffs = [abs(params0[p] - params1[p]) for p in tonic_scenario.trainable_params]
        assert max(diffs) > 0

    def test_perturbed_near_ground_truth(self, tonic_scenario):
        """Perturbed params are near ground truth (within perturbation range)."""
        params = generate_initial_params(tonic_scenario, 0)
        gt = tonic_scenario.ground_truth_params
        perturbation = tonic_scenario.perturbation

        for p in tonic_scenario.trainable_params:
            if gt[p] == 0.0:
                continue  # skip zero params (perturbation is relative)
            rel_error = abs(params[p] - gt[p]) / abs(gt[p])
            # Should be within perturbation (plus some tolerance for bounds clipping)
            assert rel_error <= perturbation + 0.05, (
                f"{p}: rel_error={rel_error:.3f} > perturbation={perturbation}"
            )

    def test_includes_non_trainable_params(self, tonic_scenario):
        """Non-trainable params from ground truth are included unchanged."""
        params = generate_initial_params(tonic_scenario, 0)
        gt = tonic_scenario.ground_truth_params
        for k, v in gt.items():
            if k not in tonic_scenario.trainable_params:
                assert params[k] == v


# ── _compute_nm_loss_from_traces: guarino objective ─────────────────────────


class TestComputeNmLossGuarino:
    """Unit tests for the guarino case in _compute_nm_loss_from_traces.

    Pure JAX/NumPy — no Jaxley required.
    """

    DT_MS = 0.025
    N = 2000  # 50 ms at 0.025 ms/step

    @pytest.fixture
    def no_spike_traces(self):
        voltage = jnp.full(self.N, -65.0)
        spikes = jnp.zeros(self.N)
        target_data = {
            "voltage": np.array(jnp.full(self.N, -65.0)),
            "stim_duration_ms": 50.0,
            "stim_end_index": self.N - 1,
        }
        return spikes, voltage, target_data

    @pytest.fixture
    def single_spike_traces(self):
        voltage = jnp.full(self.N, -65.0)
        spikes = jnp.zeros(self.N).at[500].set(1.0)
        target_data = {
            "voltage": np.array(jnp.full(self.N, -65.0)),
            "stim_duration_ms": 50.0,
            "stim_end_index": self.N - 1,
        }
        return spikes, voltage, target_data

    def test_returns_finite_scalar(self, single_spike_traces):
        sim_spikes, sim_voltage, target_data = single_spike_traces
        loss = _compute_nm_loss_from_traces(
            "guarino", sim_spikes, sim_voltage, target_data,
            loss_config=None, dt_ms=self.DT_MS,
        )
        assert np.isfinite(float(loss))
        assert float(loss) >= 0.0

    def test_no_spikes_returns_finite(self, no_spike_traces):
        sim_spikes, sim_voltage, target_data = no_spike_traces
        loss = _compute_nm_loss_from_traces(
            "guarino", sim_spikes, sim_voltage, target_data,
            loss_config=None, dt_ms=self.DT_MS,
        )
        assert np.isfinite(float(loss))

    def test_identical_traces_low_loss(self, single_spike_traces):
        """Sim identical to target should give near-zero loss."""
        sim_spikes, sim_voltage, target_data = single_spike_traces
        # Target has same spike as sim
        sim_voltage_with_peak = sim_voltage.at[500].set(35.0)
        target_data = dict(target_data)
        target_data["voltage"] = np.array(sim_voltage_with_peak)
        loss = _compute_nm_loss_from_traces(
            "guarino", sim_spikes, sim_voltage, target_data,
            loss_config=None, dt_ms=self.DT_MS,
        )
        assert float(loss) < 1.0
