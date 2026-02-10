"""Shared fixtures for the test suite.

Provides synthetic data generators so no test needs real data files or Jaxley.
"""

import jax.numpy as jnp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Basic parameters
# ---------------------------------------------------------------------------


@pytest.fixture
def dt_ms():
    """Standard simulation time step (ms)."""
    return 0.025


# ---------------------------------------------------------------------------
# Voltage traces
# ---------------------------------------------------------------------------


@pytest.fixture
def voltage_with_spikes(dt_ms):
    """12 000-sample voltage trace with 3 spikes at t=50, 150, 250 ms.

    Background is a -65 mV baseline.  Spikes are short 5-sample pulses to +20 mV.
    """
    n_samples = 12_000
    trace = np.full(n_samples, -65.0)
    for t_ms in [50.0, 150.0, 250.0]:
        idx = int(t_ms / dt_ms)
        trace[idx : idx + 5] = 20.0
    return trace


@pytest.fixture
def voltage_no_spikes(dt_ms):
    """Sub-threshold sinusoidal oscillation (no spikes)."""
    n_samples = 12_000
    t = np.arange(n_samples) * dt_ms
    return -65.0 + 5.0 * np.sin(2 * np.pi * t / 100.0)


# ---------------------------------------------------------------------------
# Spike indicator traces  (dt = 0.1 ms, length 1000 → 100 ms)
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_spike_trace_3():
    """Length-1000 binary spike trace with spikes at indices 100, 400, 700 (dt=0.1 ms)."""
    trace = np.zeros(1000)
    trace[100] = 1.0
    trace[400] = 1.0
    trace[700] = 1.0
    return jnp.array(trace)


@pytest.fixture
def soft_spike_trace_3():
    """Gaussian bumps centred at indices 100, 400, 700 (sum ≈ 3.0)."""
    x = np.arange(1000, dtype=np.float64)
    trace = np.zeros(1000)
    for centre in [100, 400, 700]:
        trace += np.exp(-0.5 * ((x - centre) / 2.0) ** 2)
    # Normalise so each bump sums to ~1.0
    trace = trace / (trace.sum() / 3.0)
    return jnp.array(trace)


@pytest.fixture
def empty_spike_trace():
    """All-zero spike trace, length 1000."""
    return jnp.zeros(1000)


@pytest.fixture
def single_spike_trace():
    """One spike at index 200, length 1000."""
    trace = np.zeros(1000)
    trace[200] = 1.0
    return jnp.array(trace)


# ---------------------------------------------------------------------------
# AdEx parameter fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_adex_params():
    """Tonic spiking parameter dict (all 10 standard AdEx keys)."""
    return {
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
    }


@pytest.fixture
def trainable_params_jaxley_format():
    """List-of-dicts mimicking ``cell.get_parameters()``."""
    return [
        {"AdEx_g_L": jnp.array([10.0])},
        {"AdEx_E_L": jnp.array([-70.0])},
        {"AdEx_tau_w": jnp.array([30.0])},
        {"AdEx_a": jnp.array([2.0])},
        {"AdEx_b": jnp.array([0.0])},
    ]


# ---------------------------------------------------------------------------
# GuarinoFeatures fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def guarino_features_3spikes():
    """Hand-built GuarinoFeatures with 3 spikes at 10, 40, 70 ms."""
    from loss.guarino import GuarinoFeatures

    return GuarinoFeatures(
        t_first_spike=jnp.array(10.0),
        t_second_spike=jnp.array(40.0),
        t_third_spike=jnp.array(70.0),
        t_last_spike=jnp.array(70.0),
        inv_first_isi=jnp.array(1.0 / 30.0),
        inv_last_isi=jnp.array(1.0 / 30.0),
        firing_frequency=jnp.array(30.0),  # 3 spikes / 0.1 s
        v_stim_end=jnp.array(-65.0),
        has_first_spike=jnp.array(1.0),
        has_second_spike=jnp.array(1.0),
        has_third_spike=jnp.array(1.0),
        n_spikes=jnp.array(3.0),
    )


@pytest.fixture
def guarino_features_0spikes():
    """Hand-built GuarinoFeatures with no spikes."""
    from loss.guarino import GuarinoFeatures

    return GuarinoFeatures(
        t_first_spike=jnp.array(0.0),
        t_second_spike=jnp.array(0.0),
        t_third_spike=jnp.array(0.0),
        t_last_spike=jnp.array(0.0),
        inv_first_isi=jnp.array(0.0),
        inv_last_isi=jnp.array(0.0),
        firing_frequency=jnp.array(0.0),
        v_stim_end=jnp.array(-65.0),
        has_first_spike=jnp.array(0.0),
        has_second_spike=jnp.array(0.0),
        has_third_spike=jnp.array(0.0),
        n_spikes=jnp.array(0.0),
    )
