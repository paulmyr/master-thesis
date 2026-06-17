"""
Time-to-First-Spike + Firing-Rate Loss for Differentiable AdEx Optimization

A deliberately minimal feature-based loss that only cares about two quantities:

- **time to first spike** (when does the neuron start firing), and
- **mean firing frequency** (how fast does it fire overall).

It is a focused subset of the Guarino et al. (2025) loss: it reuses the same
differentiable soft primitives (``soft_time_to_nth_spike``, ``soft_firing_frequency``,
``soft_spike_count``) but drops every other feature (later spike times, ISIs,
subthreshold voltage). The motivation is to give the optimizer a smooth,
low-dimensional target that separates *onset* (timing) from *throughput* (rate)
without the spike-timing credit-assignment problems of higher-order features.

The first-spike term is gated by soft spike-validity flags and carries a flat
penalty when the data fires but the simulation does not, mirroring the Guarino
missing-feature penalty. The frequency term is always active (it is 0 Hz for a
silent trace), so it provides gradient signal even before the model spikes at all.

Reference:
Guarino D, Carannante I, Destexhe A (2025) A unified model library maps how
neuromodulation reshapes the excitability landscape of neurons across the brain.
PLoS Comput Biol 21(12): e1013765.
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from .guarino import (relative_error, soft_firing_frequency, soft_spike_count,
                      soft_time_to_nth_spike)

# =============================================================================
# Loss Function Configuration
# =============================================================================


@dataclass
class TTFSRateLossConfig:
    """Configuration for the time-to-first-spike + firing-rate loss.

    Attributes:
        weight_t_first: Weight for the time-to-first-spike relative error. (default: 1.0)
        weight_firing_freq: Weight for the firing-frequency relative error. (default: 1.0)
        missing_spike_penalty: Flat penalty added when the data has a first spike
            but the simulation does not (soft-gated). Mirrors the Guarino
            missing-feature penalty. (default: 3.0)
        epsilon: Numerical-stability floor for relative-error denominators. (default: 1e-6)
    """

    weight_t_first: float = 1.0
    weight_firing_freq: float = 1.0
    missing_spike_penalty: float = 3.0
    epsilon: float = 1e-6


# =============================================================================
# Loss Function
# =============================================================================


def ttfs_rate_loss(
    sim_t_first_spike: Array,
    sim_firing_frequency: Array,
    sim_has_spike: Array,
    exp_t_first_spike: Array,
    exp_firing_frequency: Array,
    exp_has_spike: Array,
    config: Optional[TTFSRateLossConfig] = None,
) -> Array:
    """Compute the time-to-first-spike + firing-rate loss.

    Loss = weight_t_first * (gate * rel_err(t_first) + missing_penalty)
         + weight_firing_freq * rel_err(firing_frequency)

    The first-spike error is gated by ``sim_has_spike * exp_has_spike`` so it is
    only counted when both traces actually fire. The flat penalty kicks in when
    the data fires but the simulation is silent. The frequency term is always
    active (0 Hz target for silent traces).

    Args:
        sim_t_first_spike: Soft time to first spike of the simulation (ms).
        sim_firing_frequency: Soft firing frequency of the simulation (Hz).
        sim_has_spike: Soft validity flag (~1 if the simulation spikes, ~0 otherwise).
        exp_t_first_spike: Target time to first spike (ms).
        exp_firing_frequency: Target firing frequency (Hz).
        exp_has_spike: Target validity flag (1.0 if the data fires, 0.0 otherwise).
        config: Loss configuration (uses defaults if None).

    Returns:
        Total loss (scalar).
    """
    if config is None:
        config = TTFSRateLossConfig()

    # ----- Time to first spike -----
    error_t1 = relative_error(sim_t_first_spike, exp_t_first_spike, config.epsilon)
    penalty_t1 = config.missing_spike_penalty * jnp.maximum(
        0.0, exp_has_spike - sim_has_spike
    )
    loss_t1 = config.weight_t_first * (
        sim_has_spike * exp_has_spike * error_t1 + penalty_t1
    )

    # ----- Firing frequency -----
    error_freq = relative_error(
        sim_firing_frequency, exp_firing_frequency, config.epsilon
    )
    loss_freq = config.weight_firing_freq * error_freq

    return loss_t1 + loss_freq


# =============================================================================
# Training Integration
# =============================================================================


def make_ttfs_rate_loss_fn(
    cell,
    data_stimuli,
    t_max: float,
    dt_ms: float,
    exp_t_first_spike: float,
    exp_firing_frequency: float,
    stim_duration_ms: float,
    stim_end_index: int,
    exp_has_spike: float = 1.0,
    loss_config: Optional[TTFSRateLossConfig] = None,
    temperature: float = 0.5,
    beta: float = 5.0,
    validity_beta: float = 10.0,
):
    """Create a time-to-first-spike + firing-rate loss function for Jaxley training.

    Returns a ``params -> scalar`` closure compatible with ``jax.value_and_grad``.
    The closure runs ``jx.integrate``, extracts the soft spike trace (results[2]),
    derives the two soft features, and compares them to the experimental targets.

    Target values can be taken straight from an ``extract_experimental_features``
    call (``GuarinoFeatures.t_first_spike``, ``.firing_frequency``,
    ``.has_first_spike``), since this loss is a strict subset of the Guarino features.

    Args:
        cell: Jaxley Cell with trainable AdExSurrogate parameters.
        data_stimuli: Data stimuli tuple from ``cell.data_stimulate()``.
        t_max: Maximum simulation time (ms).
        dt_ms: Time step (ms).
        exp_t_first_spike: Target time to first spike (ms).
        exp_firing_frequency: Target firing frequency (Hz).
        stim_duration_ms: Stimulus duration for the frequency calculation.
        stim_end_index: Index of stimulus end for trace truncation.
        exp_has_spike: Target validity flag (1.0 if the data fires). (default: 1.0)
        loss_config: Loss configuration (uses defaults if None).
        temperature: Softmax temperature for soft first-spike extraction. (default: 0.5)
        beta: Sigmoid sharpness for spike-counting masks. (default: 5.0)
        validity_beta: Sigmoid sharpness for the soft has-spike validity flag,
            kept independent of ``beta`` so the gate stays sharp when the
            selection masks are softened. (default: 10.0)

    Returns:
        Loss function: params -> scalar loss.
    """
    import jaxley as jx

    if loss_config is None:
        loss_config = TTFSRateLossConfig()

    exp_t_first = jnp.asarray(exp_t_first_spike, dtype=float)
    exp_freq = jnp.asarray(exp_firing_frequency, dtype=float)
    exp_has = jnp.asarray(exp_has_spike, dtype=float)

    def loss_fn(params):
        results = jx.integrate(
            cell, params=params, data_stimuli=data_stimuli, delta_t=dt_ms, t_max=t_max
        )

        # results[0] = voltage, results[1] = w, results[2] = spikes
        spikes = results[2].flatten()
        min_len = min(len(spikes), stim_end_index + 1)
        spikes = spikes[:min_len]

        # Soft features
        n_spikes = soft_spike_count(spikes)
        sim_has_spike = jax.nn.sigmoid(validity_beta * (n_spikes - 0.5))
        sim_t_first = soft_time_to_nth_spike(
            spikes, n=1, dt_ms=dt_ms, temperature=temperature, beta=beta
        )
        sim_freq = soft_firing_frequency(spikes, stim_duration_ms)

        return ttfs_rate_loss(
            sim_t_first_spike=sim_t_first,
            sim_firing_frequency=sim_freq,
            sim_has_spike=sim_has_spike,
            exp_t_first_spike=exp_t_first,
            exp_firing_frequency=exp_freq,
            exp_has_spike=exp_has,
            config=loss_config,
        )

    return loss_fn