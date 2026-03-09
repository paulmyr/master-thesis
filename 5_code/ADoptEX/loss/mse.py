"""
Mean Squared Error Loss for Differentiable AdEx Optimization

This module implements MSE-based loss for gradient-based parameter optimization of
AdEx neuron models in Jaxley.

The MSE loss directly compares simulated and experimental voltage traces,
providing a simple but effective training signal.
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jax import Array


@dataclass
class MSELossConfig:
    """Configuration for MSE loss function."""

    # Whether to normalize by voltage variance
    normalize: bool = False

    # Numerical stability
    epsilon: float = 1e-6

    # Clamp voltages at this threshold (mV) before computing MSE.
    # When set, spike waveform differences above threshold are ignored,
    # focusing the loss on subthreshold dynamics only.
    clamp_threshold: Optional[float] = None

    # Inject artificial spike peaks at this voltage (mV) before computing MSE.
    # When set, the soft spike trace from surrogate gradients is used to replace
    # voltage at spike times with a visible peak, making spiking vs non-spiking
    # distinguishable in the loss. Requires results[2] (spikes) from jx.integrate.
    spike_peak_mv: Optional[float] = None


def mse_loss(
    sim_voltage: Array, exp_voltage: Array, config: Optional[MSELossConfig] = None
) -> Array:
    """
    Compute Mean Squared Error between simulated and experimental voltage traces.

    Args:
        sim_voltage: Simulated voltage trace [T] in mV
        exp_voltage: Experimental (target) voltage trace [T] in mV
        config: Loss configuration

    Returns:
        MSE loss (scalar)
    """
    if config is None:
        config = MSELossConfig()

    # Ensure same length
    min_len = min(len(sim_voltage), len(exp_voltage))
    sim_v = sim_voltage[:min_len]
    exp_v = exp_voltage[:min_len]

    # Clamp voltages at threshold to ignore spike waveform differences
    if config.clamp_threshold is not None:
        sim_v = jnp.minimum(sim_v, config.clamp_threshold)
        exp_v = jnp.minimum(exp_v, config.clamp_threshold)

    # Compute MSE
    squared_diff = jnp.square(sim_v - exp_v)
    mse = jnp.mean(squared_diff)

    # Optionally normalize by variance
    if config.normalize:
        variance = jnp.var(exp_v) + config.epsilon
        mse = mse / variance

    return mse


def make_mse_loss_fn(
    cell,
    data_stimuli,
    t_max: float,
    dt_ms: float,
    exp_voltage: Array,
    stim_end_index: int,
    loss_config: Optional[MSELossConfig] = None,
):
    """
    Create an MSE-based loss function for Jaxley training.

    This function returns a loss function that can be used with
    jax.value_and_grad for gradient-based optimization.

    Args:
        cell: Jaxley Cell with trainable AdExSurrogate parameters
        data_stimuli: Data stimuli tuple from cell.data_stimulate()
        t_max: Maximum simulation time (ms)
        dt_ms: Time step (ms)
        exp_voltage: Target voltage trace from experimental data
        stim_end_index: Index of stimulus end for voltage extraction
        loss_config: Loss configuration

    Returns:
        Loss function: params -> scalar loss
    """
    import jaxley as jx

    if loss_config is None:
        loss_config = MSELossConfig()

    def loss_fn(params):
        # Run simulation with current parameters
        results = jx.integrate(
            cell, params=params, data_stimuli=data_stimuli, delta_t=dt_ms, t_max=t_max
        )

        # Extract voltage trace
        # results[0] = voltage, results[1] = w, results[2] = spikes
        voltage = results[0].flatten()

        # Inject spike peaks if configured
        if loss_config.spike_peak_mv is not None:
            from ADoptEX.loss import inject_spike_peaks

            spikes = results[2].flatten()
            min_len = min(len(voltage), len(spikes), stim_end_index + 100)
            voltage = voltage[:min_len]
            spikes = spikes[:min_len]
            voltage = inject_spike_peaks(voltage, spikes, loss_config.spike_peak_mv)
        else:
            min_len = min(len(voltage), stim_end_index + 100)
            voltage = voltage[:min_len]

        # Compute MSE loss
        loss = mse_loss(voltage, exp_voltage[:min_len], loss_config)

        return loss

    return loss_fn
