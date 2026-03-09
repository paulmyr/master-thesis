"""
Deistler Summary Statistics Loss for Differentiable AdEx Optimization

Implements the loss function from Deistler et al. (2025) for gradient-based
biophysical model fitting. The voltage trace is split into two time windows
(pre-stimulus and stimulus), and the mean and standard deviation of each
window are computed — yielding 4 differentiable summary statistics. These
are standardized (means / 8.0, stds / 4.0) and compared via mean absolute
error (MAE).

This approach deliberately avoids discrete statistics like spike count,
making the loss fully differentiable without surrogate gradient tricks
in the loss itself.

References:
- Deistler M, et al. (2025) Differentiable simulation enables large-scale
  training of detailed biophysical models of neural dynamics.
- Goncalves PJ, et al. (2020) Training deep neural density estimators to
  identify mechanistic models of neural dynamics.
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jax import Array


@dataclass
class DeistlerLossConfig:
    """Configuration for the Deistler summary statistics loss.

    Attributes:
        mean_norm: Normalization divisor for mean statistics. (default: 8.0)
        std_norm: Normalization divisor for std statistics. (default: 4.0)
        epsilon: Small constant for numerical stability in std. (default: 1e-6)
        spike_peak_mv: If set, inject spike peaks at this voltage (mV) using
            the soft spike trace from surrogate gradients. Makes the summary
            statistics sensitive to spiking activity. (default: None)
    """

    mean_norm: float = 8.0
    std_norm: float = 4.0
    epsilon: float = 1e-6
    spike_peak_mv: float | None = None


def _window_stats(
    voltage: Array, start: int, end: int, epsilon: float = 1e-6
) -> tuple[Array, Array]:
    """Compute mean and standard deviation of voltage in a window.

    Args:
        voltage: Full voltage trace [T]
        start: Start index (inclusive)
        end: End index (exclusive)
        epsilon: Numerical stability for std

    Returns:
        (mean, std) of voltage in [start:end]
    """
    window = voltage[start:end]
    mean = jnp.mean(window)
    std = jnp.sqrt(jnp.var(window) + epsilon)
    return mean, std


def deistler_loss(
    sim_voltage: Array,
    exp_voltage: Array,
    stim_start_index: int,
    stim_end_index: int,
    config: Optional[DeistlerLossConfig] = None,
) -> Array:
    """Compute the Deistler summary statistics loss.

    Splits the voltage into pre-stimulus and stimulus windows, computes
    mean and std for each, standardizes, and returns the MAE between
    simulated and experimental statistics.

    Args:
        sim_voltage: Simulated voltage trace [T] in mV
        exp_voltage: Experimental voltage trace [T] in mV
        stim_start_index: Index where stimulus begins
        stim_end_index: Index where stimulus ends
        config: Loss configuration (uses defaults if None)

    Returns:
        MAE loss over 4 standardized summary statistics (scalar)
    """
    if config is None:
        config = DeistlerLossConfig()

    # Truncate to common length
    min_len = min(len(sim_voltage), len(exp_voltage))
    sim_v = sim_voltage[:min_len]
    exp_v = exp_voltage[:min_len]

    # Clamp indices to valid range
    stim_start = jnp.clip(stim_start_index, 0, min_len)
    stim_end = jnp.clip(stim_end_index, stim_start, min_len)

    # Standardize: means / mean_norm, stds / std_norm
    mn = config.mean_norm
    sn = config.std_norm

    has_pre_stim = stim_start_index > 0

    if has_pre_stim:
        # Pre-stimulus window statistics
        sim_pre_mean, sim_pre_std = _window_stats(sim_v, 0, stim_start, config.epsilon)
        exp_pre_mean, exp_pre_std = _window_stats(exp_v, 0, stim_start, config.epsilon)

    # Stimulus window statistics
    sim_stim_mean, sim_stim_std = _window_stats(
        sim_v, stim_start, stim_end, config.epsilon
    )
    exp_stim_mean, exp_stim_std = _window_stats(
        exp_v, stim_start, stim_end, config.epsilon
    )

    if has_pre_stim:
        sim_stats = jnp.array(
            [
                sim_pre_mean / mn,
                sim_stim_mean / mn,
                sim_pre_std / sn,
                sim_stim_std / sn,
            ]
        )
        exp_stats = jnp.array(
            [
                exp_pre_mean / mn,
                exp_stim_mean / mn,
                exp_pre_std / sn,
                exp_stim_std / sn,
            ]
        )
    else:
        # No pre-stimulus window (e.g. after crop_to_stim_window) — use 2 stats
        sim_stats = jnp.array([sim_stim_mean / mn, sim_stim_std / sn])
        exp_stats = jnp.array([exp_stim_mean / mn, exp_stim_std / sn])

    # MAE over summary statistics
    return jnp.mean(jnp.abs(sim_stats - exp_stats))


def make_deistler_loss_fn(
    cell,
    data_stimuli,
    t_max: float,
    dt_ms: float,
    exp_voltage: Array,
    stim_start_index: int,
    stim_end_index: int,
    loss_config: Optional[DeistlerLossConfig] = None,
):
    """Create a Deistler summary statistics loss function for Jaxley training.

    Returns a loss function closure compatible with jax.value_and_grad.

    The closure runs jx.integrate, extracts the voltage trace, and computes
    the Deistler loss (MAE over 4 standardized summary statistics) against
    the experimental voltage.

    Args:
        cell: Jaxley Cell with trainable AdExSurrogate parameters
        data_stimuli: Data stimuli tuple from cell.data_stimulate()
        t_max: Maximum simulation time (ms)
        dt_ms: Time step (ms)
        exp_voltage: Target voltage trace from experimental data
        stim_start_index: Index where stimulus begins
        stim_end_index: Index where stimulus ends
        loss_config: Loss configuration (uses defaults if None)

    Returns:
        Loss function: params -> scalar loss
    """
    import jaxley as jx

    if loss_config is None:
        loss_config = DeistlerLossConfig()

    def loss_fn(params):
        results = jx.integrate(
            cell,
            params=params,
            data_stimuli=data_stimuli,
            delta_t=dt_ms,
            t_max=t_max,
        )

        voltage = results[0].flatten()

        # Inject spike peaks if configured
        if loss_config.spike_peak_mv is not None:
            from ADoptEX.loss import inject_spike_peaks

            spikes = results[2].flatten()
            min_len = min(len(voltage), len(spikes), stim_end_index + 100)
            voltage = voltage[:min_len]
            spikes = spikes[:min_len]
            voltage = inject_spike_peaks(voltage, spikes, loss_config.spike_peak_mv)

        return deistler_loss(
            sim_voltage=voltage,
            exp_voltage=exp_voltage,
            stim_start_index=stim_start_index,
            stim_end_index=stim_end_index,
            config=loss_config,
        )

    return loss_fn
