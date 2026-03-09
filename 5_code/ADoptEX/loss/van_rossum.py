"""
Van Rossum Distance Loss for Differentiable AdEx Optimization

The Van Rossum distance (van Rossum, 2001) measures dissimilarity between two spike
trains by convolving each with an exponential kernel exp(-t/tau), then computing the
L2 distance between the filtered traces. This makes spike timing the primary training
signal, unlike voltage-based losses where the AdEx reset mechanism removes spike peaks.

The time constant tau controls the balance between timing precision (small tau) and
rate sensitivity (large tau). The distance is naturally differentiable through
jnp.convolve, making it compatible with JAX autodiff without custom VJP rules.

Reference:
van Rossum MCW (2001) A novel spike distance.
Neural Computation 13(4): 751-763.
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import Array


@dataclass
class VanRossumLossConfig:
    """Configuration for Van Rossum distance loss.

    Attributes:
        tau_ms: Exponential kernel time constant (ms). Small values emphasize
                precise spike timing; large values emphasize firing rate. (default: 10.0)
        kernel_n_tau: Truncate kernel at this many time constants.
                      exp(-5) ~ 0.007, so 5 tau is standard. (default: 5.0)
        weight_van_rossum: Weight for the Van Rossum distance component. (default: 1.0)
        weight_subthreshold: Weight for optional subthreshold voltage MAE.
                             Set > 0 to include subthreshold signal. (default: 0.0)
        subthreshold_clamp_mv: Clamp voltage above this value before computing
                               subthreshold MAE, to exclude spike artifacts. (default: -40.0)
    """

    tau_ms: float = 10.0
    kernel_n_tau: float = 5.0
    weight_van_rossum: float = 1.0
    weight_subthreshold: float = 0.0
    subthreshold_clamp_mv: float = -40.0


def _build_kernel(tau_ms: float, kernel_n_tau: float, dt_ms: float) -> Array:
    """Build a causal exponential kernel for spike train filtering.

    Args:
        tau_ms: Kernel time constant (ms)
        kernel_n_tau: Number of time constants before truncation
        dt_ms: Simulation time step (ms)

    Returns:
        1D kernel array starting at 1.0 and decaying exponentially
    """
    kernel_len = max(1, int(kernel_n_tau * tau_ms / dt_ms))
    return jnp.exp(-jnp.arange(kernel_len) * dt_ms / tau_ms)


def van_rossum_distance(
    sim_spikes: Array,
    exp_spikes: Array,
    dt_ms: float,
    config: Optional[VanRossumLossConfig] = None,
) -> Array:
    """Compute the Van Rossum distance between two spike trains.

    Each spike train is convolved with an exponential kernel exp(-t/tau),
    then the L2 distance (scaled by dt/tau) is computed.

    Args:
        sim_spikes: Simulated spike train [T_sim] (binary or soft indicators)
        exp_spikes: Experimental spike train [T_exp] (binary indicators)
        dt_ms: Time step (ms)
        config: Loss configuration (uses defaults if None)

    Returns:
        Van Rossum distance (scalar, non-negative)
    """
    if config is None:
        config = VanRossumLossConfig()

    kernel = _build_kernel(config.tau_ms, config.kernel_n_tau, dt_ms)

    # Truncate to common length
    min_len = min(len(sim_spikes), len(exp_spikes))
    sim = sim_spikes[:min_len]
    exp = exp_spikes[:min_len]

    # Convolve and truncate back to original length
    filtered_sim = jnp.convolve(sim, kernel)[:min_len]
    filtered_exp = jnp.convolve(exp, kernel)[:min_len]

    # Van Rossum distance: (dt/tau) * sum((f_sim - f_exp)^2)
    distance = (dt_ms / config.tau_ms) * jnp.sum(
        jnp.square(filtered_sim - filtered_exp)
    )

    return distance


def spike_train_from_voltage(
    voltage: Array,
    threshold_mv: float = 0.0,
    dt_ms: float = 0.025,
) -> Array:
    """Detect spikes via rising-edge threshold crossing, return binary spike train.

    A spike is registered at index i when voltage[i] >= threshold and
    voltage[i-1] < threshold (rising edge). This mirrors the hard detection
    logic in guarino.detect_spikes_hard but returns a full-length binary
    array suitable for convolution with jnp.convolve.

    Args:
        voltage: Voltage trace [T] in mV
        threshold_mv: Spike detection threshold (mV)
        dt_ms: Time step (ms), unused but kept for API consistency

    Returns:
        Binary spike train [T] with 1.0 at each rising-edge crossing
    """
    above = np.asarray(voltage) >= threshold_mv
    # Rising edge: above[i] and not above[i-1]
    crossings = np.zeros(len(voltage), dtype=np.float64)
    crossings[1:] = above[1:] & ~above[:-1]
    return jnp.array(crossings)


def make_van_rossum_loss_fn(
    cell,
    data_stimuli,
    t_max: float,
    dt_ms: float,
    exp_spike_train: Array,
    stim_end_index: int,
    loss_config: Optional[VanRossumLossConfig] = None,
    exp_voltage: Optional[Array] = None,
):
    """Create a Van Rossum distance loss function for Jaxley training.

    Returns a loss function closure compatible with jax.value_and_grad.

    The closure runs jx.integrate, extracts the spike trace (results[2]),
    and computes the Van Rossum distance against the experimental spike train.
    Optionally adds a subthreshold voltage MAE term.

    Args:
        cell: Jaxley Cell with trainable AdExSurrogate parameters
        data_stimuli: Data stimuli tuple from cell.data_stimulate()
        t_max: Maximum simulation time (ms)
        dt_ms: Time step (ms)
        exp_spike_train: Target binary spike train from experimental data
        stim_end_index: Index of stimulus end for trace extraction
        loss_config: Loss configuration (uses defaults if None)
        exp_voltage: Target voltage trace, required if weight_subthreshold > 0

    Returns:
        Loss function: params -> scalar loss

    Raises:
        ValueError: If weight_subthreshold > 0 but exp_voltage is None
    """
    import jaxley as jx

    if loss_config is None:
        loss_config = VanRossumLossConfig()

    if loss_config.weight_subthreshold > 0 and exp_voltage is None:
        raise ValueError("exp_voltage must be provided when weight_subthreshold > 0")

    def loss_fn(params):
        results = jx.integrate(
            cell,
            params=params,
            data_stimuli=data_stimuli,
            delta_t=dt_ms,
            t_max=t_max,
        )

        # Extract traces and truncate to stimulus window (+ small buffer)
        spikes = results[2].flatten()
        voltage = results[0].flatten()
        min_len = min(
            len(spikes), len(voltage), len(exp_spike_train), stim_end_index + 100
        )
        if loss_config.weight_subthreshold > 0:
            min_len = min(min_len, len(exp_voltage))
        spikes = spikes[:min_len]

        # Van Rossum distance
        loss = loss_config.weight_van_rossum * van_rossum_distance(
            spikes, exp_spike_train[:min_len], dt_ms, loss_config
        )

        # Optional subthreshold voltage MAE
        if loss_config.weight_subthreshold > 0:
            sim_v = voltage[:min_len]
            exp_v = exp_voltage[:min_len]
            # Clamp to exclude spike artifacts
            clamp = loss_config.subthreshold_clamp_mv
            sim_clamped = jnp.minimum(sim_v, clamp)
            exp_clamped = jnp.minimum(exp_v, clamp)
            sub_mae = jnp.mean(jnp.abs(sim_clamped - exp_clamped))
            loss = loss + loss_config.weight_subthreshold * sub_mae

        return loss

    return loss_fn
