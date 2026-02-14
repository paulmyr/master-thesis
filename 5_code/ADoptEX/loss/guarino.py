"""
Guarino et al. (2025) Feature-Based Loss for Differentiable AdEx Optimization

This module implements differentiable versions of the electrophysiological features
described in Guarino et al. (2025) for gradient-based parameter optimization of
AdEx neuron models in Jaxley.

Features:
- Time to first/second/third/last spike
- Inverse of first/last interspike interval (ISI)
- Firing frequency
- Voltage at stimulus end

The key challenge is making discrete spike-detection operations differentiable.
We use soft approximations with temperature-scaled softmax operations.

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

# =============================================================================
# Soft Feature Primitives
# =============================================================================


def soft_time_to_nth_spike(
    spike_trace: Array,
    n: int,
    dt_ms: float,
    temperature: float = 0.1,
    beta: float = 10.0,
) -> Array:
    """
    Compute soft (differentiable) approximation of time to n-th spike.

    Uses cumulative sum masking to identify the n-th spike region,
    then computes a weighted average of time using spike activity.

    Args:
        spike_trace: Soft spike indicators [T] with values in [0, 1]
        n: Which spike (1 = first, 2 = second, etc.)
        dt_ms: Time step in milliseconds
        temperature: Controls sharpness of spike selection (lower = sharper)
        beta: Sigmoid sharpness for masking (default 10.0)

    Returns:
        Soft estimate of time to n-th spike (ms)
    """
    T = len(spike_trace)
    time_indices = jnp.arange(T) * dt_ms

    # Cumulative spike count (soft)
    cumsum = jnp.cumsum(spike_trace)

    # Create mask for n-th spike region using sigmoid transitions
    # The mask is high when cumsum is around n (between n-0.5 and n+0.5)
    lower_mask = jax.nn.sigmoid(beta * (cumsum - (n - 0.5)))
    upper_mask = jax.nn.sigmoid(beta * (cumsum - (n + 0.5)))
    nth_spike_mask = lower_mask - upper_mask

    # Weight by spike strength and mask
    # Using power to sharpen the selection for sparse spikes
    weights = spike_trace * nth_spike_mask

    # Sharpen weights using temperature (lower = sharper)
    # Add small epsilon to avoid log(0)
    weights_sharp = jnp.power(weights + 1e-10, 1.0 / temperature)

    # Normalize to get probability distribution
    # Use weighted average instead of softmax to avoid exp(0)=1 problem
    weight_sum = jnp.sum(weights_sharp)
    weights_normalized = weights_sharp / (weight_sum + 1e-10)

    # Expected time (weighted average)
    expected_time = jnp.sum(time_indices * weights_normalized)

    return expected_time


def soft_time_to_last_spike(
    spike_trace: Array, dt_ms: float, temperature: float = 0.1, beta: float = 10.0
) -> Array:
    """
    Compute soft (differentiable) approximation of time to last spike.

    Uses reverse cumulative sum to identify the last spike position.

    Args:
        spike_trace: Soft spike indicators [T] with values in [0, 1]
        dt_ms: Time step in milliseconds
        temperature: Controls sharpness of selection (lower = sharper)
        beta: Sigmoid sharpness for masking

    Returns:
        Soft estimate of time to last spike (ms)
    """
    T = len(spike_trace)
    time_indices = jnp.arange(T) * dt_ms

    # Reverse cumulative sum: count spikes remaining after each time point
    reverse_cumsum = jnp.cumsum(spike_trace[::-1])[::-1]

    # Last spike: this is where reverse_cumsum transitions from >=1 to <1
    # We want positions where there's a spike and it's the last one
    # Mask: reverse_cumsum is around 0.5-1.5 (one spike remaining including this)
    last_spike_mask = (
        jax.nn.sigmoid(beta * (reverse_cumsum - 0.5))
        - jax.nn.sigmoid(beta * (reverse_cumsum - 1.5))
    ) * spike_trace

    # Sharpen weights using power (lower temperature = sharper)
    weights_sharp = jnp.power(last_spike_mask + 1e-10, 1.0 / temperature)

    # Normalize to get probability distribution
    weight_sum = jnp.sum(weights_sharp)
    weights_normalized = weights_sharp / (weight_sum + 1e-10)

    expected_time = jnp.sum(time_indices * weights_normalized)

    return expected_time


def soft_time_to_second_last_spike(
    spike_trace: Array, dt_ms: float, temperature: float = 0.1, beta: float = 10.0
) -> Array:
    """
    Compute soft approximation of time to second-to-last spike.

    Used for computing the last ISI.

    Args:
        spike_trace: Soft spike indicators [T] with values in [0, 1]
        dt_ms: Time step in milliseconds
        temperature: Controls sharpness of selection (lower = sharper)
        beta: Sigmoid sharpness for masking

    Returns:
        Soft estimate of time to second-to-last spike (ms)
    """
    T = len(spike_trace)
    time_indices = jnp.arange(T) * dt_ms

    # Reverse cumulative sum
    reverse_cumsum = jnp.cumsum(spike_trace[::-1])[::-1]

    # Second-to-last spike: reverse_cumsum is around 1.5-2.5 (two spikes remaining)
    second_last_mask = (
        jax.nn.sigmoid(beta * (reverse_cumsum - 1.5))
        - jax.nn.sigmoid(beta * (reverse_cumsum - 2.5))
    ) * spike_trace

    # Sharpen weights using power (lower temperature = sharper)
    weights_sharp = jnp.power(second_last_mask + 1e-10, 1.0 / temperature)

    # Normalize to get probability distribution
    weight_sum = jnp.sum(weights_sharp)
    weights_normalized = weights_sharp / (weight_sum + 1e-10)

    expected_time = jnp.sum(time_indices * weights_normalized)

    return expected_time


def soft_inverse_first_isi(
    spike_trace: Array,
    dt_ms: float,
    temperature: float = 0.1,
    beta: float = 10.0,
    epsilon: float = 1e-6,
) -> Array:
    """
    Compute soft approximation of inverse first ISI.

    inv_first_ISI = 1 / (t_second_spike - t_first_spike)

    Args:
        spike_trace: Soft spike indicators [T]
        dt_ms: Time step in milliseconds
        temperature: Softmax temperature
        beta: Sigmoid sharpness
        epsilon: Small value for numerical stability in division

    Returns:
        Soft estimate of inverse first ISI (1/ms, multiply by 1000 for Hz)
    """
    t1 = soft_time_to_nth_spike(
        spike_trace, n=1, dt_ms=dt_ms, temperature=temperature, beta=beta
    )
    t2 = soft_time_to_nth_spike(
        spike_trace, n=2, dt_ms=dt_ms, temperature=temperature, beta=beta
    )

    isi = t2 - t1
    # Safe inverse with epsilon and minimum ISI
    isi_safe = jnp.maximum(isi, epsilon)
    inv_isi = 1.0 / isi_safe

    return inv_isi


def soft_inverse_last_isi(
    spike_trace: Array,
    dt_ms: float,
    temperature: float = 0.1,
    beta: float = 10.0,
    epsilon: float = 1e-6,
) -> Array:
    """
    Compute soft approximation of inverse last ISI.

    inv_last_ISI = 1 / (t_last_spike - t_second_last_spike)

    Args:
        spike_trace: Soft spike indicators [T]
        dt_ms: Time step in milliseconds
        temperature: Softmax temperature
        beta: Sigmoid sharpness
        epsilon: Small value for numerical stability

    Returns:
        Soft estimate of inverse last ISI (1/ms)
    """
    t_last = soft_time_to_last_spike(
        spike_trace, dt_ms=dt_ms, temperature=temperature, beta=beta
    )
    t_second_last = soft_time_to_second_last_spike(
        spike_trace, dt_ms=dt_ms, temperature=temperature, beta=beta
    )

    isi = t_last - t_second_last
    isi_safe = jnp.maximum(isi, epsilon)
    inv_isi = 1.0 / isi_safe

    return inv_isi


def soft_firing_frequency(spike_trace: Array, stim_duration_ms: float) -> Array:
    """
    Compute soft firing frequency (Hz).

    This is naturally differentiable - just the sum of soft spike indicators
    divided by duration.

    Args:
        spike_trace: Soft spike indicators [T] with values in [0, 1]
        stim_duration_ms: Duration of stimulus in milliseconds

    Returns:
        Firing frequency in Hz
    """
    n_spikes_soft = jnp.sum(spike_trace)
    frequency_hz = n_spikes_soft / (stim_duration_ms / 1000.0)
    return frequency_hz


def soft_spike_count(spike_trace: Array) -> Array:
    """
    Compute soft spike count.

    Simply the sum of soft spike indicators.

    Args:
        spike_trace: Soft spike indicators [T]

    Returns:
        Soft spike count (float)
    """
    return jnp.sum(spike_trace)


# =============================================================================
# Feature Container
# =============================================================================


@dataclass
class GuarinoFeatures:
    """
    Container for Guarino et al. feature values.

    All values are JAX arrays to enable differentiation.
    """

    # Spike timing features
    t_first_spike: Array
    t_second_spike: Array
    t_third_spike: Array
    t_last_spike: Array

    # ISI features
    inv_first_isi: Array
    inv_last_isi: Array

    # Rate features
    firing_frequency: Array

    # Voltage features
    v_stim_end: Array

    # Validity flags (soft, for conditional loss computation)
    has_first_spike: Array  # soft: sigmoid(10*(n_spikes - 0.5))
    has_second_spike: Array  # soft: sigmoid(10*(n_spikes - 1.5))
    has_third_spike: Array  # soft: sigmoid(10*(n_spikes - 2.5))

    # Spike count
    n_spikes: Array

    def __str__(self):
        """Print feature values in a readable format."""
        return f"""\nFeatures:
    Spike count:     {float(self.n_spikes):.2f}
    t_first_spike:   {float(self.t_first_spike):.2f} ms
    t_second_spike:  {float(self.t_second_spike):.2f} ms
    t_third_spike:   {float(self.t_third_spike):.2f} ms
    t_last_spike:    {float(self.t_last_spike):.2f} ms
    inv_first_ISI:   {float(self.inv_first_isi):.4f} 1/ms
    inv_last_ISI:    {float(self.inv_last_isi):.4f} 1/ms
    firing_freq:     {float(self.firing_frequency):.2f} Hz
    v_stim_end:      {float(self.v_stim_end):.2f} mV
    has_spikes:      [{float(self.has_first_spike):.2f}
                      {float(self.has_second_spike):.2f},
                      {float(self.has_third_spike):.2f}]"""


# =============================================================================
# Feature Extractor
# =============================================================================


class GuarinoFeatureExtractor:
    """
    Extract Guarino features from voltage/spike traces.

    All operations are differentiable via soft approximations using
    temperature-scaled softmax.
    """

    def __init__(
        self,
        dt_ms: float,
        stim_duration_ms: float,
        stim_end_index: int,
        temperature: float = 0.1,
        beta: float = 10.0,
    ):
        """
        Initialize the feature extractor.

        Args:
            dt_ms: Simulation time step in milliseconds
            stim_duration_ms: Duration of stimulus for frequency calculation
            stim_end_index: Index of stimulus end for voltage extraction
            temperature: Softmax temperature (lower = sharper approximation)
            beta: Sigmoid sharpness for spike counting masks
        """
        self.dt_ms = dt_ms
        self.stim_duration_ms = stim_duration_ms
        self.stim_end_index = stim_end_index
        self.temperature = temperature
        self.beta = beta

    def extract(self, voltage_trace: Array, spike_trace: Array) -> GuarinoFeatures:
        """
        Extract all Guarino features from simulation output.

        Args:
            voltage_trace: Membrane voltage [T] in mV
            spike_trace: Soft spike indicators [T] from AdExSurrogate (values in [0, 1])

        Returns:
            GuarinoFeatures dataclass with all feature values
        """
        # Spike count (soft)
        n_spikes = soft_spike_count(spike_trace)

        # Validity indicators (soft) using sigmoid
        validity_beta = 10.0
        has_first = jax.nn.sigmoid(validity_beta * (n_spikes - 0.5))
        has_second = jax.nn.sigmoid(validity_beta * (n_spikes - 1.5))
        has_third = jax.nn.sigmoid(validity_beta * (n_spikes - 2.5))

        # Spike times
        t_first = soft_time_to_nth_spike(
            spike_trace,
            n=1,
            dt_ms=self.dt_ms,
            temperature=self.temperature,
            beta=self.beta,
        )
        t_second = soft_time_to_nth_spike(
            spike_trace,
            n=2,
            dt_ms=self.dt_ms,
            temperature=self.temperature,
            beta=self.beta,
        )
        t_third = soft_time_to_nth_spike(
            spike_trace,
            n=3,
            dt_ms=self.dt_ms,
            temperature=self.temperature,
            beta=self.beta,
        )
        t_last = soft_time_to_last_spike(
            spike_trace, dt_ms=self.dt_ms, temperature=self.temperature, beta=self.beta
        )

        # ISI features
        inv_first_isi = soft_inverse_first_isi(
            spike_trace, dt_ms=self.dt_ms, temperature=self.temperature, beta=self.beta
        )
        inv_last_isi = soft_inverse_last_isi(
            spike_trace, dt_ms=self.dt_ms, temperature=self.temperature, beta=self.beta
        )

        # Frequency
        freq = soft_firing_frequency(spike_trace, self.stim_duration_ms)

        # Voltage at stimulus end
        # Clip index to valid range
        safe_idx = jnp.minimum(self.stim_end_index, len(voltage_trace) - 1)
        v_end = voltage_trace[safe_idx]

        return GuarinoFeatures(
            t_first_spike=t_first,
            t_second_spike=t_second,
            t_third_spike=t_third,
            t_last_spike=t_last,
            inv_first_isi=inv_first_isi,
            inv_last_isi=inv_last_isi,
            firing_frequency=freq,
            v_stim_end=v_end,
            has_first_spike=has_first,
            has_second_spike=has_second,
            has_third_spike=has_third,
            n_spikes=n_spikes,
        )


# =============================================================================
# Loss Function Configuration
# =============================================================================


@dataclass
class GuarinoLossConfig:
    """Configuration for Guarino loss function."""

    # Feature weights (relative importance)
    weight_t_first: float = 1.0
    weight_t_second: float = 1.0
    weight_t_third: float = 1.0
    weight_t_last: float = 1.0
    weight_inv_first_isi: float = 1.0
    weight_inv_last_isi: float = 1.0
    weight_firing_freq: float = 1.0
    weight_v_stim_end: float = 1.0  # Increase for strongly adapting neurons

    # Spike count squared error weight
    # Adds (n_spikes_sim - n_spikes_exp)^2 * weight to the loss.
    # Unlike the flat missing_feature_penalty, this scales with how many
    # spikes are missing and has clean gradient flow through soft_spike_count.
    weight_spike_count: float = 0.0

    # Penalty for missing features (from Guarino paper: +3 per missing feature)
    missing_feature_penalty: float = 3.0

    # Numerical stability
    epsilon: float = 1e-6


# =============================================================================
# Loss Function
# =============================================================================


def relative_error(sim_value: Array, exp_value: Array, epsilon: float = 1e-6) -> Array:
    """
    Compute relative error between simulated and experimental values.

    Per Guarino et al.:
    error = |sim - exp| / |exp|

    Args:
        sim_value: Simulated feature value
        exp_value: Experimental (target) feature value
        epsilon: Small value for numerical stability

    Returns:
        Relative error (scalar)
    """
    return jnp.abs(sim_value - exp_value) / (jnp.abs(exp_value) + epsilon)


def guarino_loss(
    sim_features: GuarinoFeatures,
    exp_features: GuarinoFeatures,
    config: Optional[GuarinoLossConfig] = None,
) -> Array:
    """
    Compute Guarino-style feature-based loss.

    Loss = sum_i( weight_i * relative_error_i ) + penalties

    Penalties are applied when simulated trace lacks features present in data.
    Features are weighted by their validity (soft masks) to handle cases where
    spikes don't exist.

    Args:
        sim_features: Features extracted from simulated trace (soft/differentiable)
        exp_features: Features extracted from experimental trace (hard detection)
        config: Loss configuration (weights, penalties)

    Returns:
        Total loss (scalar)
    """
    if config is None:
        config = GuarinoLossConfig()

    total_loss = jnp.array(0.0)

    # ----- Time to first spike -----
    error_t1 = relative_error(
        sim_features.t_first_spike, exp_features.t_first_spike, config.epsilon
    )
    # Apply penalty only when simulation is missing a spike that data has
    # Using max(0, exp - sim) to handle soft validity flags correctly
    penalty_t1 = config.missing_feature_penalty * jnp.maximum(
        0.0, exp_features.has_first_spike - sim_features.has_first_spike
    )
    # Only count error when both have the feature
    loss_t1 = config.weight_t_first * (
        sim_features.has_first_spike * exp_features.has_first_spike * error_t1
        + penalty_t1
    )
    total_loss = total_loss + loss_t1

    # ----- Time to second spike -----
    error_t2 = relative_error(
        sim_features.t_second_spike, exp_features.t_second_spike, config.epsilon
    )
    penalty_t2 = config.missing_feature_penalty * jnp.maximum(
        0.0, exp_features.has_second_spike - sim_features.has_second_spike
    )
    loss_t2 = config.weight_t_second * (
        sim_features.has_second_spike * exp_features.has_second_spike * error_t2
        + penalty_t2
    )
    total_loss = total_loss + loss_t2

    # ----- Time to third spike -----
    error_t3 = relative_error(
        sim_features.t_third_spike, exp_features.t_third_spike, config.epsilon
    )
    penalty_t3 = config.missing_feature_penalty * jnp.maximum(
        0.0, exp_features.has_third_spike - sim_features.has_third_spike
    )
    loss_t3 = config.weight_t_third * (
        sim_features.has_third_spike * exp_features.has_third_spike * error_t3
        + penalty_t3
    )
    total_loss = total_loss + loss_t3

    # ----- Time to last spike -----
    # Only meaningful if at least one spike exists
    error_t_last = relative_error(
        sim_features.t_last_spike, exp_features.t_last_spike, config.epsilon
    )
    loss_t_last = config.weight_t_last * (
        sim_features.has_first_spike * exp_features.has_first_spike * error_t_last
    )
    total_loss = total_loss + loss_t_last

    # ----- Inverse first ISI -----
    # Requires at least 2 spikes
    error_inv_isi1 = relative_error(
        sim_features.inv_first_isi, exp_features.inv_first_isi, config.epsilon
    )
    loss_inv_isi1 = config.weight_inv_first_isi * (
        sim_features.has_second_spike * exp_features.has_second_spike * error_inv_isi1
    )
    total_loss = total_loss + loss_inv_isi1

    # ----- Inverse last ISI -----
    # Requires at least 2 spikes
    error_inv_isi_last = relative_error(
        sim_features.inv_last_isi, exp_features.inv_last_isi, config.epsilon
    )
    loss_inv_isi_last = config.weight_inv_last_isi * (
        sim_features.has_second_spike
        * exp_features.has_second_spike
        * error_inv_isi_last
    )
    total_loss = total_loss + loss_inv_isi_last

    # ----- Firing frequency -----
    # Always computed (0 if no spikes)
    error_freq = relative_error(
        sim_features.firing_frequency, exp_features.firing_frequency, config.epsilon
    )
    loss_freq = config.weight_firing_freq * error_freq
    total_loss = total_loss + loss_freq

    # ----- Voltage at stimulus end -----
    # Always computed
    error_v_end = relative_error(
        sim_features.v_stim_end, exp_features.v_stim_end, config.epsilon
    )
    loss_v_end = config.weight_v_stim_end * error_v_end
    total_loss = total_loss + loss_v_end

    # ----- Spike count squared error -----
    # Scales with how many spikes are missing (unlike flat penalty)
    if config.weight_spike_count > 0:
        spike_count_error = (sim_features.n_spikes - exp_features.n_spikes) ** 2
        total_loss = total_loss + config.weight_spike_count * spike_count_error

    return total_loss


# =============================================================================
# Experimental Feature Extraction (Hard Detection)
# =============================================================================


def detect_spikes_hard(
    voltage: Array,
    threshold_mv: float = 0.0,
    min_interval_ms: float = 2.0,
    dt_ms: float = 0.025,
) -> Array:
    """
    Detect spike times from voltage trace using hard threshold crossing.

    This is for extracting ground truth features from experimental data.

    Args:
        voltage: Voltage trace [T] in mV
        threshold_mv: Spike detection threshold (mV)
        min_interval_ms: Minimum interval between spikes (ms)
        dt_ms: Time step (ms)

    Returns:
        Array of spike times (ms)
    """
    import numpy as np

    voltage_np = np.array(voltage)

    # Find threshold crossings (rising edge)
    above_threshold = voltage_np > threshold_mv
    crossings = np.diff(above_threshold.astype(int)) > 0
    spike_indices = np.where(crossings)[0] + 1

    # Filter out spikes that are too close together
    if len(spike_indices) > 1:
        min_interval_samples = int(min_interval_ms / dt_ms)
        filtered_indices = [spike_indices[0]]
        for idx in spike_indices[1:]:
            if idx - filtered_indices[-1] >= min_interval_samples:
                filtered_indices.append(idx)
        spike_indices = np.array(filtered_indices)

    spike_times = spike_indices * dt_ms
    return jnp.array(spike_times)


def extract_experimental_features(
    voltage_trace: Array,
    dt_ms: float,
    stim_duration_ms: float,
    stim_end_index: Optional[int] = None,
    spike_threshold_mv: float = 0.0,
) -> GuarinoFeatures:
    """
    Extract Guarino features from experimental voltage trace.

    Uses hard spike detection for ground truth. Not differentiable,
    but doesn't need to be since this is for target features.

    Args:
        voltage_trace: Voltage trace [T] in mV
        dt_ms: Time step in milliseconds
        stim_duration_ms: Stimulus duration in milliseconds
        stim_end_index: Index of stimulus end (computed from duration if None)
        spike_threshold_mv: Threshold for spike detection (mV)

    Returns:
        GuarinoFeatures with experimental feature values
    """
    import numpy as np

    voltage_np = np.array(voltage_trace)

    # Detect spikes
    spike_times = detect_spikes_hard(
        voltage_trace, threshold_mv=spike_threshold_mv, dt_ms=dt_ms
    )
    spike_times_np = np.array(spike_times)
    n_spikes = len(spike_times_np)

    # Extract features with appropriate defaults for missing spikes
    t_first = spike_times_np[0] if n_spikes >= 1 else 0.0
    t_second = spike_times_np[1] if n_spikes >= 2 else 0.0
    t_third = spike_times_np[2] if n_spikes >= 3 else 0.0
    t_last = spike_times_np[-1] if n_spikes >= 1 else 0.0

    # ISI features (need at least 2 spikes)
    if n_spikes >= 2:
        inv_first_isi = 1.0 / (spike_times_np[1] - spike_times_np[0])
        inv_last_isi = 1.0 / (spike_times_np[-1] - spike_times_np[-2])
    else:
        inv_first_isi = 0.0
        inv_last_isi = 0.0

    # Firing frequency
    freq = n_spikes / (stim_duration_ms / 1000.0)

    # Voltage at stimulus end
    if stim_end_index is None:
        stim_end_index = int(stim_duration_ms / dt_ms)
    safe_idx = min(stim_end_index, len(voltage_np) - 1)
    v_end = voltage_np[safe_idx]

    return GuarinoFeatures(
        t_first_spike=jnp.array(t_first),
        t_second_spike=jnp.array(t_second),
        t_third_spike=jnp.array(t_third),
        t_last_spike=jnp.array(t_last),
        inv_first_isi=jnp.array(inv_first_isi),
        inv_last_isi=jnp.array(inv_last_isi),
        firing_frequency=jnp.array(freq),
        v_stim_end=jnp.array(v_end),
        has_first_spike=jnp.array(float(n_spikes >= 1)),
        has_second_spike=jnp.array(float(n_spikes >= 2)),
        has_third_spike=jnp.array(float(n_spikes >= 3)),
        n_spikes=jnp.array(float(n_spikes)),
    )


# =============================================================================
# Training Integration
# =============================================================================


def make_guarino_loss_fn(
    cell,
    data_stimuli,
    t_max: float,
    dt_ms: float,
    exp_features: GuarinoFeatures,
    stim_duration_ms: float,
    stim_end_index: int,
    loss_config: Optional[GuarinoLossConfig] = None,
    temperature: float = 0.1,
    beta: float = 10.0,
):
    """
    Create a Guarino feature-based loss function for Jaxley training.

    This function returns a loss function that can be used with
    jax.value_and_grad for gradient-based optimization.

    Args:
        cell: Jaxley Cell with trainable AdExSurrogate parameters
        data_stimuli: Data stimuli tuple from cell.data_stimulate()
        t_max: Maximum simulation time (ms)
        dt_ms: Time step (ms)
        exp_features: Target features from experimental data
        stim_duration_ms: Stimulus duration for frequency calculation
        stim_end_index: Index of stimulus end for voltage extraction
        loss_config: Loss configuration (weights, penalties)
        temperature: Softmax temperature for soft feature extraction
        beta: Sigmoid sharpness for spike counting

    Returns:
        Loss function: params -> scalar loss
    """
    import jaxley as jx

    if loss_config is None:
        loss_config = GuarinoLossConfig()

    feature_extractor = GuarinoFeatureExtractor(
        dt_ms=dt_ms,
        stim_duration_ms=stim_duration_ms,
        stim_end_index=stim_end_index,
        temperature=temperature,
        beta=beta,
    )

    def loss_fn(params):
        # Run simulation with current parameters
        results = jx.integrate(
            cell, params=params, data_stimuli=data_stimuli, delta_t=dt_ms, t_max=t_max
        )

        # Extract traces
        # results[0] = voltage, results[1] = w, results[2] = spikes
        voltage = results[0].flatten()
        spikes = results[2].flatten()

        # Ensure same length (truncate if needed)
        min_len = min(len(voltage), stim_end_index + 1)
        voltage = voltage[:min_len]
        spikes = spikes[:min_len]

        # Extract soft features
        sim_features = feature_extractor.extract(voltage, spikes)

        # Compute loss
        loss = guarino_loss(sim_features, exp_features, loss_config)

        return loss

    return loss_fn
