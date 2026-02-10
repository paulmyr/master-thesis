"""
Coincidence Factor (Gamma) for spike train similarity.

Implementation based on Jolivet et al. (2008):
"A benchmark test for a quantitative assessment of simple neuron models"
Journal of Neuroscience Methods 169 (2008) 417-424

The coincidence factor measures how well a model predicts spike times
compared to experimental recordings, normalized by chance level.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

# Optional JAX support
try:
    import jax.numpy as jnp
    from jax.typing import ArrayLike
except ImportError:
    jnp = None
    ArrayLike = np.ndarray


@dataclass
class CoincidenceResult:
    """Results from coincidence factor calculation."""

    gamma: float  # The coincidence factor Γ
    n_coincidences: int  # Number of coincident spikes
    n_data: int  # Number of spikes in reference train
    n_model: int  # Number of spikes in predicted train
    expected_coincidences: float  # Expected coincidences by chance
    firing_rate_model: float  # Firing rate of model (Hz)
    delta_ms: float  # Coincidence window used (ms)

    def __repr__(self):
        return (
            f"CoincidenceResult(\n"
            f"  gamma={self.gamma:.4f},\n"
            f"  n_coincidences={self.n_coincidences},\n"
            f"  n_data={self.n_data}, n_model={self.n_model},\n"
            f"  expected_by_chance={self.expected_coincidences:.2f},\n"
            f"  firing_rate_model={self.firing_rate_model:.2f} Hz\n"
            f")"
        )


def detect_spike_times(
    voltage_trace: ArrayLike,
    dt: float,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Detect spike times from a voltage trace using threshold crossing.

    Parameters
    ----------
    voltage_trace : array
        Voltage trace in mV
    dt : float
        Time step in ms
    threshold : float
        Spike detection threshold in mV (default: 0 mV)

    Returns
    -------
    spike_times : np.ndarray
        Array of spike times in ms
    """
    voltage = np.asarray(voltage_trace)

    # Find threshold crossings (from below)
    above_threshold = voltage > threshold
    crossings = np.diff(above_threshold.astype(int)) > 0

    # Get indices where crossings occur (add 1 because diff shifts by 1)
    spike_indices = np.where(crossings)[0] + 1

    # Convert to times
    spike_times = spike_indices * dt

    return spike_times


def detect_spike_times_from_spike_trace(
    spike_trace: ArrayLike,
    dt: float,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Detect spike times from a binary/continuous spike indicator trace.

    Parameters
    ----------
    spike_trace : array
        Spike indicator trace (1 = spike, 0 = no spike, or continuous)
    dt : float
        Time step in ms
    threshold : float
        Threshold for spike detection (default: 0.5)

    Returns
    -------
    spike_times : np.ndarray
        Array of spike times in ms
    """
    spikes = np.asarray(spike_trace)

    # Find where spikes occur
    spike_indices = np.where(spikes > threshold)[0]

    # Remove consecutive indices (keep only first of each spike)
    if len(spike_indices) > 1:
        gaps = np.diff(spike_indices) > 1
        first_of_spike = np.concatenate([[True], gaps])
        spike_indices = spike_indices[first_of_spike]

    # Convert to times
    spike_times = spike_indices * dt

    return spike_times


def count_coincidences(
    spike_times_data: np.ndarray,
    spike_times_model: np.ndarray,
    delta_ms: float = 2.0,
) -> int:
    """
    Count the number of coincident spikes between two spike trains.

    A coincidence occurs when a model spike is within ±delta_ms of a data spike.
    Each data spike can only be matched once (greedy matching).

    Parameters
    ----------
    spike_times_data : np.ndarray
        Spike times from experimental data (ms)
    spike_times_model : np.ndarray
        Spike times from model prediction (ms)
    delta_ms : float
        Coincidence window in ms (default: 2.0 ms, as in Jolivet 2008)

    Returns
    -------
    n_coincidences : int
        Number of coincident spike pairs
    """
    if len(spike_times_data) == 0 or len(spike_times_model) == 0:
        return 0

    # Track which data spikes have been matched
    matched_data = np.zeros(len(spike_times_data), dtype=bool)
    n_coincidences = 0

    for t_model in spike_times_model:
        # Find data spikes within the coincidence window
        time_diffs = np.abs(spike_times_data - t_model)
        within_window = (time_diffs <= delta_ms) & (~matched_data)

        if np.any(within_window):
            # Match to the closest unmatched data spike
            candidates = np.where(within_window)[0]
            best_match = candidates[np.argmin(time_diffs[candidates])]
            matched_data[best_match] = True
            n_coincidences += 1

    return n_coincidences


def coincidence_factor(
    spike_times_data: np.ndarray,
    spike_times_model: np.ndarray,
    duration_ms: float,
    delta_ms: float = 2.0,
) -> CoincidenceResult:
    """
    Calculate the coincidence factor Γ between experimental and model spike trains.

    From Jolivet et al. (2008), Eq. 2:

        Γ = (N_coinc - <N_coinc>) / (0.5 * (N_data + N_model)) * (1/N)

    where:
        - N_coinc: number of coincidences within ±Δ
        - <N_coinc> = 2 * f * Δ * N_data: expected coincidences by chance
        - f: firing rate of model
        - N = 1 - 2*f*Δ: normalization factor

    Γ = 1 means perfect prediction
    Γ = 0 means chance level (Poisson process with same rate)
    Γ < 0 means worse than chance

    Parameters
    ----------
    spike_times_data : np.ndarray
        Spike times from experimental data (ms)
    spike_times_model : np.ndarray
        Spike times from model prediction (ms)
    duration_ms : float
        Total duration of the recording/simulation (ms)
    delta_ms : float
        Coincidence window in ms (default: 2.0 ms)

    Returns
    -------
    result : CoincidenceResult
        Dataclass containing Γ and related statistics
    """
    n_data = len(spike_times_data)
    n_model = len(spike_times_model)

    # Handle edge cases
    if n_data == 0 and n_model == 0:
        return CoincidenceResult(
            gamma=1.0,  # Both empty = perfect match
            n_coincidences=0,
            n_data=0,
            n_model=0,
            expected_coincidences=0.0,
            firing_rate_model=0.0,
            delta_ms=delta_ms,
        )

    if n_data == 0 or n_model == 0:
        return CoincidenceResult(
            gamma=0.0,  # One empty, one not = no match
            n_coincidences=0,
            n_data=n_data,
            n_model=n_model,
            expected_coincidences=0.0,
            firing_rate_model=(
                n_model / (duration_ms / 1000) if duration_ms > 0 else 0.0
            ),
            delta_ms=delta_ms,
        )

    # Count actual coincidences
    n_coinc = count_coincidences(spike_times_data, spike_times_model, delta_ms)

    # Model firing rate (Hz)
    duration_s = duration_ms / 1000.0
    f_model = n_model / duration_s

    # Expected coincidences by chance (Poisson process)
    # <N_coinc> = 2 * f * Δ * N_data
    delta_s = delta_ms / 1000.0
    expected_coinc = 2 * f_model * delta_s * n_data

    # Normalization factor: N = 1 - 2*f*Δ
    normalization = 1 - 2 * f_model * delta_s

    # Avoid division by zero
    if normalization <= 0:
        # This happens when firing rate is very high (f > 1/(2*Δ) = 250 Hz for Δ=2ms)
        # In this case, Γ is not well-defined
        normalization = 1e-10

    denominator = 0.5 * (n_data + n_model)
    if denominator == 0:
        denominator = 1e-10

    # Calculate Γ
    gamma = ((n_coinc - expected_coinc) / denominator) * (1 / normalization)

    # Clip to reasonable range (can exceed 1 in edge cases)
    gamma = float(np.clip(gamma, -1.0, 1.0))

    return CoincidenceResult(
        gamma=gamma,
        n_coincidences=n_coinc,
        n_data=n_data,
        n_model=n_model,
        expected_coincidences=expected_coinc,
        firing_rate_model=f_model,
        delta_ms=delta_ms,
    )


def coincidence_factor_from_traces(
    voltage_data: ArrayLike,
    voltage_model: ArrayLike,
    dt: float,
    delta_ms: float = 2.0,
    threshold: float = 0.0,
) -> CoincidenceResult:
    """
    Calculate coincidence factor directly from voltage traces.

    Parameters
    ----------
    voltage_data : array
        Experimental voltage trace (mV)
    voltage_model : array
        Model voltage trace (mV)
    dt : float
        Time step (ms)
    delta_ms : float
        Coincidence window (ms)
    threshold : float
        Spike detection threshold (mV)

    Returns
    -------
    result : CoincidenceResult
    """
    # Detect spikes
    spike_times_data = detect_spike_times(voltage_data, dt, threshold)
    spike_times_model = detect_spike_times(voltage_model, dt, threshold)

    # Calculate duration
    duration_ms = len(voltage_data) * dt

    return coincidence_factor(
        spike_times_data, spike_times_model, duration_ms, delta_ms
    )


def intrinsic_reliability(
    spike_times_trials: List[np.ndarray],
    duration_ms: float,
    delta_ms: float = 2.0,
) -> float:
    """
    Calculate the intrinsic reliability Γ̂ of experimental recordings.

    This measures the trial-to-trial reliability of the neuron itself,
    computed as the average Γ between all pairs of trial repetitions.

    From Jolivet et al. (2008): Used to normalize Γ_A.

    Parameters
    ----------
    spike_times_trials : List[np.ndarray]
        List of spike time arrays, one per trial repetition
    duration_ms : float
        Duration of each trial (ms)
    delta_ms : float
        Coincidence window (ms)

    Returns
    -------
    gamma_hat : float
        Intrinsic reliability (average Γ between trial pairs)
    """
    n_trials = len(spike_times_trials)

    if n_trials < 2:
        return 1.0  # Can't measure reliability with < 2 trials

    # Compute Γ for all pairs
    gamma_pairs = []
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            result = coincidence_factor(
                spike_times_trials[i],
                spike_times_trials[j],
                duration_ms,
                delta_ms,
            )
            gamma_pairs.append(result.gamma)

    return float(np.mean(gamma_pairs))


def global_performance(
    gamma_values: List[float],
    intrinsic_reliabilities: List[float],
) -> float:
    """
    Calculate the global performance Γ_A.

    From Jolivet et al. (2008), Eq. 3:

        Γ_A = (1/K) * Σ_k (Γ_k / Γ̂_k)

    This normalizes the coincidence factor by the intrinsic reliability
    of the neuron, so that Γ_A = 1 means the model is as good as the
    neuron's own trial-to-trial reliability.

    Parameters
    ----------
    gamma_values : List[float]
        Coincidence factors for each stimulus
    intrinsic_reliabilities : List[float]
        Intrinsic reliability Γ̂ for each stimulus

    Returns
    -------
    gamma_A : float
        Global performance measure
    """
    assert len(gamma_values) == len(intrinsic_reliabilities)

    K = len(gamma_values)
    if K == 0:
        return 0.0

    # Avoid division by zero for unreliable stimuli
    ratios = []
    for gamma, gamma_hat in zip(gamma_values, intrinsic_reliabilities):
        if gamma_hat > 0.1:  # Only include reasonably reliable stimuli
            ratios.append(gamma / gamma_hat)

    if len(ratios) == 0:
        return 0.0

    return float(np.mean(ratios))


# =============================================================================
# Example usage and tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Coincidence Factor (Γ) - Test Suite")
    print("=" * 60)

    # Test 1: Perfect prediction
    print("\n--- Test 1: Perfect prediction ---")
    spike_times = np.array([10.0, 50.0, 100.0, 150.0, 200.0])
    result = coincidence_factor(spike_times, spike_times, duration_ms=250.0)
    print(f"Same spike trains: Γ = {result.gamma:.4f} (should be ~1.0)")
    print(result)

    # Test 2: Slightly shifted spikes (within window)
    print("\n--- Test 2: Shifted by 1ms (within Δ=2ms window) ---")
    spike_times_shifted = spike_times + 1.0  # Shift by 1ms
    result = coincidence_factor(spike_times, spike_times_shifted, duration_ms=250.0)
    print(f"Shifted by 1ms: Γ = {result.gamma:.4f} (should be ~1.0)")

    # Test 3: Shifted outside window
    print("\n--- Test 3: Shifted by 5ms (outside Δ=2ms window) ---")
    spike_times_far = spike_times + 5.0  # Shift by 5ms
    result = coincidence_factor(spike_times, spike_times_far, duration_ms=250.0)
    print(f"Shifted by 5ms: Γ = {result.gamma:.4f} (should be ~0 or negative)")

    # Test 4: Random spikes (should be near 0)
    print("\n--- Test 4: Random spike trains ---")
    np.random.seed(42)
    random_spikes_1 = np.sort(np.random.uniform(0, 1000, 20))
    random_spikes_2 = np.sort(np.random.uniform(0, 1000, 20))
    result = coincidence_factor(random_spikes_1, random_spikes_2, duration_ms=1000.0)
    print(f"Random spikes: Γ = {result.gamma:.4f} (should be near 0)")
    print(result)

    # Test 5: Different spike counts
    print("\n--- Test 5: Model predicts twice as many spikes ---")
    data_spikes = np.array([50.0, 150.0, 250.0, 350.0, 450.0])
    model_spikes = np.array(
        [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0]
    )
    result = coincidence_factor(data_spikes, model_spikes, duration_ms=550.0)
    print(f"2x spikes: Γ = {result.gamma:.4f}")
    print(result)

    # Test 6: From voltage traces
    print("\n--- Test 6: From voltage traces ---")
    dt = 0.025  # ms
    t = np.arange(0, 500, dt)

    # Create fake voltage trace with spikes
    voltage_data = -70 + 5 * np.sin(2 * np.pi * t / 100)
    spike_times_true = [50, 150, 250, 350, 450]
    for st in spike_times_true:
        idx = int(st / dt)
        if idx < len(voltage_data) - 10:
            voltage_data[idx : idx + 5] = 20  # Spike

    # Model trace (slightly different timing)
    voltage_model = -70 + 5 * np.sin(2 * np.pi * t / 100)
    spike_times_model_true = [51, 149, 252, 348, 451]  # Shifted slightly
    for st in spike_times_model_true:
        idx = int(st / dt)
        if idx < len(voltage_model) - 10:
            voltage_model[idx : idx + 5] = 20

    result = coincidence_factor_from_traces(voltage_data, voltage_model, dt)
    print(f"From traces: Γ = {result.gamma:.4f}")
    print(result)

    # Test 7: Intrinsic reliability
    print("\n--- Test 7: Intrinsic reliability (trial-to-trial) ---")
    trials = [
        np.array([50.0, 150.0, 250.0, 350.0]),
        np.array([51.0, 149.0, 251.0, 349.0]),  # Slightly jittered
        np.array([49.0, 152.0, 248.0, 352.0]),  # Slightly jittered
        np.array([50.5, 150.5, 250.5, 350.5]),  # Slightly jittered
    ]
    gamma_hat = intrinsic_reliability(trials, duration_ms=400.0)
    print(f"Intrinsic reliability: Γ̂ = {gamma_hat:.4f}")

    print("\n" + "=" * 60)
    print("Reference values from Jolivet et al. (2008):")
    print("  - Well-fitted aEIF: Γ_A ≈ 0.82-0.83")
    print("  - Good model: Γ_A ≈ 0.7+")
    print("  - Chance level: Γ ≈ 0")
    print("=" * 60)
