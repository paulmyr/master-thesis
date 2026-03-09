"""
Core simulation functions for AdEx neuron models.

This module provides unified interfaces for:
- Creating AdEx cells in Jaxley (with or without surrogate gradients)
- Running simulations in Jaxley
- Running simulations in Brian2 (for verification)

All simulation functions return a consistent SimulationResult dataclass.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import jax.numpy as jnp
import jaxley as jx
import numpy as np
from jaxley.channels import AdEx, AdExSurrogate


@dataclass
class SimulationResult:
    """Result from an AdEx simulation.

    Attributes:
        time: Time array in ms
        voltage: Membrane potential in mV
        w: Adaptation current in pA
        spikes: Spike indicator trace (0 or 1)
        spike_times: Array of spike times in ms
    """

    time: np.ndarray
    voltage: np.ndarray
    w: np.ndarray
    spikes: np.ndarray
    spike_times: np.ndarray

    @property
    def n_spikes(self) -> int:
        """Number of spikes detected."""
        return len(self.spike_times)

    @property
    def firing_rate_hz(self) -> float:
        """Firing rate in Hz."""
        duration_s = (self.time[-1] - self.time[0]) / 1000.0
        return self.n_spikes / duration_s if duration_s > 0 else 0.0


SurrogateType: TypeAlias = Literal["sigmoid", "exponential", "superspike"]


def geometry_for_capacitance(
    C_pF: float, specific_capacitance: float = 1.0
) -> tuple[float, float]:
    """
    Calculate cylindrical cell geometry to achieve target total capacitance.

    Jaxley expects specific capacitance (uF/cm^2) and cell geometry.
    AdEx models specify absolute capacitance (pF). This function computes
    a fictitious cylindrical geometry that yields the desired total capacitance.

    Args:
        C_pF: Target total membrane capacitance in picofarads (pF)
        specific_capacitance: Specific membrane capacitance in uF/cm^2 (default: 1.0)

    Returns:
        Tuple of (radius_um, length_um) for the cylindrical cell

    Example:
        >>> radius, length = geometry_for_capacitance(200.0)  # 200 pF
        >>> # Creates geometry with surface area = 200e-6 / 1.0 = 2e-4 cm^2
    """
    C_uF = C_pF * 1e-6
    area_cm2 = C_uF / specific_capacitance
    # For a cylinder: area = 2 * pi * r * L, with L = pi * r
    # area = 2 * pi^2 * r^2, so r = sqrt(area / (2 * pi^2))
    radius_cm = np.sqrt(area_cm2 / (2 * np.pi**2))
    radius_um = radius_cm * 1e4
    length_um = np.pi * radius_um
    return float(radius_um), float(length_um)


def create_adex_cell(
    params: dict,
    use_surrogate: bool = True,
    surrogate_type: SurrogateType = "sigmoid",
    surrogate_slope: float = 25.0,
    trainable: bool = False,
    trainable_params: list[str] | None = None,
    record: bool = True,
) -> jx.Cell:
    """
    Create a Jaxley cell with AdEx channel.

    This is the single source of truth for creating AdEx cells. It supports:
    - Standard AdEx (non-differentiable) or AdExSurrogate (differentiable)
    - Optional parameter training setup
    - Configurable recording

    Args:
        params: Dictionary of AdEx parameters. Required keys:
            - C_m: Membrane capacitance (pF)
            - g_L: Leak conductance (nS)
            - E_L: Leak reversal potential (mV)
            - v_T: Spike threshold (mV)
            - delta_T: Spike slope factor (mV)
            - v_reset: Reset potential (mV)
            - v_threshold: Detection threshold (mV)
            - tau_w: Adaptation time constant (ms)
            - a: Subthreshold adaptation (nS)
            - b: Spike-triggered adaptation (pA)
        use_surrogate: If True, use AdExSurrogate for differentiable spikes
        surrogate_type: Type of surrogate gradient function
        surrogate_slope: Steepness of surrogate gradient
        trainable: If True, make parameters trainable for optimization
        trainable_params: List of parameter names to make trainable.
            If None and trainable=True, uses default set:
            ['g_L', 'E_L', 'v_T', 'v_reset', 'tau_w', 'a', 'b']
        record: If True, set up recording for v, w, and spikes

    Returns:
        Configured Jaxley Cell object

    Example:
        >>> params = {'C_m': 200, 'g_L': 10, 'E_L': -70, ...}
        >>> cell = create_adex_cell(params, use_surrogate=True)
        >>> # For training:
        >>> cell = create_adex_cell(params, trainable=True)
    """
    # Calculate geometry from capacitance
    radius_um, length_um = geometry_for_capacitance(params["C_m"])

    # Create cell with geometry
    cell = jx.Cell()
    cell.set("radius", radius_um)
    cell.set("length", length_um)

    # Insert appropriate AdEx channel
    if use_surrogate:
        cell.insert(
            AdExSurrogate(
                surrogate_type=surrogate_type, surrogate_slope=surrogate_slope
            )
        )
        prefix = "AdEx"  # AdExSurrogate uses same prefix
    else:
        cell.insert(AdEx())
        prefix = "AdEx"

    # Set all AdEx parameters
    cell.set("capacitance", params["C_m"])
    cell.set(f"{prefix}_g_L", params["g_L"])
    cell.set(f"{prefix}_E_L", params["E_L"])
    cell.set(f"{prefix}_v_T", params["v_T"])
    cell.set(f"{prefix}_delta_T", params["delta_T"])
    cell.set(f"{prefix}_v_threshold", params["v_threshold"])
    cell.set(f"{prefix}_v_reset", params["v_reset"])
    cell.set(f"{prefix}_tau_w", params["tau_w"])
    cell.set(f"{prefix}_a", params["a"])
    cell.set(f"{prefix}_b", params["b"])

    # Set initial voltage
    cell.set("v", params["v_reset"])

    # Setup recording
    if record:
        cell.record("v")
        cell.record(f"{prefix}_w")
        cell.record(f"{prefix}_spikes")

    # Setup trainable parameters
    if trainable:
        if trainable_params is None:
            trainable_params = [
                "C_m",
                "g_L",
                "E_L",
                "v_T",
                "v_reset",
                "tau_w",
                "a",
                "b",
            ]

        for param_name in trainable_params:
            if param_name == "C_m":
                cell.make_trainable("capacitance")
            else:
                cell.make_trainable(f"{prefix}_{param_name}")

    return cell


def simulate_jaxley(
    params: dict,
    stim_current_pA: float,
    stim_duration_ms: float,
    dt_ms: float = 0.1,
    t_max_ms: float | None = None,
    stim_delay_ms: float = 0.0,
    use_surrogate: bool = False,
    surrogate_type: SurrogateType = "sigmoid",
    surrogate_slope: float = 25.0,
) -> SimulationResult:
    """
    Run AdEx simulation using Jaxley.

    Args:
        params: AdEx parameter dictionary (see create_adex_cell for required keys).
            Must also include 'I' key for compatibility, though stim_current_pA
            takes precedence if provided.
        stim_current_pA: Stimulation current in pA
        stim_duration_ms: Duration of current injection in ms
        dt_ms: Time step in ms
        t_max_ms: Total simulation time in ms. If None, uses stim_duration_ms + 100
        stim_delay_ms: Delay before stimulus onset in ms
        use_surrogate: If True, use differentiable AdExSurrogate
        surrogate_type: Type of surrogate gradient
        surrogate_slope: Steepness of surrogate gradient

    Returns:
        SimulationResult with time, voltage, w, spikes, and spike_times

    Example:
        >>> result = simulate_jaxley(params, stim_current_pA=500, stim_duration_ms=400)
        >>> print(f"Detected {result.n_spikes} spikes")
    """
    if t_max_ms is None:
        t_max_ms = stim_duration_ms + stim_delay_ms + 100.0

    # Create cell
    cell = create_adex_cell(
        params,
        use_surrogate=use_surrogate,
        surrogate_type=surrogate_type,
        surrogate_slope=surrogate_slope,
        trainable=False,
        record=True,
    )

    # Convert current: Jaxley expects nA, scaled by geometry
    # The geometry was computed for C_m, so we scale current by C_m
    I_nA = stim_current_pA * params["C_m"] / 1000.0

    # Create step current stimulus
    cell.stimulate(
        jx.step_current(
            stim_delay_ms, stim_delay_ms + stim_duration_ms, I_nA, dt_ms, t_max=t_max_ms
        )
    )

    # Run simulation
    results = jx.integrate(cell, delta_t=dt_ms, t_max=t_max_ms)

    # Extract results
    voltage = np.array(results[0]).flatten()
    w = np.array(results[1]).flatten()
    spikes = np.array(results[2]).flatten()

    # Create time array
    time = np.arange(len(voltage)) * dt_ms

    # Extract spike times
    spike_indices = np.where(spikes > 0.5)[0]
    spike_times = time[spike_indices] if len(spike_indices) > 0 else np.array([])

    return SimulationResult(
        time=time,
        voltage=voltage,
        w=w,
        spikes=spikes,
        spike_times=spike_times,
    )


def simulate_brian2(
    params: dict,
    stim_current_pA: float,
    stim_duration_ms: float,
    dt_ms: float = 0.1,
    t_max_ms: float | None = None,
    stim_delay_ms: float = 0.0,
) -> SimulationResult:
    """
    Run AdEx simulation using Brian2.

    This function has the same signature as simulate_jaxley for easy comparison.
    Used primarily for verification of the Jaxley implementation.

    Args:
        params: AdEx parameter dictionary (same as simulate_jaxley)
        stim_current_pA: Stimulation current in pA
        stim_duration_ms: Duration of current injection in ms
        dt_ms: Time step in ms
        t_max_ms: Total simulation time in ms
        stim_delay_ms: Delay before stimulus onset in ms

    Returns:
        SimulationResult with time, voltage, w, spikes, and spike_times

    Note:
        Requires brian2 to be installed. Import is done inside function
        to avoid import errors when brian2 is not available.
    """
    # Import brian2 inside function to make it optional
    from brian2 import (NeuronGroup, SpikeMonitor, StateMonitor, defaultclock,
                        ms, mV, nS, pA, pF, run)

    if t_max_ms is None:
        t_max_ms = stim_duration_ms + stim_delay_ms + 100.0

    # Convert parameters to Brian2 units
    C_m = params["C_m"] * pF
    g_L = params["g_L"] * nS
    E_L = params["E_L"] * mV
    v_T = params["v_T"] * mV
    delta_T = params["delta_T"] * mV
    v_threshold = params["v_threshold"] * mV
    v_reset = params["v_reset"] * mV
    tau_w = params["tau_w"] * ms
    a = params["a"] * nS
    b = params["b"] * pA
    I_input = stim_current_pA * pA

    # AdEx equations
    eqs = """
    dv/dt = (g_L*(E_L-v) + g_L*delta_T*exp((v-v_T)/delta_T) - w + I)/C_m : volt
    dw/dt = (a*(v-E_L) - w)/tau_w : amp
    I : amp
    """

    # Create neuron
    defaultclock.dt = dt_ms * ms
    neuron = NeuronGroup(
        1, eqs, threshold="v>v_threshold", reset="v=v_reset; w+=b", method="euler"
    )

    # Initial conditions
    neuron.v = params["v_reset"] * mV
    neuron.w = 0 * pA
    neuron.I = 0 * pA

    # Setup monitors
    mon_v = StateMonitor(neuron, "v", record=True)
    mon_w = StateMonitor(neuron, "w", record=True)
    spike_mon = SpikeMonitor(neuron)

    # Run simulation in phases
    if stim_delay_ms > 0:
        run(stim_delay_ms * ms)

    neuron.I = I_input
    run(stim_duration_ms * ms)

    neuron.I = 0 * pA
    remaining_ms = t_max_ms - stim_delay_ms - stim_duration_ms
    if remaining_ms > 0:
        run(remaining_ms * ms)

    # Extract results
    time = np.array(mon_v.t / ms)
    voltage = np.array(mon_v.v[0] / mV)
    w = np.array(mon_w.w[0] / pA)
    spike_times = np.array(spike_mon.t / ms)

    # Create spike trace (binary)
    spikes = np.zeros_like(voltage)
    for st in spike_times:
        idx = int(st / dt_ms)
        if 0 <= idx < len(spikes):
            spikes[idx] = 1.0

    return SimulationResult(
        time=time,
        voltage=voltage,
        w=w,
        spikes=spikes,
        spike_times=spike_times,
    )


def simulate_with_current_trace(
    params: dict,
    current_trace_pA: np.ndarray,
    dt_ms: float,
    use_surrogate: bool = True,
    surrogate_type: SurrogateType = "sigmoid",
    surrogate_slope: float = 25.0,
) -> SimulationResult:
    """
    Run AdEx simulation using an arbitrary current trace.

    Unlike simulate_jaxley which uses step current, this function
    accepts any current waveform (e.g., from experimental data).

    Args:
        params: AdEx parameter dictionary
        current_trace_pA: Current trace in pA, shape (T,)
        dt_ms: Time step in ms (must match current trace sampling)
        use_surrogate: If True, use differentiable AdExSurrogate
        surrogate_type: Type of surrogate gradient
        surrogate_slope: Steepness of surrogate gradient

    Returns:
        SimulationResult with time, voltage, w, spikes, and spike_times
    """
    # Create cell
    cell = create_adex_cell(
        params,
        use_surrogate=use_surrogate,
        surrogate_type=surrogate_type,
        surrogate_slope=surrogate_slope,
        trainable=False,
        record=True,
    )

    # Convert current trace: scale by C_m and convert to nA
    current_nA = current_trace_pA * params["C_m"] / 1000.0
    current_nA = jnp.array(current_nA)

    # Setup data stimulation
    t_max = len(current_trace_pA) * dt_ms
    data_stimuli = cell.comp(0).data_stimulate(current_nA, None)

    # Run simulation
    results = jx.integrate(cell, data_stimuli=data_stimuli, delta_t=dt_ms, t_max=t_max)

    # Extract results
    voltage = np.array(results[0]).flatten()
    w = np.array(results[1]).flatten()
    spikes = np.array(results[2]).flatten()

    # Create time array
    time = np.arange(len(voltage)) * dt_ms

    # Extract spike times
    spike_indices = np.where(spikes > 0.5)[0]
    spike_times = time[spike_indices] if len(spike_indices) > 0 else np.array([])

    return SimulationResult(
        time=time,
        voltage=voltage,
        w=w,
        spikes=spikes,
        spike_times=spike_times,
    )
