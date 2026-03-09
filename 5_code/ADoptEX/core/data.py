"""
Data loading and preprocessing utilities.

This module provides functions for:
- Loading experimental voltage/current traces
- Finding stimulation windows
- Detecting spikes from voltage traces
- Loading multiple traces for multi-trace training
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TraceData:
    """Container for experimental trace data.

    Traces are either simulated or read form data files.
    We use experimental data from Johnson2020, adapted by Alexander Kozlov.

    Attributes:
        time: Time array in ms
        voltage: Voltage trace in mV
        current: Current trace in pA
        dt_ms: Time step in ms
        spike_times: Detected spike times in ms
        stim_start_idx: Index where stimulation starts
        stim_end_idx: Index where stimulation ends
        stim_current_pA: Mean stimulation current in pA
    """

    time: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    dt_ms: float
    spike_times: np.ndarray
    stim_start_idx: int
    stim_end_idx: int
    stim_current_pA: float

    @property
    def n_samples(self) -> int:
        """Number of samples in the trace."""
        return len(self.time)

    @property
    def duration_ms(self) -> float:
        """Total duration of the trace in ms."""
        return self.time[-1] - self.time[0]

    @property
    def stim_duration_ms(self) -> float:
        """Duration of stimulation in ms."""
        return (self.stim_end_idx - self.stim_start_idx) * self.dt_ms

    @property
    def n_spikes(self) -> int:
        """Number of detected spikes."""
        return len(self.spike_times)

    def get_stim_window(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get time, voltage, and current during stimulation window.

        Returns:
            Tuple of (time, voltage, current) arrays for stim window only
        """
        s, e = self.stim_start_idx, self.stim_end_idx
        return self.time[s:e], self.voltage[s:e], self.current[s:e]


def detect_spikes(
    voltage: np.ndarray,
    dt_ms: float,
    threshold_mv: float = -20.0,
    min_interval_ms: float = 2.0,
) -> np.ndarray:
    """
    Detect spike times from a voltage trace using threshold crossing.

    Args:
        voltage: Voltage trace in mV
        dt_ms: Time step in ms
        threshold_mv: Spike detection threshold in mV
        min_interval_ms: Minimum interval between spikes in ms (to avoid double-counting)

    Returns:
        Array of spike times in ms

    Example:
        >>> spike_times = detect_spikes(voltage, dt_ms=0.1, threshold_mv=-20.0)
    """
    # Find threshold crossings (rising edge)
    above_threshold = voltage > threshold_mv
    crossings = np.diff(above_threshold.astype(int)) > 0
    spike_indices = np.where(crossings)[0] + 1  # +1 because diff shifts by 1

    # Filter out spikes that are too close together
    if len(spike_indices) > 1:
        min_interval_samples = int(min_interval_ms / dt_ms)
        filtered_indices = [spike_indices[0]]
        for idx in spike_indices[1:]:
            if idx - filtered_indices[-1] >= min_interval_samples:
                filtered_indices.append(idx)
        spike_indices = np.array(filtered_indices)

    spike_times = spike_indices * dt_ms
    return spike_times


def find_stim_window(
    current: np.ndarray,
    threshold_pA: float = 100.0,
) -> tuple[int, int]:
    """
    Find the start and end indices of the stimulation period.

    The stimulation window is defined as the contiguous region where
    current exceeds the threshold.

    Args:
        current: Current trace in pA
        threshold_pA: Minimum current to consider as stimulation

    Returns:
        Tuple of (start_index, end_index)

    Example:
        >>> start, end = find_stim_window(current, threshold_pA=100.0)
        >>> stim_duration_ms = (end - start) * dt_ms
    """
    above = current > threshold_pA
    indices = np.where(above)[0]
    if len(indices) == 0:
        return 0, len(current)
    return int(indices[0]), int(indices[-1])


def load_trace(
    voltage_path: str | Path,
    current_path: str | Path,
    junction_potential_mv: float = 35.0,
    spike_threshold_mv: float = 0.0,
    stim_threshold_pA: float = 100.0,
) -> TraceData:
    """
    Load experimental voltage and current traces from files.

    Expects two-column files with format: [time, value]
    Time should be in ms, voltage in mV, current in pA.

    Args:
        voltage_path: Path to voltage data file
        current_path: Path to current data file
        junction_potential_mv: Liquid junction potential correction in mV
        spike_threshold_mv: Threshold for spike detection in mV
        stim_threshold_pA: Threshold for stimulation detection in pA

    Returns:
        TraceData object with all trace information

    Example:
        >>> data = load_trace("voltage.dat", "current.dat")
        >>> print(f"Loaded {data.n_samples} samples, {data.n_spikes} spikes")
    """
    # Load voltage data
    v_data = np.loadtxt(voltage_path)
    time = v_data[:, 0]
    voltage = v_data[:, 1] - junction_potential_mv

    # Load current data
    i_data = np.loadtxt(current_path)
    current = i_data[:, 1]

    # Calculate time step
    dt_ms = float(np.mean(np.diff(time)))

    # Find stimulation window
    stim_start_idx, stim_end_idx = find_stim_window(current, stim_threshold_pA)
    stim_current_pA = float(np.mean(current[stim_start_idx:stim_end_idx]))

    # Detect spikes
    spike_times = detect_spikes(voltage, dt_ms, spike_threshold_mv)

    return TraceData(
        time=time,
        voltage=voltage,
        current=current,
        dt_ms=dt_ms,
        spike_times=spike_times,
        stim_start_idx=stim_start_idx,
        stim_end_idx=stim_end_idx,
        stim_current_pA=stim_current_pA,
    )


def load_multiple_traces(
    data_dir: str | Path,
    trace_files: dict[str, tuple[str, str]],
    junction_potential_mv: float = 0.0,
    spike_threshold_mv: float = -20.0,
    stim_threshold_pA: float = 100.0,
) -> dict[str, TraceData]:
    """
    Load multiple voltage/current trace pairs.

    Useful for multi-trace training where the same parameters must
    match across different stimulation levels.

    Args:
        data_dir: Base directory containing trace files
        trace_files: Dictionary mapping trace name to (voltage_file, current_file)
        junction_potential_mv: Liquid junction potential correction
        spike_threshold_mv: Threshold for spike detection
        stim_threshold_pA: Threshold for stimulation detection

    Returns:
        Dictionary mapping trace name to TraceData

    Example:
        >>> traces = load_multiple_traces(
        ...     data_dir="/path/to/data",
        ...     trace_files={
        ...         "low_current": ("v_low.dat", "i_low.dat"),
        ...         "high_current": ("v_high.dat", "i_high.dat"),
        ...     }
        ... )
    """
    data_dir = Path(data_dir)
    traces = {}

    for name, (v_file, i_file) in trace_files.items():
        traces[name] = load_trace(
            voltage_path=data_dir / v_file,
            current_path=data_dir / i_file,
            junction_potential_mv=junction_potential_mv,
            spike_threshold_mv=spike_threshold_mv,
            stim_threshold_pA=stim_threshold_pA,
        )

    return traces


def store_trace(
    data: TraceData,
    voltage_path: str | Path,
    current_path: str | Path
) -> None:

    v_data = np.array([data.time, data.voltage])
    c_data = np.array([data.time, data.current])
    np.savetxt(voltage_path, v_data)
    np.savetxt(current_path, c_data)


def crop_to_stim_window(
    data: TraceData,
    max_duration_ms: float | None = None,
    padding_ms_before: float = 0.0,
    padding_ms_after: float = 0.0,
) -> TraceData:
    """
    Crop trace data to the stimulation window with optional pre/post padding.

    Useful for reducing data size and focusing on the relevant period.
    The returned trace contains a pre-stimulus window, stimulus window,
    and post-stimulus window.

    Processing order:
        1. ``max_duration_ms`` caps the **stimulation window** (stim_start to stim_end).
        2. ``padding_ms_before/after`` extends **beyond** the (possibly capped) stim window.

    The total trace length is therefore up to
    ``padding_before + min(stim_duration, max_duration) + padding_after``.

    Args:
        data: Original TraceData
        max_duration_ms: Maximum stimulation duration to keep (from stim start).
            Only shortens the stim window; padding is added on top. If None, keeps
            the full stimulation window.
        padding_ms_before: Extra time to include before stimulation starts (pre-stim window)
        padding_ms_after: Extra time to include after stimulation ends (post-stim window)

    Returns:
        New TraceData cropped to [stim_start - padding_before, stim_end + padding_after]

    Example:
        >>> cropped = crop_to_stim_window(data, padding_ms_before=100.0, padding_ms_after=50.0)
    """
    stim_start = data.stim_start_idx
    stim_end = data.stim_end_idx

    # Apply max duration limit to stim end
    if max_duration_ms is not None:
        max_samples = int(max_duration_ms / data.dt_ms)
        stim_end = min(stim_end, stim_start + max_samples)

    # Compute crop boundaries with padding
    pad_before_samples = (
        int(padding_ms_before / data.dt_ms) if padding_ms_before > 0 else 0
    )
    pad_after_samples = (
        int(padding_ms_after / data.dt_ms) if padding_ms_after > 0 else 0
    )

    crop_start = max(stim_start - pad_before_samples, 0)
    crop_end = min(stim_end + pad_after_samples, len(data.time))

    # Crop arrays
    time = data.time[crop_start:crop_end] - data.time[crop_start]  # Reset time to 0
    voltage = data.voltage[crop_start:crop_end]
    current = data.current[crop_start:crop_end]

    # Recalculate spike times relative to new time origin
    spike_times = data.spike_times - data.time[crop_start]
    # Keep only spikes within the new window
    spike_times = spike_times[(spike_times >= 0) & (spike_times < time[-1])]

    # Stim indices relative to the cropped window
    new_stim_start = stim_start - crop_start
    new_stim_end = stim_end - crop_start

    return TraceData(
        time=time,
        voltage=voltage,
        current=current,
        dt_ms=data.dt_ms,
        spike_times=spike_times,
        stim_start_idx=new_stim_start,
        stim_end_idx=new_stim_end,
        stim_current_pA=data.stim_current_pA,
    )
