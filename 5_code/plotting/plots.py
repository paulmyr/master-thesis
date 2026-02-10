"""
Plotting functions for AdEx simulation and training.

Provides visualization for:
- Voltage traces and spike rasters
- Brian2 vs Jaxley comparisons
- Training progress and results
- Model fitting evaluation
"""

import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, "..")
from core.simulation import SimulationResult


def plot_voltage_trace(
    result: SimulationResult,
    ax: plt.Axes | None = None,
    color: str = "b",
    label: str | None = None,
    show_spikes: bool = True,
    spike_color: str | None = None,
    threshold_mv: float | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot a voltage trace from simulation result.

    Args:
        result: SimulationResult from simulate_jaxley or simulate_brian2
        ax: Matplotlib axes (creates new figure if None)
        color: Line color
        label: Legend label
        show_spikes: If True, mark spike times with vertical lines
        spike_color: Color for spike markers (defaults to line color)
        threshold_mv: If provided, draw horizontal threshold line
        **kwargs: Additional arguments passed to ax.plot

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(result.time, result.voltage, color=color, label=label, **kwargs)

    if show_spikes and len(result.spike_times) > 0:
        spike_c = spike_color or color
        for st in result.spike_times:
            ax.axvline(st, color=spike_c, linestyle=":", alpha=0.5)

    if threshold_mv is not None:
        ax.axhline(
            threshold_mv, color="r", linestyle="--", alpha=0.3, label="Threshold"
        )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.grid(True, alpha=0.3)

    return ax


def plot_spike_raster(
    spike_times_list: Sequence[np.ndarray],
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot spike raster for multiple spike trains.

    Args:
        spike_times_list: List of spike time arrays
        labels: Labels for each spike train
        colors: Colors for each spike train
        ax: Matplotlib axes
        **kwargs: Additional arguments passed to eventplot

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2))

    n_trains = len(spike_times_list)

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_trains))

    if labels is None:
        labels = [f"Train {i}" for i in range(n_trains)]

    for i, (spikes, color, label) in enumerate(zip(spike_times_list, colors, labels)):
        if len(spikes) > 0:
            ax.eventplot(
                [spikes], colors=[color], lineoffsets=i, linelengths=0.8, **kwargs
            )

    ax.set_yticks(range(n_trains))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (ms)")
    ax.grid(True, alpha=0.3, axis="x")

    return ax


def plot_comparison(
    result_brian2: SimulationResult,
    result_jaxley: SimulationResult,
    title: str = "Brian2 vs Jaxley Comparison",
    figsize: tuple = (12, 8),
) -> Figure:
    """
    Plot comparison between Brian2 and Jaxley simulations.

    Shows voltage overlay, adaptation current, and spike raster.

    Args:
        result_brian2: SimulationResult from simulate_brian2
        result_jaxley: SimulationResult from simulate_jaxley
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(title)

    # Voltage overlay
    ax_v = axes[0]
    ax_v.plot(
        result_brian2.time, result_brian2.voltage, "b-", label="Brian2", alpha=0.8
    )
    ax_v.plot(
        result_jaxley.time, result_jaxley.voltage, "g--", label="Jaxley", alpha=0.8
    )
    ax_v.set_ylabel("Voltage (mV)")
    ax_v.legend()
    ax_v.grid(True, alpha=0.3)

    # Adaptation current
    ax_w = axes[1]
    ax_w.plot(result_brian2.time, result_brian2.w, "b-", label="Brian2", alpha=0.8)
    ax_w.plot(result_jaxley.time, result_jaxley.w, "g--", label="Jaxley", alpha=0.8)
    ax_w.set_ylabel("w (pA)")
    ax_w.legend()
    ax_w.grid(True, alpha=0.3)

    # Spike raster
    ax_s = axes[2]
    ax_s.set_xlim(result_jaxley.time[0], result_jaxley.time[-1])
    plot_spike_raster(
        [result_brian2.spike_times, result_jaxley.spike_times],
        labels=["Brian2", "Jaxley"],
        colors=["b", "g"],
        ax=ax_s,
    )
    ax_s.set_xlabel("Time (ms)")

    plt.tight_layout()
    return fig


def plot_combined_comparison(
    results: dict[str, tuple[SimulationResult, SimulationResult]],
    figsize: tuple = (14, 8),
) -> Figure:
    """
    Plot multiple Brian2 vs Jaxley comparisons in a grid.

    Args:
        results: Dict mapping condition name to (brian2_result, jaxley_result)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_conditions = len(results)
    fig, axes = plt.subplots(3, n_conditions, figsize=figsize, squeeze=False)

    for col, (name, (result_brian2, result_jaxley)) in enumerate(results.items()):
        # Voltage overlay
        ax_v = axes[0, col]
        ax_v.plot(
            result_brian2.time,
            result_brian2.voltage,
            "b-",
            label="Brian2",
            alpha=0.8,
            linewidth=1,
        )
        ax_v.plot(
            result_jaxley.time,
            result_jaxley.voltage,
            "g--",
            label="Jaxley",
            alpha=0.8,
            linewidth=1,
        )
        ax_v.set_title(name)
        ax_v.set_ylabel("Voltage (mV)" if col == 0 else "")
        ax_v.grid(True, alpha=0.3)
        if col == 0:
            ax_v.legend(fontsize=8)

        # Adaptation current
        ax_w = axes[1, col]
        ax_w.plot(result_brian2.time, result_brian2.w, "b-", alpha=0.8, linewidth=1)
        ax_w.plot(result_jaxley.time, result_jaxley.w, "g--", alpha=0.8, linewidth=1)
        ax_w.set_ylabel("w (pA)" if col == 0 else "")
        ax_w.grid(True, alpha=0.3)

        # Spike raster
        ax_s = axes[2, col]
        if len(result_brian2.spike_times) > 0:
            ax_s.eventplot(
                [result_brian2.spike_times],
                colors="b",
                lineoffsets=1.5,
                linelengths=0.8,
            )
        if len(result_jaxley.spike_times) > 0:
            ax_s.eventplot(
                [result_jaxley.spike_times],
                colors="g",
                lineoffsets=0.5,
                linelengths=0.8,
            )
        ax_s.set_ylim(0, 2.5)
        ax_s.set_yticks([0.5, 1.5])
        ax_s.set_yticklabels(["Jaxley", "Brian2"] if col == 0 else ["", ""])
        ax_s.set_xlabel("Time (ms)")
        ax_s.set_xlim(result_jaxley.time[0], result_jaxley.time[-1])
        ax_s.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def plot_training_results(
    loss_history: list[float],
    time_per_epoch: list[float] | None = None,
    figsize: tuple = (10, 4),
) -> Figure:
    """
    Plot training loss curve.

    Args:
        loss_history: List of loss values per epoch
        time_per_epoch: Optional list of time per epoch in seconds
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_plots = 2 if time_per_epoch else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    # Loss curve
    ax_loss = axes[0]
    ax_loss.plot(loss_history, "b-", linewidth=1)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_yscale("log")

    # Time per epoch
    if time_per_epoch:
        ax_time = axes[1]
        ax_time.plot(time_per_epoch, "g-", linewidth=1)
        ax_time.set_xlabel("Epoch")
        ax_time.set_ylabel("Time (s)")
        ax_time.set_title("Time per Epoch")
        ax_time.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_fit_results(
    exp_voltage: np.ndarray,
    exp_time: np.ndarray,
    sim_result: SimulationResult,
    exp_spike_times: np.ndarray | None = None,
    figsize: tuple = (12, 8),
) -> Figure:
    """
    Plot experimental vs simulated traces after fitting.

    Args:
        exp_voltage: Experimental voltage trace
        exp_time: Experimental time array
        sim_result: Simulation result with fitted parameters
        exp_spike_times: Optional experimental spike times
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Voltage comparison
    ax_v = axes[0]
    ax_v.plot(
        exp_time, exp_voltage, "gray", alpha=0.7, label="Experimental", linewidth=1
    )
    ax_v.plot(
        sim_result.time, sim_result.voltage, "b-", label="Fitted model", linewidth=1
    )
    ax_v.set_ylabel("Voltage (mV)")
    ax_v.set_title("Voltage Trace Comparison")
    ax_v.legend()
    ax_v.grid(True, alpha=0.3)

    # Adaptation current
    ax_w = axes[1]
    ax_w.plot(sim_result.time, sim_result.w, "b-", linewidth=1)
    ax_w.set_ylabel("w (pA)")
    ax_w.set_title("Simulated Adaptation Current")
    ax_w.grid(True, alpha=0.3)

    # Spike raster
    ax_s = axes[2]
    spike_trains = [sim_result.spike_times]
    labels = ["Simulated"]
    colors = ["b"]

    if exp_spike_times is not None:
        spike_trains.append(exp_spike_times)
        labels.append("Experimental")
        colors.append("gray")

    plot_spike_raster(spike_trains, labels=labels, colors=colors, ax=ax_s)
    ax_s.set_xlabel("Time (ms)")

    n_exp = len(exp_spike_times) if exp_spike_times is not None else 0
    n_sim = sim_result.n_spikes
    ax_s.set_title(f"Spike Times (Exp: {n_exp}, Sim: {n_sim})")

    plt.tight_layout()
    return fig


def plot_coincidence_evaluation(
    gamma_values: dict[str, float],
    reference_gamma: float = 0.82,
    figsize: tuple = (8, 5),
) -> Figure:
    """
    Plot bar chart of coincidence factor values.

    Args:
        gamma_values: Dict mapping parameter set name to gamma value
        reference_gamma: Reference value for well-fitted model (dashed line)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = list(gamma_values.keys())
    values = list(gamma_values.values())

    colors = [
        "green" if v > reference_gamma else "orange" if v > 0.5 else "red"
        for v in values
    ]

    bars = ax.bar(names, values, color=colors, alpha=0.7, edgecolor="black")

    # Reference lines
    ax.axhline(
        reference_gamma,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Reference (Γ={reference_gamma})",
    )
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)

    ax.set_ylabel("Coincidence Factor (Γ)")
    ax.set_title("Coincidence Factor Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig
