"""
Plotting utilities for trace data and simulation results.

Works with TraceData and SimulationResult from the core module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ADoptEX.core.data import TraceData
    from ADoptEX.core.parameters import ParamBounds
    from ADoptEX.core.simulation import SimulationResult
    from ADoptEX.evaluation.coincidence import CoincidenceResult
    from ADoptEX.training.trainer import TrainingResult


def trace_plot(
    trace: TraceData,
    show_spikes: bool = True,
    show_current: bool = True,
    highlight_stim: bool = True,
    figsize: tuple[float, float] = (12, 5),
) -> Figure:
    """Plot a TraceData object showing voltage, current, and detected spikes.

    Args:
        trace: TraceData to visualize
        show_spikes: Mark detected spike times with vertical lines
        show_current: Show current injection as a second subplot
        highlight_stim: Shade the stimulation window
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    n_rows = 2 if show_current else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True, squeeze=False)
    ax_v = axes[0, 0]

    # Voltage trace
    ax_v.plot(trace.time, trace.voltage, color="k", linewidth=0.6)
    ax_v.set_ylabel("Voltage (mV)")

    if highlight_stim:
        t_start = trace.time[trace.stim_start_idx]
        t_end = trace.time[min(trace.stim_end_idx, len(trace.time) - 1)]
        ax_v.axvspan(t_start, t_end, color="blue", alpha=0.06, label="Stim window")

    if show_spikes and len(trace.spike_times) > 0:
        for st in trace.spike_times:
            ax_v.axvline(st, color="r", linestyle=":", alpha=0.5, linewidth=0.7)
        # Invisible line for legend entry
        ax_v.axvline(
            trace.spike_times[0],
            color="r",
            linestyle=":",
            alpha=0,
            label=f"Spikes ({trace.n_spikes})",
        )

    ax_v.legend(loc="upper right", fontsize=8)
    ax_v.grid(True, alpha=0.2)

    # Current trace
    if show_current:
        ax_i = axes[1, 0]
        ax_i.plot(trace.time, trace.current, color="tab:blue", linewidth=0.6)
        ax_i.set_ylabel("Current (pA)")
        ax_i.set_xlabel("Time (ms)")
        ax_i.grid(True, alpha=0.2)

        if highlight_stim:
            ax_i.axvspan(t_start, t_end, color="blue", alpha=0.06)
    else:
        ax_v.set_xlabel("Time (ms)")

    fig.suptitle(
        f"dt={trace.dt_ms} ms | {trace.n_samples} samples | "
        f"{trace.n_spikes} spikes | I={trace.stim_current_pA:.0f} pA",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def trace_stim_window_plot(
    trace: TraceData,
    padding_ms: float = 50.0,
    show_spikes: bool = True,
    show_current: bool = True,
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Plot only the stimulation window of a trace (zoomed in).

    Args:
        trace: TraceData to visualize
        padding_ms: Extra time before/after stim window to show
        show_spikes: Mark detected spike times
        show_current: Show current injection as a second subplot
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    n_rows = 2 if show_current else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True, squeeze=False)
    ax_v = axes[0, 0]

    t_start = trace.time[trace.stim_start_idx] - padding_ms
    t_end = trace.time[min(trace.stim_end_idx, len(trace.time) - 1)] + padding_ms

    ax_v.plot(trace.time, trace.voltage, color="k", linewidth=0.6)
    ax_v.set_xlim(t_start, t_end)
    ax_v.set_ylabel("Voltage (mV)")

    if show_spikes and len(trace.spike_times) > 0:
        for st in trace.spike_times:
            if t_start <= st <= t_end:
                ax_v.axvline(st, color="r", linestyle=":", alpha=0.5, linewidth=0.7)

    ax_v.grid(True, alpha=0.2)

    if show_current:
        ax_i = axes[1, 0]
        ax_i.plot(trace.time, trace.current, color="tab:blue", linewidth=0.6)
        ax_i.set_ylabel("Current (pA)")
        ax_i.set_xlabel("Time (ms)")
        ax_i.grid(True, alpha=0.2)
    else:
        ax_v.set_xlabel("Time (ms)")

    fig.suptitle(
        f"Stim window | {trace.n_spikes} spikes | "
        f"I={trace.stim_current_pA:.0f} pA | "
        f"{trace.stim_duration_ms:.0f} ms",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def alignment_diagnostic_plot(
    data: TraceData,
    stim_end_index: int | None = None,
    figsize: tuple[float, float] = (14, 10),
) -> Figure:
    """Diagnostic plot to verify time alignment between data, stimuli, and loss functions.

    Shows four panels:
    1. Experimental voltage with stim window markers and pre/post padding regions
    2. Current trace (what the simulation receives)
    3. What the loss function sees (voltage[:stim_end_index + margin])
    4. Spike times overlaid on voltage

    Also prints a summary of all alignment-critical indices and time windows.

    Args:
        data: Cropped TraceData (from crop_to_stim_window with padding)
        stim_end_index: The stim_end_index passed to the loss function factory.
            If None, uses data.stim_end_idx.
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if stim_end_index is None:
        stim_end_index = data.stim_end_idx

    time_ms = data.time
    stim_start_ms = data.stim_start_idx * data.dt_ms
    stim_end_ms = data.stim_end_idx * data.dt_ms

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # ── Panel 1: Voltage with stim markers ──
    ax = axes[0]
    ax.plot(time_ms, data.voltage, color="black", linewidth=0.6, label="data.voltage")
    ax.axvline(stim_start_ms, color="green", ls="--", lw=1.5,
               label=f"stim_start_idx={data.stim_start_idx}")
    ax.axvline(stim_end_ms, color="red", ls="--", lw=1.5,
               label=f"stim_end_idx={data.stim_end_idx}")
    if data.stim_start_idx > 0:
        ax.axvspan(time_ms[0], stim_start_ms, alpha=0.1, color="blue",
                   label="pre-stim padding")
    post_stim_samples = data.n_samples - data.stim_end_idx
    if post_stim_samples > 0:
        ax.axvspan(stim_end_ms, time_ms[-1], alpha=0.1, color="orange",
                   label="post-stim padding")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title("Cropped data with stim window markers")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    # ── Panel 2: Current trace ──
    ax = axes[1]
    ax.plot(time_ms, data.current, color="C1", linewidth=0.8, label="data.current (pA)")
    ax.axvline(stim_start_ms, color="green", ls="--", lw=1.5)
    ax.axvline(stim_end_ms, color="red", ls="--", lw=1.5)
    ax.set_ylabel("Current (pA)")
    ax.set_title("Stimulus current fed to simulation & training")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    # ── Panel 3: Loss function view ──
    ax = axes[2]
    loss_margin = 100  # loss functions typically use stim_end_index + 100
    loss_end = min(stim_end_index + loss_margin, len(data.voltage))
    ax.plot(time_ms, data.voltage, color="black", linewidth=0.5, alpha=0.25,
            label="full data.voltage")
    ax.plot(time_ms[:loss_end], data.voltage[:loss_end], color="C0", linewidth=0.8,
            label=f"loss view [:stim_end+100] = [:{loss_end}]")
    ax.axvline(stim_end_index * data.dt_ms, color="red", ls="--", lw=1.5,
               label=f"stim_end_index={stim_end_index}")
    if data.stim_start_idx > 0:
        ax.axvspan(time_ms[0], stim_start_ms, alpha=0.15, color="yellow",
                   label=f"pre-stim in loss: [0, {data.stim_start_idx})")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title("What the loss function sees (voltage[:stim_end_index + 100])")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    # ── Panel 4: Spike times ──
    ax = axes[3]
    ax.plot(time_ms, data.voltage, color="black", linewidth=0.6)
    for i, st in enumerate(data.spike_times):
        ax.axvline(st, color="C2", alpha=0.6, lw=0.8,
                   label="spike_times" if i == 0 else None)
    ax.set_ylabel("Voltage (mV)")
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Spike times in cropped data ({len(data.spike_times)} spikes)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()

    # ── Print summary ──
    pre_stim_ms = data.stim_start_idx * data.dt_ms
    post_stim_ms = post_stim_samples * data.dt_ms
    print("=" * 60)
    print("ALIGNMENT SUMMARY")
    print("=" * 60)
    print(f"data.n_samples        = {data.n_samples}")
    print(f"data.dt_ms            = {data.dt_ms}")
    print(f"data.duration_ms      = {data.duration_ms:.1f}")
    print(f"data.stim_start_idx   = {data.stim_start_idx}  ({pre_stim_ms:.1f} ms)")
    print(f"data.stim_end_idx     = {data.stim_end_idx}  ({stim_end_ms:.1f} ms)")
    print(f"data.stim_duration_ms = {data.stim_duration_ms:.1f}")
    print(f"data.stim_current_pA  = {data.stim_current_pA:.1f}")
    print(f"stim_end_index (loss) = {stim_end_index}")
    print(f"loss sees voltage[:{loss_end}]  ({loss_end * data.dt_ms:.1f} ms)")
    print()
    print(f"Pre-stim:   [{0}, {data.stim_start_idx})  = {pre_stim_ms:.1f} ms")
    print(f"Stim:       [{data.stim_start_idx}, {data.stim_end_idx})  = {data.stim_duration_ms:.1f} ms")
    print(f"Post-stim:  [{data.stim_end_idx}, {data.n_samples})  = {post_stim_ms:.1f} ms")
    if data.stim_start_idx > 0:
        print()
        print(f"Note: Pre-stim padding ({pre_stim_ms:.0f} ms) is included in both")
        print(f"  simulation and target. Loss functions compare from t=0,")
        print(f"  so spike times include the pre-stim offset in both sim and exp.")

    return fig


def traces_overlay_plot(
    traces: Sequence[TraceData],
    labels: Sequence[str] | None = None,
    stim_window_only: bool = False,
    padding_ms: float = 50.0,
    figsize: tuple[float, float] = (12, 5),
) -> Figure:
    """Overlay multiple traces for comparison.

    Args:
        traces: Sequence of TraceData objects
        labels: Labels for each trace (defaults to index)
        stim_window_only: If True, zoom to the stimulation window of the first trace
        padding_ms: Padding around stim window when stim_window_only=True
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if labels is None:
        labels = [f"Trace {i}" for i in range(len(traces))]

    colors = plt.cm.tab10(np.linspace(0, 1, min(len(traces), 10)))

    fig, ax = plt.subplots(figsize=figsize)

    for trace, label, color in zip(traces, labels, colors):
        ax.plot(trace.time, trace.voltage, color=color, linewidth=0.6, label=label)

    if stim_window_only and len(traces) > 0:
        ref = traces[0]
        t_start = ref.time[ref.stim_start_idx] - padding_ms
        t_end = ref.time[min(ref.stim_end_idx, len(ref.time) - 1)] + padding_ms
        ax.set_xlim(t_start, t_end)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fit_comparison_plot(
    trace: TraceData,
    sim: SimulationResult,
    stim_window_only: bool = True,
    padding_ms: float = 50.0,
    figsize: tuple[float, float] = (12, 7),
) -> Figure:
    """Plot experimental trace vs simulation result after fitting.

    Args:
        trace: Experimental TraceData
        sim: SimulationResult from fitted model
        stim_window_only: Zoom to stimulation window
        padding_ms: Padding around stim window
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Voltage comparison
    ax_v = axes[0]
    ax_v.plot(
        trace.time,
        trace.voltage,
        color="gray",
        linewidth=0.6,
        alpha=0.7,
        label=f"Experimental ({trace.n_spikes} spikes)",
    )
    ax_v.plot(
        sim.time,
        sim.voltage,
        color="tab:blue",
        linewidth=0.6,
        label=f"Simulated ({sim.n_spikes} spikes)",
    )
    ax_v.set_ylabel("Voltage (mV)")
    ax_v.legend(fontsize=8)
    ax_v.grid(True, alpha=0.2)

    # Spike raster
    ax_s = axes[1]
    trains = []
    colors = []
    y_labels = []

    if len(trace.spike_times) > 0:
        trains.append(trace.spike_times)
        colors.append("gray")
        y_labels.append("Exp")
    if len(sim.spike_times) > 0:
        trains.append(sim.spike_times)
        colors.append("tab:blue")
        y_labels.append("Sim")

    if trains:
        ax_s.eventplot(trains, colors=colors, linelengths=0.6)
        ax_s.set_yticks(range(len(y_labels)))
        ax_s.set_yticklabels(y_labels, fontsize=8)
    ax_s.set_xlabel("Time (ms)")
    ax_s.grid(True, alpha=0.2, axis="x")

    if stim_window_only:
        t_start = trace.time[trace.stim_start_idx] - padding_ms
        t_end = trace.time[min(trace.stim_end_idx, len(trace.time) - 1)] + padding_ms
        ax_v.set_xlim(t_start, t_end)

    fig.tight_layout()
    return fig


# =========================================================================
# Training & evaluation plots
# =========================================================================


def training_history_plot(
    result: TrainingResult,
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot training loss curve, gradient norms, and learning rate schedule.

    Rows shown depend on available data:
    - Loss curve (always)
    - Gradient norms (when ``grad_norms`` is non-empty)
    - Learning rate (when ``lr_history`` is non-empty, i.e. a schedule was active)

    Args:
        result: TrainingResult from training
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    has_grads = len(result.grad_norms) > 0
    has_lr = len(result.lr_history) > 0
    n_rows = 1 + has_grads + has_lr
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True, squeeze=False)

    epochs = np.arange(len(result.loss_history))
    row = 0

    # Loss curve
    ax_loss = axes[row, 0]
    ax_loss.plot(epochs, result.loss_history, color="k", linewidth=0.8)
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.2)

    # Mark best epoch
    best_epoch = int(np.argmin(result.loss_history))
    best_loss = result.loss_history[best_epoch]
    ax_loss.plot(best_epoch, best_loss, "o", color="r", markersize=5, zorder=5)
    ax_loss.annotate(
        f"best={best_loss:.4f} (ep {best_epoch})",
        xy=(best_epoch, best_loss),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        color="r",
    )

    # Gradient norms
    if has_grads:
        row += 1
        ax_grad = axes[row, 0]
        grad_epochs = np.arange(len(result.grad_norms))
        has_clipped = len(result.clipped_grad_norms) > 0
        ax_grad.plot(
            grad_epochs,
            result.grad_norms,
            color="tab:red" if has_clipped else "tab:blue",
            linewidth=0.8,
            alpha=0.6 if has_clipped else 1.0,
            label="Raw" if has_clipped else "Gradient norm",
        )
        if has_clipped:
            ax_grad.plot(
                np.arange(len(result.clipped_grad_norms)),
                result.clipped_grad_norms,
                color="tab:blue",
                linewidth=0.8,
                label="Clipped",
            )
            ax_grad.legend(fontsize=8)
        ax_grad.set_ylabel("Gradient norm")
        ax_grad.set_yscale("log")
        ax_grad.grid(True, alpha=0.2)

    # Learning rate schedule
    if has_lr:
        row += 1
        ax_lr = axes[row, 0]
        ax_lr.plot(
            np.arange(len(result.lr_history)),
            result.lr_history,
            color="tab:purple",
            linewidth=0.8,
        )
        ax_lr.set_ylabel("Learning rate")
        ax_lr.grid(True, alpha=0.2)

    # X-axis label on bottom row only
    axes[row, 0].set_xlabel("Epoch")

    cfg = result.config
    title_parts = [
        f"{cfg.optimizer}",
        f"lr={cfg.learning_rate}",
        f"{cfg.surrogate_type}",
        f"{len(result.loss_history)} epochs",
        f"{result.total_time:.1f}s",
    ]
    if cfg.lr_schedule:
        title_parts.append(f"schedule={cfg.lr_schedule}")
    if cfg.grad_clip_norm is not None:
        title_parts.append(f"clip={cfg.grad_clip_norm}")
    fig.suptitle(" | ".join(title_parts), fontsize=9)
    fig.tight_layout()
    return fig


def training_comparison_plot(
    results: Sequence[TrainingResult],
    labels: Sequence[str],
    show_grad_norms: bool = False,
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Compare training curves from multiple runs.

    Args:
        results: Sequence of TrainingResult objects
        labels: Label for each result
        show_grad_norms: If True, add a second row with gradient norms
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    n_rows = 2 if show_grad_norms else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True, squeeze=False)

    colors = plt.cm.tab10(np.linspace(0, 1, min(len(results), 10)))

    ax_loss = axes[0, 0]
    for res, label, color in zip(results, labels, colors):
        epochs = np.arange(len(res.loss_history))
        ax_loss.plot(epochs, res.loss_history, color=color, linewidth=0.8, label=label)
        best_ep = int(np.argmin(res.loss_history))
        ax_loss.plot(
            best_ep,
            res.loss_history[best_ep],
            "*",
            color=color,
            markersize=8,
            zorder=5,
        )

    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.2)

    if show_grad_norms:
        ax_grad = axes[1, 0]
        for res, label, color in zip(results, labels, colors):
            if res.grad_norms:
                ax_grad.plot(
                    np.arange(len(res.grad_norms)),
                    res.grad_norms,
                    color=color,
                    linewidth=0.8,
                    label=label,
                )
        ax_grad.set_ylabel("Gradient norm")
        ax_grad.set_yscale("log")
        ax_grad.set_xlabel("Epoch")
        ax_grad.legend(fontsize=8)
        ax_grad.grid(True, alpha=0.2)
    else:
        ax_loss.set_xlabel("Epoch")

    fig.tight_layout()
    return fig


def parameter_comparison_plot(
    result: TrainingResult,
    bounds: dict[str, ParamBounds] | None = None,
    figsize: tuple[float, float] | None = None,
    figtitle: str="Parameter changes (normalized to bounds)",
) -> Figure:
    """Horizontal bar chart showing parameter changes within bounds.

    Each parameter is normalized to [0, 1] within its bounds range.
    Black diamond = initial, blue circle = trained, arrow between them.

    Args:
        result: TrainingResult from training
        bounds: Parameter bounds dict. If None, uses PARAM_BOUNDS.
        figsize: Figure size. If None, auto-scaled to number of params.
        figtitle: Figure title.

    Returns:
        Matplotlib Figure

    """
    from ADoptEX.core.parameters import PARAM_BOUNDS

    if bounds is None:
        bounds = PARAM_BOUNDS

    trained = result.get_params_dict()
    initial = result.initial_params

    # Only show params that were actually trained
    param_names = [k for k in trained if k in bounds]

    if figsize is None:
        figsize = (8, max(2.5, 0.55 * len(param_names) + 1))

    fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(len(param_names))

    for i, name in enumerate(param_names):
        b = bounds[name]
        span = b.max - b.min
        if span == 0:
            continue

        # Normalize to [0, 1]
        init_val = initial.get(name, b.min)
        train_val = trained[name]
        init_norm = (init_val - b.min) / span
        train_norm = (train_val - b.min) / span

        # Gray bar for full bounds range
        ax.barh(i, 1.0, height=0.3, color="lightgray", edgecolor="none")

        # Arrow from initial to trained
        ax.annotate(
            "",
            xy=(train_norm, i),
            xytext=(init_norm, i),
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2),
        )

        # Markers
        ax.plot(init_norm, i, "D", color="k", markersize=5, zorder=5)
        ax.plot(train_norm, i, "o", color="tab:blue", markersize=6, zorder=5)

        # Annotate trained value
        ax.text(
            1.02,
            i,
            f"{train_val:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color="tab:blue",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(param_names, fontsize=8)
    ax.set_xlim(-0.05, 1.15)
    ax.set_xlabel("Normalized position within bounds", fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")

    # Legend
    ax.plot([], [], "D", color="k", markersize=5, label="Initial")
    ax.plot([], [], "o", color="tab:blue", markersize=6, label="Trained")
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"{figtitle}", fontsize=9)
    fig.tight_layout()
    return fig


def fit_before_after_plot(
    trace: TraceData,
    sim_initial: SimulationResult,
    sim_trained: SimulationResult | Sequence[SimulationResult],
    coincidence: CoincidenceResult | Sequence[CoincidenceResult | None] | None = None,
    labels: Sequence[str] | None = None,
    stim_window_only: bool = True,
    padding_ms: float = 20.0,
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Before/after comparison of experimental trace vs initial and trained simulations.

    Supports multiple trained results (e.g. multi-stage training). Each trained sim
    gets its own "After" row. Passing a single SimulationResult works as before.

    Args:
        trace: Experimental TraceData
        sim_initial: SimulationResult with initial parameters
        sim_trained: Single or sequence of trained SimulationResults
        coincidence: Single or sequence of CoincidenceResults (one per trained sim).
            Use None entries for sims without a coincidence result.
        labels: Labels for each trained sim (e.g. ["Stage 1", "Stage 2"]).
            Defaults to "Trained" for single, "Stage 1", "Stage 2", ... for multiple.
        stim_window_only: Zoom to stimulation window
        padding_ms: Padding around stim window
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    # Normalize to lists
    if not isinstance(sim_trained, Sequence):
        trained_sims = [sim_trained]
    else:
        trained_sims = list(sim_trained)

    n_trained = len(trained_sims)

    if coincidence is None:
        coincidences: list[CoincidenceResult | None] = [None] * n_trained
    elif not isinstance(coincidence, Sequence):
        coincidences = [coincidence]
    else:
        coincidences = list(coincidence)

    if labels is None:
        if n_trained == 1:
            sim_labels = ["Trained"]
        else:
            sim_labels = [f"Stage {i + 1}" for i in range(n_trained)]
    else:
        sim_labels = list(labels)

    trained_colors = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:cyan"]

    # 1 row for "before" + 1 row per trained sim + 1 row for spike raster
    n_rows = 1 + n_trained + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)

    n = min(
        len(trace.time), len(sim_initial.time), *(len(s.time) for s in trained_sims)
    )
    time = trace.time[:n]

    # Row 1: Before
    ax_before = axes[0]
    ax_before.plot(
        time,
        trace.voltage[:n],
        color="k",
        linewidth=0.6,
        alpha=0.7,
        label=f"Experimental ({trace.n_spikes})",
    )
    ax_before.plot(
        sim_initial.time[:n],
        sim_initial.voltage[:n],
        color="tab:orange",
        linewidth=0.6,
        label=f"Initial ({sim_initial.n_spikes})",
    )
    ax_before.set_ylabel("Voltage (mV)")
    ax_before.legend(fontsize=8, loc="upper right")
    ax_before.set_title("Before training", fontsize=9)
    ax_before.grid(True, alpha=0.2)

    # Rows 2..N: After (one per trained sim)
    for i, (sim, cf, label, color) in enumerate(
        zip(trained_sims, coincidences, sim_labels, trained_colors)
    ):
        ax = axes[1 + i]
        ax.plot(
            time,
            trace.voltage[:n],
            color="k",
            linewidth=0.6,
            alpha=0.7,
            label=f"Experimental ({trace.n_spikes})",
        )
        trained_label = f"{label} ({sim.n_spikes})"
        if cf is not None:
            trained_label += f" | Γ={cf.gamma:.3f}"
        ax.plot(
            sim.time[:n],
            sim.voltage[:n],
            color=color,
            linewidth=0.6,
            label=trained_label,
        )
        ax.set_ylabel("Voltage (mV)")
        ax.legend(fontsize=8, loc="upper right")
        title = f"After training — {label}"
        if cf is not None:
            title += f" | Γ={cf.gamma:.3f}"
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.2)

    # Last row: Spike raster
    ax_raster = axes[-1]
    trains = []
    colors = []
    y_labels = []

    if len(trace.spike_times) > 0:
        trains.append(trace.spike_times)
        colors.append("k")
        y_labels.append("Exp")
    if len(sim_initial.spike_times) > 0:
        trains.append(sim_initial.spike_times)
        colors.append("tab:orange")
        y_labels.append("Initial")
    for sim, label, color in zip(trained_sims, sim_labels, trained_colors):
        if len(sim.spike_times) > 0:
            trains.append(sim.spike_times)
            colors.append(color)
            y_labels.append(label)

    if trains:
        ax_raster.eventplot(trains, colors=colors, linelengths=0.6)
        ax_raster.set_yticks(range(len(y_labels)))
        ax_raster.set_yticklabels(y_labels, fontsize=8)
    ax_raster.set_xlabel("Time (ms)")
    ax_raster.grid(True, alpha=0.2, axis="x")

    if stim_window_only:
        t_start = trace.time[trace.stim_start_idx] - padding_ms
        t_end = trace.time[min(trace.stim_end_idx, len(trace.time) - 1)] + padding_ms
        axes[0].set_xlim(t_start, t_end)

    fig.tight_layout()
    return fig


def _match_spikes(
    spike_times_data: np.ndarray,
    spike_times_model: np.ndarray,
    delta_ms: float = 2.0,
) -> list[tuple[int, int, float]]:
    """Greedy match model spikes to data spikes within a coincidence window.

    Returns list of (data_idx, model_idx, timing_error_ms) tuples.
    Replicates the matching logic from count_coincidences.
    """
    if len(spike_times_data) == 0 or len(spike_times_model) == 0:
        return []

    matched_data = np.zeros(len(spike_times_data), dtype=bool)
    matches = []

    for m_idx, t_model in enumerate(spike_times_model):
        time_diffs = np.abs(spike_times_data - t_model)
        within_window = (time_diffs <= delta_ms) & (~matched_data)

        if np.any(within_window):
            candidates = np.where(within_window)[0]
            best_d_idx = candidates[np.argmin(time_diffs[candidates])]
            matched_data[best_d_idx] = True
            error = float(spike_times_model[m_idx] - spike_times_data[best_d_idx])
            matches.append((int(best_d_idx), m_idx, error))

    return matches


def spike_timing_plot(
    trace: TraceData,
    sim: SimulationResult,
    coincidence: CoincidenceResult,
    padding_ms: float = 20.0,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Detailed spike timing comparison with coincidence windows and error stems.

    Args:
        trace: Experimental TraceData
        sim: SimulationResult from fitted model
        coincidence: CoincidenceResult for this comparison
        padding_ms: Padding around stim window
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    delta = coincidence.delta_ms
    matches = _match_spikes(trace.spike_times, sim.spike_times, delta)

    matched_data_idxs = {m[0] for m in matches}
    matched_model_idxs = {m[1] for m in matches}

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Row 1: Spike timing with coincidence windows
    ax_top = axes[0]

    # Draw coincidence windows around data spikes
    for st in trace.spike_times:
        ax_top.axvspan(
            st - delta,
            st + delta,
            color="green",
            alpha=0.12,
            linewidth=0,
        )

    # Data spike markers
    ax_top.eventplot(
        [trace.spike_times],
        lineoffsets=1.0,
        linelengths=0.6,
        colors=["k"],
        label=f"Data ({len(trace.spike_times)})",
    )
    # Model spike markers
    ax_top.eventplot(
        [sim.spike_times],
        lineoffsets=0.0,
        linelengths=0.6,
        colors=["tab:blue"],
        label=f"Model ({len(sim.spike_times)})",
    )

    # Connecting lines for matched spikes
    for d_idx, m_idx, _ in matches:
        ax_top.plot(
            [trace.spike_times[d_idx], sim.spike_times[m_idx]],
            [1.0, 0.0],
            color="green",
            linewidth=0.6,
            alpha=0.6,
        )

    # Red x for unmatched data spikes
    unmatched_data = [
        trace.spike_times[i]
        for i in range(len(trace.spike_times))
        if i not in matched_data_idxs
    ]
    if unmatched_data:
        ax_top.scatter(
            unmatched_data,
            [1.0] * len(unmatched_data),
            marker="x",
            color="r",
            s=30,
            zorder=5,
            label="Unmatched data",
        )

    # Red x for unmatched model spikes
    unmatched_model = [
        sim.spike_times[i]
        for i in range(len(sim.spike_times))
        if i not in matched_model_idxs
    ]
    if unmatched_model:
        ax_top.scatter(
            unmatched_model,
            [0.0] * len(unmatched_model),
            marker="x",
            color="r",
            s=30,
            zorder=5,
            label="Unmatched model",
        )

    ax_top.set_yticks([0, 1])
    ax_top.set_yticklabels(["Model", "Data"], fontsize=8)
    ax_top.legend(fontsize=8, loc="upper right")
    ax_top.grid(True, alpha=0.2, axis="x")
    ax_top.set_title(
        f"Spike timing | \u0393={coincidence.gamma:.3f} | "
        f"\u0394={delta:.1f} ms | "
        f"{coincidence.n_coincidences}/{coincidence.n_data} coincidences",
        fontsize=9,
    )

    # Row 2: Timing error stems
    ax_bot = axes[1]
    if matches:
        data_times = [trace.spike_times[m[0]] for m in matches]
        errors = [m[2] for m in matches]
        ax_bot.stem(
            data_times,
            errors,
            linefmt="tab:blue",
            markerfmt="o",
            basefmt="k-",
        )
        ax_bot.axhline(0, color="k", linewidth=0.5)
        ax_bot.axhline(delta, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_bot.axhline(-delta, color="green", linewidth=0.5, linestyle="--", alpha=0.5)

        mean_err = np.mean(np.abs(errors))
        ax_bot.set_title(
            f"Timing error per spike | mean |error| = {mean_err:.2f} ms", fontsize=9
        )
    else:
        ax_bot.set_title("No matched spikes", fontsize=9)

    ax_bot.set_ylabel("Error (ms)", fontsize=8)
    ax_bot.set_xlabel("Time (ms)")
    ax_bot.grid(True, alpha=0.2)

    # Zoom to stim window
    t_start = trace.time[trace.stim_start_idx] - padding_ms
    t_end = trace.time[min(trace.stim_end_idx, len(trace.time) - 1)] + padding_ms
    ax_top.set_xlim(t_start, t_end)

    fig.tight_layout()
    return fig
