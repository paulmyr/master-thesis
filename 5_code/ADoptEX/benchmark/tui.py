"""
Interactive TUI for browsing benchmark results.

Reads result JSON files, re-simulates ground truth and best fit,
and displays voltage traces + summary statistics.

Usage:
    python -m ADoptEX.benchmark.tui [results_dir]
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .results import ScenarioResults, load_all_results
from .scenarios import _FIRING_PATTERNS

# Default stimulus parameters (shared across all synthetic scenarios)
_DEFAULT_STIM = {
    "stim_current_pA": 500.0,
    "stim_duration_ms": 400.0,
    "stim_delay_ms": 50.0,
    "dt_ms": 0.025,
}


def _pattern_name(scenario_name: str) -> str:
    """Extract the firing pattern name from a scenario name like 'tonic_25pct_membrane'."""
    for pattern in _FIRING_PATTERNS:
        if scenario_name.startswith(pattern):
            return pattern
    return scenario_name.split("_")[0]


def _stim_for_pattern(pattern: str) -> float:
    """Get stimulus current for a firing pattern."""
    entry = _FIRING_PATTERNS.get(pattern, {})
    I = entry.get("I", 500.0)
    assert isinstance(I, float)
    return I


def _simulate(params: dict, stim_current_pA: float):
    """Simulate and return (time, voltage, spike_times)."""
    from .runner import _suppress_jaxley_prints

    from ADoptEX.core.simulation import simulate_jaxley

    t_max = _DEFAULT_STIM["stim_delay_ms"] + _DEFAULT_STIM["stim_duration_ms"] + 50.0
    with _suppress_jaxley_prints():
        result = simulate_jaxley(
            params=params,
            stim_current_pA=stim_current_pA,
            stim_duration_ms=_DEFAULT_STIM["stim_duration_ms"],
            dt_ms=_DEFAULT_STIM["dt_ms"],
            t_max_ms=t_max,
            stim_delay_ms=_DEFAULT_STIM["stim_delay_ms"],
            use_surrogate=True,
            surrogate_slope=5.0,
        )
    return result.time, result.voltage, result.spike_times


def _spike_stats(spike_times: np.ndarray, stim_start_ms: float, stim_end_ms: float) -> dict:
    """Compute spike train statistics."""
    # Filter to spikes within stim window
    mask = (spike_times >= stim_start_ms) & (spike_times <= stim_end_ms)
    st = spike_times[mask] if len(spike_times) > 0 else spike_times

    n = len(st)
    isis = np.diff(st) if n > 1 else np.array([])
    stim_dur_s = (stim_end_ms - stim_start_ms) / 1000.0

    return {
        "n_spikes": n,
        "firing_rate_hz": n / stim_dur_s if stim_dur_s > 0 else 0.0,
        "first_spike_ms": float(st[0]) if n > 0 else float("nan"),
        "mean_isi_ms": float(np.mean(isis)) if len(isis) > 0 else float("nan"),
        "cv_isi": float(np.std(isis) / np.mean(isis)) if len(isis) > 1 else float("nan"),
    }


def _safe(val, fmt=".2f") -> str:
    """Format a float, handling NaN."""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "—"
    return f"{val:{fmt}}"


# ── Plotting ────────────────────────────────────────────────────────────────


def plot_scenario(sr: ScenarioResults):
    """Plot voltage traces and stats for a scenario's best run."""
    best = sr.best_run
    if best is None:
        print("  No runs found.")
        return

    pattern = _pattern_name(sr.scenario_name)
    stim_I = _stim_for_pattern(pattern)
    gt_params = best.ground_truth_params or {}
    fit_params = best.final_params

    # Re-simulate
    print(f"  Simulating ground truth and best fit (gamma={_safe(best.gamma, '.3f')})...")
    gt_time, gt_voltage, gt_spikes = _simulate(gt_params, stim_I)
    fit_time, fit_voltage, fit_spikes = _simulate(fit_params, stim_I)

    stim_start = _DEFAULT_STIM["stim_delay_ms"]
    stim_end = stim_start + _DEFAULT_STIM["stim_duration_ms"]

    gt_stats = _spike_stats(gt_spikes, stim_start, stim_end)
    fit_stats = _spike_stats(fit_spikes, stim_start, stim_end)

    # ── Figure (non-blocking) ─────────────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(14, 10), num="Benchmark Results")
    fig.clf()
    fig.suptitle(
        f"{sr.scenario_name}  —  {sr.method_name}\n"
        f"best run #{best.start_index}  |  Γ = {_safe(best.gamma, '.3f')}  |  "
        f"loss = {_safe(best.final_loss, '.4f')}",
        fontsize=12,
    )
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 2], hspace=0.35, wspace=0.3)

    # ── Top: voltage overlay ────────────────────────────────────────────
    ax_v = fig.add_subplot(gs[0, :])
    ax_v.plot(gt_time, gt_voltage, color="k", lw=0.7, alpha=0.8, label="Ground truth")
    ax_v.plot(fit_time, fit_voltage, color="tab:blue", lw=0.7, alpha=0.8, label="Best fit")
    ax_v.axvspan(stim_start, stim_end, alpha=0.06, color="green", label="Stim window")
    ax_v.set_ylabel("Voltage (mV)")
    ax_v.set_xlabel("Time (ms)")
    ax_v.legend(fontsize=8, loc="upper right")
    ax_v.grid(True, alpha=0.15)
    ax_v.set_title("Membrane potential", fontsize=10)

    # ── Middle: spike raster ────────────────────────────────────────────
    ax_r = fig.add_subplot(gs[1, :], sharex=ax_v)
    trains, colors, labels = [], [], []
    if len(gt_spikes) > 0:
        trains.append(gt_spikes)
        colors.append("k")
        labels.append("GT")
    if len(fit_spikes) > 0:
        trains.append(fit_spikes)
        colors.append("tab:blue")
        labels.append("Fit")
    if trains:
        ax_r.eventplot(trains, colors=colors, linelengths=0.6)
        ax_r.set_yticks(range(len(labels)))
        ax_r.set_yticklabels(labels, fontsize=8)
    ax_r.set_xlabel("Time (ms)")
    ax_r.grid(True, alpha=0.15, axis="x")
    ax_r.set_title("Spike raster", fontsize=10)

    # ── Bottom left: spike stats table ──────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, 0])
    ax_tbl.axis("off")

    stat_rows = [
        ["", "Ground truth", "Best fit"],
        ["Spike count", str(gt_stats["n_spikes"]), str(fit_stats["n_spikes"])],
        ["Firing rate (Hz)", _safe(gt_stats["firing_rate_hz"]), _safe(fit_stats["firing_rate_hz"])],
        ["First spike (ms)", _safe(gt_stats["first_spike_ms"]), _safe(fit_stats["first_spike_ms"])],
        ["Mean ISI (ms)", _safe(gt_stats["mean_isi_ms"]), _safe(fit_stats["mean_isi_ms"])],
        ["CV ISI", _safe(gt_stats["cv_isi"]), _safe(fit_stats["cv_isi"])],
        ["", "", ""],
        ["Γ (coincidence)", "", _safe(best.gamma, ".3f")],
        ["Spike count error", "", str(best.spike_count_error)],
        ["Wall time (s)", "", _safe(best.wall_time_s, ".1f")],
    ]

    tbl = ax_tbl.table(
        cellText=stat_rows,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    # Style header row
    for j in range(3):
        tbl[0, j].set_text_props(fontweight="bold")
    ax_tbl.set_title("Spike statistics", fontsize=10, pad=20)

    # ── Bottom right: loss history ──────────────────────────────────────
    ax_loss = fig.add_subplot(gs[2, 1])
    if best.loss_history:
        ax_loss.plot(best.loss_history, color="tab:red", lw=0.8)
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_yscale("log")
        ax_loss.grid(True, alpha=0.15)
    else:
        ax_loss.text(0.5, 0.5, "No loss history", ha="center", va="center", transform=ax_loss.transAxes)
    ax_loss.set_title("Loss convergence", fontsize=10)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ── TUI loop ────────────────────────────────────────────────────────────────


def main(results_dir: str | None = None):

    """Interactive scenario browser."""
    if results_dir is None:
        results_dir = str(Path(__file__).resolve().parents[2] / "benchmarks" / "results")

    path = Path(results_dir)
    if not path.is_dir():
        print(f"Results directory not found: {path}")
        sys.exit(1)

    all_results = load_all_results(path)
    if not all_results:
        print(f"No result files found in {path}")
        sys.exit(1)

    # Group by scenario
    scenarios: dict[str, list[ScenarioResults]] = {}
    for sr in all_results:
        scenarios.setdefault(sr.scenario_name, []).append(sr)

    scenario_names = sorted(scenarios.keys())

    print(f"\n{'='*60}")
    print(f"  Benchmark Results Browser  ({len(all_results)} files)")
    print(f"  {path}")
    print(f"{'='*60}\n")

    while True:
        print("Scenarios:")
        for i, name in enumerate(scenario_names, 1):
            methods = scenarios[name]
            # Quick summary
            gammas = []
            for sr in methods:
                if sr.best_run and not (math.isnan(sr.best_run.gamma) if isinstance(sr.best_run.gamma, float) else False):
                    gammas.append((sr.method_name, sr.best_run.gamma))
            gamma_str = "  ".join(f"{m}={_safe(g, '.2f')}" for m, g in gammas) if gammas else "no results"
            print(f"  [{i:2d}] {name:<45s} {gamma_str}")

        print(f"\n  [a]  Plot all scenarios overview")
        print(f"  [q]  Quit\n")

        choice = input("Select scenario: ").strip().lower()
        if choice == "q":
            break
        if choice == "a":
            _plot_overview(all_results, scenario_names)
            continue

        try:
            idx = int(choice) - 1
            if not (0 <= idx < len(scenario_names)):
                raise ValueError
        except ValueError:
            print("Invalid choice.\n")
            continue

        name = scenario_names[idx]
        methods = scenarios[name]

        print(f"\nMethods for {name}:")
        for j, sr in enumerate(methods, 1):
            s = sr.summary_dict()
            print(
                f"  [{j}] {sr.method_name:<20s}  "
                f"Γ_best={_safe(s['best_gamma'], '.3f')}  "
                f"Γ_mean={_safe(s['mean_gamma'], '.3f')}  "
                f"runs={s['n_runs']}  "
                f"time={_safe(s['total_wall_time_s'], '.0f')}s"
            )
        print(f"  [a] All methods for this scenario")
        print(f"  [b] Back\n")

        mchoice = input("Select method: ").strip().lower()
        if mchoice == "b":
            print()
            continue
        if mchoice == "a":
            for sr in methods:
                plot_scenario(sr)
            continue

        try:
            midx = int(mchoice) - 1
            if not (0 <= midx < len(methods)):
                raise ValueError
        except ValueError:
            print("Invalid choice.\n")
            continue

        plot_scenario(methods[midx])
        print()


def _plot_overview(all_results: list[ScenarioResults], scenario_names: list[str]):
    """Bar chart overview: best gamma per scenario, grouped by method."""
    # Collect data
    methods_seen: list[str] = []
    for sr in all_results:
        if sr.method_name not in methods_seen:
            methods_seen.append(sr.method_name)

    data: dict[str, dict[str, float]] = {}
    for sr in all_results:
        data.setdefault(sr.scenario_name, {})[sr.method_name] = sr.best_gamma

    plt.ion()
    fig, ax = plt.subplots(figsize=(16, 6), num="Benchmark Overview")
    x = np.arange(len(scenario_names))
    width = 0.8 / max(len(methods_seen), 1)
    colors = plt.colormaps["Set2"](np.linspace(0, 0.8, len(methods_seen)))

    for i, method in enumerate(methods_seen):
        vals = []
        for name in scenario_names:
            v = data.get(name, {}).get(method, float("nan"))
            vals.append(v if not (isinstance(v, float) and math.isnan(v)) else 0.0)
        offset = (i - len(methods_seen) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=method, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Best Γ (coincidence factor)")
    ax.set_title("Benchmark Overview — Best Γ per Scenario")
    ax.legend(fontsize=8)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5, label="Γ=0.5 threshold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)