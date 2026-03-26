"""
Interactive TUI for exploring AdEx parameter space.

Tweak parameters, see how the voltage trace changes, and save
parameter sets as new benchmark scenarios.

Usage:
    python -m ADoptEX.benchmark.tui_explore
"""

import json
import math
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ADoptEX.core.parameters import NAUD_PARAMETERS, PARAM_BOUNDS

# Parameters shown in the table (order matters for display)
_PARAM_ORDER = ["C_m", "g_L", "E_L", "v_T", "delta_T", "v_reset", "tau_w", "a", "b"]
_PARAM_UNITS = {
    "C_m": "pF",
    "g_L": "nS",
    "E_L": "mV",
    "v_T": "mV",
    "delta_T": "mV",
    "v_reset": "mV",
    "tau_w": "ms",
    "a": "nS",
    "b": "pA",
    "I": "pA",
}

# Stimulus defaults
_DEFAULT_STIM = {
    "stim_current_pA": 500.0,
    "stim_duration_ms": 400.0,
    "stim_delay_ms": 50.0,
    "dt_ms": 0.025,
}


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
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "--"
    return f"{val:{fmt}}"


# ── Display ─────────────────────────────────────────────────────────────────


def _print_params(params: dict, stim_I: float):
    """Print parameter table with bounds."""
    print(f"\n  {'#':<4} {'Param':<10} {'Value':>10}  {'Unit':<4}  {'Bounds'}")
    print(f"  {'─'*50}")
    for i, name in enumerate(_PARAM_ORDER, 1):
        val = params.get(name, 0.0)
        unit = _PARAM_UNITS.get(name, "")
        bounds = PARAM_BOUNDS.get(name)
        bstr = f"[{bounds.min}, {bounds.max}]" if bounds else ""
        print(f"  {i:<4} {name:<10} {val:>10.2f}  {unit:<4}  {bstr}")
    # Stimulus current as item 10
    print(f"  {len(_PARAM_ORDER)+1:<4} {'I (stim)':<10} {stim_I:>10.1f}  pA")
    print()


class _LivePlot:
    """Persistent non-blocking matplotlib figure that updates in place."""

    def __init__(self):
        self.fig = None
        self.ax_v = None
        self.ax_r = None

    def _ensure_figure(self):
        """Create the figure if it doesn't exist or was closed."""
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            plt.ion()
            self.fig, axes = plt.subplots(
                2, 1, figsize=(13, 7), height_ratios=[3, 1], sharex=True, num="AdEx Explorer"
            )
            self.ax_v = axes[0]
            self.ax_r = axes[1]
            self.fig.show()

    def update(self, params: dict, stim_I: float, history: list | None = None):
        """Simulate, update the plot, and return (time, voltage, stats)."""
        print("  Simulating...", end="", flush=True)
        time, voltage, spike_times = _simulate(params, stim_I)
        print(" done.")

        stim_start = _DEFAULT_STIM["stim_delay_ms"]
        stim_end = stim_start + _DEFAULT_STIM["stim_duration_ms"]
        stats = _spike_stats(spike_times, stim_start, stim_end)

        self._ensure_figure()

        # Clear and redraw
        self.ax_v.clear()
        self.ax_r.clear()

        self.fig.suptitle(
            f"AdEx Explorer  |  I = {stim_I:.0f} pA  |  "
            f"{stats['n_spikes']} spikes  |  "
            f"{_safe(stats['firing_rate_hz'])} Hz  |  "
            f"ISI = {_safe(stats['mean_isi_ms'])} ms",
            fontsize=11,
        )

        # Plot history traces faded
        if history:
            for i, (h_time, h_voltage, h_label) in enumerate(history):
                alpha = 0.2 + 0.15 * (i / max(len(history), 1))
                self.ax_v.plot(h_time, h_voltage, lw=0.5, alpha=alpha, color="gray", label=h_label)

        self.ax_v.plot(time, voltage, color="tab:blue", lw=0.8, label="Current")
        self.ax_v.axvspan(stim_start, stim_end, alpha=0.06, color="green")
        self.ax_v.set_ylabel("Voltage (mV)")
        self.ax_v.grid(True, alpha=0.15)
        if history:
            self.ax_v.legend(fontsize=7, loc="upper right")

        # Spike raster
        if len(spike_times) > 0:
            self.ax_r.eventplot([spike_times], colors=["tab:blue"], linelengths=0.6)
        self.ax_r.set_xlabel("Time (ms)")
        self.ax_r.set_yticks([])
        self.ax_r.grid(True, alpha=0.15, axis="x")

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        return time, voltage, stats


_plot = _LivePlot()


# ── Main loop ───────────────────────────────────────────────────────────────


def main():
    """Interactive parameter explorer."""
    presets = list(NAUD_PARAMETERS.keys())

    print(f"\n{'='*60}")
    print(f"  AdEx Parameter Explorer")
    print(f"{'='*60}\n")
    print("Start from a preset:")
    for i, name in enumerate(presets, 1):
        naud = NAUD_PARAMETERS[name]
        print(f"  [{i}] {name:<25s} I={naud.get('I', 500):.0f} pA")
    print()

    while True:
        choice = input("Select preset (or q to quit): ").strip().lower()
        if choice == "q":
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(presets):
                break
        except ValueError:
            pass
        print("Invalid choice.")

    preset_name = presets[idx]
    params = deepcopy(dict(NAUD_PARAMETERS[preset_name]))
    stim_I = float(params.pop("I", 500.0))

    # Trace history for overlay comparisons
    history: list[tuple[np.ndarray, np.ndarray, str]] = []

    print(f"\nLoaded: {preset_name}")
    _print_params(params, stim_I)

    # Initial plot
    time, voltage, stats = _plot.update(params, stim_I)

    while True:
        print("Commands:")
        print("  <n> <value>   Set param #n to value   (e.g. '3 -65.0' sets E_L)")
        print("  <n> +<val>    Relative change            (e.g. '1 +20' adds 20 to C_m)")
        print("  p             Show current params")
        print("  s             Simulate & plot")
        print("  k             Keep current trace in history (for overlay)")
        print("  c             Clear trace history")
        print("  r             Reset to preset")
        print("  save          Save params to JSON")
        print("  q             Quit\n")

        cmd = input("> ").strip()
        if not cmd:
            continue

        if cmd.lower() == "q":
            break
        elif cmd.lower() == "p":
            _print_params(params, stim_I)
            continue
        elif cmd.lower() == "s":
            time, voltage, stats = _plot.update(params, stim_I, history)
            continue
        elif cmd.lower() == "k":
            label = f"#{len(history)+1}"
            history.append((time, voltage, label))
            print(f"  Saved trace as {label} ({stats['n_spikes']} spikes)")
            continue
        elif cmd.lower() == "c":
            history.clear()
            print("  History cleared.")
            continue
        elif cmd.lower() == "r":
            params = deepcopy(dict(NAUD_PARAMETERS[preset_name]))
            stim_I = float(params.pop("I", 500.0))
            history.clear()
            print(f"  Reset to {preset_name}.")
            _print_params(params, stim_I)
            time, voltage, stats = _plot.update(params, stim_I)
            continue
        elif cmd.lower() == "save":
            _save_params(params, stim_I, preset_name)
            continue

        # Parse "<n> <value>" or "<n> +/-<delta>"
        parts = cmd.split(None, 1)
        if len(parts) != 2:
            print("  Usage: <param#> <value>  or  <param#> +/-<delta>")
            continue

        try:
            n = int(parts[0])
        except ValueError:
            print("  First argument must be a parameter number.")
            continue

        val_str = parts[1].strip()
        is_stim = n == len(_PARAM_ORDER) + 1

        if is_stim:
            name = "I"
            old_val = stim_I
        elif 1 <= n <= len(_PARAM_ORDER):
            name = _PARAM_ORDER[n - 1]
            old_val = params.get(name, 0.0)
        else:
            print(f"  Parameter number must be 1-{len(_PARAM_ORDER)+1}.")
            continue

        # Parse value or delta (only '+' prefix means relative; bare '-5' is absolute)
        try:
            if val_str.startswith("+"):
                new_val = old_val + float(val_str)
            else:
                new_val = float(val_str)
        except ValueError:
            print(f"  Cannot parse '{val_str}' as a number.")
            continue

        # Apply bounds (except stimulus current)
        if is_stim:
            stim_I = new_val
            print(f"  I: {old_val:.2f} -> {new_val:.2f} pA")
        else:
            bounds = PARAM_BOUNDS.get(name)
            if bounds and new_val not in bounds:
                print(f"  Warning: {name}={new_val:.2f} outside bounds [{bounds.min}, {bounds.max}]")
                resp = input("  Apply anyway? [y/N] ").strip().lower()
                if resp != "y":
                    continue
            params[name] = new_val
            print(f"  {name}: {old_val:.2f} -> {new_val:.2f} {_PARAM_UNITS.get(name, '')}")

        # Auto-simulate after change
        time, voltage, stats = _plot.update(params, stim_I, history)


def _save_params(params: dict, stim_I: float, base_name: str):
    """Save current parameters to a JSON file."""
    out_dir = Path(__file__).resolve().parents[2] / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    name = input(f"  Name [{base_name}_custom]: ").strip() or f"{base_name}_custom"
    path = out_dir / f"{name}.json"

    data = {
        "name": name,
        "base_preset": base_name,
        "params": {k: params[k] for k in _PARAM_ORDER if k in params},
        "stim_current_pA": stim_I,
        "stim_duration_ms": _DEFAULT_STIM["stim_duration_ms"],
        "stim_delay_ms": _DEFAULT_STIM["stim_delay_ms"],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {path}")


if __name__ == "__main__":
    main()