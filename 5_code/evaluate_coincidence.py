"""
Evaluation script using Coincidence Factor (Γ) from Jolivet et al. (2008)

This script evaluates how well fitted AdEx parameters predict spike times
compared to experimental recordings, using the coincidence factor metric.

Reference values from Jolivet et al. (2008):
  - Well-fitted aEIF: Γ_A ≈ 0.82-0.83
  - Good model: Γ_A ≈ 0.7+
  - Chance level: Γ ≈ 0
"""

import os
import sys

# Add jaxley to path
sys.path.insert(
    0, "/Users/paulmayer/Projects/university/34_project_course_hpc/3_jaxley"
)

import jax.numpy as jnp
import jaxley as jx
import matplotlib.pyplot as plt
import numpy as np
from jaxley.channels import AdExSurrogate

from core import geometry_for_capacitance
from evaluation import (CoincidenceResult, coincidence_factor,
                        coincidence_factor_from_traces, detect_spike_times)
from train_adex_guarino import (extract_experimental_features,
                                find_stim_window, load_data)


def create_cell(params):
    # TODO: move this somewhere into the training modules? core or trianing?
    """Create AdEx cell with given parameters."""
    radius_um, length_um = geometry_for_capacitance(params["C_m"])

    cell = jx.Cell()
    cell.set("radius", radius_um)
    cell.set("length", length_um)

    cell.insert(AdExSurrogate(surrogate_type="sigmoid", surrogate_slope=10.0))

    cell.set("capacitance", params["C_m"])
    cell.set("AdEx_C_m", params["C_m"])
    cell.set("AdEx_g_L", params["g_L"])
    cell.set("AdEx_E_L", params["E_L"])
    cell.set("AdEx_v_T", params["v_T"])
    cell.set("AdEx_delta_T", params["delta_T"])
    cell.set("AdEx_v_threshold", params["v_threshold"])
    cell.set("AdEx_v_reset", params["v_reset"])
    cell.set("AdEx_tau_w", params["tau_w"])
    cell.set("AdEx_a", params["a"])
    cell.set("AdEx_b", params["b"])

    cell.set("v", params["E_L"])

    cell.record("v")
    cell.record("AdEx_w")
    cell.record("AdEx_spikes")

    return cell


def evaluate_parameters(
    params: dict,
    voltage_exp: np.ndarray,
    current_exp: np.ndarray,
    time_exp: np.ndarray,
    dt_ms: float,
    stim_start_idx: int,
    stim_end_idx: int,
    stim_current_pA: float,
    spike_threshold_mv: float = -20.0,
    delta_ms: float = 2.0,
) -> tuple[CoincidenceResult, np.ndarray, np.ndarray]:
    """
    Evaluate AdEx parameters against experimental data using coincidence factor.

    Args:
        params: AdEx parameter dictionary
        voltage_exp: Experimental voltage trace (mV)
        current_exp: Experimental current trace (pA)
        time_exp: Time array (ms)
        dt_ms: Time step (ms)
        stim_start_idx: Start index of stimulation
        stim_end_idx: End index of stimulation
        stim_current_pA: Stimulation current (pA)
        spike_threshold_mv: Spike detection threshold (mV)
        delta_ms: Coincidence window (ms)

    Returns:
        (CoincidenceResult, simulated_voltage, simulated_spikes)
    """
    # Create cell with given parameters
    # cell = create_adex_cell(params, dt_ms=dt_ms)
    cell = create_cell(params)

    # Override radius/length if provided (from training)
    if "radius" in params:
        cell.set("radius", params["radius"])
    if "length" in params:
        cell.set("length", params["length"])

    # Debug: print actual cell parameters
    has_radius = "radius" in params
    has_length = "length" in params
    print(f"  [DEBUG] has_radius={has_radius}, has_length={has_length}")
    print(
        f"  [DEBUG] cell: radius={cell.nodes['radius'].values[0]:.2f}, "
        f"length={cell.nodes['length'].values[0]:.2f}, "
        f"g_L={cell.nodes['AdEx_g_L'].values[0]:.2f}, "
        f"v_T={cell.nodes['AdEx_v_T'].values[0]:.2f}"
    )

    # Setup stimulus
    stim_duration_ms = (stim_end_idx - stim_start_idx) * dt_ms
    I_nA = stim_current_pA * params["C_m"] / 1000.0
    t_max = stim_duration_ms + 100
    print(
        f"  [DEBUG] stim: I_nA={I_nA:.2f}, duration={stim_duration_ms:.1f}ms, t_max={t_max:.1f}ms"
    )

    cell.stimulate(jx.step_current(0.0, stim_duration_ms, I_nA, dt_ms, t_max=t_max))

    # Run simulation
    results = jx.integrate(cell, delta_t=dt_ms, t_max=t_max)

    voltage_sim = np.array(results[0]).flatten()
    spikes_sim = np.array(results[2]).flatten()

    # Extract spike times from experimental data (during stim window)
    voltage_exp_stim = voltage_exp[stim_start_idx:stim_end_idx]
    spike_times_exp = detect_spike_times(
        voltage_exp_stim, dt_ms, threshold=spike_threshold_mv
    )

    # Extract spike times from simulated data
    n_sim = min(len(voltage_sim), stim_end_idx - stim_start_idx)
    voltage_sim_stim = voltage_sim[:n_sim]
    spike_times_sim = detect_spike_times(
        voltage_sim_stim, dt_ms, threshold=spike_threshold_mv
    )

    # Compute coincidence factor
    duration_ms = len(voltage_exp_stim) * dt_ms
    result = coincidence_factor(
        spike_times_exp, spike_times_sim, duration_ms=duration_ms, delta_ms=delta_ms
    )

    return result, voltage_sim, spikes_sim


def compare_parameter_sets(
    param_sets: dict[str, dict],
    voltage_exp: np.ndarray,
    current_exp: np.ndarray,
    time_exp: np.ndarray,
    dt_ms: float,
    stim_start_idx: int,
    stim_end_idx: int,
    stim_current_pA: float,
    spike_threshold_mv: float = -20.0,
    delta_ms: float = 2.0,
) -> dict[str, CoincidenceResult]:
    """
    Compare multiple parameter sets using coincidence factor.

    Args:
        param_sets: Dictionary of {name: params_dict}
        ... other args same as evaluate_parameters

    Returns:
        Dictionary of {name: CoincidenceResult}
    """
    results = {}

    print("\n" + "=" * 70)
    print("Coincidence Factor Evaluation")
    print("=" * 70)
    print(f"Reference: Well-fitted aEIF achieves Γ ≈ 0.82-0.83 (Jolivet et al. 2008)")
    print(f"Coincidence window: Δ = {delta_ms} ms")
    print("-" * 70)

    for name, params in param_sets.items():
        result, voltage_sim, spikes_sim = evaluate_parameters(
            params,
            voltage_exp,
            current_exp,
            time_exp,
            dt_ms,
            stim_start_idx,
            stim_end_idx,
            stim_current_pA,
            spike_threshold_mv,
            delta_ms,
        )
        results[name] = result

        print(f"\n{name}:")
        print(f"  Γ = {result.gamma:.4f}")
        print(
            f"  Coincidences: {result.n_coincidences} / {result.n_data} experimental spikes"
        )
        print(f"  Model spikes: {result.n_model}")
        print(f"  Expected by chance: {result.expected_coincidences:.2f}")
        print(f"  Model firing rate: {result.firing_rate_model:.1f} Hz")

        # Quality assessment
        if result.gamma >= 0.8:
            quality = "EXCELLENT (matches well-fitted aEIF benchmark)"
        elif result.gamma >= 0.7:
            quality = "GOOD"
        elif result.gamma >= 0.5:
            quality = "MODERATE"
        elif result.gamma >= 0:
            quality = "POOR (near chance level)"
        else:
            quality = "VERY POOR (worse than chance)"
        print(f"  Quality: {quality}")

    print("\n" + "-" * 70)
    print("Summary:")
    for name, result in results.items():
        print(f"  {name}: Γ = {result.gamma:.4f}")
    print("=" * 70)

    return results


def plot_comparison(
    param_sets: dict[str, dict],
    voltage_exp: np.ndarray,
    time_exp: np.ndarray,
    dt_ms: float,
    stim_start_idx: int,
    stim_end_idx: int,
    stim_current_pA: float,
    results: dict[str, CoincidenceResult],
    spike_threshold_mv: float = -20.0,
    save_path: str = None,
):
    """
    Plot voltage traces and coincidence factor comparison.
    """
    n_params = len(param_sets)
    fig, axes = plt.subplots(
        n_params + 1, 1, figsize=(14, 1.3 * (n_params + 1)), sharex=True
    )

    if n_params == 1:
        axes = [axes]

    # Time array for stimulation window
    stim_duration_ms = (stim_end_idx - stim_start_idx) * dt_ms
    time_stim = np.arange(stim_end_idx - stim_start_idx) * dt_ms
    voltage_exp_stim = voltage_exp[stim_start_idx:stim_end_idx]

    # Plot experimental trace
    ax = axes[0]
    ax.plot(time_stim, voltage_exp_stim, "k", linewidth=1, label="Experimental")

    # Mark experimental spikes
    spike_times_exp = detect_spike_times(
        voltage_exp_stim, dt_ms, threshold=spike_threshold_mv
    )
    for st in spike_times_exp:
        ax.axvline(st, color="k", alpha=0.3, linestyle="--", linewidth=0.5)

    ax.set_ylabel("Voltage (mV)")
    ax.set_title(f"Experimental ({len(spike_times_exp)} spikes)")
    ax.legend(loc="upper right")

    # Plot each parameter set
    colors = plt.cm.tab10(np.linspace(0, 1, n_params))

    for i, (name, params) in enumerate(param_sets.items()):
        ax = axes[i + 1]

        # Run simulation
        result, voltage_sim, spikes_sim = evaluate_parameters(
            params,
            voltage_exp,
            np.zeros_like(voltage_exp),
            time_exp,
            dt_ms,
            stim_start_idx,
            stim_end_idx,
            stim_current_pA,
            spike_threshold_mv,
        )

        # Trim to stim window
        n_sim = min(len(voltage_sim), len(time_stim))

        ax.plot(
            time_stim[:n_sim],
            voltage_sim[:n_sim],
            color=colors[i],
            linewidth=1,
            label=name,
        )

        # Mark simulated spikes
        spike_times_sim = detect_spike_times(
            voltage_sim[:n_sim], dt_ms, threshold=spike_threshold_mv
        )
        for st in spike_times_sim:
            ax.axvline(st, color=colors[i], alpha=0.3, linestyle="--", linewidth=0.5)

        ax.set_ylabel("Voltage (mV)")
        gamma = results[name].gamma
        ax.set_title(f"{name} ({len(spike_times_sim)} spikes, Γ = {gamma:.3f})")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (ms)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved plot to {save_path}")

    plt.show()

    # Also create bar chart of Γ values
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(results.keys())
    gammas = [results[name].gamma for name in names]

    bars = ax.bar(
        names, gammas, color=colors[: len(names)], alpha=0.7, edgecolor="black"
    )

    # Add reference lines
    ax.axhline(
        0.82,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Well-fitted aEIF (0.82)",
    )
    ax.axhline(
        0.7, color="orange", linestyle="--", linewidth=1, label="Good model (0.7)"
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="Chance level (0)")

    ax.set_ylabel("Coincidence Factor (Γ)")
    ax.set_title("Parameter Set Comparison")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.2, 1.1)

    # Add value labels on bars
    for bar, gamma in zip(bars, gammas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{gamma:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        bar_path = save_path.replace(".pdf", "_bar.pdf")
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        print(f"Saved bar chart to {bar_path}")

    plt.show()


def main():
    # Data paths
    data_dir = "/Users/paulmayer/Projects/university/40_thesis/4_data/models/optimisations/mCP-dspn-e150917_c6_D1-manimal_1_n24_04102017_cel1/expdata/"
    voltage_file = data_dir + "ECBL_IDthresh_ch5_553.dat"
    current_file = data_dir + "ECBL_IDthresh_ch4_553.dat"

    # Load data
    time_exp, voltage_exp, current_exp, dt_ms = load_data(voltage_file, current_file)
    print(f"Loaded data: {len(time_exp)} samples, dt={dt_ms:.3f} ms")

    # Find stimulation window
    stim_start_idx, stim_end_idx = find_stim_window(current_exp)
    stim_current_pA = np.mean(current_exp[stim_start_idx:stim_end_idx])
    print(
        f"Stim window: {stim_start_idx} to {stim_end_idx}, current={stim_current_pA:.1f} pA"
    )

    # Crop to training window (500 ms max)
    max_duration_ms = 500.0
    stim_duration_ms = min((stim_end_idx - stim_start_idx) * dt_ms, max_duration_ms)
    stim_end_idx = stim_start_idx + int(stim_duration_ms / dt_ms)

    # Extract voltage during stimulation
    voltage_stim = voltage_exp[stim_start_idx:stim_end_idx]
    time_stim = time_exp[stim_start_idx:stim_end_idx] - time_exp[stim_start_idx]

    # Extract experimental features
    exp_features = extract_experimental_features(
        jnp.array(voltage_stim),
        dt_ms=dt_ms,
        stim_duration_ms=stim_duration_ms,
        spike_threshold_mv=-20.0,
    )

    # 1. Initial/default parameters (before training)
    initial_params = {
        "C_m": 200.0,
        "g_L": 10.0,
        "E_L": -68.0,
        "v_T": -45.0,
        "delta_T": 2.0,
        "v_threshold": -20.0,
        "v_reset": -55.0,
        "tau_w": 100.0,
        "a": 2.0,
        "b": 50.0,
    }

    # 4. Trained parameters from train_adex_mse.py (single trace, 200 epochs)
    trained_params_mse = {
        "C_m": 200.0,
        "g_L": 19.2713,
        "E_L": -60.8670,
        "delta_T": 2.0,
        "v_threshold": -20.0,
        "v_T": -34.2358,
        "v_reset": -53.3814,
        "tau_w": 90.2152,
        "a": 0.0831,
        "b": 3.2875,
    }

    # 4. Trained parameters from train_adex_guarino.py (single trace, 200 epochs)
    trained_params_guarino = {
        "C_m": 200.0,
        "g_L": 11.7734,
        "E_L": -69.7901,
        "delta_T": 2.0,
        "v_threshold": -20.0,
        "v_T": -42.6691,
        "v_reset": -58.6692,
        "tau_w": 100.7770,
        "a": 3.6024,
        "b": 3.2310,
    }

    param_sets = {
        "Initial": initial_params,
        "Trained (MSE)": trained_params_mse,
        "Trained (Guarino)": trained_params_guarino,
    }

    SPIKE_THRESHOLD = -35.0

    print("run")
    # Run comparison
    results = compare_parameter_sets(
        param_sets,
        voltage_exp,
        current_exp,
        time_exp,
        dt_ms,
        stim_start_idx,
        stim_end_idx,
        stim_current_pA,
        spike_threshold_mv=SPIKE_THRESHOLD,
        delta_ms=2.0,  # Standard from Jolivet et al.
    )

    print("plot")
    # Plot comparison
    plot_comparison(
        param_sets,
        voltage_exp,
        time_exp,
        dt_ms,
        stim_start_idx,
        stim_end_idx,
        stim_current_pA,
        results,
        spike_threshold_mv=SPIKE_THRESHOLD,
        save_path="coincidence_evaluation.pdf",
    )

    return results


if __name__ == "__main__":
    main()
