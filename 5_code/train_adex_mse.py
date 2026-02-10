"""
Simple AdEx training with MSE loss.

Trains on a single current/voltage trace.
"""

import sys

import jax
import jax.numpy as jnp
import jaxley as jx
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import config
from jaxley.channels import AdExSurrogate

from loss import MSELossConfig, mse_loss

# sys.path.insert(0, '/Users/paulmayer/Projects/university/35_project_course/3_jaxley')




# =============================================================================
# Setup
# =============================================================================


def geometry_for_capacitance(C_pF, specific_capacitance=1.0):
    """Calculate cylindrical cell geometry to achieve target total capacitance."""
    C_uF = C_pF * 1e-6
    area_cm2 = C_uF / specific_capacitance
    radius_cm = np.sqrt(area_cm2 / (6.28 * np.pi))
    radius_um = radius_cm * 1e4
    length_um = 3.14 * radius_um
    return radius_um, length_um


def load_data(voltage_path, current_path):
    """Load voltage and current traces from files."""
    v_data = np.loadtxt(voltage_path)
    i_data = np.loadtxt(current_path)

    time = v_data[:, 0]
    voltage = v_data[:, 1]
    current = i_data[:, 1]
    dt_ms = time[1] - time[0]

    return time, voltage, current, dt_ms


def find_stim_window(current, threshold_pA=100.0):
    """Find stimulation start and end indices."""
    above = current > threshold_pA
    indices = np.where(above)[0]
    if len(indices) == 0:
        return 0, len(current)
    return indices[0], indices[-1]


def create_cell(params):
    """Create AdEx cell with given parameters."""
    radius_um, length_um = geometry_for_capacitance(params["C_m"])

    cell = jx.Cell()
    cell.set("radius", radius_um)
    cell.set("length", length_um)

    cell.insert(AdExSurrogate(surrogate_type="sigmoid", surrogate_slope=25.0))

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


# =============================================================================
# Training
# =============================================================================


def train(
    cell,
    t_max,
    dt_ms,
    exp_voltage,
    stim_end_index,
    trainable_params,
    n_epochs=200,
    lr=0.005,
):
    """Train AdEx parameters using MSE loss."""

    loss_config = MSELossConfig(normalize=False, epsilon=1e-6)

    # Convert experimental voltage to JAX array
    exp_voltage_jax = jnp.array(exp_voltage)

    def loss_fn(params):
        results = jx.integrate(cell, params=params, delta_t=dt_ms, t_max=t_max)

        voltage = results[0].flatten()

        min_len = min(len(voltage), stim_end_index + 100, len(exp_voltage_jax))
        voltage = voltage[:min_len]
        exp_v = exp_voltage_jax[:min_len]

        return mse_loss(voltage, exp_v, loss_config)

    optimizer = optax.chain(optax.clip_by_global_norm(2.0), optax.adam(lr))
    opt_state = optimizer.init(trainable_params)

    loss_history = []
    best_loss = float("inf")
    best_params = trainable_params
    best_epoch = 0

    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 50)

    for epoch in range(n_epochs):
        loss_val, grads = jax.value_and_grad(loss_fn)(trainable_params)

        # Skip NaN updates
        has_nan = any(jnp.any(jnp.isnan(g[k])) for g in grads for k in g)
        if has_nan or jnp.isnan(loss_val):
            print(f"  Epoch {epoch}: NaN detected, skipping")
            continue

        updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)

        loss_history.append(float(loss_val))

        # Checkpoint best parameters
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = trainable_params
            best_epoch = epoch

        if epoch % 20 == 0:
            print(f"Epoch {epoch:4d}: loss={loss_val:.4f}")

    print("-" * 50)
    print(
        f"Final loss: {loss_history[-1]:.4f}, Best: {best_loss:.4f} (epoch {best_epoch})"
    )

    return best_params, loss_history


# =============================================================================
# Plotting
# =============================================================================


def plot_results(time_exp, voltage_exp, time_sim, voltage_sim, loss_history):
    """Plot comparison and training loss."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(time_exp, voltage_exp, "gray", label="Experimental", alpha=0.7)
    axes[0].plot(time_sim, voltage_sim, "b", label="Simulated", alpha=0.8)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].legend()
    axes[0].set_title("Voltage Comparison")

    axes[1].plot(loss_history)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].set_title("Training Loss (MSE)")

    plt.tight_layout()
    plt.savefig("mse_training_results.pdf", dpi=150)
    plt.show()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    config.update("jax_platform_name", "cpu")

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

    print(f"Training on {len(voltage_stim)} samples ({stim_duration_ms:.1f} ms)")

    # Initial parameters
    params = {
        "C_m": 200.0,
        "g_L": 10.0,
        "E_L": -68.0,
        "v_T": -45.0,
        "delta_T": 2.0,
        "v_threshold": -20.0,
        "v_reset": -55.0,
        "tau_w": 100.0,
        "a": 2.0,
        "b": 1.0,
    }

    # Create cell
    cell = create_cell(params)

    # Setup stimulus
    I_nA = stim_current_pA * params["C_m"] / 1000.0
    t_max = stim_duration_ms + 100
    stim_end_index = int(stim_duration_ms / dt_ms)

    cell.stimulate(jx.step_current(0.0, stim_duration_ms, I_nA, dt_ms, t_max=t_max))

    # Make parameters trainable
    # cell.make_trainable("AdEx_C_m")
    cell.make_trainable("AdEx_g_L")
    cell.make_trainable("AdEx_E_L")
    cell.make_trainable("AdEx_v_T")
    cell.make_trainable("AdEx_v_reset")
    cell.make_trainable("AdEx_tau_w")
    cell.make_trainable("AdEx_a")
    cell.make_trainable("AdEx_b")
    # cell.make_trainable("radius")
    # cell.make_trainable("length")

    trainable_params = cell.get_parameters()

    # Train
    trained_params, loss_history = train(
        cell,
        t_max,
        dt_ms,
        voltage_stim,
        stim_end_index,
        trainable_params,
        n_epochs=500,
        lr=0.1,
    )

    # Final simulation
    results = jx.integrate(cell, params=trained_params, delta_t=dt_ms, t_max=t_max)
    voltage_sim = np.array(results[0]).flatten()
    time_sim = np.arange(len(voltage_sim)) * dt_ms

    # Print trained parameters
    print("\nTrained Parameters:")
    for p in trained_params:
        for k, v in p.items():
            print(f"  {k}: {float(v.flatten()[0]):.4f}")

    # Compute final MSE
    min_len = min(len(voltage_sim), len(voltage_stim))
    final_mse = np.mean((voltage_sim[:min_len] - voltage_stim[:min_len]) ** 2)
    print(f"\nFinal MSE: {final_mse:.4f}")

    # Plot
    plot_results(time_stim, voltage_stim, time_sim, voltage_sim, loss_history)
