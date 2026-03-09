"""
AdEx Implementation Verification: Brian2 vs Jaxley

This script verifies that the Jaxley AdEx implementation matches Brian2
by comparing voltage traces and spike times across different parameter sets.

Uses the new modular structure from core/, plotting/.
"""

import matplotlib.pyplot as plt
from core import NAUD_PARAMETERS, simulate_brian2, simulate_jaxley
from plotting import plot_combined_comparison, plot_comparison


def run_single_comparison(
    param_set: str,
    duration_ms: float = 400.0,
    t_max_ms: float = 500.0,
    dt_ms: float = 0.01,
):
    """
    Run and plot comparison for a single parameter set.

    Args:
        param_set: Name of parameter set ('tonic', 'adaptation', 'original')
        duration_ms: Stimulus duration
        t_max_ms: Total simulation time
        dt_ms: Time step
    """
    params = NAUD_PARAMETERS[param_set]

    print(f"\n{'='*60}")
    print(f"Running comparison for: {param_set}")
    print(f"{'='*60}")

    # Run simulations
    result_brian2 = simulate_brian2(
        params,
        stim_current_pA=params["I"],
        stim_duration_ms=duration_ms,
        dt_ms=dt_ms,
        t_max_ms=t_max_ms,
    )

    result_jaxley = simulate_jaxley(
        params,
        stim_current_pA=params["I"],
        stim_duration_ms=duration_ms,
        dt_ms=dt_ms,
        t_max_ms=t_max_ms,
        use_surrogate=False,  # Use standard AdEx for verification
    )

    # Print comparison
    print(
        f"Brian2:  {result_brian2.n_spikes} spikes, "
        f"rate = {result_brian2.firing_rate_hz:.1f} Hz"
    )
    print(
        f"Jaxley:  {result_jaxley.n_spikes} spikes, "
        f"rate = {result_jaxley.firing_rate_hz:.1f} Hz"
    )

    if result_brian2.n_spikes > 0 and result_jaxley.n_spikes > 0:
        # Compare first spike times
        t1_brian = result_brian2.spike_times[0]
        t1_jaxley = result_jaxley.spike_times[0]
        print(
            f"First spike: Brian2={t1_brian:.2f}ms, Jaxley={t1_jaxley:.2f}ms, "
            f"diff={abs(t1_brian - t1_jaxley):.3f}ms"
        )

    # Plot
    fig = plot_comparison(
        result_brian2,
        result_jaxley,
        title=f"AdEx Comparison: {param_set.capitalize()} Spiking",
    )
    plt.savefig(f"new_adex_comparison_{param_set}.pdf", dpi=150, bbox_inches="tight")
    plt.show()

    return result_brian2, result_jaxley


def run_all_comparisons(
    param_sets: list[str] = ["tonic", "adaptation", "original"],
    duration_ms: float = 400.0,
    t_max_ms: float = 500.0,
    dt_ms: float = 0.01,
):
    """
    Run comparisons for all parameter sets and create combined plot.

    Args:
        param_sets: List of parameter set names
        duration_ms: Stimulus duration
        t_max_ms: Total simulation time
        dt_ms: Time step

    Returns:
        Dictionary of results
    """
    results = {}

    for param_set in param_sets:
        params = NAUD_PARAMETERS[param_set]

        result_brian2 = simulate_brian2(
            params,
            stim_current_pA=params["I"],
            stim_duration_ms=duration_ms,
            dt_ms=dt_ms,
            t_max_ms=t_max_ms,
        )

        result_jaxley = simulate_jaxley(
            params,
            stim_current_pA=params["I"],
            stim_duration_ms=duration_ms,
            dt_ms=dt_ms,
            t_max_ms=t_max_ms,
            use_surrogate=False,
        )

        results[param_set.capitalize()] = (result_brian2, result_jaxley)

        print(
            f"{param_set:12s}: Brian2={result_brian2.n_spikes} spikes, "
            f"Jaxley={result_jaxley.n_spikes} spikes"
        )

    # Create combined comparison plot
    fig = plot_combined_comparison(results, figsize=(14, 6))
    plt.savefig("new_adex_comparison_combined.pdf", dpi=150, bbox_inches="tight")
    plt.show()

    return results


def verify_surrogate_gradients():
    import jax
    import jax.numpy as jnp
    import jaxley as jx
    from core.simulation import geometry_for_capacitance
    from jaxley.channels import AdEx, AdExSurrogate

    params = NAUD_PARAMETERS["tonic"]

    radius_um, length_um = geometry_for_capacitance(params["C_m"])
    dt_ms = 0.1
    duration_ms = 400.0
    t_max_ms = 500.0
    I_nA = params["I"] * params["C_m"] / 1000.0

    # TODO: change this cell creation -> use create_cell or create_adex_cell or create_trainiable_cell or whatever
    # not enough time to do this now but cleab all these evaluation shit up!!!

    cell_surrogate = jx.Cell()
    cell_surrogate.set("radius", radius_um)
    cell_surrogate.set("length", length_um)
    cell_surrogate.insert(AdExSurrogate(surrogate_type="sigmoid", surrogate_slope=2.5))

    # Set parameters
    cell_surrogate.set("capacitance", params["C_m"])
    cell_surrogate.set("AdEx_C_m", params["C_m"])
    cell_surrogate.set("AdEx_g_L", params["g_L"])
    cell_surrogate.set("AdEx_E_L", params["E_L"])
    cell_surrogate.set("AdEx_v_T", params["v_T"])
    cell_surrogate.set("AdEx_delta_T", params["delta_T"])
    cell_surrogate.set("AdEx_v_threshold", params["v_threshold"])
    cell_surrogate.set("AdEx_v_reset", params["v_reset"])
    cell_surrogate.set("AdEx_tau_w", params["tau_w"])
    cell_surrogate.set("AdEx_a", params["a"])
    cell_surrogate.set("AdEx_b", params["b"])
    cell_surrogate.set("v", params["v_reset"])

    cell_surrogate.record("v")
    cell_surrogate.record("AdEx_spikes")

    # Make v_threshold trainable (we'll compute gradient w.r.t. this)
    cell_surrogate.make_trainable("AdEx_v_threshold")
    trainable_params_surrogate = cell_surrogate.get_parameters()

    # Setup stimulus
    cell_surrogate.stimulate(
        jx.step_current(0.0, duration_ms, I_nA, dt_ms, t_max=t_max_ms)
    )

    def loss_surrogate(params):
        results = jx.integrate(
            cell_surrogate, params=params, delta_t=dt_ms, t_max=t_max_ms
        )
        spikes = results[1].flatten()  # index 1 = AdEx_spikes
        return -jnp.sum(spikes)  # Negative to maximize spikes

    loss_val, grad_surrogate = jax.value_and_grad(loss_surrogate)(
        trainable_params_surrogate
    )
    grad_value_surrogate = float(grad_surrogate[0]["AdEx_v_threshold"][0])
    print(f"   Loss: {loss_val:.4f}")
    print(f"   Gradient w.r.t. v_threshold: {grad_value_surrogate:.6e}")

    cell_standard = jx.Cell()
    cell_standard.set("radius", radius_um)
    cell_standard.set("length", length_um)
    cell_standard.insert(AdEx())

    # Set same parameters
    cell_standard.set("capacitance", params["C_m"])
    cell_standard.set("AdEx_C_m", params["C_m"])
    cell_standard.set("AdEx_g_L", params["g_L"])
    cell_standard.set("AdEx_E_L", params["E_L"])
    cell_standard.set("AdEx_v_T", params["v_T"])
    cell_standard.set("AdEx_delta_T", params["delta_T"])
    cell_standard.set("AdEx_v_threshold", params["v_threshold"])
    cell_standard.set("AdEx_v_reset", params["v_reset"])
    cell_standard.set("AdEx_tau_w", params["tau_w"])
    cell_standard.set("AdEx_a", params["a"])
    cell_standard.set("AdEx_b", params["b"])
    cell_standard.set("v", params["v_reset"])

    cell_standard.record("v")
    cell_standard.record("AdEx_spikes")

    cell_standard.make_trainable("AdEx_v_threshold")
    trainable_params_standard = cell_standard.get_parameters()

    cell_standard.stimulate(
        jx.step_current(0.0, duration_ms, I_nA, dt_ms, t_max=t_max_ms)
    )

    def loss_standard(params):
        results = jx.integrate(
            cell_standard, params=params, delta_t=dt_ms, t_max=t_max_ms
        )
        spikes = results[1].flatten()  # index 1 = AdEx_spikes
        return -jnp.sum(spikes)  # Negative to maximize spikes

    loss_val_std, grad_standard = jax.value_and_grad(loss_standard)(
        trainable_params_standard
    )
    grad_value_standard = float(grad_standard[0]["AdEx_v_threshold"][0])

    print(f"   Loss: {loss_val_std:.4f}")
    print(f"   Gradient w.r.t. v_threshold: {grad_value_standard:.6e}")


if __name__ == "__main__":
    # Run individual comparisons
    # run_single_comparison('tonic')
    # run_single_comparison('adaptation')
    # run_single_comparison('original')

    # Run combined comparison
    results = run_all_comparisons()  # plot used in report

    # Verify surrogate gradients
    verify_surrogate_gradients()
