#!/usr/bin/env python3
"""Deeper NaN gradient diagnosis.

Finding so far: forward pass is fine, backward pass produces NaN.
Now: print ALL gradients, and test with varying simulation lengths.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import contextlib
import io
import builtins
import logging

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.WARNING)

from ADoptEX.benchmark.methods import get_tonic_methods
from ADoptEX.benchmark.runner import _build_loss_fn, generate_initial_params, prepare_target_data
from ADoptEX.benchmark.scenarios import get_tonic_scenarios
from ADoptEX.core.parameters import PARAM_BOUNDS
from ADoptEX.training.trainer import (
    _build_param_transform,
    _compute_grad_norm,
    _nudge_from_bounds,
    _param_key_to_bounds_key,
    setup_trainable_cell_step,
)


def suppress_prints():
    """Context manager to suppress Jaxley prints."""
    _real = builtins.print
    builtins.print = lambda *a, **kw: None
    class CM:
        def __enter__(self): return self
        def __exit__(self, *args): builtins.print = _real
    return CM()


def print_all_grads(label, grads):
    """Print all gradient values."""
    for i, d in enumerate(grads):
        for name, val in d.items():
            arr = jnp.asarray(val).flatten()
            status = ""
            if jnp.any(jnp.isnan(arr)):
                status = " *** NaN ***"
            elif jnp.any(jnp.isinf(arr)):
                status = " *** Inf ***"
            print(f"  {label} [{_param_key_to_bounds_key(name):8s}] = {float(arr[0]):15.6f}{status}")


def test_gradient_all_params():
    """Test gradient for all parameters."""
    print("=" * 70)
    print("TEST 1: All gradient values (no transform)")
    print("=" * 70)

    scenarios = get_tonic_scenarios(n_starts=1, perturbations=[0.01])
    scenario = scenarios[0]
    methods = get_tonic_methods()
    method = methods[0]  # grad_vanrossum

    initial_params = generate_initial_params(scenario, start_index=0)

    from dataclasses import replace
    config = replace(method.training_config, trainable_params=scenario.trainable_params)

    with suppress_prints():
        cell, _, t_max, trainable_params = setup_trainable_cell_step(
            initial_params=initial_params,
            stim_current_pA=scenario.stim_current_pA,
            stim_duration_ms=scenario.stim_duration_ms,
            stim_delay_ms=scenario.stim_delay_ms,
            dt_ms=scenario.dt_ms,
            config=config,
            trainable_params=scenario.trainable_params,
        )

    from ADoptEX.benchmark.runner import _build_loss_fn, prepare_target_data
    target_data = prepare_target_data(scenario)
    loss_fn = _build_loss_fn(method, cell, t_max, target_data)

    # Forward check
    loss_val = loss_fn(trainable_params)
    print(f"  Forward loss = {float(loss_val):.6f}")

    # Gradient check
    loss_val, grads = jax.value_and_grad(loss_fn)(trainable_params)
    print(f"  Loss from value_and_grad = {float(loss_val):.6f}")
    print_all_grads("grad", grads)
    print()


def test_gradient_per_param():
    """Test gradient for each parameter individually to find the offending one."""
    print("=" * 70)
    print("TEST 2: Gradient per parameter (isolate the NaN source)")
    print("=" * 70)

    scenarios = get_tonic_scenarios(n_starts=1, perturbations=[0.01])
    scenario = scenarios[0]
    methods = get_tonic_methods()
    method = methods[0]

    from dataclasses import replace

    # Test each param individually
    for param_name in scenario.trainable_params:
        initial_params = generate_initial_params(scenario, start_index=0)
        config = replace(method.training_config, trainable_params=[param_name])

        with suppress_prints():
            cell, _, t_max, trainable_params = setup_trainable_cell_step(
                initial_params=initial_params,
                stim_current_pA=scenario.stim_current_pA,
                stim_duration_ms=scenario.stim_duration_ms,
                stim_delay_ms=scenario.stim_delay_ms,
                dt_ms=scenario.dt_ms,
                config=config,
                trainable_params=[param_name],
            )

        target_data = prepare_target_data(scenario)
        loss_fn = _build_loss_fn(method, cell, t_max, target_data)

        loss_val, grads = jax.value_and_grad(loss_fn)(trainable_params)
        grad_norm = _compute_grad_norm(grads)
        status = ""
        if jnp.isnan(grad_norm):
            status = " *** NaN ***"
        elif jnp.isinf(grad_norm):
            status = " *** Inf ***"
        print(f"  {param_name:12s}: loss={float(loss_val):.6f}  grad_norm={grad_norm:.6e}{status}")

    print()


def test_gradient_varying_dt():
    """Test if larger dt (fewer timesteps) avoids NaN."""
    print("=" * 70)
    print("TEST 3: Varying dt (fewer timesteps)")
    print("=" * 70)

    scenarios = get_tonic_scenarios(n_starts=1, perturbations=[0.01])
    scenario = scenarios[0]
    methods = get_tonic_methods()
    method = methods[0]

    from dataclasses import replace

    for dt in [0.025, 0.05, 0.1, 0.5, 1.0]:
        initial_params = generate_initial_params(scenario, start_index=0)
        config = replace(method.training_config, trainable_params=scenario.trainable_params)

        # Manually create scenario with different dt
        from ADoptEX.benchmark.scenarios import SyntheticScenario
        test_scenario = SyntheticScenario(
            name=scenario.name,
            ground_truth_params=scenario.ground_truth_params,
            trainable_params=scenario.trainable_params,
            stim_current_pA=scenario.stim_current_pA,
            stim_duration_ms=scenario.stim_duration_ms,
            stim_delay_ms=scenario.stim_delay_ms,
            perturbation=scenario.perturbation,
            n_starts=1,
            dt_ms=dt,
        )

        with suppress_prints():
            cell, _, t_max, trainable_params = setup_trainable_cell_step(
                initial_params=initial_params,
                stim_current_pA=test_scenario.stim_current_pA,
                stim_duration_ms=test_scenario.stim_duration_ms,
                stim_delay_ms=test_scenario.stim_delay_ms,
                dt_ms=dt,
                config=config,
                trainable_params=test_scenario.trainable_params,
            )

        target_data = prepare_target_data(test_scenario)
        loss_fn = _build_loss_fn(method, cell, t_max, target_data)

        n_timesteps = int(t_max / dt)
        loss_val, grads = jax.value_and_grad(loss_fn)(trainable_params)
        grad_norm = _compute_grad_norm(grads)
        status = ""
        if jnp.isnan(grad_norm):
            status = " *** NaN ***"
        elif jnp.isinf(grad_norm):
            status = " *** Inf ***"
        print(f"  dt={dt:5.3f}ms ({n_timesteps:6d} steps): loss={float(loss_val):.4f}  grad_norm={grad_norm:.4e}{status}")

    print()


def test_gradient_shorter_sim():
    """Test gradient with shorter simulation duration."""
    print("=" * 70)
    print("TEST 4: Shorter simulation durations")
    print("=" * 70)

    scenarios = get_tonic_scenarios(n_starts=1, perturbations=[0.01])
    scenario = scenarios[0]
    methods = get_tonic_methods()
    method = methods[0]

    from dataclasses import replace

    for stim_dur in [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0]:
        initial_params = generate_initial_params(scenario, start_index=0)
        config = replace(method.training_config, trainable_params=scenario.trainable_params)

        from ADoptEX.benchmark.scenarios import SyntheticScenario
        test_scenario = SyntheticScenario(
            name=scenario.name,
            ground_truth_params=scenario.ground_truth_params,
            trainable_params=scenario.trainable_params,
            stim_current_pA=scenario.stim_current_pA,
            stim_duration_ms=stim_dur,
            stim_delay_ms=scenario.stim_delay_ms,
            perturbation=scenario.perturbation,
            n_starts=1,
            dt_ms=0.025,
        )

        with suppress_prints():
            cell, _, t_max, trainable_params = setup_trainable_cell_step(
                initial_params=initial_params,
                stim_current_pA=test_scenario.stim_current_pA,
                stim_duration_ms=stim_dur,
                stim_delay_ms=test_scenario.stim_delay_ms,
                dt_ms=0.025,
                config=config,
                trainable_params=test_scenario.trainable_params,
            )

        target_data = prepare_target_data(test_scenario)
        loss_fn = _build_loss_fn(method, cell, t_max, target_data)

        n_timesteps = int(t_max / 0.025)
        loss_val, grads = jax.value_and_grad(loss_fn)(trainable_params)
        grad_norm = _compute_grad_norm(grads)
        status = ""
        if jnp.isnan(grad_norm):
            status = " *** NaN ***"
        elif jnp.isinf(grad_norm):
            status = " *** Inf ***"
        print(f"  stim={stim_dur:6.1f}ms ({n_timesteps:6d} steps): loss={float(loss_val):.4f}  grad_norm={grad_norm:.4e}{status}")

    print()


if __name__ == "__main__":
    test_gradient_all_params()
    # test_gradient_per_param()
    test_gradient_varying_dt()
    test_gradient_shorter_sim()
