#!/usr/bin/env python3
"""Diagnose NaN/Inf gradients at epoch 0 in tonic benchmark.

Traces every step in the gradient computation chain:
1. Initial params → cell setup → trainable_params
2. Param transform: nudge → inverse (constrained → unconstrained)
3. Forward pass: unconstrained → forward → simulate → loss value
4. Backward pass: jax.value_and_grad → gradient values
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np

from ADoptEX.benchmark.methods import get_tonic_methods
from ADoptEX.benchmark.runner import (
    _build_loss_fn,
    generate_initial_params,
    prepare_target_data,
)
from ADoptEX.benchmark.sampling import TONIC_GROUND_TRUTHS
from ADoptEX.benchmark.scenarios import get_tonic_scenarios
from ADoptEX.core.parameters import PARAM_BOUNDS
from ADoptEX.training.trainer import (
    _build_param_transform,
    _compute_grad_norm,
    _nudge_from_bounds,
    _param_key_to_bounds_key,
    setup_trainable_cell_step,
)

# Suppress Jaxley noise
import logging
logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
logging.getLogger("jaxley").setLevel(logging.WARNING)


def check_nan_inf(name, obj):
    """Check a pytree for NaN/Inf values. Returns True if any found."""
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                for k, v in item.items():
                    arr = jnp.asarray(v)
                    has_nan = bool(jnp.any(jnp.isnan(arr)))
                    has_inf = bool(jnp.any(jnp.isinf(arr)))
                    status = ""
                    if has_nan:
                        status += " [NaN!]"
                    if has_inf:
                        status += " [Inf!]"
                    print(f"  {name}[{i}][{k}] = {arr.flatten()}{status}")
                    if has_nan or has_inf:
                        return True
    elif isinstance(obj, (int, float)):
        has_nan = np.isnan(obj)
        has_inf = np.isinf(obj)
        status = ""
        if has_nan:
            status += " [NaN!]"
        if has_inf:
            status += " [Inf!]"
        print(f"  {name} = {obj}{status}")
        if has_nan or has_inf:
            return True
    return False


def diagnose_single_run():
    """Run diagnosis for GT 0, perturbation 1%, Van Rossum method."""

    # Pick ground truth 0 with smallest perturbation
    scenarios = get_tonic_scenarios(n_starts=1, perturbations=[0.01])
    scenario = scenarios[0]  # gt00, 1% perturbation
    methods = get_tonic_methods()
    method = methods[0]  # grad_vanrossum

    print(f"Scenario: {scenario.name}")
    print(f"Method: {method.name}")
    print(f"Perturbation: {scenario.perturbation}")
    print()

    # Step 1: Generate initial params
    initial_params = generate_initial_params(scenario, start_index=0)
    print("=" * 70)
    print("STEP 1: Initial parameters")
    print("=" * 70)
    for k, v in sorted(initial_params.items()):
        gt = scenario.ground_truth_params.get(k, "N/A")
        print(f"  {k:12s} = {v:10.4f}  (GT: {gt})")
    print()

    # Step 2: Prepare target data (simulate ground truth)
    print("=" * 70)
    print("STEP 2: Simulate ground truth for target data")
    print("=" * 70)
    target_data = prepare_target_data(scenario)
    target_v = target_data["voltage"]
    print(f"  Target voltage: shape={target_v.shape}, "
          f"min={target_v.min():.2f}, max={target_v.max():.2f}")
    print(f"  Target spikes: {len(target_data['spike_times'])}")
    has_nan_target = np.any(np.isnan(target_v)) or np.any(np.isinf(target_v))
    print(f"  Target NaN/Inf: {has_nan_target}")
    print()

    # Step 3: Setup trainable cell
    print("=" * 70)
    print("STEP 3: Setup trainable cell")
    print("=" * 70)

    from dataclasses import replace
    config = method.training_config
    config = replace(config, trainable_params=scenario.trainable_params)

    import contextlib, io, builtins
    _real_print = builtins.print
    builtins.print = lambda *a, **kw: None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cell, _, t_max, trainable_params = setup_trainable_cell_step(
                initial_params=initial_params,
                stim_current_pA=scenario.stim_current_pA,
                stim_duration_ms=scenario.stim_duration_ms,
                stim_delay_ms=scenario.stim_delay_ms,
                dt_ms=scenario.dt_ms,
                config=config,
                trainable_params=scenario.trainable_params,
            )
    finally:
        builtins.print = _real_print

    _real_print(f"  t_max = {t_max} ms")
    _real_print(f"  Trainable params ({len(trainable_params)} dicts):")
    check_nan_inf("trainable_params", trainable_params)
    _real_print()

    # Step 4: Build loss function
    _real_print("=" * 70)
    _real_print("STEP 4: Build loss function")
    _real_print("=" * 70)
    loss_fn = _build_loss_fn(method, cell, t_max, target_data)
    _real_print(f"  Loss function built: {method.loss_type}")
    _real_print()

    # Step 5: Param transform (this is what train() does internally)
    _real_print("=" * 70)
    _real_print("STEP 5: Sigmoid param transform")
    _real_print("=" * 70)

    # 5a: Build transform
    param_transform = _build_param_transform(trainable_params, PARAM_BOUNDS)
    _real_print("  Transform built OK")

    # 5b: Nudge from bounds
    nudged = _nudge_from_bounds(trainable_params, PARAM_BOUNDS)
    _real_print("\n  After nudge_from_bounds:")
    nan_found = check_nan_inf("nudged", nudged)
    if nan_found:
        _real_print("  >>> NaN/Inf found after nudging! Stopping here.")
        return

    # 5c: Inverse transform (constrained → unconstrained)
    unconstrained = param_transform.inverse(nudged)
    _real_print("\n  After param_transform.inverse (unconstrained space):")
    nan_found = check_nan_inf("unconstrained", unconstrained)
    if nan_found:
        _real_print("  >>> NaN/Inf found after inverse transform! This is the bug.")
        _real_print("  Checking which params are at or near bounds...")
        for param_dict in nudged:
            for name, val in param_dict.items():
                bounds_key = _param_key_to_bounds_key(name)
                b = PARAM_BOUNDS[bounds_key]
                _real_print(f"    {name}: {float(val):.6f}  bounds=[{b.min}, {b.max}]  "
                           f"range_frac={(float(val)-b.min)/(b.max-b.min):.6f}")
        return
    _real_print()

    # Step 6: Forward pass with original loss_fn
    _real_print("=" * 70)
    _real_print("STEP 6: Forward pass (loss evaluation)")
    _real_print("=" * 70)

    # 6a: With original params (no transform)
    _real_print("  6a: Direct loss_fn(trainable_params) [no transform]:")
    try:
        loss_direct = loss_fn(trainable_params)
        _real_print(f"      loss = {float(loss_direct)}")
        check_nan_inf("loss_direct", float(loss_direct))
    except Exception as e:
        _real_print(f"      FAILED: {e}")

    # 6b: With transformed params
    _real_print("  6b: loss_fn(param_transform.forward(unconstrained)):")
    try:
        constrained_back = param_transform.forward(unconstrained)
        loss_transformed = loss_fn(constrained_back)
        _real_print(f"      loss = {float(loss_transformed)}")
        check_nan_inf("loss_transformed", float(loss_transformed))
    except Exception as e:
        _real_print(f"      FAILED: {e}")

    _real_print()

    # Step 7: Gradient computation
    _real_print("=" * 70)
    _real_print("STEP 7: Gradient computation (jax.value_and_grad)")
    _real_print("=" * 70)

    # 7a: Gradient without transform
    _real_print("  7a: Gradient of loss_fn w.r.t. trainable_params (no transform):")
    try:
        loss_val, grads = jax.value_and_grad(loss_fn)(trainable_params)
        _real_print(f"      loss = {float(loss_val)}")
        grad_norm = _compute_grad_norm(grads)
        _real_print(f"      grad_norm = {grad_norm}")
        check_nan_inf("grads", grads)
    except Exception as e:
        _real_print(f"      FAILED: {e}")

    # 7b: Gradient with transform (what train() actually does)
    _real_print("\n  7b: Gradient of transformed_loss w.r.t. unconstrained params:")
    try:
        transformed_loss_fn = lambda p: loss_fn(param_transform.forward(p))
        loss_val, grads = jax.value_and_grad(transformed_loss_fn)(unconstrained)
        _real_print(f"      loss = {float(loss_val)}")
        grad_norm = _compute_grad_norm(grads)
        _real_print(f"      grad_norm = {grad_norm}")
        check_nan_inf("grads_transformed", grads)
    except Exception as e:
        _real_print(f"      FAILED: {e}")

    _real_print()
    _real_print("=" * 70)
    _real_print("DIAGNOSIS COMPLETE")
    _real_print("=" * 70)


if __name__ == "__main__":
    diagnose_single_run()
