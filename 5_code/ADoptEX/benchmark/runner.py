"""
Execution engine for benchmark runs.

Handles initial parameter generation, target data preparation,
gradient and Nelder-Mead optimization, and metric evaluation.

Synthetic scenarios use step current stimulation via jx.step_current.
"""

import contextlib
import io
import logging
import time
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

from ADoptEX.core.parameters import PARAM_BOUNDS
from ADoptEX.core.simulation import simulate_jaxley
from ADoptEX.evaluation.coincidence import coincidence_factor
from ADoptEX.loss import inject_spike_peaks

from .methods import MethodConfig
from .results import RunResult, ScenarioResults, save_results
from .scenarios import SyntheticScenario

log = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_jaxley_prints():
    """Suppress stdout prints from Jaxley (recordings, trainable params, externals).

    Uses both redirect_stdout and a builtins.print patch since Jaxley calls
    print() directly and some code paths may capture stdout before redirect.
    """
    import builtins

    _real_print = builtins.print
    builtins.print = lambda *a, **kw: None  # type: ignore[assignment]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        builtins.print = _real_print


# ── Initial Parameter Generation ────────────────────────────────────────────


def generate_initial_params(
    scenario: SyntheticScenario,
    start_index: int,
) -> dict[str, float]:
    """Generate reproducible initial parameters using Latin Hypercube Sampling.

    Perturbs the ground truth parameters within bounds.

    Args:
        scenario: Scenario definition.
        start_index: Index of this start (0 to n_starts-1).

    Returns:
        Dictionary of initial parameter values.
    """
    rng = np.random.default_rng(scenario.seed + start_index)

    base_params = scenario.ground_truth_params
    perturbation = scenario.perturbation
    trainable = scenario.trainable_params
    n_params = len(trainable)

    # Use LHS to generate space-filling sample in [0, 1]^n
    sampler = LatinHypercube(d=n_params, seed=rng)
    sample = sampler.random(n=1)[0]  # shape (n_params,)

    initial = {}
    for i, param_name in enumerate(trainable):
        base_val = base_params.get(param_name, 0.0)
        bounds = PARAM_BOUNDS[param_name]

        # Perturb: map LHS sample from [0,1] to [-perturbation, +perturbation]
        rel_perturbation = (2 * sample[i] - 1) * perturbation
        perturbed = base_val * (1.0 + rel_perturbation)

        # Clamp to bounds
        initial[param_name] = bounds.clip(perturbed)

    # Include non-trainable params from base
    for k, v in base_params.items():
        if k not in initial:
            initial[k] = v

    return initial


# ── Target Data Preparation ─────────────────────────────────────────────────


def prepare_target_data(scenario: SyntheticScenario) -> dict:
    """Simulate ground truth with step current for a synthetic scenario.

    Args:
        scenario: Scenario definition with stimulus parameters.

    Returns:
        Dictionary with keys:
        - 'voltage': target voltage trace (np.ndarray)
        - 'spike_times': target spike times in ms (np.ndarray)
        - 'dt_ms': time step
        - 'duration_ms': total simulation duration
        - 'stim_end_index': index of stimulus end
        - 'stim_current_pA': stimulus amplitude
        - 'stim_duration_ms': stimulus duration
        - 'stim_delay_ms': stimulus delay
    """
    with _suppress_jaxley_prints():
        result = simulate_jaxley(
            params=scenario.ground_truth_params,
            stim_current_pA=scenario.stim_current_pA,
            stim_duration_ms=scenario.stim_duration_ms,
            dt_ms=scenario.dt_ms,
            t_max_ms=scenario.t_max_ms,
            stim_delay_ms=scenario.stim_delay_ms,
            use_surrogate=False,
        )

    stim_end_index = int(
        (scenario.stim_delay_ms + scenario.stim_duration_ms) / scenario.dt_ms
    )

    # Jaxley's continuous reset means voltage never produces tall spike peaks.
    # Inject artificial peaks at spike times so threshold-based feature
    # extraction (guarino, van_rossum) works on the target voltage.
    voltage = inject_spike_peaks(
        jnp.array(result.voltage), jnp.array(result.spikes), v_peak_mv=35.0
    )

    return {
        "voltage": np.array(voltage),
        "spike_times": result.spike_times,
        "dt_ms": scenario.dt_ms,
        "duration_ms": scenario.t_max_ms,
        "stim_end_index": stim_end_index,
        "stim_current_pA": scenario.stim_current_pA,
        "stim_duration_ms": scenario.stim_duration_ms,
        "stim_delay_ms": scenario.stim_delay_ms,
    }


# ── Metric Evaluation ───────────────────────────────────────────────────────


def _evaluate_metrics(
    scenario: SyntheticScenario,
    final_params: dict[str, float],
    target_data: dict,
) -> dict:
    """Evaluate a fitted model against target data.

    Returns dict with gamma, spike counts, timing errors, and param recovery.
    """
    with _suppress_jaxley_prints():
        result = simulate_jaxley(
            params=final_params,
            stim_current_pA=target_data["stim_current_pA"],
            stim_duration_ms=target_data["stim_duration_ms"],
            dt_ms=target_data["dt_ms"],
            t_max_ms=target_data["duration_ms"],
            stim_delay_ms=target_data["stim_delay_ms"],
            use_surrogate=True,
            surrogate_slope=5.0,
        )

    model_spikes = result.spike_times
    target_spikes = target_data["spike_times"]
    duration_ms = target_data["duration_ms"]

    # Coincidence factor
    gamma_result = coincidence_factor(target_spikes, model_spikes, duration_ms)

    # Spike counts
    n_model = len(model_spikes)
    n_target = len(target_spikes)

    # First spike error
    first_spike_error = float("nan")
    if n_model > 0 and n_target > 0:
        first_spike_error = abs(float(model_spikes[0]) - float(target_spikes[0]))

    # Parameter recovery
    param_recovery = {}
    for p in scenario.trainable_params:
        gt_val = scenario.ground_truth_params.get(p, 0.0)
        fit_val = final_params.get(p, 0.0)
        abs_err = abs(fit_val - gt_val)
        pct_err = abs_err / abs(gt_val) * 100 if gt_val != 0 else float("nan")
        param_recovery[p] = {"error_abs": abs_err, "error_pct": pct_err}

    return {
        "gamma": gamma_result.gamma,
        "n_spikes_model": n_model,
        "n_spikes_target": n_target,
        "spike_count_error": n_model - n_target,
        "first_spike_error_ms": first_spike_error,
        "param_recovery": param_recovery,
    }


# ── Loss Function Construction ──────────────────────────────────────────────


def _build_loss_fn(method: MethodConfig, cell, t_max, target_data):
    """Build a loss function closure for the given method and target data.

    For step current stimulation, data_stimuli is None — the cell already
    has the stimulus applied via cell.stimulate().
    """
    dt_ms = target_data["dt_ms"]
    stim_end_index = target_data["stim_end_index"]

    if method.loss_type == "van_rossum":
        from ADoptEX.loss.van_rossum import make_van_rossum_loss_fn, spike_train_from_voltage

        exp_spike_train = spike_train_from_voltage(
            jnp.array(target_data["voltage"]),
            threshold_mv=0.0,
            dt_ms=dt_ms,
        )
        return make_van_rossum_loss_fn(
            cell=cell,
            data_stimuli=None,
            t_max=t_max,
            dt_ms=dt_ms,
            exp_spike_train=exp_spike_train,
            stim_end_index=stim_end_index,
            loss_config=method.loss_config,
            exp_voltage=jnp.array(target_data["voltage"]),
        )

    elif method.loss_type == "guarino":
        from ADoptEX.loss.guarino import (
            extract_experimental_features,
            make_guarino_loss_fn,
        )

        stim_duration_ms = target_data["stim_duration_ms"]
        exp_features = extract_experimental_features(
            voltage_trace=jnp.array(target_data["voltage"]),
            dt_ms=dt_ms,
            stim_duration_ms=stim_duration_ms,
            stim_end_index=stim_end_index,
        )
        return make_guarino_loss_fn(
            cell=cell,
            data_stimuli=None,
            t_max=t_max,
            dt_ms=dt_ms,
            exp_features=exp_features,
            stim_duration_ms=stim_duration_ms,
            stim_end_index=stim_end_index,
            loss_config=method.loss_config,
        )

    elif method.loss_type == "mse":
        from ADoptEX.loss.mse import make_mse_loss_fn

        return make_mse_loss_fn(
            cell=cell,
            data_stimuli=None,
            t_max=t_max,
            dt_ms=dt_ms,
            exp_voltage=jnp.array(target_data["voltage"]),
            stim_end_index=stim_end_index,
            loss_config=method.loss_config,
        )

    else:
        raise ValueError(f"Unknown loss type: {method.loss_type}")


# ── Gradient Optimization ───────────────────────────────────────────────────


def run_gradient(
    scenario: SyntheticScenario,
    method: MethodConfig,
    initial_params: dict[str, float],
    target_data: dict,
    cell=None,
    loss_fn=None,
    grad_fn=None,
    config=None,
    t_max: float | None = None,
) -> RunResult:
    """Run gradient-based optimization for a single start.

    Uses step current stimulation via setup_trainable_cell_step.

    When *cell*, *loss_fn*, and *grad_fn* are provided (from
    ``run_scenario``), the expensive cell-creation and JIT-compilation
    steps are skipped -- only the initial parameter values change.
    """
    from ADoptEX.training.trainer import (
        make_initial_trainable_params,
        setup_trainable_cell_step,
        train,
    )

    if config is None:
        config = method.training_config
    assert config is not None, "training_config required for gradient methods"

    # Override trainable params from scenario
    config = replace(config, trainable_params=scenario.trainable_params)

    t0 = time.time()

    # Reuse pre-built cell or create a new one
    if cell is None:
        with _suppress_jaxley_prints():
            cell, _, t_max, trainable_params = setup_trainable_cell_step(
                initial_params=initial_params,
                stim_current_pA=scenario.stim_current_pA,
                stim_duration_ms=scenario.stim_duration_ms,
                stim_delay_ms=scenario.stim_delay_ms,
                dt_ms=scenario.dt_ms,
                config=config,
                trainable_params=scenario.trainable_params,
            )
    else:
        # Cell already exists -- just inject new initial values
        trainable_params = make_initial_trainable_params(cell, initial_params)

    # Reuse pre-built loss function or create a new one
    if loss_fn is None:
        loss_fn = _build_loss_fn(method, cell, t_max, target_data)

    # Run training (pass pre-compiled grad_fn if available)
    train_result = train(
        loss_fn=loss_fn,
        trainable_params=trainable_params,
        config=config,
        initial_params=initial_params,
        grad_fn=grad_fn,
    )

    wall_time = time.time() - t0

    # Extract final params
    final_params = train_result.get_params_dict()
    # Merge non-trainable params
    for k, v in initial_params.items():
        if k not in final_params:
            final_params[k] = v

    # Evaluate metrics
    metrics = _evaluate_metrics(scenario, final_params, target_data)

    return RunResult(
        scenario_name=scenario.name,
        method_name=method.name,
        start_index=-1,  # set by caller
        seed=scenario.seed,
        initial_params=initial_params,
        final_params=final_params,
        ground_truth_params=scenario.ground_truth_params,
        final_loss=train_result.final_loss,
        best_loss=train_result.best_loss,
        gamma=metrics["gamma"],
        n_spikes_model=metrics["n_spikes_model"],
        n_spikes_target=metrics["n_spikes_target"],
        spike_count_error=metrics["spike_count_error"],
        first_spike_error_ms=metrics["first_spike_error_ms"],
        param_recovery=metrics["param_recovery"],
        wall_time_s=wall_time,
        n_function_evals=config.n_epochs,
        time_per_eval_ms=wall_time / max(config.n_epochs, 1) * 1000,
        loss_history=train_result.loss_history,
        grad_norms=train_result.grad_norms,
    )


# ── Nelder-Mead Optimization ────────────────────────────────────────────────


def _param_to_jaxley_key(name: str) -> str:
    """Map standard param name to Jaxley column key for data_set()."""
    if name == "C_m":
        return "capacitance"
    return f"AdEx_{name}"


def _build_nm_sim_fn(cell, trainable_names, dt_ms, t_max_ms):
    """Build a JIT-compiled simulation function using data_set().

    The cell is created once; each call to the returned function only
    changes parameter values via ``data_set`` + ``param_state``, avoiding
    expensive cell recreation.
    """
    import jax
    import jaxley as jx

    jaxley_keys = [_param_to_jaxley_key(name) for name in trainable_names]

    def simulate(param_values):
        param_state = None
        for i, key in enumerate(jaxley_keys):
            param_state = cell.data_set(key, param_values[i], param_state)
        return jx.integrate(cell, param_state=param_state, delta_t=dt_ms, t_max=t_max_ms)

    return jax.jit(simulate)


def run_nelder_mead(
    scenario: SyntheticScenario,
    method: MethodConfig,
    initial_params: dict[str, float],
    target_data: dict,
    sim_fn=None,
) -> RunResult:
    """Run Nelder-Mead optimization for a single start.

    When *sim_fn* is provided (pre-built via ``_build_nm_sim_fn`` at the
    ``run_scenario`` level), the expensive cell creation and JIT compilation
    are skipped -- each function evaluation reuses the compiled simulation.
    """
    import jaxley as jx

    trainable_names = scenario.trainable_params
    dt_ms = scenario.dt_ms

    # Build JIT-compiled simulation if not provided
    if sim_fn is None:
        from ADoptEX.core.simulation import create_adex_cell

        with _suppress_jaxley_prints():
            cell = create_adex_cell(
                initial_params,
                use_surrogate=True,
                surrogate_slope=method.surrogate_slope,
                trainable=False,
                record=True,
            )
            I_nA = scenario.stim_current_pA / 1000.0
            current = jx.step_current(
                scenario.stim_delay_ms, scenario.stim_duration_ms,
                I_nA, dt_ms, t_max=scenario.t_max_ms,
            )
            cell.stimulate(current)
            sim_fn = _build_nm_sim_fn(cell, trainable_names, dt_ms, scenario.t_max_ms)

    # Build objective: param vector -> scalar loss
    loss_config = method.nm_loss_config
    loss_history = []

    def objective(x: np.ndarray) -> float:
        """Evaluate loss for a parameter vector (uses JIT-compiled sim)."""
        clipped = jnp.array([
            float(PARAM_BOUNDS[name].clip(float(x[i])))
            for i, name in enumerate(trainable_names)
        ])

        try:
            results = sim_fn(clipped)
            spikes = results[2].flatten()
            voltage = results[0].flatten()

            loss = _compute_nm_loss_from_traces(
                method.nm_objective, spikes, voltage, target_data, loss_config, dt_ms
            )
            loss_history.append(float(loss))
            return float(loss)
        except Exception as e:
            log.warning("NM eval failed: %s", e)
            loss_history.append(float("inf"))
            return float("inf")

    # Initial parameter vector
    x0 = np.array([initial_params[name] for name in trainable_names])

    t0 = time.time()

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxfev": method.nm_max_fev,
            "adaptive": True,
            "xatol": 1e-4,
            "fatol": 1e-6,
        },
    )

    wall_time = time.time() - t0

    # Extract final params
    final_params = dict(initial_params)
    for i, name in enumerate(trainable_names):
        bounds = PARAM_BOUNDS[name]
        final_params[name] = bounds.clip(float(result.x[i]))

    # Evaluate metrics
    metrics = _evaluate_metrics(scenario, final_params, target_data)

    return RunResult(
        scenario_name=scenario.name,
        method_name=method.name,
        start_index=-1,
        seed=scenario.seed,
        initial_params=initial_params,
        final_params=final_params,
        ground_truth_params=scenario.ground_truth_params,
        final_loss=float(result.fun),
        best_loss=min(loss_history) if loss_history else float(result.fun),
        gamma=metrics["gamma"],
        n_spikes_model=metrics["n_spikes_model"],
        n_spikes_target=metrics["n_spikes_target"],
        spike_count_error=metrics["spike_count_error"],
        first_spike_error_ms=metrics["first_spike_error_ms"],
        param_recovery=metrics["param_recovery"],
        wall_time_s=wall_time,
        n_function_evals=result.nfev,
        time_per_eval_ms=wall_time / max(result.nfev, 1) * 1000,
        loss_history=loss_history,
    )


def _compute_nm_loss(objective_type, sim_result, target_data, loss_config, dt_ms):
    """Compute the loss for Nelder-Mead evaluation (SimulationResult variant)."""
    return _compute_nm_loss_from_traces(
        objective_type,
        jnp.array(sim_result.spikes),
        jnp.array(sim_result.voltage),
        target_data,
        loss_config,
        dt_ms,
    )


def _compute_nm_loss_from_traces(
    objective_type, sim_spikes, sim_voltage, target_data, loss_config, dt_ms
):
    """Compute NM loss from raw traces (works with jx.integrate output)."""
    if objective_type == "van_rossum":
        from ADoptEX.loss.van_rossum import VanRossumLossConfig, spike_train_from_voltage, van_rossum_distance

        config = loss_config or VanRossumLossConfig()
        target_voltage = jnp.array(target_data["voltage"])
        exp_spikes = spike_train_from_voltage(target_voltage, threshold_mv=0.0, dt_ms=dt_ms)
        return van_rossum_distance(sim_spikes, exp_spikes, dt_ms, config)

    elif objective_type == "mse":
        from ADoptEX.loss.mse import MSELossConfig, mse_loss

        config = loss_config or MSELossConfig()
        target_v = jnp.array(target_data["voltage"])
        min_len = min(len(sim_voltage), len(target_v))
        return mse_loss(sim_voltage[:min_len], target_v[:min_len], config)

    elif objective_type == "guarino":
        raise NotImplementedError(
            "Guarino loss for Nelder-Mead not supported (requires cell object)"
        )
    else:
        raise ValueError(f"Unknown NM objective: {objective_type}")


# ── Orchestration ────────────────────────────────────────────────────────────


def _run_gradient_vmapped(
    scenario: SyntheticScenario,
    method: MethodConfig,
    target_data: dict,
    cell,
    loss_fn,
    config,
) -> list[RunResult]:
    """Run all gradient starts in parallel via ``jit(vmap(train_scan))``.

    Returns a list of :class:`RunResult` objects, one per start.
    """
    import jax

    from ADoptEX.training.trainer import make_initial_trainable_params, train_scan

    n_starts = scenario.n_starts
    log.info("  Running %d starts in parallel via vmap", n_starts)

    # Generate all initial params and stack into a batched pytree
    all_initial_params = [generate_initial_params(scenario, i) for i in range(n_starts)]
    all_trainable = [
        make_initial_trainable_params(cell, ip) for ip in all_initial_params
    ]
    batched_params = jax.tree.map(lambda *xs: jnp.stack(xs), *all_trainable)

    # vmap the scan-based training loop across the batch dimension
    def train_one(params):
        return train_scan(loss_fn, params, config)

    t0 = time.time()
    vmapped_train = jax.jit(jax.vmap(train_one))
    all_final_params, all_loss_histories = vmapped_train(batched_params)
    total_time = time.time() - t0

    log.info(
        "  vmap complete: %d starts in %.1fs (%.1fs per start equivalent)",
        n_starts, total_time, total_time / n_starts,
    )

    # Unstack results and evaluate metrics per start
    runs = []
    for i in range(n_starts):
        # Extract single-start params from batched result
        final_trainable_i = jax.tree.map(lambda x: x[i], all_final_params)
        loss_history_i = all_loss_histories[i]

        # Convert to standard param dict
        from ADoptEX.training.trainer import _param_key_to_bounds_key

        final_params = {}
        for param_dict in final_trainable_i:
            for name, value in param_dict.items():
                clean_name = _param_key_to_bounds_key(name)
                final_params[clean_name] = float(value.flatten()[0])
        # Merge non-trainable params
        for k, v in all_initial_params[i].items():
            if k not in final_params:
                final_params[k] = v

        # Evaluate metrics
        metrics = _evaluate_metrics(scenario, final_params, target_data)

        loss_list = [float(x) for x in loss_history_i]
        runs.append(
            RunResult(
                scenario_name=scenario.name,
                method_name=method.name,
                start_index=i,
                seed=scenario.seed,
                initial_params=all_initial_params[i],
                final_params=final_params,
                ground_truth_params=scenario.ground_truth_params,
                final_loss=loss_list[-1] if loss_list else float("nan"),
                best_loss=min(loss_list) if loss_list else float("nan"),
                gamma=metrics["gamma"],
                n_spikes_model=metrics["n_spikes_model"],
                n_spikes_target=metrics["n_spikes_target"],
                spike_count_error=metrics["spike_count_error"],
                first_spike_error_ms=metrics["first_spike_error_ms"],
                param_recovery=metrics["param_recovery"],
                wall_time_s=total_time / n_starts,
                n_function_evals=config.n_epochs,
                time_per_eval_ms=total_time / n_starts / max(config.n_epochs, 1) * 1000,
                loss_history=loss_list,
            )
        )

    return runs


def run_single(
    scenario: SyntheticScenario,
    method: MethodConfig,
    start_index: int,
    target_data: dict,
) -> RunResult:
    """Run a single optimization start."""
    initial_params = generate_initial_params(scenario, start_index)

    if method.method_type == "gradient":
        result = run_gradient(scenario, method, initial_params, target_data)
    elif method.method_type == "nelder_mead":
        result = run_nelder_mead(scenario, method, initial_params, target_data)
    else:
        raise ValueError(f"Unknown method type: {method.method_type}")

    result.start_index = start_index
    return result


def run_scenario(
    scenario: SyntheticScenario,
    method: MethodConfig,
    target_data: dict,
    output_dir: str | Path | None = None,
) -> ScenarioResults:
    """Run all starts for a (scenario, method) pair.

    For gradient methods, the cell, loss function, and JIT-compiled gradient
    function are created **once** and reused across all starts.  Only the
    initial parameter values change per start.
    """
    import jax

    log.info(
        "Running %s x %s (%d starts)",
        scenario.name,
        method.name,
        scenario.n_starts,
    )

    # ── Pre-build shared objects (cell, JIT-compiled functions) ──────────
    shared_cell = None
    shared_loss_fn = None
    shared_grad_fn = None
    shared_config = None
    shared_t_max = None
    shared_sim_fn = None  # for NM

    if method.method_type == "gradient":
        from ADoptEX.training.trainer import setup_trainable_cell_step

        shared_config = method.training_config
        assert shared_config is not None
        shared_config = replace(shared_config, trainable_params=scenario.trainable_params)

        # Create cell ONCE with ground-truth params (values don't matter,
        # they'll be overridden per start via make_initial_trainable_params)
        with _suppress_jaxley_prints():
            shared_cell, _, shared_t_max, _ = setup_trainable_cell_step(
                initial_params=scenario.ground_truth_params,
                stim_current_pA=scenario.stim_current_pA,
                stim_duration_ms=scenario.stim_duration_ms,
                stim_delay_ms=scenario.stim_delay_ms,
                dt_ms=scenario.dt_ms,
                config=shared_config,
                trainable_params=scenario.trainable_params,
            )

        # Build loss function ONCE (captures cell)
        shared_loss_fn = _build_loss_fn(method, shared_cell, shared_t_max, target_data)

        # JIT-compile gradient function ONCE
        shared_grad_fn = jax.jit(jax.value_and_grad(shared_loss_fn))

        log.info("  Pre-built cell + JIT-compiled grad_fn (reused across starts)")

    elif method.method_type == "nelder_mead":
        import jaxley as jx

        from ADoptEX.core.simulation import create_adex_cell

        # Create cell ONCE, build JIT-compiled simulation for all NM starts
        with _suppress_jaxley_prints():
            nm_cell = create_adex_cell(
                scenario.ground_truth_params,
                use_surrogate=True,
                surrogate_slope=method.surrogate_slope,
                trainable=False,
                record=True,
            )
            I_nA = scenario.stim_current_pA / 1000.0
            current = jx.step_current(
                scenario.stim_delay_ms, scenario.stim_duration_ms,
                I_nA, scenario.dt_ms, t_max=scenario.t_max_ms,
            )
            nm_cell.stimulate(current)
            shared_sim_fn = _build_nm_sim_fn(
                nm_cell, scenario.trainable_params, scenario.dt_ms, scenario.t_max_ms,
            )

        log.info("  Pre-built cell + JIT-compiled sim_fn (reused across NM starts)")

    # ── Run starts ─────────────────────────────────────────────────────────
    runs = []

    if method.method_type == "gradient" and scenario.n_starts > 1:
        # ── vmap path: run all starts in parallel on GPU ───────────────
        runs = _run_gradient_vmapped(
            scenario, method, target_data,
            shared_cell, shared_loss_fn, shared_config,
        )
        for r in runs:
            log.info(
                "  Start %d: gamma=%.3f loss=%.6f time=%.1fs",
                r.start_index, r.gamma, r.final_loss, r.wall_time_s,
            )
    else:
        for i in range(scenario.n_starts):
            log.info("  Start %d/%d", i + 1, scenario.n_starts)
            try:
                initial_params = generate_initial_params(scenario, i)

                if method.method_type == "gradient":
                    result = run_gradient(
                        scenario, method, initial_params, target_data,
                        cell=shared_cell,
                        loss_fn=shared_loss_fn,
                        grad_fn=shared_grad_fn,
                        config=shared_config,
                        t_max=shared_t_max,
                    )
                elif method.method_type == "nelder_mead":
                    result = run_nelder_mead(
                        scenario, method, initial_params, target_data,
                        sim_fn=shared_sim_fn,
                    )
                else:
                    raise ValueError(f"Unknown method type: {method.method_type}")

                result.start_index = i
                runs.append(result)
                log.info(
                    "    gamma=%.3f loss=%.6f time=%.1fs",
                    result.gamma,
                    result.final_loss,
                    result.wall_time_s,
                )
            except Exception as e:
                log.error("  Start %d failed: %s", i, e)

    scenario_results = ScenarioResults(
        scenario_name=scenario.name,
        method_name=method.name,
        runs=runs,
    )

    if output_dir is not None:
        save_results(scenario_results, output_dir)

    return scenario_results


def run_benchmark(
    scenarios: list[SyntheticScenario],
    methods: list[MethodConfig],
    output_dir: str | Path,
    resume: bool = True,
) -> list[ScenarioResults]:
    """Run the full benchmark suite.

    Args:
        scenarios: List of synthetic scenarios to run.
        methods: List of methods to run.
        output_dir: Directory for result files.
        resume: If True, skip (scenario, method) pairs with existing results.

    Returns:
        List of all ScenarioResults.
    """
    from .results import _result_filename, load_results

    output_dir = Path(output_dir)
    all_results = []
    total = len(scenarios) * len(methods)
    completed = 0

    for scenario in scenarios:
        # Prepare target data once per scenario (simulates ground truth)
        target_data = prepare_target_data(scenario)

        for method in methods:
            completed += 1
            result_path = output_dir / _result_filename(scenario.name, method.name)

            if resume and result_path.exists():
                log.info(
                    "[%d/%d] Skipping %s x %s (exists)",
                    completed,
                    total,
                    scenario.name,
                    method.name,
                )
                all_results.append(load_results(result_path))
                continue

            log.info(
                "[%d/%d] Running %s x %s",
                completed,
                total,
                scenario.name,
                method.name,
            )

            try:
                result = run_scenario(scenario, method, target_data, output_dir)
                all_results.append(result)
            except Exception as e:
                log.error(
                    "Scenario %s x %s failed: %s", scenario.name, method.name, e
                )

    return all_results
