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
) -> RunResult:
    """Run gradient-based optimization for a single start.

    Uses step current stimulation via setup_trainable_cell_step.
    """
    from ADoptEX.training.trainer import setup_trainable_cell_step, train

    config = method.training_config
    assert config is not None, "training_config required for gradient methods"

    # Override trainable params from scenario
    config = replace(config, trainable_params=scenario.trainable_params)

    t0 = time.time()

    # Setup cell with step current (suppress Jaxley print noise)

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

    # Build loss function
    loss_fn = _build_loss_fn(method, cell, t_max, target_data)

    # Run training
    train_result = train(
        loss_fn=loss_fn,
        trainable_params=trainable_params,
        config=config,
        initial_params=initial_params,
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


def run_nelder_mead(
    scenario: SyntheticScenario,
    method: MethodConfig,
    initial_params: dict[str, float],
    target_data: dict,
) -> RunResult:
    """Run Nelder-Mead optimization for a single start.

    Uses simulate_jaxley with step current for each function evaluation.
    """
    trainable_names = scenario.trainable_params
    dt_ms = scenario.dt_ms

    # Build objective: param vector -> scalar loss
    loss_config = method.nm_loss_config
    loss_history = []

    def objective(x: np.ndarray) -> float:
        """Evaluate loss for a parameter vector."""
        params = dict(initial_params)
        for i, name in enumerate(trainable_names):
            params[name] = float(x[i])

        # Clip to bounds
        for name in trainable_names:
            bounds = PARAM_BOUNDS[name]
            params[name] = bounds.clip(params[name])

        try:
            with _suppress_jaxley_prints():
                result = simulate_jaxley(
                    params=params,
                    stim_current_pA=scenario.stim_current_pA,
                    stim_duration_ms=scenario.stim_duration_ms,
                    dt_ms=dt_ms,
                    t_max_ms=scenario.t_max_ms,
                    stim_delay_ms=scenario.stim_delay_ms,
                    use_surrogate=True,
                    surrogate_slope=method.surrogate_slope,
                )

            loss = _compute_nm_loss(
                method.nm_objective, result, target_data, loss_config, dt_ms
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
    """Compute the loss for Nelder-Mead evaluation."""
    if objective_type == "van_rossum":
        from ADoptEX.loss.van_rossum import VanRossumLossConfig, spike_train_from_voltage, van_rossum_distance

        config = loss_config or VanRossumLossConfig()
        sim_spikes = jnp.array(sim_result.spikes)
        target_voltage = jnp.array(target_data["voltage"])
        exp_spikes = spike_train_from_voltage(target_voltage, threshold_mv=0.0, dt_ms=dt_ms)
        return van_rossum_distance(sim_spikes, exp_spikes, dt_ms, config)

    elif objective_type == "mse":
        from ADoptEX.loss.mse import MSELossConfig, mse_loss

        config = loss_config or MSELossConfig()
        sim_v = jnp.array(sim_result.voltage)
        target_v = jnp.array(target_data["voltage"])
        min_len = min(len(sim_v), len(target_v))
        return mse_loss(sim_v[:min_len], target_v[:min_len], config)

    elif objective_type == "guarino":
        raise NotImplementedError(
            "Guarino loss for Nelder-Mead not supported (requires cell object)"
        )
    else:
        raise ValueError(f"Unknown NM objective: {objective_type}")


# ── Orchestration ────────────────────────────────────────────────────────────


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
    """Run all starts for a (scenario, method) pair."""
    log.info(
        "Running %s x %s (%d starts)",
        scenario.name,
        method.name,
        scenario.n_starts,
    )

    runs = []
    for i in range(scenario.n_starts):
        log.info("  Start %d/%d", i + 1, scenario.n_starts)
        try:
            result = run_single(scenario, method, i, target_data)
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
