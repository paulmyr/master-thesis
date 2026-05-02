"""Warm-start benchmark: 4 methods × 3 scenarios × 100 inits.

Per Deistler 2025 training recipe, every run begins with a best-of-K warm-start
(K=100 random draws from PARAM_BOUNDS, evaluated under the run's loss, lowest
kept) and then refines via gradient descent (25 steps) or Nelder-Mead (50
fevals). Per-run budget: 100 + 50 = 150 forward-simulation-equivalents.

NM and gradient methods sharing the same loss receive the same K=100 candidate
pool (per (scenario, init_idx)); they pick the same warm-start point, so the
comparison isolates the optimiser.

Hyperparameters are pre-registered per method (see METHOD_CONFIGS). Runs
sequentially, flushes CSV per row, resumes on restart via run_id deduplication.
"""

import contextlib
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

SurrogateType = Literal["sigmoid", "exponential", "superspike"]

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from jax import config

config.update("jax_platform_name", "cpu")

from ADoptEX.core.parameters import NAUD_PARAMETERS, PARAM_BOUNDS
from ADoptEX.core.simulation import simulate_jaxley
from ADoptEX.evaluation.coincidence import coincidence_factor
from ADoptEX.loss import (GuarinoLossConfig, MSELossConfig, VanRossumLossConfig,
                          extract_experimental_features, make_guarino_loss_fn,
                          make_mse_loss_fn, make_van_rossum_loss_fn)
from ADoptEX.training.trainer import TrainingConfig, setup_trainable_cell, train

# --- Constants ---

DT_MS = 0.025
T_MAX_MS = 100.0
STIM_DELAY_MS = 5.0
STIM_DUR_MS = 90.0
SPIKE_PEAK_MV = 35.0

STIM_END_IDX = int(round((STIM_DELAY_MS + STIM_DUR_MS) / DT_MS))
STIM_START_IDX = int(round(STIM_DELAY_MS / DT_MS))
T = np.arange(int(round(T_MAX_MS / DT_MS))) * DT_MS
N = len(T)

TRAINABLE = ["C_m", "g_L", "E_L", "v_T", "delta_T", "v_reset"]

N_INITS = 100
K_WARMSTART = 100   # forward-only candidate sims per run
N_GRAD_STEPS = 25   # post-warmstart gradient steps (1 step ≈ 2 sim-equivalents)
NM_MAXFEV = 50      # post-warmstart Nelder-Mead fevals
BUDGET_SIMS = K_WARMSTART + max(2 * N_GRAD_STEPS, NM_MAXFEV)  # 150, for documentation
INIT_SEED_BASE = 0

SCENARIOS = {
    "tonic":            {**NAUD_PARAMETERS["tonic"]},
    "adaptation":       {**NAUD_PARAMETERS["adaptation"]},
    "initial_bursting": {**NAUD_PARAMETERS["initial_bursting"]},
}

METHODS = ["grad_guarino", "grad_vanrossum", "nm_guarino"]

# Per-method hyperparameters (edit in-place to retune; defaults from §0c sweep medians).
METHOD_CONFIGS = {
    "grad_guarino": {
        "loss_type": "guarino",
        "surrogate_type": "superspike",
        "surrogate_slope": 2.7,
        "optimizer": "polyak",
        "learning_rate": 0.002,
        "use_param_transform": True,
        "polyak_alpha": 0.75,
        "polyak_beta": 0.5,
        "temperature": 0.25,
        "beta": 10.0,
        "validity_beta": 10.0,
        "loss_kwargs": {
            "weight_spike_count" : 0.1,
        },
    },
    "grad_vanrossum": {
        "loss_type": "van_rossum",
        "surrogate_type": "superspike",
        "surrogate_slope": 3.5,
        "optimizer": "polyak",
        "learning_rate": 0.001,
        "use_param_transform": True,
        "polyak_alpha": 0.8,
        "polyak_beta": 0.5,
        "loss_kwargs": {
            "tau_ms": 6.5,
            "weight_subthreshold": 0.3,
        },
    },
    "nm_guarino": {
        "loss_type": "guarino",
        "temperature": 0.25,
        "beta": 10.0,
        "validity_beta": 10.0,
        "nm_options": {"adaptive": True, "xatol": 1e-4, "fatol": 1e-4},
    },
}

RESULTS_PATH = Path(__file__).parent / "results_benchmark_warmstart.csv"


# --- Helpers ---

@contextlib.contextmanager
def _silence():
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def _build_current_trace(I_pA):
    current: np.ndarray = np.zeros(N)
    current[STIM_START_IDX:STIM_END_IDX] = I_pA
    return current


def simulate_ground_truth(scenario_params):
    with _silence():
        sim = simulate_jaxley(
            params=scenario_params, stim_current_pA=scenario_params["I"],
            stim_duration_ms=STIM_DUR_MS, dt_ms=DT_MS,
            t_max_ms=T_MAX_MS, stim_delay_ms=STIM_DELAY_MS, use_surrogate=False,
        )
    for st in sim.spike_times:
        idx = int(round(st / DT_MS))
        if 0 <= idx < len(sim.voltage):
            sim.voltage[idx] = SPIKE_PEAK_MV
    exp_features = extract_experimental_features(
        voltage_trace=jnp.array(sim.voltage), dt_ms=DT_MS,
        stim_duration_ms=STIM_DUR_MS, stim_end_index=STIM_END_IDX,
    )
    return {
        "spike_times": sim.spike_times,
        "exp_features": exp_features,
        "exp_spike_train": jnp.array(sim.spikes),
        "voltage": jnp.array(sim.voltage),
    }


def evaluate(final_params, gt_spike_times, scenario_params):
    params = {**scenario_params, **final_params}
    with _silence():
        sim = simulate_jaxley(
            params=params, stim_current_pA=scenario_params["I"],
            stim_duration_ms=STIM_DUR_MS, dt_ms=DT_MS,
            t_max_ms=T_MAX_MS, stim_delay_ms=STIM_DELAY_MS, use_surrogate=False,
        )
    if len(sim.spike_times) == 0 or len(gt_spike_times) == 0:
        return 0.0, len(sim.spike_times)
    cf = coincidence_factor(
        spike_times_data=gt_spike_times, spike_times_model=sim.spike_times,
        duration_ms=T_MAX_MS, delta_ms=2.0,
    )
    return float(cf.gamma), len(sim.spike_times)


def sample_scaffold_inits(seed):
    """Return N_INITS scaffold param dicts (uniform-random within bounds).

    Under the warm-start protocol, the scaffold's role is only to give
    `_setup_cell` something valid for handle construction — the actual
    optimisation start point is whatever warm-start picks from its K=100
    candidate pool. Distance-from-GT stratification (used by the old
    `sample_stratified_inits`) is therefore meaningless here and dropped.
    """
    rng = np.random.default_rng(seed)
    return [
        {n: float(rng.uniform(PARAM_BOUNDS[n].min, PARAM_BOUNDS[n].max))
         for n in TRAINABLE}
        for _ in range(N_INITS)
    ]


# --- Loss factory (shared by all methods) ---

def _build_loss_fn(method_cfg, cell, data_stimuli, t_max, gt):
    lt = method_cfg["loss_type"]
    if lt == "mse":
        return make_mse_loss_fn(
            cell=cell, data_stimuli=data_stimuli, t_max=t_max, dt_ms=DT_MS,
            exp_voltage=gt["voltage"], stim_end_index=STIM_END_IDX,
            loss_config=MSELossConfig(**method_cfg.get("loss_kwargs", {})),
        )
    if lt == "guarino":
        return make_guarino_loss_fn(
            cell=cell, data_stimuli=data_stimuli, t_max=t_max, dt_ms=DT_MS,
            exp_features=gt["exp_features"], stim_duration_ms=STIM_DUR_MS,
            stim_end_index=STIM_END_IDX,
            temperature=method_cfg["temperature"], beta=method_cfg["beta"],
            validity_beta=method_cfg["validity_beta"],
            loss_config=GuarinoLossConfig(
                weight_spike_count=method_cfg.get("weight_spike_count", 0.0),
            ),
        )
    if lt == "van_rossum":
        return make_van_rossum_loss_fn(
            cell=cell, data_stimuli=data_stimuli, t_max=t_max, dt_ms=DT_MS,
            exp_spike_train=gt["exp_spike_train"], stim_end_index=STIM_END_IDX,
            exp_voltage=gt["voltage"],  # required when weight_subthreshold > 0
            loss_config=VanRossumLossConfig(**method_cfg.get("loss_kwargs", {})),
        )
    raise ValueError(f"unknown loss_type: {lt}")


def _setup_cell(scenario_params, init_params, surrogate_slope=5.0,
                surrogate_type: SurrogateType = "sigmoid"):
    cell_params = {**scenario_params, **init_params}
    cfg = TrainingConfig(
        optimizer="polyak", learning_rate=0.1, n_epochs=N_GRAD_STEPS,
        surrogate_type=surrogate_type, surrogate_slope=surrogate_slope,
        clip_to_bounds=False, use_param_transform=True,
        return_best=True, verbose=False,
    )
    with _silence():
        return setup_trainable_cell(
            initial_params=cell_params,
            current_trace_pA=jnp.asarray(_build_current_trace(scenario_params["I"])),
            dt_ms=DT_MS, config=cfg, trainable_params=TRAINABLE,
        )


# --- Loss-vector wrapper + warm-start ---

def _scipy_loss_wrapper(loss_jit, jaxley_keys):
    """Wrap a jitted Jaxley-format loss into x ∈ R^n → scalar."""
    def fn(x):
        params = [{k: jnp.array([float(x[i])])} for i, k in enumerate(jaxley_keys)]
        try:
            v = float(loss_jit(params))
            return v if math.isfinite(v) else 1e6
        except Exception:
            return 1e6
    return fn


def warm_start(eval_loss, K, rng):
    """Best-of-K random draws within PARAM_BOUNDS (Deistler 2025).

    Returns (best_x, best_loss). Each method's warm-start uses its own loss;
    fairness across methods is preserved by sharing the rng seed (so the
    candidate pool is identical, and methods using the same loss converge to
    the same starting point).
    """
    best_x = None
    best_loss = math.inf
    for _ in range(K):
        x = np.array([rng.uniform(PARAM_BOUNDS[n].min, PARAM_BOUNDS[n].max)
                      for n in TRAINABLE])
        v = eval_loss(x)
        if math.isfinite(v) and v < best_loss:
            best_loss = v
            best_x = x
    if best_x is None:
        raise RuntimeError(f"warm-start: all K={K} candidates returned non-finite loss")
    return best_x, float(best_loss)


# --- Method runners ---

def _eval_loss_for(scenario_params, scaffold_params, method_cfg, gt):
    """Build a (cell, eval_loss, jaxley_keys) tuple shared by warm-start + NM."""
    surrogate_slope = method_cfg.get("surrogate_slope", 5.0)
    surrogate_type = method_cfg.get("surrogate_type", "sigmoid")
    cell, data_stimuli, t_max, _ = _setup_cell(
        scenario_params, scaffold_params, surrogate_slope, surrogate_type
    )
    loss_fn = _build_loss_fn(method_cfg, cell, data_stimuli, t_max, gt)
    loss_jit = jax.jit(loss_fn)
    jaxley_keys = [list(p.keys())[0] for p in cell.get_parameters()]
    eval_loss = _scipy_loss_wrapper(loss_jit, jaxley_keys)
    return cell, eval_loss, jaxley_keys


def _keep_better(ws_params, ws_loss, train_params, train_best_loss):
    """Return (final_params, final_loss, kept_source).

    Practitioner's workflow: keep whichever of (warm-start, trained) has lower
    loss. This is what `return_best=True` *should* do but currently doesn't —
    the trainer's `best_loss` initialises at +inf and never compares against
    the pre-training (warm-start) loss, so a wholly regressing run returns
    its least-bad post-step params, not the warm-start it should have kept.
    """
    if (not math.isfinite(train_best_loss)) or train_best_loss >= ws_loss:
        return ws_params, ws_loss, "warmstart"
    return train_params, train_best_loss, "training"


def run_gradient(scenario_params, scaffold_params, method_cfg, gt, rng,
                 k_warmstart=None, n_grad_steps=None):
    """Warm-start (K candidates) → rebuild → train.

    `k_warmstart` and `n_grad_steps` default to module constants but can be
    overridden by the HP sweep (which uses smaller K for trial cost).
    All other HPs come from `method_cfg`; safe defaults for missing keys.
    """
    K = K_WARMSTART if k_warmstart is None else k_warmstart
    n_epochs = N_GRAD_STEPS if n_grad_steps is None else n_grad_steps

    # Phase 1: scaffold cell → eval_loss → warm-start.
    _, eval_loss, _ = _eval_loss_for(scenario_params, scaffold_params, method_cfg, gt)
    ws_x, ws_loss = warm_start(eval_loss, K, rng)
    ws_params = {n: float(ws_x[i]) for i, n in enumerate(TRAINABLE)}

    # Phase 2: rebuild cell with ws_params so handles point at the warm-start.
    surrogate_slope = method_cfg["surrogate_slope"]
    surrogate_type: SurrogateType = method_cfg.get("surrogate_type", "sigmoid")
    cell, data_stimuli, t_max, handles = _setup_cell(
        scenario_params, ws_params, surrogate_slope, surrogate_type
    )
    loss_fn = _build_loss_fn(method_cfg, cell, data_stimuli, t_max, gt)

    optimizer = method_cfg.get("optimizer", "polyak")
    cfg_kwargs = dict(
        optimizer=optimizer,
        learning_rate=method_cfg["learning_rate"],
        n_epochs=n_epochs,
        surrogate_type=surrogate_type,
        surrogate_slope=surrogate_slope,
        clip_to_bounds=False,
        use_param_transform=method_cfg.get("use_param_transform", True),
        grad_clip_norm=method_cfg.get("grad_clip_norm", None),
        return_best=True,
        verbose=False,
    )
    if optimizer == "polyak":
        cfg_kwargs["polyak_alpha"] = method_cfg.get("polyak_alpha", 0.85)
        cfg_kwargs["polyak_beta"] = method_cfg.get("polyak_beta", 0.8)
    cfg = TrainingConfig(**cfg_kwargs)
    with _silence():
        result = train(loss_fn, handles, cfg)
    train_best_loss = float(result.best_loss)
    train_final_loss = float(result.final_loss)
    train_params = result.get_params_dict()

    final_params, final_loss, kept_source = _keep_better(
        ws_params, ws_loss, train_params, train_best_loss
    )
    return {
        "warmstart_params": ws_params,
        "warmstart_loss": ws_loss,
        "train_best_loss": train_best_loss,
        "train_final_loss": train_final_loss,
        "kept_source": kept_source,
        "final_params": final_params,
        "initial_loss": ws_loss,
        "final_loss": final_loss,
        "n_iter": len(result.loss_history),
        "converged": math.isfinite(train_best_loss),
    }


def run_nelder_mead(scenario_params, scaffold_params, method_cfg, gt, rng):
    _, eval_loss, _ = _eval_loss_for(scenario_params, scaffold_params, method_cfg, gt)
    ws_x, ws_loss = warm_start(eval_loss, K_WARMSTART, rng)
    ws_params = {n: float(ws_x[i]) for i, n in enumerate(TRAINABLE)}

    bounds = [(PARAM_BOUNDS[n].min, PARAM_BOUNDS[n].max) for n in TRAINABLE]
    res = scipy.optimize.minimize(
        eval_loss, ws_x, method="Nelder-Mead", bounds=bounds,
        options={"maxfev": NM_MAXFEV, **method_cfg["nm_options"]},
    )
    nm_best_loss = float(res.fun)
    nm_params = {n: float(res.x[i]) for i, n in enumerate(TRAINABLE)}

    final_params, final_loss, kept_source = _keep_better(
        ws_params, ws_loss, nm_params, nm_best_loss
    )
    return {
        "warmstart_params": ws_params,
        "warmstart_loss": ws_loss,
        "train_best_loss": nm_best_loss,
        "train_final_loss": nm_best_loss,  # NM has no separate final/best.
        "kept_source": kept_source,
        "final_params": final_params,
        "initial_loss": ws_loss,
        "final_loss": final_loss,
        "n_iter": int(res.nfev),
        "converged": bool(res.success),
    }


def dispatch(method_name, scenario_params, scaffold_params, method_cfg, gt,
             scenario_name, init_idx):
    # Same rng seed across methods (within a (scenario, init_idx)) → identical
    # K=100 candidate pool. Methods sharing a loss therefore land at the same
    # warm-start point; methods with different losses pick differently.
    seed = INIT_SEED_BASE + (abs(hash(scenario_name)) % 100_000) * N_INITS + init_idx
    rng = np.random.default_rng(seed)
    if method_name.startswith("nm_"):
        return run_nelder_mead(scenario_params, scaffold_params, method_cfg, gt, rng)
    return run_gradient(scenario_params, scaffold_params, method_cfg, gt, rng)


# --- CSV ---

def csv_fieldnames():
    base = ["run_id", "scenario", "init_idx", "method",
            "final_gamma", "n_spikes_gt", "n_spikes_sim",
            "warmstart_loss", "train_best_loss", "train_final_loss",
            "final_loss", "kept_source",
            "n_iter", "wall_time_s", "converged",
            "warmstart_params_json", "final_params_json", "gt_params_json",
            "timestamp_iso"]
    return base + [f"norm_err_{n}" for n in TRAINABLE]


def make_row(scenario_name, init_idx, method, r, gamma, n_spikes_sim,
             n_spikes_gt, scenario_params, wall):
    row = {
        "run_id": f"{scenario_name}_{init_idx:03d}_{method}",
        "scenario": scenario_name,
        "init_idx": init_idx,
        "method": method,
        "final_gamma": gamma,
        "n_spikes_gt": n_spikes_gt,
        "n_spikes_sim": n_spikes_sim,
        "warmstart_loss": r.get("warmstart_loss", float("nan")),
        "train_best_loss": r.get("train_best_loss", float("nan")),
        "train_final_loss": r.get("train_final_loss", float("nan")),
        "final_loss": r["final_loss"],
        "kept_source": r.get("kept_source", ""),
        "n_iter": r["n_iter"],
        "wall_time_s": wall,
        "converged": r["converged"],
        "warmstart_params_json": json.dumps(r.get("warmstart_params", {})),
        "final_params_json": json.dumps(r["final_params"]),
        "gt_params_json": json.dumps({k: scenario_params[k] for k in TRAINABLE}),
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
    }
    for n in TRAINABLE:
        gt_v = scenario_params[n]
        fv = r["final_params"].get(n, float("nan"))
        row[f"norm_err_{n}"] = abs(fv - gt_v) / (PARAM_BOUNDS[n].max - PARAM_BOUNDS[n].min)
    return row


# --- Main ---

def benchmark():
    completed = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            completed = {row["run_id"] for row in csv.DictReader(f)}
    write_header = not RESULTS_PATH.exists()

    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
        if write_header:
            writer.writeheader()
        for scenario_name, scenario_params in SCENARIOS.items():
            print(f"scenario {scenario_name}: {scenario_params}")
            gt = simulate_ground_truth(scenario_params)
            n_spikes_gt = len(gt["spike_times"])
            scaffold_seed = INIT_SEED_BASE + abs(hash(scenario_name)) % 10000
            scaffolds = sample_scaffold_inits(scaffold_seed)
            print(f"\n=== {scenario_name} ({n_spikes_gt} GT spikes, {len(scaffolds)} inits) ===")
            for init_idx, scaffold in enumerate(scaffolds):
                for method_name in METHODS:
                    run_id = f"{scenario_name}_{init_idx:03d}_{method_name}"
                    if run_id in completed:
                        continue
                    method_cfg = METHOD_CONFIGS[method_name]
                    t0 = time.time()
                    try:
                        r = dispatch(method_name, scenario_params, scaffold,
                                     method_cfg, gt, scenario_name, init_idx)
                        gamma, n_sim = evaluate(r["final_params"], gt["spike_times"],
                                                scenario_params)
                        wall = time.time() - t0
                        writer.writerow(make_row(scenario_name, init_idx,
                                                 method_name, r, gamma, n_sim,
                                                 n_spikes_gt, scenario_params, wall))
                        f.flush()
                        ws_loss = r.get("warmstart_loss", float("nan"))
                        print(f"{run_id} ws_loss={ws_loss:.3g} "
                              f"final_loss={r['final_loss']:.3g} "
                              f"Γ={gamma:.3f} t={wall:.1f}s")
                    except Exception as e:
                        print(f"{run_id} ERROR: {e}")


if __name__ == "__main__":
    print("starting benchmark...")
    benchmark()
