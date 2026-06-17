"""Recovery-radius benchmark (Exp A: basin of attraction).

Per (scenario, method, δ, direction): start at θ_GT + δ·direction (clipped
to bounds), run optimization (no warm-start), record Γ + per-parameter
recovery error. Direction RNG is seeded per scenario so all methods see the
*same* direction set at each (scenario, δ) — the comparison is paired.

Headline plot (analysis notebook): P(Γ > 0.5) vs δ, faceted by scenario,
one curve per method. Crossing point of the curve is the recovery radius.

Run with `python benchmark_recovery.py`. Total ≈ 1620 runs (~1 h on CPU).
CSV is flushed per row. Smoke: edit DELTAS / N_DIRS / METHODS at the top.
"""

import contextlib
import csv
import math
import os
import time
from pathlib import Path
from typing import Any, Literal

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

SurrogateType = Literal["sigmoid", "exponential", "superspike"]


# --- Simulation constants --------------------------------------------------

DT_MS = 0.025
T_MAX_MS = 100.0
STIM_DELAY_MS = 5.0
STIM_DUR_MS = 90.0
SPIKE_PEAK_MV = 35.0

STIM_END_IDX = int(round((STIM_DELAY_MS + STIM_DUR_MS) / DT_MS))
STIM_START_IDX = int(round(STIM_DELAY_MS / DT_MS))
N_TIMESTEPS = int(round(T_MAX_MS / DT_MS))

TRAINABLE = ["C_m", "g_L", "E_L", "v_T", "delta_T", "v_reset"]
N_GRAD_STEPS = 500
NM_MAXFEV = 1000


# --- Experiment constants --------------------------------------------------

DELTAS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
N_DIRS = 30
SEED = 42
METHODS = ["grad_guarino", "grad_vanrossum", "nm_guarino"]
OUT_CSV = Path(__file__).parent / "results_recovery.csv"

SCENARIOS = {
    "tonic":            dict(NAUD_PARAMETERS["tonic"]),
    "adaptation":       dict(NAUD_PARAMETERS["adaptation"]),
    "initial_bursting": dict(NAUD_PARAMETERS["initial_bursting"]),
}

# HPs validated in `notebooks/training_synthetic_singlestage_guarino.ipynb`
# for close-to-GT inits (small loss, large gradient regime). Tweak when the
# perturbation HP sweep produces something better.
METHOD_CONFIGS: dict[str, dict] = {
    "grad_guarino": {
        "loss_type": "guarino",
        "surrogate_type": "superspike",
        "surrogate_slope": 10,
        "optimizer": "adam",
        "learning_rate": 0.05,
        "use_param_transform": False,
        #"polyak_alpha": 0.8,
        #"polyak_beta": 0.6,
        "temperature": 0.5,
        "beta": 10.0,
        "validity_beta": 10.0,
        "weight_spike_count": 0.1,
    },
    "grad_vanrossum": {
        "loss_type": "van_rossum",
        "surrogate_type": "superspike",
        "surrogate_slope": 10,
        "optimizer": "adam",
        "learning_rate": 0.05,
        "use_param_transform": False,
        #"polyak_alpha": 0.8,
        #"polyak_beta": 0.6,
        "loss_kwargs": {"tau_ms": 8.0, "weight_subthreshold": 0.3},
    },
    "nm_guarino": {
        "loss_type": "guarino",
        "temperature": 0.25,
        "beta": 10.0,
        "validity_beta": 10.0,
        "nm_options": {"adaptive": True, "xatol": 1e-4, "fatol": 1e-4},
    },
}

CSV_FIELDS = [
    "scenario", "method", "delta", "direction_idx",
    "gamma", "n_spikes_sim", "n_spikes_gt",
    "init_loss", "final_loss", "kept_source", "n_iter",
    "n_clipped", "elapsed_s",
] + [f"param_err_{n}" for n in TRAINABLE]


# --- Forward-sim & loss helpers --------------------------------------------

@contextlib.contextmanager
def _silence():
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def _current_trace(I_pA: float) -> np.ndarray:
    current = np.zeros(N_TIMESTEPS)
    current[STIM_START_IDX:STIM_END_IDX] = I_pA
    return current


def simulate_gt(scenario_params: dict) -> dict:
    """Run the non-differentiable AdEx forward sim and pre-compute everything
    each loss flavour needs as `gt`."""
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


def evaluate_gamma(final_params: dict, gt_spike_times, scenario_params: dict
                   ) -> tuple[float, int]:
    """Final Γ via a clean (non-surrogate) forward sim at the final params."""
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


def setup_cell(scenario_params: dict, init_params: dict,
               surrogate_slope: float, surrogate_type: SurrogateType):
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
            current_trace_pA=jnp.asarray(_current_trace(scenario_params["I"])),
            dt_ms=DT_MS, config=cfg, trainable_params=TRAINABLE,
        )


def build_loss_fn(method_cfg: dict, cell, data_stimuli, t_max: float, gt: dict):
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
            exp_voltage=gt["voltage"],  # used iff weight_subthreshold > 0
            loss_config=VanRossumLossConfig(**method_cfg.get("loss_kwargs", {})),
        )
    raise ValueError(f"unknown loss_type: {lt}")


# --- Perturbation sampling -------------------------------------------------

def sample_directions(rng: np.random.Generator, n_dirs: int, n_params: int
                      ) -> np.ndarray:
    """Uniform-on-sphere directions in R^n_params (Gaussian + L2-normalize)."""
    raw = rng.standard_normal((n_dirs, n_params))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw


def perturb(theta_gt: dict, direction: np.ndarray, delta: float
            ) -> tuple[dict, int]:
    """θ[i] = clip(θ_gt[i] + δ · direction[i] · (max - min), bounds).
    Returns (init_dict, n_clipped). `direction` is indexed by TRAINABLE order.
    """
    out = dict(theta_gt)
    n_clipped = 0
    for i, name in enumerate(TRAINABLE):
        b = PARAM_BOUNDS[name]
        proposed = theta_gt[name] + delta * direction[i] * (b.max - b.min)
        clipped = float(np.clip(proposed, b.min, b.max))
        if clipped != proposed:
            n_clipped += 1
        out[name] = clipped
    return out, n_clipped


def param_errors(final_params: dict, theta_gt: dict) -> dict:
    """Per-parameter |final - GT| / range, keyed `param_err_<name>`."""
    return {
        f"param_err_{n}": abs(final_params[n] - theta_gt[n])
                          / (PARAM_BOUNDS[n].max - PARAM_BOUNDS[n].min)
        for n in TRAINABLE
    }


# --- Optimisation drivers (no warm-start) ----------------------------------

def _keep_better(init_params, init_loss, trained_params, trained_best_loss):
    """If training regressed (or NaN'd), return the perturbed init."""
    if (not math.isfinite(trained_best_loss)) or trained_best_loss >= init_loss:
        return init_params, init_loss, "init"
    return trained_params, trained_best_loss, "training"


def run_grad(scenario_params: dict, init_params: dict, method_cfg: dict,
             gt: dict) -> dict:
    surrogate_slope = method_cfg["surrogate_slope"]
    surrogate_type: SurrogateType = method_cfg.get("surrogate_type", "sigmoid")
    cell, data_stimuli, t_max, handles = setup_cell(
        scenario_params, init_params, surrogate_slope, surrogate_type
    )
    loss_fn = build_loss_fn(method_cfg, cell, data_stimuli, t_max, gt)
    init_loss = float(jax.jit(loss_fn)(handles))

    cfg_kwargs: dict[str, Any] = dict(
        optimizer=method_cfg.get("optimizer", "polyak"),
        learning_rate=method_cfg["learning_rate"],
        n_epochs=N_GRAD_STEPS,
        surrogate_type=surrogate_type,
        surrogate_slope=surrogate_slope,
        clip_to_bounds=False,
        use_param_transform=method_cfg.get("use_param_transform", True),
        grad_clip_norm=method_cfg.get("grad_clip_norm", None),
        return_best=True,
        verbose=False,
    )
    if cfg_kwargs["optimizer"] == "polyak":
        cfg_kwargs["polyak_alpha"] = method_cfg.get("polyak_alpha", 1.0)
        cfg_kwargs["polyak_beta"] = method_cfg.get("polyak_beta", 0.8)
    with _silence():
        result = train(loss_fn, handles, TrainingConfig(**cfg_kwargs))

    trained_best = float(result.best_loss)
    trained_params = result.get_params_dict()
    final_params, final_loss, kept_source = _keep_better(
        init_params, init_loss, trained_params, trained_best
    )
    return {
        "init_loss": init_loss, "final_loss": final_loss,
        "kept_source": kept_source, "final_params": final_params,
        "n_iter": len(result.loss_history),
    }


def run_nm(scenario_params: dict, init_params: dict, method_cfg: dict,
           gt: dict) -> dict:
    # NM uses a hard-Heaviside surrogate (slope is irrelevant; type defaults
    # are fine) — we only need the cell + jitted loss for scipy to call.
    cell, data_stimuli, t_max, _ = setup_cell(
        scenario_params, init_params,
        surrogate_slope=method_cfg.get("surrogate_slope", 5.0),
        surrogate_type=method_cfg.get("surrogate_type", "sigmoid"),
    )
    loss_fn = build_loss_fn(method_cfg, cell, data_stimuli, t_max, gt)
    loss_jit = jax.jit(loss_fn)
    jaxley_keys = [list(p.keys())[0] for p in cell.get_parameters()]

    def eval_loss(x: np.ndarray) -> float:
        params = [{k: jnp.array([float(x[i])])} for i, k in enumerate(jaxley_keys)]
        try:
            v = float(loss_jit(params))
            return v if math.isfinite(v) else 1e6
        except Exception:
            return 1e6

    init_x = np.array([init_params[n] for n in TRAINABLE])
    init_loss = float(eval_loss(init_x))

    bounds = [(PARAM_BOUNDS[n].min, PARAM_BOUNDS[n].max) for n in TRAINABLE]
    nm_options = dict(method_cfg.get("nm_options", {}))
    nm_options["maxfev"] = NM_MAXFEV
    res = scipy.optimize.minimize(
        eval_loss, x0=init_x, method="Nelder-Mead",
        bounds=bounds, options=nm_options,
    )
    trained_best = float(res.fun) if math.isfinite(res.fun) else float("inf")
    trained_params = {n: float(res.x[i]) for i, n in enumerate(TRAINABLE)}
    final_params, final_loss, kept_source = _keep_better(
        init_params, init_loss, trained_params, trained_best
    )
    return {
        "init_loss": init_loss, "final_loss": final_loss,
        "kept_source": kept_source, "final_params": final_params,
        "n_iter": int(res.nfev),
    }


# --- Main loop -------------------------------------------------------------

def run():
    rng_master = np.random.default_rng(SEED)
    n_params = len(TRAINABLE)

    # GTs and direction sets are scenario-specific but method-shared (paired).
    cache: dict[str, dict] = {}
    for scenario_name, scenario_params in SCENARIOS.items():
        print(f"GT simulating ({scenario_name})...")
        gt = simulate_gt(scenario_params)
        dir_seed = int(rng_master.integers(0, 2**31))
        directions = sample_directions(np.random.default_rng(dir_seed),
                                       N_DIRS, n_params)
        cache[scenario_name] = {
            "scenario_params": scenario_params,
            "gt": gt,
            "theta_gt": {n: float(scenario_params[n]) for n in TRAINABLE},
            "directions": directions,
            "n_spikes_gt": len(gt["spike_times"]),
        }
        print(f"  done. n_spikes_gt={cache[scenario_name]['n_spikes_gt']}")

    write_header = not OUT_CSV.exists()
    total = len(SCENARIOS) * len(METHODS) * len(DELTAS) * N_DIRS
    run_idx = 0
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for scenario_name, c in cache.items():
            for delta in DELTAS:
                for dir_idx in range(N_DIRS):
                    init_params, n_clipped = perturb(
                        c["theta_gt"], c["directions"][dir_idx], delta
                    )
                    for method in METHODS:
                        run_idx += 1
                        t0 = time.time()
                        cfg = METHOD_CONFIGS[method]
                        row: dict[str, Any] = {f: float("nan") for f in CSV_FIELDS}
                        row.update({
                            "scenario": scenario_name, "method": method,
                            "delta": delta, "direction_idx": dir_idx,
                            "n_clipped": n_clipped,
                            "n_spikes_gt": c["n_spikes_gt"],
                        })
                        try:
                            runner = run_grad if method.startswith("grad_") else run_nm
                            result = runner(c["scenario_params"], init_params,
                                            cfg, c["gt"])
                            gamma, n_sim = evaluate_gamma(
                                result["final_params"], c["gt"]["spike_times"],
                                c["scenario_params"],
                            )
                            row.update({
                                "gamma": gamma, "n_spikes_sim": n_sim,
                                "init_loss": result["init_loss"],
                                "final_loss": result["final_loss"],
                                "kept_source": result["kept_source"],
                                "n_iter": result["n_iter"],
                            })
                            row.update(param_errors(result["final_params"],
                                                    c["theta_gt"]))
                        except Exception as e:
                            row["kept_source"] = "ERROR"
                            print(f"  ERROR ({scenario_name}/{method}/"
                                  f"δ={delta}/dir={dir_idx}): {e}")
                        row["elapsed_s"] = time.time() - t0
                        writer.writerow(row)
                        f.flush()
                        if run_idx % 25 == 0 or run_idx == total:
                            print(f"  [{run_idx:4d}/{total}] {scenario_name} "
                                  f"{method} δ={delta} dir={dir_idx} "
                                  f"γ={row['gamma']:.3f} "
                                  f"({row['elapsed_s']:.1f}s)")

    print(f"\nDone. Results: {OUT_CSV}")


if __name__ == "__main__":
    print(f"Recovery benchmark: {len(SCENARIOS)} scenarios × {len(METHODS)} "
          f"methods × {len(DELTAS)} deltas × {N_DIRS} dirs = "
          f"{len(SCENARIOS) * len(METHODS) * len(DELTAS) * N_DIRS} runs")
    run()
