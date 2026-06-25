"""Subthreshold recovery benchmark: gradient descent vs. Nelder-Mead (MSE).

A focused spinoff of `benchmark_recovery.py`. In the *non-spiking* regime only
C_m, E_L, g_L shape the voltage trace, so this isolates a clean 3-parameter
recovery and asks: how does MSE-based gradient descent compare to Nelder-Mead
(also MSE) on (1) convergence speed and (2) fit quality?

Design (see plan):
  * 3 bases (tonic, adaptation, initial_bursting) x 4 deltas x 25 directions =
    300 scenarios. Each scenario is a *fully* perturbed ground truth (ALL AdEx
    params perturbed, not just the three subthreshold ones) so the target
    realistically reflects unknown parameters. A non-spiking guard rejects any
    perturbation that crosses rheobase.
  * Every fit starts from the SAME fixed init: C_m/E_L/g_L = tonic values, spike
    params = the notebook's non-spiking defaults. Only C_m, E_L, g_L are trained.
  * Two optimizers per scenario (paired): Adam (fixed HPs) and Nelder-Mead, both
    on the MSE voltage loss.

Outputs (both CSVs are overwritten at the start of each run, then flushed per
row so progress is visible / background-able):
  * results_subthreshold.csv         one row per run (summary metrics)
  * results_subthreshold_curves.csv  one row per loss evaluation (loss curves)

Run with `python benchmark_subthreshold.py`. Smoke config: DELTAS=[0.2],
N_DIRS=2, BASES=["tonic"].
"""

import contextlib
import csv
import logging
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from jax import config

config.update("jax_platform_name", "cpu")
logging.getLogger().setLevel(logging.ERROR)

from ADoptEX.core.data import TraceData
from ADoptEX.core.parameters import NAUD_PARAMETERS, PARAM_BOUNDS
from ADoptEX.core.simulation import simulate_with_current_trace
from ADoptEX.loss import make_mse_loss_fn
from ADoptEX.training.trainer import (TrainingConfig, setup_trainable_cell,
                                      train)

# --- Simulation constants --------------------------------------------------

DT_MS = 0.025
SUB_CURRENT_PA = 100.0                       # subthreshold step (below rheobase)
SUB_DELAY, SUB_DUR = 30.0, 100.0      # pre-stim baseline, then subthreshold step
SUB_T_MAX = SUB_DELAY + SUB_DUR       # full trace = baseline + stim; trained as-is (no crop)

# --- Experiment constants --------------------------------------------------

DELTAS = [0.1, 0.2, 0.3, 0.4]
N_DIRS = 50
SEED = 42
BASES = ["tonic"]
METHODS = ["grad", "nm"]

OUT_CSV = Path(__file__).parent / "results_subthreshold.csv"
CURVES_CSV = Path(__file__).parent / "results_subthreshold_curves.csv"

# All AdEx params are perturbed in the GT; only the three subthreshold ones are fit.
ALL_PARAMS = ["C_m", "g_L", "E_L", "v_T", "delta_T", "v_reset", "tau_w", "a", "b"]
PERTURB_PARAMS = ALL_PARAMS
TRAINABLE = ["C_m", "E_L", "g_L"]

# Fixed init (always identical): tonic subthreshold params + non-spiking defaults.
NON_SPIKING_DEFAULTS = {"v_T": 0.0, "delta_T": 1.0, "v_reset": -100.0,
                        "tau_w": 20.0, "a": 0.0, "b": 0.0}
INIT = {**dict(NAUD_PARAMETERS["tonic"]), **NON_SPIKING_DEFAULTS}

CONV_TOL = 0.1          # mV^2; convergence = loss <= tol (perfect recovery -> ~0)
MAX_RESAMPLE = 200       # non-spiking-guard attempts per scenario slot

# Gradient hyperparameters (fixed). Surrogate is irrelevant subthreshold.
# Both optimizers are capped at MAX_ITERS loss evaluations for a like-for-like
# budget comparison (Adam epochs == NM function evaluations).
MAX_ITERS = 25
# Tuned via subthreshold_grad_tuning.py: adam + sigmoid transform (normalizes the
# ~30x parameter-range spread so one lr moves all three) + lr=0.2 reaches CONV_TOL
# in ~7 evals and beats NM at every budget <= 25. Polyak/SGD and lr=0.05 were the
# misconfiguration that made NM look better.
GRAD = dict(optimizer="adam", lr=0.2, epochs=MAX_ITERS,
            surrogate_type="superspike", surrogate_slope=10.0, transform=True)
NM_MAXFEV = MAX_ITERS

CSV_FIELDS = [
    "scenario_idx", "base", "method", "delta", "direction_idx", "n_clipped",
    "init_loss", "final_loss", "converged", "iters_to_tol", "n_iter", "elapsed_s",
    "param_err_C_m", "param_err_E_L", "param_err_g_L",
    "C_m_gt", "E_L_gt", "g_L_gt", "C_m_fit", "E_L_fit", "g_L_fit",
]
CURVE_FIELDS = ["scenario_idx", "base", "method", "delta", "direction_idx",
                "iter", "loss"]


@contextlib.contextmanager
def _silence():
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


# --- Trace generation ------------------------------------------------------

def _step_current(n_samples, delay_ms, dur_ms, amp_pA):
    cur = np.zeros(n_samples)
    s = int(round(delay_ms / DT_MS))
    e = int(round((delay_ms + dur_ms) / DT_MS))
    cur[s:e] = amp_pA
    return cur, s, e


def make_synthetic_trace(amp_pA, t_max_ms, delay_ms, dur_ms, gt_params) -> TraceData:
    """GT forward sim (non-surrogate) packaged as TraceData."""
    n = int(round(t_max_ms / DT_MS))
    current, s_idx, e_idx = _step_current(n, delay_ms, dur_ms, amp_pA)
    with _silence():
        sim = simulate_with_current_trace(
            params=gt_params, current_trace_pA=current, dt_ms=DT_MS,
            use_surrogate=False, v_init=gt_params["E_L"],
        )
    return TraceData(
        time=np.arange(n) * DT_MS, voltage=sim.voltage[:n].copy(), current=current,
        dt_ms=DT_MS, spike_times=sim.spike_times, stim_start_idx=s_idx,
        stim_end_idx=e_idx, stim_current_pA=float(amp_pA),
    )


# --- Perturbation ----------------------------------------------------------

def sample_direction(rng, n_params):
    raw = rng.standard_normal(n_params)
    return raw / np.linalg.norm(raw)


def make_gt(base_params, direction, delta):
    """Perturb ALL params of the base; clip to bounds. Returns (gt, n_clipped)."""
    out = dict(base_params)
    n_clipped = 0
    for i, name in enumerate(PERTURB_PARAMS):
        b = PARAM_BOUNDS[name]
        proposed = base_params[name] + delta * direction[i] * (b.max - b.min)
        clipped = float(np.clip(proposed, b.min, b.max))
        n_clipped += clipped != proposed
        out[name] = clipped
    return out, n_clipped


def generate_scenarios():
    """300 fully-perturbed, guaranteed-non-spiking subthreshold scenarios."""
    rng_master = np.random.default_rng(SEED)
    scenarios = []
    sidx = 0
    for base in BASES:
        base_params = dict(NAUD_PARAMETERS[base])
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31)))
        for delta in DELTAS:
            count = 0
            attempts = 0
            while count < N_DIRS:
                attempts += 1
                if attempts > MAX_RESAMPLE * N_DIRS:
                    raise RuntimeError(f"non-spiking guard exhausted: {base} δ={delta}")
                direction = sample_direction(rng, len(PERTURB_PARAMS))
                gt, n_clipped = make_gt(base_params, direction, delta)
                trace = make_synthetic_trace(SUB_CURRENT_PA, SUB_T_MAX, SUB_DELAY,
                                             SUB_DUR, gt)
                if len(trace.spike_times) > 0:
                    continue  # crossed rheobase -> reject, resample
                scenarios.append(dict(
                    scenario_idx=sidx, base=base, delta=delta, direction_idx=count,
                    gt=gt, trace=trace, n_clipped=n_clipped))
                sidx += 1
                count += 1
    return scenarios


# --- Cell / loss build (shared by both optimizers) -------------------------

def build(trace):
    cfg = TrainingConfig(surrogate_type=GRAD["surrogate_type"],
                         surrogate_slope=GRAD["surrogate_slope"],
                         clip_to_bounds=False, use_param_transform=False, verbose=False)
    with _silence():
        cell, data_stimuli, t_max, handles = setup_trainable_cell(
            initial_params=INIT, current_trace_pA=jnp.asarray(trace.current),
            dt_ms=DT_MS, config=cfg, trainable_params=TRAINABLE,
            v_init=float(trace.voltage[0]))
    loss_fn = make_mse_loss_fn(cell=cell, data_stimuli=data_stimuli, t_max=t_max,
                               dt_ms=DT_MS, exp_voltage=jnp.asarray(trace.voltage),
                               stim_end_index=trace.stim_end_idx)
    return cell, handles, loss_fn


def run_grad(trace):
    _, handles, loss_fn = build(trace)
    cfg = TrainingConfig(optimizer=GRAD["optimizer"], learning_rate=GRAD["lr"],
                         n_epochs=GRAD["epochs"], surrogate_type=GRAD["surrogate_type"],
                         surrogate_slope=GRAD["surrogate_slope"],
                         use_param_transform=GRAD["transform"], clip_to_bounds=False,
                         return_best=True, verbose=False)
    with _silence():
        result = train(loss_fn, handles, cfg, initial_params=INIT)
    return result.get_params_dict(), [float(x) for x in result.loss_history]


def run_nm(trace):
    cell, _, loss_fn = build(trace)
    loss_jit = jax.jit(loss_fn)
    jaxley_keys = [list(p.keys())[0] for p in cell.get_parameters()]
    curve = []

    def eval_loss(x):
        p = [{k: jnp.array([float(x[i])])} for i, k in enumerate(jaxley_keys)]
        try:
            v = float(loss_jit(p))
            v = v if math.isfinite(v) else 1e6
        except Exception:
            v = 1e6
        curve.append(v)
        return v

    x0 = np.array([INIT[n] for n in TRAINABLE])
    bounds = [(PARAM_BOUNDS[n].min, PARAM_BOUNDS[n].max) for n in TRAINABLE]
    res = scipy.optimize.minimize(
        eval_loss, x0=x0, method="Nelder-Mead", bounds=bounds,
        options={"maxfev": NM_MAXFEV, "adaptive": True, "xatol": 1e-3, "fatol": 1e-5})
    fitted = {n: float(res.x[i]) for i, n in enumerate(TRAINABLE)}
    return fitted, curve


# --- Evaluation (clean, non-surrogate) -------------------------------------

def clean_mse(fitted, trace):
    params = {**INIT, **fitted}
    with _silence():
        sim = simulate_with_current_trace(
            params=params, current_trace_pA=trace.current, dt_ms=DT_MS,
            use_surrogate=False, v_init=float(trace.voltage[0]))
    sim_v = np.asarray(sim.voltage)
    gt_v = np.asarray(trace.voltage)
    n = min(len(sim_v), len(gt_v), trace.stim_end_idx + 100)  # match mse.py window
    return float(np.mean((sim_v[:n] - gt_v[:n]) ** 2))


def iters_to_tol(curve):
    for i, v in enumerate(curve):
        if v <= CONV_TOL:
            return i
    return -1


# --- Main loop -------------------------------------------------------------

def run():
    print("Generating scenarios (non-spiking guard)...")
    scenarios = generate_scenarios()
    print(f"  {len(scenarios)} scenarios.")

    fitters = {"grad": run_grad, "nm": run_nm}
    total = len(scenarios) * len(METHODS)
    run_idx = 0

    # Overwrite (not append) so each run starts from a clean slate — no stale rows
    # from a previous run leaking into the CSVs.
    with open(OUT_CSV, "w", newline="") as f, open(CURVES_CSV, "w", newline="") as fc:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        cwriter = csv.DictWriter(fc, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        cwriter.writeheader()

        for sc in scenarios:
            init_loss = clean_mse({}, sc["trace"])
            for method in METHODS:
                run_idx += 1
                t0 = time.time()
                row = {k: float("nan") for k in CSV_FIELDS}
                row.update({"scenario_idx": sc["scenario_idx"], "base": sc["base"],
                            "method": method, "delta": sc["delta"],
                            "direction_idx": sc["direction_idx"],
                            "n_clipped": sc["n_clipped"], "init_loss": init_loss})
                for p in TRAINABLE:
                    row[f"{p}_gt"] = sc["gt"][p]
                try:
                    fitted, curve = fitters[method](sc["trace"])
                    final_loss = clean_mse(fitted, sc["trace"])
                    row["final_loss"] = final_loss
                    row["converged"] = int(final_loss <= CONV_TOL)
                    row["iters_to_tol"] = iters_to_tol(curve)
                    row["n_iter"] = len(curve)
                    for p in TRAINABLE:
                        b = PARAM_BOUNDS[p]
                        row[f"param_err_{p}"] = abs(fitted[p] - sc["gt"][p]) / (b.max - b.min)
                        row[f"{p}_fit"] = fitted[p]
                    for it, loss in enumerate(curve):
                        cwriter.writerow({
                            "scenario_idx": sc["scenario_idx"], "base": sc["base"],
                            "method": method, "delta": sc["delta"],
                            "direction_idx": sc["direction_idx"], "iter": it, "loss": loss})
                except Exception as e:
                    row["converged"] = -1  # ERROR marker
                    print(f"  ERROR (s={sc['scenario_idx']}/{method}): {e}")
                row["elapsed_s"] = time.time() - t0
                writer.writerow(row)
                f.flush()
                fc.flush()
                if run_idx % 20 == 0 or run_idx == total:
                    print(f"  [{run_idx:4d}/{total}] {sc['base']} {method} "
                          f"δ={sc['delta']} final={row['final_loss']:.4g} "
                          f"conv={row['converged']} ({row['elapsed_s']:.1f}s)")
    print(f"\nDone. Summary: {OUT_CSV}\n      Curves:  {CURVES_CSV}")


if __name__ == "__main__":
    print(f"Subthreshold benchmark: {len(BASES)} bases × {len(DELTAS)} deltas × "
          f"{N_DIRS} dirs × {len(METHODS)} methods = "
          f"{len(BASES) * len(DELTAS) * N_DIRS * len(METHODS)} runs")
    run()
