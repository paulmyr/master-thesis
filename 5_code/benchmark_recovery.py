"""Multistage recovery benchmark: gradient descent vs. Nelder-Mead.

Faithfully runs the multistage curriculum demonstrated in
`notebooks/training_synthetic_multistage.ipynb` (MSE subthreshold -> TTFS-rate
spike-init -> Guarino adaptation -> soft-DTW polish) against Nelder-Mead, in a
2x2 design {grad, nm} x {multistage, naive}:

  grad_multistage  the demonstrated curriculum, Adam per stage
  nm_multistage    same curriculum, Nelder-Mead per stage (optimizer swap)
  grad_naive       single stage, all 9 params at once, Guarino, Adam
  nm_naive         single stage, all 9 params at once, Guarino, Nelder-Mead

Per (scenario, δ, direction) all four methods start from the *same* perturbed
init (paired comparison). Only the subthreshold params (C_m, E_L, g_L) are
perturbed by δ·direction; the spike/adaptation params always start at fixed
physiological *spiking* defaults (SPIKE_DEFAULTS) so gradient descent never
starts in a non-spiking dead zone — this is the mechanism the benchmark tests.
Every cell is built with `v_init = trace.voltage[0]` (GT start voltage).

Primary metric (relaxed, supervisor-requested): first-spike-timing error +
mean firing-rate relative error on a clean (non-surrogate) sim at the recovered
params. Γ (coincidence factor) and per-parameter recovery error are secondary.

Run with `python benchmark_recovery.py`. CSV is flushed per row (resumable /
background-able). Smoke config: set N_DIRS=2, DELTAS=[0.2], SCENARIOS to one key.
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

from ADoptEX.core.data import TraceData, crop_to_stim_window
from ADoptEX.core.parameters import NAUD_PARAMETERS, PARAM_BOUNDS
from ADoptEX.core.simulation import simulate_with_current_trace
from ADoptEX.evaluation.coincidence import coincidence_factor
from ADoptEX.loss import (GuarinoLossConfig, SoftDTWLossConfig,
                          extract_experimental_features, make_guarino_loss_fn,
                          make_mse_loss_fn, make_soft_dtw_loss_fn,
                          make_ttfs_rate_loss_fn)
from ADoptEX.training.trainer import (TrainingConfig, setup_trainable_cell,
                                      train)

# --- Simulation constants --------------------------------------------------

DT_MS = 0.025
SPIKE_PEAK_MV = 10.0  # peak injected into target voltage; soft-DTW must match

SUB_CURRENT_PA = 100.0      # subthreshold step (below rheobase ~200 pA)
SUB_T_MAX, SUB_DELAY, SUB_DUR = 350.0, 20.0, 300.0
SPK_DELAY, SPK_DUR, SPK_T_MAX = 20.0, 280.0, 300.0  # spiking step uses scenario["I"]

# --- Experiment constants --------------------------------------------------

DELTAS = [0.1, 0.2, 0.4]
N_DIRS = 30
SEED = 42
METHODS = ["grad_multistage", "nm_multistage", "grad_naive", "nm_naive"]
OUT_CSV = Path(__file__).parent / "results_recovery.csv"

SCENARIOS = {
    "tonic":            dict(NAUD_PARAMETERS["tonic"]),
    "adaptation":       dict(NAUD_PARAMETERS["adaptation"]),
    "initial_bursting": dict(NAUD_PARAMETERS["initial_bursting"]),
}

# δ perturbs only the subthreshold params; spike/adaptation params are reset to
# physiological spiking defaults regardless of perturbation (decision: keep grad
# in a spiking regime).
PERTURB_PARAMS = ["C_m", "E_L", "g_L"]
SPIKE_DEFAULTS = {"v_T": -50.0, "delta_T": 2.0, "v_reset": -60.0,
                  "tau_w": 30.0, "a": 2.0, "b": 0.0}
ALL_PARAMS = ["C_m", "g_L", "E_L", "v_T", "delta_T", "v_reset", "tau_w", "a", "b"]

# Accuracy-metric tolerances (define "success" and normalize accuracy_err).
FIRST_SPIKE_TOL_MS = 5.0
RATE_TOL = 0.10

NM_MAXFEV_STAGE = 600    # per-stage NM budget (low-dim groups)
NM_MAXFEV_NAIVE = 2500   # all-9-params single-stage NM budget

# Multistage curriculum (mirrors training_synthetic_multistage.ipynb).
STAGES = [
    dict(name="s1_mse", trace="s1", trainable=["C_m", "E_L", "g_L"], loss="mse",
         grad=dict(optimizer="adam", lr=0.2, epochs=100,
                   surrogate_type="superspike", surrogate_slope=10.0, transform=False)),
    dict(name="s2_mse", trace="s2", trainable=["C_m", "E_L", "g_L"], loss="mse",
         grad=dict(optimizer="adam", lr=0.05, epochs=100,
                   surrogate_type="superspike", surrogate_slope=10.0, transform=False)),
    dict(name="s3_ttfs", trace="spk", reset_spike_defaults=True,
         trainable=["v_T", "delta_T", "v_reset"], loss="ttfs",
         grad=dict(optimizer="adam", lr=0.5, epochs=100,
                   surrogate_type="superspike", surrogate_slope=10.0, transform=False)),
    dict(name="s4_guarino", trace="spk",
         trainable=["C_m", "g_L", "E_L", "v_T", "delta_T", "v_reset", "tau_w", "a", "b"],
         loss="guarino",
         grad=dict(optimizer="adam", lr=0.005, epochs=100,
                   surrogate_type="superspike", surrogate_slope=5.0, transform=True)),
    dict(name="s5_softdtw", trace="spk", trainable=list(ALL_PARAMS), loss="softdtw",
         grad=dict(optimizer="adam", lr=0.01, epochs=150,
                   surrogate_type="superspike", surrogate_slope=10.0, transform=True)),
]

NAIVE_STAGE = dict(
    name="naive", trace="spk", trainable=list(ALL_PARAMS), loss="guarino",
    grad=dict(optimizer="adam", lr=0.01, epochs=300,
              surrogate_type="superspike", surrogate_slope=10.0, transform=True),
)

CSV_FIELDS = [
    "scenario", "method", "delta", "direction_idx", "n_clipped",
    "n_spikes_gt", "n_spikes_sim",
    "first_spike_gt_ms", "first_spike_sim_ms", "first_spike_err_ms",
    "rate_gt_hz", "rate_sim_hz", "rate_rel_err",
    "accuracy_err", "accuracy_success", "gamma",
    "n_iter", "elapsed_s",
] + [f"param_err_{n}" for n in ALL_PARAMS]


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
    """GT forward sim (non-surrogate) packaged as TraceData, peaks injected."""
    n = int(round(t_max_ms / DT_MS))
    current, s_idx, e_idx = _step_current(n, delay_ms, dur_ms, amp_pA)
    with _silence():
        sim = simulate_with_current_trace(
            params=gt_params, current_trace_pA=current, dt_ms=DT_MS,
            use_surrogate=False, v_init=gt_params["E_L"],
        )
    voltage = sim.voltage[:n].copy()
    for st in sim.spike_times:
        idx = int(st / DT_MS)
        if 0 <= idx < len(voltage):
            voltage[idx] = SPIKE_PEAK_MV
    return TraceData(
        time=np.arange(n) * DT_MS, voltage=voltage, current=current, dt_ms=DT_MS,
        spike_times=sim.spike_times, stim_start_idx=s_idx, stim_end_idx=e_idx,
        stim_current_pA=float(amp_pA),
    )


def build_traces(scenario_params) -> dict:
    """Subthreshold trace (cropped for S1/S2) + spiking trace (S3-S5, naive)."""
    sub = make_synthetic_trace(SUB_CURRENT_PA, SUB_T_MAX, SUB_DELAY, SUB_DUR,
                               scenario_params)
    spk = make_synthetic_trace(scenario_params["I"], SPK_T_MAX, SPK_DELAY, SPK_DUR,
                               scenario_params)
    return {
        "s1": crop_to_stim_window(sub, max_duration_ms=30.0, padding_ms_before=10.0),
        "s2": crop_to_stim_window(sub, max_duration_ms=100.0, padding_ms_before=30.0),
        "spk": spk,
    }


# --- Perturbation ----------------------------------------------------------

def sample_directions(rng, n_dirs, n_params):
    raw = rng.standard_normal((n_dirs, n_params))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw


def make_init(scenario_params, direction, delta):
    """Perturb subthreshold params; reset spike/adaptation params to defaults.

    Starts from the full scenario param dict (keeps v_threshold, I, ...) so the
    cell builder has every field it needs.
    """
    out = dict(scenario_params)
    n_clipped = 0
    for i, name in enumerate(PERTURB_PARAMS):
        b = PARAM_BOUNDS[name]
        proposed = scenario_params[name] + delta * direction[i] * (b.max - b.min)
        clipped = float(np.clip(proposed, b.min, b.max))
        n_clipped += clipped != proposed
        out[name] = clipped
    out.update(SPIKE_DEFAULTS)
    return out, n_clipped


# --- Loss + optimizer dispatch ---------------------------------------------

def build_stage_loss(stage, cell, data_stimuli, t_max, trace):
    lt = stage["loss"]
    se = trace.stim_end_idx
    if lt == "mse":
        return make_mse_loss_fn(cell=cell, data_stimuli=data_stimuli, t_max=t_max,
                                dt_ms=DT_MS, exp_voltage=jnp.asarray(trace.voltage),
                                stim_end_index=se)
    feat = extract_experimental_features(
        voltage_trace=jnp.asarray(trace.voltage), dt_ms=DT_MS,
        stim_duration_ms=trace.stim_duration_ms, stim_end_index=se,
        spike_threshold_mv=-10.0,
    )
    if lt == "ttfs":
        return make_ttfs_rate_loss_fn(
            cell=cell, data_stimuli=data_stimuli, t_max=t_max, dt_ms=DT_MS,
            exp_t_first_spike=feat.t_first_spike,
            exp_firing_frequency=feat.firing_frequency,
            stim_duration_ms=trace.stim_duration_ms, stim_end_index=se,
            exp_has_spike=feat.has_first_spike,
        )
    if lt == "guarino":
        return make_guarino_loss_fn(
            cell=cell, data_stimuli=data_stimuli, t_max=t_max, dt_ms=DT_MS,
            exp_features=feat, stim_duration_ms=trace.stim_duration_ms,
            stim_end_index=se, loss_config=GuarinoLossConfig(weight_spike_count=0.3),
            temperature=0.3, beta=5.0, validity_beta=5.0,
        )
    if lt == "softdtw":
        return make_soft_dtw_loss_fn(
            cell=cell, data_stimuli=data_stimuli, t_max=t_max, dt_ms=DT_MS,
            exp_voltage=jnp.asarray(trace.voltage), stim_end_index=se,
            loss_config=SoftDTWLossConfig(spike_peak_mv=SPIKE_PEAK_MV),
        )
    raise ValueError(f"unknown loss: {lt}")


def _setup(stage, params, trace):
    g = stage["grad"]
    cfg = TrainingConfig(surrogate_type=g["surrogate_type"],
                         surrogate_slope=g["surrogate_slope"],
                         clip_to_bounds=False, use_param_transform=False, verbose=False)
    with _silence():
        return setup_trainable_cell(
            initial_params=params,
            current_trace_pA=jnp.asarray(trace.current), dt_ms=DT_MS, config=cfg,
            trainable_params=stage["trainable"], v_init=float(trace.voltage[0]),
        )


def run_grad_stage(stage, handles, loss_fn, params):
    g = stage["grad"]
    cfg = TrainingConfig(optimizer=g["optimizer"], learning_rate=g["lr"],
                         n_epochs=g["epochs"], surrogate_type=g["surrogate_type"],
                         surrogate_slope=g["surrogate_slope"],
                         use_param_transform=g["transform"], clip_to_bounds=False,
                         return_best=True, verbose=False)
    with _silence():
        result = train(loss_fn, handles, cfg, initial_params=params)
    return result.get_params_dict(), float(result.best_loss), len(result.loss_history)


def run_nm_stage(stage, cell, loss_fn, params, maxfev):
    loss_jit = jax.jit(loss_fn)
    jaxley_keys = [list(p.keys())[0] for p in cell.get_parameters()]
    trainable = stage["trainable"]

    def eval_loss(x):
        p = [{k: jnp.array([float(x[i])])} for i, k in enumerate(jaxley_keys)]
        try:
            v = float(loss_jit(p))
            return v if math.isfinite(v) else 1e6
        except Exception:
            return 1e6

    x0 = np.array([params[n] for n in trainable])
    bounds = [(PARAM_BOUNDS[n].min, PARAM_BOUNDS[n].max) for n in trainable]
    res = scipy.optimize.minimize(
        eval_loss, x0=x0, method="Nelder-Mead", bounds=bounds,
        options={"maxfev": maxfev, "adaptive": True, "xatol": 1e-3, "fatol": 1e-4},
    )
    trained = {n: float(res.x[i]) for i, n in enumerate(trainable)}
    best = float(res.fun) if math.isfinite(res.fun) else float("inf")
    return trained, best, int(res.nfev)


def _run_stage(stage, params, trace, optimizer_kind, nm_maxfev):
    cell, data_stimuli, t_max, handles = _setup(stage, params, trace)
    loss_fn = build_stage_loss(stage, cell, data_stimuli, t_max, trace)
    init_loss = float(jax.jit(loss_fn)(handles))
    if optimizer_kind == "grad":
        trained, best, n_iter = run_grad_stage(stage, handles, loss_fn, params)
    else:
        trained, best, n_iter = run_nm_stage(stage, cell, loss_fn, params, nm_maxfev)
    # Keep the stage update only if it improved; otherwise carry params forward.
    if math.isfinite(best) and best < init_loss:
        params = {**params, **trained}
    return params, n_iter


def fit_multistage(traces, init_params, optimizer_kind):
    params = dict(init_params)
    total_iter = 0
    for stage in STAGES:
        if stage.get("reset_spike_defaults"):
            params.update(SPIKE_DEFAULTS)
        params, n_iter = _run_stage(stage, params, traces[stage["trace"]],
                                    optimizer_kind, NM_MAXFEV_STAGE)
        total_iter += n_iter
    return params, total_iter


def fit_naive(traces, init_params, optimizer_kind):
    params, n_iter = _run_stage(NAIVE_STAGE, dict(init_params), traces["spk"],
                                optimizer_kind, NM_MAXFEV_NAIVE)
    return params, n_iter


# --- Evaluation (clean, non-differentiable) --------------------------------

def evaluate(final_params, scenario_params, spk_trace) -> dict:
    params = {**scenario_params, **final_params}
    with _silence():
        sim = simulate_with_current_trace(
            params=params, current_trace_pA=spk_trace.current, dt_ms=DT_MS,
            use_surrogate=False, v_init=float(spk_trace.voltage[0]),
        )
    dur_s = spk_trace.stim_duration_ms / 1000.0
    gt_st, sim_st = spk_trace.spike_times, sim.spike_times
    n_gt, n_sim = len(gt_st), len(sim_st)
    rate_gt, rate_sim = n_gt / dur_s, n_sim / dur_s

    t0_gt = float(gt_st[0]) if n_gt else float("nan")
    if n_sim and n_gt:
        t0_sim = float(sim_st[0])
        first_err = abs(t0_sim - t0_gt)
    else:
        t0_sim = float("nan")
        first_err = SPK_DUR  # maximally-late penalty when sim is silent
    rate_rel = abs(rate_sim - rate_gt) / rate_gt if rate_gt > 0 else float("nan")

    acc_err = first_err / FIRST_SPIKE_TOL_MS + rate_rel / RATE_TOL
    success = (first_err < FIRST_SPIKE_TOL_MS) and (rate_rel < RATE_TOL)

    if n_sim and n_gt:
        cf = coincidence_factor(spike_times_data=gt_st, spike_times_model=sim_st,
                                duration_ms=SPK_T_MAX, delta_ms=2.0)
        gamma = float(cf.gamma)
    else:
        gamma = 0.0
    return {
        "n_spikes_sim": n_sim,
        "first_spike_gt_ms": t0_gt, "first_spike_sim_ms": t0_sim,
        "first_spike_err_ms": first_err,
        "rate_gt_hz": rate_gt, "rate_sim_hz": rate_sim, "rate_rel_err": rate_rel,
        "accuracy_err": acc_err, "accuracy_success": int(success), "gamma": gamma,
    }


def param_errors(final_params, theta_gt):
    return {
        f"param_err_{n}": abs(final_params[n] - theta_gt[n])
                          / (PARAM_BOUNDS[n].max - PARAM_BOUNDS[n].min)
        for n in ALL_PARAMS
    }


# --- Main loop -------------------------------------------------------------

def run():
    rng_master = np.random.default_rng(SEED)
    cache = {}
    for name, sp in SCENARIOS.items():
        print(f"GT simulating ({name})...")
        traces = build_traces(sp)
        dir_seed = int(rng_master.integers(0, 2**31))
        directions = sample_directions(np.random.default_rng(dir_seed), N_DIRS,
                                       len(PERTURB_PARAMS))
        cache[name] = {
            "scenario_params": sp, "traces": traces, "directions": directions,
            "theta_gt": {n: float(sp[n]) for n in ALL_PARAMS},
            "n_spikes_gt": len(traces["spk"].spike_times),
        }
        print(f"  done. n_spikes_gt(spiking)={cache[name]['n_spikes_gt']}, "
              f"n_spikes(sub)={len(traces['s2'].spike_times)}")

    fitters = {
        "grad_multistage": lambda tr, ip: fit_multistage(tr, ip, "grad"),
        "nm_multistage":   lambda tr, ip: fit_multistage(tr, ip, "nm"),
        "grad_naive":      lambda tr, ip: fit_naive(tr, ip, "grad"),
        "nm_naive":        lambda tr, ip: fit_naive(tr, ip, "nm"),
    }

    write_header = not OUT_CSV.exists()
    total = len(SCENARIOS) * len(DELTAS) * N_DIRS * len(METHODS)
    run_idx = 0
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for name, c in cache.items():
            for delta in DELTAS:
                for d in range(N_DIRS):
                    init_params, n_clipped = make_init(c["scenario_params"],
                                                       c["directions"][d], delta)
                    for method in METHODS:
                        run_idx += 1
                        t0 = time.time()
                        row = {k: float("nan") for k in CSV_FIELDS}
                        row.update({"scenario": name, "method": method,
                                    "delta": delta, "direction_idx": d,
                                    "n_clipped": n_clipped,
                                    "n_spikes_gt": c["n_spikes_gt"]})
                        try:
                            final_params, n_iter = fitters[method](
                                c["traces"], init_params)
                            row.update(evaluate(final_params, c["scenario_params"],
                                                c["traces"]["spk"]))
                            row.update(param_errors(final_params, c["theta_gt"]))
                            row["n_iter"] = n_iter
                        except Exception as e:
                            row["accuracy_success"] = -1  # ERROR marker
                            print(f"  ERROR ({name}/{method}/δ={delta}/d={d}): {e}")
                        row["elapsed_s"] = time.time() - t0
                        writer.writerow(row)
                        f.flush()
                        if run_idx % 10 == 0 or run_idx == total:
                            print(f"  [{run_idx:4d}/{total}] {name} {method} "
                                  f"δ={delta} d={d} acc_err={row['accuracy_err']:.2f} "
                                  f"γ={row['gamma']:.2f} ({row['elapsed_s']:.1f}s)")
    print(f"\nDone. Results: {OUT_CSV}")


if __name__ == "__main__":
    print(f"Recovery benchmark: {len(SCENARIOS)} scenarios × {len(DELTAS)} deltas "
          f"× {N_DIRS} dirs × {len(METHODS)} methods = "
          f"{len(SCENARIOS) * len(DELTAS) * N_DIRS * len(METHODS)} runs")
    run()
