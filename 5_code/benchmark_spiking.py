"""Spike-initiation recovery benchmark (stage 1): gradient descent vs. Nelder-Mead
on the ttfs_rate loss, with adaptation switched off (a = b = 0).

The companion to `benchmark_subthreshold.py`. There the only parameters shaping a
non-spiking trace were C_m/E_L/g_L and MSE sufficed. Here we *assume those three are
already fitted perfectly* and recover the spike-initiation parameters —

    v_T, delta_T, v_reset

— from a *spiking* target trace. Adaptation is disabled (a = b = 0), so the
adaptation current w stays 0 for all time and tau_w is inert: these are the only
identifiable spike-shaping params. A later stage will fit the adaptation params
(a, b, tau_w) with Soft-DTW or other losses, building on the params recovered here.

Because these params act only through spike timing, MSE is the wrong loss (it
collapses to non-spiking solutions). We use a single spike-aware loss, ttfs_rate
(time-to-first-spike + firing rate), and ask the simple question: *how low can each
optimizer drive that loss?* — grad (Adam) vs Nelder-Mead on an *equal* per-run
budget (Adam epochs == NM function evaluations).

Design (subthreshold-style perturbation):
  * Single base (tonic, with a=b=0) × DELTAS × N_DIRS directions. For each scenario
    the INIT is the unperturbed base; the GROUND TRUTH is the base with the three
    trainable params perturbed by δ·direction. The fit recovers init -> GT.
    C_m/E_L/g_L/tau_w are identical in init and GT and never trained.
  * A spiking guard rejects any GT perturbation that yields < 2 spikes (so first
    ISI / firing rate / first-spike time are well defined).

Headline metric: the achieved ttfs_rate loss (init_loss vs final = min over the
optimization curve). Secondary diagnostics from a clean non-surrogate re-sim at the
recovered params: coincidence factor Γ, first-spike error, firing-rate error, and
normalized per-parameter recovery error.

Outputs (overwritten at start, flushed per row -> background-able):
  * results_spiking.csv         one row per run (summary metrics)
  * results_spiking_curves.csv  one row per loss evaluation (loss curves)

Run with `python benchmark_spiking.py`. Smoke config: DELTAS=[0.2], N_DIRS=2,
MAX_ITERS=20.
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
from ADoptEX.evaluation.coincidence import coincidence_factor
from ADoptEX.loss import (GuarinoLossConfig, SoftDTWLossConfig,
                          VanRossumLossConfig, extract_experimental_features,
                          make_guarino_loss_fn, make_soft_dtw_loss_fn,
                          make_ttfs_rate_loss_fn, make_van_rossum_loss_fn,
                          spike_train_from_voltage)
from ADoptEX.training.trainer import (TrainingConfig, setup_trainable_cell,
                                      train)

# --- Simulation constants --------------------------------------------------

DT_MS = 0.025
SPIKE_PEAK_MV = 10.0  # peak injected into target voltage; soft-DTW must match
SPK_DELAY, SPK_DUR, SPK_T_MAX = 20.0, 280.0, 300.0  # step uses base["I"] (500 pA)

# --- Experiment constants --------------------------------------------------

DELTAS = [0.1, 0.2, 0.4]
N_DIRS = 20
SEED = 42
BASES = ["tonic"]                      # a=b=0 collapses the other NAUD bases
LOSSES = ["ttfs_rate"]                 # stage 1: spike-initiation only
OPTIMIZERS = ["grad", "nm"]

OUT_CSV = Path(__file__).parent / "results_spiking.csv"
CURVES_CSV = Path(__file__).parent / "results_spiking_curves.csv"

# Adaptation is switched off (a=b=0) -> w stays 0, tau_w inert. The only
# identifiable spike-shaping params are these three: perturbed in the GT, fit from
# the base init. C_m/E_L/g_L/tau_w are held at the base value and never trained.
TRAINABLE = ["v_T", "delta_T", "v_reset"]
PERTURB_PARAMS = TRAINABLE

# Accuracy-metric tolerances (define "success" and normalize accuracy_err).
FIRST_SPIKE_TOL_MS = 5.0
RATE_TOL = 0.10

MIN_GT_SPIKES = 2        # spiking guard: GT must fire at least this many spikes
MAX_RESAMPLE = 200       # guard attempts per scenario slot

# Equal like-for-like budget: Adam epochs == NM function evaluations.
MAX_ITERS = 200

# Per-loss gradient hyperparameters. Seeded from benchmark_recovery.py's tuned
# per-stage values; refine with benchmark_spiking_grad_tuning.py and paste back.
# optimizer=adam and surrogate_type=superspike are fixed across losses.
GRAD_HP = {
    "guarino":    dict(lr=0.005, surrogate_slope=5.0, transform=True),
    "van_rossum": dict(lr=0.05, surrogate_slope=10.0, transform=True),
    "soft_dtw":   dict(lr=0.01, surrogate_slope=10.0, transform=True),
    "ttfs_rate":  dict(lr=0.5, surrogate_slope=10.0, transform=False),
}

CSV_FIELDS = [
    "scenario_idx", "base", "loss", "optimizer", "delta", "direction_idx",
    "n_clipped", "n_spikes_gt", "n_spikes_sim",
    "first_spike_gt_ms", "first_spike_sim_ms", "first_spike_err_ms",
    "rate_gt_hz", "rate_sim_hz", "rate_rel_err",
    "accuracy_err", "accuracy_success", "gamma", "init_gamma",
    "init_loss", "final_loss", "n_iter", "elapsed_s",
] + [f"param_err_{n}" for n in TRAINABLE] \
  + [f"{n}_gt" for n in TRAINABLE] + [f"{n}_fit" for n in TRAINABLE]

CURVE_FIELDS = ["scenario_idx", "base", "loss", "optimizer", "delta",
                "direction_idx", "iter", "loss"]


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


# --- Perturbation ----------------------------------------------------------

def sample_direction(rng, n_params):
    raw = rng.standard_normal(n_params)
    return raw / np.linalg.norm(raw)


def make_gt(base_params, direction, delta):
    """Perturb the 6 trainable params of the base; clip to bounds.

    Starts from the full base dict (keeps C_m/E_L/g_L, v_threshold, I, ...) so the
    cell builder and the GT sim have every field. Returns (gt, n_clipped).
    """
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
    """Fully-perturbed, guaranteed-spiking recovery scenarios (subthreshold-style:
    init = base, GT = base with the 6 trainable params perturbed)."""
    rng_master = np.random.default_rng(SEED)
    scenarios = []
    sidx = 0
    for base in BASES:
        base_params = dict(NAUD_PARAMETERS[base])
        base_params["a"] = 0.0  # adaptation off -> w stays 0, tau_w inert
        base_params["b"] = 0.0
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31)))
        for delta in DELTAS:
            count = 0
            attempts = 0
            while count < N_DIRS:
                attempts += 1
                if attempts > MAX_RESAMPLE * N_DIRS:
                    raise RuntimeError(f"spiking guard exhausted: {base} δ={delta}")
                direction = sample_direction(rng, len(PERTURB_PARAMS))
                gt, n_clipped = make_gt(base_params, direction, delta)
                trace = make_synthetic_trace(base_params["I"], SPK_T_MAX, SPK_DELAY,
                                             SPK_DUR, gt)
                if len(trace.spike_times) < MIN_GT_SPIKES:
                    continue  # GT (near-)silent -> reject, resample
                scenarios.append(dict(
                    scenario_idx=sidx, base=base, delta=delta, direction_idx=count,
                    init=dict(base_params), gt=gt, trace=trace, n_clipped=n_clipped,
                    n_spikes_gt=len(trace.spike_times)))
                sidx += 1
                count += 1
    return scenarios


# --- Cell / loss build (shared by both optimizers) -------------------------

def build_loss(loss_name, cell, data_stimuli, t_max, trace):
    se = trace.stim_end_idx
    if loss_name == "soft_dtw":
        return make_soft_dtw_loss_fn(
            cell=cell, data_stimuli=data_stimuli, t_max=t_max, dt_ms=DT_MS,
            exp_voltage=jnp.asarray(trace.voltage), stim_end_index=se,
            loss_config=SoftDTWLossConfig(spike_peak_mv=SPIKE_PEAK_MV))
    if loss_name == "van_rossum":
        exp_train = spike_train_from_voltage(jnp.asarray(trace.voltage),
                                             threshold_mv=0.0, dt_ms=DT_MS)
        return make_van_rossum_loss_fn(
            cell=cell, data_stimuli=data_stimuli, t_max=t_max, dt_ms=DT_MS,
            exp_spike_train=exp_train, stim_end_index=se,
            loss_config=VanRossumLossConfig())
    # guarino & ttfs_rate both need soft experimental features
    feat = extract_experimental_features(
        voltage_trace=jnp.asarray(trace.voltage), dt_ms=DT_MS,
        stim_duration_ms=trace.stim_duration_ms, stim_end_index=se,
        spike_threshold_mv=-10.0)
    if loss_name == "guarino":
        return make_guarino_loss_fn(
            cell=cell, data_stimuli=data_stimuli, t_max=t_max, dt_ms=DT_MS,
            exp_features=feat, stim_duration_ms=trace.stim_duration_ms,
            stim_end_index=se, loss_config=GuarinoLossConfig(weight_spike_count=0.3),
            temperature=0.3, beta=5.0, validity_beta=5.0)
    if loss_name == "ttfs_rate":
        return make_ttfs_rate_loss_fn(
            cell=cell, data_stimuli=data_stimuli, t_max=t_max, dt_ms=DT_MS,
            exp_t_first_spike=feat.t_first_spike,
            exp_firing_frequency=feat.firing_frequency,
            stim_duration_ms=trace.stim_duration_ms, stim_end_index=se,
            exp_has_spike=feat.has_first_spike)
    raise ValueError(f"unknown loss: {loss_name}")


def build(init, trace, loss_name, surrogate_slope=None):
    slope = surrogate_slope if surrogate_slope is not None \
        else GRAD_HP[loss_name]["surrogate_slope"]
    cfg = TrainingConfig(surrogate_type="superspike", surrogate_slope=slope,
                         clip_to_bounds=False, use_param_transform=False,
                         verbose=False)
    with _silence():
        cell, data_stimuli, t_max, handles = setup_trainable_cell(
            initial_params=init, current_trace_pA=jnp.asarray(trace.current),
            dt_ms=DT_MS, config=cfg, trainable_params=TRAINABLE,
            v_init=float(trace.voltage[0]))
    loss_fn = build_loss(loss_name, cell, data_stimuli, t_max, trace)
    return cell, handles, loss_fn


def run_grad(init, trace, loss_name, hp=None):
    hp = hp or GRAD_HP[loss_name]
    _, handles, loss_fn = build(init, trace, loss_name, hp["surrogate_slope"])
    cfg = TrainingConfig(optimizer="adam", learning_rate=hp["lr"],
                         n_epochs=MAX_ITERS, surrogate_type="superspike",
                         surrogate_slope=hp["surrogate_slope"],
                         use_param_transform=hp["transform"], clip_to_bounds=False,
                         return_best=True, verbose=False)
    with _silence():
        result = train(loss_fn, handles, cfg, initial_params=init)
    return result.get_params_dict(), [float(x) for x in result.loss_history]


def run_nm(init, trace, loss_name):
    cell, _, loss_fn = build(init, trace, loss_name)
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

    x0 = np.array([init[n] for n in TRAINABLE])
    bounds = [(PARAM_BOUNDS[n].min, PARAM_BOUNDS[n].max) for n in TRAINABLE]
    res = scipy.optimize.minimize(
        eval_loss, x0=x0, method="Nelder-Mead", bounds=bounds,
        options={"maxfev": MAX_ITERS, "adaptive": True, "xatol": 1e-3, "fatol": 1e-4})
    fitted = {n: float(res.x[i]) for i, n in enumerate(TRAINABLE)}
    return fitted, curve


# --- Evaluation (clean, non-surrogate; loss-agnostic) ----------------------

def evaluate(fitted, base, gt_trace) -> dict:
    """Re-sim with base params overridden by the recovered trainable params and
    score against the GT trace. Mirrors benchmark_recovery.evaluate."""
    params = {**base, **fitted}
    with _silence():
        sim = simulate_with_current_trace(
            params=params, current_trace_pA=gt_trace.current, dt_ms=DT_MS,
            use_surrogate=False, v_init=float(gt_trace.voltage[0]))
    dur_s = gt_trace.stim_duration_ms / 1000.0
    gt_st, sim_st = gt_trace.spike_times, sim.spike_times
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


def param_errors(fitted, gt):
    return {
        f"param_err_{n}": abs(fitted[n] - gt[n])
                          / (PARAM_BOUNDS[n].max - PARAM_BOUNDS[n].min)
        for n in TRAINABLE
    }


# --- Main loop -------------------------------------------------------------

def run():
    print("Generating scenarios (spiking guard)...")
    scenarios = generate_scenarios()
    print(f"  {len(scenarios)} scenarios.")

    fitters = {"grad": run_grad, "nm": run_nm}
    total = len(scenarios) * len(LOSSES) * len(OPTIMIZERS)
    run_idx = 0

    # Overwrite (not append) so each run starts from a clean slate.
    with open(OUT_CSV, "w", newline="") as f, open(CURVES_CSV, "w", newline="") as fc:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        cwriter = csv.DictWriter(fc, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        cwriter.writeheader()

        for sc in scenarios:
            init_eval = evaluate({}, sc["init"], sc["trace"])  # base vs GT baseline
            for loss_name in LOSSES:
                for optimizer in OPTIMIZERS:
                    run_idx += 1
                    t0 = time.time()
                    row = {k: float("nan") for k in CSV_FIELDS}
                    row.update({
                        "scenario_idx": sc["scenario_idx"], "base": sc["base"],
                        "loss": loss_name, "optimizer": optimizer,
                        "delta": sc["delta"], "direction_idx": sc["direction_idx"],
                        "n_clipped": sc["n_clipped"], "n_spikes_gt": sc["n_spikes_gt"],
                        "init_gamma": init_eval["gamma"]})
                    for p in TRAINABLE:
                        row[f"{p}_gt"] = sc["gt"][p]
                    try:
                        fitted, curve = fitters[optimizer](
                            sc["init"], sc["trace"], loss_name)
                        row["init_loss"] = curve[0] if curve else float("nan")
                        row["final_loss"] = min(curve) if curve else float("nan")
                        row["n_iter"] = len(curve)
                        row.update(evaluate(fitted, sc["init"], sc["trace"]))
                        row.update(param_errors(fitted, sc["gt"]))
                        for p in TRAINABLE:
                            row[f"{p}_fit"] = fitted[p]
                        for it, loss in enumerate(curve):
                            cwriter.writerow({
                                "scenario_idx": sc["scenario_idx"], "base": sc["base"],
                                "loss": loss_name, "optimizer": optimizer,
                                "delta": sc["delta"],
                                "direction_idx": sc["direction_idx"],
                                "iter": it, "loss": loss})
                    except Exception as e:
                        row["accuracy_success"] = -1  # ERROR marker
                        print(f"  ERROR (s={sc['scenario_idx']}/{loss_name}/"
                              f"{optimizer}): {e}")
                    row["elapsed_s"] = time.time() - t0
                    writer.writerow(row)
                    f.flush()
                    fc.flush()
                    if run_idx % 20 == 0 or run_idx == total:
                        print(f"  [{run_idx:4d}/{total}] {sc['base']} {loss_name} "
                              f"{optimizer} δ={sc['delta']} γ={row['gamma']:.2f} "
                              f"acc_err={row['accuracy_err']:.2f} "
                              f"({row['elapsed_s']:.1f}s)")
    print(f"\nDone. Summary: {OUT_CSV}\n      Curves:  {CURVES_CSV}")


if __name__ == "__main__":
    print(f"Spiking benchmark: {len(BASES)} bases × {len(DELTAS)} deltas × "
          f"{N_DIRS} dirs × {len(LOSSES)} losses × {len(OPTIMIZERS)} optimizers = "
          f"{len(BASES) * len(DELTAS) * N_DIRS * len(LOSSES) * len(OPTIMIZERS)} runs")
    run()
