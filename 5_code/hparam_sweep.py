"""Random hyperparameter sweep for Guarino loss on synthetic ground truth.

Saves (hparams, metrics) for each trial to results_hparam_sweep.csv.
Plot heatmaps separately from the saved CSV.

Parallelized via multiprocessing: N_WORKERS processes each handle many trials.
Each worker pays its own JIT compile cost (once), so set N_WORKERS to your
core count and N_TRIALS to be much larger than N_WORKERS for amortization.
"""

import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import jax.numpy as jnp
import numpy as np
from jax import config

from ADoptEX.core.data import TraceData
from ADoptEX.core.parameters import NAUD_PARAMETERS, PARAM_BOUNDS
from ADoptEX.core.simulation import simulate_with_current_trace
from ADoptEX.evaluation.coincidence import coincidence_factor
from ADoptEX.loss import extract_experimental_features, make_guarino_loss_fn
from ADoptEX.training.trainer import (TrainingConfig, setup_trainable_cell,
                                      train_scan)

config.update("jax_platform_name", "cpu")

N_TRIALS = 1000
N_EPOCHS = 50
SEED = 42
N_WORKERS = 8
TRAINABLE = ["C_m", "g_L", "v_reset", "v_T", "E_L", "delta_T"]
OUT_CSV = "results_hparam_sweep.csv"

# --- Synthetic ground truth (runs in main + every worker on spawn) ---
dt_ms = 0.025
t_max_ms = 100.0
n = int(round(t_max_ms / dt_ms))
current = np.zeros(n)
current[int(5.0 / dt_ms): int(100.0 / dt_ms)] = 500.0
gt = dict(NAUD_PARAMETERS["tonic"])

sim_gt = simulate_with_current_trace(gt, current, dt_ms, use_surrogate=False)
voltage_gt = sim_gt.voltage[:n].copy()
for st in sim_gt.spike_times:
    idx = int(st / dt_ms)
    if 0 <= idx < n:
        voltage_gt[idx] = 35.0
data = TraceData(
    time=np.arange(n) * dt_ms, voltage=voltage_gt, current=current, dt_ms=dt_ms,
    spike_times=sim_gt.spike_times, stim_start_idx=int(5.0 / dt_ms),
    stim_end_idx=n, stim_current_pA=500.0,
)


def sample_init(rng_, gt_):
    """Sample initial parameters uniformly from PARAM_BOUNDS for trainable params."""
    init_ = dict(gt_)
    for name in TRAINABLE:
        b = PARAM_BOUNDS[name]
        init_[name] = float(rng_.uniform(b.min, b.max))
    return init_


def init_distance(init_, gt_):
    """Mean relative distance from GT, normalized by bounds range."""
    dists = []
    for name in TRAINABLE:
        b = PARAM_BOUNDS[name]
        dists.append(abs(init_[name] - gt_[name]) / (b.max - b.min))
    return float(np.mean(dists))


def run_trial(args):
    """Single-trial worker function. Picklable; safe in subprocess."""
    i, slope, temp, beta_, lr, alpha, beta_p, init = args
    t0 = time.time()
    cfg = TrainingConfig(
        optimizer="polyak", learning_rate=lr, n_epochs=N_EPOCHS,
        polyak_alpha=alpha, polyak_beta=beta_p,
        surrogate_type="sigmoid", surrogate_slope=slope,
        clip_to_bounds=False, use_param_transform=True, verbose=False,
    )
    try:
        cell, stim, t_max, params = setup_trainable_cell(
            init, current, dt_ms, cfg, TRAINABLE,
        )
        exp_feats = extract_experimental_features(
            jnp.array(data.voltage), dt_ms, data.stim_duration_ms, data.stim_end_idx,
        )
        loss_fn = make_guarino_loss_fn(
            cell, stim, t_max, dt_ms, exp_feats, data.stim_duration_ms,
            data.stim_end_idx, temperature=temp, beta=beta_,
        )
        final_params, history = train_scan(loss_fn, params, cfg)
        final_loss = float(history[-1])
        min_loss = float(jnp.nanmin(history))
        recovered = {**init}
        for d in final_params:
            for k_, v in d.items():
                name = k_.replace("AdEx_", "").replace("capacitance", "C_m")
                recovered[name] = float(v.flatten()[0])
        sim = simulate_with_current_trace(recovered, current, dt_ms, use_surrogate=False)
        if sim.n_spikes > 0 and data.n_spikes > 0:
            gamma = float(coincidence_factor(
                data.spike_times, sim.spike_times, data.stim_duration_ms, 2.0,
            ).gamma)
        else:
            gamma = float("nan")
        n_spk = sim.n_spikes
    except Exception as e:
        final_loss, min_loss, gamma, n_spk = float("nan"), float("nan"), float("nan"), 0
        print(f"trial {i}: {e}")

    elapsed = time.time() - t0
    d_init = init_distance(init, gt)
    init_vals = [init[name] for name in TRAINABLE]
    return (i, slope, temp, beta_, lr, alpha, beta_p, d_init, *init_vals,
            final_loss, min_loss, gamma, n_spk, elapsed)


if __name__ == "__main__":
    # --- Sample all hparams + inits in main process ---
    rng = np.random.default_rng(SEED)
    slopes = rng.uniform(0.5, 10.0, N_TRIALS)
    temps = rng.uniform(0.1, 0.5, N_TRIALS)
    betas = rng.uniform(1.0, 15.0, N_TRIALS)
    lrs = 10 ** rng.uniform(-2, 0, N_TRIALS)
    polyak_alphas = rng.uniform(0.5, 1.5, N_TRIALS)
    polyak_betas = rng.uniform(0.5, 1.0, N_TRIALS)
    inits = [sample_init(rng, gt) for _ in range(N_TRIALS)]
    args_list = [(i, float(slopes[i]), float(temps[i]), float(betas[i]),
                  float(lrs[i]), float(polyak_alphas[i]), float(polyak_betas[i]),
                  inits[i]) for i in range(N_TRIALS)]

    init_cols = [f"init_{name}" for name in TRAINABLE]
    header = ["trial", "slope", "temperature", "beta", "lr",
              "polyak_alpha", "polyak_beta", "init_dist", *init_cols,
              "final_loss", "min_loss", "gamma", "n_spikes_sim", "elapsed_s"]

    t_start = time.time()
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(run_trial, a): a[0] for a in args_list}
            done = 0
            for fut in as_completed(futures):
                row = fut.result()
                w.writerow(row); f.flush()
                done += 1
                i, slope, temp, beta_, lr, alpha, beta_p, d_init, *rest = row
                min_loss = rest[len(TRAINABLE) + 1]
                gamma = rest[len(TRAINABLE) + 2]
                elapsed = rest[len(TRAINABLE) + 4]
                print(f"[{done:4d}/{N_TRIALS}] trial={i:4d} slope={slope:.2f} "
                      f"temp={temp:.2f} beta={beta_:.1f} lr={lr:.1e} "
                      f"d={d_init:.2f} -> loss={min_loss:.3f} gamma={gamma:.2f} "
                      f"({elapsed:.1f}s)")

    print(f"\nDone. Results: {OUT_CSV}  (total wall time: {time.time() - t_start:.1f}s)")