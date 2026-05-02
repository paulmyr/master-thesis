"""Random hyperparameter sweep under the warm-start protocol.

Replaces the cold-start sweeps (`hparam_sweep.py`,
`hparam_sweep_van_rossum.py`) which were tuned for the old random-init regime
and don't transfer to the warm-start benchmark in `benchmark.py`.

Per trial:
  1. Sample loss_type ∈ {guarino, van_rossum}; MSE skipped (design-failure).
  2. Sample shared HPs: surrogate_type, surrogate_slope, optimizer,
     learning_rate, use_param_transform, grad_clip_norm.
  3. Sample optimizer-conditional HPs (polyak_alpha/beta if polyak).
  4. Sample loss-conditional HPs (Guarino: temperature, beta;
     Van Rossum: tau_ms, weight_subthreshold).
  5. Build cell + jitted loss; warm-start with K_SWEEP=20 random candidates.
  6. Rebuild cell at warm-start params; train 25 epochs.
  7. Record warmstart_loss, train_best_loss, train_final_loss, gamma.

Outputs `results_hparam_sweep_warmstart.csv`. Tuned HPs feed into
`benchmark.py`'s METHOD_CONFIGS for the headline benchmark run.

Run with `python hparam_sweep_warmstart.py [--smoke]` — `--smoke` runs 5
trials with verbose output for sanity-checking before the overnight run.
"""

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from jax import config

config.update("jax_platform_name", "cpu")

import benchmark_warmstart as benchmark  # for SCENARIOS, simulate_ground_truth, run_gradient, evaluate

# --- Sweep constants ---

N_TRIALS_FULL = 2000
N_TRIALS_SMOKE = 5
K_SWEEP = 100         # smaller than benchmark's K_WARMSTART=100, for per-trial speed
N_GRAD_STEPS_SWEEP = 25  # matches benchmark's N_GRAD_STEPS
SCENARIO_NAME = "tonic"
SEED = 42
OUT_CSV = Path(__file__).parent / "results_hparam_sweep_warmstart.csv"

LOSS_TYPES = ["guarino", "van_rossum"]
SURROGATE_TYPES = ["superspike"]
OPTIMIZERS = ["polyak"]


# --- HP sampling ---

def sample_hp(rng):
    """Draw one trial's HPs. Returns (method_cfg, sampled_hp_record).

    method_cfg is what `run_gradient` consumes. sampled_hp_record is a flat
    dict of every value drawn (with NaN for non-applicable conditional HPs)
    for CSV logging.
    """
    loss_type = LOSS_TYPES[rng.integers(len(LOSS_TYPES))]
    surrogate_type = SURROGATE_TYPES[rng.integers(len(SURROGATE_TYPES))]
    surrogate_slope = float(10 ** rng.uniform(np.log10(0.5), np.log10(20.0)))
    optimizer = OPTIMIZERS[rng.integers(len(OPTIMIZERS))]
    lr = float(10 ** rng.uniform(-5.0, -1.0))
    use_param_transform = bool(rng.integers(2))
    # grad_clip_norm is incompatible with polyak (its normalisation replaces
    # clipping); only sample for adam.
    if optimizer == "polyak":
        grad_clip_norm = None
    else:
        grad_clip_norm = (
            None if rng.uniform() < 0.5
            else float(10 ** rng.uniform(-1.0, 2.0))
        )

    cfg = {
        "loss_type": loss_type,
        "surrogate_type": surrogate_type,
        "surrogate_slope": surrogate_slope,
        "optimizer": optimizer,
        "learning_rate": lr,
        "use_param_transform": use_param_transform,
        "grad_clip_norm": grad_clip_norm,
    }

    record = dict(cfg)
    record["polyak_alpha"] = float("nan")
    record["polyak_beta"] = float("nan")
    record["temperature"] = float("nan")
    record["feature_beta"] = float("nan")
    record["validity_beta"] = float("nan")
    record["weight_spike_count"] = float("nan")
    record["tau_ms"] = float("nan")
    record["weight_subthreshold"] = float("nan")

    if optimizer == "polyak":
        polyak_alpha = float(rng.uniform(0.0, 1.5))
        polyak_beta = float(rng.uniform(0.0, 1.0))
        cfg["polyak_alpha"] = polyak_alpha
        cfg["polyak_beta"] = polyak_beta
        record["polyak_alpha"] = polyak_alpha
        record["polyak_beta"] = polyak_beta

    if loss_type == "guarino":
        temperature = float(rng.uniform(0.05, 0.5))
        feature_beta = float(rng.uniform(1.0, 20.0))
        validity_beta = float(rng.uniform(1.0, 20.0))

        weight_spike_count = float(rng.uniform(0, 0.3))
        # weight_spike_count = (
        #     0.0 if rng.uniform() < 0.5
        #     else float(rng.uniform(0, 0.2))
        # )

        cfg["temperature"] = temperature
        cfg["beta"] = feature_beta
        cfg["validity_beta"] = validity_beta
        cfg["weight_spike_count"] = weight_spike_count
        record["temperature"] = temperature
        record["feature_beta"] = feature_beta
        record["validity_beta"] = validity_beta
        record["weight_spike_count"] = weight_spike_count
    elif loss_type == "van_rossum":
        tau_ms = float(10 ** rng.uniform(0.0, np.log10(50.0)))
        weight_subthreshold = float(rng.uniform(0.0, 1.0))
        cfg["loss_kwargs"] = {
            "tau_ms": tau_ms,
            "weight_subthreshold": weight_subthreshold,
        }
        record["tau_ms"] = tau_ms
        record["weight_subthreshold"] = weight_subthreshold

    return cfg, record


# --- Scaffold init (warm-start replaces it; just need something valid) ---

def make_scaffold(rng):
    from ADoptEX.core.parameters import PARAM_BOUNDS
    return {n: float(rng.uniform(PARAM_BOUNDS[n].min, PARAM_BOUNDS[n].max))
            for n in benchmark.TRAINABLE}


# --- CSV ---

CSV_FIELDS = [
    "trial",
    "loss_type", "surrogate_type", "surrogate_slope",
    "optimizer", "learning_rate", "use_param_transform", "grad_clip_norm",
    "polyak_alpha", "polyak_beta",
    "temperature", "feature_beta", "validity_beta", "weight_spike_count",
    "tau_ms", "weight_subthreshold",
    "warmstart_loss", "train_best_loss", "train_final_loss",
    "kept_source", "kept_loss",
    "improvement",  # warmstart_loss - train_best_loss; positive = useful
    "gamma", "n_spikes_sim", "n_spikes_gt",
    "n_iter", "elapsed_s",
]


def smoke_print(trial_idx, record, result, gamma, elapsed):
    color = '\033[92m' if (gamma is not None and gamma > 0.5) else '\033[91m'
    end = '\033[0m'
    grad_clip = record["grad_clip_norm"]
    grad_clip_str = f"{grad_clip:.2g}" if grad_clip is not None else "None"
    extra = ""
    if record["loss_type"] == "guarino":
        extra = f" T={record['temperature']:.2f} fβ={record['feature_beta']:.1f}"
    elif record["loss_type"] == "van_rossum":
        extra = f" τ={record['tau_ms']:.1f} wsub={record['weight_subthreshold']:.2f}"
    if record["optimizer"] == "polyak":
        extra += f" α={record['polyak_alpha']:.2f} β={record['polyak_beta']:.2f}"
    ws = result["warmstart_loss"]
    tb = result["train_best_loss"]
    print(
        f"{color}[{trial_idx:4d}] {record['loss_type']:<11} "
        f"{record['surrogate_type']:<11} slope={record['surrogate_slope']:.2f} "
        f"opt={record['optimizer']:<6} lr={record['learning_rate']:.1e} "
        f"tx={int(record['use_param_transform'])} clip={grad_clip_str}{extra} "
        f"-> ws={ws:.3g} tb={tb:.3g} src={result['kept_source']:<9} "
        f"γ={gamma:.3f} ({elapsed:.1f}s){end}"
    )


# --- Main ---

def run_sweep(n_trials, smoke=False):
    rng = np.random.default_rng(SEED)
    scenario_params = benchmark.SCENARIOS[SCENARIO_NAME]
    print(f"GT simulating ({SCENARIO_NAME})...")
    gt = benchmark.simulate_ground_truth(scenario_params)
    n_spikes_gt = len(gt["spike_times"])
    print(f"GT done. n_spikes_gt={n_spikes_gt}")

    write_header = not OUT_CSV.exists() or smoke
    mode = "w" if smoke else "a"
    with open(OUT_CSV if not smoke else OUT_CSV.with_suffix(".smoke.csv"),
              mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for trial_idx in range(n_trials):
            t0 = time.time()
            cfg, record = sample_hp(rng)
            scaffold = make_scaffold(rng)
            warmstart_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
            row: dict[str, Any] = {f: float("nan") for f in CSV_FIELDS}
            row.update(record)
            row["trial"] = trial_idx

            try:
                result = benchmark.run_gradient(
                    scenario_params, scaffold, cfg, gt, warmstart_rng,
                    k_warmstart=K_SWEEP, n_grad_steps=N_GRAD_STEPS_SWEEP,
                )
                gamma, n_sim = benchmark.evaluate(
                    result["final_params"], gt["spike_times"], scenario_params
                )
                elapsed = time.time() - t0

                row.update({
                    "warmstart_loss": result["warmstart_loss"],
                    "train_best_loss": result["train_best_loss"],
                    "train_final_loss": result["train_final_loss"],
                    "kept_source": result["kept_source"],
                    "kept_loss": result["final_loss"],
                    "improvement": (result["warmstart_loss"]
                                    - result["train_best_loss"])
                        if math.isfinite(result["train_best_loss"]) else float("nan"),
                    "gamma": gamma,
                    "n_spikes_sim": n_sim,
                    "n_spikes_gt": n_spikes_gt,
                    "n_iter": result["n_iter"],
                    "elapsed_s": elapsed,
                })
                writer.writerow(row)
                f.flush()
                if smoke:
                    smoke_print(trial_idx, record, result, gamma, elapsed)
                else:
                    if (trial_idx + 1) % 10 == 0:
                        print(f"[{trial_idx + 1:4d}/{n_trials}] "
                              f"ws={result['warmstart_loss']:.3g} "
                              f"src={result['kept_source']} γ={gamma:.3f}")
            except Exception as e:
                row["elapsed_s"] = time.time() - t0
                row["kept_source"] = "ERROR"
                writer.writerow(row)
                f.flush()
                print(f"[{trial_idx:4d}] ERROR ({record['loss_type']}, "
                      f"{record['surrogate_type']}, {record['optimizer']}): {e}")

    print(f"\nDone. Results: {OUT_CSV if not smoke else OUT_CSV.with_suffix('.smoke.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help=f"Run {N_TRIALS_SMOKE} trials with verbose output")
    args = parser.parse_args()
    n = N_TRIALS_SMOKE if args.smoke else N_TRIALS_FULL
    print(f"Running {'smoke' if args.smoke else 'full'} sweep: {n} trials, "
          f"K_WARMSTART={K_SWEEP}, n_grad_steps={N_GRAD_STEPS_SWEEP}")
    run_sweep(n, smoke=args.smoke)
