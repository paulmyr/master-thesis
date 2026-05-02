"""HP sensitivity benchmark (Exp B: OFAT around the anchor config).

For each gradient method, anchor at its METHOD_CONFIGS entry (defined here,
edit when a better sweep result lands), then sweep one HP at a time across
a 7-point grid. At each (method, hp_name, hp_value), run N_DIRS perturbations
at fixed δ to measure how much that single HP moves Γ.

Headline plots (analysis notebook): per-HP violin/strip of Γ vs HP value;
tornado of HP-induced Γ range, sorted, per method — identifies brittle HPs.

Run with `python benchmark_hp_sensitivity.py`. ~1120 runs (~40 min on CPU).
Smoke: trim HP_GRIDS values and N_DIRS at the top.
"""

import contextlib
import csv
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
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
N_GRAD_STEPS = 25


# --- Experiment constants --------------------------------------------------

SCENARIO_PARAMS = dict(NAUD_PARAMETERS["tonic"])
DELTA = 0.2
N_DIRS = 10
SEED = 42
OUT_CSV = Path(__file__).parent / "results_hp_sensitivity.csv"

# Anchor configs. Same notebook-validated HPs as benchmark_recovery.py.
METHOD_CONFIGS: dict[str, dict] = {
    "grad_guarino": {
        "loss_type": "guarino",
        "surrogate_type": "superspike",
        "surrogate_slope": 10.0,
        "optimizer": "polyak",
        "learning_rate": 0.01,
        "use_param_transform": True,
        "polyak_alpha": 1.0,
        "polyak_beta": 0.8,
        "temperature": 0.3,
        "beta": 10.0,
        "validity_beta": 10.0,
        "weight_spike_count": 0.0,
    },
    "grad_vanrossum": {
        "loss_type": "van_rossum",
        "surrogate_type": "superspike",
        "surrogate_slope": 10.0,
        "optimizer": "polyak",
        "learning_rate": 0.01,
        "use_param_transform": True,
        "polyak_alpha": 1.0,
        "polyak_beta": 0.8,
        "loss_kwargs": {"tau_ms": 10.0, "weight_subthreshold": 0.3},
    },
}

# HPs swept per method. Log-spaced for scale parameters; linear elsewhere.
HP_GRIDS = {
    "grad_guarino": {
        "learning_rate":      np.logspace(-5, -1, 7).tolist(),
        "surrogate_slope":    np.logspace(np.log10(0.5), np.log10(20.0), 7).tolist(),
        "polyak_alpha":       np.linspace(0.0, 1.5, 7).tolist(),
        "polyak_beta":        np.linspace(0.0, 1.0, 7).tolist(),
        "temperature":        np.linspace(0.05, 0.5, 7).tolist(),
        "beta":               np.linspace(1.0, 20.0, 7).tolist(),
        "validity_beta":      np.linspace(1.0, 20.0, 7).tolist(),
        "weight_spike_count": np.linspace(0.0, 0.2, 7).tolist(),
    },
    "grad_vanrossum": {
        "learning_rate":       np.logspace(-5, -1, 7).tolist(),
        "surrogate_slope":     np.logspace(np.log10(0.5), np.log10(20.0), 7).tolist(),
        "polyak_alpha":        np.linspace(0.0, 1.5, 7).tolist(),
        "polyak_beta":         np.linspace(0.0, 1.0, 7).tolist(),
        # van-Rossum-specific HPs are nested under loss_kwargs.
        "tau_ms":              np.logspace(0.0, np.log10(50.0), 7).tolist(),
        "weight_subthreshold": np.linspace(0.0, 1.0, 7).tolist(),
    },
}

CSV_FIELDS = [
    "method", "hp_name", "hp_value", "direction_idx",
    "gamma", "n_spikes_sim", "n_spikes_gt",
    "init_loss", "final_loss", "kept_source", "n_iter", "elapsed_s",
]


# --- Forward-sim & loss helpers (duplicated from benchmark_recovery.py to
# keep this script standalone — the user explicitly chose self-contained over
# DRY). If these get out of sync, that's a real divergence between the two
# experiments; treat it as a code smell, not a refactor target. ------------

@contextlib.contextmanager
def _silence():
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def _current_trace(I_pA: float) -> np.ndarray:
    current = np.zeros(N_TIMESTEPS)
    current[STIM_START_IDX:STIM_END_IDX] = I_pA
    return current


def simulate_gt(scenario_params: dict) -> dict:
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
            exp_voltage=gt["voltage"],
            loss_config=VanRossumLossConfig(**method_cfg.get("loss_kwargs", {})),
        )
    raise ValueError(f"unknown loss_type: {lt}")


# --- Perturbation sampling -------------------------------------------------

def sample_directions(rng: np.random.Generator, n_dirs: int, n_params: int
                      ) -> np.ndarray:
    raw = rng.standard_normal((n_dirs, n_params))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw


def perturb(theta_gt: dict, direction: np.ndarray, delta: float
            ) -> tuple[dict, int]:
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


# --- Optimisation driver (gradient only — Exp B is HP sensitivity, not the
# baseline comparison) ------------------------------------------------------

def _keep_better(init_params, init_loss, trained_params, trained_best_loss):
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


# --- Main loop -------------------------------------------------------------

def _override_hp(anchor_cfg: dict, hp_name: str, hp_value: float) -> dict:
    """Return a copy of anchor_cfg with `hp_name` overridden. Van-Rossum's
    tau_ms / weight_subthreshold live inside loss_kwargs; everything else is
    a top-level cfg key."""
    cfg = deepcopy(anchor_cfg)
    if hp_name in ("tau_ms", "weight_subthreshold"):
        cfg.setdefault("loss_kwargs", {})
        cfg["loss_kwargs"][hp_name] = hp_value
    else:
        cfg[hp_name] = hp_value
    return cfg


def run():
    rng = np.random.default_rng(SEED)
    n_params = len(TRAINABLE)

    print(f"GT simulating ({SCENARIO_PARAMS['I']} pA tonic stim)...")
    gt = simulate_gt(SCENARIO_PARAMS)
    n_spikes_gt = len(gt["spike_times"])
    theta_gt = {n: float(SCENARIO_PARAMS[n]) for n in TRAINABLE}
    directions = sample_directions(rng, N_DIRS, n_params)
    # Pre-compute the perturbed inits so all (method, hp) combos see the same
    # set of init points — sensitivity is then about the HP, not the inits.
    init_set = [perturb(theta_gt, directions[i], DELTA) for i in range(N_DIRS)]
    print(f"  done. n_spikes_gt={n_spikes_gt}")

    write_header = not OUT_CSV.exists()
    total = sum(len(grid) * N_DIRS
                for grids in HP_GRIDS.values()
                for grid in grids.values())
    run_idx = 0
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for method, hp_grid in HP_GRIDS.items():
            anchor_cfg = METHOD_CONFIGS[method]
            for hp_name, values in hp_grid.items():
                for hp_value in values:
                    cfg = _override_hp(anchor_cfg, hp_name, hp_value)
                    for dir_idx, (init_params, _n_clipped) in enumerate(init_set):
                        run_idx += 1
                        t0 = time.time()
                        row: dict[str, Any] = {f: float("nan") for f in CSV_FIELDS}
                        row.update({
                            "method": method, "hp_name": hp_name,
                            "hp_value": float(hp_value),
                            "direction_idx": dir_idx,
                            "n_spikes_gt": n_spikes_gt,
                        })
                        try:
                            result = run_grad(SCENARIO_PARAMS, init_params,
                                              cfg, gt)
                            gamma, n_sim = evaluate_gamma(
                                result["final_params"], gt["spike_times"],
                                SCENARIO_PARAMS,
                            )
                            row.update({
                                "gamma": gamma, "n_spikes_sim": n_sim,
                                "init_loss": result["init_loss"],
                                "final_loss": result["final_loss"],
                                "kept_source": result["kept_source"],
                                "n_iter": result["n_iter"],
                            })
                        except Exception as e:
                            row["kept_source"] = "ERROR"
                            print(f"  ERROR ({method}/{hp_name}={hp_value}/"
                                  f"dir={dir_idx}): {e}")
                        row["elapsed_s"] = time.time() - t0
                        writer.writerow(row)
                        f.flush()
                        if run_idx % 25 == 0 or run_idx == total:
                            print(f"  [{run_idx:4d}/{total}] {method} "
                                  f"{hp_name}={hp_value:.4g} dir={dir_idx} "
                                  f"γ={row['gamma']:.3f} "
                                  f"({row['elapsed_s']:.1f}s)")

    print(f"\nDone. Results: {OUT_CSV}")


if __name__ == "__main__":
    print(f"HP sensitivity benchmark: tonic δ={DELTA} N_DIRS={N_DIRS}")
    run()
