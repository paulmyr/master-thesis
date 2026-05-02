import contextlib
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import scipy.stats

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from jax import config

from ADoptEX.core import TraceData, simulate_jaxley
from ADoptEX.plotting import trace_stim_window_plot

config.update("jax_platform_name", "cpu")

from ADoptEX.core.parameters import NAUD_PARAMETERS, PARAM_BOUNDS
from ADoptEX.core.simulation import simulate_with_current_trace
from ADoptEX.evaluation.coincidence import coincidence_factor
from ADoptEX.loss import (
    extract_experimental_features,
    make_guarino_loss_fn,
    make_van_rossum_loss_fn,
)
from ADoptEX.training.trainer import TrainingConfig, setup_trainable_cell, train

# --- Constants ---

DT_MS = 0.025
T_MAX_MS = 100.0
STIM_DELAY_MS = 5.0
STIM_DURATION_MS = 90.0
SPIKE_PEAK_MV = 35.0

assert STIM_DELAY_MS < T_MAX_MS, "Stim delay has to be smaller than maximum simulation duration"

T = jnp.linspace(0, T_MAX_MS, int(T_MAX_MS / DT_MS) + 1)
N = len(T)

STIM_START_IDX = int(STIM_DELAY_MS / DT_MS)
STIM_END_IDX = int(min(((STIM_DURATION_MS + STIM_DELAY_MS) / DT_MS), len(T))) # first 0 current index

TRAINABLE = ["C_m", "g_L", "E_L", "v_T", "delta_T", "v_reset"]
SEEDS = list(range(10))
METHODS = ["grad_guarino", "grad_vanrossum", "nm_guarino", "nm_vanrossum"]

SCENARIOS = {
    "tonic": {**NAUD_PARAMETERS["tonic"]},
    "adaptation": {**NAUD_PARAMETERS["adaptation"]},
    "initial_bursting": {**NAUD_PARAMETERS["initial_bursting"]},
}

training_config = TrainingConfig(
    optimizer="polyak",
    learning_rate=0.05,
    n_epochs=200,
    surrogate_type="sigmoid",
    surrogate_slope=2,
    polyak_alpha=1.0,
    polyak_beta=0.85,
    clip_to_bounds=False,
    use_param_transform=True,
    return_best=True,
    verbose=False,
)

results_path = Path(__file__).parent / "results_benchmark.csv"

# --- Helpers ---

@contextlib.contextmanager
def _silence():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def generate_initial_params(ground_truth_params, seed):
    pass


def run(ground_truth, initial_params, method):
    pass

def benchmark():

    for scenario in SCENARIOS:
        ground_truth = None # todo...

        for seed in SEEDS:
            initial_params = generate_initial_params(ground_truth, seed)

            for method in METHODS:
                run(ground_truth, initial_params, method)

    pass

def evaluation():
    pass


if __name__ == "__main__":
    benchmark()
    evaluation()
