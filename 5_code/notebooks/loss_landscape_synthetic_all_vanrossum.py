import copy
import logging
import time
from itertools import combinations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import config

from ADoptEX.core import simulate_jaxley
from ADoptEX.core.data import TraceData, crop_to_stim_window, detect_spikes, load_trace
from ADoptEX.core.parameters import DEFAULT_PARAMS, NAUD_PARAMETERS, PARAM_BOUNDS
from ADoptEX.core.simulation import simulate_with_current_trace
from ADoptEX.evaluation.coincidence import coincidence_factor
from ADoptEX.loss import (
    GuarinoFeatures,
    GuarinoLossConfig,
    VanRossumLossConfig,
    extract_experimental_features,
    guarino_loss,
    make_guarino_loss_fn,
    make_van_rossum_loss_fn,
    spike_train_from_voltage,
)
from ADoptEX.plotting import (
    fit_before_after_plot,
    fit_comparison_plot,
    parameter_comparison_plot,
    spike_timing_plot,
    trace_plot,
    trace_stim_window_plot,
    training_comparison_plot,
    training_history_plot,
)
from ADoptEX.training import TrainingResult
from ADoptEX.training.trainer import TrainingConfig, setup_trainable_cell, train

config.update("jax_platform_name", "cpu")
logging.basicConfig(level=logging.INFO, format="%(message)s")

import matplotlib.pyplot as plt
from matplotlib import cm


initial_params = dict(NAUD_PARAMETERS["adaptation"])
initial_params["v_threshold"] = 0.0
print(initial_params)

current_pA = initial_params["I"]

t_max_ms = 100
dt_ms = 0.025
T = np.linspace(0, t_max_ms, int(t_max_ms / dt_ms))

sim_init = simulate_jaxley(
    params=initial_params,
    stim_current_pA=current_pA,
    stim_duration_ms=100,
    dt_ms=dt_ms,
    t_max_ms=t_max_ms,
    stim_delay_ms=0.0,
    use_surrogate=False,
)

T = np.append(T, T[-1] + dt_ms)
print(len(T))

from ADoptEX.loss import (
    extract_experimental_features,
    GuarinoLossConfig,
    make_guarino_loss_fn,
)

training_config = TrainingConfig(
    optimizer="adam",
    learning_rate=0.01,
    n_epochs=1,
    surrogate_type="sigmoid",
    surrogate_slope=5.0,
)

trace_stim_window_plot(TraceData(
    T,
    sim_init.voltage,
    np.full(len(T), initial_params["I"]),
    dt_ms,
    sim_init.spike_times,
    0,
    len(T),
    initial_params["I"]))

cell, data_stimuli, t_max, trainable_params = setup_trainable_cell(
    initial_params=initial_params,
    current_trace_pA=np.full(len(T), initial_params["I"]),
    dt_ms=dt_ms,
    config=training_config,
)

target_voltage = jnp.array(sim_init.voltage)
stim_end_index = t_max_ms  # already 0-based in cropped trace

# loss_name = 'MSE'
# loss_fn = make_mse_loss_fn(
#     cell=cell,
#     data_stimuli=data_stimuli,
#     t_max=t_max,
#     dt_ms=data.dt_ms,
#     exp_voltage=target_voltage,
#     stim_end_index=data.stim_end_idx,
#     loss_config=MSELossConfig(normalize=False, clamp_threshold=None),
# )


# Extract target features from experimental trace (hard spike detection)
exp_features = extract_experimental_features(
    voltage_trace=target_voltage,
    dt_ms=dt_ms,
    stim_duration_ms=stim_end_index,
    stim_end_index=len(sim_init.voltage),
    spike_threshold_mv=-35
)

loss_name = "VanRossum"
loss_fn = make_guarino_loss_fn(
    cell=cell,
    data_stimuli=data_stimuli,
    t_max=t_max,
    dt_ms=dt_ms,
    exp_features=exp_features,
    stim_duration_ms=t_max_ms,
    stim_end_index=len(sim_init.voltage),
    loss_config=GuarinoLossConfig(
        weight_spike_count=0.0,
    ),
)


# Build param name -> index mapping
param_index = {}
for i, p in enumerate(trainable_params):
    for k in p:
        param_index[k.replace("AdEx_", "")] = i

PARAM_NAMES = list(param_index.keys())
N_PARAMS = len(PARAM_NAMES)

loss_init = loss_fn(trainable_params)
print(f"Initial loss: {float(loss_init):.4f}")
print(f"Trainable parameters ({N_PARAMS}): {PARAM_NAMES}")
print(f"Initial parameters: {initial_params}")


GRID_SIZE = 20  # 20x20 per pair; 21 pairs * 400 = 8400 simulations

pairs = list(combinations(range(N_PARAMS), 2))
print(f"{len(pairs)} parameter pairs, {GRID_SIZE}x{GRID_SIZE} grid each")
print(f"Total simulations: {len(pairs) * GRID_SIZE**2}")

# Storage for all grids
grids = {}  # (i, j) -> {"X": ..., "Y": ..., "Z": ...}

t_start = time.time()

for pair_idx, (pi, pj) in enumerate(pairs):
    px, py = PARAM_NAMES[pi], PARAM_NAMES[pj]
    bx, by = PARAM_BOUNDS[px], PARAM_BOUNDS[py]

    x_vals = np.linspace(bx.min, bx.max, GRID_SIZE)
    y_vals = np.linspace(by.min, by.max, GRID_SIZE)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.full((GRID_SIZE, GRID_SIZE), np.nan)

    ix, iy = param_index[px], param_index[py]

    pxrc = f"AdEx_{px}" if px != "capacitance" else px
    pyrc = f"AdEx_{py}" if py != "capacitance" else py
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            params_rc = copy.deepcopy(trainable_params)
            params_rc[ix] = {pxrc: jnp.array([X[r, c]])}
            params_rc[iy] = {pyrc: jnp.array([Y[r, c]])}
            try:
                Z[r, c] = float(loss_fn(params_rc))
            except Exception:
                Z[r, c] = np.nan

    grids[(pi, pj)] = {"X": X, "Y": Y, "Z": Z}

    elapsed = time.time() - t_start
    eta = elapsed / (pair_idx + 1) * (len(pairs) - pair_idx - 1)
    print(
        f"  [{pair_idx+1:2d}/{len(pairs)}] {px:>8s} vs {py:<8s}  "
        f"loss=[{np.nanmin(Z):.1f}, {np.nanmax(Z):.1f}]  "
        f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
    )

print(f"\nTotal time: {time.time() - t_start:.0f}s")

fig, axes = plt.subplots(
    N_PARAMS,
    N_PARAMS,
    figsize=(3 * N_PARAMS, 3 * N_PARAMS),
    squeeze=False,
)

# Hide all axes first
for ax_row in axes:
    for ax in ax_row:
        ax.set_visible(False)

# Fill lower triangle
for (pi, pj), g in grids.items():
    ax = axes[pj, pi]  # row=pj (y-axis param), col=pi (x-axis param)
    ax.set_visible(True)

    Z_plot = np.clip(g["Z"], np.nanmin(g["Z"]), np.nanpercentile(g["Z"], 95))
    ax.contourf(g["X"], g["Y"], Z_plot, levels=20, cmap=cm.jet)
    ax.contour(
        g["X"], g["Y"], Z_plot, levels=8, colors="white", linewidths=0.3, alpha=0.5
    )

    def map_name_to_trainable_params(s):
        if s == "capacitance":
            return "C_m"
        return s
    # Mark default params
    ax.plot(
        initial_params[map_name_to_trainable_params(PARAM_NAMES[pi])],
        initial_params[map_name_to_trainable_params(PARAM_NAMES[pj])],
        "k+",
        markersize=8,
        markeredgewidth=2,
    )

    # Labels on edges only
    if pj == N_PARAMS - 1:  # bottom row
        ax.set_xlabel(PARAM_NAMES[pi], fontsize=10)
    else:
        ax.set_xticklabels([])
    if pi == 0:  # left column
        ax.set_ylabel(PARAM_NAMES[pj], fontsize=10)
    else:
        ax.set_yticklabels([])

    ax.tick_params(labelsize=7)

# Diagonal labels
for i in range(N_PARAMS):
    ax = axes[i, i]
    ax.set_visible(True)
    ax.text(
        0.5,
        0.5,
        PARAM_NAMES[i],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle(f"{loss_name} Loss Landscape — All Parameter Pairs", fontsize=18, y=1.01)
plt.tight_layout()
plt.savefig(
    f"{loss_name.lower()}_loss_landscape_grid.pdf", dpi=150, bbox_inches="tight"
)
plt.show()
print(f"Saved to {loss_name.lower()}_loss_landscape_grid.pdf")

ax = plt.subplot(projection="3d")

surf = ax.plot_surface(
    X,
    Y,
    Z_plot,
    cmap=cm.jet,
    edgecolor="none",
    alpha=0.9,
    rstride=1,
    cstride=1,
)

# Contour projection on the floor
z_floor = ax.get_zlim()[0]
ax.contour(
    X,
    Y,
    Z_plot,
    levels=10,
    zdir="z",
    offset=z_floor,
    cmap=cm.jet,
    alpha=0.6,
)

ax.set_xlabel(px, fontsize=8, labelpad=2)
ax.set_ylabel(py, fontsize=8, labelpad=2)
ax.set_zlabel("Loss", fontsize=8, labelpad=2)
ax.set_title(f"{px} vs {py}", fontsize=10, pad=2)
ax.tick_params(labelsize=6)
ax.view_init(elev=30, azim=-60)

plt.tight_layout()
# plt.savefig(
#     f"{loss_name.lower()}_loss_landscape_3d_focus.pdf", dpi=150, bbox_inches="tight"
# )
plt.show()
# print(f"Saved to {loss_name.lower()}_loss_landscape_3d_focus.png")
