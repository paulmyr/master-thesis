import copy
import logging
import time
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
from jax import config
from jax.scipy.special import logsumexp

from ADoptEX.core import simulate_jaxley
from ADoptEX.core.data import TraceData
from ADoptEX.core.parameters import DEFAULT_PARAMS, NAUD_PARAMETERS, PARAM_BOUNDS
from ADoptEX.loss import inject_spike_peaks

from ADoptEX.plotting import (
    trace_stim_window_plot,
)

from ADoptEX.training.trainer import TrainingConfig, setup_trainable_cell

config.update("jax_platform_name", "cpu")
logging.basicConfig(level=logging.INFO, format="%(message)s")

import matplotlib.pyplot as plt
from matplotlib import cm


initial_params = dict(NAUD_PARAMETERS["tonic"])
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

SPIKE_PEAK_MV = 35.0
for st in sim_init.spike_times:
    idx = int(st / dt_ms)
    if 0 <= idx < len(sim_init.voltage):
        sim_init.voltage[idx] = SPIKE_PEAK_MV

T = np.append(T, T[-1] + dt_ms)
print(len(T))

# --- Deistler preprocessing ---
def sliding_window_max(v: jnp.ndarray, window_size: int, stride: int) -> jnp.ndarray:
    """Sliding-window max reduction via jax.lax.reduce_window (differentiable)."""
    return jax.lax.reduce_window(
        v,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=(window_size,),
        window_strides=(stride,),
        padding="VALID",
    )


def rescale_unit(v: jnp.ndarray, v_min: float, v_max: float, eps: float = 1e-8) -> jnp.ndarray:
    """Rescale to [0, 1] using pre-computed min/max (shared across sim and target)."""
    return (v - v_min) / (v_max - v_min + eps)


# --- Soft-DTW ---
_LARGE = 1e9  # stand-in for +inf; softmin treats -_LARGE/gamma as negligible


def _softmin(values: jnp.ndarray, gamma: float) -> jnp.ndarray:
    return -gamma * logsumexp(-values / gamma)


def soft_dtw_from_cost(C: jnp.ndarray, gamma: float = 1.0) -> jnp.ndarray:
    """Soft-DTW distance from a pre-built cost matrix C of shape (n, m).

    R[0, 0] = 0; R[0, j>0] = R[i>0, 0] = +inf (approximated by _LARGE).
    R[i, j] = C[i-1, j-1] + softmin_gamma(R[i-1, j-1], R[i-1, j], R[i, j-1]).
    Returns R[n, m].
    """
    n, m = C.shape

    def process_row(R_prev: jnp.ndarray, C_row: jnp.ndarray):
        # R_prev has shape (m+1,); C_row has shape (m,).
        def col_step(R_curr_prev, inputs):
            R_diag, R_up, C_val = inputs
            R_curr = C_val + _softmin(jnp.stack([R_diag, R_up, R_curr_prev]), gamma)
            return R_curr, R_curr

        col_inputs = (R_prev[:-1], R_prev[1:], C_row)
        _, R_curr_rest = jax.lax.scan(col_step, _LARGE, col_inputs)
        R_curr = jnp.concatenate([jnp.array([_LARGE]), R_curr_rest])
        return R_curr, None

    R_init = jnp.concatenate([jnp.zeros(1), jnp.full(m, _LARGE)])
    R_final, _ = jax.lax.scan(process_row, R_init, C)
    return R_final[-1]

# --- Soft-DTW loss factory (Deistler preprocessing + divergence form) ---
def make_soft_dtw_loss_fn(
    cell,
    data_stimuli,
    t_max: float,
    dt_ms: float,
    exp_voltage: jnp.ndarray,
    stim_end_index: int,
    *,
    window_size: int = 50,
    stride: int = 30,
    gamma: float = 0.5,
    lambda_temp: float = 10.0,
    spike_peak_mv: float | None = 35.0,
    use_divergence: bool = True,
):
    """Build a closure `params -> scalar loss` computing soft-DTW on the
    sliding-window-max-reduced, unit-rescaled voltage traces with an L1 +
    (normalized) temporal-penalty cost.

    Cost: C[i, j] = |x_i - y_j| + lambda_temp * |i - j| / (n - 1)
    Both terms live in [0, 1], so lambda_temp directly controls timing vs. amplitude.

    If `use_divergence=True` (default), returns the Cuturi & Blondel soft-DTW
    divergence D(x, y) = sdtw(x, y) - 0.5 * (sdtw(x, x) + sdtw(y, y)), which is
    non-negative and = 0 when x = y. This removes soft-DTW's intrinsic length-
    dependent bias (~gamma * log(3) per cell) that otherwise pushes the loss
    to large negative values, confusingly making a "better" fit look "worse".
    """
    import jaxley as jx

    exp_v = jnp.asarray(exp_voltage)

    # Precompute target preprocessing (static across training).
    exp_reduced = sliding_window_max(exp_v, window_size, stride)
    # Scale using the *reduced* target's min/max so the rescaled target covers [0, 1].
    v_min = float(jnp.min(exp_reduced))
    v_max = float(jnp.max(exp_reduced))
    exp_scaled = rescale_unit(exp_reduced, v_min, v_max)
    m = exp_scaled.shape[0]

    # Precompute sdtw(y, y) — constant baseline used by the divergence form.
    if use_divergence:
        j_idx_y = jnp.arange(m, dtype=exp_scaled.dtype)
        denom_y = jnp.maximum(jnp.asarray(m - 1, dtype=exp_scaled.dtype), 1.0)
        C_yy = jnp.abs(exp_scaled[:, None] - exp_scaled[None, :]) + lambda_temp * (
            jnp.abs(j_idx_y[:, None] - j_idx_y[None, :]) / denom_y
        )
        sdtw_yy = soft_dtw_from_cost(C_yy, gamma=gamma)
    else:
        sdtw_yy = jnp.asarray(0.0, dtype=exp_scaled.dtype)

    def loss_fn(params):
        results = jx.integrate(
            cell,
            params=params,
            data_stimuli=data_stimuli,
            delta_t=dt_ms,
            t_max=t_max,
        )
        voltage = results[0].flatten()
        min_len = min(len(voltage), len(exp_v))
        sim_v = voltage[:min_len]

        # Hard-inject peaks at spike timesteps (results[2] is 0/1 in the forward
        # pass from AdExSurrogate's Heaviside). Mirrors the manual injection on
        # the target trace so the sliding-window max sees the same peak height.
        if spike_peak_mv is not None:
            spikes = results[2].flatten()[:min_len]
            sim_v = inject_spike_peaks(sim_v, spikes, spike_peak_mv)

        sim_reduced = sliding_window_max(sim_v, window_size, stride)
        sim_scaled = rescale_unit(sim_reduced, v_min, v_max)
        n = sim_scaled.shape[0]

        denom = jnp.maximum(jnp.asarray(max(n, m) - 1, dtype=exp_scaled.dtype), 1.0)
        i_idx = jnp.arange(n, dtype=exp_scaled.dtype)[:, None]
        j_idx = jnp.arange(m, dtype=exp_scaled.dtype)[None, :]
        C_xy = jnp.abs(sim_scaled[:, None] - exp_scaled[None, :]) + lambda_temp * (
            jnp.abs(i_idx - j_idx) / denom
        )
        sdtw_xy = soft_dtw_from_cost(C_xy, gamma=gamma)

        if not use_divergence:
            return sdtw_xy

        # sdtw(x, x): full cost (amplitude + temporal), same form as C_yy.
        # Using only the temporal penalty here would make sdtw(x, x) != sdtw(y, y)
        # even when x == y, breaking the divergence identity D(x, x) = 0.
        i_only = jnp.arange(n, dtype=exp_scaled.dtype)
        C_xx = jnp.abs(sim_scaled[:, None] - sim_scaled[None, :]) + lambda_temp * (
            jnp.abs(i_only[:, None] - i_only[None, :]) / denom
        )
        sdtw_xx = soft_dtw_from_cost(C_xx, gamma=gamma)

        return sdtw_xy - 0.5 * (sdtw_xx + sdtw_yy)

    return loss_fn

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

data = TraceData(
    T,
    sim_init.voltage,
    np.full(len(T), initial_params["I"]),
    dt_ms,
    sim_init.spike_times,
    0,
    len(T),
    initial_params["I"])

trace_stim_window_plot(data)

cell, data_stimuli, t_max, trainable_params = setup_trainable_cell(
    initial_params=initial_params,
    current_trace_pA=np.full(len(T), initial_params["I"]),
    dt_ms=dt_ms,
    config=training_config,
)

target_voltage = jnp.array(sim_init.voltage)
stim_end_index = t_max_ms  # already 0-based in cropped trace

loss_name = "Soft-DTW"
loss_fn = make_soft_dtw_loss_fn(
    cell=cell,
    data_stimuli=data_stimuli,
    t_max=t_max,
    dt_ms=dt_ms,
    exp_voltage=target_voltage,
    stim_end_index=len(sim_init.voltage),
    window_size=50,
    stride=30,
    gamma=0.5,
    lambda_temp=10.0,
    spike_peak_mv=SPIKE_PEAK_MV,
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


GRID_SIZE = 10  # 20x20 per pair; 21 pairs * 400 = 8400 simulations

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
