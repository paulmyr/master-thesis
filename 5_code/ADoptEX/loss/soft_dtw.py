"""
Soft-DTW Loss for Differentiable AdEx Optimization

Soft-DTW (Cuturi & Blondel, 2017) is a differentiable relaxation of Dynamic Time
Warping: the hard ``min`` in the DTW recursion is replaced by a soft-min
(``-gamma * logsumexp(-./gamma)``), so the alignment cost is smooth in both
sequences and compatible with JAX autodiff. Unlike voltage MSE it tolerates small
temporal misalignments (it aligns rather than subtracts point-by-point), which is
exactly what spike-timing comparison needs.

Pipeline applied here:
  1. sliding-window max reduction of both traces (collapses spike peaks into a
     coarse envelope and shortens the sequence so the O(n*m) recursion is cheap),
  2. unit rescale to [0, 1] using a shared min/max,
  3. soft-DTW on a cost matrix combining an L1 amplitude term and a normalized
     temporal-penalty term, both in [0, 1] so ``lambda_temp`` directly trades
     timing against amplitude.

With ``use_divergence=True`` the loss returns the Cuturi & Blondel divergence
``D(x, y) = sdtw(x, y) - 0.5 * (sdtw(x, x) + sdtw(y, y))`` which is >= 0 and = 0
when x = y, removing soft-DTW's length-dependent bias.

Reference:
Cuturi M, Blondel M (2017) Soft-DTW: a Differentiable Loss Function for
Time-Series. ICML 2017.
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from . import inject_spike_peaks

_LARGE = 1e9  # stand-in for +inf; softmin treats -_LARGE/gamma as negligible


@dataclass
class SoftDTWLossConfig:
    """Configuration for the soft-DTW voltage-trace loss.

    Attributes:
        window_size: Sliding-window length (samples) for the max reduction. Larger
                     windows give a coarser envelope and a shorter sequence. (default: 50)
        stride: Sliding-window stride (samples). (default: 30)
        gamma: Soft-min temperature in the DTW recursion. Smaller -> closer to hard
               DTW (sharper alignment); larger -> smoother. (default: 0.5)
        lambda_temp: Weight of the normalized temporal-penalty term relative to the
                     [0, 1] amplitude term; trades timing vs amplitude. (default: 10.0)
        spike_peak_mv: Peak voltage injected at spike timesteps before the reduction
                       so the envelope sees the same peak height the target was built
                       with. Set None to disable injection. (default: 35.0)
        use_divergence: If True return the Cuturi & Blondel divergence (>= 0, = 0 at a
                        perfect fit); if False return the raw soft-DTW value. (default: True)
    """

    window_size: int = 50
    stride: int = 30
    gamma: float = 0.5
    lambda_temp: float = 10.0
    spike_peak_mv: Optional[float] = 35.0
    use_divergence: bool = True


def sliding_window_max(v: Array, window_size: int, stride: int) -> Array:
    """Sliding-window max reduction via jax.lax.reduce_window (differentiable)."""
    return jax.lax.reduce_window(
        v,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=(window_size,),
        window_strides=(stride,),
        padding="VALID",
    )


def rescale_unit(v: Array, v_min: float, v_max: float, eps: float = 1e-8) -> Array:
    """Rescale to [0, 1] using pre-computed min/max (shared across sim and target)."""
    return (v - v_min) / (v_max - v_min + eps)


def _softmin(values: Array, gamma: float) -> Array:
    return -gamma * logsumexp(-values / gamma)


def soft_dtw_from_cost(C: Array, gamma: float = 1.0) -> Array:
    """Soft-DTW distance from a pre-built cost matrix C of shape (n, m).

    R[0, 0] = 0; R[0, j>0] = R[i>0, 0] = +inf (approximated by _LARGE).
    R[i, j] = C[i-1, j-1] + softmin_gamma(R[i-1, j-1], R[i-1, j], R[i, j-1]).
    """
    n, m = C.shape

    def process_row(R_prev, C_row):
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


def make_soft_dtw_loss_fn(
    cell,
    data_stimuli,
    t_max: float,
    dt_ms: float,
    exp_voltage: Array,
    stim_end_index: int,
    loss_config: Optional[SoftDTWLossConfig] = None,
):
    """Create a soft-DTW voltage-trace loss function for Jaxley training.

    Returns a ``params -> scalar`` closure compatible with ``jax.value_and_grad``.
    The closure runs ``jx.integrate``, injects spike peaks into the simulated
    voltage (so the sliding-window max sees the target's peak height), reduces and
    rescales both traces, builds the amplitude+temporal cost matrix, and returns
    the soft-DTW distance (or the Cuturi & Blondel divergence).

    Cost: ``C[i, j] = |x_i - y_j| + lambda_temp * |i - j| / (max(n, m) - 1)``; both
    terms in [0, 1].

    Args:
        cell: Jaxley Cell with trainable AdExSurrogate parameters.
        data_stimuli: Data stimuli tuple from ``cell.data_stimulate()``.
        t_max: Maximum simulation time (ms).
        dt_ms: Time step (ms).
        exp_voltage: Target voltage trace from experimental data.
        stim_end_index: Index of stimulus end (kept for API consistency).
        loss_config: Loss configuration (uses defaults if None).

    Returns:
        Loss function: params -> scalar loss.
    """
    import jaxley as jx

    if loss_config is None:
        loss_config = SoftDTWLossConfig()

    window_size = loss_config.window_size
    stride = loss_config.stride
    gamma = loss_config.gamma
    lambda_temp = loss_config.lambda_temp
    spike_peak_mv = loss_config.spike_peak_mv
    use_divergence = loss_config.use_divergence

    exp_v = jnp.asarray(exp_voltage)
    exp_reduced = sliding_window_max(exp_v, window_size, stride)
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
            cell, params=params, data_stimuli=data_stimuli,
            delta_t=dt_ms, t_max=t_max,
        )
        voltage = results[0].flatten()
        min_len = min(len(voltage), len(exp_v))
        sim_v = voltage[:min_len]

        # Inject peaks at spike timesteps so the sliding-window max sees the same
        # peak height the target trace was built with (results[2] is the 0/1 spike
        # indicator from AdExSurrogate's forward pass).
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

        i_only = jnp.arange(n, dtype=exp_scaled.dtype)
        C_xx = jnp.abs(sim_scaled[:, None] - sim_scaled[None, :]) + lambda_temp * (
            jnp.abs(i_only[:, None] - i_only[None, :]) / denom
        )
        sdtw_xx = soft_dtw_from_cost(C_xx, gamma=gamma)

        return sdtw_xy - 0.5 * (sdtw_xx + sdtw_yy)

    return loss_fn
