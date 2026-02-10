"""
Soft-DTW + MAE Loss for Differentiable AdEx Optimization

This module implements a combined Soft-DTW (Dynamic Time Warping) and Mean Absolute
Error loss for gradient-based parameter optimization of AdEx neuron models in Jaxley.

Soft-DTW (Cuturi & Blondel, 2017) replaces the hard min in classical DTW with a
differentiable softmin (log-sum-exp), making it fully compatible with JAX autodiff.
This is particularly useful for neuron model fitting because DTW naturally handles
temporal misalignments in spike timing — the core weakness of MSE-based losses.

The MAE component provides a direct amplitude signal to complement DTW's temporal
alignment capability.

Reference:
Cuturi M, Blondel M (2017) Soft-DTW: a Differentiable Loss Function for Time-Series.
Proceedings of the 34th International Conference on Machine Learning (ICML).
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array


@dataclass
class SoftDTWLossConfig:
    """Configuration for Soft-DTW + MAE loss function.

    Attributes:
        gamma_sdtw: Softmin smoothing parameter. Lower values approximate hard DTW
                    more closely; higher values give smoother gradients. (default: 1.0)
        weight_dtw: Weight for the Soft-DTW component of the combined loss. (default: 1.0)
        weight_mae: Weight for the MAE component of the combined loss. (default: 0.5)
        normalize: Whether to length-normalize the DTW distance. (default: True)
        max_length: Maximum trace length before downsampling to manage O(T^2) memory.
                    Set to None to disable downsampling. (default: 2000)
    """

    gamma_sdtw: float = 1.0
    weight_dtw: float = 1.0
    weight_mae: float = 0.5
    normalize: bool = True
    max_length: int | None = 2000


def _softmin(a: Array, b: Array, c: Array, gamma: float) -> Array:
    """Compute softmin of three values using log-sum-exp.

    softmin(a, b, c) = -gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma))

    Args:
        a, b, c: Input values
        gamma: Smoothing parameter (>0)

    Returns:
        Soft minimum (scalar)
    """
    neg_inv_gamma = -1.0 / gamma
    return -gamma * jnp.logaddexp(
        jnp.logaddexp(neg_inv_gamma * a, neg_inv_gamma * b),
        neg_inv_gamma * c,
    )


def _downsample(x: Array, max_length: int) -> Array:
    """Downsample a 1D trace to at most max_length points via linear interpolation."""
    n = len(x)
    if n <= max_length:
        return x
    new_indices = jnp.linspace(0, n - 1, max_length)
    return jnp.interp(new_indices, jnp.arange(n), x)


def soft_dtw(sim_voltage: Array, exp_voltage: Array, gamma: float = 1.0) -> Array:
    """Compute the Soft-DTW distance between two 1D time series.

    Builds an (N x M) cost matrix and fills the alignment matrix R using a
    softmin recurrence implemented with jax.lax.scan for JIT compatibility.

    Args:
        sim_voltage: Simulated voltage trace [N]
        exp_voltage: Experimental (target) voltage trace [M]
        gamma: Softmin smoothing parameter (>0)

    Returns:
        Soft-DTW distance (scalar)
    """
    m = len(exp_voltage)

    # Cost matrix: C[i, j] = |sim[i] - exp[j]|
    cost = jnp.abs(sim_voltage[:, None] - exp_voltage[None, :])  # (N, M)

    # Initialize alignment matrix R with infinity everywhere except R[0, 0] = C[0, 0]
    # We process row-by-row using scan; each row depends on the previous row.
    # R has shape (N, M).

    # Process the first row: R[0, j] = sum(C[0, 0:j+1])
    # (only left-to-left transitions are valid in the first row)
    first_row = jnp.cumsum(cost[0])

    # For remaining rows, we scan row by row.
    # For each new row i, we need to fill R[i, :] using R[i-1, :] (prev_row) and
    # the current cost row cost[i, :].
    #
    # R[i, 0] = R[i-1, 0] + C[i, 0]  (only vertical transition)
    # R[i, j] = C[i, j] + softmin(R[i-1, j-1], R[i-1, j], R[i, j-1])

    def _fill_row(prev_row: Array, cost_row: Array) -> tuple[Array, Array]:
        """Fill one row of the alignment matrix given the previous row."""
        # R[i, 0] = prev_row[0] + cost_row[0]
        r_i_0 = prev_row[0] + cost_row[0]

        # For j >= 1, scan left-to-right within the row
        def _fill_cell(r_left: Array, j: Array) -> tuple[Array, Array]:
            # Three predecessors: diagonal=prev_row[j-1], up=prev_row[j], left=r_left
            diag = prev_row[j - 1]
            up = prev_row[j]
            val = cost_row[j] + _softmin(diag, up, r_left, gamma)
            return val, val

        # Scan over columns 1..M-1
        _, row_rest = jax.lax.scan(_fill_cell, r_i_0, jnp.arange(1, m))

        new_row = jnp.concatenate([r_i_0[None], row_rest])
        return new_row, new_row

    # Scan over rows 1..N-1
    _, all_rows = jax.lax.scan(_fill_row, first_row, cost[1:])

    # Full R matrix rows (for extracting final value)
    # all_rows has shape (N-1, M); we need the last element of the last row
    return all_rows[-1, -1]


def mae_loss(sim_voltage: Array, exp_voltage: Array) -> Array:
    """Compute Mean Absolute Error between two voltage traces.

    Args:
        sim_voltage: Simulated voltage trace [T]
        exp_voltage: Experimental (target) voltage trace [T]

    Returns:
        MAE loss (scalar)
    """
    min_len = min(len(sim_voltage), len(exp_voltage))
    return jnp.mean(jnp.abs(sim_voltage[:min_len] - exp_voltage[:min_len]))


def soft_dtw_mae_loss(
    sim_voltage: Array,
    exp_voltage: Array,
    config: Optional[SoftDTWLossConfig] = None,
) -> Array:
    """Compute combined Soft-DTW + MAE loss.

    Args:
        sim_voltage: Simulated voltage trace [T]
        exp_voltage: Experimental (target) voltage trace [T]
        config: Loss configuration

    Returns:
        Weighted combination of Soft-DTW distance and MAE (scalar)
    """
    if config is None:
        config = SoftDTWLossConfig()

    # Ensure same length
    min_len = min(len(sim_voltage), len(exp_voltage))
    sim_v = sim_voltage[:min_len]
    exp_v = exp_voltage[:min_len]

    # Optionally downsample for DTW (O(T^2) memory)
    if config.max_length is not None and min_len > config.max_length:
        sim_v_dtw = _downsample(sim_v, config.max_length)
        exp_v_dtw = _downsample(exp_v, config.max_length)
    else:
        sim_v_dtw = sim_v
        exp_v_dtw = exp_v

    # Soft-DTW distance
    dtw_dist = soft_dtw(sim_v_dtw, exp_v_dtw, gamma=config.gamma_sdtw)

    # Normalize by alignment path length if requested
    if config.normalize:
        path_len = len(sim_v_dtw) + len(exp_v_dtw)
        dtw_dist = dtw_dist / path_len

    # MAE on full-resolution traces
    mae = mae_loss(sim_v, exp_v)

    return config.weight_dtw * dtw_dist + config.weight_mae * mae


def make_soft_dtw_loss_fn(
    cell,
    data_stimuli,
    t_max: float,
    dt_ms: float,
    exp_voltage: Array,
    stim_end_index: int,
    loss_config: Optional[SoftDTWLossConfig] = None,
):
    """Create a Soft-DTW + MAE loss function for Jaxley training.

    This function returns a loss function closure that can be used with
    jax.value_and_grad for gradient-based optimization.

    Args:
        cell: Jaxley Cell with trainable AdExSurrogate parameters
        data_stimuli: Data stimuli tuple from cell.data_stimulate()
        t_max: Maximum simulation time (ms)
        dt_ms: Time step (ms)
        exp_voltage: Target voltage trace from experimental data
        stim_end_index: Index of stimulus end for voltage extraction
        loss_config: Loss configuration

    Returns:
        Loss function: params -> scalar loss
    """
    import jaxley as jx

    if loss_config is None:
        loss_config = SoftDTWLossConfig()

    def loss_fn(params):
        # Run simulation with current parameters
        results = jx.integrate(
            cell,
            params=params,
            data_stimuli=data_stimuli,
            delta_t=dt_ms,
            t_max=t_max,
        )

        # Extract voltage trace
        voltage = results[0].flatten()

        # Truncate to stimulus window (+ small buffer)
        min_len = min(len(voltage), stim_end_index + 100)
        voltage = voltage[:min_len]

        # Compute combined loss
        loss = soft_dtw_mae_loss(voltage, exp_voltage[:min_len], loss_config)

        return loss

    return loss_fn
