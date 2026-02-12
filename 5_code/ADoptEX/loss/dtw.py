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
from jax.scipy.special import logsumexp


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


# =========================================================================
# Vendored from softdtw_jax by khdlr (https://github.com/khdlr/softdtw_jax)
# MIT License — Copyright (c) 2021 Konrad Heidler
#
# Anti-diagonal scan with custom_vjp for numerically stable gradients.
# Modifications: extracted from class to module-level functions, reformatted,
# simplified _distance_matrix for 1D inputs.
# =========================================================================


def _distance_matrix(a: Array, b: Array) -> Array:
    """Squared Euclidean cost matrix between two 1D sequences."""
    return jnp.square(a[:, None] - b[None, :])


def _pad_inf(inp: Array, before: int, after: int) -> Array:
    """Pad a 1D array with inf values."""
    return jnp.pad(inp, (before, after), constant_values=jnp.inf)


def _make_softmin(gamma: float):
    """Create a softmin function with custom_vjp for gradient stability.

    The custom gradient handles inf values correctly, preventing NaN
    propagation during backprop through the DTW alignment matrix.
    """

    def softmin_raw(array):
        return -gamma * logsumexp(array / -gamma, axis=-1)

    softmin = jax.custom_vjp(softmin_raw)

    def softmin_fwd(array):
        return softmin(array), (array / -gamma,)

    def softmin_bwd(res, g):
        (scaled_array,) = res
        grad = jnp.where(
            jnp.isinf(scaled_array),
            jnp.zeros(scaled_array.shape),
            jax.nn.softmax(scaled_array) * jnp.expand_dims(g, 1),
        )
        return (grad,)

    softmin.defvjp(softmin_fwd, softmin_bwd)
    return softmin


def _soft_dtw_impl(prediction: Array, target: Array, gamma: float) -> Array:
    """Compute Soft-DTW distance using anti-diagonal scan.

    Processes the cost matrix along anti-diagonals using jax.lax.scan,
    which is more parallelizable than row-by-row processing.

    Args:
        prediction: First sequence [N]
        target: Second sequence [M]
        gamma: Softmin smoothing parameter (>0)

    Returns:
        Soft-DTW distance (scalar)
    """
    softmin = _make_softmin(gamma)

    D = _distance_matrix(prediction, target)
    # Ensure H >= W for the anti-diagonal indexing
    if D.shape[0] < D.shape[1]:
        D = D.T
    H, W = D.shape

    # Rearrange cost matrix into anti-diagonals
    rows = []
    for row in range(H):
        rows.append(_pad_inf(D[row], row, H - row - 1))
    model_matrix = jnp.stack(rows, axis=1)

    # First two anti-diagonals as initial carry
    init = (
        _pad_inf(model_matrix[0], 1, 0),
        _pad_inf(model_matrix[1] + model_matrix[0, 0], 1, 0),
    )

    def scan_step(carry, current_antidiagonal):
        two_ago, one_ago = carry

        diagonal = two_ago[:-1]
        right = one_ago[:-1]
        down = one_ago[1:]
        best = softmin(jnp.stack([diagonal, right, down], axis=-1))

        next_row = best + current_antidiagonal
        next_row = _pad_inf(next_row, 1, 0)

        return (one_ago, next_row), next_row

    carry, _ = jax.lax.scan(scan_step, init, model_matrix[2:], unroll=4)
    return carry[1][-1]


# =========================================================================
# End vendored code
# =========================================================================


def _downsample(x: Array, max_length: int) -> Array:
    """Downsample a 1D trace to at most max_length points via linear interpolation."""
    n = len(x)
    if n <= max_length:
        return x
    new_indices = jnp.linspace(0, n - 1, max_length)
    return jnp.interp(new_indices, jnp.arange(n), x)


def soft_dtw(sim_voltage: Array, exp_voltage: Array, gamma: float = 1.0) -> Array:
    """Compute the Soft-DTW distance between two 1D voltage traces.

    Uses squared Euclidean cost matrix and anti-diagonal scan with
    numerically stable gradients (custom_vjp handles inf correctly).

    Args:
        sim_voltage: Simulated voltage trace [N]
        exp_voltage: Experimental (target) voltage trace [M]
        gamma: Softmin smoothing parameter (>0)

    Returns:
        Soft-DTW distance (scalar)
    """
    return _soft_dtw_impl(sim_voltage, exp_voltage, gamma)


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
