"""
Unified training infrastructure for AdEx parameter optimization.

This module provides a single training interface that works with any
loss function (MSE, Guarino, or custom). The loss function is passed
as an argument, making it easy to experiment with different objectives.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import optax
from jaxley.channels import AdExSurrogate

log = logging.getLogger(__name__)

from ADoptEX.core.parameters import PARAM_BOUNDS, clip_params
from ADoptEX.core.simulation import geometry_for_capacitance


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimizer settings
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"
    learning_rate: float = 0.1
    n_epochs: int = 500

    # Surrogate gradient settings
    surrogate_type: Literal["sigmoid", "exponential", "superspike"] = "sigmoid"
    surrogate_slope: float = 25.0

    # Parameter constraints
    clip_to_bounds: bool = True

    # Logging
    print_every: int = 50
    verbose: bool = True

    # Which parameters to train (None = default set)
    trainable_params: list[str] | None = None

    # Return best-loss parameters instead of last-epoch parameters
    return_best: bool = False


@dataclass
class TrainingResult:
    """Result from training."""

    # Final parameters (in Jaxley trainable format)
    trainable_params: list[dict]

    # Training history
    loss_history: list[float]
    time_per_epoch: list[float]

    # Configuration used
    config: TrainingConfig

    # Initial parameters for comparison
    initial_params: dict

    # Gradient diagnostics
    grad_norms: list[float] = field(default_factory=list)

    # Epoch that achieved the lowest training loss
    best_epoch: int = -1

    @property
    def final_loss(self) -> float:
        """Final loss value."""
        return self.loss_history[-1] if self.loss_history else float("nan")

    @property
    def best_loss(self) -> float:
        """Best (lowest) loss seen during training."""
        return self.loss_history[self.best_epoch] if self.best_epoch >= 0 else float("nan")

    @property
    def total_time(self) -> float:
        """Total training time in seconds."""
        return sum(self.time_per_epoch)

    @property
    def mean_time_per_epoch(self) -> float:
        """Mean time per epoch in seconds."""
        return np.mean(self.time_per_epoch) if self.time_per_epoch else 0.0

    def get_params_dict(self) -> dict:
        """Extract parameters as a simple dictionary."""
        result = {}
        for param_dict in self.trainable_params:
            for name, value in param_dict.items():
                clean_name = name.replace("AdEx_", "")
                result[clean_name] = float(value.flatten()[0])
        return result


def setup_trainable_cell(
    initial_params: dict,
    current_trace_nA: jnp.ndarray,
    dt_ms: float,
    config: TrainingConfig | None = None,
    trainable_params: list[str] | None = None,
) -> tuple[jx.Cell, tuple, float, list[dict]]:
    """
    Create a Jaxley cell with trainable AdEx parameters.

    This sets up the cell with make_trainable() so parameters can be
    optimized via jx.integrate(cell, params=params).

    Args:
        initial_params: Dictionary of initial AdEx parameters
        current_trace_nA: Current trace in nA for data_stimulate
        dt_ms: Time step in ms
        config: Training configuration (optional)
        trainable_params: List of parameter names to make trainable.
            If None, uses default set: ['g_L', 'E_L', 'v_T', 'v_reset', 'tau_w', 'a', 'b']

    Returns:
        Tuple of (cell, data_stimuli, t_max, trainable_params_list)

    Example:
        >>> cell, data_stimuli, t_max, params = setup_trainable_cell(
        ...     initial_params, current_nA, dt_ms=0.1
        ... )
    """
    if config is None:
        config = TrainingConfig()

    if trainable_params is None:
        trainable_params = config.trainable_params
    if trainable_params is None:
        trainable_params = ["g_L", "E_L", "v_T", "v_reset", "tau_w", "a", "b"]

    # Calculate geometry from capacitance
    radius_um, length_um = geometry_for_capacitance(initial_params["C_m"])

    # Create cell
    cell = jx.Cell()
    cell.set("radius", radius_um)
    cell.set("length", length_um)

    # Insert AdExSurrogate for differentiable spikes
    cell.insert(
        AdExSurrogate(
            surrogate_type=config.surrogate_type, surrogate_slope=config.surrogate_slope
        )
    )

    # Set initial parameter values
    cell.set("capacitance", initial_params["C_m"])
    cell.set("AdEx_C_m", initial_params["C_m"])
    cell.set("AdEx_g_L", initial_params["g_L"])
    cell.set("AdEx_E_L", initial_params["E_L"])
    cell.set("AdEx_v_T", initial_params["v_T"])
    cell.set("AdEx_delta_T", initial_params["delta_T"])
    cell.set("AdEx_v_threshold", initial_params["v_threshold"])
    cell.set("AdEx_v_reset", initial_params["v_reset"])
    cell.set("AdEx_tau_w", initial_params["tau_w"])
    cell.set("AdEx_a", initial_params["a"])
    cell.set("AdEx_b", initial_params["b"])

    # Set initial voltage
    cell.set("v", initial_params.get("E_L", -70.0))

    # Setup recording
    cell.record("v")
    cell.record("AdEx_w")
    cell.record("AdEx_spikes")

    # Make specified parameters trainable
    for param_name in trainable_params:
        cell.make_trainable(f"AdEx_{param_name}")

    # Setup data stimulation
    data_stimuli = cell.comp(0).data_stimulate(current_trace_nA, None)
    t_max = len(current_trace_nA) * dt_ms

    # Get trainable parameters
    trainable_params_list = cell.get_parameters()

    return cell, data_stimuli, t_max, trainable_params_list


def _create_optimizer(config: TrainingConfig):
    """Create optimizer based on config."""
    if config.optimizer == "adam":
        return optax.adam(config.learning_rate)
    elif config.optimizer == "sgd":
        return optax.sgd(config.learning_rate)
    elif config.optimizer == "rmsprop":
        return optax.rmsprop(config.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def _clip_trainable_params(params: list[dict], bounds: dict) -> list[dict]:
    """Clip trainable parameters to bounds."""
    clipped = []
    for param_dict in params:
        clipped_dict = {}
        for name, value in param_dict.items():
            bounds_key = name.replace("AdEx_", "")
            if bounds_key in bounds:
                bound = bounds[bounds_key]
                clipped_dict[name] = jnp.clip(value, bound.min, bound.max)
            else:
                clipped_dict[name] = value
        clipped.append(clipped_dict)
    return clipped


def _compute_grad_norm(grads: list[dict]) -> float:
    """Compute the L2 norm of all gradients."""
    sq_sum = sum(jnp.sum(g**2) for d in grads for g in d.values())
    return float(jnp.sqrt(sq_sum))


def _format_params(params: list[dict]) -> str:
    """Format trainable params as a compact string for logging."""
    parts = []
    for d in params:
        for name, val in d.items():
            clean = name.replace("AdEx_", "")
            parts.append(f"{clean}={float(val.flatten()[0]):.4f}")
    return ", ".join(parts)


def train(
    loss_fn: Callable,
    trainable_params: list[dict],
    config: TrainingConfig | None = None,
    initial_params: dict | None = None,
) -> TrainingResult:
    """
    Train AdEx parameters using gradient descent.

    This is the unified training function that works with any loss function.
    The loss function should take trainable_params and return a scalar loss.

    Args:
        loss_fn: Loss function: params -> scalar. Must be differentiable via JAX.
        trainable_params: Initial trainable parameters from cell.get_parameters()
        config: Training configuration
        initial_params: Initial parameter values (for logging only)

    Returns:
        TrainingResult with final parameters and training history

    Example:
        >>> # With MSE loss
        >>> from loss import make_mse_loss_fn
        >>> loss_fn = make_mse_loss_fn(cell, data_stimuli, t_max, dt_ms, target)
        >>> result = train(loss_fn, trainable_params)
        >>>
        >>> # With Guarino loss
        >>> from loss import make_guarino_loss_fn
        >>> loss_fn = make_guarino_loss_fn(cell, data_stimuli, ...)
        >>> result = train(loss_fn, trainable_params)
    """
    if config is None:
        config = TrainingConfig()

    if initial_params is None:
        initial_params = {}

    # Create optimizer
    optimizer = _create_optimizer(config)
    opt_state = optimizer.init(trainable_params)

    # Training loop
    loss_history = []
    time_per_epoch = []
    grad_norms = []
    best_loss = float("inf")
    best_trainable_params = None
    best_epoch = -1

    if config.verbose:
        log.info(
            "Training: optimizer=%s lr=%s epochs=%d surrogate=%s slope=%.1f",
            config.optimizer, config.learning_rate, config.n_epochs,
            config.surrogate_type, config.surrogate_slope,
        )

    for epoch in range(config.n_epochs):
        t0 = time.time()

        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(trainable_params)

        # Gradient diagnostics
        grad_norm = _compute_grad_norm(grads)

        # Early exit on NaN/Inf
        if jnp.isnan(loss) or jnp.isinf(loss):
            log.warning("NaN/Inf loss at epoch %d — stopping early", epoch)
            break
        if jnp.isnan(grad_norm):
            log.warning("NaN gradients at epoch %d — stopping early", epoch)
            break

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        trainable_params = optax.apply_updates(trainable_params, updates)

        # Apply parameter bounds
        if config.clip_to_bounds:
            trainable_params = _clip_trainable_params(trainable_params, PARAM_BOUNDS)

        t1 = time.time()

        loss_val = float(loss)
        loss_history.append(loss_val)
        time_per_epoch.append(t1 - t0)
        grad_norms.append(float(grad_norm))

        # Track best parameters
        if loss_val < best_loss:
            best_loss = loss_val
            best_trainable_params = trainable_params
            best_epoch = epoch

        # Periodic logging
        if epoch % config.print_every == 0:
            log.info(
                "Epoch %4d | loss=%.6f | grad_norm=%.2e | %s",
                epoch, loss_val, float(grad_norm),
                _format_params(trainable_params),
            )

    if config.verbose and loss_history:
        log.info(
            "Training complete: final_loss=%.6f best_loss=%.6f (epoch %d) "
            "total_time=%.1fs mean_epoch=%.1fms%s",
            loss_history[-1], loss_history[best_epoch] if best_epoch >= 0 else float("nan"),
            best_epoch,
            sum(time_per_epoch), np.mean(time_per_epoch) * 1000,
            " [returning best]" if config.return_best else "",
        )

    # Return best-epoch or last-epoch params based on config
    output_params = best_trainable_params if config.return_best and best_trainable_params is not None else trainable_params

    return TrainingResult(
        trainable_params=output_params,
        loss_history=loss_history,
        time_per_epoch=time_per_epoch,
        config=config,
        initial_params=initial_params,
        grad_norms=grad_norms,
        best_epoch=best_epoch,
    )
