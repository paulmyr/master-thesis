"""
Unified training infrastructure for AdEx parameter optimization.

This module provides a single training interface that works with any
loss function (MSE, Guarino, or custom). The loss function is passed
as an argument, making it easy to experiment with different objectives.
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import optax
from jaxley.channels import AdExSurrogate

sys.path.insert(0, "..")
from core.parameters import PARAM_BOUNDS, clip_params
from core.simulation import geometry_for_capacitance


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

    @property
    def final_loss(self) -> float:
        """Final loss value."""
        return self.loss_history[-1] if self.loss_history else float("nan")

    @property
    def total_time(self) -> float:
        """Total training time in seconds."""
        return sum(self.time_per_epoch)

    @property
    def mean_time_per_epoch(self) -> float:
        """Mean time per epoch in seconds."""
        return np.mean(self.time_per_epoch) if self.time_per_epoch else 0.0

    def get_params_dict(self) -> dict:
        """Extract final parameters as a simple dictionary."""
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

    if config.verbose:
        print(f"Starting training with {config.n_epochs} epochs...")
        print(f"Optimizer: {config.optimizer}, LR: {config.learning_rate}")

    for epoch in range(config.n_epochs):
        t0 = time.time()

        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(trainable_params)

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        trainable_params = optax.apply_updates(trainable_params, updates)

        # Apply parameter bounds
        if config.clip_to_bounds:
            trainable_params = _clip_trainable_params(trainable_params, PARAM_BOUNDS)

        t1 = time.time()

        loss_history.append(float(loss))
        time_per_epoch.append(t1 - t0)

        if config.verbose and epoch % config.print_every == 0:
            print(
                f"Epoch {epoch:4d}: loss = {loss:.6f}, "
                f"time = {time_per_epoch[-1]:.3f}s"
            )

    if config.verbose:
        print(f"\nTraining complete!")
        print(f"Final loss: {loss_history[-1]:.6f}")
        print(f"Total time: {sum(time_per_epoch):.1f}s")

    return TrainingResult(
        trainable_params=trainable_params,
        loss_history=loss_history,
        time_per_epoch=time_per_epoch,
        config=config,
        initial_params=initial_params,
    )


def train_with_mse(
    cell,
    data_stimuli,
    t_max: float,
    dt_ms: float,
    target_spike_times: np.ndarray,
    trainable_params: list[dict],
    config: TrainingConfig | None = None,
    sigma_ms: float = 2.0,
) -> TrainingResult:
    """
    Convenience function to train with MSE spike timing loss.

    Args:
        cell: Jaxley cell with trainable parameters
        data_stimuli: Data stimuli tuple
        t_max: Maximum simulation time in ms
        dt_ms: Time step in ms
        target_spike_times: Target spike times in ms
        trainable_params: Initial trainable parameters
        config: Training configuration
        sigma_ms: Gaussian kernel width for soft targets

    Returns:
        TrainingResult
    """
    from loss.mse import create_soft_spike_target, make_mse_loss_fn

    # Create soft target
    n_timesteps = int(t_max / dt_ms)
    target_soft_spikes = create_soft_spike_target(
        jnp.array(target_spike_times), n_timesteps, dt_ms, sigma_ms=sigma_ms
    )

    # Create loss function
    loss_fn = make_mse_loss_fn(
        cell, data_stimuli, t_max, dt_ms, target_soft_spikes=target_soft_spikes
    )

    return train(loss_fn, trainable_params, config)


def train_with_guarino(
    cell,
    data_stimuli,
    t_max: float,
    dt_ms: float,
    exp_features,
    stim_duration_ms: float,
    stim_end_index: int,
    trainable_params: list[dict],
    config: TrainingConfig | None = None,
    temperature: float = 0.1,
    beta: float = 10.0,
) -> TrainingResult:
    """
    Convenience function to train with Guarino feature-based loss.

    Args:
        cell: Jaxley cell with trainable parameters
        data_stimuli: Data stimuli tuple
        t_max: Maximum simulation time in ms
        dt_ms: Time step in ms
        exp_features: Target GuarinoFeatures from experimental data
        stim_duration_ms: Stimulus duration in ms
        stim_end_index: Index of stimulus end
        trainable_params: Initial trainable parameters
        config: Training configuration
        temperature: Softmax temperature for soft feature extraction
        beta: Sigmoid sharpness

    Returns:
        TrainingResult
    """
    from loss.guarino import make_guarino_loss_fn

    # Create loss function
    loss_fn = make_guarino_loss_fn(
        cell,
        data_stimuli,
        t_max,
        dt_ms,
        exp_features,
        stim_duration_ms,
        stim_end_index,
        temperature=temperature,
        beta=beta,
    )

    return train(loss_fn, trainable_params, config)
