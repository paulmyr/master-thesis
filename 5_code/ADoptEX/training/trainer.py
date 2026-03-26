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

log = logging.getLogger(__name__)

from ADoptEX.core.parameters import PARAM_BOUNDS, clip_params
from ADoptEX.core.simulation import create_adex_cell


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimizer settings
    optimizer: Literal["adam", "sgd", "rmsprop", "polyak"] = "adam"
    learning_rate: float = 0.1
    n_epochs: int = 500

    # Gradient clipping (roadmap item 1)
    grad_clip_norm: float | None = None

    # Polyak gradient normalization (roadmap item 3)
    polyak_alpha: float = 1.0  # loss exponent (0 = ignore loss, 1 = scale by loss)
    polyak_beta: float = 1.0  # gradient norm exponent (1 = full normalization)

    # Learning rate schedule (roadmap item 2)
    lr_schedule: Literal["constant", "cosine", "exponential"] | None = None
    lr_decay_rate: float = 0.01  # final/initial LR ratio for exponential decay

    # Surrogate gradient settings
    surrogate_type: Literal["sigmoid", "exponential", "superspike"] = "sigmoid"
    surrogate_slope: float = 5.0

    # Parameter constraints
    clip_to_bounds: bool = True
    use_param_transform: bool = False

    # Logging
    print_every: int = 50
    verbose: bool = True

    # Which parameters to train (None = default set)
    trainable_params: list[str] | None = None

    # Return best-loss parameters instead of last-epoch parameters
    return_best: bool = False

    def __post_init__(self):
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive, got {self.grad_clip_norm}"
            )
        if self.lr_schedule == "exponential" and self.lr_decay_rate <= 0:
            raise ValueError(
                f"lr_decay_rate must be positive for exponential schedule, got {self.lr_decay_rate}"
            )
        if (
            self.lr_schedule is not None
            and self.lr_schedule != "constant"
            and self.n_epochs <= 0
        ):
            raise ValueError(
                f"n_epochs must be positive when using lr_schedule, got {self.n_epochs}"
            )
        if self.use_param_transform and self.clip_to_bounds:
            log.warning(
                "use_param_transform=True with clip_to_bounds=True: "
                "sigmoid transform implicitly enforces bounds, "
                "hard clipping will be skipped."
            )
        if self.optimizer == "polyak":
            if self.polyak_beta <= 0:
                raise ValueError(
                    f"polyak_beta must be positive, got {self.polyak_beta}"
                )
            if self.lr_schedule is not None and self.lr_schedule != "constant":
                raise ValueError(
                    "lr_schedule is incompatible with polyak optimizer "
                    "(Polyak has implicit decay via loss scaling)"
                )
            if self.grad_clip_norm is not None:
                raise ValueError(
                    "grad_clip_norm is incompatible with polyak optimizer "
                    "(normalization replaces clipping)"
                )


@dataclass
class TrainingResult:
    """Result from training."""

    # Final parameters (in Jaxley trainable format)
    trainable_params: list[dict]

    # Training history
    loss_history: list[float]
    time_per_epoch: list[float]

    # Initial parameters for comparison
    initial_params: dict

    # Configuration used (optional when constructing manually in notebooks)
    config: TrainingConfig | None = None

    # Gradient diagnostics (pre-clipping raw norms)
    grad_norms: list[float] = field(default_factory=list)

    # Post-clipping gradient norms (populated when grad_clip_norm is set)
    clipped_grad_norms: list[float] = field(default_factory=list)

    # Learning rate history (populated when a schedule is active)
    lr_history: list[float] = field(default_factory=list)

    # Epoch that achieved the lowest training loss
    best_epoch: int = -1

    # Optional label for display/plotting
    label: str = ""

    @property
    def final_loss(self) -> float:
        """Final loss value."""
        return self.loss_history[-1] if self.loss_history else float("nan")

    @property
    def best_loss(self) -> float:
        """Best (lowest) loss seen during training."""
        return (
            self.loss_history[self.best_epoch] if self.best_epoch >= 0 else float("nan")
        )

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
                clean_name = _param_key_to_bounds_key(name)
                result[clean_name] = float(value.flatten()[0])
        return result


def setup_trainable_cell(
    initial_params: dict,
    current_trace_pA: jnp.ndarray,
    dt_ms: float,
    config: TrainingConfig | None = None,
    trainable_params: list[str] | None = None,
) -> tuple[jx.Cell, tuple, float, list[dict]]:
    """
    Create a Jaxley cell with trainable AdEx parameters and data stimulation.

    Delegates cell creation to ``create_adex_cell``, then adds data stimulation
    and returns the trainable parameter handles.

    Args:
        initial_params: Dictionary of initial AdEx parameters
        current_trace_pA: Current trace in pA (converted to nA internally)
        dt_ms: Time step in ms
        config: Training configuration (optional, provides surrogate settings
            and default trainable_params list)
        trainable_params: List of parameter names to make trainable.
            If None, uses config.trainable_params or default set.

    Returns:
        Tuple of (cell, data_stimuli, t_max, trainable_params_list)
    """
    if config is None:
        config = TrainingConfig()

    if trainable_params is None:
        trainable_params = config.trainable_params
    if trainable_params is None:
        trainable_params = ["C_m", "g_L", "E_L", "v_T", "v_reset", "tau_w", "a", "b"]

    cell = create_adex_cell(
        initial_params,
        use_surrogate=True,
        surrogate_type=config.surrogate_type,
        surrogate_slope=config.surrogate_slope,
        trainable=True,
        trainable_params=trainable_params,
        record=True,
    )

    # Convert pA → nA and setup data stimulation
    current_trace_nA = current_trace_pA / 1000.0
    data_stimuli = cell.comp(0).data_stimulate(current_trace_nA, None)
    t_max = len(current_trace_pA) * dt_ms

    return cell, data_stimuli, t_max, cell.get_parameters()


def setup_trainable_cell_step(
    initial_params: dict,
    stim_current_pA: float,
    stim_duration_ms: float,
    stim_delay_ms: float,
    dt_ms: float,
    config: TrainingConfig | None = None,
    trainable_params: list[str] | None = None,
) -> tuple[jx.Cell, None, float, list[dict]]:
    """
    Create a Jaxley cell with trainable AdEx parameters and step current stimulus.

    Uses ``jx.step_current`` + ``cell.stimulate`` (idiomatic Jaxley step current).

    Args:
        initial_params: Dictionary of initial AdEx parameters
        stim_current_pA: Step current amplitude in pA
        stim_duration_ms: Duration of current injection in ms
        stim_delay_ms: Delay before stimulus onset in ms
        dt_ms: Time step in ms
        config: Training configuration (optional)
        trainable_params: List of parameter names to make trainable

    Returns:
        Tuple of (cell, None, t_max, trainable_params_list).
        Second element is None (no data_stimuli for step current).
    """
    if config is None:
        config = TrainingConfig()

    if trainable_params is None:
        trainable_params = config.trainable_params
    if trainable_params is None:
        trainable_params = ["C_m", "g_L", "E_L", "v_T", "v_reset", "tau_w", "a", "b"]

    cell = create_adex_cell(
        initial_params,
        use_surrogate=True,
        surrogate_type=config.surrogate_type,
        surrogate_slope=config.surrogate_slope,
        trainable=True,
        trainable_params=trainable_params,
        record=True,
    )

    # Step current via jx.step_current (pA -> nA conversion)
    t_max = stim_delay_ms + stim_duration_ms + 50.0
    I_nA = stim_current_pA / 1000.0
    current = jx.step_current(stim_delay_ms, stim_duration_ms, I_nA, dt_ms, t_max=t_max)
    cell.stimulate(current)

    return cell, None, t_max, cell.get_parameters()


def make_initial_trainable_params(
    cell: jx.Cell,
    initial_params: dict[str, float],
) -> list[dict]:
    """Build trainable_params with new initial values for an existing cell.

    This allows reusing a single cell across multiple starts -- the cell
    topology, channels, stimulus, and ``make_trainable()`` calls stay the same,
    only the parameter *values* change.

    Args:
        cell: Cell that already has ``make_trainable()`` called on it.
        initial_params: Dict mapping standard param names (``g_L``, ``C_m``, ...)
            to desired initial values.

    Returns:
        Trainable params list in Jaxley format (same structure as
        ``cell.get_parameters()``), but with values from *initial_params*.
    """
    base_params = cell.get_parameters()
    result = []
    for param_dict in base_params:
        new_dict = {}
        for key, val in param_dict.items():
            clean_key = _param_key_to_bounds_key(key)
            if clean_key in initial_params:
                new_dict[key] = jnp.array([initial_params[clean_key]])
            else:
                new_dict[key] = val
        result.append(new_dict)
    return result


def _create_optimizer(config: TrainingConfig):
    """Create optimizer based on config.

    Supports optional gradient clipping (prepended via ``optax.chain``) and
    learning-rate schedules (passed as schedule callable to the base optimizer).
    """
    # 1. Determine learning rate: scalar or schedule callable
    lr = config.learning_rate
    if config.lr_schedule == "cosine":
        lr = optax.cosine_decay_schedule(
            init_value=config.learning_rate, decay_steps=config.n_epochs
        )
    elif config.lr_schedule == "exponential":
        lr = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.n_epochs,
            decay_rate=config.lr_decay_rate,
        )
    # "constant" or None -> keep scalar lr

    # 2. Create base optimizer
    if config.optimizer == "polyak":
        # Polyak uses inject_hyperparams(sgd) so we can mutate LR per step
        optimizer = optax.inject_hyperparams(optax.sgd)(
            learning_rate=config.learning_rate
        )
        return optimizer

    base_opts = {"adam": optax.adam, "sgd": optax.sgd, "rmsprop": optax.rmsprop}
    if config.optimizer not in base_opts:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    optimizer = base_opts[config.optimizer](lr)

    # 3. Prepend gradient clipping if configured
    if config.grad_clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_norm), optimizer
        )

    return optimizer


def _param_key_to_bounds_key(name: str) -> str:
    """Map a Jaxley parameter key to the corresponding PARAM_BOUNDS key.

    Strips the ``AdEx_`` prefix and maps ``capacitance`` → ``C_m``.
    """
    key = name.replace("AdEx_", "")
    if key == "capacitance":
        key = "C_m"
    return key


def _clip_trainable_params(params: list[dict], bounds: dict) -> list[dict]:
    """Clip trainable parameters to bounds."""
    clipped = []
    for param_dict in params:
        clipped_dict = {}
        for name, value in param_dict.items():
            bounds_key = _param_key_to_bounds_key(name)
            if bounds_key in bounds:
                bound = bounds[bounds_key]
                clipped_dict[name] = jnp.clip(value, bound.min, bound.max)
            else:
                clipped_dict[name] = value
        clipped.append(clipped_dict)
    return clipped


def build_param_transform(params: list[dict], bounds: dict):
    """Build a ``ParamTransform`` mapping each trainable parameter through a sigmoid.

    Imports ``SigmoidTransform`` and ``ParamTransform`` lazily so the Jaxley
    transforms module is only required when this feature is actually used.
    """
    from jaxley.optimize.transforms import ParamTransform, SigmoidTransform

    tf_list: list[dict] = []
    for param_dict in params:
        tf_dict = {}
        for name in param_dict:
            bounds_key = _param_key_to_bounds_key(name)
            if bounds_key not in bounds:
                raise ValueError(
                    f"No bounds for parameter '{name}' (bounds key '{bounds_key}'). "
                    "Cannot build sigmoid transform without bounds."
                )
            b = bounds[bounds_key]
            tf_dict[name] = SigmoidTransform(b.min, b.max)
        tf_list.append(tf_dict)
    return ParamTransform(tf_list)


def nudge_from_bounds(params: list[dict], bounds: dict) -> list[dict]:
    """Nudge parameters sitting at exact bounds inward by a tiny epsilon.

    This prevents ``SigmoidTransform.inverse()`` from returning +/-Inf when
    a parameter value exactly equals a bound.
    """
    nudged = []
    for param_dict in params:
        nudged_dict = {}
        for name, value in param_dict.items():
            bounds_key = _param_key_to_bounds_key(name)
            if bounds_key in bounds:
                # TODO: check if all parameters get nudged or only parameters within
                #  bounds. And if all parameters get nudged, is that really bad?
                b = bounds[bounds_key]
                eps = 1e-4 * (b.max - b.min)
                nudged_dict[name] = jnp.clip(value, b.min + eps, b.max - eps)
            else:
                nudged_dict[name] = value
        nudged.append(nudged_dict)
    return nudged


def _compute_grad_norm(grads: list[dict]) -> float:
    """Compute the L2 norm of all gradients."""
    sq_sum = sum(jnp.sum(g**2) for d in grads for g in d.values())
    return float(jnp.sqrt(sq_sum))


def _format_params(params: list[dict]) -> str:
    """Format trainable params as a compact string for logging."""
    parts = []
    for d in params:
        for name, val in d.items():
            clean = _param_key_to_bounds_key(name)
            parts.append(f"{clean}={float(val.flatten()[0]):.4f}")
    return ", ".join(parts)


def train(
    loss_fn: Callable,
    trainable_params: list[dict],
    config: TrainingConfig | None = None,
    initial_params: dict | None = None,
    grad_fn: Callable | None = None,
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
        grad_fn: Pre-compiled JIT gradient function. If provided, reuses it
            instead of building a new one. Useful when running multiple starts
            that share the same cell and loss structure.

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

    # Sigmoid reparameterization: work in unconstrained space
    param_transform = None
    if config.use_param_transform:
        param_transform = build_param_transform(trainable_params, PARAM_BOUNDS)
        trainable_params = nudge_from_bounds(trainable_params, PARAM_BOUNDS)
        trainable_params = param_transform.inverse(trainable_params)
        # Wrap loss_fn so it maps unconstrained -> constrained before evaluation
        _original_loss_fn = loss_fn
        loss_fn = lambda p: _original_loss_fn(param_transform.forward(p))

    # Create optimizer
    optimizer = _create_optimizer(config)
    opt_state = optimizer.init(trainable_params)

    # Build LR schedule function for logging (if schedule active)
    lr_schedule_fn = None
    if config.lr_schedule == "cosine":
        lr_schedule_fn = optax.cosine_decay_schedule(
            init_value=config.learning_rate, decay_steps=config.n_epochs
        )
    elif config.lr_schedule == "exponential":
        lr_schedule_fn = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.n_epochs,
            decay_rate=config.lr_decay_rate,
        )

    # Training loop
    loss_history = []
    time_per_epoch = []
    grad_norms = []
    clipped_grad_norms = []
    lr_history = []
    best_loss = float("inf")
    best_trainable_params = None
    best_epoch = -1

    if config.verbose:
        extras = []
        if config.optimizer == "polyak":
            extras.append(f"alpha={config.polyak_alpha} beta={config.polyak_beta}")
        if config.grad_clip_norm is not None:
            extras.append(f"clip={config.grad_clip_norm}")
        if config.lr_schedule is not None:
            extras.append(f"schedule={config.lr_schedule}")
        if config.use_param_transform:
            extras.append("sigmoid_transform")
        log.info(
            "Training: optimizer=%s lr=%s epochs=%d surrogate=%s slope=%.1f%s",
            config.optimizer,
            config.learning_rate,
            config.n_epochs,
            config.surrogate_type,
            config.surrogate_slope,
            (" " + " ".join(extras)) if extras else "",
        )

    # JIT-compile value_and_grad once; first call triggers XLA compilation,
    # all subsequent epochs reuse the compiled code (10-100x faster per epoch).
    # Callers can pass a pre-compiled grad_fn to reuse across multiple starts.
    if grad_fn is None:
        grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    for epoch in range(config.n_epochs):
        t0 = time.time()

        # Compute loss and gradients
        loss, grads = grad_fn(trainable_params)

        # Gradient diagnostics
        grad_norm = _compute_grad_norm(grads)

        # Early exit on NaN/Inf
        if jnp.isnan(loss) or jnp.isinf(loss):
            log.warning("NaN/Inf loss at epoch %d — stopping early", epoch)
            break
        if jnp.isnan(grad_norm) or jnp.isinf(grad_norm):
            log.warning("NaN/Inf gradients at epoch %d — stopping early", epoch)
            break

        # Update parameters
        if config.optimizer == "polyak":
            # Polyak normalization: normalize grad direction, scale LR by loss
            norm = _compute_grad_norm(grads)
            normalized_grads = jax.tree.map(
                lambda g: g / (norm**config.polyak_beta + 1e-8), grads
            )
            effective_lr = config.learning_rate * abs(float(loss)) ** config.polyak_alpha
            opt_state.hyperparams["learning_rate"] = effective_lr
            updates, opt_state = optimizer.update(normalized_grads, opt_state)
        else:
            updates, opt_state = optimizer.update(grads, opt_state)
        trainable_params = optax.apply_updates(trainable_params, updates)

        # Apply parameter bounds (skip when sigmoid transform handles it)
        if config.clip_to_bounds and param_transform is None:
            trainable_params = _clip_trainable_params(trainable_params, PARAM_BOUNDS)

        t1 = time.time()

        loss_val = float(loss)
        loss_history.append(loss_val)
        time_per_epoch.append(t1 - t0)
        grad_norms.append(float(grad_norm))

        # Post-clipping gradient norm: clip_by_global_norm caps the L2 norm
        if config.grad_clip_norm is not None:
            clipped_grad_norms.append(min(float(grad_norm), config.grad_clip_norm))

        # Track learning rate when schedule is active or polyak is used
        if config.optimizer == "polyak":
            lr_history.append(effective_lr)
        elif lr_schedule_fn is not None:
            lr_history.append(float(lr_schedule_fn(epoch)))

        # Track best parameters (use abs to handle losses that can go negative,
        # e.g. Soft-DTW where the floor is negative due to softmin bias)
        if abs(loss_val) < abs(best_loss):
            best_loss = loss_val
            best_trainable_params = trainable_params
            best_epoch = epoch

        # Periodic logging
        if epoch % config.print_every == 0:
            lr_str = f" | lr={lr_history[-1]:.2e}" if lr_history else ""
            display_params = (
                param_transform.forward(trainable_params)
                if param_transform is not None
                else trainable_params
            )
            log.info(
                "Epoch %4d | loss=%.6f | grad_norm=%.2e%s | %s",
                epoch,
                loss_val,
                float(grad_norm),
                lr_str,
                _format_params(display_params),
            )

    if config.verbose and loss_history:
        lr_decay_str = ""
        if lr_history:
            lr_decay_str = f" lr={lr_history[0]:.2e}->{lr_history[-1]:.2e}"
        log.info(
            "Training complete: final_loss=%.6f best_loss=%.6f (epoch %d) "
            "total_time=%.1fs mean_epoch=%.1fms%s%s",
            loss_history[-1],
            loss_history[best_epoch] if best_epoch >= 0 else float("nan"),
            best_epoch,
            sum(time_per_epoch),
            np.mean(time_per_epoch) * 1000,
            lr_decay_str,
            " [returning best]" if config.return_best else "",
        )

    # Return best-epoch or last-epoch params based on config
    output_params = (
        best_trainable_params
        if config.return_best and best_trainable_params is not None
        else trainable_params
    )

    # Map back to constrained space when using sigmoid transform
    if param_transform is not None:
        output_params = param_transform.forward(output_params)

    return TrainingResult(
        trainable_params=output_params,
        loss_history=loss_history,
        time_per_epoch=time_per_epoch,
        config=config,
        initial_params=initial_params,
        grad_norms=grad_norms,
        clipped_grad_norms=clipped_grad_norms,
        lr_history=lr_history,
        best_epoch=best_epoch,
    )


def train_scan(
    loss_fn: Callable,
    trainable_params: list[dict],
    config: TrainingConfig,
) -> tuple[list[dict], jnp.ndarray]:
    """Pure-JAX training loop using ``jax.lax.scan``.

    Unlike :func:`train`, this function has **no side effects** (no logging,
    timing, early stopping, or best-param tracking).  This makes it compatible
    with ``jax.jit`` and ``jax.vmap``, enabling GPU-parallel multi-start
    optimization.

    Args:
        loss_fn: Differentiable loss function ``params -> scalar``.
        trainable_params: Initial trainable parameters (Jaxley format).
        config: Training configuration (optimizer, LR, epochs, etc.).

    Returns:
        ``(final_params, loss_history)`` where *loss_history* is a 1-D
        ``jnp.ndarray`` of length ``config.n_epochs``.
    """
    # Sigmoid reparameterization (if enabled)
    param_transform = None
    if config.use_param_transform:
        param_transform = build_param_transform(trainable_params, PARAM_BOUNDS)
        trainable_params = nudge_from_bounds(trainable_params, PARAM_BOUNDS)
        trainable_params = param_transform.inverse(trainable_params)
        _original_loss_fn = loss_fn
        loss_fn = lambda p: _original_loss_fn(param_transform.forward(p))

    optimizer = _create_optimizer(config)
    opt_state = optimizer.init(trainable_params)

    clip = config.clip_to_bounds and param_transform is None

    if config.optimizer == "polyak":

        def step(carry, _):
            params, opt_st = carry
            loss, grads = jax.value_and_grad(loss_fn)(params)
            norm = jnp.sqrt(
                sum(jnp.sum(g**2) for d in grads for g in d.values())
            )
            normalized_grads = jax.tree.map(
                lambda g: g / (norm**config.polyak_beta + 1e-8), grads
            )
            effective_lr = config.learning_rate * jnp.abs(loss) ** config.polyak_alpha
            opt_st.hyperparams["learning_rate"] = effective_lr
            updates, new_opt_st = optimizer.update(normalized_grads, opt_st)
            new_params = optax.apply_updates(params, updates)
            if clip:
                new_params = _clip_trainable_params(new_params, PARAM_BOUNDS)
            return (new_params, new_opt_st), loss

    else:

        def step(carry, _):
            params, opt_st = carry
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_st = optimizer.update(grads, opt_st)
            new_params = optax.apply_updates(params, updates)
            if clip:
                new_params = _clip_trainable_params(new_params, PARAM_BOUNDS)
            return (new_params, new_opt_st), loss

    (final_params, _), loss_history = jax.lax.scan(
        step, (trainable_params, opt_state), None, length=config.n_epochs
    )

    # Map back to constrained space
    if param_transform is not None:
        final_params = param_transform.forward(final_params)

    return final_params, loss_history
