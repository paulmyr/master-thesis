"""
Experiment configuration dataclasses.

This module defines the configuration schema for optimization experiments.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class GuarinoWeights:
    """Feature weights for Guarino loss function."""

    weight_t_first: float = 1.0
    weight_t_second: float = 1.0
    weight_t_third: float = 1.0
    weight_t_last: float = 1.0
    weight_inv_first_isi: float = 1.0
    weight_inv_last_isi: float = 1.0
    weight_firing_freq: float = 1.0
    weight_v_stim_end: float = 1.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for loss config."""
        return {
            "weight_t_first": self.weight_t_first,
            "weight_t_second": self.weight_t_second,
            "weight_t_third": self.weight_t_third,
            "weight_t_last": self.weight_t_last,
            "weight_inv_first_isi": self.weight_inv_first_isi,
            "weight_inv_last_isi": self.weight_inv_last_isi,
            "weight_firing_freq": self.weight_firing_freq,
            "weight_v_stim_end": self.weight_v_stim_end,
        }


@dataclass
class ExperimentConfig:
    """
    Complete configuration for an optimization experiment.

    This dataclass captures all hyperparameters needed to run a gradient-based
    optimization experiment on AdEx neuron parameters.
    """

    # Experiment identification
    name: str = "unnamed_experiment"

    # Data selection
    trace_paths: list[tuple[str, str]] = field(default_factory=list)

    # Loss function configuration
    loss_type: Literal["mse", "guarino", "soft_dtw"] = "guarino"
    guarino_weights: GuarinoWeights = field(default_factory=GuarinoWeights)
    missing_penalty: float = 3.0

    # Surrogate gradient configuration
    surrogate_type: Literal["sigmoid", "exponential", "superspike"] = "sigmoid"
    surrogate_slope: float = 25.0

    # Feature extraction (Guarino loss)
    temperature: float = 0.1
    beta: float = 10.0

    # Optimizer configuration
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"
    learning_rate: float = 0.1
    n_epochs: int = 500
    clip_to_bounds: bool = True
    print_every: int = 50

    # Parameters to train
    trainable_params: list[str] = field(
        default_factory=lambda: ["g_L", "E_L", "v_T", "v_reset", "tau_w", "a", "b"]
    )

    # Initial parameters (if None, uses defaults from core.parameters)
    initial_params: dict[str, float] | None = None

    # Hyperparameter search
    enable_hyperparam_search: bool = False
    hyperparam_grid: dict | None = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "trace_paths": self.trace_paths,
            "loss_type": self.loss_type,
            "guarino_weights": self.guarino_weights.to_dict(),
            "missing_penalty": self.missing_penalty,
            "surrogate_type": self.surrogate_type,
            "surrogate_slope": self.surrogate_slope,
            "temperature": self.temperature,
            "beta": self.beta,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "clip_to_bounds": self.clip_to_bounds,
            "print_every": self.print_every,
            "trainable_params": self.trainable_params,
            "initial_params": self.initial_params,
            "enable_hyperparam_search": self.enable_hyperparam_search,
            "hyperparam_grid": self.hyperparam_grid,
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create from dictionary."""
        weights_data = data.pop("guarino_weights", {})
        weights = GuarinoWeights(**weights_data)
        return cls(guarino_weights=weights, **data)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
