"""
Experiment configuration and result dataclasses.

This module defines the configuration schema for optimization experiments
and the structure for storing results.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
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

    Attributes:
        name: Human-readable experiment name
        trace_paths: List of (voltage_file, current_file) path tuples
        loss_type: Type of loss function to use
        guarino_weights: Feature weights for Guarino loss
        missing_penalty: Penalty for missing features (Guarino loss)
        surrogate_type: Type of surrogate gradient function
        surrogate_slope: Steepness of surrogate gradient (beta)
        temperature: Temperature for soft feature extraction
        beta: Sigmoid sharpness for spike counting
        optimizer: Optimizer type
        learning_rate: Learning rate
        n_epochs: Number of training epochs
        clip_to_bounds: Whether to clip parameters to bounds
        trainable_params: List of parameter names to optimize
        initial_params: Optional initial parameter values
        enable_hyperparam_search: Whether to run hyperparameter search
        hyperparam_grid: Grid of hyperparameters to search
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


@dataclass
class ExperimentResult:
    """
    Results from a completed optimization experiment.

    Stores the configuration, training results, evaluation metrics,
    and paths to generated artifacts.

    Attributes:
        config: The experiment configuration used
        timestamp: When the experiment was run
        loss_history: Loss values at each epoch
        time_per_epoch: Time in seconds for each epoch
        final_params: Optimized parameter values
        gamma: Coincidence factor (primary evaluation metric)
        spike_count_target: Number of spikes in target trace
        spike_count_model: Number of spikes in fitted model
        first_spike_error_ms: Error in first spike timing (ms)
        plot_path: Path to generated plot
        result_dir: Directory containing all result files
    """

    config: ExperimentConfig
    timestamp: str

    # Training results
    loss_history: list[float] = field(default_factory=list)
    time_per_epoch: list[float] = field(default_factory=list)
    final_params: dict[str, float] = field(default_factory=dict)

    # Evaluation metrics
    gamma: float = 0.0
    spike_count_target: int = 0
    spike_count_model: int = 0
    first_spike_error_ms: float = float("inf")

    # Artifact paths
    plot_path: Path | None = None
    result_dir: Path | None = None

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
        return (
            sum(self.time_per_epoch) / len(self.time_per_epoch)
            if self.time_per_epoch
            else 0.0
        )

    @property
    def spike_count_diff(self) -> int:
        """Difference in spike count (model - target)."""
        return self.spike_count_model - self.spike_count_target

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "config": self.config.to_dict(),
            "timestamp": self.timestamp,
            "loss_history": self.loss_history,
            "time_per_epoch": self.time_per_epoch,
            "final_params": self.final_params,
            "gamma": self.gamma,
            "spike_count_target": self.spike_count_target,
            "spike_count_model": self.spike_count_model,
            "first_spike_error_ms": self.first_spike_error_ms,
            "plot_path": str(self.plot_path) if self.plot_path else None,
            "result_dir": str(self.result_dir) if self.result_dir else None,
        }

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentResult":
        """Create from dictionary."""
        config_data = data.pop("config")
        config = ExperimentConfig.from_dict(config_data)

        # Convert paths
        if data.get("plot_path"):
            data["plot_path"] = Path(data["plot_path"])
        if data.get("result_dir"):
            data["result_dir"] = Path(data["result_dir"])

        return cls(config=config, **data)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentResult":
        """Load result from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def summary(self) -> str:
        """Generate a text summary of results."""
        return f"""
Experiment: {self.config.name}
Timestamp: {self.timestamp}
Loss Type: {self.config.loss_type}

Training:
  Epochs: {len(self.loss_history)}
  Final Loss: {self.final_loss:.6f}
  Total Time: {self.total_time:.1f}s

Evaluation:
  Gamma (Γ): {self.gamma:.4f}
  Spikes: {self.spike_count_model} / {self.spike_count_target} (diff: {self.spike_count_diff:+d})
  First Spike Error: {self.first_spike_error_ms:.2f} ms

Final Parameters:
{self._format_params()}
"""

    def _format_params(self) -> str:
        """Format parameters as string."""
        if not self.final_params:
            return "  (none)"
        return "\n".join(f"  {k}: {v:.4f}" for k, v in self.final_params.items())


def create_result_dir(base_dir: str | Path = "results") -> Path:
    """
    Create a timestamped directory for experiment results.

    Args:
        base_dir: Base directory for results

    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(base_dir) / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir
