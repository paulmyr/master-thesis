"""
ExperimentRunner - orchestrates training and evaluation.

This module provides the main interface for running optimization experiments,
handling data loading, training, evaluation, and result saving.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np

from config.experiment import (ExperimentConfig, ExperimentResult,
                               create_result_dir)
from core.data import TraceData, load_trace
from core.parameters import DEFAULT_PARAMS, PARAM_BOUNDS
from core.simulation import SimulationResult, simulate_with_current_trace
from loss.dtw import SoftDTWLossConfig, make_soft_dtw_loss_fn
from loss.guarino import (GuarinoLossConfig, extract_experimental_features,
                          make_guarino_loss_fn)
from loss.mse import make_mse_loss_fn
from runner.evaluator import EvaluationMetrics, Evaluator
from training.trainer import (TrainingConfig, TrainingResult,
                              setup_trainable_cell, train)


@dataclass
class TrainingProgress:
    """Progress information during training."""

    epoch: int
    total_epochs: int
    loss: float
    elapsed_time: float
    estimated_remaining: float


class ExperimentRunner:
    """
    Run optimization experiments with full lifecycle management.

    This class handles:
    - Loading experimental data
    - Setting up the training infrastructure
    - Running gradient-based optimization
    - Evaluating results
    - Saving outputs (metrics, plots, config)

    Attributes:
        config: Experiment configuration
        result_dir: Directory for saving results
        evaluator: Evaluator instance for metrics computation
        progress_callback: Optional callback for progress updates

    Example:
        >>> config = ExperimentConfig(
        ...     trace_paths=[("voltage.dat", "current.dat")],
        ...     n_epochs=500,
        ... )
        >>> runner = ExperimentRunner(config)
        >>> result = runner.run()
        >>> print(f"Gamma: {result.gamma:.4f}")
    """

    def __init__(
        self,
        config: ExperimentConfig,
        result_dir: str | Path | None = None,
        progress_callback: Callable[[TrainingProgress], None] | None = None,
    ):
        """
        Initialize the experiment runner.

        Args:
            config: Experiment configuration
            result_dir: Directory for results (auto-created if None)
            progress_callback: Callback for progress updates during training
        """
        self.config = config
        self.result_dir = Path(result_dir) if result_dir else None
        self.progress_callback = progress_callback
        self.evaluator = Evaluator()

        # Internal state
        self._traces: list[TraceData] = []
        self._training_result: TrainingResult | None = None
        self._stop_requested = False

    def request_stop(self) -> None:
        """Request early stopping of training."""
        self._stop_requested = True

    def load_data(self) -> list[TraceData]:
        """
        Load experimental traces specified in config.

        Returns:
            List of TraceData objects
        """
        self._traces = []

        for voltage_path, current_path in self.config.trace_paths:
            trace = load_trace(voltage_path, current_path)
            self._traces.append(trace)

        return self._traces

    def run(self) -> ExperimentResult:
        """
        Run the complete experiment.

        This is the main entry point that orchestrates:
        1. Data loading
        2. Training setup
        3. Optimization
        4. Evaluation
        5. Result saving

        Returns:
            ExperimentResult with all metrics and paths
        """
        self._stop_requested = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create result directory
        if self.result_dir is None:
            self.result_dir = create_result_dir("results")

        # Load data
        if not self._traces:
            self.load_data()

        if not self._traces:
            raise ValueError("No traces loaded. Set trace_paths in config.")

        # For now, use single trace (multi-trace support can be added later)
        trace = self._traces[0]

        # Get initial parameters
        initial_params = self._get_initial_params()

        # Run training
        training_result = self._run_training(trace, initial_params)
        self._training_result = training_result

        # Run evaluation
        metrics = self._run_evaluation(trace, training_result)

        # Generate plots
        plot_path = self._generate_plots(trace, training_result, metrics)

        # Create result object
        result = ExperimentResult(
            config=self.config,
            timestamp=timestamp,
            loss_history=training_result.loss_history,
            time_per_epoch=training_result.time_per_epoch,
            final_params=training_result.get_params_dict(),
            gamma=metrics.gamma,
            spike_count_target=metrics.spike_count_target,
            spike_count_model=metrics.spike_count_model,
            first_spike_error_ms=metrics.first_spike_error_ms,
            plot_path=plot_path,
            result_dir=self.result_dir,
        )

        # Save results
        self._save_results(result)

        return result

    def _get_initial_params(self) -> dict:
        """Get initial parameters from config or defaults."""
        if self.config.initial_params:
            # Merge with defaults for any missing params
            params = DEFAULT_PARAMS.copy()
            params.update(self.config.initial_params)
            return params
        return DEFAULT_PARAMS.copy()

    def _run_training(
        self,
        trace: TraceData,
        initial_params: dict,
    ) -> TrainingResult:
        """
        Run the training loop.

        Args:
            trace: Experimental trace data
            initial_params: Initial parameter values

        Returns:
            TrainingResult with final parameters and history
        """
        # Convert current from pA to nA (Jaxley expects nA)
        current_nA = jnp.array(trace.current * initial_params["C_m"] / 1000.0)

        # Create training config
        training_config = TrainingConfig(
            optimizer=self.config.optimizer,
            learning_rate=self.config.learning_rate,
            n_epochs=self.config.n_epochs,
            surrogate_type=self.config.surrogate_type,
            surrogate_slope=self.config.surrogate_slope,
            clip_to_bounds=self.config.clip_to_bounds,
            print_every=self.config.print_every,
            verbose=True,
            trainable_params=self.config.trainable_params,
        )

        # Setup trainable cell
        cell, data_stimuli, t_max, trainable_params = setup_trainable_cell(
            initial_params,
            current_nA,
            trace.dt_ms,
            training_config,
        )

        # Create loss function
        loss_fn = self._create_loss_fn(cell, data_stimuli, t_max, trace)

        # Run training
        result = train(
            loss_fn,
            trainable_params,
            training_config,
            initial_params,
        )

        return result

    def _create_loss_fn(
        self,
        cell,
        data_stimuli,
        t_max: float,
        trace: TraceData,
    ) -> Callable:
        """Create the appropriate loss function based on config."""
        if self.config.loss_type == "guarino":
            # Extract experimental features
            exp_features = extract_experimental_features(
                jnp.array(trace.voltage),
                trace.dt_ms,
                trace.stim_duration_ms,
                trace.stim_end_idx,
                spike_threshold_mv=0.0,
            )

            # Create loss config from guarino weights
            loss_config = GuarinoLossConfig(
                weight_t_first=self.config.guarino_weights.weight_t_first,
                weight_t_second=self.config.guarino_weights.weight_t_second,
                weight_t_third=self.config.guarino_weights.weight_t_third,
                weight_t_last=self.config.guarino_weights.weight_t_last,
                weight_inv_first_isi=self.config.guarino_weights.weight_inv_first_isi,
                weight_inv_last_isi=self.config.guarino_weights.weight_inv_last_isi,
                weight_firing_freq=self.config.guarino_weights.weight_firing_freq,
                weight_v_stim_end=self.config.guarino_weights.weight_v_stim_end,
                missing_feature_penalty=self.config.missing_penalty,
            )

            return make_guarino_loss_fn(
                cell,
                data_stimuli,
                t_max,
                trace.dt_ms,
                exp_features,
                trace.stim_duration_ms,
                trace.stim_end_idx,
                loss_config=loss_config,
                temperature=self.config.temperature,
                beta=self.config.beta,
            )

        elif self.config.loss_type == "mse":
            # Use experimental voltage trace directly
            return make_mse_loss_fn(
                cell,
                data_stimuli,
                t_max,
                trace.dt_ms,
                jnp.array(trace.voltage),
                trace.stim_end_idx,
            )

        elif self.config.loss_type == "soft_dtw":
            return make_soft_dtw_loss_fn(
                cell,
                data_stimuli,
                t_max,
                trace.dt_ms,
                jnp.array(trace.voltage),
                trace.stim_end_idx,
                loss_config=SoftDTWLossConfig(),
            )

        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

    def _run_evaluation(
        self,
        trace: TraceData,
        training_result: TrainingResult,
    ) -> EvaluationMetrics:
        """
        Evaluate the trained model.

        Args:
            trace: Experimental trace data
            training_result: Result from training

        Returns:
            EvaluationMetrics with all computed metrics
        """
        # Get final parameters
        final_params = self._get_initial_params()
        final_params.update(training_result.get_params_dict())

        # Run simulation with final parameters
        sim_result = simulate_with_current_trace(
            final_params,
            trace.current,
            trace.dt_ms,
            use_surrogate=False,  # No surrogate needed for evaluation
        )

        # Compute metrics
        metrics = self.evaluator.compute_metrics(
            trace.voltage,
            sim_result.voltage,
            trace.dt_ms,
            trace.duration_ms,
            target_spike_times=trace.spike_times,
            model_spike_times=sim_result.spike_times,
        )

        return metrics

    def _generate_plots(
        self,
        trace: TraceData,
        training_result: TrainingResult,
        metrics: EvaluationMetrics,
    ) -> Path:
        """
        Generate and save evaluation plots.

        Returns:
            Path to the main comparison plot
        """
        # Get final parameters
        final_params = self._get_initial_params()
        final_params.update(training_result.get_params_dict())

        # Run simulation for plotting
        sim_result = simulate_with_current_trace(
            final_params,
            trace.current,
            trace.dt_ms,
            use_surrogate=False,
        )

        # Comparison plot
        comparison_path = self.result_dir / "comparison.png"
        self.evaluator.plot_comparison(
            trace.voltage,
            sim_result.voltage,
            trace.time,
            sim_result.time,
            trace.spike_times,
            sim_result.spike_times,
            title=f"{self.config.name} - Gamma: {metrics.gamma:.4f}",
            save_path=comparison_path,
        )

        # Loss curve
        loss_path = self.result_dir / "loss_curve.png"
        self.evaluator.plot_loss_curve(
            training_result.loss_history,
            title=f"Training Loss - {self.config.loss_type}",
            save_path=loss_path,
        )

        # Metrics summary
        metrics_path = self.result_dir / "metrics.png"
        self.evaluator.plot_metrics_summary(
            metrics,
            save_path=metrics_path,
        )

        return comparison_path

    def _save_results(self, result: ExperimentResult) -> None:
        """Save all result files."""
        # Save config
        config_path = self.result_dir / "config.json"
        self.config.save(config_path)

        # Save result
        result_path = self.result_dir / "result.json"
        result.save(result_path)

        # Print summary
        print(result.summary())
        print(f"\nResults saved to: {self.result_dir}")


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Convenience function to run an experiment.

    Args:
        config: Experiment configuration

    Returns:
        ExperimentResult

    Example:
        >>> from config.presets import get_preset
        >>> config = get_preset("standard")
        >>> config.trace_paths = [("v.dat", "i.dat")]
        >>> result = run_experiment(config)
    """
    runner = ExperimentRunner(config)
    return runner.run()
