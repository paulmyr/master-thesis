"""
Post-training evaluation of AdEx model fits.

Computes evaluation metrics (coincidence factor, spike count accuracy,
timing errors) and generates visualization plots.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.simulation import SimulationResult
from evaluation.coincidence import (CoincidenceResult, coincidence_factor,
                                    detect_spike_times)


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""

    # Coincidence factor
    gamma: float
    coincidence_result: CoincidenceResult

    # Spike counts
    spike_count_target: int
    spike_count_model: int

    # Timing
    first_spike_error_ms: float
    mean_spike_timing_error_ms: float

    @property
    def spike_count_diff(self) -> int:
        """Difference in spike count (model - target)."""
        return self.spike_count_model - self.spike_count_target

    @property
    def spike_count_accurate(self) -> bool:
        """Whether spike count is within tolerance (+-1)."""
        return abs(self.spike_count_diff) <= 1

    def summary(self) -> str:
        """Generate text summary of metrics."""
        return f"""
Evaluation Metrics:
  Coincidence Factor (Γ): {self.gamma:.4f}
  Spike Count: {self.spike_count_model} / {self.spike_count_target} (diff: {self.spike_count_diff:+d})
  First Spike Error: {self.first_spike_error_ms:.2f} ms
  Mean Timing Error: {self.mean_spike_timing_error_ms:.2f} ms
  Spike Count Accurate: {self.spike_count_accurate}
"""


class Evaluator:
    """
    Evaluate fitted AdEx model against experimental data.

    This class computes evaluation metrics and generates plots for
    assessing the quality of parameter optimization.

    Attributes:
        delta_ms: Coincidence window for gamma calculation (default 2.0 ms)
        spike_threshold_mv: Threshold for spike detection (default 0.0 mV)

    Example:
        >>> evaluator = Evaluator()
        >>> metrics = evaluator.compute_metrics(
        ...     target_voltage, model_voltage, dt_ms=0.1, duration_ms=500.0
        ... )
        >>> print(f"Gamma: {metrics.gamma:.4f}")
    """

    def __init__(
        self,
        delta_ms: float = 2.0,
        spike_threshold_mv: float = 0.0,
    ):
        """
        Initialize the evaluator.

        Args:
            delta_ms: Coincidence window in ms for gamma calculation
            spike_threshold_mv: Voltage threshold for spike detection
        """
        self.delta_ms = delta_ms
        self.spike_threshold_mv = spike_threshold_mv

    def compute_metrics(
        self,
        target_voltage: np.ndarray,
        model_voltage: np.ndarray,
        dt_ms: float,
        duration_ms: float | None = None,
        target_spike_times: np.ndarray | None = None,
        model_spike_times: np.ndarray | None = None,
    ) -> EvaluationMetrics:
        """
        Compute all evaluation metrics.

        Args:
            target_voltage: Experimental voltage trace (mV)
            model_voltage: Simulated voltage trace (mV)
            dt_ms: Time step in ms
            duration_ms: Duration of trace (computed from length if None)
            target_spike_times: Pre-computed target spike times (optional)
            model_spike_times: Pre-computed model spike times (optional)

        Returns:
            EvaluationMetrics with all computed metrics
        """
        if duration_ms is None:
            duration_ms = len(target_voltage) * dt_ms

        # Detect spikes if not provided
        if target_spike_times is None:
            target_spike_times = detect_spike_times(
                target_voltage, dt_ms, threshold=self.spike_threshold_mv
            )
        if model_spike_times is None:
            model_spike_times = detect_spike_times(
                model_voltage, dt_ms, threshold=self.spike_threshold_mv
            )

        # Compute coincidence factor
        coincidence_result = coincidence_factor(
            target_spike_times,
            model_spike_times,
            duration_ms,
            self.delta_ms,
        )

        # Spike counts
        n_target = len(target_spike_times)
        n_model = len(model_spike_times)

        # Timing errors
        first_spike_error = self._compute_first_spike_error(
            target_spike_times, model_spike_times
        )
        mean_timing_error = self._compute_mean_timing_error(
            target_spike_times, model_spike_times
        )

        return EvaluationMetrics(
            gamma=coincidence_result.gamma,
            coincidence_result=coincidence_result,
            spike_count_target=n_target,
            spike_count_model=n_model,
            first_spike_error_ms=first_spike_error,
            mean_spike_timing_error_ms=mean_timing_error,
        )

    def _compute_first_spike_error(
        self,
        target_spikes: np.ndarray,
        model_spikes: np.ndarray,
    ) -> float:
        """Compute error in first spike timing."""
        if len(target_spikes) == 0 or len(model_spikes) == 0:
            return float("inf")
        return abs(model_spikes[0] - target_spikes[0])

    def _compute_mean_timing_error(
        self,
        target_spikes: np.ndarray,
        model_spikes: np.ndarray,
    ) -> float:
        """
        Compute mean timing error for matched spikes.

        Uses greedy matching within the coincidence window.
        """
        if len(target_spikes) == 0 or len(model_spikes) == 0:
            return float("inf")

        errors = []
        matched_target = set()

        for model_t in model_spikes:
            # Find closest unmatched target spike
            best_error = float("inf")
            best_idx = -1

            for i, target_t in enumerate(target_spikes):
                if i in matched_target:
                    continue
                error = abs(model_t - target_t)
                if error < best_error and error <= self.delta_ms:
                    best_error = error
                    best_idx = i

            if best_idx >= 0:
                matched_target.add(best_idx)
                errors.append(best_error)

        return np.mean(errors) if errors else float("inf")

    def plot_comparison(
        self,
        target_voltage: np.ndarray,
        model_voltage: np.ndarray,
        target_time: np.ndarray,
        model_time: np.ndarray | None = None,
        target_spike_times: np.ndarray | None = None,
        model_spike_times: np.ndarray | None = None,
        title: str = "Model Fit Comparison",
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Generate comparison plot of target vs model traces.

        Args:
            target_voltage: Experimental voltage trace
            model_voltage: Simulated voltage trace
            target_time: Time array for target trace
            model_time: Time array for model trace (uses target_time if None)
            target_spike_times: Target spike times for raster
            model_spike_times: Model spike times for raster
            title: Plot title
            save_path: Path to save figure (optional)
            show: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        if model_time is None:
            model_time = target_time

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(title)

        # Voltage overlay
        ax_v = axes[0]
        ax_v.plot(
            target_time, target_voltage, "gray", alpha=0.7, label="Target", linewidth=1
        )
        ax_v.plot(model_time, model_voltage, "b-", label="Model", linewidth=1)
        ax_v.set_ylabel("Voltage (mV)")
        ax_v.legend()
        ax_v.grid(True, alpha=0.3)

        # Voltage difference
        ax_diff = axes[1]
        # Interpolate if needed
        if len(model_voltage) != len(target_voltage):
            model_interp = np.interp(target_time, model_time, model_voltage)
        else:
            model_interp = model_voltage
        diff = model_interp - target_voltage
        ax_diff.plot(target_time, diff, "r-", linewidth=0.5)
        ax_diff.set_ylabel("Difference (mV)")
        ax_diff.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax_diff.grid(True, alpha=0.3)

        # Spike raster
        ax_s = axes[2]
        if target_spike_times is not None and len(target_spike_times) > 0:
            ax_s.eventplot(
                [target_spike_times], colors=["gray"], lineoffsets=1, linelengths=0.8
            )
        if model_spike_times is not None and len(model_spike_times) > 0:
            ax_s.eventplot(
                [model_spike_times], colors=["blue"], lineoffsets=0, linelengths=0.8
            )
        ax_s.set_yticks([0, 1])
        ax_s.set_yticklabels(["Model", "Target"])
        ax_s.set_xlabel("Time (ms)")
        ax_s.set_ylabel("Spikes")
        ax_s.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_loss_curve(
        self,
        loss_history: list[float],
        title: str = "Training Loss",
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot training loss curve.

        Args:
            loss_history: List of loss values
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(loss_history, "b-", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Add final loss annotation
        final_loss = loss_history[-1]
        ax.annotate(
            f"Final: {final_loss:.6f}",
            xy=(len(loss_history) - 1, final_loss),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_metrics_summary(
        self,
        metrics: EvaluationMetrics,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot summary of evaluation metrics as a bar chart.

        Args:
            metrics: EvaluationMetrics to visualize
            save_path: Path to save figure
            show: Whether to display the plot

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Gamma score
        ax_gamma = axes[0]
        color = (
            "green" if metrics.gamma > 0.5 else "orange" if metrics.gamma > 0 else "red"
        )
        ax_gamma.bar(["Γ"], [metrics.gamma], color=color, alpha=0.7)
        ax_gamma.axhline(
            0.82, color="green", linestyle="--", alpha=0.5, label="Target (0.82)"
        )
        ax_gamma.axhline(
            0.5, color="orange", linestyle="--", alpha=0.5, label="Good (0.5)"
        )
        ax_gamma.set_ylim(-0.1, 1.1)
        ax_gamma.set_ylabel("Coincidence Factor")
        ax_gamma.legend(fontsize=8)
        ax_gamma.set_title("Coincidence Factor")

        # Spike count
        ax_count = axes[1]
        ax_count.bar(
            ["Target", "Model"],
            [metrics.spike_count_target, metrics.spike_count_model],
            color=["gray", "blue"],
            alpha=0.7,
        )
        ax_count.set_ylabel("Spike Count")
        ax_count.set_title(f"Spike Count (diff: {metrics.spike_count_diff:+d})")

        # Timing errors
        ax_timing = axes[2]
        errors = [metrics.first_spike_error_ms, metrics.mean_spike_timing_error_ms]
        labels = ["First Spike", "Mean"]
        colors = ["blue", "purple"]
        ax_timing.bar(labels, errors, color=colors, alpha=0.7)
        ax_timing.set_ylabel("Error (ms)")
        ax_timing.set_title("Timing Errors")
        ax_timing.axhline(
            5.0, color="orange", linestyle="--", alpha=0.5, label="5ms threshold"
        )
        ax_timing.legend(fontsize=8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig
