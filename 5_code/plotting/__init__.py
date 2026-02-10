"""
Plotting utilities for AdEx simulation and training visualization.

This package provides:
- traces: Voltage trace and spike raster plotting
- comparison: Brian2 vs Jaxley comparison plots
- training: Loss curves and training progress
"""

from .plots import (plot_coincidence_evaluation, plot_combined_comparison,
                    plot_comparison, plot_fit_results, plot_spike_raster,
                    plot_training_results, plot_voltage_trace)

__all__ = [
    "plot_voltage_trace",
    "plot_spike_raster",
    "plot_comparison",
    "plot_training_results",
    "plot_fit_results",
    "plot_coincidence_evaluation",
    "plot_combined_comparison",
]
