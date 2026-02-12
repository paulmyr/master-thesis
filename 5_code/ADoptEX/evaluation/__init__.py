"""
Evaluation metrics for AdEx model fitting.

This package provides:
- coincidence: Coincidence factor (Gamma) from Jolivet et al. (2008)
"""

from .coincidence import (CoincidenceResult, coincidence_factor,
                          coincidence_factor_from_traces, count_coincidences,
                          detect_spike_times,
                          detect_spike_times_from_spike_trace,
                          global_performance, intrinsic_reliability)

__all__ = [
    "CoincidenceResult",
    "detect_spike_times",
    "detect_spike_times_from_spike_trace",
    "count_coincidences",
    "coincidence_factor",
    "coincidence_factor_from_traces",
    "intrinsic_reliability",
    "global_performance",
]
