"""
Core modules for AdEx simulation and evaluation.

This package provides the foundational components:
- simulation: AdEx simulation in Jaxley and Brian2
- data: Experimental data loading and preprocessing
- parameters: Standard parameter sets
"""

from .data import (TraceData, crop_to_stim_window, detect_spikes,
                   find_stim_window, load_multiple_traces, load_trace)
from .parameters import (DATA_PATHS, DEFAULT_PARAMS, NAUD_PARAMETERS,
                         PARAM_BOUNDS, AdExParams, ParamBounds, clip_params,
                         convert_trainable_to_params,
                         params_to_trainable_format)
from .simulation import (SimulationResult, SurrogateType, create_adex_cell,
                         geometry_for_capacitance, simulate_brian2,
                         simulate_jaxley, simulate_with_current_trace)

__all__ = [
    # simulation
    "SimulationResult",
    "SurrogateType",
    "geometry_for_capacitance",
    "create_adex_cell",
    "simulate_jaxley",
    "simulate_brian2",
    "simulate_with_current_trace",
    # data
    "TraceData",
    "detect_spikes",
    "find_stim_window",
    "load_trace",
    "load_multiple_traces",
    "crop_to_stim_window",
    # parameters
    "AdExParams",
    "ParamBounds",
    "NAUD_PARAMETERS",
    "DEFAULT_PARAMS",
    "PARAM_BOUNDS",
    "DATA_PATHS",
    "clip_params",
    "convert_trainable_to_params",
    "params_to_trainable_format",
]
