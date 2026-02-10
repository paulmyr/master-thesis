"""
Experiment configuration module.

Provides dataclasses for experiment configuration and results,
as well as preset configurations for common use cases.
"""

from .experiment import ExperimentConfig, ExperimentResult, GuarinoWeights
from .presets import PRESETS, get_preset

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "GuarinoWeights",
    "PRESETS",
    "get_preset",
]
