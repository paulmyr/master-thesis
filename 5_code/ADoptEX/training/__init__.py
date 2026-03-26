"""
Training infrastructure for AdEx parameter optimization.

This package provides a unified training interface that works with
different loss functions (MSE, Guarino, custom).
"""

from .trainer import (TrainingConfig, TrainingResult, build_param_transform,
                      nudge_from_bounds, setup_trainable_cell,
                      setup_trainable_cell_step, train)

__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "build_param_transform",
    "nudge_from_bounds",
    "setup_trainable_cell",
    "setup_trainable_cell_step",
    "train",
]
