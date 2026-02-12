"""
Training infrastructure for AdEx parameter optimization.

This package provides a unified training interface that works with
different loss functions (MSE, Guarino, custom).
"""

from .trainer import (TrainingConfig, TrainingResult, setup_trainable_cell,
                      train)

__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "setup_trainable_cell",
    "train",
]
