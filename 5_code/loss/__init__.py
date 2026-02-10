"""
Loss functions for AdEx parameter optimization.

This package provides:
- mse: Mean squared error based losses (voltage trace, spike timing)
- guarino: Feature-based loss from Guarino et al. (2025)
"""

from .dtw import SoftDTWLossConfig, make_soft_dtw_loss_fn, soft_dtw_mae_loss
from .guarino import (GuarinoFeatureExtractor, GuarinoFeatures,
                      GuarinoLossConfig, extract_experimental_features,
                      guarino_loss, make_guarino_loss_fn, relative_error)
from .mse import MSELossConfig, make_mse_loss_fn, mse_loss

__all__ = [
    # MSE losses
    "MSELossConfig",
    "mse_loss",
    "make_mse_loss_fn",
    # Guarino losses
    "GuarinoFeatures",
    "GuarinoFeatureExtractor",
    "GuarinoLossConfig",
    "guarino_loss",
    "relative_error",
    "extract_experimental_features",
    "make_guarino_loss_fn",
    # Soft-DTW losses
    "SoftDTWLossConfig",
    "soft_dtw_mae_loss",
    "make_soft_dtw_loss_fn",
]
