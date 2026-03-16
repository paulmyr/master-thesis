"""
Loss functions for AdEx parameter optimization.

This package provides:
- mse: Mean squared error based losses (voltage trace, spike timing)
- guarino: Feature-based loss from Guarino et al. (2025)
- van_rossum: Van Rossum spike train distance (van Rossum, 2001)
- inject_spike_peaks: Differentiable spike peak injection for voltage traces
"""

from jax import Array

from .guarino import (GuarinoFeatureExtractor, GuarinoFeatures,
                      GuarinoLossConfig, extract_experimental_features,
                      guarino_loss, make_guarino_loss_fn, relative_error)
from .mse import MSELossConfig, make_mse_loss_fn, mse_loss
from .van_rossum import (VanRossumLossConfig, make_van_rossum_loss_fn,
                         spike_train_from_voltage, van_rossum_distance)


def inject_spike_peaks(voltage: Array, spikes: Array, v_peak_mv: float = 35.0) -> Array:
    """Replace voltage at spike times with a peak value (differentiable).

    Uses the soft spike trace from surrogate gradients, so gradients flow
    through the spike indicator to the loss.

    Args:
        voltage: Voltage trace [T] in mV
        spikes: Soft spike indicator [T] (0 = no spike, 1 = spike)
        v_peak_mv: Peak voltage to inject at spike times (mV)

    Returns:
        Modified voltage trace with spike peaks injected [T]
    """
    return spikes * v_peak_mv + (1.0 - spikes) * voltage


__all__ = [
    # Utilities
    "inject_spike_peaks",
    # MSE losses
    "MSELossConfig",
    "make_mse_loss_fn",
    # Guarino losses
    "GuarinoFeatures",
    "GuarinoFeatureExtractor",
    "GuarinoLossConfig",
    "extract_experimental_features",
    "make_guarino_loss_fn",
    # Van Rossum losses
    "VanRossumLossConfig",
    "spike_train_from_voltage",
    "make_van_rossum_loss_fn",
]
