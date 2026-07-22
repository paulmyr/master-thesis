"""
Loss functions for AdEx parameter optimization.

This package provides:
- mse: Mean squared error based losses (voltage trace, spike timing)
- guarino: Feature-based loss from Guarino et al. (2025)
- van_rossum: Van Rossum spike train distance (van Rossum, 2001)
- ttfs_rate: Minimal time-to-first-spike + firing-rate feature loss
- soft_dtw: Soft-DTW voltage-trace distance (Cuturi & Blondel, 2017)
- inject_spike_peaks: Differentiable spike peak injection for voltage traces
"""

from jax import Array

from .guarino import (GuarinoFeatureExtractor, GuarinoFeatures,
                      GuarinoLossConfig, extract_experimental_features,
                      guarino_loss, make_guarino_loss_fn, relative_error)
from .mse import MSELossConfig, make_mse_loss_fn, mse_loss
from .ttfs_rate import (TTFSRateLossConfig, make_ttfs_rate_loss_fn,
                        ttfs_rate_loss)
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


# Imported after inject_spike_peaks is defined: soft_dtw depends on it, so this
# import must follow the definition to avoid a circular-import failure.
from .soft_dtw import (SoftDTWLossConfig, make_soft_dtw_loss_fn,  # noqa: E402
                       sliding_window_max, soft_dtw_from_cost)


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
    # Time-to-first-spike + firing-rate loss
    "TTFSRateLossConfig",
    "ttfs_rate_loss",
    "make_ttfs_rate_loss_fn",
    # Soft-DTW loss
    "SoftDTWLossConfig",
    "make_soft_dtw_loss_fn",
    "sliding_window_max",
    "soft_dtw_from_cost",
]
