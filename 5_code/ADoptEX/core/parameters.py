"""
Standard parameter sets for AdEx neuron models.

This module provides:
- NAUD_PARAMETERS: Parameter sets from Naud et al. (2008)
- DEFAULT_PARAMS: Default starting point for optimization
- PARAM_BOUNDS: Biophysically realistic parameter bounds
- DATA_PATHS: Paths to experimental data files

References:
- Naud R, et al. (2008) Firing patterns in the adaptive exponential integrate-and-fire model.
- Brette R, Gerstner W (2005) Adaptive exponential integrate-and-fire model.
"""

from dataclasses import dataclass
from typing import TypedDict


class AdExParams(TypedDict, total=False):
    """Type definition for AdEx parameters."""

    C_m: float  # Membrane capacitance (pF)
    g_L: float  # Leak conductance (nS)
    E_L: float  # Leak reversal potential (mV)
    v_T: float  # Spike threshold (mV)
    delta_T: float  # Spike slope factor (mV)
    v_reset: float  # Reset potential (mV)
    v_threshold: float  # Detection threshold (mV)
    tau_w: float  # Adaptation time constant (ms)
    a: float  # Subthreshold adaptation (nS)
    b: float  # Spike-triggered adaptation (pA)
    I: float  # Injected current (pA) - for step current simulations


# =============================================================================
# Naud et al. (2008) Parameter Sets
# =============================================================================

NAUD_PARAMETERS: dict[str, AdExParams] = {
    "tonic": {
        "C_m": 200.0,  # pF
        "g_L": 10.0,  # nS
        "E_L": -70.0,  # mV
        "v_T": -50.0,  # mV
        "delta_T": 2.0,  # mV
        "v_reset": -58.0,  # mV
        "v_threshold": 0.0,  # mV
        "tau_w": 30.0,  # ms
        "a": 2.0,  # nS
        "b": 0.0,  # pA - no spike-triggered adaptation
        "I": 500.0,  # pA
    },
    "adaptation": {
        "C_m": 200.0,  # pF
        "g_L": 12.0,  # nS - slightly higher leak
        "E_L": -70.0,  # mV
        "v_T": -50.0,  # mV
        "delta_T": 2.0,  # mV
        "v_reset": -58.0,  # mV
        "v_threshold": 0.0,  # mV
        "tau_w": 300.0,  # ms - 10x slower adaptation
        "a": 2.0,  # nS
        "b": 60.0,  # pA - strong spike-triggered adaptation
        "I": 500.0,  # pA
    },
    "original": {
        # Parameters from Brette & Gerstner (2005)
        "C_m": 281.0,  # pF
        "g_L": 30.0,  # nS
        "E_L": -70.6,  # mV
        "v_T": -50.4,  # mV
        "delta_T": 2.0,  # mV
        "v_reset": -70.6,  # mV
        "v_threshold": 20.0,  # mV
        "tau_w": 144.0,  # ms
        "a": 4.0,  # nS
        "b": 80.5,  # pA
        "I": 2500.0,  # pA
    },
}


# =============================================================================
# Default Parameters for Optimization
# =============================================================================

DEFAULT_PARAMS: AdExParams = {
    "C_m": 200.0,
    "g_L": 10.0,
    "E_L": -70.0,
    "v_T": -50.0,
    "delta_T": 2.0,
    "v_reset": -58.0,
    "v_threshold": 0.0,
    "tau_w": 30.0,
    "a": 2.0,
    "b": 0.0,
}


# =============================================================================
# Parameter Bounds for Optimization
# =============================================================================


@dataclass
class ParamBounds:
    """Bounds for a single parameter."""

    min: float
    max: float

    # TODO: think if this data class should do more work, e.g. is sigmoid inverse
    #  transformation reponsibility of this class?

    def clip(self, value: float) -> float:
        """Clip value to bounds."""
        return max(self.min, min(self.max, value))

    def __contains__(self, value: float) -> bool:
        """Check if value is within bounds."""
        return self.min <= value <= self.max


PARAM_BOUNDS: dict[str, ParamBounds] = {
    "C_m": ParamBounds(180.0, 220.0),
    "capacitance": ParamBounds(180.0, 220.0),  # alias for C_m (Jaxley compartment param)
    "g_L": ParamBounds(9.0, 13.0),
    "E_L": ParamBounds(-75.0, -68.0),
    "v_T": ParamBounds(-60.0, -30.0),
    "delta_T": ParamBounds(2, 20.0),
    "v_reset": ParamBounds(-60.0, -45.0),
    "v_threshold": ParamBounds(-20.0, 30.0),
    "tau_w": ParamBounds(20.0, 320.0),
    "a": ParamBounds(-10.0, 3.0),
    "b": ParamBounds(0.0, 65.0),
}


def clip_params(params: dict, bounds: dict[str, ParamBounds] | None = None) -> dict:
    """
    Clip parameters to their biophysically valid bounds.

    Args:
        params: Dictionary of parameter values
        bounds: Dictionary of ParamBounds. If None, uses PARAM_BOUNDS.

    Returns:
        Dictionary with clipped parameter values

    Example:
        >>> params = {'g_L': 100.0, 'E_L': -60.0}  # g_L out of bounds
        >>> clipped = clip_params(params)
        >>> print(clipped['g_L'])  # 50.0 (max bound)
    """
    if bounds is None:
        bounds = PARAM_BOUNDS

    clipped = {}
    for key, value in params.items():
        # Strip AdEx_ prefix if present
        bounds_key = key.replace("AdEx_", "")
        if bounds_key in bounds:
            clipped[key] = bounds[bounds_key].clip(value)
        else:
            clipped[key] = value

    return clipped


def convert_trainable_to_params(trainable_params: list[dict]) -> dict:
    """
    Convert Jaxley trainable parameters to standard format.

    Jaxley's get_parameters() returns a list of dicts with 'AdEx_*' keys.
    This function extracts the values and strips the prefix.

    Args:
        trainable_params: List of dicts from cell.get_parameters()

    Returns:
        Dictionary with standard parameter names and float values

    Example:
        >>> trainable = [{'AdEx_g_L': Array([10.5])}, {'AdEx_E_L': Array([-68.0])}]
        >>> params = convert_trainable_to_params(trainable)
        >>> print(params)  # {'g_L': 10.5, 'E_L': -68.0}
    """
    result = {}
    for param_dict in trainable_params:
        for name, value in param_dict.items():
            # Strip AdEx_ prefix
            clean_name = name.replace("AdEx_", "")
            if clean_name == "capacitance":
                clean_name = "C_m"
            # Extract scalar value
            result[clean_name] = float(value.flatten()[0])
    return result


def params_to_trainable_format(params: dict) -> list[dict]:
    """
    Convert standard parameters to Jaxley trainable format.

    Args:
        params: Dictionary of parameter values

    Returns:
        List of single-key dicts with 'AdEx_*' keys

    Example:
        >>> params = {'g_L': 10.0, 'E_L': -70.0}
        >>> trainable = params_to_trainable_format(params)
        >>> # [{'AdEx_g_L': 10.0}, {'AdEx_E_L': -70.0}]
    """
    import jax.numpy as jnp

    result = []
    for name, value in params.items():
        # Add AdEx_ prefix if not present
        if not name.startswith("AdEx_"):
            name = f"AdEx_{name}"
        result.append({name: jnp.array([value])})
    return result


# =============================================================================
# Experimental Data Paths
# =============================================================================

DATA_PATHS = {
    "mCP-dspn-e150917_c6_D1-manimal_1_n24_04102017_cel1": {
        "base": "expdata/",
        "traces": {
            "IV_499": ("ECBL_IV_ch5_499.dat", "ECBL_IV_ch4_499.dat"),
            "IV_502": ("ECBL_IV_ch5_502.dat", "ECBL_IV_ch4_502.dat"),
            "IDthresh_541": ("ECBL_IDthresh_ch5_541.dat", "ECBL_IDthresh_ch4_541.dat"),
            "IDthresh_543": ("ECBL_IDthresh_ch5_543.dat", "ECBL_IDthresh_ch4_543.dat"),
            "IDthresh_544": ("ECBL_IDthresh_ch5_544.dat", "ECBL_IDthresh_ch4_544.dat"),
            "IDthresh_553": ("ECBL_IDthresh_ch5_553.dat", "ECBL_IDthresh_ch4_553.dat"),
        },
    },
}
