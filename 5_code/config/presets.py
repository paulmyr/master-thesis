"""
Preset configurations for common experiment types.

These presets provide sensible defaults for different use cases:
- quick_test: Fast iteration for debugging (50 epochs)
- standard: Standard training run (500 epochs)
- thorough: Thorough optimization (2000 epochs)
- compare_surrogates: Compare different surrogate gradient types
"""

from .experiment import ExperimentConfig, GuarinoWeights

# Quick test preset - fast iteration for debugging
QUICK_TEST = ExperimentConfig(
    name="quick_test",
    loss_type="guarino",
    surrogate_type="sigmoid",
    surrogate_slope=25.0,
    temperature=0.1,
    beta=10.0,
    optimizer="adam",
    learning_rate=0.1,
    n_epochs=50,
    print_every=10,
    clip_to_bounds=True,
    trainable_params=["g_L", "E_L", "v_T", "v_reset", "tau_w", "a", "b"],
)

# Standard preset - balanced speed and quality
STANDARD = ExperimentConfig(
    name="standard",
    loss_type="guarino",
    surrogate_type="sigmoid",
    surrogate_slope=25.0,
    temperature=0.1,
    beta=10.0,
    optimizer="adam",
    learning_rate=0.1,
    n_epochs=500,
    print_every=50,
    clip_to_bounds=True,
    trainable_params=["g_L", "E_L", "v_T", "v_reset", "tau_w", "a", "b"],
)

# Thorough preset - longer training for best results
THOROUGH = ExperimentConfig(
    name="thorough",
    loss_type="guarino",
    surrogate_type="sigmoid",
    surrogate_slope=25.0,
    temperature=0.1,
    beta=10.0,
    optimizer="adam",
    learning_rate=0.05,  # Lower LR for stability
    n_epochs=2000,
    print_every=100,
    clip_to_bounds=True,
    trainable_params=["g_L", "E_L", "v_T", "v_reset", "tau_w", "a", "b"],
)

# MSE loss preset
MSE_STANDARD = ExperimentConfig(
    name="mse_standard",
    loss_type="mse",
    surrogate_type="sigmoid",
    surrogate_slope=25.0,
    optimizer="adam",
    learning_rate=0.1,
    n_epochs=500,
    print_every=50,
    clip_to_bounds=True,
    trainable_params=["g_L", "E_L", "v_T", "v_reset", "tau_w", "a", "b"],
)

# Superspike surrogate preset
SUPERSPIKE = ExperimentConfig(
    name="superspike",
    loss_type="guarino",
    surrogate_type="superspike",
    surrogate_slope=10.0,  # Different optimal slope for superspike
    temperature=0.1,
    beta=10.0,
    optimizer="adam",
    learning_rate=0.1,
    n_epochs=500,
    print_every=50,
    clip_to_bounds=True,
    trainable_params=["g_L", "E_L", "v_T", "v_reset", "tau_w", "a", "b"],
)

# Exponential surrogate preset
EXPONENTIAL = ExperimentConfig(
    name="exponential",
    loss_type="guarino",
    surrogate_type="exponential",
    surrogate_slope=5.0,  # Different optimal slope for exponential
    temperature=0.1,
    beta=10.0,
    optimizer="adam",
    learning_rate=0.1,
    n_epochs=500,
    print_every=50,
    clip_to_bounds=True,
    trainable_params=["g_L", "E_L", "v_T", "v_reset", "tau_w", "a", "b"],
)

# High frequency weight preset (emphasize spike count/frequency)
HIGH_FREQ_WEIGHT = ExperimentConfig(
    name="high_freq_weight",
    loss_type="guarino",
    guarino_weights=GuarinoWeights(
        weight_t_first=1.0,
        weight_t_second=1.0,
        weight_t_third=1.0,
        weight_t_last=1.0,
        weight_inv_first_isi=1.0,
        weight_inv_last_isi=1.0,
        weight_firing_freq=5.0,  # Higher weight on frequency
        weight_v_stim_end=0.5,
    ),
    surrogate_type="sigmoid",
    surrogate_slope=25.0,
    temperature=0.1,
    beta=10.0,
    optimizer="adam",
    learning_rate=0.1,
    n_epochs=500,
    print_every=50,
    clip_to_bounds=True,
    trainable_params=["g_L", "E_L", "v_T", "v_reset", "tau_w", "a", "b"],
)

# High timing weight preset (emphasize spike timing)
HIGH_TIMING_WEIGHT = ExperimentConfig(
    name="high_timing_weight",
    loss_type="guarino",
    guarino_weights=GuarinoWeights(
        weight_t_first=3.0,  # Higher weight on timing
        weight_t_second=3.0,
        weight_t_third=2.0,
        weight_t_last=2.0,
        weight_inv_first_isi=2.0,
        weight_inv_last_isi=2.0,
        weight_firing_freq=1.0,
        weight_v_stim_end=0.5,
    ),
    surrogate_type="sigmoid",
    surrogate_slope=25.0,
    temperature=0.1,
    beta=10.0,
    optimizer="adam",
    learning_rate=0.1,
    n_epochs=500,
    print_every=50,
    clip_to_bounds=True,
    trainable_params=["g_L", "E_L", "v_T", "v_reset", "tau_w", "a", "b"],
)


# Soft-DTW + MAE loss preset
SOFT_DTW = ExperimentConfig(
    name="soft_dtw",
    loss_type="soft_dtw",
    surrogate_type="sigmoid",
    surrogate_slope=25.0,
    optimizer="adam",
    learning_rate=0.1,
    n_epochs=500,
    print_every=50,
    clip_to_bounds=True,
    trainable_params=["g_L", "E_L", "v_T", "v_reset", "tau_w", "a", "b"],
)

# Dictionary of all presets
PRESETS: dict[str, ExperimentConfig] = {
    "quick_test": QUICK_TEST,
    "standard": STANDARD,
    "thorough": THOROUGH,
    "mse_standard": MSE_STANDARD,
    "superspike": SUPERSPIKE,
    "exponential": EXPONENTIAL,
    "high_freq_weight": HIGH_FREQ_WEIGHT,
    "high_timing_weight": HIGH_TIMING_WEIGHT,
    "soft_dtw": SOFT_DTW,
}


def get_preset(name: str) -> ExperimentConfig:
    """
    Get a preset configuration by name.

    Args:
        name: Name of the preset (e.g., "standard", "quick_test")

    Returns:
        A copy of the preset configuration

    Raises:
        KeyError: If preset name is not found

    Example:
        >>> config = get_preset("standard")
        >>> config.n_epochs = 1000  # Modify without affecting preset
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")

    # Return a copy so modifications don't affect the preset
    import copy

    return copy.deepcopy(PRESETS[name])


def list_presets() -> list[str]:
    """Get list of available preset names."""
    return list(PRESETS.keys())


def describe_presets() -> str:
    """Get descriptions of all presets."""
    descriptions = []
    for name, config in PRESETS.items():
        desc = f"{name}:\n"
        desc += f"  Loss: {config.loss_type}\n"
        desc += (
            f"  Surrogate: {config.surrogate_type} (slope={config.surrogate_slope})\n"
        )
        desc += f"  Optimizer: {config.optimizer} (lr={config.learning_rate})\n"
        desc += f"  Epochs: {config.n_epochs}\n"
        descriptions.append(desc)
    return "\n".join(descriptions)
