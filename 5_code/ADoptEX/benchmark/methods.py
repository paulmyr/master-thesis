"""
Method definitions for benchmark runs.

Each method is a MethodConfig that specifies the optimization approach,
loss function, and hyperparameters. HP sensitivity variants are generated
programmatically as additional MethodConfig objects.
"""

from dataclasses import dataclass
from typing import Literal

from ADoptEX.loss.guarino import GuarinoLossConfig
from ADoptEX.loss.mse import MSELossConfig
from ADoptEX.loss.van_rossum import VanRossumLossConfig
from ADoptEX.training.trainer import TrainingConfig


@dataclass
class MethodConfig:
    """Configuration for a single optimization method."""

    name: str
    method_type: Literal["gradient", "nelder_mead"]

    # Gradient-specific
    training_config: TrainingConfig | None = None
    loss_type: Literal["van_rossum", "guarino", "mse"] | None = None
    loss_config: VanRossumLossConfig | GuarinoLossConfig | MSELossConfig | None = None

    # Nelder-Mead specific
    nm_max_fev: int = 2000
    nm_objective: Literal["van_rossum", "guarino", "mse"] = "van_rossum"
    nm_loss_config: VanRossumLossConfig | GuarinoLossConfig | MSELossConfig | None = (
        None
    )

    # Common
    surrogate_type: Literal["sigmoid", "exponential", "superspike"] = "sigmoid"
    surrogate_slope: float = 5.0


def get_standard_methods() -> list[MethodConfig]:
    """Return the 4 headline methods for benchmarking.

    - grad_vanrossum: Polyak + Van Rossum (tau=10)
    - grad_guarino: Polyak + Guarino feature loss
    - grad_mse: Adam + MSE voltage loss
    - nelder_mead: Nelder-Mead + Van Rossum objective
    """
    return [
        MethodConfig(
            name="grad_vanrossum",
            method_type="gradient",
            loss_type="van_rossum",
            loss_config=VanRossumLossConfig(tau_ms=15.0, weight_subthreshold=0.5, subthreshold_clamp_mv=-40),
            training_config=TrainingConfig(
                optimizer="polyak",
                learning_rate=0.005,
                n_epochs=50,
                print_every=10,
                surrogate_type="sigmoid",
                surrogate_slope=2.0,
                use_param_transform=True,
                clip_to_bounds=False,
                return_best=True,
                verbose=False,
            ),
        ),
        MethodConfig(
            name="grad_guarino",
            method_type="gradient",
            loss_type="guarino",
            loss_config=GuarinoLossConfig(),
            training_config=TrainingConfig(
                optimizer="polyak",
                learning_rate=0.005,
                n_epochs=50,
                print_every=10,
                surrogate_type="sigmoid",
                surrogate_slope=2.0,
                polyak_alpha=1.0,
                polyak_beta=0.85,
                use_param_transform=True,
                clip_to_bounds=False,
                return_best=True,
                verbose=False,
            ),
        ),
#        MethodConfig(
#            name="grad_mse",
#            method_type="gradient",
#            loss_type="mse",
#            loss_config=MSELossConfig(),
#            training_config=TrainingConfig(
#                optimizer="adam",
#                learning_rate=0.01,
#                n_epochs=50,
#                print_every=10,
#                surrogate_type="sigmoid",
#                surrogate_slope=2.0,
#                use_param_transform=True,
#                clip_to_bounds=False,
#                return_best=True,
#                verbose=False,
#            ),
#        ),
        MethodConfig(
            name="nelder_mead",
            method_type="nelder_mead",
            nm_max_fev=50,
            nm_objective="van_rossum",
            nm_loss_config=VanRossumLossConfig(tau_ms=10.0),
            surrogate_slope=5.0,
        ),
    ]


def get_hp_sensitivity_methods() -> list[MethodConfig]:
    """Generate HP sensitivity variants for Van Rossum gradient method.

    Varies: tau, surrogate slope, learning rate, optimizer.
    Returns additional MethodConfig objects (not including the headline methods).
    """
    variants = []

    # Van Rossum tau variants
    for tau in [5.0, 20.0, 50.0]:
        variants.append(
            MethodConfig(
                name=f"grad_vr_tau{int(tau)}",
                method_type="gradient",
                loss_type="van_rossum",
                loss_config=VanRossumLossConfig(tau_ms=tau),
                training_config=TrainingConfig(
                    optimizer="polyak",
                    learning_rate=0.01,
                    n_epochs=500,
                    surrogate_type="sigmoid",
                    surrogate_slope=5.0,
                    use_param_transform=True,
                    clip_to_bounds=True,
                    return_best=True,
                    verbose=False,
                ),
            )
        )

    # Surrogate slope variants
    for slope in [2.0, 10.0]:
        variants.append(
            MethodConfig(
                name=f"grad_vr_slope{int(slope)}",
                method_type="gradient",
                loss_type="van_rossum",
                loss_config=VanRossumLossConfig(tau_ms=10.0),
                training_config=TrainingConfig(
                    optimizer="polyak",
                    learning_rate=0.01,
                    n_epochs=500,
                    surrogate_type="sigmoid",
                    surrogate_slope=slope,
                    use_param_transform=True,
                    clip_to_bounds=True,
                    return_best=True,
                    verbose=False,
                ),
            )
        )

    # Learning rate variants
    for lr in [0.005, 0.05]:
        variants.append(
            MethodConfig(
                name=f"grad_vr_lr{lr}",
                method_type="gradient",
                loss_type="van_rossum",
                loss_config=VanRossumLossConfig(tau_ms=10.0),
                training_config=TrainingConfig(
                    optimizer="polyak",
                    learning_rate=lr,
                    n_epochs=500,
                    surrogate_type="sigmoid",
                    surrogate_slope=5.0,
                    use_param_transform=True,
                    clip_to_bounds=True,
                    return_best=True,
                    verbose=False,
                ),
            )
        )

    # Adam variant (instead of Polyak)
    variants.append(
        MethodConfig(
            name="grad_vr_adam",
            method_type="gradient",
            loss_type="van_rossum",
            loss_config=VanRossumLossConfig(tau_ms=10.0),
            training_config=TrainingConfig(
                optimizer="adam",
                learning_rate=0.01,
                n_epochs=500,
                surrogate_type="sigmoid",
                surrogate_slope=5.0,
                use_param_transform=True,
                clip_to_bounds=True,
                return_best=True,
                verbose=False,
            ),
        )
    )

    return variants


def get_single_param_methods() -> list[MethodConfig]:
    """Return methods for the single-parameter convergence benchmark.

    Same as tonic methods but with fewer epochs/fev since 1-param
    optimization should converge faster.

    - grad_vanrossum: Polyak + Van Rossum, 50 epochs
    - grad_guarino: Polyak + Guarino feature loss, 50 epochs
    - nelder_mead: Nelder-Mead + Van Rossum, 500 max_fev
    """
    return [
        MethodConfig(
            name="grad_vanrossum",
            method_type="gradient",
            loss_type="van_rossum",
            loss_config=VanRossumLossConfig(tau_ms=10.0),
            training_config=TrainingConfig(
                optimizer="polyak",
                learning_rate=0.01,
                n_epochs=50,
                print_every=10,
                surrogate_type="sigmoid",
                surrogate_slope=2.0,
                use_param_transform=True,
                clip_to_bounds=False,
                return_best=True,
                verbose=False,
            ),
        ),
        MethodConfig(
            name="grad_guarino",
            method_type="gradient",
            loss_type="guarino",
            loss_config=GuarinoLossConfig(),
            training_config=TrainingConfig(
                optimizer="polyak",
                learning_rate=0.01,
                n_epochs=50,
                print_every=10,
                surrogate_type="sigmoid",
                surrogate_slope=2.0,
                use_param_transform=True,
                clip_to_bounds=False,
                return_best=True,
                verbose=False,
            ),
        ),
        MethodConfig(
            name="nelder_mead",
            method_type="nelder_mead",
            nm_max_fev=500,
            nm_objective="van_rossum",
            nm_loss_config=VanRossumLossConfig(tau_ms=10.0),
            surrogate_slope=5.0,
        ),
    ]


def get_tonic_methods() -> list[MethodConfig]:
    """Return the 3 methods for the tonic convergence benchmark.

    - grad_vanrossum: Polyak + Van Rossum (tau=10), 200 epochs
    - grad_guarino: Polyak + Guarino feature loss, 200 epochs
    - nelder_mead: Nelder-Mead + Van Rossum objective, 2000 max_fev
    """
    return [
        MethodConfig(
            name="grad_vanrossum",
            method_type="gradient",
            loss_type="van_rossum",
            loss_config=VanRossumLossConfig(tau_ms=10.0),
            training_config=TrainingConfig(
                optimizer="polyak",
                learning_rate=0.01,
                n_epochs=50,
                print_every=10,
                surrogate_type="sigmoid",
                surrogate_slope=2.0,
                use_param_transform=True,
                clip_to_bounds=False,
                return_best=True,
                verbose=False,
            ),
        ),
        MethodConfig(
            name="grad_guarino",
            method_type="gradient",
            loss_type="guarino",
            loss_config=GuarinoLossConfig(),
            training_config=TrainingConfig(
                optimizer="polyak",
                learning_rate=0.05,
                n_epochs=200,
                print_every=50,
                surrogate_type="sigmoid",
                polyak_alpha=1.0,
                polyak_beta=0.85,
                surrogate_slope=2.0,
                use_param_transform=True,
                clip_to_bounds=False,
                return_best=True,
                verbose=False,
            ),
        ),
        MethodConfig(
            name="nelder_mead_vanrossum",
            method_type="nelder_mead",
            nm_max_fev=2000,
            nm_objective="van_rossum",
            nm_loss_config=VanRossumLossConfig(tau_ms=10.0),
            surrogate_slope=5.0,
        ),
        MethodConfig(
            name="nelder_mead_guarino",
            method_type="nelder_mead",
            nm_max_fev=2000,
            nm_objective="guarino",
            nm_loss_config=GuarinoLossConfig(weight_spike_count=0.2),
            surrogate_slope=5.0,
        ),
    ]
