"""
Scenario definitions for benchmark runs.

SyntheticScenario: known ground truth with step current stimulation.
Each scenario specifies the injected current amplitude, duration, and delay.
"""

from dataclasses import dataclass

from ADoptEX.core.parameters import PARAM_BOUNDS

# Standard trainable parameter groups
MEMBRANE_PARAMS = ["C_m", "g_L", "E_L", "v_T", "delta_T", "v_reset"]
FULL_PARAMS = ["C_m", "g_L", "E_L", "v_T", "delta_T", "v_reset", "a", "b", "tau_w"]


@dataclass
class SyntheticScenario:
    """Scenario with known ground truth parameters and step current stimulus."""

    name: str
    ground_truth_params: dict[str, float]
    trainable_params: list[str]

    # Stimulus
    stim_current_pA: float = 500.0
    stim_duration_ms: float = 400.0
    stim_delay_ms: float = 50.0

    # Optimization
    perturbation: float = 0.15  # relative perturbation for initial params
    n_starts: int = 10
    dt_ms: float = 0.1
    seed: int = 42

    @property
    def t_max_ms(self) -> float:
        """Total simulation time including post-stimulus window."""
        return self.stim_delay_ms + self.stim_duration_ms + 50.0

    def __post_init__(self):
        # Validate all trainable params have bounds
        for p in self.trainable_params:
            if p not in PARAM_BOUNDS:
                raise ValueError(f"No bounds defined for trainable param '{p}'")
        # Validate ground truth within bounds
        for p in self.trainable_params:
            if p in self.ground_truth_params:
                bounds = PARAM_BOUNDS[p]
                val = self.ground_truth_params[p]
                if val < bounds.min or val > bounds.max:
                    raise ValueError(
                        f"Ground truth {p}={val} outside bounds [{bounds.min}, {bounds.max}]"
                    )


# ── Synthetic Scenario Registry ─────────────────────────────────────────────

# 5 firing patterns from Naud et al. (2008)
# Each includes the stimulation current I that produces the named pattern.
_FIRING_PATTERNS: dict[str, dict[str, dict[str, float] | float]] = {
    "tonic": {
        "params": {
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
        },
        "I": 500.0,
    },
    "adaptation": {
        "params": {
            "C_m": 200.0,
            "g_L": 12.0,
            "E_L": -70.0,
            "v_T": -50.0,
            "delta_T": 2.0,
            "v_reset": -58.0,
            "v_threshold": 0.0,
            "tau_w": 300.0,
            "a": 2.0,
            "b": 60.0,
        },
        "I": 500.0,
    },
    "initial_bursting": {
        "params": {
            "C_m": 200.0,
            "g_L": 10.0,
            "E_L": -70.0,
            "v_T": -50.0,
            "delta_T": 2.0,
            "v_reset": -58.0,
            "v_threshold": 0.0,
            "tau_w": 50.0,
            "a": -5.0,
            "b": 60.0,
        },
        "I": 500.0,
    },
    "regular_bursting": {
        "params": {
            "C_m": 200.0,
            "g_L": 10.0,
            "E_L": -70.0,
            "v_T": -50.0,
            "delta_T": 2.0,
            "v_reset": -58.0,
            "v_threshold": 0.0,
            "tau_w": 300.0,
            "a": -5.0,
            "b": 60.0,
        },
        "I": 500.0,
    },
    "delayed_accelerating": {
        "params": {
            "C_m": 200.0,
            "g_L": 10.0,
            "E_L": -70.0,
            "v_T": -50.0,
            "delta_T": 2.0,
            "v_reset": -58.0,
            "v_threshold": 0.0,
            "tau_w": 300.0,
            "a": -2.0,
            "b": 0.0,
        },
        "I": 500.0,
    },
}


def get_synthetic_scenarios(
    n_starts: int = 10,
    seed: int = 42,
) -> list[SyntheticScenario]:
    """Generate the standard synthetic scenario matrix.

    5 patterns x 2 param groups x 2 perturbations = 20 scenarios.

    Args:
        n_starts: Number of random initializations per scenario.
        seed: Base random seed.

    Returns:
        List of SyntheticScenario objects.
    """
    scenarios = []
    for pattern_name, pattern in _FIRING_PATTERNS.items():
        for group_name, trainable in [
            ("membrane", MEMBRANE_PARAMS),
            ("full", FULL_PARAMS),
        ]:
            for perturbation in [0.15, 0.25]:
                pct = int(perturbation * 100)
                name = f"{pattern_name}_{pct}pct_{group_name}"
                params = pattern["params"]
                assert isinstance(params, dict)
                stim_I = pattern["I"]
                assert isinstance(stim_I, float)
                scenarios.append(
                    SyntheticScenario(
                        name=name,
                        ground_truth_params=params.copy(),
                        trainable_params=trainable.copy(),
                        stim_current_pA=stim_I,
                        perturbation=perturbation,
                        n_starts=n_starts,
                        seed=seed,
                    )
                )
    return scenarios


# ── Tonic Benchmark (Q1: Convergence Reliability) ─────────────────────────

TONIC_PERTURBATIONS: list[float] = [0.01, 0.03, 0.05, 0.10, 0.15, 0.25]


# ── Single-Parameter Benchmark (Q: Per-Parameter Convergence) ───────────────

SINGLE_PARAM_PERTURBATIONS: list[float] = [0.05, 0.10, 0.15, 0.25, 0.50]


def get_single_param_scenarios(
    n_starts: int = 10,
    seed: int = 42,
    perturbations: list[float] | None = None,
    ground_truths: list[dict] | None = None,
    param_names: list[str] | None = None,
) -> list[SyntheticScenario]:
    """Generate scenarios that optimize one parameter at a time.

    Tests each trainable parameter in isolation to identify which parameters
    converge easily and which are difficult.

    Args:
        n_starts: Random initializations per scenario.
        seed: Base seed for initial param generation.
        perturbations: Perturbation fractions. Default: SINGLE_PARAM_PERTURBATIONS.
        ground_truths: Ground truth dicts. Default: TONIC_GROUND_TRUTHS.
        param_names: Which parameters to test individually.
            Default: MEMBRANE_PARAMS.

    Returns:
        List of SyntheticScenario objects.
    """
    if perturbations is None:
        perturbations = list(SINGLE_PARAM_PERTURBATIONS)

    if ground_truths is None:
        from .sampling import TONIC_GROUND_TRUTHS

        ground_truths = TONIC_GROUND_TRUTHS

    if param_names is None:
        param_names = list(MEMBRANE_PARAMS)

    scenarios = []
    for param in param_names:
        for i, gt in enumerate(ground_truths):
            for perturbation in perturbations:
                pct = int(perturbation * 100)
                name = f"single_{param}_gt{i:02d}_{pct}pct"
                scenarios.append(
                    SyntheticScenario(
                        name=name,
                        ground_truth_params=gt["params"].copy(),
                        trainable_params=[param],
                        stim_current_pA=gt["stim_current_pA"],
                        perturbation=perturbation,
                        n_starts=n_starts,
                        seed=seed,
                    )
                )

    return scenarios


def get_tonic_scenarios(
    n_starts: int = 10,
    seed: int = 42,
    perturbations: list[float] | None = None,
    ground_truths: list[dict] | None = None,
) -> list[SyntheticScenario]:
    """Generate tonic convergence benchmark scenarios.

    20 random ground truths x 6 perturbation levels = 120 scenarios.
    All tonic firing, membrane params only.

    Args:
        n_starts: Random initializations per scenario.
        seed: Base seed for initial param generation.
        perturbations: Perturbation fractions. Default: TONIC_PERTURBATIONS.
        ground_truths: Ground truth dicts (from TONIC_GROUND_TRUTHS or
            sample_tonic_ground_truths). Default: TONIC_GROUND_TRUTHS.

    Returns:
        List of SyntheticScenario objects.
    """
    if perturbations is None:
        perturbations = list(TONIC_PERTURBATIONS)

    if ground_truths is None:
        from .sampling import TONIC_GROUND_TRUTHS

        ground_truths = TONIC_GROUND_TRUTHS

    scenarios = []
    for i, gt in enumerate(ground_truths):
        for perturbation in perturbations:
            pct = int(perturbation * 100)
            name = f"tonic_gt{i:02d}_{pct}pct_membrane"
            scenarios.append(
                SyntheticScenario(
                    name=name,
                    ground_truth_params=gt["params"].copy(),
                    trainable_params=list(MEMBRANE_PARAMS),
                    stim_current_pA=gt["stim_current_pA"],
                    perturbation=perturbation,
                    n_starts=n_starts,
                    seed=seed,
                )
            )

    return scenarios
