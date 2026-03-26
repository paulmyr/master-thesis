"""
Benchmark system for systematic AdEx parameter optimization evaluation.

Provides scenario definitions, method configs, result storage, and execution
logic for comparing gradient-based and derivative-free optimization approaches.
"""

from .analysis import format_latex_table, print_table, recovery_table, summary_table
from .methods import MethodConfig, get_hp_sensitivity_methods, get_standard_methods, get_tonic_methods
from .results import RunResult, ScenarioResults, load_all_results, load_results, save_results
from .runner import (
    generate_initial_params,
    prepare_target_data,
    run_benchmark,
    run_scenario,
    run_single,
)
from .sampling import (
    TONIC_FIXED_PARAMS,
    TONIC_GROUND_TRUTHS,
    TONIC_TRAINABLE_PARAMS,
)
from .scenarios import (
    TONIC_PERTURBATIONS,
    SyntheticScenario,
    get_synthetic_scenarios,
    get_tonic_scenarios,
)

__all__ = [
    # sampling
    "TONIC_GROUND_TRUTHS",
    "TONIC_TRAINABLE_PARAMS",
    "TONIC_FIXED_PARAMS",
    # scenarios
    "SyntheticScenario",
    "get_synthetic_scenarios",
    "get_tonic_scenarios",
    "TONIC_PERTURBATIONS",
    # methods
    "MethodConfig",
    "get_standard_methods",
    "get_hp_sensitivity_methods",
    "get_tonic_methods",
    # results
    "RunResult",
    "ScenarioResults",
    "save_results",
    "load_results",
    "load_all_results",
    # runner
    "generate_initial_params",
    "prepare_target_data",
    "run_single",
    "run_scenario",
    "run_benchmark",
    # analysis
    "summary_table",
    "recovery_table",
    "print_table",
    "format_latex_table",
]
