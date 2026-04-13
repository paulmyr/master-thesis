# Benchmark System for ADoptEX

## Context

Gradient-based optimization recovers membrane parameters (g_L, E_L, v_T, v_reset, delta_T, C_m) but NOT adaptation parameters (a, b, tau_w). Nelder-Mead also fails on adaptation params. The user needs a systematic benchmarking system to:
1. Quantify what works and what doesn't across firing regimes and methods
2. Compare gradient-based vs derivative-free approaches fairly
3. Test across a variety of parameters and initializations
4. Produce thesis-quality figures and tables

Current state: all experiments are ad-hoc in notebooks, results never saved to disk, no structured evaluation.

## Architecture Overview

```
ADoptEX/benchmark/          <- library module (scenarios, methods, runner, results, analysis)
benchmarks/                 <- runner scripts, configs, results dir, analysis notebook
    run_benchmark.py        <- CLI entry point
    run_hp_sensitivity.py   <- HP grid search script
    results/                <- JSON output (gitignored)
    analysis.ipynb          <- loads results, produces thesis figures
```

**Separation of concerns:**
- Library (`ADoptEX/benchmark/`) defines types and execution logic
- Scripts (`benchmarks/`) orchestrate and configure
- Results stored as JSON files (one per scenario x method)
- Analysis notebook loads results for visualization

## File Plan

### 1. `ADoptEX/benchmark/scenarios.py` — Scenario Definitions

Two scenario types:

**SyntheticScenario** — known ground truth, simulated target:
- `name`, `ground_truth_params`, `trainable_params`
- `stimulus_source`: "experimental" (reuse real current waveform) or "step"
- `perturbation`: relative noise for initial params (e.g., 0.15)
- `n_starts`: number of random initializations
- `max_duration_ms`, `dt_ms`, `seed`

**ExperimentalScenario** — real voltage trace, no ground truth:
- `name`, `cell_id`, `trace_id`, `voltage_path`, `current_path`
- `trainable_params`, `initial_params` (starting point, e.g., NAUD tonic)
- `n_starts`, `max_duration_ms`, `dt_ms`, `seed`

**Registry functions:**
- `get_synthetic_scenarios()` — returns standard suite
- `get_experimental_scenarios()` — returns standard suite

**Synthetic scenario matrix (5 patterns x 2 param groups x 2 perturbations = 20 scenarios):**

| Pattern | Key params | Source |
|---------|-----------|--------|
| Tonic spiking | b=0, tau_w=30, a=2 | NAUD existing |
| Adaptation | b=60, tau_w=300, a=2 | NAUD existing |
| Initial bursting | b=100, tau_w=50, a=-5 | Naud 2008 (NEW) |
| Regular bursting | b=100, tau_w=300, a=-5 | Naud 2008 (NEW) |
| Delayed accelerating | b=0, tau_w=300, a=-2 | Naud 2008 (NEW) |

Each pattern tested with:
- **membrane_only**: trainable = [C_m, g_L, E_L, v_T, delta_T, v_reset]
- **full**: trainable = [C_m, g_L, E_L, v_T, delta_T, v_reset, a, b, tau_w]
- Perturbation: 15% and 25%

All synthetic scenarios use the experimental current waveform (IDthresh_553) for realism. Ground truth generated with `use_surrogate=True` to match training forward pass.

**Experimental scenarios** — 3-4 curated IDthresh traces from primary cell, selected for different spike counts (e.g., 2, 5, 10+ spikes).

### 2. `ADoptEX/benchmark/methods.py` — Method Definitions

**MethodConfig dataclass:**
- `name`, `method_type` ("gradient" | "nelder_mead")
- Gradient: `training_config` (TrainingConfig), `loss_type`, `loss_config`
- NM: `nm_max_fev`, `nm_objective`, `nm_loss_config`
- Common: `surrogate_type`, `surrogate_slope`

**Standard methods (4 headline + HP variants):**

| Method | Type | Loss | Key HPs |
|--------|------|------|---------|
| `grad_vanrossum` | gradient | Van Rossum (tau=10) | Polyak, slope=5, sigmoid transform |
| `grad_guarino` | gradient | Guarino | Polyak, slope=5, sigmoid transform |
| `nelder_mead` | nelder_mead | Van Rossum (tau=10) | max_fev=2000, adaptive |

**HP sensitivity variants** — additional MethodConfigs generated programmatically:
- Van Rossum tau: {5, 10, 20, 50} ms
- Surrogate slope: {2, 5, 10}
- Learning rate: {0.005, 0.01, 0.05}
- Optimizer: {polyak}

These are just more MethodConfig objects fed through the same runner. No separate system needed.

### 3. `ADoptEX/benchmark/results.py` — Result Storage

**RunResult dataclass** (one per optimization run):
- Identity: `scenario_name`, `method_name`, `start_index`, `seed`
- Params: `initial_params`, `final_params`, `ground_truth_params` (None for experimental)
- Metrics: `final_loss`, `best_loss`, `gamma`, `n_spikes_model`, `n_spikes_target`, `spike_count_error`, `first_spike_error_ms`
- Recovery: `param_recovery` dict — per-param `{error_abs, error_pct}` (synthetic only)
- Timing: `wall_time_s`, `n_function_evals`, `time_per_eval_ms`
- Diagnostics: `loss_history`, `grad_norms` (gradient methods only)
- Metadata: `timestamp`
- Methods: `to_dict()`, `from_dict()`

**ScenarioResults** (all runs for one scenario x method):
- `scenario_name`, `method_name`, `runs: list[RunResult]`
- Properties: `best_run`, `mean_gamma`, `best_gamma`
- `summary_dict()` for aggregation

**I/O functions:**
- `save_results(ScenarioResults, output_dir)` → writes `{scenario}__{method}.json`
- `load_results(path)` → ScenarioResults
- `load_all_results(results_dir)` → list[ScenarioResults]

One JSON file per (scenario, method) pair. ~80 files for full benchmark. Supports incremental/resumable execution.

### 4. `ADoptEX/benchmark/runner.py` — Execution Engine

**Core functions:**
- `generate_initial_params(scenario, start_index)` — Latin Hypercube Sampling via `scipy.stats.qmc.LatinHypercube`, reproducible using `seed + start_index`
- `prepare_target_data(scenario)` — simulate ground truth (synthetic) or load trace (experimental)
- `run_gradient(scenario, method, initial_params, target_data)` → RunResult
  - Uses existing `setup_trainable_cell()`, `make_*_loss_fn()`, `train()`
  - Evaluates with `simulate_with_current_trace()` + `coincidence_factor()`
- `run_nelder_mead(scenario, method, initial_params, target_data)` → RunResult
  - Uses `scipy.optimize.minimize` with Van Rossum or Gamma objective
- `run_single(scenario, method, start_index)` → RunResult
- `run_scenario(scenario, method, output_dir)` → ScenarioResults (all starts)
- `run_benchmark(scenarios, methods, output_dir, resume=True)` → full suite

**Important**: runner uses updated API (`current_trace_pA`, not the old `current_trace_nA * C_m` pattern).

### 5. `ADoptEX/benchmark/analysis.py` — Result Analysis

- `summary_table(results)` → list of flat dicts (scenario, method, metrics)
- `recovery_table(results, param_names)` → per-parameter recovery stats
- `print_table(rows)` — formatted console output
- `format_latex_table(rows, caption)` — LaTeX tabular for thesis

### 6. `benchmarks/run_benchmark.py` — CLI Entry Point

```
python run_benchmark.py                              # full suite
python run_benchmark.py --scenario tonic_15pct_full  # single scenario
python run_benchmark.py --method grad_vanrossum      # single method
python run_benchmark.py --dry-run                    # show what would run
python run_benchmark.py --resume                     # skip existing results
python run_benchmark.py --n-starts 5                 # override start count
```

### 7. `benchmarks/run_hp_sensitivity.py` — HP Grid Search

Generates MethodConfig variants programmatically and feeds them through the same `run_benchmark()`. Results in `results/hp_sensitivity/` subdirectory.

### 8. `benchmarks/analysis.ipynb` — Analysis Notebook

Loads results, produces thesis figures:
1. Summary heatmap (Gamma by scenario x method)
2. Per-parameter recovery bar charts (membrane vs adaptation)
3. Loss convergence comparison
4. Multi-start distribution (box plots of Gamma)
5. Wall time comparison
6. Firing pattern difficulty ranking
7. LaTeX tables for thesis

### 9. Supporting Changes

**`ADoptEX/core/parameters.py`** — Add 3 new NAUD patterns:
- `initial_bursting`, `regular_bursting`, `delayed_accelerating`
- Requires careful selection of params within PARAM_BOUNDS

**`ADoptEX/benchmark/__init__.py`** — Re-exports

**`benchmarks/results/.gitignore`** — Ignore JSON results

### 10. Tests — `ADoptEX/tests/test_benchmark/`

- `test_scenarios.py` — scenario generation, parameter validation
- `test_results.py` — serialization roundtrip (to_dict/from_dict, save/load)
- `test_methods.py` — method config validation
- `test_runner.py` — `generate_initial_params` reproducibility, metric computation (mark `@slow` for integration)

## Implementation Order

1. **Phase 1**: `results.py` (RunResult, ScenarioResults, JSON I/O) — foundation
2. **Phase 2**: `scenarios.py` + update `parameters.py` with new Naud patterns
3. **Phase 3**: `methods.py` (MethodConfig, standard methods)
4. **Phase 4**: `runner.py` (core execution logic)
5. **Phase 5**: `run_benchmark.py` CLI + `analysis.py` helpers
6. **Phase 6**: Tests
7. **Phase 7**: Analysis notebook + doc sync

## Decisions (confirmed)

- **10 starts** per scenario (good balance of statistics vs runtime)
- **Latin Hypercube Sampling** via `scipy.stats.qmc.LatinHypercube` (space-filling, better coverage with fewer samples)
- **HP variants as methods** — each (loss x slope x lr) combo is a MethodConfig, all go through same runner, analysis groups them
- **3-4 curated IDthresh traces** from primary cell, selected for different spike counts (e.g., 2, 5, 10+ spikes)
- **NM gets same 10 starts** but may take ~30 min per scenario. Run on subset or overnight.

## Runtime Estimate

- Synthetic: 20 scenarios x 4 headline methods x 10 starts = 800 gradient runs (~1-5s each = ~30-60 min) + 200 NM runs (~200s each = ~11 hours)
- Experimental: 4 scenarios x 4 methods x 10 starts = 160 runs (similar proportions)
- HP sensitivity: ~30 HP variants x 2-3 representative scenarios x 10 starts = 600-900 gradient runs (~30-60 min)
- **Total gradient: ~2-3 hours. Total NM: ~15 hours (can run overnight or on cluster).**
- Resume support means runs can be split across sessions.

## Key Reused Components

| Existing function | File | Used by runner for |
|---|---|---|
| `setup_trainable_cell()` | `training/trainer.py` | Cell creation + trainable params |
| `train()` | `training/trainer.py` | Gradient optimization loop |
| `make_guarino_loss_fn()` | `loss/guarino.py` | Guarino loss closure |
| `make_van_rossum_loss_fn()` | `loss/van_rossum.py` | Van Rossum loss closure |
| `make_mse_loss_fn()` | `loss/mse.py` | MSE loss closure |
| `simulate_with_current_trace()` | `core/simulation.py` | Ground truth + evaluation sims |
| `coincidence_factor()` | `evaluation/coincidence.py` | Gamma metric |
| `load_trace()`, `crop_to_stim_window()` | `core/data.py` | Experimental data loading |
| `NAUD_PARAMETERS`, `PARAM_BOUNDS` | `core/parameters.py` | Scenario definitions |
| `build_param_transform()`, `nudge_from_bounds()` | `training/trainer.py` | Sigmoid reparameterization |
| `scipy.stats.qmc.LatinHypercube` | scipy (already installed) | Multi-start initialization |
| `scipy.optimize.minimize` | scipy | Nelder-Mead baseline |