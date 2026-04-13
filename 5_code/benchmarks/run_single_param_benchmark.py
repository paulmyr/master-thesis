#!/usr/bin/env python3
"""
CLI entry point for the single-parameter convergence benchmark.

Tests each trainable parameter in isolation to identify which parameters
converge easily and which are difficult for gradient-based optimization.

6 params x 20 GTs x 5 perturbations x 3 methods x 10 starts = 18,000 total runs.
(Each run optimizes only 1 parameter, so individual runs are fast.)

Usage:
    python run_single_param_benchmark.py                        # full suite
    python run_single_param_benchmark.py --dry-run              # show what would run
    python run_single_param_benchmark.py --param g_L            # single parameter
    python run_single_param_benchmark.py --gt-index 0           # single ground truth
    python run_single_param_benchmark.py --perturbation 0.15    # single perturbation
    python run_single_param_benchmark.py --method grad_vanrossum # single method
    python run_single_param_benchmark.py --n-starts 5           # override start count
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ADoptEX.benchmark.methods import get_single_param_methods
from ADoptEX.benchmark.sampling import TONIC_GROUND_TRUTHS
from ADoptEX.benchmark.scenarios import (
    MEMBRANE_PARAMS,
    get_single_param_scenarios,
)

log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results" / "single_param"


def main():
    parser = argparse.ArgumentParser(
        description="Run single-parameter convergence benchmark"
    )
    parser.add_argument(
        "--param",
        type=str,
        choices=MEMBRANE_PARAMS,
        help="Run only this parameter (e.g., g_L, E_L, v_T)",
    )
    parser.add_argument(
        "--gt-index", type=int, help="Run only specific ground truth (0-19)"
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        help="Run only this perturbation level (e.g., 0.15)",
    )
    parser.add_argument("--method", type=str, help="Run only this method (by name)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip existing results (default)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Re-run existing results"
    )
    parser.add_argument(
        "--n-starts", type=int, default=10, help="Starts per scenario"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR)
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("jaxley").setLevel(logging.WARNING)

    resume = not args.no_resume

    # ── Ground truths ──────────────────────────────────────────────────────
    ground_truths = TONIC_GROUND_TRUTHS

    # ── Which parameters to test ───────────────────────────────────────────
    param_names = [args.param] if args.param else None

    # ── Build scenarios ────────────────────────────────────────────────────
    perturbations = None
    if args.perturbation is not None:
        perturbations = [args.perturbation]

    scenarios = get_single_param_scenarios(
        n_starts=args.n_starts,
        ground_truths=ground_truths,
        perturbations=perturbations,
        param_names=param_names,
    )

    # Filter by GT index
    if args.gt_index is not None:
        tag = f"gt{args.gt_index:02d}"
        scenarios = [s for s in scenarios if tag in s.name]
        if not scenarios:
            print(f"No scenarios matching GT index {args.gt_index}")
            sys.exit(1)

    # ── Build methods ──────────────────────────────────────────────────────
    methods = get_single_param_methods()
    if args.method:
        methods = [m for m in methods if m.name == args.method]
        if not methods:
            print(f"No method matching '{args.method}'")
            print("Available methods:")
            for m in get_single_param_methods():
                print(f"  {m.name}")
            sys.exit(1)

    # ── Summary ────────────────────────────────────────────────────────────
    tested_params = sorted({s.trainable_params[0] for s in scenarios})
    tested_gts = sorted({s.name.split("_gt")[1].split("_")[0] for s in scenarios})
    tested_perts = sorted({s.perturbation for s in scenarios})

    print(f"\n{'='*70}")
    print("SINGLE-PARAMETER BENCHMARK")
    print(f"{'='*70}")
    print(f"  Parameters:    {', '.join(tested_params)}")
    print(f"  Ground truths: {len(tested_gts)}")
    print(f"  Perturbations: {tested_perts}")
    print(f"  Methods:       {', '.join(m.name for m in methods)}")
    print(f"  Starts/scenario: {args.n_starts}")

    total_runs = len(scenarios) * len(methods) * args.n_starts
    print(f"  Total runs:    {total_runs}")
    print()

    # ── Dry run ────────────────────────────────────────────────────────────
    if args.dry_run:
        print("Scenarios breakdown by parameter:")
        for param in tested_params:
            n = sum(1 for s in scenarios if s.trainable_params[0] == param)
            runs = n * len(methods) * args.n_starts
            print(f"  {param:10s}: {n:3d} scenarios, {runs:5d} runs")
        print()
        print("Methods:")
        for m in methods:
            if m.method_type == "gradient":
                print(
                    f"  {m.name} ({m.loss_type}, {m.training_config.n_epochs} epochs)"
                )
            else:
                print(f"  {m.name} (NM, {m.nm_max_fev} max_fev)")
        return

    # ── Run ─────────────────────────────────────────────────────────────────
    from ADoptEX.benchmark.runner import run_benchmark

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Starting single-param benchmark: %d scenarios x %d methods = %d pairs",
        len(scenarios),
        len(methods),
        len(scenarios) * len(methods),
    )

    results = run_benchmark(
        scenarios=scenarios,
        methods=methods,
        output_dir=output_dir,
        resume=resume,
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    from ADoptEX.benchmark.analysis import recovery_table

    print(f"\n{'='*70}")
    print("SINGLE-PARAMETER BENCHMARK — RECOVERY SUMMARY")
    print(f"{'='*70}")

    # Per-parameter recovery table
    for param in tested_params:
        param_results = [r for r in results if f"single_{param}_" in r.scenario_name]
        if not param_results:
            continue

        print(f"\n--- {param} ---")
        table = recovery_table(param_results, [param])
        if table:
            # Aggregate across GTs: group by (method, perturbation)
            from collections import defaultdict

            agg = defaultdict(list)
            for row in table:
                # Extract perturbation from scenario name
                parts = row["scenario"].split("_")
                pct = [p for p in parts if p.endswith("pct")][0]
                key = (row["method"], pct)
                agg[key].append(row["mean_error_pct"])

            print(f"  {'Method':<25s} {'Pert':>6s} {'Mean Err%':>10s} {'Std':>8s}")
            print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*8}")
            for (method, pct), errors in sorted(agg.items()):
                import math

                mean = sum(errors) / len(errors)
                std = (
                    math.sqrt(sum((e - mean) ** 2 for e in errors) / len(errors))
                    if len(errors) > 1
                    else 0.0
                )
                print(f"  {method:<25s} {pct:>6s} {mean:>10.2f} {std:>8.2f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
