#!/usr/bin/env python3
"""
CLI entry point for the tonic convergence benchmark.

Tests convergence reliability across 20 random ground truths,
6 perturbation levels, and 3 optimization methods.

20 GTs x 6 perturbations x 3 methods x 10 starts = 3,600 total runs.

Usage:
    python run_tonic_benchmark.py                        # full suite
    python run_tonic_benchmark.py --dry-run              # show what would run
    python run_tonic_benchmark.py --gt-index 0           # single ground truth
    python run_tonic_benchmark.py --perturbation 0.05    # single perturbation
    python run_tonic_benchmark.py --method grad_vanrossum # single method
    python run_tonic_benchmark.py --n-starts 5           # override start count
    python run_tonic_benchmark.py --no-resume            # re-run existing results
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ADoptEX.benchmark.methods import get_tonic_methods
from ADoptEX.benchmark.runner import run_benchmark
from ADoptEX.benchmark.sampling import TONIC_GROUND_TRUTHS
from ADoptEX.benchmark.scenarios import TONIC_PERTURBATIONS, get_tonic_scenarios

log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results" / "tonic"


def main():
    parser = argparse.ArgumentParser(
        description="Run tonic convergence benchmark (Q1)"
    )
    parser.add_argument(
        "--scenario-index", type=int, help="Run only specific scenario (0-19)"
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        help="Run only this perturbation level (e.g., 0.05)",
    )
    parser.add_argument("--method", type=str, help="Run only this method (by name)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument(
        "--resume", action="store_true", default=True, help="Skip existing results (default)"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Re-run existing results"
    )
    parser.add_argument("--n-starts", type=int, default=10, help="Starts per scenario")
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

    # ── Ground truths (pre-generated, no simulation needed) ────────────────
    ground_truths = TONIC_GROUND_TRUTHS

    print(f"\n{'='*70}")
    print("TONIC BENCHMARK — Ground Truth Summary")
    print(f"{'='*70}")
    for i, gt in enumerate(ground_truths):
        print(
            f"  GT {i:02d}: I={gt['stim_current_pA']:6.1f} pA, "
            f"g_L={gt['params']['g_L']:.1f}, "
            f"E_L={gt['params']['E_L']:.1f}, "
            f"v_T={gt['params']['v_T']:.1f}, "
            f"delta_T={gt['params']['delta_T']:.1f}"
        )
    print()

    # ── Build scenarios ─────────────────────────────────────────────────────
    perturbations = None
    if args.perturbation is not None:
        perturbations = [args.perturbation]

    scenarios = get_tonic_scenarios(
        n_starts=args.n_starts,
        ground_truths=ground_truths,
        perturbations=perturbations,
    )

    # Filter by scenario index
    if args.scenario_index is not None:
        tag = f"gt{args.scenario_index:02d}"
        scenarios = [s for s in scenarios if tag in s.name]
        if not scenarios:
            print(f"No scenarios matching GT index {args.scenario_index}")
            sys.exit(1)

    # ── Build methods ───────────────────────────────────────────────────────
    methods = get_tonic_methods()
    if args.method:
        methods = [m for m in methods if m.name == args.method]
        if not methods:
            print(f"No method matching '{args.method}'")
            print("Available methods:")
            for m in get_tonic_methods():
                print(f"  {m.name}")
            sys.exit(1)

    # ── Dry run ─────────────────────────────────────────────────────────────
    total_runs = len(scenarios) * len(methods) * args.n_starts
    n_gradient = sum(1 for m in methods if m.method_type == "gradient")
    n_nm = sum(1 for m in methods if m.method_type == "nelder_mead")
    grad_runs = len(scenarios) * n_gradient * args.n_starts
    nm_runs = len(scenarios) * n_nm * args.n_starts

    if args.dry_run:
        print(
            f"Would run {len(scenarios)} scenarios x {len(methods)} methods "
            f"x {args.n_starts} starts = {total_runs} total runs"
        )
        print(f"  Gradient runs: {grad_runs} (est. {grad_runs * 5 / 60:.0f} min)")
        print(f"  Nelder-Mead runs: {nm_runs} (est. {nm_runs * 30 / 60:.0f} min)")
        print()
        print("Perturbation levels:", sorted({s.perturbation for s in scenarios}))
        print(f"Ground truths: {len({s.name.split('_')[1] for s in scenarios})}")
        print()
        print("Methods:")
        for m in methods:
            if m.method_type == "gradient":
                print(f"  {m.name} ({m.loss_type}, {m.training_config.n_epochs} epochs)")
            else:
                print(f"  {m.name} (NM, {m.nm_max_fev} max_fev)")
        return

    # ── Run ──────────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Starting tonic benchmark: %d scenarios x %d methods x %d starts = %d runs",
        len(scenarios),
        len(methods),
        args.n_starts,
        total_runs,
    )

    results = run_benchmark(
        scenarios=scenarios,
        methods=methods,
        output_dir=output_dir,
        resume=resume,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    from ADoptEX.benchmark.analysis import print_table, summary_table

    print(f"\n{'='*70}")
    print("TONIC BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}")
    table = summary_table(results)
    print_table(table)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
