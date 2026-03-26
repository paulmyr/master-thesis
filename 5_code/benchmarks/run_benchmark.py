#!/usr/bin/env python3
"""
CLI entry point for running the benchmark suite.

Usage:
    python run_benchmark.py                              # full suite
    python run_benchmark.py --scenario tonic_15pct_full  # single scenario
    python run_benchmark.py --method grad_vanrossum      # single method
    python run_benchmark.py --dry-run                    # show what would run
    python run_benchmark.py --resume                     # skip existing results
    python run_benchmark.py --n-starts 5                 # override start count
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ADoptEX.benchmark.methods import get_standard_methods
from ADoptEX.benchmark.runner import run_benchmark
from ADoptEX.benchmark.scenarios import get_synthetic_scenarios

log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def main():
    parser = argparse.ArgumentParser(description="Run ADoptEX benchmark suite")
    parser.add_argument("--scenario", type=str, help="Run only this scenario (by name)")
    parser.add_argument("--method", type=str, help="Run only this method (by name)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip existing results")
    parser.add_argument("--no-resume", action="store_true", help="Re-run existing results")
    parser.add_argument("--n-starts", type=int, help="Override number of starts")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
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

    # Build scenario list
    n_starts = args.n_starts or 10
    scenarios = list(get_synthetic_scenarios(n_starts=n_starts))

    # Filter by name
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]
        if not scenarios:
            print(f"No scenario matching '{args.scenario}'")
            print("Available scenarios:")
            for s in get_synthetic_scenarios():
                print(f"  {s.name}")
            sys.exit(1)

    # Build method list
    methods = get_standard_methods()
    if args.method:
        methods = [m for m in methods if m.name == args.method]
        if not methods:
            print(f"No method matching '{args.method}'")
            print("Available methods:")
            for m in get_standard_methods():
                print(f"  {m.name}")
            sys.exit(1)

    # Dry run
    if args.dry_run:
        print(f"Would run {len(scenarios)} scenarios x {len(methods)} methods "
              f"x {n_starts} starts = {len(scenarios) * len(methods) * n_starts} total runs")
        print()
        print("Scenarios:")
        for s in scenarios:
            print(f"  {s.name} (I={s.stim_current_pA} pA, {s.stim_duration_ms} ms)")
        print()
        print("Methods:")
        for m in methods:
            print(f"  {m.name} ({m.method_type})")
        return

    # Run benchmark (no external data needed — scenarios define their own stimulus)
    log.info(
        "Starting benchmark: %d scenarios x %d methods x %d starts",
        len(scenarios),
        len(methods),
        n_starts,
    )

    results = run_benchmark(
        scenarios=scenarios,
        methods=methods,
        output_dir=args.output_dir,
        resume=resume,
    )

    # Print summary
    from ADoptEX.benchmark.analysis import print_table, summary_table

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    table = summary_table(results)
    print_table(table)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
