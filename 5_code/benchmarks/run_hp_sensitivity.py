#!/usr/bin/env python3
"""
HP sensitivity grid search.

Generates MethodConfig variants programmatically and feeds them
through the same run_benchmark() machinery.

Results go to results/hp_sensitivity/ subdirectory.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ADoptEX.benchmark.methods import get_hp_sensitivity_methods
from ADoptEX.benchmark.runner import run_benchmark
from ADoptEX.benchmark.scenarios import get_synthetic_scenarios

log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results" / "hp_sensitivity"


def main():
    parser = argparse.ArgumentParser(description="Run HP sensitivity analysis")
    parser.add_argument("--n-starts", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["tonic_15pct_full", "adaptation_15pct_full", "initial_bursting_15pct_full"],
        help="Scenario names to test (default: 3 representative patterns)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Get HP variants
    methods = get_hp_sensitivity_methods()

    # Get representative scenarios
    all_scenarios = get_synthetic_scenarios(n_starts=args.n_starts)
    scenarios = [s for s in all_scenarios if s.name in args.scenarios]

    if not scenarios:
        print("No matching scenarios. Available:")
        for s in all_scenarios:
            print(f"  {s.name}")
        sys.exit(1)

    if args.dry_run:
        print(f"Would run {len(scenarios)} scenarios x {len(methods)} HP variants "
              f"x {args.n_starts} starts = {len(scenarios) * len(methods) * args.n_starts} runs")
        print("\nScenarios:", [s.name for s in scenarios])
        print("Methods:", [m.name for m in methods])
        return

    resume = not args.no_resume

    results = run_benchmark(
        scenarios=scenarios,
        methods=methods,
        output_dir=args.output_dir,
        resume=resume,
    )

    from ADoptEX.benchmark.analysis import print_table, summary_table

    print("\n" + "=" * 80)
    print("HP SENSITIVITY RESULTS")
    print("=" * 80)
    print_table(summary_table(results))


if __name__ == "__main__":
    main()
