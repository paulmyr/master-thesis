#!/usr/bin/env python3
"""
Loss Function Testing Framework - Main Entry Point

Run with:
    python main.py --help              # Show help
    python main.py --quick             # Run quick test with default trace
    python main.py --trace V.dat I.dat # Run with specific trace files

Usage:
    1. Select a preset configuration or use defaults
    2. Specify trace files from 4_data/
    3. Run optimization and view results
"""

import argparse
import sys
from pathlib import Path

from config.presets import describe_presets, get_preset
from runner.experiment import ExperimentRunner

DATA_DIRECTORY = "/Users/paulmayer/Projects/university/40_thesis/4_data/"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AdEx Loss Function Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --quick                          # Quick test with default trace
    python main.py --config standard --trace V I    # Use 'standard' preset with trace
    python main.py --list-presets                    # Show available presets

For more information, see the README or run with --help.
        """,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick test with default settings",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Use a preset configuration (quick_test, standard, thorough)",
    )

    parser.add_argument(
        "--trace",
        type=str,
        nargs=2,
        metavar=("VOLTAGE", "CURRENT"),
        help="Path to voltage and current trace files",
    )

    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available configuration presets",
    )

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        print(describe_presets())
        return

    # Quick test mode
    if args.quick:
        run_quick_test(args.config)
        return

    # Run with specified trace
    if args.trace:
        run_cli_mode(args.trace, args.config)
        return

    # No action specified — show help
    parser.print_help()
    sys.exit(1)


def run_quick_test(preset_name: str | None = None):
    """Run a quick test with the first available trace."""
    print("Running quick test...")

    # Get configuration
    preset = preset_name or "quick_test"
    config = get_preset(preset)  # ExperimentConfig

    # Find a default trace
    data_root = Path(f"{DATA_DIRECTORY}/models/optimisations")
    if not data_root.exists():
        print("Error: Data directory not found")
        sys.exit(1)

    # Find first available trace
    for cell_dir in sorted(data_root.iterdir()):
        expdata_dir = cell_dir / "expdata"
        if not expdata_dir.exists():
            continue

        voltage_files = sorted(expdata_dir.glob("*_ch5_*.dat"))
        for v_file in voltage_files:
            i_file_name = v_file.name.replace("_ch5_", "_ch4_")
            i_file = expdata_dir / i_file_name
            if i_file.exists():
                config.trace_paths = [(str(v_file), str(i_file))]
                break
        if config.trace_paths:
            break

    if not config.trace_paths:
        print("Error: No trace files found")
        sys.exit(1)

    print(f"Using trace: {config.trace_paths[0][0]}")
    print(f"Configuration: {preset}")

    # Run experiment
    runner = ExperimentRunner(config)
    result = runner.run()

    print(result.summary())


def run_cli_mode(trace_paths: tuple[str, str], preset_name: str | None = None):
    """Run optimization in CLI mode."""
    print("Running optimization...")

    # Get configuration
    preset = preset_name or "standard"
    config = get_preset(preset)
    config.trace_paths = [trace_paths]

    print(f"Voltage: {trace_paths[0]}")
    print(f"Current: {trace_paths[1]}")
    print(f"Configuration: {preset}")

    # Run experiment
    runner = ExperimentRunner(config)
    result = runner.run()

    print(result.summary())


if __name__ == "__main__":
    main()
