"""Command-line interface for benchmark execution."""

import argparse
from pathlib import Path
from collections.abc import Sequence

from benchmark.orchestrator import run_and_save_benchmark
from config.loader import load_benchmark_config, load_dataset_configs


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        argparse.ArgumentParser: Configured benchmark argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the reproducible tabular semi-supervised "
            "learning benchmark."
        ),
    )

    parser.add_argument(
        "--datasets-config",
        type=Path,
        required=True,
        help="Path to the dataset configuration YAML file.",
    )

    parser.add_argument(
        "--benchmark-config",
        type=Path,
        required=True,
        help="Path to the benchmark configuration YAML file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where benchmark run artifacts are stored.",
    )

    return parser

def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the benchmark command-line interface.

    Args:
        argv (Sequence[str] | None): Optional command-line arguments.
            When omitted, arguments are read from the process command line.

    Returns:
        int: Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    datasets = load_dataset_configs(
        args.datasets_config,
    )

    benchmark = load_benchmark_config(
        args.benchmark_config,
    )

    artifacts = run_and_save_benchmark(
        datasets=datasets,
        benchmark=benchmark,
        output_dir=args.output_dir,
    )

    print(
        f"Benchmark completed successfully: "
        f"{artifacts.results_csv.parent}"
    )

    return 0