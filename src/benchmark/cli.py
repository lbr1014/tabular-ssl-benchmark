"""Command-line interface for benchmark execution."""

import argparse
from pathlib import Path
from collections.abc import Sequence

from benchmark.orchestrator import run_and_save_benchmark
from config.loader import load_benchmark_config, load_dataset_configs
from config.matrix import generate_experiment_matrix
from config.validation import validate_benchmark_compatibility

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
    """Run the benchmark command-line interface with early configuration validation.

    Args:
        argv (Sequence[str] | None): Optional command-line arguments.
            When omitted, arguments are read from the process command line.

    Returns:
        int: Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        datasets = load_dataset_configs(
            args.datasets_config,
        )
        benchmark = load_benchmark_config(
            args.benchmark_config,
        )

        validate_benchmark_compatibility(benchmark)

        # Validate the experiment matrix before loading OpenML data.
        generate_experiment_matrix(
            datasets=datasets,
            benchmark=benchmark,
        )

    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    artifacts = run_and_save_benchmark(
        datasets=datasets,
        benchmark=benchmark,
        output_dir=args.output_dir,
    )

    print(
        "Benchmark completed successfully: "
        f"{artifacts.results_csv.parent}"
    )

    return 0
