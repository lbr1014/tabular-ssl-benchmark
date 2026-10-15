"""Tests for the benchmark CLI."""

from unittest.mock import patch

from pathlib import Path

import pytest

from benchmark.cli import build_parser, main
from benchmark.persistence import BenchmarkRunArtifacts
from config.models import BenchmarkConfig, DatasetConfig, SSLMethodConfig

def test_build_parser_parses_required_arguments():
    """CLI parser should parse benchmark input and output paths."""
    parser = build_parser()

    args = parser.parse_args([
        "--datasets-config",
        "config/datasets.yaml",
        "--benchmark-config",
        "config/benchmark.yaml",
        "--output-dir",
        "results/test-run",
    ])

    assert args.datasets_config == Path(
        "config/datasets.yaml"
    )
    assert args.benchmark_config == Path(
        "config/benchmark.yaml"
    )
    assert args.output_dir == Path(
        "results/test-run"
    )
    
def test_build_parser_requires_configuration_arguments():
    """CLI should require benchmark configuration paths."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
        
def test_main_runs_and_persists_benchmark(
    monkeypatch,
    tmp_path,
):
    """CLI should validate configuration and execute a persistent run."""
    datasets = (
        DatasetConfig(
            name="iris",
            openml_id=61,
            enabled=True,
        ),
    )

    benchmark = BenchmarkConfig(
        models=("logistic_regression",),
        ssl_methods=(
            SSLMethodConfig(name="supervised"),
        ),
        label_fractions=(0.1,),
        seeds=(42,),
        test_size=0.2,
    )

    datasets_path = tmp_path / "datasets.yaml"
    benchmark_path = tmp_path / "benchmark.yaml"
    output_dir = tmp_path / "results"

    calls = {}

    monkeypatch.setattr(
        "benchmark.cli.load_dataset_configs",
        lambda path: datasets,
    )

    monkeypatch.setattr(
        "benchmark.cli.load_benchmark_config",
        lambda path: benchmark,
    )

    def fake_run_and_save_benchmark(
        *,
        datasets,
        benchmark,
        output_dir,
    ):
        calls["datasets"] = datasets
        calls["benchmark"] = benchmark
        calls["output_dir"] = output_dir

        return BenchmarkRunArtifacts(
            results_jsonl=output_dir / "results.jsonl",
            results_csv=output_dir / "results.csv",
            metadata_json=output_dir / "metadata.json",
            config_json=output_dir / "config.json",
        )

    monkeypatch.setattr(
        "benchmark.cli.run_and_save_benchmark",
        fake_run_and_save_benchmark,
    )

    exit_code = main([
        "--datasets-config",
        str(datasets_path),
        "--benchmark-config",
        str(benchmark_path),
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 0
    assert calls["datasets"] is datasets
    assert calls["benchmark"] is benchmark
    assert calls["output_dir"] == output_dir


def test_cli_rejects_unknown_model_before_execution(capsys):
    """Invalid models should produce a CLI error without running."""
    datasets = (
        DatasetConfig(name="iris", openml_id=61),
    )
    benchmark = BenchmarkConfig(
        models=("unknown_model",),
        ssl_methods=(SSLMethodConfig(name="supervised"),),
        label_fractions=(0.1,),
        seeds=(42,),
    )

    with (
        patch(
            "benchmark.cli.load_dataset_configs",
            return_value=datasets,
        ),
        patch(
            "benchmark.cli.load_benchmark_config",
            return_value=benchmark,
        ),
        patch(
            "benchmark.cli.run_and_save_benchmark",
        ) as mock_run,
    ):
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--datasets-config", "datasets.yaml",
                "--benchmark-config", "benchmark.yaml",
                "--output-dir", "results",
            ])

    assert exc_info.value.code == 2
    assert "Invalid benchmark model" in capsys.readouterr().err
    mock_run.assert_not_called()


def test_cli_rejects_unknown_ssl_method_before_execution(capsys):
    """Unknown SSL methods should fail before benchmark execution."""
    datasets = (
        DatasetConfig(name="iris", openml_id=61),
    )
    benchmark = BenchmarkConfig(
        models=("logistic_regression",),
        ssl_methods=(SSLMethodConfig(name="unknown_ssl"),),
        label_fractions=(0.1,),
        seeds=(42,),
    )

    with (
        patch(
            "benchmark.cli.load_dataset_configs",
            return_value=datasets,
        ),
        patch(
            "benchmark.cli.load_benchmark_config",
            return_value=benchmark,
        ),
        patch(
            "benchmark.cli.run_and_save_benchmark",
        ) as mock_run,
    ):
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--datasets-config", "datasets.yaml",
                "--benchmark-config", "benchmark.yaml",
                "--output-dir", "results",
            ])

    assert exc_info.value.code == 2
    assert "Invalid SSL method" in capsys.readouterr().err
    mock_run.assert_not_called()
