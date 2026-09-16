"""Serialization utilities for validated benchmark configuration."""
import json

from config.models import BenchmarkConfig, DatasetConfig
from config.serialization import serialize_benchmark_config


def test_serialize_benchmark_config_preserves_effective_configuration():
    """Serialization should preserve the validated benchmark configuration."""
    datasets = (
        DatasetConfig(
            name="iris",
            openml_id=61,
            enabled=True,
        ),
        DatasetConfig(
            name="disabled-dataset",
            openml_id=123,
            enabled=False,
        ),
    )

    benchmark = BenchmarkConfig(
        models=(
            "logistic_regression",
            "random_forest",
        ),
        ssl_methods=("supervised",),
        label_fractions=(0.1, 0.5),
        seeds=(1, 2),
        test_size=0.25,
    )

    serialized = serialize_benchmark_config(
        datasets=datasets,
        benchmark=benchmark,
    )

    assert serialized == {
        "datasets": [
            {
                "name": "iris",
                "openml_id": 61,
                "enabled": True,
            },
            {
                "name": "disabled-dataset",
                "openml_id": 123,
                "enabled": False,
            },
        ],
        "benchmark": {
            "models": [
                "logistic_regression",
                "random_forest",
            ],
            "ssl_methods":[
                "supervised",    
            ],
            "label_fractions": [0.1, 0.5],
            "seeds": [1, 2],
            "test_size": 0.25,
        },
    }
    
def test_serialized_benchmark_config_is_json_serializable():
    """Serialized benchmark configuration should be JSON compatible."""
    datasets = (
        DatasetConfig(
            name="iris",
            openml_id=61,
        ),
    )

    benchmark = BenchmarkConfig(
        models=("logistic_regression",),
        ssl_methods=("supervised",),
        label_fractions=(0.1,),
        seeds=(1,),
        test_size=0.2,
    )

    serialized = serialize_benchmark_config(
        datasets=datasets,
        benchmark=benchmark,
    )

    json.dumps(serialized)