
"""Validate benchmark configuration against available implementations.

Structural constraints are enforced by the configuration dataclasses.
This module checks whether requested models and SSL methods can
actually be instantiated and executed by the current benchmark.
"""

from config.models import BenchmarkConfig
from models.factory import create_classifier
from ssl_methods.factory import create_ssl_method


def validate_benchmark_compatibility(
    benchmark: BenchmarkConfig,
) -> None:
    """Validate executable model and SSL method configurations.

    Validation is performed without loading datasets or fitting models.

    Args:
        benchmark: Benchmark-wide experiment configuration.

    Raises:
        TypeError: If the configuration has an invalid type.
        ValueError: If a model or SSL method is unavailable or invalid.
    """
    if not isinstance(benchmark, BenchmarkConfig):
        raise TypeError(
            "benchmark must be a BenchmarkConfig instance."
        )

    # Check that every configured classifier can be created.
    for model_name in benchmark.models:
        try:
            create_classifier(model_name, seed=0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid benchmark model {model_name!r}: {exc}"
            ) from exc

    # Check that every SSL method exists and accepts its parameters.
    for method_config in benchmark.ssl_methods:
        try:
            method = create_ssl_method(
                method_config.name,
                **method_config.params,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid SSL method {method_config.name!r}: {exc}"
            ) from exc

        # The current experiment runner always supplies a base model.
        if method_config.requires_base_model != method.requires_base_model:
            raise ValueError(
                "Configured requires_base_model does not match "
                f"SSL method {method_config.name!r}."
            )
