"""Typed configuration models for the benchmark.

This module defines the validated configuration structures used to
describe datasets and benchmark-wide experimental settings.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration describing a dataset used by the benchmark."""

    name: str
    openml_id: int
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate the dataset configuration."""
        if not isinstance(self.name, str):
            raise TypeError("name must be a string.")

        if not self.name.strip():
            raise ValueError("name must not be empty.")

        if not isinstance(self.openml_id, int) or isinstance(
            self.openml_id,
            bool,
        ):
            raise TypeError("openml_id must be an integer.")

        if self.openml_id <= 0:
            raise ValueError("openml_id must be positive.")

        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean.")
        
@dataclass(frozen=True)
class SSLMethodConfig:
    """Configuration for a semi-supervised learning strategy.

    Attributes:
        name: Stable identifier of the SSL strategy.
        requires_base_model: Whether the strategy is combined with each
            configured benchmark classifier.
        params: Strategy-specific hyperparameters.
    """

    name: str
    requires_base_model: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the SSL method configuration."""
        if not isinstance(self.name, str):
            raise TypeError("name must be a string.")

        if not self.name.strip():
            raise ValueError("name must not be empty.")

        if not isinstance(self.requires_base_model, bool):
            raise TypeError("requires_base_model must be a boolean.")
        
        if not isinstance(self.params, dict):
            raise TypeError("params must be a dictionary.")
        
@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration describing the benchmark experiment grid."""

    models: tuple[str, ...]
    ssl_methods: tuple[SSLMethodConfig, ...]
    label_fractions: tuple[float, ...]
    seeds: tuple[int, ...]
    test_size: float = 0.2

    def __post_init__(self) -> None:
        """Validate benchmark-wide experimental settings."""
        self._validate_models()
        self._validate_ssl_methods()
        self._validate_label_fractions()
        self._validate_seeds()
        self._validate_test_size()
        
    def _validate_models(self) -> None:
        """Validate benchmark model identifiers."""
        if not isinstance(self.models, tuple):
            raise TypeError("models must be a tuple.")

        if not self.models:
            raise ValueError(
                "models must contain at least one model."
            )

        for model_name in self.models:
            if not isinstance(model_name, str):
                raise TypeError(
                    "models must contain string values."
                )

            if not model_name.strip():
                raise ValueError(
                    "models must not contain empty names."
                )

        if len(set(self.models)) != len(self.models):
            raise ValueError(
                "models must not contain duplicate values."
            )
            
    def _validate_ssl_methods(self) -> None:
        """Validate benchmark SSL method configurations."""
        if not isinstance(self.ssl_methods, tuple):
            raise TypeError("ssl_methods must be a tuple.")

        if not self.ssl_methods:
            raise ValueError(
                "ssl_methods must contain at least one method."
            )

        if not all(
            isinstance(method, SSLMethodConfig)
            for method in self.ssl_methods
        ):
            raise TypeError(
                "ssl_methods must contain SSLMethodConfig instances."
            )

        names = tuple(
            method.name
            for method in self.ssl_methods
        )

        if len(names) != len(set(names)):
            raise ValueError(
                "ssl_methods must contain unique method names."
            )
    
    def _validate_label_fractions(self) -> None:
        """Validate labelled training fractions."""
        if not isinstance(self.label_fractions, tuple):
            raise TypeError("label_fractions must be a tuple.")

        if not self.label_fractions:
            raise ValueError(
                "label_fractions must contain at least one value."
            )

        for fraction in self.label_fractions:
            if (
                not isinstance(fraction, (int, float))
                or isinstance(fraction, bool)
            ):
                raise TypeError(
                    "label_fractions must contain numeric values."
                )

            if not 0 < fraction <= 1:
                raise ValueError(
                    "label_fractions must contain values in the "
                    "interval (0, 1]."
                )

        if len(set(self.label_fractions)) != len(self.label_fractions):
            raise ValueError(
                "label_fractions must not contain duplicate values."
            )

    def _validate_seeds(self) -> None:
        """Validate experimental root seeds."""
        if not isinstance(self.seeds, tuple):
            raise TypeError("seeds must be a tuple.")

        if not self.seeds:
            raise ValueError(
                "seeds must contain at least one value."
            )

        for seed in self.seeds:
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise TypeError(
                    "seeds must contain integer values."
                )

            if seed < 0:
                raise ValueError(
                    "seeds must contain non-negative values."
                )

        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(
                "seeds must not contain duplicate values."
            )

    def _validate_test_size(self) -> None:
        """Validate the held-out test fraction."""
        if (
            not isinstance(self.test_size, (int, float))
            or isinstance(self.test_size, bool)
        ):
            raise TypeError("test_size must be numeric.")

        if not 0 < self.test_size < 1:
            raise ValueError(
                "test_size must be in the interval (0, 1)."
            )
            
