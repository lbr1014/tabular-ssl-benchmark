"""Typed configuration models for the benchmark.

This module defines the validated configuration structures used to
describe datasets and benchmark-wide experimental settings.
"""

from dataclasses import dataclass


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
class BenchmarkConfig:
    """Configuration describing the benchmark experiment grid."""

    label_fractions: tuple[float, ...]
    seeds: tuple[int, ...]
    test_size: float = 0.2

    def __post_init__(self) -> None:
        """Validate benchmark-wide experimental settings."""
        self._validate_label_fractions()
        self._validate_seeds()
        self._validate_test_size()

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