"""Experiment configuration and result data structures for the benchmark."""
import hashlib
import json

from dataclasses import dataclass, field
from typing import Any

def _ssl_params_digest(
    params: dict[str, Any],
) -> str:
    """Return a deterministic digest for SSL method parameters."""
    serialized = json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()[:8]

@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single benchmark experiment."""

    dataset_name: str
    model_name: str
    ssl_method: str | None

    label_fraction: float
    seed: int

    test_size: float = 0.2
    ssl_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the configuration parameters."""
        if not 0 < self.label_fraction <= 1:
            raise ValueError(
                "label_fraction must be in the interval (0, 1]."
            )

        if not 0 < self.test_size < 1:
            raise ValueError(
                "test_size must be in the interval (0, 1)."
            )
        
    @property
    def experiment_id(self) -> str:
        """Build a unique identifier for the experiment based on its configuration.
        
        Returns:
            str: A unique identifier string for the experiment.
        """
        ssl_identifier = self.ssl_method or "supervised"

        if self.ssl_params:
            ssl_identifier  = (
                f"{ssl_identifier }-"
                f"{_ssl_params_digest(self.ssl_params)}"
            )

        return (
            f"{self.dataset_name}"
            f"__model-{self.model_name}"
            f"__ssl-{ssl_identifier}"
            f"__lf-{self.label_fraction:g}"
            f"__test-{self.test_size:g}"
            f"__seed-{self.seed}"
        )
            
@dataclass
class ExperimentResult:
    """Results produced by a single benchmark experiment."""

    config: ExperimentConfig

    metrics: dict[str, float]

    fit_time: float
    predict_time: float

    n_train: int
    n_labeled: int
    n_unlabeled: int
    n_test: int

    metadata: dict[str, Any] = field(default_factory=dict)