from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single benchmark experiment."""

    dataset_name: str
    model_name: str
    ssl_method: str | None

    label_fraction: float
    seed: int

    test_size: float = 0.2

    def __post_init__(self) -> None:
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
        ssl = self.ssl_method or "supervised"

        return (
            f"{self.dataset_name}"
            f"__{self.model_name}"
            f"__{ssl}"
            f"__lf-{self.label_fraction:g}"
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