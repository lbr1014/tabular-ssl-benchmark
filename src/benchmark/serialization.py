"""Serialization utilities for benchmark experiment results."""

from typing import Any

from benchmark.experiment import ExperimentResult


def serialize_experiment_result(
    result: ExperimentResult,
) -> dict[str, Any]:
    """Convert an experiment result into a flat serializable record.

    Configuration, metrics, execution times, sample counts, and dataset
    provenance are combined into a single record suitable for tabular
    persistence and later analysis.

    Args:
        result (ExperimentResult): Experiment result to serialize.

    Returns:
        dict[str, Any]: Flat serializable representation of the result.
    """
    record = {
        "experiment_id": result.config.experiment_id,
        "dataset_name": result.config.dataset_name,
        "model_name": result.config.model_name,
        "ssl_method": result.config.ssl_method,
        "label_fraction": result.config.label_fraction,
        "seed": result.config.seed,
        "test_size": result.config.test_size,
        "fit_time": result.fit_time,
        "predict_time": result.predict_time,
        "n_train": result.n_train,
        "n_labeled": result.n_labeled,
        "n_unlabeled": result.n_unlabeled,
        "n_test": result.n_test,
    }
    _merge_without_collisions(
        record,
        result.metrics,
        source="metrics",
    )

    _merge_without_collisions(
        record,
        result.metadata,
        source="metadata",
    )

    return record

def _merge_without_collisions(
    target: dict[str, Any],
    values: dict[str, Any],
    *,
    source: str,
) -> None:
    """Merge values into a record without overwriting existing keys."""
    collisions = target.keys() & values.keys()

    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(
            f"{source} contains reserved result fields: {names}."
        )

    target.update(values)