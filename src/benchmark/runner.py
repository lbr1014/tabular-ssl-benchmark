"""Execution utilities for benchmark experiments.
This module orchestrates dataset splitting, leakage-safe preprocessing,
learning-strategy execution, prediction, and evaluation for benchmark
experiments.
"""

from time import perf_counter

from benchmark.experiment import (
    ExperimentConfig,
    ExperimentResult,
)
from datasets.dataset import TabularDataset
from datasets.preprocessing import (
    create_tabular_preprocessor,
    fit_transform_tabular,
    transform_tabular,
)
from datasets.split import create_ssl_split
from evaluation.calibration import compute_calibration_metrics
from evaluation.metrics import (
    compute_classification_metrics,
    compute_probabilistic_metrics,
)
from models.base import BenchmarkClassifier
from config.matrix import ExperimentSpec
from models.factory import create_classifier
from collections.abc import Mapping

from ssl_methods.base import SSLMethod
from ssl_methods.factory import create_ssl_method

def run_experiment(
    *,
    dataset: TabularDataset,
    model: BenchmarkClassifier,
    ssl_method: SSLMethod,
    label_fraction: float,
    test_size: float,
    seed: int,
) -> ExperimentResult:
    """Run one benchmark experiment.
    The experiment creates a deterministic semi-supervised split, applies
    leakage-safe preprocessing, delegates model fitting to the selected learning
    strategy, and evaluates the fitted classifier exclusively on the held-out
    test partition.
        
    Args:
        dataset (TabularDataset): Dataset used by the experiment.
        model (BenchmarkClassifier): Classifier implementing the benchmark model interface.
        ssl_method (SSLMethod): Semi-supervised learning method to evaluate.
        label_fraction (float): Fraction of the training partition whose labels are available.
        test_size (float): Fraction of the complete dataset reserved for held-out testing.
        seed (int): Root random seed controlling the experiment split and model.

    Returns:
        ExperimentResult: Metrics, timing information, sample counts, and experiment
        metadata produced by the run.
    """
    
    split = create_ssl_split(
        dataset.y,
        label_fraction=label_fraction,
        test_size=test_size,
        seed=seed,
    )
    
    x_labeled = dataset.X.iloc[
        split.labeled_indices
    ]
    
    x_unlabeled = dataset.X.iloc[
        split.unlabeled_indices
    ]

    y_labeled = dataset.y.iloc[
        split.labeled_indices
    ]

    x_test = dataset.X.iloc[
        split.test_indices
    ]

    y_test = dataset.y.iloc[
        split.test_indices
    ]
    
    preprocessor = create_tabular_preprocessor(
        numerical_features=dataset.numerical_features,
        categorical_features=dataset.categorical_features,
    )

    x_labeled_transformed = fit_transform_tabular(
        preprocessor,
        x_labeled,
    )
    
    if len(x_unlabeled) > 0:
        x_unlabeled_transformed = transform_tabular(
            preprocessor,
            x_unlabeled,
        )
    else:
        x_unlabeled_transformed = x_labeled_transformed[:0].copy()

    x_test_transformed = transform_tabular(
        preprocessor,
        x_test,
    )
    
    fit_start = perf_counter()

    fitted_model = ssl_method.fit(
        model=model,
        x_labeled=x_labeled_transformed,
        y_labeled=y_labeled.to_numpy(),
        x_unlabeled=x_unlabeled_transformed,
    )

    fit_time = perf_counter() - fit_start
    
    predict_start = perf_counter()

    predictions = fitted_model.predict(
        x_test_transformed
    )

    probabilities = fitted_model.predict_proba(
        x_test_transformed
    )

    predict_time = (
        perf_counter() - predict_start
    )
    
    y_test_array = y_test.to_numpy()

    metrics = {}

    metrics.update(
        compute_classification_metrics(
            y_test_array,
            predictions,
        )
    )

    metrics.update(
        compute_probabilistic_metrics(
            y_test_array,
            probabilities,
            fitted_model.classes_,
        )
    )

    metrics.update(
        compute_calibration_metrics(
            y_test_array,
            probabilities,
            fitted_model.classes_,
        )
    )
    
    config = ExperimentConfig(
        dataset_name=dataset.name,
        model_name=model.name,
        ssl_method=ssl_method.name,
        label_fraction=label_fraction,
        seed=seed,
        test_size=test_size,
    )
    
    return ExperimentResult(
        config=config,
        metrics=metrics,
        fit_time=fit_time,
        predict_time=predict_time,
        n_train=len(split.train_indices),
        n_labeled=len(split.labeled_indices),
        n_unlabeled=len(split.unlabeled_indices),
        n_test=len(split.test_indices),
        metadata={
            "dataset_source": dataset.source,
            "dataset_source_id": dataset.source_id,
            "dataset_source_version": dataset.source_version,
        },
    )
    
def run_supervised_experiment(
    *,
    dataset: TabularDataset,
    model: BenchmarkClassifier,
    label_fraction: float,
    test_size: float,
    seed: int,
) -> ExperimentResult:
    """Run one supervised reference experiment.
    This compatibility wrapper delegates execution to the generic experiment
    runner using the supervised learning strategy.

    Args:
        dataset: Dataset used by the experiment.
        model: Classifier implementing the benchmark model interface.
        label_fraction: Fraction of the training partition whose labels are
            available.
        test_size: Fraction of the complete dataset reserved for held-out
            testing.
        seed: Root random seed controlling the experiment split and model.

    Returns:
        ExperimentResult: Result produced by the supervised reference
        strategy.
    """
    return run_experiment(
        dataset=dataset,
        model=model,
        ssl_method=create_ssl_method("supervised"),
        label_fraction=label_fraction,
        test_size=test_size,
        seed=seed,
    )
    
def run_experiment_spec(
    spec: ExperimentSpec,
    dataset: TabularDataset,
) -> ExperimentResult:
    """Execute one experiment specification on a loaded dataset.
    The classifier is created from the model identifier and experiment
    seed stored in the specification. The experiment is then delegated
    to the supervised experiment runner.

    Args:
        spec (ExperimentSpec): Experimental specification to execute.
        dataset (TabularDataset): Loaded tabular dataset used by the
            experiment.

    Returns:
        ExperimentResult: Result produced by the supervised experiment.

    Raises:
        ValueError: If the loaded dataset does not match the dataset
            requested by the experiment specification.
    """
    if dataset.name != spec.dataset.name:
        raise ValueError(
            "Loaded dataset does not match experiment specification: "
            f"expected {spec.dataset.name!r}, "
            f"received {dataset.name!r}."
        )

    model = create_classifier(
        spec.model_name,
        seed=spec.seed,
    )
    
    ssl_method = create_ssl_method("supervised")

    return run_experiment(
        dataset=dataset,
        model=model,
        ssl_method=ssl_method,
        label_fraction=spec.label_fraction,
        seed=spec.seed,
        test_size=spec.test_size,
    )
    
def run_benchmark_matrix(
    matrix: tuple[ExperimentSpec, ...],
    datasets: Mapping[str, TabularDataset],
) -> tuple[ExperimentResult, ...]:
    """Execute all experiment specifications in a benchmark matrix.

    Args:
        matrix (tuple[ExperimentSpec, ...]): Ordered experiment
            specifications to execute.
        datasets (Mapping[str, TabularDataset]): Preloaded datasets keyed
            by dataset name.

    Returns:
        tuple[ExperimentResult, ...]: Experiment results in the same order
        as the input matrix.

    Raises:
        TypeError: If matrix is not a tuple.
        ValueError: If an experiment references a dataset that has not
            been provided.
    """
    if not isinstance(matrix, tuple):
        raise TypeError(
            "matrix must be a tuple of ExperimentSpec instances."
        )

    results = []

    for spec in matrix:
        try:
            dataset = datasets[spec.dataset.name]
        except KeyError as exc:
            raise ValueError(
                "Dataset required by experiment specification "
                f"is not loaded: {spec.dataset.name!r}."
            ) from exc

        result = run_experiment_spec(
            spec=spec,
            dataset=dataset,
        )

        results.append(result)

    return tuple(results)