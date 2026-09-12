"""Execution utilities for benchmark experiments.
This module orchestrates dataset splitting, leakage-safe preprocessing,
model fitting, prediction, and evaluation for a single supervised
benchmark experiment.
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

def run_supervised_experiment(
    *,
    dataset: TabularDataset,
    model: BenchmarkClassifier,
    label_fraction: float,
    test_size: float,
    seed: int,
) -> ExperimentResult:
    """Run one supervised baseline experiment.
    The experiment creates a deterministic semi-supervised split but
    trains the supervised baseline exclusively on the labelled training
    subset. Preprocessing is also fitted only on labelled data to prevent
    information leakage from unlabeled or held-out test samples.

    Args:
        dataset (TabularDataset): Dataset used by the experiment.
        model (BenchmarkClassifier): Classifier implementing the benchmark model interface.
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
    
    X_labeled = dataset.X.iloc[
        split.labeled_indices
    ]

    y_labeled = dataset.y.iloc[
        split.labeled_indices
    ]

    X_test = dataset.X.iloc[
        split.test_indices
    ]

    y_test = dataset.y.iloc[
        split.test_indices
    ]
    
    preprocessor = create_tabular_preprocessor(
        numerical_features=dataset.numerical_features,
        categorical_features=dataset.categorical_features,
    )

    X_labeled_transformed = fit_transform_tabular(
        preprocessor,
        X_labeled,
    )

    X_test_transformed = transform_tabular(
        preprocessor,
        X_test,
    )
    
    fit_start = perf_counter()

    model.fit(
        X_labeled_transformed,
        y_labeled.to_numpy(),
    )

    fit_time = perf_counter() - fit_start
    
    predict_start = perf_counter()

    predictions = model.predict(
        X_test_transformed
    )

    probabilities = model.predict_proba(
        X_test_transformed
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
            model.classes_,
        )
    )

    metrics.update(
        compute_calibration_metrics(
            y_test_array,
            probabilities,
            model.classes_,
        )
    )
    
    config = ExperimentConfig(
        dataset_name=dataset.name,
        model_name=model.name,
        ssl_method="supervised",
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