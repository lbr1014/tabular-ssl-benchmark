from benchmark.orchestrator import _load_enabled_datasets
from config.models import DatasetConfig


def test_load_enabled_datasets_skips_disabled_datasets(
    monkeypatch,
    mixed_dataset,
):
    """Disabled datasets should not be loaded."""
    requested_ids = []

    def fake_loader(data_id):
        requested_ids.append(data_id)
        return mixed_dataset

    monkeypatch.setattr(
        "benchmark.orchestrator.load_openml_dataset",
        fake_loader,
    )

    configs = (
        DatasetConfig(
            name=mixed_dataset.name,
            openml_id=61,
            enabled=True,
        ),
        DatasetConfig(
            name="disabled_dataset",
            openml_id=999,
            enabled=False,
        ),
    )

    loaded = _load_enabled_datasets(configs)

    assert requested_ids == [61]
    assert set(loaded) == {
        mixed_dataset.name,
    }