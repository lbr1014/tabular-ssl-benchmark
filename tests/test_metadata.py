"""Unit tests for benchmark metadata utilities. 
This module tests the creation and serialization of benchmark run metadata, including timestamp generation, Python version detection, 
platform information retrieval, Git commit SHA extraction, and experiment count validation.
"""
from datetime import datetime, timedelta
import json
import subprocess

import pytest

from benchmark.metadata import BenchmarkRunMetadata, _get_git_commit, create_run_metadata

def test_create_run_metadata_contains_execution_information():
    """Run metadata should describe the benchmark execution."""
    metadata = create_run_metadata(
        n_experiments=50,
    )

    assert metadata.n_experiments == 50
    assert metadata.python_version
    assert metadata.platform
    assert metadata.created_at
    
def test_create_run_metadata_uses_utc_timestamp():
    """Run metadata timestamps should be timezone-aware UTC values."""
    metadata = create_run_metadata(
        n_experiments=1,
    )

    timestamp = datetime.fromisoformat(
        metadata.created_at,
    )

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)
    
def test_get_git_commit_returns_commit_hash(monkeypatch):
    """Git commit detection should return the current commit SHA.
    
    Args:
        monkeypatch: pytest fixture for modifying behavior during tests.
    """
    completed_process = subprocess.CompletedProcess(
        args=["git", "rev-parse", "HEAD"],
        returncode=0,
        stdout="abc123\n",
        stderr="",
    )

    monkeypatch.setattr(
        "benchmark.metadata.subprocess.run",
        lambda *args, **kwargs: completed_process,
    )

    assert _get_git_commit() == "abc123"
    
def test_get_git_commit_returns_none_when_git_is_unavailable(monkeypatch):
    """Git commit detection should gracefully handle missing Git."""

    def raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(
        "benchmark.metadata.subprocess.run",
        raise_file_not_found,
    )

    assert _get_git_commit() is None
    
def test_create_run_metadata_rejects_negative_experiment_count():
    """Negative experiment counts should be rejected."""
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        create_run_metadata(
            n_experiments=-1,
        )
        
def test_run_metadata_to_dict_returns_serializable_fields():
    """Run metadata should expose a serializable representation."""
    metadata = BenchmarkRunMetadata(
        created_at="2026-09-14T10:00:00+00:00",
        python_version="3.12.0",
        platform="test-platform",
        git_commit="abc123",
        n_experiments=10,
    )

    record = metadata.to_dict()

    assert record == {
        "created_at": "2026-09-14T10:00:00+00:00",
        "python_version": "3.12.0",
        "platform": "test-platform",
        "git_commit": "abc123",
        "n_experiments": 10,
    }
    
    json.dumps(record)
    
    