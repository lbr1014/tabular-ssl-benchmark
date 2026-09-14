"""Metadata utilities for reproducible benchmark runs.
This module provides functionality to capture and serialize metadata describing a complete 
benchmark execution, including the creation timestamp, Python version, platform information, 
Git commit SHA, and the number of experiments executed."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import platform
import subprocess
from typing import Any


@dataclass(frozen=True)
class BenchmarkRunMetadata:
    """Metadata describing a complete benchmark execution."""

    created_at: str
    python_version: str
    platform: str
    git_commit: str | None
    n_experiments: int

    def to_dict(self) -> dict[str, Any]:
        """Convert benchmark run metadata to a serializable dictionary.
        
        Returns:
            dict[str, Any]: Dictionary representation of the metadata.
        """
        return asdict(self)
    
def create_run_metadata(
    *,
    n_experiments: int,
) -> BenchmarkRunMetadata:
    """Create metadata describing a benchmark execution.

    Args:
        n_experiments (int): Number of experiments in the benchmark run.

    Returns:
        BenchmarkRunMetadata: Metadata describing the execution.
    """
    if n_experiments < 0:
        raise ValueError(
            "n_experiments must be non-negative."
        )

    return BenchmarkRunMetadata(
        created_at=datetime.now(timezone.utc).isoformat(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        git_commit=_get_git_commit(),
        n_experiments=n_experiments,
    )
    
def _get_git_commit() -> str | None:
    """Return the current Git commit SHA when available.
    
    Returns:
        str | None: Current Git commit SHA or None if unavailable.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip()