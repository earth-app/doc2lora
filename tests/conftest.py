"""Shared pytest fixtures and helpers for the doc2lora test suite."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Yield a fresh temporary directory as a Path, cleaned up afterward."""
    import shutil

    path = tempfile.mkdtemp()
    try:
        yield Path(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)
