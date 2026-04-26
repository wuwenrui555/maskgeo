"""Shared pytest fixtures for maskgeo tests.

Fixture files in tests/data/ were captured during the 2026-04-26 brainstorming
session and validated against QuPath. See docs/superpowers/specs/.
"""
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir() -> Path:
    return DATA_DIR
