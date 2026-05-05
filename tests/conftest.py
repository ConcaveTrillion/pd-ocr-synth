"""Shared pytest fixtures for pd-ocr-synth tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def recipes_dir() -> str:
    """Path to the bundled recipes directory."""
    from pathlib import Path

    return str(Path(__file__).resolve().parent.parent / "recipes")
