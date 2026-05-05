"""Shared pytest fixtures for pd-ocr-synth tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def recipes_dir() -> str:
    """Path to the bundled recipes directory."""
    return str(Path(__file__).resolve().parent.parent / "recipes")


_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _bundled_font_bytes() -> bytes | None:
    if _BUNDLED_FONT.exists():
        return _BUNDLED_FONT.read_bytes()
    return None


@pytest.fixture
def writable_font_bytes() -> bytes:
    """Bytes of a real font, or skip if not fetched.

    Used by validation / CLI tests that need ``font.path.exists()``
    to point at something the font-coverage check can actually open.
    """
    data = _bundled_font_bytes()
    if data is None:
        pytest.skip("Real Gaelic font not fetched. Run ./scripts/fetch-fonts-gaelic.sh to enable.")
    return data
