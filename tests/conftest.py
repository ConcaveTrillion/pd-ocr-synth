"""Shared pytest fixtures for pd-ocr-synth tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.audit import GLOBAL_AUDIT_DISABLE_ENV


@pytest.fixture(autouse=True)
def _isolate_global_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress the global audit mirror by default in every test.

    The global aggregate audit log lives at
    ``<cache_root>/audit.jsonl`` (default ``~/.cache/pd-ocr-synth``).
    Without this fixture, every test that drives ``run_recipe`` would
    silently append to the developer's real cache dir, mixing test
    runs into the production timeline. Tests that *want* to exercise
    the global mirror must monkeypatch ``PD_OCR_SYNTH_CACHE`` to a
    ``tmp_path`` and explicitly ``monkeypatch.delenv`` this var so the
    mirror is enabled inside the isolated cache root.
    """

    monkeypatch.setenv(GLOBAL_AUDIT_DISABLE_ENV, "1")


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
