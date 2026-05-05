"""Smoke tests for the CLI surface.

M01 only ships argument parsing; behavior tests come with each
implementation milestone.
"""

from __future__ import annotations

import pytest

from pd_ocr_synth import __version__
from pd_ocr_synth.cli import build_parser, main

SUBCOMMANDS = [
    "init",
    "list",
    "validate",
    "describe",
    "fetch",
    "preview",
    "render",
    "publish",
    "clean",
]


def test_parser_builds() -> None:
    parser = build_parser()
    assert parser.prog == "pd-ocr-synth"


def test_no_args_prints_help_and_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "usage:" in captured.err.lower()


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_subcommand_help(subcommand: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Every subcommand exposes --help and exits 0."""
    with pytest.raises(SystemExit) as exc_info:
        main([subcommand, "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()


@pytest.mark.parametrize("subcommand", ["validate", "describe", "fetch", "preview", "clean"])
def test_subcommand_stub_returns_not_implemented(subcommand: str) -> None:
    """Subcommands that take only a recipe name run the stub and exit 1."""
    rc = main([subcommand, "dummy-recipe"])
    assert rc == 1


def test_list_stub_returns_not_implemented() -> None:
    rc = main(["list"])
    assert rc == 1


def test_init_stub_returns_not_implemented() -> None:
    rc = main(["init", "dummy"])
    assert rc == 1


def test_render_stub_returns_not_implemented() -> None:
    rc = main(["render", "dummy-recipe"])
    assert rc == 1


def test_publish_stub_returns_not_implemented() -> None:
    rc = main(["publish", "dummy-recipe"])
    assert rc == 1
