"""Unit tests for ``pd_ocr_synth.publish.cli_runner`` (M08).

The CLI integration tests in ``test_cli_publish.py`` cover the full
argparse → dispatch → exit-code path. These tests exercise the
runner's pure helpers (size formatting, front-matter slicing, plan
formatting) directly — no recipe loader, no render, no CLI parsing —
so a regression in the small string utilities is caught locally
without re-running the heavier end-to-end tests.
"""

from __future__ import annotations

from pathlib import Path

from pd_ocr_synth.publish.cli_runner import (
    DryRunPlan,
    _format_size_mb,
    _front_matter_preview,
    _walk_dir_stats,
    format_dry_run_plan,
)

# ---------------------------------------------------------------------------
# _format_size_mb
# ---------------------------------------------------------------------------


def test_format_size_mb_uses_byte_units_for_small_sizes() -> None:
    assert _format_size_mb(0) == "0 B"
    assert _format_size_mb(500) == "500 B"


def test_format_size_mb_uses_kb_at_the_thousands_threshold() -> None:
    assert _format_size_mb(1_500) == "1.5 KB"


def test_format_size_mb_uses_mb_at_the_millions_threshold() -> None:
    # Spec example: ``247.3 MB`` for ~247 MB. We match the format.
    assert _format_size_mb(247_300_000) == "247.3 MB"


def test_format_size_mb_uses_gb_at_the_billions_threshold() -> None:
    assert _format_size_mb(2_500_000_000) == "2.50 GB"


# ---------------------------------------------------------------------------
# _front_matter_preview
# ---------------------------------------------------------------------------


def test_front_matter_preview_returns_only_the_fenced_block(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "---\n"
        "license: cc-by-4.0\n"
        "pd-ocr-shape: recognition/v1\n"
        "---\n"
        "\n"
        "# Body should not appear\n"
        "Lots of prose...\n",
        encoding="utf-8",
    )

    preview = _front_matter_preview(readme)
    assert "license: cc-by-4.0" in preview
    assert "pd-ocr-shape: recognition/v1" in preview
    # Body is stripped.
    assert "Body should not appear" not in preview
    # Both fences are kept so consumers can re-parse if they want.
    assert preview.startswith("---")
    assert preview.rstrip().endswith("---")


def test_front_matter_preview_handles_missing_readme(tmp_path: Path) -> None:
    assert _front_matter_preview(tmp_path / "missing.md") == "(no README.md)"


def test_front_matter_preview_handles_no_front_matter(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Just a body\n", encoding="utf-8")
    assert _front_matter_preview(readme) == "(no front matter)"


# ---------------------------------------------------------------------------
# _walk_dir_stats
# ---------------------------------------------------------------------------


def test_walk_dir_stats_counts_files_and_bytes_recursively(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"abcde")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"123")
    (tmp_path / "empty-dir").mkdir()  # not counted

    file_count, total_bytes = _walk_dir_stats(tmp_path)
    assert file_count == 2
    assert total_bytes == 8


# ---------------------------------------------------------------------------
# format_dry_run_plan
# ---------------------------------------------------------------------------


def test_format_dry_run_plan_includes_every_spec_section() -> None:
    """Per ``docs/specs/10-publishing.md`` § Dry run, the plan output
    must include the upload target, file count, dataset card preview,
    and content SHA. Each shows up here exactly once."""

    plan = DryRunPlan(
        repo="alice/dataset",
        visibility="public",
        file_count=10,
        total_bytes=1_500_000,
        content_sha="abcdef0123456789" + "0" * 48,
        front_matter_preview="---\nlicense: cc-by-4.0\n---",
        summary_block="Rows: 10\nFonts: 2 distinct",
        token_source="env",
        auth_chain=None,
    )

    text = format_dry_run_plan(plan)
    assert "Would upload to: alice/dataset (public)" in text
    assert "Files: 10" in text
    assert "1,500,000 bytes" in text
    assert "1.5 MB" in text
    assert "Auth: token resolved from env" in text
    assert "Dataset card preview:" in text
    assert "license: cc-by-4.0" in text
    assert "Manifest summary:" in text
    assert "Rows: 10" in text
    # Content SHA is truncated to 12 chars per the formatter.
    assert "Content SHA: abcdef012345" in text


def test_format_dry_run_plan_shows_auth_chain_when_no_token() -> None:
    plan = DryRunPlan(
        repo="alice/x",
        visibility="private",
        file_count=1,
        total_bytes=10,
        content_sha="0" * 64,
        front_matter_preview="---\n---",
        summary_block="Rows: 0",
        token_source=None,
        auth_chain="No Hugging Face token found.\n  1. --token <token>",
    )

    text = format_dry_run_plan(plan)
    assert "no token resolved" in text
    assert "1. --token <token>" in text
    assert "(private)" in text


def test_format_dry_run_plan_handles_empty_content_sha() -> None:
    """Defensive: an empty SHA (e.g. README couldn't be written because
    the snapshot was missing) should print a sentinel rather than a
    bare colon."""

    plan = DryRunPlan(
        repo="alice/x",
        visibility="public",
        file_count=0,
        total_bytes=0,
        content_sha="",
        front_matter_preview="(no README.md)",
        summary_block="Rows: 0",
        token_source="flag",
        auth_chain=None,
    )

    text = format_dry_run_plan(plan)
    assert "Content SHA: (none)" in text
