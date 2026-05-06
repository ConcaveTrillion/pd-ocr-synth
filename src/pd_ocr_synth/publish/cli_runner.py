"""CLI orchestration for ``pd-ocr-synth publish`` (M08).

This module is the glue between the argparse-built CLI surface in
``pd_ocr_synth.cli`` and the publish primitives that already exist
(staging build, preflight, summary, content-SHA, token resolver). The
split keeps ``cli.py`` short and lets the dry-run flow be exercised in
isolation by tests that don't need to round-trip through ``main()``.

Per ``docs/specs/10-publishing.md`` and ``docs/roadmap/08-publishing-hf.md``:

- ``publish --dry-run`` shows what would be uploaded **without
  contacting HF**: target repo, file count, total size, dataset-card
  preview, content SHA. No ``huggingface_hub`` import; no network.
- Auth resolution attempts are reported by *source label only*
  ("flag", "env", "cache") — the token value is never echoed. A
  missing token is **not** fatal in dry-run mode (you can preview
  without credentials), only in real upload mode (M08 later chunk).
- Real upload (no ``--dry-run``) is **not yet implemented** here. The
  CLI returns the documented exit-code-1 stub for that path until the
  upload chunk lands. (Spec 01 § Exit codes maps real publish failures
  to exit 7; the dry-run path returns 0 on success.)

Dry-run as the first user-visible publish surface is deliberate: it
exercises every primitive that landed in earlier chunks end-to-end,
gives users a pre-flight they can run before committing to an upload,
and makes the format-conversion work auditable without an HF token.

The temp-staging strategy lets ``--dry-run`` work even when a real
``--repo`` upload would refuse: we never write into the
``recipe.output.destination``'s parent (which would be presumptuous)
and the temp dir is auto-cleaned. The cost is that the staging build
runs each time; that's deliberate — the SHA depends on the staging
contents, so reusing a stale staging dir would lie about the digest.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pd_ocr_synth.publish.auth import (
    AuthError,
    ResolvedToken,
    format_resolution_chain,
    resolve_hf_token,
)
from pd_ocr_synth.publish.preflight import (
    PreflightError,
    assert_staging_publish_ready,
)
from pd_ocr_synth.publish.recognition import (
    StagingError,
    build_recognition_staging,
)
from pd_ocr_synth.publish.summary import (
    SummaryError,
    format_summary,
    summarize_metadata,
)

# Exit codes (must match ``docs/specs/01-cli.md``). Mirrored here as
# constants so the CLI dispatch and the runner share one source of
# truth — the CLI module imports these rather than re-declaring them.
PUBLISH_OK_EXIT = 0
PUBLISH_USAGE_EXIT = 2
PUBLISH_RENDER_EXIT = 5  # missing local render
PUBLISH_DESTINATION_EXIT = 6  # corrupt local output / preflight failure
PUBLISH_AUTH_EXIT = 7  # auth or repo-state failure


# Filenames the README front-matter preview keeps. Mirrors the dry-run
# example in ``docs/specs/10-publishing.md`` § Dry run, which truncates
# the body and shows just the front-matter block.
_FRONT_MATTER_FENCE = "---"


@dataclass(frozen=True)
class DryRunPlan:
    """Structured output of a successful dry-run.

    Returned for tests that want to assert on individual fields
    without pattern-matching against the printed text. The CLI
    converts this into the human-readable block via
    :func:`format_dry_run_plan`.
    """

    repo: str
    visibility: str
    file_count: int
    total_bytes: int
    content_sha: str
    front_matter_preview: str
    summary_block: str
    token_source: str | None
    auth_chain: str | None


def run_publish_dry_run(
    *,
    local_output_dir: Path,
    repo: str,
    private: bool | None,
    flag_token: str | None,
) -> DryRunPlan:
    """Execute the dry-run pipeline and return a structured plan.

    Pure function over filesystem inputs — no globals, no network.
    The CLI layer wraps this with argparse parsing and exception →
    exit-code mapping.

    Parameters
    ----------
    local_output_dir:
        Where the recognition writer dropped its layout. Must exist
        and contain at minimum ``labels.json`` + ``images/``; missing
        snapshot triggers a preflight failure (no README written).
    repo:
        The ``OWNER/NAME`` target. Resolved by the caller from
        ``--repo`` flag → recipe ``publish.hf_dataset.repo``.
    private:
        Tri-state visibility: ``True`` → "private", ``False`` →
        "public", ``None`` → "public" (spec default for synth).
    flag_token:
        Value of the ``--token`` CLI flag, or ``None``. Passed to the
        resolver so dry-run can report the resolution source
        ("flag" / "env" / "cache") without echoing the secret.

    Raises
    ------
    StagingError
        Local output is missing pieces the staging builder needs.
    PreflightError
        Staging built, but the README front matter is incomplete /
        the staging dir is structurally invalid.
    SummaryError
        ``metadata.jsonl`` is missing entirely after staging build.
    """

    # Build into a temp dir — the dry-run is read-only from the
    # user's perspective; we don't want to clobber anything next to
    # ``local_output_dir`` and we don't want to leave staging
    # artifacts on disk after the preview.
    with tempfile.TemporaryDirectory(prefix="pd-ocr-synth-publish-") as tmp:
        staging = Path(tmp) / "staging"

        result = build_recognition_staging(local_output_dir, staging)

        # Run the same preflight the upload step will run. A failure
        # here is the spec's "local output corrupt" case — exit 6.
        assert_staging_publish_ready(staging)

        # Now compute the upload-time facts from the staging dir.
        file_count, total_bytes = _walk_dir_stats(staging)
        front_matter_preview = _front_matter_preview(staging / "README.md")
        summary = summarize_metadata(staging)
        summary_block = format_summary(summary)

        # Token resolution. Failure is *not* fatal in dry-run; we
        # report the chain so the user can fix it before the real
        # upload, but they may legitimately preview without a token.
        token_source: str | None = None
        auth_chain: str | None = None
        try:
            resolved: ResolvedToken = resolve_hf_token(flag_token=flag_token)
            token_source = resolved.source
        except AuthError:
            auth_chain = format_resolution_chain()

        return DryRunPlan(
            repo=repo,
            visibility="private" if private else "public",
            file_count=file_count,
            total_bytes=total_bytes,
            content_sha=result.content_sha or "",
            front_matter_preview=front_matter_preview,
            summary_block=summary_block,
            token_source=token_source,
            auth_chain=auth_chain,
        )


def format_dry_run_plan(plan: DryRunPlan) -> str:
    """Render the plan as the human-readable block from spec 10.

    Mirrors the example in ``docs/specs/10-publishing.md`` § Dry run.
    Kept separate from :func:`run_publish_dry_run` so tests can assert
    on the structured plan without pattern-matching prose.
    """

    lines: list[str] = []
    lines.append(f"Would upload to: {plan.repo} ({plan.visibility})")
    lines.append(
        f"Files: {plan.file_count} "
        f"({plan.total_bytes:,} bytes, {_format_size_mb(plan.total_bytes)})"
    )
    if plan.token_source is not None:
        lines.append(f"Auth: token resolved from {plan.token_source}")
    else:
        lines.append("Auth: no token resolved (preview only)")
        if plan.auth_chain:
            for chain_line in plan.auth_chain.splitlines():
                lines.append(f"  {chain_line}")

    lines.append("Dataset card preview:")
    for fm_line in plan.front_matter_preview.splitlines():
        lines.append(f"  {fm_line}")

    lines.append("Manifest summary:")
    for s_line in plan.summary_block.splitlines():
        lines.append(f"  {s_line}")

    sha_short = plan.content_sha[:12] if plan.content_sha else "(none)"
    lines.append(f"Content SHA: {sha_short}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def cmd_publish(
    *,
    recipe_arg: str,
    repo_flag: str | None,
    private: bool,
    public: bool,
    token_flag: str | None,
    output_override: str | None,
    dry_run: bool,
) -> int:
    """Top-level dispatch for ``pd-ocr-synth publish``.

    Returns the exit code. The argparse layer in ``pd_ocr_synth.cli``
    maps argparse's ``args.repo`` / ``args.dry_run`` / etc. directly
    onto the keyword params here so this function is the single place
    that knows about the publish exit-code mapping.

    Real upload (``--dry-run`` False) is not implemented yet — the
    upload chunk lands later in M08. We return the canonical
    "not implemented" exit (1) in that case so existing M07-era
    tooling behavior is preserved.
    """

    if private and public:
        print("error: --private and --public are mutually exclusive", file=sys.stderr)
        return PUBLISH_USAGE_EXIT

    # Late imports keep the publish CLI from paying the recipe-loader
    # cost on ``--help`` / parser-only paths.
    from pd_ocr_synth.recipe import RecipeLoadError, load_recipe
    from pd_ocr_synth.recipe_search import RecipeNotFoundError, resolve_recipe

    try:
        path = resolve_recipe(recipe_arg)
    except RecipeNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        # Validation-family failure rather than publish-family — the
        # recipe couldn't be located at all, which is upstream of
        # anything HF-related.
        return 3  # VALIDATION_EXIT, mirrored from cli.py

    try:
        recipe = load_recipe(path)
    except RecipeLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"error: schema validation failed for {path}:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 3

    # Resolve the target repo. CLI flag wins; otherwise the recipe's
    # ``publish.hf_dataset.repo`` default; otherwise it's a usage
    # error per spec 10 § CLI summary "default from recipe".
    repo = repo_flag
    if repo is None and recipe.publish and recipe.publish.hf_dataset:
        repo = recipe.publish.hf_dataset.repo
    if repo is None:
        print(
            "error: no target repo. Pass --repo OWNER/NAME, or set "
            "publish.hf_dataset.repo in the recipe.",
            file=sys.stderr,
        )
        return PUBLISH_USAGE_EXIT

    # Visibility resolution. Spec 10 default for synth is public when
    # neither flag is set; the recipe's ``private: true`` carries
    # otherwise. Explicit ``--public`` overrides recipe ``private``.
    if public:
        resolved_private: bool = False
    elif private:
        resolved_private = True
    elif recipe.publish and recipe.publish.hf_dataset:
        resolved_private = recipe.publish.hf_dataset.private
    else:
        resolved_private = False

    local_output = (
        Path(output_override).expanduser() if output_override else recipe.output.destination
    )
    if not local_output.is_dir():
        print(
            f"error: local render output not found at {local_output}. "
            "Run `pd-ocr-synth render <recipe>` first or pass --output PATH.",
            file=sys.stderr,
        )
        return PUBLISH_RENDER_EXIT

    if not dry_run:
        # Real upload chunk hasn't landed. Keep the documented
        # "not implemented yet" exit so callers don't think we silently
        # uploaded.
        print(
            "publish: real upload is not implemented yet (see docs/roadmap/08-publishing-hf.md). "
            "Use --dry-run to preview the plan.",
            file=sys.stderr,
        )
        return 1

    try:
        plan = run_publish_dry_run(
            local_output_dir=local_output,
            repo=repo,
            private=resolved_private,
            flag_token=token_flag,
        )
    except StagingError as exc:
        # Local output is structurally incomplete. Spec 10 maps
        # corrupt local output to exit 6.
        print(f"error: {exc}", file=sys.stderr)
        return PUBLISH_DESTINATION_EXIT
    except PreflightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return PUBLISH_DESTINATION_EXIT
    except SummaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return PUBLISH_DESTINATION_EXIT

    print(format_dry_run_plan(plan))
    return PUBLISH_OK_EXIT


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _walk_dir_stats(root: Path) -> tuple[int, int]:
    """Count files and total bytes under ``root`` (recursive).

    Walked here rather than via ``shutil.disk_usage`` or
    ``os.walk(...).total`` because the staging dir is small enough
    (<1 GB typical) for a direct rglob to dominate, and we want byte
    counts that match what HF will actually upload (regular files
    only — symlinks materialized into copy2 already, no special
    handling needed).
    """

    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
    return file_count, total_bytes


def _front_matter_preview(readme_path: Path) -> str:
    """Slice the YAML front-matter block out of the README.

    Returns just the ``---``-fenced block (inclusive of the fences).
    The dry-run prints this prefix as the "Dataset card preview"
    section per spec 10's example. Body text is intentionally
    elided — the front matter is the contract; the body is prose.

    If the README is missing or has no front matter we return a
    sentinel rather than raising: preflight already caught those
    cases by the time we're here, so this is purely defensive.
    """

    if not readme_path.is_file():
        return "(no README.md)"
    text = readme_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_FENCE:
        return "(no front matter)"
    # Find the closing fence.
    out: list[str] = [lines[0]]
    for line in lines[1:]:
        out.append(line)
        if line.strip() == _FRONT_MATTER_FENCE:
            return "\n".join(out)
    return "\n".join(out)  # unterminated; show what we have


def _format_size_mb(size: int) -> str:
    """Render bytes as ``X.Y MB`` per spec 10's dry-run example.

    KB/MB/GB threshold rules favored over a generic humanize helper
    because the spec example uses ``247.3 MB`` literally. A staging
    dir with a few KB of metadata + no images would otherwise show
    ``0.0 MB`` which is unhelpful — fall through to KB / B for
    smaller sizes.
    """

    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.2f} GB"
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    if size >= 1_000:
        return f"{size / 1_000:.1f} KB"
    return f"{size} B"
