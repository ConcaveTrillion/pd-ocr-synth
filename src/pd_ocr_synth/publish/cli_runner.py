"""CLI orchestration for ``pd-ocr-synth publish`` (M08).

This module is the glue between the argparse-built CLI surface in
``pd_ocr_synth.cli`` and the publish primitives that already exist
(staging build, preflight, summary, content-SHA, token resolver,
upload orchestrator). The split keeps ``cli.py`` short and lets each
flow (dry-run preview, real upload) be exercised in isolation by
tests that don't need to round-trip through ``main()``.

Per ``docs/specs/10-publishing.md`` and ``docs/roadmap/08-publishing-hf.md``:

- ``publish --dry-run`` shows what would be uploaded **without
  contacting HF**: target repo, file count, total size, dataset-card
  preview, content SHA. No ``huggingface_hub`` import; no network.
- Auth resolution attempts are reported by *source label only*
  ("flag", "env", "cache") — the token value is never echoed. A
  missing token is **not** fatal in dry-run mode (you can preview
  without credentials), only in real upload mode where the orchestrator
  needs a transport.
- Real upload (no ``--dry-run``) calls
  :func:`pd_ocr_synth.publish.publish_recognition` against the
  transport produced by ``transport_factory`` (default
  :func:`pd_ocr_synth.publish.make_default_transport`). The factory
  is *injectable* so tests substitute a
  :class:`pd_ocr_synth.publish.FakeTransport` and the production CLI
  default lazy-imports ``huggingface_hub``. Until the SDK adapter
  lands, the production factory raises
  :class:`pd_ocr_synth.publish.SdkUnavailableError`, which is a
  :class:`pd_ocr_synth.publish.TransportError` and therefore maps to
  the documented exit-7 publish failure.
- ``--render-first`` (spec 10 § When to publish) chains a render
  step in front of the publish pipeline so a single command produces
  the local layout AND ships it. The render callable is *injectable*
  so tests don't pay the real-rendering cost; the production default
  delegates to :func:`pd_ocr_synth.render.run_recipe`. Render failure
  short-circuits with exit 5 (the spec-01 RENDER_EXIT), distinct from
  publish-family failures (exit 7).

Dry-run remains the first user-visible publish surface: it exercises
every primitive that landed in earlier chunks end-to-end, gives users
a pre-flight they can run before committing to an upload, and makes
the format-conversion work auditable without an HF token.

The temp-staging strategy is shared by both modes — we never write
into ``recipe.output.destination``'s parent (which would be
presumptuous) and the temp dir is auto-cleaned. The cost is that the
staging build runs each time; that's deliberate — the SHA depends on
the staging contents, so reusing a stale staging dir would lie about
the digest.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pd_ocr_synth.publish.auth import (
    AuthError,
    ResolvedToken,
    format_resolution_chain,
    resolve_hf_token,
)
from pd_ocr_synth.publish.commit_message import resolve_commit_message
from pd_ocr_synth.publish.content_sha import ContentShaError
from pd_ocr_synth.publish.orchestrator import (
    PublishError,
    PublishResult,
    PublishState,
    publish_recognition,
)
from pd_ocr_synth.publish.preflight import (
    PreflightError,
    assert_staging_publish_ready,
)
from pd_ocr_synth.publish.recognition import (
    StagingError,
    build_recognition_staging,
)
from pd_ocr_synth.publish.sdk_transport import make_default_transport
from pd_ocr_synth.publish.summary import (
    SummaryError,
    format_summary,
    summarize_metadata,
)
from pd_ocr_synth.publish.transport import HfTransport, TransportError

# Type alias for the injectable transport factory. Production passes
# :func:`make_default_transport`; tests pass a lambda that closes over
# a :class:`FakeTransport`. Captured as a named alias so the public
# :func:`cmd_publish` signature is documented in one place rather than
# inlining a long callable type.
TransportFactory = Callable[[str], HfTransport]


# Type alias for the injectable render callable used by
# ``--render-first``. Implementation must accept the recipe path,
# the resolved output dir, and a cache_dir override (or ``None`` for
# the default). It runs to completion or raises — the runner does NOT
# pass count/seed/workers because spec 10 § When to publish describes
# ``--render-first`` as "render before publishing", i.e. produce
# whatever the recipe says (and recipe-level overrides via
# ``--output`` still flow through).
#
# The callable returns nothing on success. Any failure must raise so
# the runner can map it to RENDER_EXIT (5). The default production
# implementation, :func:`_default_render_first`, calls
# :func:`pd_ocr_synth.render.run_recipe` and re-raises its
# :class:`RenderError` / :class:`DestinationNotEmptyError` /
# :class:`SnapshotMismatchError` for the runner to catch.
RenderFirstCallable = Callable[[Path, Path, "Path | None"], None]

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

# User-facing notice for the ``--message`` / ``upload_large_folder``
# limitation. Spec 10 § Tooling used pins the SDK call to
# ``HfApi.upload_large_folder``, whose docstring (huggingface_hub
# 1.13.0, ``hf_api.py:5919-5921``) explicitly states:
#
#   > you cannot set a custom commit_message and commit_description
#   > since multiple commits are created.
#
# The flag is therefore accepted but **cannot** be stamped on the
# remote commit. We surface a single-line stderr warning when the
# user explicitly passes ``--message`` so the gap is visible at the
# command line rather than buried in the spec. The default-message
# path (no ``--message`` flag) does not warn — there is no user
# expectation to violate when the message was generated by us.
#
# Strategy A from ``docs/roadmap/08-publishing-hf.md`` § Residual M08
# work — picked over (D) "switch to upload_folder" because the spec
# deliberately mandates the resumable variant for large recognition
# datasets.
_MESSAGE_LIMITATION_WARNING = (
    "warning: --message accepted but huggingface_hub.upload_large_folder "
    "auto-generates per-shard commit messages; your message will not "
    "appear on the HF commit. See docs/specs/10-publishing.md "
    "§ Tooling used."
)


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
    license_override: str | None = None,
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
    license_override:
        Value of the ``--license`` CLI flag, or ``None``. Forwarded
        to :func:`build_recognition_staging` so the front-matter
        preview reflects the same license the real upload will stamp.

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

        result = build_recognition_staging(
            local_output_dir, staging, license_override=license_override
        )

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
    no_create: bool = False,
    tag: str | None = None,
    message: str | None = None,
    license_override: str | None = None,
    render_first: bool = False,
    transport_factory: TransportFactory | None = None,
    render_first_callable: RenderFirstCallable | None = None,
) -> int:
    """Top-level dispatch for ``pd-ocr-synth publish``.

    Returns the exit code. The argparse layer in ``pd_ocr_synth.cli``
    maps argparse's ``args.repo`` / ``args.dry_run`` / etc. directly
    onto the keyword params here so this function is the single place
    that knows about the publish exit-code mapping.

    Parameters
    ----------
    recipe_arg, repo_flag, private, public, token_flag, output_override, dry_run:
        Direct passthroughs from the argparse layer; see spec 01's
        ``publish`` section for semantics.
    no_create:
        Spec 10 § Errors and recovery: ``--no-create`` flips
        ``allow_create=False`` on the orchestrator. A missing repo
        then fails with exit 7 instead of being auto-created.
    tag:
        Spec 10 § Versioning: optional version tag created against
        the upload commit.
    message:
        Spec 10 § Versioning: ``--message <MSG>`` overrides the
        auto-generated commit message. ``None`` falls back to the
        ``pd-ocr-synth render @<recipe-sha>`` default.
    license_override:
        Spec 10 § Recipe ``publish:`` block: ``--license <LICENSE>``
        overrides ``recipe.publish.hf_dataset.license`` in the staged
        dataset card's front matter. ``None`` falls back to the
        recipe value (or omits the key entirely).
    render_first:
        Spec 10 § When to publish: ``--render-first`` chains a render
        step in front of the publish pipeline, equivalent to running
        ``pd-ocr-synth render <recipe>`` first. Errors from the
        render step short-circuit with exit 5 (RENDER_EXIT) — the
        publish pipeline never runs. Compatible with ``--dry-run``:
        rendering happens, then the dry-run preview is generated
        against the freshly-rendered output.
    transport_factory:
        Production callers leave this ``None`` to get the default
        :func:`make_default_transport` (which currently raises
        :class:`SdkUnavailableError` until the adapter lands; that
        error is a :class:`TransportError` and maps to exit 7). Tests
        inject a fake-returning factory to drive the full upload path
        hermetically.
    render_first_callable:
        Production callers leave this ``None`` to get
        :func:`_default_render_first` (which delegates to
        :func:`pd_ocr_synth.render.run_recipe`). Tests inject a
        no-op or recording stub so they don't pay the real-rendering
        cost; only the chaining contract is exercised.
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

    # ``--render-first`` chains the render step in front of publish
    # (spec 10 § When to publish). Run BEFORE the local-output
    # existence check — that check is what ``--render-first`` is
    # there to satisfy. Render failure short-circuits with the
    # spec-01 RENDER_EXIT (5), which is *distinct from* the
    # missing-render-without-flag exit-5 path so users keep getting
    # actionable messages either way.
    if render_first:
        render_callable = render_first_callable or _default_render_first
        try:
            render_callable(path, local_output, None)
        except Exception as exc:  # noqa: BLE001 — render exceptions are diverse
            print(f"error: render failed before publish: {exc}", file=sys.stderr)
            return PUBLISH_RENDER_EXIT

    if not local_output.is_dir():
        # Distinct hint depending on whether the user asked us to
        # render. Without ``--render-first`` we point at the spec's
        # canonical remediation; with the flag, the directory ought
        # to exist by now, so something went wrong upstream.
        if render_first:
            hint = (
                f"error: local render output still missing at {local_output} "
                "after --render-first. Did the render step write to a different path?"
            )
        else:
            hint = (
                f"error: local render output not found at {local_output}. "
                "Run `pd-ocr-synth render <recipe>` first or pass --output PATH."
            )
        print(hint, file=sys.stderr)
        return PUBLISH_RENDER_EXIT

    if dry_run:
        return _run_dry_run(
            local_output=local_output,
            repo=repo,
            resolved_private=resolved_private,
            token_flag=token_flag,
            license_override=license_override,
        )

    return _run_upload(
        local_output=local_output,
        repo=repo,
        resolved_private=resolved_private,
        token_flag=token_flag,
        no_create=no_create,
        tag=tag,
        message=message,
        license_override=license_override,
        transport_factory=transport_factory or make_default_transport,
    )


def _run_dry_run(
    *,
    local_output: Path,
    repo: str,
    resolved_private: bool,
    token_flag: str | None,
    license_override: str | None = None,
) -> int:
    """Execute the dry-run path and map its errors to exit codes.

    Split out so :func:`cmd_publish` remains a top-level dispatcher
    rather than a multi-page function. Each branch maps a typed
    exception onto the documented spec-01 exit code.
    """

    try:
        plan = run_publish_dry_run(
            local_output_dir=local_output,
            repo=repo,
            private=resolved_private,
            flag_token=token_flag,
            license_override=license_override,
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


def _run_upload(
    *,
    local_output: Path,
    repo: str,
    resolved_private: bool,
    token_flag: str | None,
    no_create: bool,
    tag: str | None,
    message: str | None,
    license_override: str | None,
    transport_factory: TransportFactory,
) -> int:
    """Execute the real upload path: build staging, resolve auth, dispatch.

    The sequence mirrors :func:`run_publish_dry_run` for the staging
    + preflight steps so a corrupt-local-output failure produces the
    same exit-6 outcome whether or not ``--dry-run`` was passed. From
    there we diverge: resolve the token (auth failure → exit 7),
    construct the transport via the injected factory (SDK unavailable
    → exit 7), and call the orchestrator. The orchestrator's typed
    exceptions all map to exit 7 (publish-family failures) per
    spec 01. The :class:`PublishResult` is rendered to stdout via
    :func:`format_publish_result`.
    """

    with tempfile.TemporaryDirectory(prefix="pd-ocr-synth-publish-") as tmp:
        staging = Path(tmp) / "staging"
        try:
            build_recognition_staging(local_output, staging, license_override=license_override)
        except StagingError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return PUBLISH_DESTINATION_EXIT

        try:
            preflight_report = assert_staging_publish_ready(staging)
        except PreflightError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return PUBLISH_DESTINATION_EXIT

        # Resolve auth. Real upload requires a token; we surface the
        # spec-mandated resolution chain so the user knows which knob
        # to turn.
        try:
            resolved_token: ResolvedToken = resolve_hf_token(flag_token=token_flag)
        except AuthError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return PUBLISH_AUTH_EXIT

        # Construct the transport. The factory is the seam where
        # production wires the SDK adapter and tests wire a fake.
        # ``SdkUnavailableError`` (TransportError subclass) lands
        # here when the SDK adapter isn't yet installed.
        try:
            transport = transport_factory(resolved_token.token)
        except TransportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return PUBLISH_AUTH_EXIT

        # Build the commit message: caller's --message, or the spec's
        # default ``pd-ocr-synth render @<recipe-sha>`` derived from
        # the staging README's pd-ocr-recipe-sha key.
        recipe_sha = preflight_report.front_matter.get("pd-ocr-recipe-sha")
        commit_message = resolve_commit_message(
            override=message,
            recipe_sha=recipe_sha if isinstance(recipe_sha, str) else None,
        )

        # Warn if the user explicitly supplied ``--message``: the
        # underlying ``upload_large_folder`` SDK call ignores it (the
        # chunked upload generates its own per-shard commit messages).
        # See ``_MESSAGE_LIMITATION_WARNING`` for the spec citation.
        # We mirror :func:`resolve_commit_message`'s "whitespace-only is
        # not provided" rule so a shell that expanded
        # ``--message "$VAR"`` to empty string doesn't trigger spurious
        # noise.
        if message is not None and message.strip():
            print(_MESSAGE_LIMITATION_WARNING, file=sys.stderr)

        try:
            result = publish_recognition(
                transport,
                repo,
                staging,
                commit_message=commit_message,
                private=resolved_private,
                allow_create=not no_create,
                tag=tag,
            )
        except PreflightError as exc:
            # The orchestrator re-runs preflight; if it fires here a
            # local edit happened between our check and its.
            print(f"error: {exc}", file=sys.stderr)
            return PUBLISH_DESTINATION_EXIT
        except ContentShaError as exc:
            # Could not hash the staging dir. Treat as corrupt local
            # output rather than an upload-family failure — the
            # remediation is "fix your render", not "fix your token".
            print(f"error: {exc}", file=sys.stderr)
            return PUBLISH_DESTINATION_EXIT
        except PublishError as exc:
            # Policy-level failure (e.g. --no-create + missing repo).
            print(f"error: {exc}", file=sys.stderr)
            return PUBLISH_AUTH_EXIT
        except TransportError as exc:
            # Network / auth / conflict during the actual upload.
            print(f"error: {exc}", file=sys.stderr)
            return PUBLISH_AUTH_EXIT

    print(format_publish_result(result))
    return PUBLISH_OK_EXIT


def format_publish_result(result: PublishResult) -> str:
    """Render a :class:`PublishResult` as the human-readable summary.

    Each :class:`PublishState` gets its own opening line so the user
    immediately knows what happened (no upload, created repo and
    uploaded, plain re-upload). The content SHA + commit URL follow
    so log scrapers can grab them. Tag, when present, is on the last
    line so a user who didn't ``--tag`` doesn't see clutter.

    Kept separate from :func:`run_publish_dry_run` / its formatter so
    a regression in either format doesn't cross-pollute the other.
    """

    lines: list[str] = []
    if result.state is PublishState.NO_CHANGE:
        lines.append(f"No changes to upload: {result.repo_id}")
        lines.append(f"  remote already has pd-ocr-content-sha={result.content_sha[:12]}")
    elif result.state is PublishState.CREATED:
        lines.append(f"Created and uploaded: {result.repo_id}")
        lines.append(f"  commit: {result.commit_sha[:12]}")
        if result.commit_url:
            lines.append(f"  url: {result.commit_url}")
        lines.append(f"  content-sha: {result.content_sha[:12]}")
    elif result.state is PublishState.UPLOADED:
        lines.append(f"Uploaded: {result.repo_id}")
        lines.append(f"  commit: {result.commit_sha[:12]}")
        if result.commit_url:
            lines.append(f"  url: {result.commit_url}")
        lines.append(f"  content-sha: {result.content_sha[:12]}")
    else:  # pragma: no cover — defensive; StrEnum is exhaustive
        lines.append(f"Publish completed in unexpected state {result.state!r}")

    if result.tag:
        lines.append(f"  tag: {result.tag}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_render_first(recipe_path: Path, output_dir: Path, cache_dir: Path | None) -> None:
    """Default ``--render-first`` callable.

    Loads the recipe at ``recipe_path``, then calls
    :func:`pd_ocr_synth.render.run_recipe` with the resolved
    ``output_dir`` (matches what the publish pipeline will read).

    Recipe-level overrides (count / seed / workers) are intentionally
    **not** plumbed here: spec 10 § When to publish describes
    ``--render-first`` as "render before publishing", i.e. run the
    recipe as-written. Users who want to render a smoke-sized subset
    should invoke ``render -c N`` separately and then plain
    ``publish`` — that way the chained form remains "ship a real
    recipe-shaped dataset" rather than an under-specified shorthand.

    Failures bubble up as their native exception types
    (:class:`RenderError`, :class:`DestinationNotEmptyError`,
    :class:`SnapshotMismatchError`); the CLI runner's broad
    ``except Exception`` catch in :func:`cmd_publish` maps them all to
    RENDER_EXIT.

    Implementation note on idempotency: the publish pipeline expects
    the destination to *already exist* (that's the whole reason
    ``--render-first`` is needed). On a re-run with this flag we
    therefore pass ``force=True`` so the render step can clear the
    destination and write fresh — otherwise the existing render's
    "non-empty destination" guard would refuse and exit 6, which would
    be surprising for someone who just asked us to chain render +
    publish.
    """

    # Local import: avoids paying the recipe-loader / render-graph
    # cost on every publish invocation, only when --render-first is
    # actually requested.
    from pd_ocr_synth.recipe import load_recipe
    from pd_ocr_synth.render import run_recipe

    recipe = load_recipe(recipe_path)
    run_recipe(
        recipe,
        output_dir=output_dir,
        cache_dir=cache_dir,
        force=True,
        progress=False,
    )


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
