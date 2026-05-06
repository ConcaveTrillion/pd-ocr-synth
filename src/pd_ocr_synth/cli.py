"""Command-line interface for pd-ocr-synth.

Subcommands wired to date:

- M02: ``list``, ``validate``, ``describe``, ``init``, ``schema``.
- M03: ``fetch``, ``clean`` (corpus cache management).
- M05: ``preview`` (render N samples to a preview directory).
- M07: ``render`` (full dataset → ``pd-ocr-trainer/v1`` recognition).
- M08: ``publish --dry-run`` (preview HF upload plan; real upload
  lands in a later chunk of M08).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pd_ocr_synth import __version__

NOT_IMPLEMENTED_EXIT = 1
USAGE_EXIT = 2
VALIDATION_EXIT = 3
RENDER_EXIT = 5
DESTINATION_EXIT = 6


# ---------------------------------------------------------------------------
# Stub helper (used by render-side subcommands until their milestones land)
# ---------------------------------------------------------------------------


def _stub(name: str) -> int:
    print(f"{name}: not implemented yet (see docs/roadmap/)", file=sys.stderr)
    return NOT_IMPLEMENTED_EXIT


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def _add_recipe_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "recipe",
        help="recipe name (resolved on the recipe search path) or path to a YAML file",
    )


def _add_common_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--count", type=int, help="override sample count from the recipe")
    parser.add_argument("-o", "--output", help="override output destination")
    parser.add_argument("-s", "--seed", type=int, help="override random seed")
    parser.add_argument("-w", "--workers", type=int, help="parallel render workers")
    parser.add_argument("--cache-dir", help="corpus cache root")
    parser.add_argument("--no-cache", action="store_true", help="bypass corpus cache")
    parser.add_argument("--dry-run", action="store_true", help="validate + plan only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pd-ocr-synth",
        description="Synthetic OCR training-data generator (recipe-driven).",
    )
    parser.add_argument("--version", action="version", version=f"pd-ocr-synth {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<subcommand>")

    p_init = subparsers.add_parser("init", help="scaffold a new recipe")
    p_init.add_argument("name", help="recipe name to create")
    p_init.add_argument(
        "--dir",
        default="recipes",
        help="directory to scaffold under (default: ./recipes)",
    )
    p_init.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing recipe with the same name",
    )

    subparsers.add_parser("list", help="list recipes on the recipe search path")

    p_validate = subparsers.add_parser("validate", help="schema-check a recipe")
    _add_recipe_arg(p_validate)
    p_validate.add_argument(
        "--offline",
        action="store_true",
        help="skip network-touching checks (M03+)",
    )

    p_describe = subparsers.add_parser(
        "describe", help="print resolved config + corpus stats for a recipe"
    )
    _add_recipe_arg(p_describe)
    p_describe.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )

    p_schema = subparsers.add_parser(
        "schema",
        help="emit the recipe JSON Schema (default writes docs/specs/recipe.schema.json)",
    )
    p_schema.add_argument(
        "-o",
        "--output",
        help="write the schema to this path instead of stdout",
    )

    p_fetch = subparsers.add_parser("fetch", help="pre-fetch and cache web corpora for a recipe")
    _add_recipe_arg(p_fetch)
    _add_common_render_args(p_fetch)

    p_preview = subparsers.add_parser("preview", help="render N samples to a preview directory")
    _add_recipe_arg(p_preview)
    _add_common_render_args(p_preview)
    p_preview.add_argument(
        "--no-degrade",
        action="store_true",
        help="skip the recipe's degradation pipeline; output raw render only",
    )

    p_render = subparsers.add_parser("render", help="render the full dataset for a recipe")
    _add_recipe_arg(p_render)
    _add_common_render_args(p_render)
    p_render.add_argument("--force", action="store_true", help="clear destination before render")
    p_render.add_argument("--resume", action="store_true", help="resume an interrupted render")

    p_publish = subparsers.add_parser("publish", help="upload rendered output to a HF dataset repo")
    _add_recipe_arg(p_publish)
    p_publish.add_argument("--repo", help="OWNER/NAME on the Hugging Face Hub")
    p_publish.add_argument("--private", action="store_true")
    p_publish.add_argument("--public", action="store_true")
    p_publish.add_argument("--license")
    p_publish.add_argument("--tag")
    p_publish.add_argument("--message")
    p_publish.add_argument("--token")
    # ``--output`` is not in spec 10's CLI summary because the spec's
    # default is ``recipe.output.destination`` and most users never need
    # to override. We accept it here so dry-run / publish can target a
    # one-off render at e.g. ``/tmp/...`` without editing the recipe.
    p_publish.add_argument("-o", "--output", help="override local render output path")
    p_publish.add_argument("--render-first", action="store_true")
    p_publish.add_argument("--no-create", action="store_true")
    p_publish.add_argument("--dry-run", action="store_true")

    p_clean = subparsers.add_parser("clean", help="remove cached corpora for a recipe")
    _add_recipe_arg(p_clean)
    p_clean.add_argument(
        "--cache-dir",
        help="cache root (default: $PD_OCR_SYNTH_CACHE or ~/.cache/pd-ocr-synth)",
    )

    return parser


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_list() -> int:
    from pd_ocr_synth.recipe_search import iter_recipes

    entries = iter_recipes()
    if not entries:
        print("(no recipes found on the search path)", file=sys.stderr)
        return 0
    width = max(len(e.name) for e in entries)
    for entry in entries:
        print(f"{entry.name:<{width}}  {entry.path}")
    return 0


def _cmd_validate(recipe_arg: str, *, offline: bool) -> int:
    from pd_ocr_synth.recipe import RecipeLoadError, load_recipe
    from pd_ocr_synth.recipe_search import RecipeNotFoundError, resolve_recipe
    from pd_ocr_synth.validation import validate_recipe

    try:
        path = resolve_recipe(recipe_arg)
    except RecipeNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return VALIDATION_EXIT

    try:
        recipe = load_recipe(path)
    except RecipeLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return VALIDATION_EXIT
    except Exception as exc:
        # pydantic.ValidationError lands here.
        print(f"error: schema validation failed for {path}:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return VALIDATION_EXIT

    report = validate_recipe(recipe, offline=offline)
    for issue in report.issues:
        stream = sys.stderr if issue.severity == "error" else sys.stdout
        print(issue.format(), file=stream)
    if report.is_ok:
        print(f"OK: {recipe.name} ({path})")
        return 0
    return VALIDATION_EXIT


def _cmd_describe(recipe_arg: str, *, output_format: str) -> int:
    from pd_ocr_synth.recipe import RecipeLoadError, load_recipe
    from pd_ocr_synth.recipe_search import RecipeNotFoundError, resolve_recipe

    try:
        path = resolve_recipe(recipe_arg)
    except RecipeNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return VALIDATION_EXIT

    try:
        recipe = load_recipe(path)
    except RecipeLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return VALIDATION_EXIT
    except Exception as exc:
        print(f"error: schema validation failed for {path}:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return VALIDATION_EXIT

    payload = recipe.model_dump(mode="json")
    if output_format == "json":
        print(json.dumps(payload, indent=2, sort_keys=False))
    else:
        print(f"recipe: {recipe.name}")
        print(f"source: {recipe.source_path}")
        print(f"schema_version: {recipe.schema_version}")
        print(f"output.mode: {recipe.output.mode}")
        print(f"output.destination: {recipe.output.destination}")
        print(f"output.count: {recipe.output.count}")
        print(f"corpus: {len(recipe.corpus)} entries (not fetched)")
        print(f"text_transforms: {len(recipe.text_transforms)}")
        print(f"fonts: {len(recipe.fonts)}")
        print(f"layout.mode: {recipe.layout.mode}")
        print(f"degradation: {len(recipe.degradation)} stages")
        if recipe.publish and recipe.publish.hf_dataset:
            print(f"publish.hf_dataset.repo: {recipe.publish.hf_dataset.repo}")
        print()
        print("--- resolved config (json) ---")
        print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


def _cmd_init(name: str, *, dir_: str, force: bool) -> int:
    from pd_ocr_synth.recipe_init import scaffold_recipe

    target_dir = Path(dir_) / name
    if target_dir.exists() and not force:
        print(
            f"error: {target_dir} already exists; use --force to overwrite",
            file=sys.stderr,
        )
        return USAGE_EXIT

    written = scaffold_recipe(name=name, target_dir=target_dir)
    print(f"created recipe '{name}' at {target_dir}")
    for path in written:
        print(f"  + {path.relative_to(target_dir.parent)}")
    return 0


def _cmd_fetch(recipe_arg: str, *, cache_dir: str | None, no_cache: bool) -> int:
    import time

    from pd_ocr_synth.corpus import (
        CacheStore,
        CorpusError,
        ProviderContext,
        default_cache_root,
        default_registry,
    )
    from pd_ocr_synth.corpus.filters import apply_filter
    from pd_ocr_synth.recipe import RecipeLoadError, load_recipe
    from pd_ocr_synth.recipe_search import RecipeNotFoundError, resolve_recipe

    try:
        path = resolve_recipe(recipe_arg)
    except RecipeNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return VALIDATION_EXIT

    try:
        recipe = load_recipe(path)
    except RecipeLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return VALIDATION_EXIT

    cache_root = Path(cache_dir).expanduser() if cache_dir else default_cache_root()
    cache = CacheStore(root=cache_root)
    ctx = ProviderContext(recipe_dir=path.parent, cache=cache)
    registry = default_registry()

    print(f"recipe: {recipe.name} ({path})")
    print(f"cache:  {cache_root}")
    print()

    total_chars = 0
    failures = 0
    for index, entry in enumerate(recipe.corpus):
        options = entry.model_dump(mode="python")
        if no_cache:
            options["cache"] = False

        try:
            provider = registry.get(entry.type)
        except CorpusError as exc:
            failures += 1
            print(
                f"  corpus[{index}] {entry.type}: ERROR {exc}",
                file=sys.stderr,
            )
            continue

        cache_key = provider.cache_key(options)
        was_cached = cache.has(provider.type_name, cache_key)
        started = time.monotonic()
        try:
            chunks = list(provider.fetch(ctx, options))
        except CorpusError as exc:
            failures += 1
            print(
                f"  corpus[{index}] {entry.type}: ERROR {exc}",
                file=sys.stderr,
            )
            continue
        elapsed = time.monotonic() - started

        text = apply_filter("\n".join(chunks), options.get("filter"))
        chars = len(text)
        total_chars += chars
        marker = "cache" if was_cached and not no_cache else "fetch"
        print(
            f"  corpus[{index}] {entry.type}: {marker} "
            f"{chars:>9d} chars  {elapsed:>5.2f}s  ({cache_key})"
        )

    print()
    print(f"total: {total_chars:,} chars across {len(recipe.corpus)} entries")
    if failures:
        print(f"failures: {failures}", file=sys.stderr)
        return 4  # CORPUS_EXIT per docs/specs/01-cli.md
    return 0


def _cmd_preview(
    recipe_arg: str,
    *,
    count: int | None,
    output: str | None,
    seed: int | None,
    cache_dir: str | None,
    workers: int | None,
    no_degrade: bool,
) -> int:
    """Render N samples to a preview directory.

    Default count: ``DEFAULT_PREVIEW_COUNT`` (50).
    Default output: ``./preview/<recipe-name>/``.
    Default workers: ``max(1, min(cpu_count - 1, 8))`` — see
    :func:`pd_ocr_synth.render.preview.resolve_workers`.

    Degradation: applied by default (M06). Pass ``--no-degrade`` to
    skip the pipeline and inspect raw render output.
    """

    from pd_ocr_synth.recipe import RecipeLoadError, load_recipe
    from pd_ocr_synth.recipe_search import RecipeNotFoundError, resolve_recipe
    from pd_ocr_synth.render import RenderError
    from pd_ocr_synth.render.preview import (
        DEFAULT_PREVIEW_COUNT,
        resolve_workers,
        run_preview,
    )

    try:
        path = resolve_recipe(recipe_arg)
    except RecipeNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return VALIDATION_EXIT

    try:
        recipe = load_recipe(path)
    except RecipeLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return VALIDATION_EXIT
    except Exception as exc:
        print(f"error: schema validation failed for {path}:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return VALIDATION_EXIT

    sample_count = count if count is not None else DEFAULT_PREVIEW_COUNT
    if sample_count <= 0:
        print(f"error: --count must be positive (got {sample_count})", file=sys.stderr)
        return USAGE_EXIT

    if workers is not None and workers <= 0:
        print(f"error: --workers must be positive (got {workers})", file=sys.stderr)
        return USAGE_EXIT
    worker_count = resolve_workers(workers)

    output_dir = Path(output).expanduser() if output else Path("preview") / recipe.name
    cache_root = Path(cache_dir).expanduser() if cache_dir else None

    print(f"recipe:  {recipe.name} ({path})")
    print(f"output:  {output_dir}")
    print(f"count:   {sample_count}")
    print(f"workers: {worker_count}")

    try:
        stats = run_preview(
            recipe,
            output_dir=output_dir,
            count=sample_count,
            seed=seed,
            cache_dir=cache_root,
            workers=worker_count,
            apply_degrade=not no_degrade,
        )
    except RenderError as exc:
        print(f"error: render failed: {exc}", file=sys.stderr)
        return 5  # RENDER_EXIT per docs/specs/01-cli.md

    print()
    print(f"rendered: {stats.rendered}/{stats.count}")
    if stats.skipped:
        print(f"skipped:  {stats.skipped}")
        for reason, n in sorted(stats.skip_reasons.items()):
            print(f"  {reason}: {n}")
    print(f"manifest: {output_dir / 'manifest.jsonl'}")
    return 0


def _cmd_render(
    recipe_arg: str,
    *,
    count: int | None,
    output: str | None,
    seed: int | None,
    cache_dir: str | None,
    workers: int | None,
    force: bool,
    resume: bool,
    dry_run: bool,
) -> int:
    """Render the full recipe dataset into the ``pd-ocr-trainer/v1`` layout.

    Default output: ``recipe.output.destination`` from the YAML.
    Pass ``-o`` to override (handy for smoke runs into ``/tmp``).

    ``--force`` and ``--resume`` are mutually exclusive. Default
    behavior on a non-empty destination is to refuse and exit 6.
    Snapshot mismatch on ``--resume`` exits 6 too — same family of
    "destination is in a state I won't silently clobber."
    """

    from pd_ocr_synth.output import RecognitionWriter
    from pd_ocr_synth.output.recognition import DestinationNotEmptyError
    from pd_ocr_synth.output.snapshot import SnapshotMismatchError
    from pd_ocr_synth.recipe import RecipeLoadError, load_recipe
    from pd_ocr_synth.recipe_search import RecipeNotFoundError, resolve_recipe
    from pd_ocr_synth.render import RenderError, plan_recipe, run_recipe
    from pd_ocr_synth.render.preview import resolve_workers

    _ = RecognitionWriter  # imported for re-use; quiets the linter

    if force and resume:
        print("error: --force and --resume are mutually exclusive", file=sys.stderr)
        return USAGE_EXIT

    try:
        path = resolve_recipe(recipe_arg)
    except RecipeNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return VALIDATION_EXIT

    try:
        recipe = load_recipe(path)
    except RecipeLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return VALIDATION_EXIT
    except Exception as exc:
        print(f"error: schema validation failed for {path}:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return VALIDATION_EXIT

    sample_count = count if count is not None else recipe.output.count
    if sample_count <= 0:
        print(f"error: --count must be positive (got {sample_count})", file=sys.stderr)
        return USAGE_EXIT

    if workers is not None and workers <= 0:
        print(f"error: --workers must be positive (got {workers})", file=sys.stderr)
        return USAGE_EXIT
    worker_count = resolve_workers(workers)

    output_dir = Path(output).expanduser() if output else Path(recipe.output.destination)
    cache_root = Path(cache_dir).expanduser() if cache_dir else None

    if dry_run:
        try:
            plan = plan_recipe(
                recipe,
                output_dir=output_dir,
                count=sample_count,
                seed=seed,
                workers=worker_count,
                cache_dir=cache_root,
            )
        except RenderError as exc:
            print(f"error: dry-run failed: {exc}", file=sys.stderr)
            return RENDER_EXIT
        print(f"recipe:           {plan.recipe_name}")
        print(f"output:           {plan.output_dir}")
        print(f"count:            {plan.count}")
        print(f"seed:             {plan.seed}")
        print(f"workers:          {plan.workers}")
        print(f"layout.mode:      {plan.layout_mode}")
        print(
            f"fonts present:    {plan.fonts_present} (optional missing: {plan.fonts_missing_optional})"
        )
        print(f"transforms:       {', '.join(plan.transforms) or '-'}")
        print(f"degradation:      {', '.join(plan.degradation_stages) or '-'}")
        print(f"corpus entries:   {plan.corpus_entries}")
        print(f"corpus chars:     {plan.corpus_total_chars:,}")
        return 0

    print(f"recipe:  {recipe.name} ({path})")
    print(f"output:  {output_dir}")
    print(f"count:   {sample_count}")
    print(f"workers: {worker_count}")
    if force:
        print("mode:    --force (destination will be cleared)")
    elif resume:
        print("mode:    --resume (continuing from existing snapshot)")

    try:
        result = run_recipe(
            recipe,
            output_dir=output_dir,
            count=sample_count,
            seed=seed,
            workers=worker_count,
            cache_dir=cache_root,
            force=force,
            resume=resume,
        )
    except DestinationNotEmptyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return DESTINATION_EXIT
    except SnapshotMismatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return DESTINATION_EXIT
    except RenderError as exc:
        print(f"error: render failed: {exc}", file=sys.stderr)
        return RENDER_EXIT

    print()
    print(f"rendered: {result.rendered}/{sample_count}")
    if result.skipped:
        print(f"skipped:  {result.skipped}")
        for reason, n in sorted(result.skip_reasons.items()):
            print(f"  {reason}: {n}")
    print(f"wall:     {result.wall_time_seconds:.1f}s")
    return 0


def _cmd_publish(
    recipe_arg: str,
    *,
    repo: str | None,
    private: bool,
    public: bool,
    token: str | None,
    output: str | None,
    dry_run: bool,
    no_create: bool,
    tag: str | None,
    message: str | None,
) -> int:
    """Dispatch ``publish`` (M08).

    Both dry-run and real upload paths are implemented. Real upload
    requires the ``huggingface_hub`` SDK; until the adapter chunk
    lands, the production transport factory raises
    :class:`pd_ocr_synth.publish.SdkUnavailableError` (a
    :class:`TransportError`) and the runner maps it to exit 7 with
    a clear remediation message. Exit-code mapping matches
    ``docs/specs/01-cli.md`` (canonical) — spec 10 was reconciled in
    the dry-run dispatch commit.
    """

    from pd_ocr_synth.publish.cli_runner import cmd_publish

    return cmd_publish(
        recipe_arg=recipe_arg,
        repo_flag=repo,
        private=private,
        public=public,
        token_flag=token,
        output_override=output,
        dry_run=dry_run,
        no_create=no_create,
        tag=tag,
        message=message,
    )


def _cmd_clean(recipe_arg: str, *, cache_dir: str | None) -> int:
    """Remove cache entries owned by the given recipe.

    For each corpus entry we look up the provider, ask it for its
    ``cache_key(options)``, and remove the on-disk pair. Entries
    whose providers are unknown (e.g. provider not yet implemented)
    are surfaced as warnings — they aren't fatal because the
    recipe might still be usable for other purposes.
    """

    from pd_ocr_synth.corpus import (
        CacheStore,
        CorpusError,
        default_cache_root,
        default_registry,
    )
    from pd_ocr_synth.recipe import RecipeLoadError, load_recipe
    from pd_ocr_synth.recipe_search import RecipeNotFoundError, resolve_recipe

    try:
        path = resolve_recipe(recipe_arg)
    except RecipeNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return VALIDATION_EXIT

    try:
        recipe = load_recipe(path)
    except RecipeLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return VALIDATION_EXIT

    cache_root = Path(cache_dir).expanduser() if cache_dir else default_cache_root()
    cache = CacheStore(root=cache_root)
    registry = default_registry()

    print(f"recipe: {recipe.name} ({path})")
    print(f"cache:  {cache_root}")
    print()

    removed = 0
    skipped = 0
    for index, entry in enumerate(recipe.corpus):
        options = entry.model_dump(mode="python")
        try:
            provider = registry.get(entry.type)
        except CorpusError:
            print(
                f"  corpus[{index}] {entry.type}: SKIP (provider not registered)",
                file=sys.stderr,
            )
            skipped += 1
            continue
        cache_key = provider.cache_key(options)
        if cache.remove(provider.type_name, cache_key):
            print(f"  corpus[{index}] {entry.type}: removed {cache_key}")
            removed += 1
        else:
            print(f"  corpus[{index}] {entry.type}: nothing to remove ({cache_key})")
    print()
    print(f"removed: {removed}  skipped: {skipped}")
    return 0


def _cmd_schema(output: str | None) -> int:
    from pd_ocr_synth.recipe.models import Recipe

    schema = Recipe.model_json_schema()
    text = json.dumps(schema, indent=2, sort_keys=False)
    if output is None:
        print(text)
        return 0
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {target}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_IMPLEMENTED_DISPATCH = {
    "list": lambda args: _cmd_list(),
    "validate": lambda args: _cmd_validate(args.recipe, offline=args.offline),
    "describe": lambda args: _cmd_describe(args.recipe, output_format=args.format),
    "init": lambda args: _cmd_init(args.name, dir_=args.dir, force=args.force),
    "schema": lambda args: _cmd_schema(args.output),
    "fetch": lambda args: _cmd_fetch(
        args.recipe,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
    ),
    "preview": lambda args: _cmd_preview(
        args.recipe,
        count=args.count,
        output=args.output,
        seed=args.seed,
        cache_dir=args.cache_dir,
        workers=args.workers,
        no_degrade=args.no_degrade,
    ),
    "render": lambda args: _cmd_render(
        args.recipe,
        count=args.count,
        output=args.output,
        seed=args.seed,
        cache_dir=args.cache_dir,
        workers=args.workers,
        force=args.force,
        resume=args.resume,
        dry_run=args.dry_run,
    ),
    "publish": lambda args: _cmd_publish(
        args.recipe,
        repo=args.repo,
        private=args.private,
        public=args.public,
        token=args.token,
        output=args.output,
        dry_run=args.dry_run,
        no_create=args.no_create,
        tag=args.tag,
        message=args.message,
    ),
    "clean": lambda args: _cmd_clean(args.recipe, cache_dir=args.cache_dir),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return USAGE_EXIT
    handler = _IMPLEMENTED_DISPATCH.get(args.command)
    if handler is not None:
        return handler(args)
    return _stub(args.command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
