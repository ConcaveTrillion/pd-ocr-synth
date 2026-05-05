"""Command-line interface for pd-ocr-synth.

M02 wires ``list``, ``validate``, ``describe``, ``init``, and the new
``schema`` subcommand. Render-side subcommands (``fetch``, ``preview``,
``render``, ``publish``, ``clean``) remain stubs until the milestones
that own them.
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
