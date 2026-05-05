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
