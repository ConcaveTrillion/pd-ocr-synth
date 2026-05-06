"""Dataset-card README generator for the HF imagefolder staging dir.

Per ``docs/specs/10-publishing.md``: the staging dir ships a generated
``README.md`` with YAML front matter that carries the ``pd-ocr-*``
conventional keys (see ``DATASETS.md`` workspace contract) plus enough
human-readable body that a viewer landing on the HF dataset page knows
what they're looking at.

Inputs (all read from local disk; no recipe object required, no
network):

- ``recipe.snapshot.yaml`` — the resolved recipe written by the M07
  writer. We read ``recipe.publish.hf_dataset`` for license / tags /
  language / description-file overrides, ``recipe.name`` /
  ``recipe.description`` / ``recipe.fonts`` for the body, and the file
  itself (its bytes) for the recipe-SHA front-matter key.
- ``stats.json`` — run counters (samples, fonts used, tokens unique,
  render wall time). Optional: an old run / a partial render may not
  have one, in which case the stats section degrades gracefully.

Output: a single string that the staging builder writes to
``<staging>/README.md``. Generation is pure (no I/O): the helper
functions take already-loaded dicts and the staging builder owns the
read+write boilerplate. That split keeps tests fast and avoids
re-reading files when the future ``--dry-run`` path wants to preview
the card.

The conventional ``pd-ocr-*`` keys mirror the spec table:

- ``pd-ocr-shape: recognition/v1`` — fixed for recognition mode.
- ``pd-ocr-source: pd-ocr-synth`` — fixed; this tool is the producer.
- ``pd-ocr-recipe-sha`` — SHA-256 of the snapshot YAML bytes. The
  snapshot is the canonical resolved recipe (presets expanded, paths
  absolute), so its hash is the right reproducibility pin.
- ``pd-ocr-render-tool-version`` — copied from
  ``snapshot.tool_version``; ties the dataset to the synth release
  that generated it.

``pd-ocr-content-sha`` (idempotency check on the staging dir contents)
is *not* set here — it's an upload-time concern computed after the
staging build completes. Leaving it out of the README written by the
staging step means re-reading and re-writing the README when the
upload code lands; that's a deliberate trade so this chunk stays
network-free and self-contained.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pd_ocr_synth.output.snapshot import SNAPSHOT_FILENAME

# README written into the staging dir. Centralized so the upload code
# (which will rewrite the front matter to add ``pd-ocr-content-sha``)
# and the tests share the same constant.
README_FILENAME = "README.md"

# Default values per spec 10. ``shape`` is the workspace-wide DATASETS.md
# contract for recognition output; ``source`` identifies the producer
# tool so consumers can route different shapes through different
# loaders. Detection-mode staging overrides ``shape`` / ``task`` via
# :class:`DatasetCardInputs.shape` and
# :class:`DatasetCardInputs.task_categories`; the defaults match
# recognition so existing call sites stay untouched.
_DEFAULT_SHAPE = "recognition/v1"
_SOURCE = "pd-ocr-synth"
_DEFAULT_TASK_CATEGORIES: tuple[str, ...] = ("text-recognition",)

# HF size_categories buckets. Picked the closest matching bracket to
# ``samples`` so the dataset shows up under the right size filter on
# the HF hub. These are the canonical buckets the HF docs list.
_SIZE_BUCKETS: tuple[tuple[int, str], ...] = (
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
)
_SIZE_OVERFLOW = "n>10M"


@dataclass(slots=True)
class DatasetCardInputs:
    """Bundle of already-loaded dicts the renderer needs.

    Plain dataclass rather than recipe / pydantic models so the
    renderer is decoupled from the recipe schema's evolution. The
    staging builder loads + adapts; this layer just shapes text.

    ``license_override`` is the spec-10 ``--license`` CLI flag value:
    when set it wins over ``recipe.publish.hf_dataset.license`` per
    spec 10 § Recipe ``publish:`` block ("CLI flags override recipe
    values when both are present"). ``None`` falls back to the recipe
    value (or omits the key entirely if neither is set).
    """

    snapshot: dict[str, Any]
    snapshot_bytes: bytes
    stats: dict[str, Any] | None = None
    description_override: str | None = None
    license_override: str | None = None
    # ``shape`` lands as ``pd-ocr-shape`` in the front matter. Default
    # is the recognition-mode contract; the detection-mode staging
    # builder overrides this to ``"detection/v1"`` so consumers can
    # route the two shapes through different loaders (per
    # ``DATASETS.md``).
    shape: str = _DEFAULT_SHAPE
    # ``task_categories`` lands as the HF front-matter key of the same
    # name. Default matches recognition; detection mode overrides to
    # ``["object-detection"]`` per HF's task taxonomy.
    task_categories: tuple[str, ...] | list[str] = _DEFAULT_TASK_CATEGORIES


def render_dataset_card(inputs: DatasetCardInputs) -> str:
    """Render the README.md text for the staging dir.

    The output is a YAML front-matter block followed by a Markdown
    body. The front matter keys are ordered for readable diffs across
    runs (HF doesn't care about order, but humans reading
    `git diff README.md` do).
    """

    front_matter = _front_matter(inputs)
    body = _body(inputs)
    fm_text = yaml.safe_dump(
        front_matter,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).rstrip("\n")
    return f"---\n{fm_text}\n---\n\n{body}\n"


def write_dataset_card(staging_dir: Path, inputs: DatasetCardInputs) -> Path:
    """Write the rendered card to ``<staging_dir>/README.md``.

    Returns the written path. The staging-build orchestrator calls
    this after copying images / metadata / snapshot.
    """

    target = staging_dir / README_FILENAME
    target.write_text(render_dataset_card(inputs), encoding="utf-8")
    return target


def load_card_inputs(
    local_output_dir: Path,
    *,
    license_override: str | None = None,
) -> DatasetCardInputs:
    """Read snapshot + stats from a local recognition output.

    Convenience for the staging builder; tests typically construct
    :class:`DatasetCardInputs` directly so the renderer can be
    exercised without writing a snapshot file. Returns an "empty-ish"
    inputs object if the snapshot is missing — the caller decides
    whether to skip card generation in that case.

    Parameters
    ----------
    license_override:
        Forwarded to :class:`DatasetCardInputs.license_override`. The
        staging builder threads ``--license`` through here so the flag
        wins over the recipe-declared license per spec 10.
    """

    snapshot_path = local_output_dir / SNAPSHOT_FILENAME
    snapshot_bytes = b""
    snapshot: dict[str, Any] = {}
    if snapshot_path.is_file():
        snapshot_bytes = snapshot_path.read_bytes()
        loaded = yaml.safe_load(snapshot_bytes.decode("utf-8"))
        if isinstance(loaded, dict):
            snapshot = loaded

    stats_path = local_output_dir / "stats.json"
    stats: dict[str, Any] | None = None
    if stats_path.is_file():
        try:
            import json

            loaded_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupt stats shouldn't block card generation. The Stats
            # section will just degrade gracefully.
            loaded_stats = None
        if isinstance(loaded_stats, dict):
            stats = loaded_stats

    description_override = _resolve_description_override(snapshot, local_output_dir)

    return DatasetCardInputs(
        snapshot=snapshot,
        snapshot_bytes=snapshot_bytes,
        stats=stats,
        description_override=description_override,
        license_override=license_override,
    )


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------


def _front_matter(inputs: DatasetCardInputs) -> dict[str, Any]:
    """Assemble the ordered front-matter dict.

    Built as a plain dict literal in the desired order so PyYAML
    preserves it under ``sort_keys=False``. Empty-ish values are
    omitted rather than written as ``null`` / ``[]``: HF's parquet
    inference and lint rules are friendlier to absent keys than to
    typed nulls.
    """

    recipe = _recipe_from_snapshot(inputs.snapshot)
    publish = _hf_publish_block(recipe)

    fm: dict[str, Any] = {}

    # ``--license`` flag (carried as ``license_override``) wins over
    # ``recipe.publish.hf_dataset.license``. Falsy overrides ("" or
    # whitespace-only) fall through to the recipe value rather than
    # writing a blank ``license:`` to the front matter.
    override = (inputs.license_override or "").strip()
    license_value: Any
    if override:
        license_value = override
    else:
        license_value = publish.get("license")
    if license_value:
        fm["license"] = str(license_value)

    fm["task_categories"] = [str(t) for t in inputs.task_categories]

    language = _as_string_list(publish.get("language"))
    if language:
        fm["language"] = language

    tags = _as_string_list(publish.get("tags"))
    if tags:
        fm["tags"] = tags

    samples = _samples_written(inputs.stats)
    bucket = _size_bucket(samples)
    if bucket is not None:
        fm["size_categories"] = [bucket]

    # pd-ocr conventional keys (see DATASETS.md). These live alongside
    # the standard HF keys; the spec's example also puts them flat at
    # the top level.
    fm["pd-ocr-shape"] = inputs.shape
    fm["pd-ocr-source"] = _SOURCE

    recipe_sha = _recipe_sha(inputs.snapshot_bytes)
    if recipe_sha:
        fm["pd-ocr-recipe-sha"] = recipe_sha

    tool_version = inputs.snapshot.get("tool_version")
    if tool_version:
        fm["pd-ocr-render-tool-version"] = str(tool_version)

    return fm


def _recipe_sha(snapshot_bytes: bytes) -> str | None:
    """SHA-256 of the snapshot YAML bytes.

    Falls back to ``None`` (so the front-matter key is omitted) when
    we have no snapshot to hash — only happens in tests that pass an
    explicit empty payload.
    """

    if not snapshot_bytes:
        return None
    return hashlib.sha256(snapshot_bytes).hexdigest()


def _size_bucket(samples: int | None) -> str | None:
    """Pick the smallest HF bucket whose upper bound covers ``samples``.

    Returns ``None`` for a missing / zero count so the front-matter
    key is omitted rather than wrong.
    """

    if samples is None or samples <= 0:
        return None
    for upper, label in _SIZE_BUCKETS:
        if samples < upper:
            return label
    return _SIZE_OVERFLOW


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------


def _body(inputs: DatasetCardInputs) -> str:
    """Render the human-readable Markdown body.

    Sections (each elided if its data isn't present):

    1. H1 + tagline (recipe name + description).
    2. Reproduce block (CLI commands per the spec example).
    3. Stats block.
    4. Provenance block (recipe SHA + tool version + corpus + fonts).
    """

    recipe = _recipe_from_snapshot(inputs.snapshot)
    sections: list[str] = []

    name = str(recipe.get("name") or "")
    title = f"# pd-ocr-synth — {name}" if name else "# pd-ocr-synth dataset"
    sections.append(title)

    description = inputs.description_override
    if not description:
        description = recipe.get("description")
    if description:
        sections.append(str(description).strip())

    if name:
        sections.append(_reproduce_block(name))

    stats_section = _stats_section(inputs.stats, recipe)
    if stats_section:
        sections.append(stats_section)

    provenance_section = _provenance_section(inputs, recipe)
    if provenance_section:
        sections.append(provenance_section)

    return "\n\n".join(sections)


def _reproduce_block(recipe_name: str) -> str:
    """Render the spec's reproduce-this-dataset shell snippet.

    The repo-name in ``--repo`` is left as a placeholder so the
    snippet works regardless of where the dataset got published; the
    publish step will *not* rewrite this section.
    """

    return (
        "Generated by [pd-ocr-synth](https://github.com/) from "
        f"`recipes/{recipe_name}.yaml`. Reproduce with:\n\n"
        "```bash\n"
        f"pd-ocr-synth fetch {recipe_name}\n"
        f"pd-ocr-synth render {recipe_name}\n"
        f"pd-ocr-synth publish {recipe_name} --repo <this-repo>\n"
        "```"
    )


def _stats_section(stats: dict[str, Any] | None, recipe: Mapping[str, Any]) -> str | None:
    """Render the Stats H2 section, or ``None`` if there's nothing to say.

    Pulls fonts-used from stats first; falls back to recipe ``fonts``
    block if the stats counters are absent (e.g. older runs).
    """

    lines: list[str] = []
    samples = _samples_written(stats)
    if samples is not None:
        lines.append(f"- Samples: {samples}")

    fonts_used = _fonts_used_names(stats)
    if not fonts_used:
        # Fall back to recipe-declared fonts. We use the basename of
        # the path (without extension) for readability; the snapshot's
        # ``fonts`` list carries absolute paths.
        fonts_used = [
            Path(str(f.get("path", ""))).stem for f in (recipe.get("fonts") or []) if f.get("path")
        ]
    if fonts_used:
        lines.append(f"- Fonts used: {', '.join(fonts_used)}")

    if stats is not None:
        tokens_unique = stats.get("tokens_unique")
        if isinstance(tokens_unique, int) and tokens_unique > 0:
            lines.append(f"- Tokens (unique): {tokens_unique}")
        # Detection-mode counters (M09): per-page line / word / paragraph
        # totals roll up the structural ground truth a detection run
        # produces. Recognition stats.json doesn't carry these keys, so
        # ``> 0`` gating keeps the section quiet on recognition cards
        # rather than emitting "Lines: 0 / Words: 0 / Paragraphs: 0"
        # noise. Without this, the dataset card silently dropped the
        # main scale signal a detection consumer wants ("100 pages" is
        # informative; "100 pages, 800 lines, 4000 words" is what a
        # trainer plans capacity around).
        lines_total = stats.get("lines_total")
        if isinstance(lines_total, int) and lines_total > 0:
            lines.append(f"- Lines: {lines_total}")
        words_total = stats.get("words_total")
        if isinstance(words_total, int) and words_total > 0:
            lines.append(f"- Words: {words_total}")
        paragraphs_total = stats.get("paragraphs_total")
        if isinstance(paragraphs_total, int) and paragraphs_total > 0:
            lines.append(f"- Paragraphs: {paragraphs_total}")
        wall = stats.get("wall_time_seconds")
        if isinstance(wall, (int, float)) and wall > 0:
            lines.append(f"- Render time: {int(round(float(wall)))}s")

    if not lines:
        return None
    return "## Stats\n" + "\n".join(lines)


def _provenance_section(inputs: DatasetCardInputs, recipe: Mapping[str, Any]) -> str | None:
    """Render the Provenance H2 section.

    Includes the recipe SHA (same value that lands in front matter),
    the tool version, and a one-line corpus listing built from the
    snapshot's ``corpus`` block. Fonts get a "see recipe" pointer
    rather than a license dump — we don't have license metadata at
    this layer, and the recipe author owns that.
    """

    lines: list[str] = []
    sha = _recipe_sha(inputs.snapshot_bytes)
    if sha:
        lines.append(f"- Recipe SHA: {sha}")
    tool_version = inputs.snapshot.get("tool_version")
    if tool_version:
        lines.append(f"- Tool version: pd-ocr-synth {tool_version}")
    corpus = _corpus_summary(recipe)
    if corpus:
        lines.append(f"- Corpus sources: {corpus}")
    if recipe.get("fonts"):
        lines.append("- Fonts: see recipe for licenses; not bundled")

    if not lines:
        return None
    return "## Provenance\n" + "\n".join(lines)


def _corpus_summary(recipe: Mapping[str, Any]) -> str:
    """One-line summary of the recipe's corpus sources.

    For each entry we emit ``<type>:<key>`` where the key is the most
    specific identifier we can pull (``key`` for HF, ``path`` basename
    for local, ``id`` for wikisource/CELT). Entries with neither a
    type nor a key are skipped silently rather than surfaced as a
    confusing partial string.
    """

    out: list[str] = []
    for entry in recipe.get("corpus") or []:
        if not isinstance(entry, Mapping):
            continue
        kind = entry.get("type")
        identifier = (
            entry.get("key")
            or entry.get("id")
            or entry.get("repo")
            or (Path(str(entry["path"])).name if entry.get("path") else None)
        )
        if kind and identifier:
            out.append(f"{kind}:{identifier}")
        elif kind:
            out.append(str(kind))
    return ", ".join(out)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def _recipe_from_snapshot(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    """Pluck the resolved recipe payload out of the snapshot.

    Returns an empty dict for snapshots without a ``recipe`` key
    (legacy / synthetic-test snapshots) so callers can `.get()`
    without guarding.
    """

    inner = snapshot.get("recipe")
    if isinstance(inner, Mapping):
        return inner
    return {}


def _hf_publish_block(recipe: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve ``recipe.publish.hf_dataset`` (an empty dict if absent)."""

    publish = recipe.get("publish")
    if not isinstance(publish, Mapping):
        return {}
    hf = publish.get("hf_dataset")
    if not isinstance(hf, Mapping):
        return {}
    return hf


def _resolve_description_override(
    snapshot: Mapping[str, Any], local_output_dir: Path
) -> str | None:
    """Read ``publish.hf_dataset.description_file`` if it points at a real file.

    The description file is the user's hand-authored body intro: spec
    10's "Users can override the generated README by placing
    ``recipes/<recipe>/README.md.template`` next to the recipe."
    Resolution rules:

    - absolute path → use as-is.
    - relative path → resolve relative to the output dir's parent
      (where the recipe was loaded from in typical usage). We don't
      have the recipe source path here, so the parent of the local
      output is the safest anchor; the recipe loader has already made
      the field optional anyway.
    - missing file → return ``None``; the body falls back to
      ``recipe.description``.
    """

    recipe = _recipe_from_snapshot(snapshot)
    publish = _hf_publish_block(recipe)
    desc_file = publish.get("description_file")
    if not desc_file:
        return None
    candidate = Path(str(desc_file))
    if not candidate.is_absolute():
        candidate = (local_output_dir.parent / candidate).resolve()
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8").strip() or None
    return None


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _samples_written(stats: dict[str, Any] | None) -> int | None:
    """Pull the rendered-sample count out of stats.json, ``None`` if absent.

    Spec calls for the count of *successfully rendered* samples, not
    the planned count: a recipe with ``count: 50000`` that skipped
    1000 should publish as 49000.
    """

    if not stats:
        return None
    value = stats.get("samples_written")
    if not isinstance(value, int):
        return None
    return value


def _fonts_used_names(stats: dict[str, Any] | None) -> list[str]:
    """Pull the list of font names actually used at render time.

    The stats writer keys ``fonts_used`` by font *filename* (e.g.
    ``bungc.otf``); for the README we drop the extension to match the
    spec example ("bungc, seangc, glangc, ...").
    """

    if not stats:
        return []
    fonts = stats.get("fonts_used")
    if not isinstance(fonts, Mapping):
        return []
    return [Path(str(name)).stem for name in fonts]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _as_string_list(value: Any) -> list[str]:
    """Coerce a value to a list of non-empty strings.

    The publish block's ``tags`` and ``language`` are typed as
    ``list[str]`` by the recipe model, but the snapshot YAML round-
    trips through plain dicts; we don't trust the shape.
    """

    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
    return out
