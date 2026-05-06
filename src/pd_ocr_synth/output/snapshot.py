"""``recipe.snapshot.yaml`` writer + comparator.

Per ``docs/specs/08-output-format.md``: the snapshot is the resolved
recipe written next to the output. It carries:

- absolute paths (the loader already does this; we round-trip the
  pydantic ``model_dump``)
- expanded presets (the loader already inlines these)
- the running tool version + the seed actually used at render time
- a SHA-256 of every font + local-corpus file at render time

The snapshot is what ``--resume`` checks against to detect "the recipe
or its inputs changed since the previous run; refusing to keep
appending samples that no longer match." Snapshot equality is by-value
on the resolved recipe dict (modulo ``count``, which can grow on
resume) plus per-input file hashes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pd_ocr_synth import __version__
from pd_ocr_synth.recipe import Recipe
from pd_ocr_synth.recipe.models import LocalCorpus

# Filename written into the output directory. Centralized here so the
# writer + the resume check + the test suite all agree.
SNAPSHOT_FILENAME = "recipe.snapshot.yaml"

# Top-level keys under the ``snapshot`` document. Stable on-disk shape.
_SNAPSHOT_KEYS = (
    "tool_version",
    "seed",
    "recipe",
    "input_hashes",
)


class SnapshotMismatchError(Exception):
    """Raised when ``--resume`` finds a snapshot that doesn't match the
    current recipe + inputs.

    ``detail`` is a human-readable explanation of the first mismatch
    encountered (recipes might differ on a single block; we don't try
    to enumerate every diff).
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Build / write
# ---------------------------------------------------------------------------


def build_snapshot(recipe: Recipe, *, seed: int) -> dict[str, Any]:
    """Construct the on-disk snapshot dict for ``recipe`` at render time.

    ``seed`` is captured separately because it's the *effective* seed
    used by the run (CLI ``--seed`` overrides the recipe's). Absent
    an override, callers should pass ``recipe.seed``.
    """

    payload = recipe.model_dump(mode="json")
    # ``source_path`` is loader-only metadata, not part of the YAML
    # contract, so drop it from the snapshot. Keeping it in would make
    # the snapshot non-portable across machines (the absolute path is
    # specific to the dev box that ran the render).
    payload.pop("source_path", None)

    return {
        "tool_version": __version__,
        "seed": int(seed),
        "recipe": payload,
        "input_hashes": _hash_inputs(recipe),
    }


def write_snapshot(snapshot: dict[str, Any], output_dir: Path) -> Path:
    """Write the snapshot to ``output_dir / recipe.snapshot.yaml``.

    Returns the written path. Existing files are overwritten — the
    caller (``RecognitionWriter``) decides when to do this based on
    ``--force`` / ``--resume`` semantics.
    """

    target = output_dir / SNAPSHOT_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        {key: snapshot[key] for key in _SNAPSHOT_KEYS},
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    target.write_text(text, encoding="utf-8")
    return target


def load_snapshot(output_dir: Path) -> dict[str, Any] | None:
    """Read an existing snapshot from ``output_dir`` if present.

    Returns ``None`` if the file is missing. Yaml parse errors raise
    so the caller can surface them as a render error rather than
    silently treating a corrupt snapshot as "no previous run."
    """

    path = output_dir / SNAPSHOT_FILENAME
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SnapshotMismatchError(f"snapshot at {path} is not a YAML mapping")
    return data


def snapshot_matches(
    existing: dict[str, Any],
    current: dict[str, Any],
    *,
    allow_count_growth: bool = True,
) -> tuple[bool, str | None]:
    """Compare two snapshots for resume eligibility.

    Returns ``(True, None)`` on match, ``(False, reason)`` otherwise.
    ``allow_count_growth=True`` (the default) lets ``output.count``
    increase between runs — common case for resume after a crash with
    a higher target.
    """

    if existing.get("tool_version") != current.get("tool_version"):
        return False, (
            f"tool_version changed: snapshot={existing.get('tool_version')!r} "
            f"current={current.get('tool_version')!r}"
        )
    if int(existing.get("seed", -1)) != int(current.get("seed", -2)):
        return False, (
            f"seed changed: snapshot={existing.get('seed')} current={current.get('seed')}"
        )
    if existing.get("input_hashes") != current.get("input_hashes"):
        return False, "input file hashes differ (a font or local corpus changed on disk)"

    e_recipe = dict(existing.get("recipe") or {})
    c_recipe = dict(current.get("recipe") or {})
    if allow_count_growth:
        e_out = dict(e_recipe.get("output") or {})
        c_out = dict(c_recipe.get("output") or {})
        e_count = e_out.pop("count", None)
        c_count = c_out.pop("count", None)
        e_recipe["output"] = e_out
        c_recipe["output"] = c_out
        if c_count is not None and e_count is not None and int(c_count) < int(e_count):
            return False, f"count shrank: snapshot={e_count} current={c_count}"

    if e_recipe != c_recipe:
        return False, "recipe content differs (something other than count changed)"
    return True, None


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _InputFile:
    role: str  # "font" | "corpus_local"
    path: Path


def _hash_inputs(recipe: Recipe) -> dict[str, str]:
    """SHA-256 every font + local corpus path the recipe references.

    Output is a flat ``{role:absolute-path: sha256-hex}`` dict so the
    snapshot is comparable by exact equality. Paths are absolute (the
    loader's job) so we don't have to re-resolve here.

    Missing files are recorded as ``"<missing>"`` rather than
    suppressed; that way a snapshot from an environment with a
    different optional font set still surfaces as different.
    """

    inputs: list[_InputFile] = []
    for font in recipe.fonts:
        inputs.append(_InputFile(role="font", path=font.path))
    for entry in recipe.corpus:
        if isinstance(entry, LocalCorpus):
            inputs.append(_InputFile(role="corpus_local", path=entry.path))

    out: dict[str, str] = {}
    for item in inputs:
        key = f"{item.role}:{item.path}"
        if not item.path.exists():
            out[key] = "<missing>"
            continue
        out[key] = _sha256_file(item.path)
    return out


def _sha256_file(path: Path) -> str:
    """Stream a file through SHA-256.

    Chunk size of 1 MiB keeps memory bounded for large fonts (the
    Gaelic OTFs are <1 MB, but recipe authors will eventually point
    at larger corpora).
    """

    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
