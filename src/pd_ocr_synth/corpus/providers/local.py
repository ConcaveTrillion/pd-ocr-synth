"""Local file-backed corpus provider.

Supports three forms of ``path`` (per ``docs/specs/04-corpus-providers.md``):

- a single file
- a glob pattern (anything with ``*``, ``?``, or ``[`` in it)
- a directory (walked recursively, alphabetical order)

Parser inference for M03 ships with ``plain`` only; ``.html`` and
``.xml`` files raise a clear error pointing at the ``parser:`` option,
which the web provider commit will activate. This keeps the local
provider's first cut small without lying about coverage.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

from pd_ocr_synth.corpus.context import ProviderContext
from pd_ocr_synth.corpus.exceptions import ProviderError

_GLOB_CHARS = re.compile(r"[*?\[]")


class LocalProvider:
    """Read text from local files on disk."""

    type_name: ClassVar[str] = "local"
    schema_version: ClassVar[int] = 1

    def cache_key(self, options: dict) -> str:
        path = str(options["path"])
        parser = options.get("parser") or "plain"
        digest = hashlib.sha256(f"{parser}|{path}".encode()).hexdigest()[:16]
        return f"local-{digest}"

    def fetch(self, ctx: ProviderContext, options: dict) -> Iterable[str]:
        raw_path = str(options["path"])
        explicit_parser = options.get("parser")

        for source_path in _expand(raw_path, base_dir=ctx.recipe_dir):
            parser = explicit_parser or _infer_parser(source_path)
            if parser != "plain":
                raise ProviderError(
                    f"local provider only supports parser='plain' in M03; "
                    f"got '{parser}' for {source_path}. Set parser explicitly "
                    f"or wait for the M03 web/HTML parser commit."
                )
            try:
                yield source_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ProviderError(
                    f"could not decode {source_path} as UTF-8: {exc}. "
                    "Convert the source file or set parser explicitly."
                ) from exc


def _expand(raw: str, *, base_dir: Path) -> list[Path]:
    """Expand a raw path argument into the sorted list of files to read."""

    p = Path(raw)
    if not p.is_absolute():
        p = (base_dir / p).resolve()

    if _GLOB_CHARS.search(str(p)):
        # Anchor the glob to its parent that actually exists. Use
        # Path.glob() relative to the deepest non-glob ancestor.
        anchor, pattern = _split_glob_anchor(p)
        if not anchor.exists():
            raise ProviderError(f"glob anchor does not exist: {anchor}")
        matches = sorted(child for child in anchor.glob(pattern) if child.is_file())
        if not matches:
            raise ProviderError(f"glob matched no files: {raw}")
        return matches

    if not p.exists():
        raise ProviderError(f"local corpus path does not exist: {p}")

    if p.is_file():
        return [p]

    if p.is_dir():
        files = sorted(child for child in p.rglob("*") if child.is_file())
        if not files:
            raise ProviderError(f"local corpus directory is empty: {p}")
        return files

    raise ProviderError(f"unsupported local corpus path type: {p}")


def _split_glob_anchor(p: Path) -> tuple[Path, str]:
    """Split a glob path at the first segment containing a glob char.

    ``/a/b/*.txt`` → (``/a/b``, ``*.txt``).
    ``/a/*/b/*.txt`` → (``/a``, ``*/b/*.txt``).
    """

    parts = p.parts
    for i, part in enumerate(parts):
        if _GLOB_CHARS.search(part):
            anchor = Path(*parts[:i]) if i else Path(parts[0])
            pattern = str(Path(*parts[i:]))
            return anchor, pattern
    return p.parent, p.name


def _infer_parser(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".text", ""}:
        return "plain"
    if suffix in {".html", ".htm"}:
        return "html-text"
    if suffix in {".xml", ".tei"}:
        return "tei-text"
    if suffix == ".json":
        return "json"
    return "plain"
