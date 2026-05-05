"""Path expansion + resolution helpers used by the recipe loader.

The recipe spec (``docs/specs/02-recipe-format.md``) defines this
resolution order for any path-like string:

1. Absolute paths used as-is.
2. ``~`` and ``${ENV_VAR}`` expanded.
3. Relative paths resolved against the directory of the recipe file.

The loader pre-processes the parsed YAML dict so pydantic only ever sees
absolute paths. Keys that hold paths are listed explicitly per block —
no string-content guessing.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any


def expand_path(value: str, base_dir: Path) -> str:
    """Apply env/home expansion and resolve relative paths against ``base_dir``.

    Returns a string (not a Path) so the result can be substituted back
    into the parsed dict before pydantic validation. Pydantic will coerce
    to ``Path`` itself when the model declares one.
    """

    expanded = os.path.expanduser(os.path.expandvars(value))
    p = Path(expanded)
    if not p.is_absolute():
        p = base_dir / p
    return str(p)


# Path-bearing keys per block. Kept explicit rather than inferred so that
# a recipe author cannot surprise us by, say, putting a URL in a field
# we'd otherwise treat as a path.
_OUTPUT_PATH_KEYS = ("destination",)
_FONT_PATH_KEYS = ("path",)
_LOCAL_CORPUS_PATH_KEYS = ("path",)
_PUBLISH_HF_PATH_KEYS = ("description_file",)

# Degradation stages declare path-like options. The loader walks
# ``degradation[*]`` and resolves these where the entry's ``kind``
# matches.
_DEGRADATION_PATH_KEYS_BY_KIND: Mapping[str, tuple[str, ...]] = {
    "paper_texture": ("directory",),
}


def _resolve_at(d: MutableMapping[str, Any], key: str, base_dir: Path) -> None:
    val = d.get(key)
    if isinstance(val, str):
        d[key] = expand_path(val, base_dir)


def _resolve_degradation_stage_list(stages: Any, base_dir: Path) -> None:
    """Walk a list of degradation stages (raw dicts) and resolve known path keys."""

    if not isinstance(stages, list):
        return
    for stage in stages:
        if not isinstance(stage, MutableMapping):
            continue
        kind = stage.get("kind")
        if isinstance(kind, str):
            for key in _DEGRADATION_PATH_KEYS_BY_KIND.get(kind, ()):
                _resolve_at(stage, key, base_dir)


def resolve_paths(data: MutableMapping[str, Any], base_dir: Path) -> MutableMapping[str, Any]:
    """Walk the parsed recipe dict in place and resolve path-bearing keys.

    Called by the loader after YAML parsing and before pydantic
    validation. The dict is mutated and returned for convenience.
    Unknown keys are ignored — semantic validation runs separately.
    """

    output = data.get("output")
    if isinstance(output, MutableMapping):
        for key in _OUTPUT_PATH_KEYS:
            _resolve_at(output, key, base_dir)

    fonts = data.get("fonts")
    if isinstance(fonts, list):
        for font in fonts:
            if isinstance(font, MutableMapping):
                for key in _FONT_PATH_KEYS:
                    _resolve_at(font, key, base_dir)

    corpus = data.get("corpus")
    if isinstance(corpus, list):
        for entry in corpus:
            if isinstance(entry, MutableMapping) and entry.get("type") == "local":
                for key in _LOCAL_CORPUS_PATH_KEYS:
                    _resolve_at(entry, key, base_dir)

    _resolve_degradation_stage_list(data.get("degradation"), base_dir)

    presets = data.get("degradation_presets")
    if isinstance(presets, MutableMapping):
        for stages in presets.values():
            _resolve_degradation_stage_list(stages, base_dir)

    publish = data.get("publish")
    if isinstance(publish, MutableMapping):
        hf = publish.get("hf_dataset")
        if isinstance(hf, MutableMapping):
            for key in _PUBLISH_HF_PATH_KEYS:
                _resolve_at(hf, key, base_dir)

    return data
