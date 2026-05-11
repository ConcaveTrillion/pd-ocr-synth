"""Pre-flight checks against a built HF staging directory.

Per ``docs/specs/10-publishing.md`` § Dataset card and the matching
deliverable in ``docs/roadmap/08-publishing-hf.md``, the README that
ships in the staging dir must carry a fixed set of conventional
``pd-ocr-*`` front-matter keys:

- ``pd-ocr-shape`` — fixed shape label, e.g. ``recognition/v1``.
- ``pd-ocr-source`` — producer tool name; always ``pd-ocr-synth``.
- ``pd-ocr-recipe-sha`` — SHA over the snapshot YAML bytes.
- ``pd-ocr-render-tool-version`` — tool version that produced the
  output, copied from the snapshot.
- ``pd-ocr-content-sha`` — SHA over the staging-dir contents,
  written into the front matter by ``apply_content_sha_to_readme``.

The staging builder (`recognition.build_recognition_staging`) populates
these in two passes: the dataset-card writer fills in the first four
from the local snapshot, and ``apply_content_sha_to_readme`` rewrites
the README to add ``pd-ocr-content-sha`` once the rest of the dir has
settled. The risk this module guards against is a *silently dropped*
key — if the snapshot is missing a ``tool_version`` field the
``pd-ocr-render-tool-version`` line never gets written, and the
upload flow produces a card the trainer can't pin against.

Why a separate module rather than inline checks in ``recognition``:

- The pre-flight is the natural last step before contacting HF, and
  the upcoming ``--dry-run`` chunk + the eventual upload step both
  want to invoke it. Hoisting it here keeps the call-site short and
  testable in isolation.
- The check is *read-only* over the staging dir — no mutation, no
  network. That isolation is important: a future detection-mode
  staging path (M09) will share the same required-key contract and
  will reuse this validator unchanged.

This module is deliberately **pure file-IO** — it does not import
``huggingface_hub`` and never reads the network. It assumes the
staging dir was produced by ``build_recognition_staging`` (or a
future detection-mode equivalent) so the README path and front-matter
syntax are known.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pd_ocr_synth.publish.content_sha import CONTENT_SHA_KEY
from pd_ocr_synth.publish.dataset_card import README_FILENAME

# The conventional ``pd-ocr-*`` keys the dataset card MUST carry by
# the time the staging dir is ready for upload. Centralized as a
# tuple (ordered, immutable) so the error message lists missing keys
# in a stable order — humans diffing logs across runs benefit from
# that even when the set itself is small.
#
# ``pd-ocr-content-sha`` is included because the staging build runs
# ``apply_content_sha_to_readme`` immediately after the dataset-card
# writer; pre-flight is run *after* both of those, so by then the key
# must be present. A future ``--dry-run`` path that wants to validate
# the card *before* the SHA is applied can call
# :func:`check_required_front_matter` with a custom ``required`` set.
REQUIRED_FRONT_MATTER_KEYS: tuple[str, ...] = (
    "pd-ocr-shape",
    "pd-ocr-source",
    "pd-ocr-recipe-sha",
    "pd-ocr-render-tool-version",
    CONTENT_SHA_KEY,
)


class PreflightError(Exception):
    """Raised when a pre-flight check fails.

    Distinct exception type so the future CLI can map it cleanly: a
    pre-flight failure means the staging dir was *built* but is
    structurally incomplete — closer to "render output corrupt"
    (exit 6) than "auth failure" (exit 7). The CLI chunk that lands
    later will pick the final mapping; the helper just raises this
    typed exception with an actionable message.
    """


@dataclass(frozen=True)
class PreflightReport:
    """Outcome of a pre-flight check over the staging dir.

    Always returned (even on success) so callers like the future
    ``--dry-run`` summary can echo the populated keys + values without
    re-parsing the README. ``ok`` is the single-bit "publish-ready?"
    answer; ``missing_keys`` and ``empty_keys`` give the human-readable
    detail when not.

    Attributes
    ----------
    front_matter:
        The parsed YAML front-matter dict, exactly as it sits on disk.
        Empty dict if the README has no front matter (which itself
        triggers a pre-flight failure).
    missing_keys:
        Required keys that are absent from the front matter entirely.
        Stable order, matching :data:`REQUIRED_FRONT_MATTER_KEYS`.
    empty_keys:
        Required keys that are *present* but resolve to an empty /
        whitespace-only value. Treated as just-as-bad as missing
        because HF lint and the trainer's pin-by-key both fail on
        empty strings.
    """

    front_matter: dict[str, Any]
    missing_keys: tuple[str, ...] = ()
    empty_keys: tuple[str, ...] = ()
    # Internal raw README path so the error message can name it.
    readme_path: Path | None = None

    @property
    def ok(self) -> bool:
        """True iff every required key is present and non-empty."""

        return not self.missing_keys and not self.empty_keys


def check_required_front_matter(
    staging_dir: Path,
    *,
    required: Iterable[str] | None = None,
) -> PreflightReport:
    """Validate the staging README's front matter has every required key.

    Parses ``<staging_dir>/README.md``'s YAML front matter and checks
    that every key in ``required`` is present and resolves to a
    non-empty value. Returns a :class:`PreflightReport` either way; the
    caller decides whether a missing-key result raises or just warns.

    Parameters
    ----------
    staging_dir:
        Built staging dir produced by
        :func:`pd_ocr_synth.publish.build_recognition_staging`.
    required:
        Override the required-keys set. Defaults to
        :data:`REQUIRED_FRONT_MATTER_KEYS`. Useful for
        ``check_required_front_matter(staging, required=...)`` calls
        that run *before* ``apply_content_sha_to_readme`` (and so don't
        yet expect ``pd-ocr-content-sha``).

    Returns
    -------
    PreflightReport
        The parse outcome plus any missing / empty keys. Use the
        ``ok`` property as the single boolean check; use the lists
        for actionable error messages.

    Raises
    ------
    PreflightError
        If the README is missing or unparseable. Missing required
        *keys* do **not** raise — they're returned in the report so
        the caller can format a structured error or list everything
        wrong at once. (Distinguishing "broken README" from "missing
        keys" matters: the first is a bug in our writer, the second
        is a bug in the inputs.)
    """

    required_keys = tuple(required) if required is not None else REQUIRED_FRONT_MATTER_KEYS

    readme_path = Path(staging_dir) / README_FILENAME
    if not readme_path.is_file():
        raise PreflightError(
            f"staging dir {staging_dir} is missing {README_FILENAME}; "
            "did `build_recognition_staging` run to completion?"
        )

    text = readme_path.read_text(encoding="utf-8")
    front_matter = _parse_front_matter(text, readme_path)

    missing: list[str] = []
    empty: list[str] = []
    for key in required_keys:
        if key not in front_matter:
            missing.append(key)
            continue
        if _is_empty(front_matter[key]):
            empty.append(key)

    return PreflightReport(
        front_matter=front_matter,
        missing_keys=tuple(missing),
        empty_keys=tuple(empty),
        readme_path=readme_path,
    )


def assert_staging_publish_ready(
    staging_dir: Path,
    *,
    required: Iterable[str] | None = None,
) -> PreflightReport:
    """Strict variant: raise :class:`PreflightError` on any problem.

    Wraps :func:`check_required_front_matter` and turns a non-``ok``
    report into a typed exception with an actionable message. The
    upload step uses this; ``--dry-run`` may prefer the non-raising
    variant so it can print the report and exit 0 with a warning.
    """

    report = check_required_front_matter(staging_dir, required=required)
    if report.ok:
        return report

    parts: list[str] = [f"staging README {report.readme_path} is not publish-ready:"]
    if report.missing_keys:
        parts.append("  missing front-matter keys: " + ", ".join(report.missing_keys))
    if report.empty_keys:
        parts.append("  empty front-matter values: " + ", ".join(report.empty_keys))
    raise PreflightError("\n".join(parts))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Match a YAML front-matter block delimited by ``---`` lines at the
# very start of the file. ``DOTALL`` so the body capture can span
# newlines; ``\A`` anchors the leading delimiter so an inline ``---``
# deeper in the body can't be misread as the front matter.
_FRONT_MATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\s*(?:\n|\Z)", re.DOTALL)


def _parse_front_matter(text: str, readme_path: Path) -> dict[str, Any]:
    """Extract and YAML-parse the front-matter block.

    Returns an empty dict when no front matter is present *and the
    body is empty too* — a generated card always has front matter, so
    in practice this signals a malformed README. We surface that as a
    :class:`PreflightError` (with the path) rather than silently
    treating it as "no required keys present".
    """

    match = _FRONT_MATTER_RE.match(text)
    if match is None:
        raise PreflightError(
            f"staging README {readme_path} has no YAML front matter; "
            "the dataset-card writer always emits one — this looks like a corrupt staging build"
        )

    body = match.group("body")
    try:
        loaded = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        raise PreflightError(
            f"staging README {readme_path} has invalid YAML front matter: {exc}"
        ) from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise PreflightError(
            f"staging README {readme_path} front matter is not a YAML mapping "
            f"(got {type(loaded).__name__})"
        )
    return loaded


def _is_empty(value: Any) -> bool:
    """True for ``None``, ``""``, whitespace-only str, or empty list / dict.

    ``False`` for booleans, numerics including ``0``, and structured
    types with at least one element. The published spec keys are all
    string-valued (SHAs, version strings, fixed labels), so the
    string branch is the practically-relevant one — the others guard
    against a future required key landing as a list / mapping.
    """

    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False
