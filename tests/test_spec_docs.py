"""Spec-doc ↔ implementation drift guards.

Each test in this module pairs a claim made in ``docs/specs/`` with
the source-of-truth in ``src/pd_ocr_synth/`` and fails if the two get
out of sync. The pattern mirrors the degradation-kind catalog meta-
tests in ``test_validation.py`` (iter 66).

Why this file exists. The spec docs are the contract this project is
implemented against, and a stale doc is a real bug — downstream
trainers, recipe authors, and contributors all read the spec to
discover field names, file names, exit codes, etc. When the code
moves but the doc lags, the contract silently fragments. Iter 25
caught the original ``pages.json``/``labels.json`` rename in spec 08;
iter N (this commit) catches a follow-on case where spec 06 and spec
10 were still calling the detection labels file by its old name even
though the writer + reader had moved on.

Each test reads the spec markdown verbatim — no parsing of canonical
data structures we control — so the test only passes when the doc
text on disk genuinely names the right artifact.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Canonical filename constants live with the writers. Importing them
# here means a future rename in code surfaces as a static failure
# (ImportError) rather than a silent doc/code mismatch.
from pd_ocr_synth.cli import build_parser
from pd_ocr_synth.output.detection import LABELS_FILENAME as DETECTION_LABELS_FILENAME
from pd_ocr_synth.output.recognition import LABELS_FILENAME as RECOGNITION_LABELS_FILENAME

SPECS_DIR = Path(__file__).resolve().parent.parent / "docs" / "specs"


# ---------------------------------------------------------------------------
# pages.json drift (iter 25 caught the original; iter N caught two more)
#
# The detection writer's labels file is named after the constant
# ``output.detection.LABELS_FILENAME``, currently ``"labels.json"``.
# Earlier drafts of spec 08 called it ``pages.json`` and that name
# leaked into spec 06 and spec 10. The trainer's reader is the
# canonical contract, so the spec docs need to follow the constant.
#
# Spec 08 is allowed to mention ``pages.json`` exactly twice — the
# historical-context paragraph that explains the rename precedent.
# Anywhere else in any spec, a bare reference to ``pages.json`` is
# stale and we fail loudly.
# ---------------------------------------------------------------------------


def _spec_text(name: str) -> str:
    return (SPECS_DIR / name).read_text(encoding="utf-8")


def test_recognition_and_detection_labels_filename_consistent() -> None:
    """Both writers must agree on the labels filename.

    The drift guard below reads ``"labels.json"`` from the recognition
    constant; if a future change splits the two writers' filenames
    apart (e.g. detection moves to ``annotations.json``), the spec-
    doc tests below would silently still pass because they hardcode
    ``"pages.json"`` as the *forbidden* name. Pin the invariant the
    rest of this file relies on.
    """

    assert RECOGNITION_LABELS_FILENAME == DETECTION_LABELS_FILENAME == "labels.json", (
        "Recognition and detection writers disagree on labels filename: "
        f"recognition={RECOGNITION_LABELS_FILENAME!r}, "
        f"detection={DETECTION_LABELS_FILENAME!r}. "
        "If this is intentional, update test_spec_docs.py to track each side "
        "separately."
    )


def _paragraphs_with_token(text: str, token: str) -> list[tuple[int, str]]:
    """Yield ``(starting_line_number, paragraph_text)`` for paragraphs
    containing ``token``.

    A "paragraph" is a run of consecutive non-blank lines (the standard
    Markdown source convention). This lets the drift checks tolerate
    historical-context notes that span multiple wrapped lines without
    losing the ability to flag a bare reference somewhere else in the
    same file.
    """

    paragraphs: list[tuple[int, str]] = []
    current: list[str] = []
    start_line = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.strip() == "":
            if current and any(token in lines for lines in current):
                paragraphs.append((start_line, "\n".join(current)))
            current = []
            start_line = 0
        else:
            if not current:
                start_line = lineno
            current.append(line)
    if current and any(token in lines for lines in current):
        paragraphs.append((start_line, "\n".join(current)))
    return paragraphs


def _stale_pages_json_paragraphs(text: str) -> list[tuple[int, str]]:
    """Paragraphs that mention ``pages.json`` without also naming
    ``labels.json`` in the same paragraph.

    The single-paragraph context window is what makes a multi-line
    historical-context note ("an earlier draft of this spec specified
    `pages.json`; the trainer's existing reader is the canonical
    contract, so we matched `labels.json` ...") safe — both names
    appear in the same block, so the reader sees the rename in
    context. A bare mention of the old name in any other paragraph
    is what we flag.
    """

    return [
        (lineno, paragraph)
        for lineno, paragraph in _paragraphs_with_token(text, "pages.json")
        if "labels.json" not in paragraph
    ]


def test_spec_06_no_stale_pages_json_reference() -> None:
    """Spec 06 (rendering) must not bare-reference ``pages.json``.

    Detection-mode output is written to ``labels.json`` per the
    rename in M09 (mirroring the M07 ``labels.csv`` → ``labels.json``
    rename in recognition). A passing reference to the old name in
    spec 06 sends recipe authors hunting for a file the renderer
    never produces.

    A historical-context paragraph that explicitly contrasts the old
    name against ``labels.json`` is allowed; a bare reference is not.
    """

    offending = _stale_pages_json_paragraphs(_spec_text("06-rendering.md"))
    assert not offending, (
        "docs/specs/06-rendering.md mentions 'pages.json' in a paragraph "
        f"that does not also name {RECOGNITION_LABELS_FILENAME!r} (see "
        "src/pd_ocr_synth/output/detection.py:LABELS_FILENAME):\n"
        + "\n\n".join(f"  L{ln}:\n{paragraph}" for ln, paragraph in offending)
    )


def test_spec_10_no_stale_pages_json_reference() -> None:
    """Spec 10 (publishing) must not bare-reference ``pages.json``.

    Spec 10 describes the producer side of the HF dataset contract.
    The detection imagefolder publish path copies ``labels.json``
    verbatim from the local render; the parquet path (future) reads
    the same file and projects it into a Sequence schema. Either way
    the on-disk source is ``labels.json``.

    A historical-context paragraph naming both old and new is allowed;
    a bare reference is not. Same rule as spec 06.
    """

    offending = _stale_pages_json_paragraphs(_spec_text("10-publishing.md"))
    assert not offending, (
        "docs/specs/10-publishing.md has 'pages.json' references in a "
        f"paragraph that does not also name {RECOGNITION_LABELS_FILENAME!r}:\n"
        + "\n\n".join(f"  L{ln}:\n{paragraph}" for ln, paragraph in offending)
    )


def test_spec_08_pages_json_only_in_historical_context() -> None:
    """Spec 08 documents the rename; the only mentions of ``pages.json``
    must live in a paragraph that also names ``labels.json``.

    This is the inverse of the spec-10 guard: if a future edit moves
    an explanatory note around and ends up bare-referencing the old
    name elsewhere, surface that here. Spec 08's normative sections
    must use ``labels.json`` only.
    """

    offending = _stale_pages_json_paragraphs(_spec_text("08-output-format.md"))
    assert not offending, (
        "docs/specs/08-output-format.md has 'pages.json' references in a "
        f"paragraph that does not also name {RECOGNITION_LABELS_FILENAME!r}:\n"
        + "\n\n".join(f"  L{ln}:\n{paragraph}" for ln, paragraph in offending)
    )


# ---------------------------------------------------------------------------
# Spec 01 ↔ argparse drift (iter 68)
#
# Spec 01 lists every subcommand and the flags it accepts. The CLI
# parser is the source of truth, but the spec is the user-facing
# contract. Without a meta-test, flag additions silently leave the
# spec stale (which is exactly how spec 01's "M10 stretch" label on
# ``audit`` survived through M10 finishing in iters 45-50, plus the
# audit window flags ``--since`` / ``--until`` / ``--recipe-sha`` /
# ``--summary`` / ``--audit-file`` / ``--global`` from iter 49+).
#
# Both directions are checked:
#   1. Every subcommand named in spec 01's Subcommands table must
#      exist in argparse.
#   2. Every flag listed in a per-subcommand flag table must exist
#      on the matching subparser.
#   3. Every non-render-family flag that argparse declares must be
#      listed in the spec's per-subcommand table (catches "added a
#      flag, forgot to update the spec" — the bug class iter 67's
#      header-row prep work was setting up).
#
# Render-family flags (``-c/--count``, ``-o/--output``, ``-s/--seed``,
# ``-w/--workers``, ``--cache-dir``, ``--no-cache``, ``--dry-run``)
# are documented once in the spec's shared "Render-family options"
# section; the per-subcommand check skips them so we don't demand
# that every render-family subcommand re-list them.
# ---------------------------------------------------------------------------


_RENDER_FAMILY_FLAGS = frozenset(
    {
        "-c",
        "--count",
        "-o",
        "--output",
        "-s",
        "--seed",
        "-w",
        "--workers",
        "--cache-dir",
        "--no-cache",
        "--dry-run",
    }
)

# Top-level flags we don't expect documented per-subcommand: the
# program-wide ``--help`` / ``--version`` and the implicit
# subcommand recipe positional. They're captured by the Invocation
# section of spec 01, not the flag tables.
_OMIT_FROM_SPEC_CHECK = frozenset(
    {
        "-h",
        "--help",
        "--version",
    }
)


def _argparse_subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """Return the parser's subcommand → subparser mapping.

    argparse stores this on a ``_SubParsersAction`` registered as a
    parser action. The attribute is private but the layout has been
    stable for a decade and a future refactor would surface here as
    a clear AttributeError, exactly the failure mode the meta-test
    is designed to catch.
    """

    for action in parser._actions:  # noqa: SLF001 — argparse-internal but stable
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("build_parser() has no subparsers action")


def _argparse_flags(subparser: argparse.ArgumentParser) -> set[str]:
    """All option strings accepted by ``subparser`` (e.g. ``--force``).

    Excludes positional arguments — the spec lists them in the
    subcommand title (e.g. ``audit [output-dir]``), not in the flag
    tables.
    """

    flags: set[str] = set()
    for action in subparser._actions:  # noqa: SLF001
        for opt in action.option_strings:
            flags.add(opt)
    return flags


_SUBCOMMAND_HEADER = re.compile(r"^### `([a-z]+)(?:\s.*)?`\s*$")
# Match every option-looking token *inside* a backticked code span on
# the row's first cell. The backtick-wrapped form is e.g.
# ``--dir PATH``, ``-o, --output PATH``, or ``--private`` — we want
# ``--dir``, ``-o``, ``--output``, ``--private`` etc. The token rule
# is "starts with - or -- and is followed by alnum/-". We allow word
# boundaries on either side (start of cell, ``,``, space, backtick).
_FLAG_TOKEN = re.compile(r"(?<![A-Za-z0-9-])(--?[A-Za-z][A-Za-z0-9-]*)")


def _spec_subcommand_names(spec_text: str) -> list[str]:
    """Names from the Subcommands table — first ``code`` cell per row.

    The table format is ``| `init <name>` | ... |`` so we strip the
    trailing decoration to recover the bare subcommand name.
    """

    names: list[str] = []
    in_table = False
    for line in spec_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Subcommands"):
            in_table = True
            continue
        if in_table and stripped.startswith("## "):
            break
        if not in_table:
            continue
        if not stripped.startswith("| `"):
            continue
        # Skip the header separator row: ``| Command | Purpose |``.
        # That row doesn't start with ``| `<token>`` so the prefix
        # filter above already excludes it; nothing to do here.
        match = re.match(r"\|\s*`([a-z]+)", line)
        if match:
            name = match.group(1)
            if name not in names:
                names.append(name)
    return names


def _spec_flag_tables(spec_text: str) -> dict[str, set[str]]:
    """Per-subcommand flag sets parsed from the ``### `<name>``` blocks.

    Walks the spec line-by-line: each ``### `<name>``` heading opens a
    new section, each ``| `--flag` | ... |`` row inside contributes a
    flag, and any other ``###`` heading or higher closes the section.
    """

    tables: dict[str, set[str]] = {}
    current: str | None = None
    for raw_line in spec_text.splitlines():
        line = raw_line.rstrip()
        header = _SUBCOMMAND_HEADER.match(line)
        if header:
            current = header.group(1)
            tables.setdefault(current, set())
            continue
        # A heading at any level closes the section. ``####`` would be
        # a sub-section, but spec 01 doesn't currently use one and
        # closing on it would be safer than silently absorbing.
        if line.startswith("##") and not line.startswith("###"):
            current = None
            continue
        if line.startswith("## "):
            current = None
            continue
        if current is None:
            continue
        if not line.lstrip().startswith("| `"):
            continue
        # First cell of the row holds the flag (or comma-list of
        # short+long aliases). Extract every backticked code span and
        # scan inside it for option-looking tokens; argparse treats
        # each alias as a separate option string so the spec table
        # should too. Confining the scan to backticked spans avoids
        # picking up flag-like tokens from the description column.
        first_cell = line.split("|", 2)[1] if "|" in line else ""
        for code_span in re.findall(r"`([^`]+)`", first_cell):
            for match in _FLAG_TOKEN.finditer(code_span):
                tables[current].add(match.group(1))
    return tables


def test_spec_01_subcommands_match_argparse() -> None:
    """Every subcommand named in spec 01 must exist in the parser.

    Catches the common drift: spec lists a subcommand the parser no
    longer ships (or never shipped). The reverse direction —
    parser-only subcommands — is covered below.
    """

    parser = build_parser()
    sub_map = _argparse_subparsers(parser)
    spec_names = _spec_subcommand_names(_spec_text("01-cli.md"))

    spec_only = [name for name in spec_names if name not in sub_map]
    assert not spec_only, (
        "docs/specs/01-cli.md lists subcommands not in build_parser():\n  "
        + ", ".join(spec_only)
        + f"\nargparse has: {sorted(sub_map)}"
    )


def test_argparse_subcommands_match_spec_01() -> None:
    """Reverse direction: every parser subcommand must be documented.

    A new subcommand added to the parser without a spec entry would
    be invisible to recipe authors. Force the spec update.
    """

    parser = build_parser()
    sub_map = _argparse_subparsers(parser)
    spec_names = set(_spec_subcommand_names(_spec_text("01-cli.md")))

    parser_only = sorted(name for name in sub_map if name not in spec_names)
    assert not parser_only, (
        "build_parser() registers subcommands missing from "
        "docs/specs/01-cli.md Subcommands table:\n  "
        + ", ".join(parser_only)
        + "\nAdd a row to the table with a one-line purpose."
    )


def test_spec_01_flag_tables_match_argparse() -> None:
    """Every flag in a per-subcommand table must exist on its subparser.

    A flag in the spec but not in the parser is a documentation bug:
    users will type a flag the program rejects. Render-family flags
    are documented in the shared section, not the per-subcommand
    tables, so we only check the local-flag tables here.
    """

    parser = build_parser()
    sub_map = _argparse_subparsers(parser)
    flag_tables = _spec_flag_tables(_spec_text("01-cli.md"))

    failures: list[str] = []
    for name, spec_flags in flag_tables.items():
        if name not in sub_map:
            failures.append(f"  {name}: spec has flag table but no subparser")
            continue
        argparse_flags = _argparse_flags(sub_map[name])
        spec_only = sorted(f for f in spec_flags if f not in argparse_flags)
        if spec_only:
            failures.append(f"  {name}: documented flag(s) not in argparse: {spec_only}")
    assert not failures, (
        "docs/specs/01-cli.md lists flags missing from build_parser():\n" + "\n".join(failures)
    )


def test_argparse_flags_match_spec_01() -> None:
    """Reverse direction: every parser flag must be documented somewhere.

    Each flag must appear either in the spec's per-subcommand flag
    table or in the render-family allow-list. A new flag that lands
    in argparse without a spec entry is a silent contract change —
    fail until the spec is updated.

    The allow-list ``_RENDER_FAMILY_FLAGS`` covers flags documented
    once in the shared "Render-family options" table. ``--help`` and
    related top-level flags are excluded via ``_OMIT_FROM_SPEC_CHECK``.
    """

    parser = build_parser()
    sub_map = _argparse_subparsers(parser)
    flag_tables = _spec_flag_tables(_spec_text("01-cli.md"))

    failures: list[str] = []
    for name, subparser in sub_map.items():
        argparse_flags = _argparse_flags(subparser)
        spec_flags = flag_tables.get(name, set())
        for flag in sorted(argparse_flags):
            if flag in _OMIT_FROM_SPEC_CHECK:
                continue
            if flag in _RENDER_FAMILY_FLAGS:
                continue
            if flag in spec_flags:
                continue
            failures.append(f"  {name} {flag}: in argparse, not in spec 01")
    assert not failures, (
        "docs/specs/01-cli.md is missing flag entries the parser declares:\n"
        + "\n".join(failures)
        + "\nAdd them to the relevant '### `<subcommand>`' table, or "
        "extend _RENDER_FAMILY_FLAGS / _OMIT_FROM_SPEC_CHECK in the "
        "test if the flag is intentionally shared/hidden."
    )
