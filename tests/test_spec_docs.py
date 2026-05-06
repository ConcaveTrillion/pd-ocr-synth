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

# Canonical filename constants live with the writers. Importing them
# here means a future rename in code surfaces as a static failure
# (ImportError) rather than a silent doc/code mismatch.
import dataclasses
import re
from pathlib import Path

from pd_ocr_synth.audit import AUDIT_SCHEMA_VERSION, AuditEntry
from pd_ocr_synth.cli import build_parser
from pd_ocr_synth.lint import LINT_CODES
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


# ---------------------------------------------------------------------------
# Spec 01 "Lint codes" table ↔ ``lint.LINT_CODES`` (this iter)
#
# Spec 01 documents every code ``lint <recipe>`` can emit so users can
# grep / filter on stable identifiers. The catalog source-of-truth is
# ``pd_ocr_synth.lint.LINT_CODES``; behavioural tests in
# ``tests/test_lint.py`` already pin both directions of the runtime
# contract (every emitted code ⊆ catalog, every catalog entry reachable
# by some recipe). This meta-test closes the third side: the spec doc
# table must list exactly the codes in ``LINT_CODES``, no more and no
# fewer.
#
# A new lint helper that lands code without updating spec 01 fails
# here; a stale doc entry referring to a removed helper also fails.
# ---------------------------------------------------------------------------


def _spec_lint_codes(spec_text: str) -> set[str]:
    """Codes parsed from the ``## Lint codes`` table in spec 01.

    Walks each row in the table and extracts the first backticked
    span as the code. Rows that don't start with a backticked code
    cell are skipped, matching the pattern used by
    ``_spec_flag_tables`` for the per-subcommand flag tables.
    """

    codes: set[str] = set()
    in_section = False
    for raw_line in spec_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("## Lint codes"):
            in_section = True
            continue
        # Any other ``## `` heading closes the section.
        if in_section and stripped.startswith("## ") and not stripped.startswith("## Lint codes"):
            break
        if not in_section:
            continue
        if not stripped.startswith("| `"):
            continue
        match = re.match(r"\|\s*`([a-z_]+)`", stripped)
        if match:
            codes.add(match.group(1))
    return codes


def test_spec_01_lint_codes_match_LINT_CODES() -> None:
    """The Lint codes table in spec 01 must equal ``LINT_CODES``.

    Both directions checked in one assertion:
      - ``spec_only`` — codes documented but not registered (stale doc).
      - ``code_only`` — codes registered but not documented (silent ship).

    Either case is a contract drift between user-visible documentation
    and the linter's actual behaviour.
    """

    spec_codes = _spec_lint_codes(_spec_text("01-cli.md"))
    spec_only = sorted(spec_codes - LINT_CODES)
    code_only = sorted(LINT_CODES - spec_codes)
    failures: list[str] = []
    if spec_only:
        failures.append(f"spec 01 documents codes not in pd_ocr_synth.lint.LINT_CODES: {spec_only}")
    if code_only:
        failures.append(
            "pd_ocr_synth.lint.LINT_CODES has codes missing from the spec 01 "
            f"'Lint codes' table: {code_only}"
        )
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Spec 01 "Audit log schema" table ↔ ``audit.AuditEntry`` (this iter)
#
# Spec 01 documents every field the audit JSONL row carries so users
# can grep / filter / project on stable identifiers (``jq``, pandas,
# trainer-side scripts that key off ``recipe_sha``, etc.). The on-disk
# shape is determined by the ``AuditEntry`` dataclass — its
# ``to_jsonl`` serialization keys equal the dataclass field names.
#
# Without a meta-test, a future schema bump that adds a field (say
# ``planned_count`` next to ``count``, or ``host`` for cross-host
# triage) lands in the writer without surfacing in the user-facing
# spec, and consumers of the audit log only discover it by reading a
# real audit file. Conversely, a stale doc entry referring to a
# removed field would point users at JSON keys that never appear.
#
# Both directions are checked in one assertion:
#   - ``spec_only`` — fields documented but not on the dataclass.
#   - `dataclass_only` — fields on the dataclass but not documented.
# ---------------------------------------------------------------------------


def _spec_audit_fields(spec_text: str) -> set[str]:
    """Field names parsed from the ``## Audit log schema`` table in spec 01.

    Walks each row in the schema table and extracts the first
    backticked span as the field name. Same shape as
    ``_spec_lint_codes`` but the ``## `` heading match must close on
    any other ``## `` (including the trailing ``### Forward-…``
    subheading, which is one level deeper and therefore not a section
    boundary — the schema table sits before that subheading).
    """

    fields: set[str] = set()
    in_section = False
    for raw_line in spec_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("## Audit log schema"):
            in_section = True
            continue
        # Any other ``## `` heading closes the section. ``### `` is a
        # subheading inside the section (e.g. forward-compat policy)
        # and should not close it; the table itself sits before that
        # subheading anyway.
        if (
            in_section
            and stripped.startswith("## ")
            and not stripped.startswith("## Audit log schema")
        ):
            break
        if not in_section:
            continue
        if not stripped.startswith("| `"):
            continue
        match = re.match(r"\|\s*`([a-z_]+)`", stripped)
        if match:
            fields.add(match.group(1))
    return fields


def test_spec_01_audit_schema_matches_AuditEntry() -> None:
    """The Audit log schema table in spec 01 must equal ``AuditEntry`` fields.

    Both directions checked in one assertion:
      - ``spec_only`` — fields documented but not on the dataclass (stale doc).
      - ``dataclass_only`` — fields on the dataclass but not documented (silent ship).

    Either case is a contract drift between the user-visible audit log
    documentation and the actual on-disk JSONL shape.
    """

    spec_fields = _spec_audit_fields(_spec_text("01-cli.md"))
    dataclass_fields = {f.name for f in dataclasses.fields(AuditEntry)}
    spec_only = sorted(spec_fields - dataclass_fields)
    dataclass_only = sorted(dataclass_fields - spec_fields)
    failures: list[str] = []
    if spec_only:
        failures.append(
            f"spec 01 'Audit log schema' table documents fields not on AuditEntry: {spec_only}"
        )
    if dataclass_only:
        failures.append(
            "pd_ocr_synth.audit.AuditEntry has fields missing from the spec 01 "
            f"'Audit log schema' table: {dataclass_only}"
        )
    assert not failures, "\n".join(failures)


def test_spec_01_audit_schema_version_matches_constant() -> None:
    """Spec 01 must name the current ``AUDIT_SCHEMA_VERSION``.

    The schema description says "current: `<N>`" and the
    ``schema_version`` row's "Since" column references the same
    version label. A bump in ``AUDIT_SCHEMA_VERSION`` that doesn't
    update spec 01 would leave the doc claiming the wrong current
    version — a silent contract drift especially nasty for v1 readers
    that branch on the constant.

    The check is intentionally lightweight: we look for the literal
    ``current: `<N>``` substring in the audit schema section. Any
    other phrasing is fine as long as the constant appears verbatim.
    """

    spec_text = _spec_text("01-cli.md")
    needle = f"current: `{AUDIT_SCHEMA_VERSION}`"
    # Restrict the search to the audit-schema section so a stray
    # mention elsewhere in the spec doesn't paper over a stale
    # in-section claim.
    section_start = spec_text.find("## Audit log schema")
    assert section_start != -1, "spec 01 missing '## Audit log schema' section"
    next_section = spec_text.find("\n## ", section_start + 1)
    section = (
        spec_text[section_start:next_section] if next_section != -1 else spec_text[section_start:]
    )
    assert needle in section, (
        f"docs/specs/01-cli.md '## Audit log schema' section does not name "
        f"AUDIT_SCHEMA_VERSION={AUDIT_SCHEMA_VERSION}. Expected substring "
        f"{needle!r}; update the spec to match the constant in audit.py."
    )


# ---------------------------------------------------------------------------
# Spec 04 corpus provider catalog ↔ recipe model + runtime registry
#
# Spec 04 has one ``## `<type>` `` heading per documented provider
# (``local``, ``web``, ``web_list``, ``wikisource``, ``hf_dataset``,
# ``internet_archive``, ``gutenberg``). The recipe model declares a
# pydantic discriminated union over the ``type`` field — anything not
# in the union is rejected at YAML load. The runtime registry registers
# the M03 builtins; a recipe whose ``type`` model-validates but isn't
# registered crashes at render with ``ProviderError``.
#
# The companion validation check ``corpus_provider_not_implemented``
# (this iter, mirroring iter 65's degradation pattern) bridges the gap
# between model-accepted and runtime-registered. This meta-test fixes
# the remaining drift surface: every provider documented in spec 04 is
# either (a) registered with the runtime registry, or (b) explicitly
# tracked as deferred in the M03 roadmap. Without this guard, spec 04
# could quietly accumulate "documented but not implemented" entries
# that never get flagged anywhere.
#
# Direction-by-direction:
#   1. Every spec 04 provider name must be either runtime-registered
#      or roadmap-deferred. A bare spec entry with no roadmap ack is
#      vapourware — fail.
#   2. Every runtime-registered provider must appear in spec 04.
#      Otherwise users have a working feature with no public docs.
#   3. Every ``CorpusEntry`` union member's ``type`` literal must be
#      named in spec 04. The recipe model is what gates "valid YAML",
#      so anything it accepts must be discoverable in the spec.
# ---------------------------------------------------------------------------


_SPEC_04_PROVIDER_HEADER = re.compile(r"^## `([a-z_]+)`\s*$")


def _spec_04_provider_names() -> set[str]:
    """Provider types parsed from ``## `<name>` `` headings in spec 04.

    Only the per-provider section headings match this pattern; the
    other ``##`` headings (``Common keys``, ``Caching``, etc.) use
    plain text and are filtered out by the backtick-wrapping regex.
    """

    names: set[str] = set()
    for line in _spec_text("04-corpus-providers.md").splitlines():
        match = _SPEC_04_PROVIDER_HEADER.match(line.rstrip())
        if match:
            names.add(match.group(1))
    return names


def _registered_corpus_provider_types() -> set[str]:
    """Provider type names registered with the runtime ``default_registry``.

    Wrapped in a helper so a future move of the registry source-of-
    truth (e.g. switching to a different module) is one edit, not
    spread across every meta-test.
    """

    from pd_ocr_synth.corpus.registry import default_registry

    return set(default_registry().types())


def _model_corpus_provider_types() -> set[str]:
    """``type`` literal values from the ``CorpusEntry`` discriminated union.

    Walks the union members and reads the ``type`` field's literal
    annotation. Pydantic v2 stores the literal value on the field's
    annotation as ``Literal[<value>]``; iterate the union and collect
    each member's value. A future refactor that flattens or replaces
    the union would surface here as an ``AssertionError``.
    """

    from typing import get_args, get_type_hints

    from pd_ocr_synth.recipe.models import (
        HFDatasetCorpus,
        LocalCorpus,
        WebCorpus,
        WikisourceCorpus,
    )

    members = (WebCorpus, LocalCorpus, HFDatasetCorpus, WikisourceCorpus)
    names: set[str] = set()
    for member in members:
        hints = get_type_hints(member)
        type_hint = hints["type"]
        # Literal[<value>] → tuple of literal values; corpus uses single-
        # element literals so we just unpack them all to be safe.
        for arg in get_args(type_hint):
            names.add(str(arg))
    return names


# Roadmap-tracked deferred providers, per
# ``docs/roadmap/03-corpus.md`` "Built-in providers". Keep this list
# short; it should shrink to empty once the providers ship. If a new
# provider gets added to spec 04 ahead of implementation, it lands
# here too — but the long-term goal is for this set to equal the empty
# set and for ``test_spec_04_providers_match_runtime_or_roadmap`` to
# enforce that direction directly.
_ROADMAP_DEFERRED_CORPUS_PROVIDERS: frozenset[str] = frozenset(
    {
        "web_list",
        "hf_dataset",
        "internet_archive",
        "gutenberg",
    }
)


def test_spec_04_providers_match_runtime_or_roadmap() -> None:
    """Every provider documented in spec 04 must be implemented or roadmap-deferred.

    Catches "vapourware drift": a new provider gets a section in spec
    04 but never lands code, never lands a roadmap entry, and quietly
    presents itself to recipe authors as a real option. This test
    mirrors the degradation-kind catalog meta-tests in
    ``test_validation.py`` (iter 66, ``test_known_degradation_kinds_matches_spec_doc``)
    but on the corpus side: spec is the user-facing contract; runtime
    + roadmap are the ground truth.
    """

    spec_names = _spec_04_provider_names()
    registered = _registered_corpus_provider_types()
    accounted_for = registered | _ROADMAP_DEFERRED_CORPUS_PROVIDERS
    spec_only = sorted(spec_names - accounted_for)
    assert not spec_only, (
        "docs/specs/04-corpus-providers.md documents providers that are "
        "neither registered with default_registry() nor tracked as "
        f"deferred in docs/roadmap/03-corpus.md: {spec_only}.\n"
        "Either ship the provider (and add it to default_registry()), "
        "remove the spec section, or add the name to "
        "_ROADMAP_DEFERRED_CORPUS_PROVIDERS in this test "
        "after adding a roadmap entry."
    )


def test_registered_corpus_providers_documented_in_spec_04() -> None:
    """Every registered provider must have a section in spec 04.

    Reverse direction of the catalog match: a runtime provider with
    no public spec entry is a working feature recipe authors can't
    discover. Force the spec update.
    """

    spec_names = _spec_04_provider_names()
    registered = _registered_corpus_provider_types()
    runtime_only = sorted(registered - spec_names)
    assert not runtime_only, (
        "default_registry() registers providers missing from "
        f"docs/specs/04-corpus-providers.md: {runtime_only}.\n"
        "Add a `## `<name>`` section to spec 04 documenting the "
        "options each provider accepts."
    )


def test_corpus_model_types_documented_in_spec_04() -> None:
    """Every ``CorpusEntry`` union member's ``type`` literal must be in spec 04.

    The recipe model is what decides "valid YAML"; if the model
    accepts a ``type`` value the spec doesn't describe, recipe authors
    can write YAML that loads but has no documented semantics. This
    test enforces the model-side ↔ doc-side contract independently of
    runtime registration so the gap caught by
    ``corpus_provider_not_implemented`` (model-accepted but not
    registered) still has a public spec page describing intended
    behaviour.
    """

    spec_names = _spec_04_provider_names()
    model_names = _model_corpus_provider_types()
    model_only = sorted(model_names - spec_names)
    assert not model_only, (
        "CorpusEntry union members' ``type`` literals not documented in "
        f"docs/specs/04-corpus-providers.md: {model_only}.\n"
        "Add a `## `<name>`` section to spec 04 (or remove the "
        "model member if the type was withdrawn)."
    )


# ---------------------------------------------------------------------------
# Spec 05 ↔ text-transform registry drift
#
# Every transform documented in ``docs/specs/05-text-transforms.md`` must
# be either registered with the runtime ``default_registry()`` or tracked
# as deferred in ``docs/roadmap/04-text-transforms.md``. The companion
# validation check ``text_transform_not_implemented`` (this iter) catches
# *recipe-named* drift at validate time; this meta-test catches *spec-vs-
# code* drift at test time. Together they mirror the iter-65 / iter-73
# pattern for degradation kinds and corpus providers.
#
# Direction-by-direction:
#   1. Every spec 05 transform name must be either runtime-registered or
#      roadmap-deferred. A bare spec entry with no roadmap ack is
#      vapourware — fail.
#   2. Every runtime-registered transform must appear in spec 05.
#      Otherwise users have a working feature with no public docs.
# ---------------------------------------------------------------------------


# Match ``### `name` / `name2` / ...`` headers (one or more inline-code
# names, slash-separated). Each captured group becomes a separate
# transform name. Spec 05 uses this form for ``lowercase / uppercase``
# and ``nfc / nfd / nfkc / nfkd``.
_SPEC_05_TRANSFORM_HEADER = re.compile(r"^### (`[a-z_]+`(?:\s*/\s*`[a-z_]+`)*)\s*$")
_SPEC_05_INLINE_CODE = re.compile(r"`([a-z_]+)`")


def _spec_05_transform_names() -> set[str]:
    """Transform names parsed from ``### `name`` (and slash-joined) headings in spec 05.

    The spec groups related transforms onto a single ``###`` line when
    convenient — ``### `lowercase` / `uppercase```, ``### `nfc` / `nfd`
    / `nfkc` / `nfkd```, etc. Each name in the line is a distinct
    registered callable, so the parser unpacks them all.
    """

    names: set[str] = set()
    for line in _spec_text("05-text-transforms.md").splitlines():
        match = _SPEC_05_TRANSFORM_HEADER.match(line.rstrip())
        if not match:
            continue
        for inline in _SPEC_05_INLINE_CODE.findall(match.group(1)):
            names.add(inline)
    return names


def _registered_text_transform_names() -> set[str]:
    """Transform names registered with the runtime ``default_registry``.

    Wrapped in a helper so a future move of the registry source-of-
    truth (e.g. switching to a different module) is one edit, not
    spread across every meta-test.
    """

    from pd_ocr_synth.text_transforms import default_registry

    return set(default_registry().names())


# Roadmap-tracked deferred transforms, per
# ``docs/roadmap/04-text-transforms.md`` "Antique-conventions
# built-ins" and "``python:`` inline loader". Keep this list short; it
# should shrink to empty once the transforms ship. If a new transform
# gets added to spec 05 ahead of implementation, it lands here too —
# but the long-term goal is for this set to equal the empty set and
# for ``test_spec_05_transforms_match_registry_or_roadmap`` to enforce
# that direction directly.
_ROADMAP_DEFERRED_TEXT_TRANSFORMS: frozenset[str] = frozenset(
    {
        "u_v_swap",
        "i_j_swap",
        "ct_st_ligature_marker",
    }
)


def test_spec_05_transforms_match_registry_or_roadmap() -> None:
    """Every transform documented in spec 05 must be registered or roadmap-deferred.

    Catches "vapourware drift": a new transform gets a section in spec
    05 but never lands code, never lands a roadmap entry, and quietly
    presents itself to recipe authors as a real option. This test
    mirrors the corpus-provider catalog meta-test
    (``test_spec_04_providers_match_runtime_or_roadmap``) and the
    degradation-kind catalog meta-test
    (``test_known_degradation_kinds_matches_spec_doc`` in
    ``test_validation.py``).
    """

    spec_names = _spec_05_transform_names()
    registered = _registered_text_transform_names()
    accounted_for = registered | _ROADMAP_DEFERRED_TEXT_TRANSFORMS
    spec_only = sorted(spec_names - accounted_for)
    assert not spec_only, (
        "docs/specs/05-text-transforms.md documents transforms that are "
        "neither registered with default_registry() nor tracked as "
        f"deferred in docs/roadmap/04-text-transforms.md: {spec_only}.\n"
        "Either ship the transform (and add it to register_builtins()), "
        "remove the spec section, or add the name to "
        "_ROADMAP_DEFERRED_TEXT_TRANSFORMS in this test "
        "after adding a roadmap entry."
    )


def test_registered_text_transforms_documented_in_spec_05() -> None:
    """Every registered transform must have a section in spec 05.

    Reverse direction of the catalog match: a runtime transform with
    no public spec entry is a working feature recipe authors can't
    discover. Force the spec update.
    """

    spec_names = _spec_05_transform_names()
    registered = _registered_text_transform_names()
    runtime_only = sorted(registered - spec_names)
    assert not runtime_only, (
        "register_builtins() registers transforms missing from "
        f"docs/specs/05-text-transforms.md: {runtime_only}.\n"
        "Add a `### `<name>`` section to spec 05 documenting the "
        "options each transform accepts."
    )


# ---------------------------------------------------------------------------
# Argparse dest ↔ dispatch reads (iter 80 audit)
#
# Every argparse flag the CLI declares must be read by some code path,
# otherwise users pass it expecting an effect and get silent
# drop-on-the-floor — same drift class as iter 76 (antialiasing) and
# iter 78 (per-stage degradation options).
#
# Iter 80 found 8 such silent-ignore drifts across the
# ``fetch`` / ``preview`` / ``render`` subparsers (mostly inherited
# from ``_add_common_render_args`` and never plumbed). The fix wired
# ``--no-cache`` end-to-end for ``preview`` and ``render``; the rest
# of the iter-80 audit is captured here and below as future drift to
# be addressed (currently allow-listed so the guard doesn't fire on
# pre-existing gaps).
# ---------------------------------------------------------------------------


def _cli_source() -> str:
    cli_path = Path(__file__).resolve().parent.parent / "src" / "pd_ocr_synth" / "cli.py"
    return cli_path.read_text(encoding="utf-8")


# Dests we deliberately do not require a read for. ``command`` is read
# in ``main()`` to dispatch and never bound to a subparser-level
# meaning. Positional ``recipe`` and ``output_dir`` and ``name`` are
# read by every implemented dispatch entry — the broad regex below
# already counts them, but listing them here documents intent.
_DEST_READ_EXEMPT: frozenset[str] = frozenset(
    {
        "command",  # read in main(), not a subparser dest
    }
)

# Known pre-existing silent-ignore drifts the iter-80 audit found.
# Each entry is a (subparser, dest) pair; the meta-test allow-lists
# them so it can pass today and start failing the moment the next
# regression sneaks in. Removing an entry from this set is the gating
# step when each drift is fixed in a follow-up commit.
#
# Notes on each:
#   fetch:count/output/seed/workers/dry_run — inherited from
#       ``_add_common_render_args`` but spec 01's render-family
#       table claims they apply to fetch too. ``--dry-run`` is the
#       most defensible follow-up (validate + plan only); the rest
#       are spec-side ambiguity (``count`` for fetch is undefined).
#   (preview:dry_run — fixed in iter 81; the dispatch now plumbs
#    ``args.dry_run`` to ``_cmd_preview`` which delegates to
#    ``plan_recipe`` for the same dry-run summary as ``render``.)
_KNOWN_UNREAD_DESTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("fetch", "count"),
        ("fetch", "output"),
        ("fetch", "seed"),
        ("fetch", "workers"),
        ("fetch", "dry_run"),
    }
)


def test_every_argparse_dest_is_read_by_dispatch() -> None:
    """Every argparse flag dest must be read by some ``args.<dest>``.

    Catches the iter-76 / iter-80 drift class: the parser declares a
    flag, the user passes it, and the dispatch never reads it. The
    fix is to either implement the flag (preferred) or remove it.

    The CLI module's ``args.<dest>`` reads are the canonical record.
    Static-scan ``cli.py`` rather than running the parser — that way
    we catch a removed dispatch line even if argparse still accepts
    the flag.

    Allow-listed pre-existing gaps live in ``_KNOWN_UNREAD_DESTS``;
    fixing each is a separate commit that also removes the entry.
    """

    parser = build_parser()
    sub_map = _argparse_subparsers(parser)

    cli_src = _cli_source()
    read_pattern = re.compile(r"\bargs\.([a-z_][a-z0-9_]*)")
    read_dests: set[str] = set(read_pattern.findall(cli_src))

    drift: list[str] = []
    for sub_name, sub_parser in sub_map.items():
        for action in sub_parser._actions:  # noqa: SLF001
            dest = action.dest
            if dest in _DEST_READ_EXEMPT:
                continue
            if dest in {"help", "version"}:
                continue
            if (sub_name, dest) in _KNOWN_UNREAD_DESTS:
                continue
            if not action.option_strings and dest in {"recipe"}:
                # Positional ``recipe`` is read on every implemented
                # subcommand; the regex picks it up.
                pass
            if dest not in read_dests:
                drift.append(
                    f"  {sub_name}: --{action.option_strings[0] if action.option_strings else dest} (dest={dest!r})"
                )

    assert not drift, (
        "argparse declares flags that no dispatch path reads — "
        "users pass them and get silent no-op (same drift class as "
        "iter 76 antialiasing and iter 78 per-stage options).\n"
        "Either implement the flag (plumb args.<dest> through the "
        "dispatch lambda + cmd helper) or remove the add_argument "
        "call.\nUnread:\n" + "\n".join(drift)
    )


def test_known_unread_dests_are_actually_unread() -> None:
    """Sanity-check that the allow-list itself isn't stale.

    If a previously-unread dest gets wired up in a follow-up commit,
    its ``_KNOWN_UNREAD_DESTS`` entry should be removed at the same
    time. This test catches the case where someone adds the read but
    forgets to delete the allow-list entry, which would mask future
    regressions of the same flag.
    """

    cli_src = _cli_source()

    parser = build_parser()
    sub_map = _argparse_subparsers(parser)

    stale: list[tuple[str, str]] = []
    for sub_name, dest in _KNOWN_UNREAD_DESTS:
        # The allow-list entry only makes sense if (a) the subparser
        # still exists and (b) the dest actually appears as a flag on
        # that subparser. If both are true and the dispatch now reads
        # ``args.<dest>``, drop the allow-list entry — the drift is
        # gone. We also need to be careful: a dest like ``output``
        # is read globally (publish, schema, render, ...). To keep
        # the allow-list useful we only flag the entry as stale when
        # there's strong evidence the *specific* subparser's
        # dispatch path now reads it. Use a conservative heuristic:
        # if the dispatch lambda for this subparser passes
        # ``args.<dest>`` literally, it's wired.
        if sub_name not in sub_map:
            stale.append((sub_name, dest))
            continue
        # Find the dispatch lambda block for this subparser. The
        # dispatch dict in cli.py uses ``"<name>": lambda args: ...``
        # so we slice around that signature.
        marker = f'"{sub_name}": lambda args:'
        idx = cli_src.find(marker)
        if idx == -1:
            # Subparser exists but has no dispatch entry yet (e.g.
            # only a stub). Allow-list entries for stubs are fine.
            continue
        # Find the closing of this lambda's argument call. The
        # dispatch dict entries are short — the next ``"<word>":
        # lambda args:`` or the closing ``}`` of the dict marks the
        # end. Take everything up to whichever comes first.
        rest = cli_src[idx + len(marker) :]
        next_entry = re.search(r'\n\s*"[a-z_]+": lambda args:', rest)
        end = next_entry.start() if next_entry else rest.find("\n}")
        block = rest[:end] if end != -1 else rest
        if dest in set(re.findall(r"\bargs\.([a-z_][a-z0-9_]*)", block)):
            stale.append((sub_name, dest))

    assert not stale, (
        "stale entries in _KNOWN_UNREAD_DESTS — the dispatch now "
        "reads these flags but they're still allow-listed. Remove "
        f"them: {sorted(stale)}"
    )
