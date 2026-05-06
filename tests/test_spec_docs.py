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

from pathlib import Path

# Canonical filename constants live with the writers. Importing them
# here means a future rename in code surfaces as a static failure
# (ImportError) rather than a silent doc/code mismatch.
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
