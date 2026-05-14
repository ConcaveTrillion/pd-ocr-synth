# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Extract failure-relevant sections from a captured make CI log."""

import re
import sys
from pathlib import Path

MAX_OUTPUT_LINES = 300
FALLBACK_TAIL_LINES = 50


def extract_pytest_sections(text: str) -> list[str]:
    sections = []
    for header in ("FAILURES", "ERRORS", "short test summary info"):
        # Match lines like "===== FAILURES ====="
        pattern = r"(=+\s+" + re.escape(header) + r"\s+=+.*?)(?=\n=+|\Z)"
        m = re.search(pattern, text, re.DOTALL)
        if m:
            sections.append(m.group(1).rstrip())
    return sections


def extract_cargo_failures(text: str) -> list[str]:
    m = re.search(r"(failures:\n\n.*?)(?=\ntest result:|\Z)", text, re.DOTALL)
    return [m.group(1).rstrip()] if m else []


def extract_ruff_lines(text: str) -> list[str]:
    lines = [ln for ln in text.splitlines() if re.match(r".+:\d+:\d+: [A-Z]\d+", ln)]
    return ["\n".join(lines)] if lines else []


def extract_pre_commit_failures(text: str) -> list[str]:
    m = re.search(r"(Failed.*?)(?=\n-{10,}|\Z)", text, re.DOTALL)
    return [m.group(1).rstrip()] if m else []


def extract_error_lines(text: str) -> list[str]:
    """Catch-all: error: lines (mypy, tsc, build errors) with 3-line context."""
    lines = text.splitlines()
    error_re = re.compile(r"error:", re.IGNORECASE)
    hits = [i for i, ln in enumerate(lines) if error_re.search(ln)]
    if not hits:
        return []
    seen: set[int] = set()
    out: list[str] = []
    for i in hits:
        for j in range(max(0, i - 3), min(len(lines), i + 4)):
            if j not in seen:
                out.append(lines[j])
                seen.add(j)
    return ["\n".join(out)]


def fallback_tail(text: str) -> list[str]:
    lines = text.splitlines()
    return ["\n".join(lines[-FALLBACK_TAIL_LINES:])]


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: ai-filter-log.py <logfile>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"log not found: {path}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text(errors="replace")

    extractors = [
        extract_pytest_sections,
        extract_cargo_failures,
        extract_ruff_lines,
        extract_pre_commit_failures,
        extract_error_lines,
    ]

    found: list[str] = []
    for extractor in extractors:
        found.extend(extractor(text))

    if not found:
        found = fallback_tail(text)

    output = "\n\n".join(found)
    output_lines = output.splitlines()
    if len(output_lines) > MAX_OUTPUT_LINES:
        output_lines = output_lines[:MAX_OUTPUT_LINES]
        output_lines.append(f"... (truncated to {MAX_OUTPUT_LINES} lines; see full log)")

    print("\n".join(output_lines))


if __name__ == "__main__":
    main()
