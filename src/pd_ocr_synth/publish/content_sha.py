"""Content-SHA digest over a built HF staging directory.

Per ``docs/specs/10-publishing.md`` § Idempotency and the matching
deliverable in ``docs/roadmap/08-publishing-hf.md``:

> Compute a content SHA over the staging directory (sorted file list +
> per-file SHA-256). Persist it as ``pd-ocr-content-sha`` in the
> dataset card front matter.

The SHA closes the idempotency loop: ``publish`` reads the latest HF
commit's ``card_data.pd-ocr-content-sha``; if it matches the freshly
computed digest, nothing has changed and the publish exits 0 with "no
changes" without creating a commit. ``--dry-run`` prints the same
digest so a caller can preview whether a re-run would be a no-op.

Two pieces live here, both pure file-IO (no network, no HF SDK):

1. :func:`compute_content_sha` — given a built staging directory,
   produce a deterministic 64-char hex digest.
2. :func:`apply_content_sha_to_readme` — rewrite
   ``<staging>/README.md`` so its front matter carries
   ``pd-ocr-content-sha: <digest>``. The dataset-card writer in
   ``dataset_card.py`` deliberately omits this key (see its module
   docstring): the SHA is computed *over* the staging dir contents
   (which include the README), so it must be inserted *after* the
   rest of the staging build has settled.

## Determinism contract

The digest must be invariant across:

- **Filesystem walk order.** We sort relative paths byte-wise (POSIX
  separator) before hashing, so any two filesystems / OSes produce
  the same digest for the same logical content.
- **Pre-existing ``pd-ocr-content-sha`` in README.md.** The README is
  hashed as it sits on disk, but ``apply_content_sha_to_readme``
  always strips any old ``pd-ocr-content-sha`` line before inserting
  the new one — see :func:`apply_content_sha_to_readme` for why this
  matters and how round-trips stay stable.

The digest must change for *any* logical change to the dataset:

- Image bytes added, removed, or modified.
- ``metadata.jsonl`` row text or column changes.
- ``recipe.snapshot.yaml`` byte-level changes.
- README body / front-matter changes (excluding the
  ``pd-ocr-content-sha`` line itself, which is intentionally part of
  the strip-and-replace cycle).

What we *don't* hash:

- File mode, mtime, owner — content only.
- Empty / hidden directories — only files contribute.

We do hash filenames: a file rename is a real content change and
should produce a new digest.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pd_ocr_synth.publish.dataset_card import README_FILENAME

# Public algorithm name. Centralized so callers (CLI summary,
# upload-time card-writer) can label the digest consistently and so
# tests can assert against one name.
CONTENT_SHA_ALGORITHM = "sha256"

# Front-matter key. Mirrors the conventional ``pd-ocr-*`` namespace
# the rest of ``dataset_card`` already uses. Defined here (rather than
# imported back from ``dataset_card``) to keep that module free of any
# notion of content-SHA — it stays a pure render of the static front
# matter.
CONTENT_SHA_KEY = "pd-ocr-content-sha"

# Chunk size for streaming file reads. 1 MiB balances syscall
# overhead against memory pressure on the largest images we expect
# (recognition crops are tiny but the publisher must survive
# detection-mode parquet shards in the future).
_READ_CHUNK_BYTES = 1 << 20


class ContentShaError(Exception):
    """Raised when the staging dir can't be hashed.

    Distinct exception type so the future CLI can map it cleanly: a
    missing staging dir is a programmer-side bug (publish was called
    before staging built), distinct from the recipe / corpus / render
    error families.
    """


def compute_content_sha(staging_dir: Path) -> str:
    """Compute the content SHA over a built staging directory.

    Walks every file under ``staging_dir`` recursively, hashes each
    file's bytes with SHA-256, and returns the SHA-256 of the sorted
    ``"<relpath>\\n<file_sha>\\n"`` lines. The two-level structure
    (per-file SHAs feeding into a top-level SHA) means a callers can,
    in future, expose the per-file SHAs for sub-tree comparisons
    without changing the top-level digest.

    Path normalization:

    - Relative paths are POSIX-style (``data/0000000.png``) regardless
      of host OS, so digests are portable across Linux / macOS /
      Windows render hosts.
    - Sorting is byte-wise on the POSIX form. ASCII filenames sort
      identically to lexicographic; non-ASCII metadata names (we don't
      currently emit any) would sort by UTF-8 byte order.

    Parameters
    ----------
    staging_dir:
        Built staging directory. Must exist; need not yet have a
        README, but typically does (call this *after*
        ``build_recognition_staging``).

    Returns
    -------
    str
        Lower-case hex digest, 64 characters.

    Raises
    ------
    ContentShaError
        If ``staging_dir`` doesn't exist or isn't a directory.
    """

    root = Path(staging_dir)
    if not root.exists():
        raise ContentShaError(f"staging dir does not exist: {root}")
    if not root.is_dir():
        raise ContentShaError(f"staging path is not a directory: {root}")

    entries: list[tuple[str, str]] = []
    for path in _walk_files(root):
        relpath = path.relative_to(root).as_posix()
        entries.append((relpath, _hash_file(path)))

    # Sort by relpath only. The per-file SHA is fully determined by
    # the relpath in a built staging dir, so a separate tie-break
    # isn't needed; sorting the tuple would still be stable but is
    # marginally slower on long lists.
    entries.sort(key=lambda item: item[0])

    top = hashlib.sha256()
    for relpath, file_sha in entries:
        top.update(relpath.encode("utf-8"))
        top.update(b"\n")
        top.update(file_sha.encode("ascii"))
        top.update(b"\n")
    return top.hexdigest()


def apply_content_sha_to_readme(staging_dir: Path, content_sha: str) -> Path:
    """Insert ``pd-ocr-content-sha`` into the staging README's front matter.

    Strips any pre-existing ``pd-ocr-content-sha:`` line before
    inserting the new value. This makes the operation idempotent:
    calling it twice with the same SHA leaves the README byte-for-byte
    unchanged, and calling it twice with different SHAs leaves only
    the latest value (no stale duplicates).

    Why we strip-and-rewrite rather than recompute the digest with the
    old line included: a publish workflow conceptually computes the
    SHA *over the staging dir as it would be uploaded*. Once the SHA
    is in the README, hashing again with that line would change the
    digest. The stable convention is "hash the staging dir with no
    ``pd-ocr-content-sha`` line in the README, then write that digest
    in" — :func:`compute_content_sha` is called *before* this helper.

    The insertion point is right after the last ``pd-ocr-*`` key in
    the front matter so the conventional keys stay grouped. If no
    front matter is found, one is created.

    Parameters
    ----------
    staging_dir:
        Directory containing the README to rewrite.
    content_sha:
        The hex digest to record. Stored verbatim; callers should
        pass the output of :func:`compute_content_sha`.

    Returns
    -------
    Path
        The README path that was rewritten.

    Raises
    ------
    ContentShaError
        If ``<staging_dir>/README.md`` doesn't exist. The publish
        flow always builds the staging dir (which writes a README)
        before calling this; the missing-README case indicates a
        programmer-side ordering bug worth surfacing.
    """

    if not content_sha:
        raise ContentShaError("content_sha must be a non-empty string")

    readme_path = staging_dir / README_FILENAME
    if not readme_path.is_file():
        raise ContentShaError(f"staging dir has no README to update: {readme_path}")

    text = readme_path.read_text(encoding="utf-8")
    rewritten = _embed_content_sha(text, content_sha)
    readme_path.write_text(rewritten, encoding="utf-8")
    return readme_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _walk_files(root: Path) -> list[Path]:
    """Collect every file under ``root`` (recursive, files only).

    ``Path.rglob('*')`` yields directories too; we filter to files so
    empty directories don't perturb the hash. Symlinks are followed
    (``Path.is_file()`` returns ``True`` for a symlink to a regular
    file); this matches what ``upload_large_folder`` would actually
    upload in practice.
    """

    return [p for p in root.rglob("*") if p.is_file()]


def _hash_file(path: Path) -> str:
    """SHA-256 of a single file's bytes, streamed to bound memory."""

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# Match a top-level YAML key line for ``pd-ocr-content-sha``. Tolerant
# of leading whitespace (none expected in our writer but cheap to
# allow) and of the line being the last in the front-matter body
# without a trailing newline (``\Z``). The trailing-``\n`` consume
# keeps the surrounding body byte-stable so an idempotent re-apply
# doesn't grow the file.
_CONTENT_SHA_LINE_RE = re.compile(
    r"^[ \t]*" + re.escape(CONTENT_SHA_KEY) + r"[ \t]*:[^\n]*(?:\n|\Z)",
    re.MULTILINE,
)

# Match a YAML front-matter block delimited by ``---`` lines at the
# start of the file. ``DOTALL`` so the body capture includes
# newlines; ``\A`` anchors at the file head so we don't match a
# secondary ``---`` deeper in the body.
_FRONT_MATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)


def _embed_content_sha(text: str, content_sha: str) -> str:
    """Return ``text`` with the front matter carrying ``content_sha``.

    Three cases:

    1. Front matter exists — strip any old content-SHA line, append a
       fresh one inside the front-matter block, return the rewritten
       text.
    2. Front matter is missing entirely — synthesize a one-key block
       and prepend it. (Defensive; the staging builder always writes
       a front matter.)

    The new line is appended at the *end* of the front-matter body so
    diffs across runs only ever touch one line.
    """

    match = _FRONT_MATTER_RE.match(text)
    new_line = f"{CONTENT_SHA_KEY}: {content_sha}\n"

    if match is None:
        return f"---\n{new_line}---\n\n{text}" if text else f"---\n{new_line}---\n"

    body = match.group("body")
    body_stripped = _CONTENT_SHA_LINE_RE.sub("", body)
    if not body_stripped.endswith("\n"):
        body_stripped += "\n"
    new_body = body_stripped + new_line
    rest = text[match.end() :]
    return f"---\n{new_body}---\n{rest}"
