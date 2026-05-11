"""Unit tests for the staging-dir content SHA (M08).

Covers ``pd_ocr_synth.publish.content_sha``: deterministic digest over
the staging-dir contents and the README front-matter rewrite that
embeds the digest under ``pd-ocr-content-sha``. Pure file-IO; no
network, no HF SDK.

The tests treat the digest as a *contract*, not an implementation
detail: equal contents must produce equal digests across two builds,
unequal contents must produce unequal digests, and inserting the SHA
into the README is idempotent (same SHA written twice = no-op).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from PIL import Image

from pd_ocr_synth.publish import (
    CONTENT_SHA_ALGORITHM,
    CONTENT_SHA_KEY,
    README_FILENAME,
    ContentShaError,
    apply_content_sha_to_readme,
    build_recognition_staging,
    compute_content_sha,
)
from pd_ocr_synth.publish.content_sha import (
    _CONTENT_SHA_LINE_RE,
    _embed_content_sha,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _build_staging(tmp_path: Path, *, suffix: str = "") -> Path:
    """Build a small but realistic staging dir from a synthetic local output.

    Reuses the production ``build_recognition_staging`` so the test
    exercises the real layout (``data/<NNNNNNN>.png`` +
    ``metadata.jsonl`` + ``recipe.snapshot.yaml`` + ``README.md``).
    The ``suffix`` keeps multiple builds in the same ``tmp_path``
    isolated.
    """

    local = tmp_path / f"local{suffix}"
    images = local / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(180, 180, 180)).save(images / "0000000.png", format="PNG")
    Image.new("RGB", (8, 8), color=(120, 120, 120)).save(images / "0000001.png", format="PNG")
    (local / "labels.json").write_text(
        '{"0000000.png": "Séadna", "0000001.png": "agus"}\n', encoding="utf-8"
    )
    (local / "manifest.jsonl").write_text(
        '{"index": 0, "id": "0000000", "image": "images/0000000.png", '
        '"text": "Séadna", "status": "rendered", '
        '"font": {"name": "bungc.otf", "path": "/abs/bungc.otf", "size_pt": 14.0}}\n'
        '{"index": 1, "id": "0000001", "image": "images/0000001.png", '
        '"text": "agus", "status": "rendered", '
        '"font": {"name": "seangc.otf", "path": "/abs/seangc.otf", "size_pt": 14.0}}\n',
        encoding="utf-8",
    )
    (local / "recipe.snapshot.yaml").write_text(
        "tool_version: 0.0.0\n"
        "seed: 5\n"
        "recipe:\n"
        "  name: gaelic-test\n"
        "  schema_version: 1\n"
        "  fonts:\n"
        "    - {path: /abs/fonts/bungc.otf}\n",
        encoding="utf-8",
    )
    (local / "stats.json").write_text(
        '{"samples_written": 2, "fonts_used": {"bungc.otf": 1, "seangc.otf": 1}}\n',
        encoding="utf-8",
    )

    staging = tmp_path / f"staging{suffix}"
    build_recognition_staging(local, staging)
    return staging


# ---------------------------------------------------------------------------
# compute_content_sha — happy path / format
# ---------------------------------------------------------------------------


def test_compute_content_sha_returns_lowercase_hex_64(tmp_path: Path) -> None:
    """SHA must look like a SHA-256 hex digest."""

    staging = _build_staging(tmp_path)

    digest = compute_content_sha(staging)

    assert isinstance(digest, str)
    assert len(digest) == 64
    assert digest == digest.lower()
    assert re.fullmatch(r"[0-9a-f]{64}", digest), digest


def test_compute_content_sha_algorithm_constant_matches_hashlib() -> None:
    """The exposed algorithm name should be a hashlib-known one."""

    assert CONTENT_SHA_ALGORITHM in hashlib.algorithms_guaranteed
    # Sanity: tests above assert SHA-256 width; lock the constant
    # alongside that assumption.
    assert CONTENT_SHA_ALGORITHM == "sha256"


# ---------------------------------------------------------------------------
# Determinism — equal content → equal digest
# ---------------------------------------------------------------------------


def test_compute_content_sha_is_deterministic_across_rebuilds(tmp_path: Path) -> None:
    """Two staging dirs built from the same inputs hash identically.

    This is the load-bearing property: idempotency relies on a
    re-render with no logical change producing the same digest as the
    previous run.
    """

    staging_a = _build_staging(tmp_path, suffix="a")
    staging_b = _build_staging(tmp_path, suffix="b")

    assert compute_content_sha(staging_a) == compute_content_sha(staging_b)


def test_compute_content_sha_ignores_filesystem_walk_order(tmp_path: Path) -> None:
    """Adding files in a different order leaves the digest unchanged.

    Builds a synthetic flat staging dir (no real images) so we can
    control creation order: the digest must depend only on the final
    content, not the sequence of writes.
    """

    a = tmp_path / "a"
    a.mkdir()
    (a / "z.txt").write_text("zeta\n")
    (a / "m.txt").write_text("mu\n")
    (a / "a.txt").write_text("alpha\n")

    b = tmp_path / "b"
    b.mkdir()
    (b / "a.txt").write_text("alpha\n")
    (b / "z.txt").write_text("zeta\n")
    (b / "m.txt").write_text("mu\n")

    assert compute_content_sha(a) == compute_content_sha(b)


def test_compute_content_sha_ignores_empty_directories(tmp_path: Path) -> None:
    """Empty subdirectories don't contribute to the digest.

    HF's ``upload_large_folder`` won't upload an empty dir; matching
    that behavior keeps the digest aligned with what's actually
    shipped to the hub.
    """

    a = tmp_path / "a"
    (a / "data").mkdir(parents=True)
    (a / "data" / "x.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")

    b = tmp_path / "b"
    (b / "data").mkdir(parents=True)
    (b / "data" / "x.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    (b / "empty_extra").mkdir()

    assert compute_content_sha(a) == compute_content_sha(b)


# ---------------------------------------------------------------------------
# Sensitivity — any content change → different digest
# ---------------------------------------------------------------------------


def test_compute_content_sha_changes_on_image_byte_change(tmp_path: Path) -> None:
    staging_a = _build_staging(tmp_path, suffix="a")
    staging_b = _build_staging(tmp_path, suffix="b")
    sha_a = compute_content_sha(staging_a)

    # Mutate one image byte in B.
    target = staging_b / "data" / "0000000.png"
    target.write_bytes(target.read_bytes() + b"\x00")

    sha_b = compute_content_sha(staging_b)
    assert sha_a != sha_b


def test_compute_content_sha_changes_on_metadata_change(tmp_path: Path) -> None:
    staging_a = _build_staging(tmp_path, suffix="a")
    staging_b = _build_staging(tmp_path, suffix="b")
    sha_a = compute_content_sha(staging_a)

    metadata = staging_b / "metadata.jsonl"
    text = metadata.read_text(encoding="utf-8")
    metadata.write_text(text.replace("Séadna", "Séadnaí"), encoding="utf-8")

    sha_b = compute_content_sha(staging_b)
    assert sha_a != sha_b


def test_compute_content_sha_changes_on_filename_rename(tmp_path: Path) -> None:
    """Renames count as content change — the upload would commit them."""

    a = tmp_path / "a"
    a.mkdir()
    (a / "alpha.txt").write_text("hello\n")

    b = tmp_path / "b"
    b.mkdir()
    (b / "beta.txt").write_text("hello\n")

    assert compute_content_sha(a) != compute_content_sha(b)


def test_compute_content_sha_changes_on_added_file(tmp_path: Path) -> None:
    a = tmp_path / "a"
    a.mkdir()
    (a / "x.txt").write_text("x\n")

    b = tmp_path / "b"
    b.mkdir()
    (b / "x.txt").write_text("x\n")
    (b / "y.txt").write_text("y\n")

    assert compute_content_sha(a) != compute_content_sha(b)


def test_compute_content_sha_changes_on_readme_body_edit(tmp_path: Path) -> None:
    """README body changes propagate to the SHA."""

    staging_a = _build_staging(tmp_path, suffix="a")
    staging_b = _build_staging(tmp_path, suffix="b")
    sha_a = compute_content_sha(staging_a)

    readme = staging_b / README_FILENAME
    readme.write_text(readme.read_text(encoding="utf-8") + "\n# extra\n", encoding="utf-8")
    sha_b = compute_content_sha(staging_b)

    assert sha_a != sha_b


# ---------------------------------------------------------------------------
# compute_content_sha — error paths
# ---------------------------------------------------------------------------


def test_compute_content_sha_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ContentShaError, match="does not exist"):
        compute_content_sha(tmp_path / "nope")


def test_compute_content_sha_path_is_file_raises(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(ContentShaError, match="not a directory"):
        compute_content_sha(f)


def test_compute_content_sha_handles_empty_dir(tmp_path: Path) -> None:
    """An empty staging dir is a valid input — hashes the empty list."""

    staging = tmp_path / "empty"
    staging.mkdir()
    digest = compute_content_sha(staging)
    assert len(digest) == 64
    # The empty-input digest is the SHA-256 of nothing — i.e., the
    # well-known constant. Locks the canonical-empty contract.
    assert digest == hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# apply_content_sha_to_readme — happy path
# ---------------------------------------------------------------------------


def test_apply_content_sha_to_readme_inserts_key(tmp_path: Path) -> None:
    staging = _build_staging(tmp_path)
    digest = compute_content_sha(staging)

    apply_content_sha_to_readme(staging, digest)

    readme_text = (staging / README_FILENAME).read_text(encoding="utf-8")
    assert f"{CONTENT_SHA_KEY}: {digest}" in readme_text


def test_apply_content_sha_to_readme_keeps_front_matter_block(tmp_path: Path) -> None:
    """Existing keys must survive untouched after the rewrite."""

    staging = _build_staging(tmp_path)
    digest = compute_content_sha(staging)
    before = (staging / README_FILENAME).read_text(encoding="utf-8")

    apply_content_sha_to_readme(staging, digest)

    after = (staging / README_FILENAME).read_text(encoding="utf-8")
    # Preserve the key markers that the dataset-card writer always
    # emits: shape, source, recipe-sha, render-tool-version.
    for required in (
        "pd-ocr-shape:",
        "pd-ocr-source:",
        "pd-ocr-recipe-sha:",
        "pd-ocr-render-tool-version:",
    ):
        assert required in before, f"fixture missing {required}"
        assert required in after, f"rewrite dropped {required}"


def test_apply_content_sha_to_readme_preserves_body(tmp_path: Path) -> None:
    """Markdown body (after the front matter) must be byte-stable."""

    staging = _build_staging(tmp_path)
    digest = compute_content_sha(staging)
    before = (staging / README_FILENAME).read_text(encoding="utf-8")
    body_before = before.split("---\n", 2)[2]

    apply_content_sha_to_readme(staging, digest)

    after = (staging / README_FILENAME).read_text(encoding="utf-8")
    body_after = after.split("---\n", 2)[2]
    assert body_before == body_after


def test_apply_content_sha_returns_readme_path(tmp_path: Path) -> None:
    staging = _build_staging(tmp_path)
    digest = compute_content_sha(staging)
    result = apply_content_sha_to_readme(staging, digest)
    assert result == staging / README_FILENAME


# ---------------------------------------------------------------------------
# apply_content_sha_to_readme — idempotency
# ---------------------------------------------------------------------------


def test_apply_content_sha_to_readme_is_idempotent(tmp_path: Path) -> None:
    """Applying the same SHA twice is a no-op on the second call."""

    staging = _build_staging(tmp_path)
    digest = compute_content_sha(staging)
    apply_content_sha_to_readme(staging, digest)
    once = (staging / README_FILENAME).read_text(encoding="utf-8")

    apply_content_sha_to_readme(staging, digest)
    twice = (staging / README_FILENAME).read_text(encoding="utf-8")

    assert once == twice
    # Exactly one occurrence of the key, ever.
    assert once.count(f"{CONTENT_SHA_KEY}:") == 1


def test_apply_content_sha_to_readme_replaces_old_value(tmp_path: Path) -> None:
    """A second SHA replaces (not duplicates) the first."""

    staging = _build_staging(tmp_path)
    apply_content_sha_to_readme(staging, "a" * 64)
    apply_content_sha_to_readme(staging, "b" * 64)

    text = (staging / README_FILENAME).read_text(encoding="utf-8")
    assert f"{CONTENT_SHA_KEY}: {'b' * 64}" in text
    assert f"{CONTENT_SHA_KEY}: {'a' * 64}" not in text
    assert text.count(f"{CONTENT_SHA_KEY}:") == 1


def test_apply_content_sha_to_readme_strips_old_line_only(tmp_path: Path) -> None:
    """Only the matching key line is stripped — siblings stay intact."""

    staging = _build_staging(tmp_path)
    # Inject a sibling key the dataset-card writer doesn't add to be
    # sure the line-strip is targeted.
    readme = staging / README_FILENAME
    text = readme.read_text(encoding="utf-8")
    text = text.replace("---\n\n", "extra-key: hello\n---\n\n", 1)
    readme.write_text(text, encoding="utf-8")

    apply_content_sha_to_readme(staging, "c" * 64)

    after = readme.read_text(encoding="utf-8")
    assert "extra-key: hello" in after
    assert f"{CONTENT_SHA_KEY}: {'c' * 64}" in after


# ---------------------------------------------------------------------------
# apply_content_sha_to_readme — error paths
# ---------------------------------------------------------------------------


def test_apply_content_sha_to_readme_missing_readme_raises(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(ContentShaError, match="no README"):
        apply_content_sha_to_readme(staging, "deadbeef" * 8)


def test_apply_content_sha_to_readme_empty_sha_raises(tmp_path: Path) -> None:
    staging = _build_staging(tmp_path)
    with pytest.raises(ContentShaError, match="non-empty"):
        apply_content_sha_to_readme(staging, "")


# ---------------------------------------------------------------------------
# Internal helper coverage — keeps the strip-and-replace logic honest
# ---------------------------------------------------------------------------


def test_embed_content_sha_creates_front_matter_when_missing() -> None:
    """A README without front matter gets a fresh block prepended."""

    plain = "# title\n\nbody\n"
    out = _embed_content_sha(plain, "f" * 64)
    assert out.startswith("---\n")
    assert f"{CONTENT_SHA_KEY}: {'f' * 64}" in out
    assert "# title" in out
    assert "body" in out


def test_content_sha_line_regex_matches_key_only() -> None:
    """The strip regex shouldn't fire on similarly-named keys."""

    text = "pd-ocr-content-sha-old: nope\npd-ocr-content-sha: real\nother-content-sha: nope\n"
    stripped = _CONTENT_SHA_LINE_RE.sub("", text)
    assert "pd-ocr-content-sha-old: nope" in stripped
    assert "other-content-sha: nope" in stripped
    assert "pd-ocr-content-sha: real" not in stripped


# ---------------------------------------------------------------------------
# Integration — content SHA over a freshly built staging dir matches a
# round-trip rebuild after embedding (sanity, not a stable contract)
# ---------------------------------------------------------------------------


def test_full_round_trip_compute_then_embed(tmp_path: Path) -> None:
    """End-to-end: build → compute → embed → verify the key is there."""

    staging = _build_staging(tmp_path)
    digest = compute_content_sha(staging)
    apply_content_sha_to_readme(staging, digest)

    text = (staging / README_FILENAME).read_text(encoding="utf-8")
    # Key written
    assert f"{CONTENT_SHA_KEY}: {digest}" in text
    # Key sits inside the front-matter block, not loose in the body.
    front_matter, _, body = text.partition("\n---\n")
    assert f"{CONTENT_SHA_KEY}:" in front_matter
    assert f"{CONTENT_SHA_KEY}:" not in body


def test_compute_content_sha_is_invariant_after_apply(tmp_path: Path) -> None:
    """The upload orchestrator's idempotency loop depends on
    ``compute_content_sha`` returning the same value before and after
    :func:`apply_content_sha_to_readme` writes the digest into the
    README. Without this invariance, a re-publish over an unchanged
    staging dir would never see ``UP_TO_DATE``.
    """

    staging = _build_staging(tmp_path)
    digest_before = compute_content_sha(staging)

    apply_content_sha_to_readme(staging, digest_before)
    digest_after = compute_content_sha(staging)

    assert digest_before == digest_after


def test_compute_content_sha_invariant_holds_for_any_pinned_value(
    tmp_path: Path,
) -> None:
    """A README with a *different* (e.g. stale or tampered)
    ``pd-ocr-content-sha`` value must still hash to the canonical
    pre-embed digest. The line is stripped before hashing regardless
    of its current value.
    """

    staging = _build_staging(tmp_path)
    canonical = compute_content_sha(staging)

    # First apply: legitimate digest.
    apply_content_sha_to_readme(staging, canonical)
    assert compute_content_sha(staging) == canonical

    # Second apply: a stale value (e.g. left over from a previous
    # build). The hash is still the canonical pre-embed digest.
    apply_content_sha_to_readme(staging, "0" * 64)
    assert compute_content_sha(staging) == canonical

    # Even an obviously-bogus value (not a hex digest at all).
    apply_content_sha_to_readme(staging, "definitely-not-a-real-sha")
    assert compute_content_sha(staging) == canonical
