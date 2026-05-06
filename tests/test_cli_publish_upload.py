"""End-to-end tests for ``pd-ocr-synth publish`` real-upload path (M08).

The dry-run path is covered in ``test_cli_publish.py``. These tests
exercise the actual upload orchestration through ``cmd_publish`` with
the ``transport_factory`` seam injected as a :class:`FakeTransport`-
returning callable. No network, no ``huggingface_hub`` import.

Per ``docs/specs/10-publishing.md`` we lock down:

- Spec § Idempotency: a re-publish over an unchanged staging dir is
  a no-op (NO_CHANGE state, exit 0, no upload commit).
- Spec § Versioning: ``--message`` overrides the auto-generated commit
  message; ``--tag`` creates a tag against the upload commit.
- Spec § Errors and recovery: ``--no-create`` on a missing repo fails
  exit 7; transport errors map to exit 7; auth errors map to exit 7.

The tests run a real recipe + render so the staging build hits the
M07 → M08 boundary; only the upload itself is faked. This gives us
the strongest possible "wire-the-CLI" coverage without an HF token.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pd_ocr_synth.publish.cli_runner import cmd_publish
from pd_ocr_synth.publish.content_sha import (
    CONTENT_SHA_KEY,
    compute_content_sha,
)
from pd_ocr_synth.publish.recognition import build_recognition_staging
from pd_ocr_synth.publish.transport import FakeTransport

# Bundled font, same one M07/M08 dry-run tests use.
_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; publish upload test skipped.")
    return _BUNDLED_FONT


_RECIPE_TEMPLATE = """\
schema_version: 1
name: publish-upload-smoke
seed: 11
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {dest}
  count: 4
corpus:
  - type: local
    path: ./seed-words.txt
fonts:
  - path: {font}
    weight: 1.0
rendering:
  font_size_pt: {{ min: 14, max: 16 }}
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 235, b: 220 }}
layout:
  mode: word_crops
  padding_px: 4
"""

_SEED_WORDS = "\n".join(["ḃeaḋ", "ċeann", "ḋuine", "ḟear"]) + "\n"


def _setup_recipe(tmp_path: Path, dest: Path) -> Path:
    font = _require_font()
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_RECIPE_TEMPLATE.format(font=font, dest=dest), encoding="utf-8")
    (tmp_path / "seed-words.txt").write_text(_SEED_WORDS, encoding="utf-8")
    return rp


def _do_render(recipe_path: Path, out: Path) -> None:
    from pd_ocr_synth.cli import main

    rc = main(
        [
            "render",
            str(recipe_path),
            "--count",
            "4",
            "--output",
            str(out),
            "--seed",
            "11",
            "--workers",
            "1",
        ]
    )
    assert rc == 0


def _factory_for(transport: FakeTransport):
    """Build a ``transport_factory`` that returns the seeded fake.

    The factory closure intentionally ignores the token argument so
    tests don't need to thread a real HF token through. The CLI's
    auth resolver still runs (and fails if no token is found), but
    the transport itself is the fake.
    """

    def _factory(token: str) -> FakeTransport:
        # The token reaches us — meaning auth resolution succeeded.
        # We don't need to do anything with it; the fake doesn't
        # authenticate.
        del token
        return transport

    return _factory


# ---------------------------------------------------------------------------
# Happy path: first publish creates + uploads
# ---------------------------------------------------------------------------


def test_real_upload_first_publish_creates_repo_and_uploads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """First publish to a brand-new repo: state CREATED, exit 0,
    summary mentions the commit SHA and content-SHA."""

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_for_upload_" + "a" * 12)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    transport = FakeTransport()
    rc = cmd_publish(
        recipe_arg=str(rp),
        repo_flag="alice/x",
        private=False,
        public=False,
        token_flag=None,
        output_override=None,
        dry_run=False,
        no_create=False,
        tag=None,
        message=None,
        transport_factory=_factory_for(transport),
    )

    captured = capsys.readouterr()
    assert rc == 0, captured.err

    # The fake transport recorded the create + upload sequence.
    op_names = [name for name, _ in transport.calls]
    assert "create_repo" in op_names
    assert "upload_folder" in op_names
    assert op_names.index("create_repo") < op_names.index("upload_folder")

    # Stdout summary: the runner reports the new state plus the
    # short SHAs.
    text = captured.out
    assert "Created and uploaded: alice/x" in text
    assert "commit:" in text
    assert "content-sha:" in text


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_real_upload_unchanged_staging_is_no_op(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Spec 10 § Idempotency: matching content-SHA → exit 0, no upload."""

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_idempotent_" + "b" * 12)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    # Pre-compute the content-SHA the runner would derive: build a
    # staging dir, hash it, then seed the fake repo with that digest.
    # The runner builds its own staging dir but the SHA is a function
    # of the local output, so seeding with this digest exercises the
    # "remote already up to date" branch.
    pre_staging = tmp_path / "pre-staging"
    build_recognition_staging(out, pre_staging)
    expected_sha = compute_content_sha(pre_staging)

    transport = FakeTransport()
    transport.seed_repo("alice/x", card_data={CONTENT_SHA_KEY: expected_sha})

    rc = cmd_publish(
        recipe_arg=str(rp),
        repo_flag="alice/x",
        private=False,
        public=False,
        token_flag=None,
        output_override=None,
        dry_run=False,
        no_create=False,
        tag=None,
        message=None,
        transport_factory=_factory_for(transport),
    )

    captured = capsys.readouterr()
    assert rc == 0, captured.err

    op_names = [name for name, _ in transport.calls]
    assert "upload_folder" not in op_names
    assert "create_repo" not in op_names

    assert "No changes to upload: alice/x" in captured.out
    assert "pd-ocr-content-sha=" in captured.out


# ---------------------------------------------------------------------------
# --message override
# ---------------------------------------------------------------------------


def test_real_upload_message_override_propagates_to_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--message <MSG>`` overrides the auto-generated default. The
    fake transport records the commit message verbatim so we can
    assert on it."""

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_message_" + "c" * 12)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    transport = FakeTransport()
    rc = cmd_publish(
        recipe_arg=str(rp),
        repo_flag="alice/x",
        private=False,
        public=False,
        token_flag=None,
        output_override=None,
        dry_run=False,
        no_create=False,
        tag=None,
        message="ship it: v2026.05.06",
        transport_factory=_factory_for(transport),
    )

    captured = capsys.readouterr()
    assert rc == 0, captured.err

    upload_call = next(call for call in transport.calls if call[0] == "upload_folder")
    assert upload_call[1]["commit_message"] == "ship it: v2026.05.06"


def test_real_upload_default_commit_message_uses_recipe_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No ``--message`` → the default ``pd-ocr-synth render @<recipe-sha>``
    format. The recipe SHA comes from the staging README's
    ``pd-ocr-recipe-sha`` key, set by the dataset-card builder."""

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_default_msg_" + "d" * 12)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    transport = FakeTransport()
    rc = cmd_publish(
        recipe_arg=str(rp),
        repo_flag="alice/x",
        private=False,
        public=False,
        token_flag=None,
        output_override=None,
        dry_run=False,
        no_create=False,
        tag=None,
        message=None,
        transport_factory=_factory_for(transport),
    )

    captured = capsys.readouterr()
    assert rc == 0, captured.err

    upload_call = next(call for call in transport.calls if call[0] == "upload_folder")
    msg = upload_call[1]["commit_message"]
    # Default prefix is the spec-mandated literal.
    assert msg.startswith("pd-ocr-synth render")
    # ``@<recipe-sha>`` carries because the staging README front matter
    # always includes ``pd-ocr-recipe-sha`` (preflight enforces it).
    assert "@" in msg
    sha = msg.split("@", 1)[1]
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)


# ---------------------------------------------------------------------------
# --message limitation warning (spec 10 § Tooling used → Known limitation,
# roadmap 08 § Residual M08 work → Commit-message limitation)
# ---------------------------------------------------------------------------


def test_real_upload_message_override_emits_limitation_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Spec 10 § Tooling used: ``HfApi.upload_large_folder`` does not
    accept ``commit_message`` — the chunked uploader auto-generates
    its own per-shard commit messages. The CLI surfaces this gap by
    printing a single-line stderr warning when ``--message`` is
    explicitly supplied so the user is not silently misled.

    The warning text references both the flag name and the spec
    section so a user grepping logs can find the explanation. We
    assert on those substrings rather than the full string so a
    future copy-edit doesn't break the test gratuitously.
    """

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_msg_warn_" + "e" * 12)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    transport = FakeTransport()
    rc = cmd_publish(
        recipe_arg=str(rp),
        repo_flag="alice/x",
        private=False,
        public=False,
        token_flag=None,
        output_override=None,
        dry_run=False,
        no_create=False,
        tag=None,
        message="ship it: v2026.05.06",
        transport_factory=_factory_for(transport),
    )

    captured = capsys.readouterr()
    assert rc == 0, captured.err

    # Warning lands on stderr, not stdout — stdout is reserved for
    # the publish-result summary that downstream tooling may parse.
    assert "--message" in captured.err
    assert "upload_large_folder" in captured.err
    # Spec citation so the user can find the explanation.
    assert "10-publishing" in captured.err

    # And the message still flows to the transport (FakeTransport
    # honors it; the Protocol stays intact for the future
    # detection-mode push_to_hub path).
    upload_call = next(call for call in transport.calls if call[0] == "upload_folder")
    assert upload_call[1]["commit_message"] == "ship it: v2026.05.06"


def test_real_upload_default_message_does_not_emit_limitation_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Inverse of the previous test: when ``--message`` is omitted,
    no warning should appear. The default ``pd-ocr-synth render
    @<recipe-sha>`` is generated by the runner, not the user, so
    there is no user expectation about it appearing on the remote
    commit and a warning would just be noise.
    """

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_no_warn_" + "f" * 12)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    transport = FakeTransport()
    rc = cmd_publish(
        recipe_arg=str(rp),
        repo_flag="alice/x",
        private=False,
        public=False,
        token_flag=None,
        output_override=None,
        dry_run=False,
        no_create=False,
        tag=None,
        message=None,
        transport_factory=_factory_for(transport),
    )

    captured = capsys.readouterr()
    assert rc == 0, captured.err
    # Warning is only for explicit user override; absent here.
    assert "upload_large_folder" not in captured.err


def test_real_upload_whitespace_only_message_does_not_emit_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A whitespace-only ``--message`` (e.g. shell expansion of an
    unset variable: ``--message "$VAR"``) is treated by
    :func:`resolve_commit_message` as "not provided" and falls
    through to the default. The warning must mirror that rule so we
    don't emit spurious noise when the user did not actually pass a
    real message.
    """

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_ws_msg_" + "g" * 12)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    transport = FakeTransport()
    rc = cmd_publish(
        recipe_arg=str(rp),
        repo_flag="alice/x",
        private=False,
        public=False,
        token_flag=None,
        output_override=None,
        dry_run=False,
        no_create=False,
        tag=None,
        message="   ",
        transport_factory=_factory_for(transport),
    )

    captured = capsys.readouterr()
    assert rc == 0, captured.err
    # Same silent fallthrough as the default-message test above.
    assert "upload_large_folder" not in captured.err


# ---------------------------------------------------------------------------
# --tag
# ---------------------------------------------------------------------------


def test_real_upload_tag_creates_tag_after_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--tag v1`` calls ``create_tag`` against the upload commit
    AFTER ``upload_folder`` — never before."""

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_tag_" + "e" * 12)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    transport = FakeTransport()
    rc = cmd_publish(
        recipe_arg=str(rp),
        repo_flag="alice/x",
        private=False,
        public=False,
        token_flag=None,
        output_override=None,
        dry_run=False,
        no_create=False,
        tag="v2026.05.06",
        message=None,
        transport_factory=_factory_for(transport),
    )

    captured = capsys.readouterr()
    assert rc == 0, captured.err

    op_names = [name for name, _ in transport.calls]
    upload_idx = op_names.index("upload_folder")
    tag_idx = op_names.index("create_tag")
    assert tag_idx > upload_idx
    assert "v2026.05.06" in transport.repos["alice/x"].tags
    assert "tag: v2026.05.06" in captured.out


# ---------------------------------------------------------------------------
# --no-create
# ---------------------------------------------------------------------------


def test_real_upload_no_create_on_missing_repo_exits_seven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Spec 10 § Errors and recovery: ``--no-create`` on a missing
    repo fails. The orchestrator raises :class:`PublishError` which
    the runner maps to exit 7."""

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_nocreate_" + "f" * 12)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    transport = FakeTransport()  # repo does not exist
    rc = cmd_publish(
        recipe_arg=str(rp),
        repo_flag="alice/x",
        private=False,
        public=False,
        token_flag=None,
        output_override=None,
        dry_run=False,
        no_create=True,
        tag=None,
        message=None,
        transport_factory=_factory_for(transport),
    )

    err = capsys.readouterr().err
    assert rc == 7
    assert "does not exist" in err
    op_names = [name for name, _ in transport.calls]
    assert "create_repo" not in op_names
    assert "upload_folder" not in op_names


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------


def test_real_upload_private_flag_propagates_to_create_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_private_" + "g" * 12)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    transport = FakeTransport()
    rc = cmd_publish(
        recipe_arg=str(rp),
        repo_flag="alice/x",
        private=True,
        public=False,
        token_flag=None,
        output_override=None,
        dry_run=False,
        no_create=False,
        tag=None,
        message=None,
        transport_factory=_factory_for(transport),
    )

    captured = capsys.readouterr()
    assert rc == 0, captured.err
    create_call = next(call for call in transport.calls if call[0] == "create_repo")
    assert create_call[1]["private"] is True
    assert transport.repos["alice/x"].private is True


# ---------------------------------------------------------------------------
# Token never leaks
# ---------------------------------------------------------------------------


def test_real_upload_does_not_leak_token_into_stdout_or_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Audit invariant: the token value must never reach stdout/stderr,
    even on the success path."""

    secret = "hf_top_secret_value_aaaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    transport = FakeTransport()
    rc = cmd_publish(
        recipe_arg=str(rp),
        repo_flag="alice/x",
        private=False,
        public=False,
        token_flag=None,
        output_override=None,
        dry_run=False,
        no_create=False,
        tag=None,
        message=None,
        transport_factory=_factory_for(transport),
    )

    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert secret not in captured.out
    assert secret not in captured.err


# ---------------------------------------------------------------------------
# Round-trip: first publish + idempotent re-publish
# ---------------------------------------------------------------------------


def test_real_upload_round_trip_first_then_no_op(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The realistic CI loop: first invocation creates+uploads, second
    invocation over the unchanged local output is NO_CHANGE. Spec 10
    § Idempotency: 'safe to invoke from CI on every run'."""

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_round_" + "h" * 12)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)

    transport = FakeTransport()
    factory = _factory_for(transport)

    # First call: should create + upload.
    buf1 = io.StringIO()
    with redirect_stdout(buf1):
        rc1 = cmd_publish(
            recipe_arg=str(rp),
            repo_flag="alice/x",
            private=False,
            public=False,
            token_flag=None,
            output_override=None,
            dry_run=False,
            no_create=False,
            tag=None,
            message=None,
            transport_factory=factory,
        )
    assert rc1 == 0
    assert "Created and uploaded" in buf1.getvalue()

    # Second call: should NO_CHANGE.
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = cmd_publish(
            recipe_arg=str(rp),
            repo_flag="alice/x",
            private=False,
            public=False,
            token_flag=None,
            output_override=None,
            dry_run=False,
            no_create=False,
            tag=None,
            message=None,
            transport_factory=factory,
        )
    assert rc2 == 0
    assert "No changes to upload" in buf2.getvalue()

    # Only one upload happened across both invocations.
    assert len(transport.repos["alice/x"].commits) == 1


# ---------------------------------------------------------------------------
# Transport error during upload
# ---------------------------------------------------------------------------


def test_real_upload_transport_error_during_upload_exits_seven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A network-shaped failure during upload (simulated via the fake's
    ``raise_on_upload`` flag) maps to exit 7."""

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_neterr_" + "i" * 12)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    transport = FakeTransport(raise_on_upload=True)
    rc = cmd_publish(
        recipe_arg=str(rp),
        repo_flag="alice/x",
        private=False,
        public=False,
        token_flag=None,
        output_override=None,
        dry_run=False,
        no_create=False,
        tag=None,
        message=None,
        transport_factory=_factory_for(transport),
    )

    err = capsys.readouterr().err
    assert rc == 7
    assert "raise_on_upload" in err


# ---------------------------------------------------------------------------
# --license override (spec 10 § Recipe ``publish:`` block)
# ---------------------------------------------------------------------------


def test_real_upload_license_override_lands_in_uploaded_readme(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--license <LICENSE>`` must end up in the staged README that
    actually gets uploaded. We assert against the bytes the fake
    transport snapshotted from ``upload_folder`` so this test catches
    a regression where the flag is accepted but silently dropped."""

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_license_" + "h" * 12)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    transport = FakeTransport()
    rc = cmd_publish(
        recipe_arg=str(rp),
        repo_flag="alice/x",
        private=False,
        public=False,
        token_flag=None,
        output_override=None,
        dry_run=False,
        no_create=False,
        tag=None,
        message=None,
        license_override="mit",
        transport_factory=_factory_for(transport),
    )

    captured = capsys.readouterr()
    assert rc == 0, captured.err

    # The fake parses the uploaded README's front matter into
    # card_data on every upload — that's what HF does too.
    card = transport.repos["alice/x"].card_data
    assert card.get("license") == "mit"

    # And the raw README bytes carry the same line in the YAML block.
    readme_bytes = transport.repos["alice/x"].files["README.md"]
    assert b"license: mit" in readme_bytes
