"""Publish — turn local render output into shippable HF dataset payloads.

M08 builds out the Hugging Face publish path. The first piece, landed
here, is the **staging-dir builder**: a pure-Python file transformer
that reads a recognition-mode local layout (per
``docs/specs/08-output-format.md``) and emits the HF imagefolder
layout (per ``docs/specs/10-publishing.md``) into a separate directory
ready for upload.

Network / SDK calls (``huggingface_hub``, auth resolution, idempotency
via ``card_data``) land in later chunks. Splitting the staging step
out keeps the surface testable end-to-end without an HF token, and
keeps the upload-time decisions (private, tag, message) cleanly
separated from the format-conversion decisions.
"""

from __future__ import annotations

from pd_ocr_synth.publish.auth import (
    HF_TOKEN_ENV_VAR,
    AuthError,
    ResolvedToken,
    format_resolution_chain,
    resolve_hf_token,
)
from pd_ocr_synth.publish.commit_message import (
    default_commit_message,
    resolve_commit_message,
)
from pd_ocr_synth.publish.content_sha import (
    CONTENT_SHA_ALGORITHM,
    CONTENT_SHA_KEY,
    ContentShaError,
    apply_content_sha_to_readme,
    compute_content_sha,
)
from pd_ocr_synth.publish.dataset_card import (
    README_FILENAME,
    DatasetCardInputs,
    load_card_inputs,
    render_dataset_card,
    write_dataset_card,
)
from pd_ocr_synth.publish.detection import (
    build_detection_staging,
)
from pd_ocr_synth.publish.idempotency import (
    IdempotencyDecision,
    IdempotencyState,
    check_idempotency,
)
from pd_ocr_synth.publish.orchestrator import (
    PublishError,
    PublishResult,
    PublishState,
    publish_recognition,
)
from pd_ocr_synth.publish.preflight import (
    REQUIRED_FRONT_MATTER_KEYS,
    PreflightError,
    PreflightReport,
    assert_staging_publish_ready,
    check_required_front_matter,
)
from pd_ocr_synth.publish.recognition import (
    DATA_DIRNAME,
    METADATA_FILENAME,
    StagingError,
    StagingResult,
    build_recognition_staging,
)
from pd_ocr_synth.publish.redaction import (
    REDACTED_SENTINEL,
    redact_token,
)
from pd_ocr_synth.publish.sdk_transport import (
    SdkUnavailableError,
    make_default_transport,
)

# ``HfHubTransport`` is the only file that imports ``huggingface_hub``
# (an optional extra). We re-export the symbol here as a *lazy*
# attribute so importing :mod:`pd_ocr_synth.publish` itself never
# triggers the SDK import — users who only render locally don't pay
# the cost. ``__getattr__`` (PEP 562) provides the deferred import
# without leaking the SDK dependency at module-load time.
from pd_ocr_synth.publish.summary import (
    ManifestSummary,
    SummaryError,
    format_summary,
    summarize_metadata,
)
from pd_ocr_synth.publish.transport import (
    CommitInfo,
    FakeTransport,
    HfTransport,
    TransportError,
)


def __getattr__(name: str):
    """PEP 562 lazy-import for the SDK-backed transport.

    Importing :mod:`pd_ocr_synth.publish.hf_hub_transport` pulls in
    ``huggingface_hub``; we route its only public symbol through this
    hook so the SDK is loaded the first time *someone asks for it*
    rather than at package import. Render-only / dry-run code paths
    never touch ``HfHubTransport`` and therefore never require the
    optional ``[publish]`` extra.

    Raises
    ------
    AttributeError
        For any name other than ``HfHubTransport`` (the standard
        contract for ``__getattr__``).
    ImportError
        Bubbles up unchanged when the SDK isn't installed; callers
        that go through :func:`make_default_transport` get a typed
        :class:`SdkUnavailableError` instead, which is the
        recommended entry point.
    """

    if name == "HfHubTransport":
        from pd_ocr_synth.publish.hf_hub_transport import HfHubTransport

        return HfHubTransport
    raise AttributeError(f"module 'pd_ocr_synth.publish' has no attribute {name!r}")


__all__ = [
    "CONTENT_SHA_ALGORITHM",
    "CONTENT_SHA_KEY",
    "DATA_DIRNAME",
    "HF_TOKEN_ENV_VAR",
    "METADATA_FILENAME",
    "README_FILENAME",
    "REDACTED_SENTINEL",
    "REQUIRED_FRONT_MATTER_KEYS",
    "AuthError",
    "CommitInfo",
    "ContentShaError",
    "DatasetCardInputs",
    "FakeTransport",
    "HfHubTransport",
    "HfTransport",
    "IdempotencyDecision",
    "IdempotencyState",
    "ManifestSummary",
    "PreflightError",
    "PreflightReport",
    "PublishError",
    "PublishResult",
    "PublishState",
    "ResolvedToken",
    "SdkUnavailableError",
    "StagingError",
    "StagingResult",
    "SummaryError",
    "TransportError",
    "apply_content_sha_to_readme",
    "assert_staging_publish_ready",
    "build_detection_staging",
    "build_recognition_staging",
    "check_idempotency",
    "check_required_front_matter",
    "compute_content_sha",
    "default_commit_message",
    "format_resolution_chain",
    "format_summary",
    "load_card_inputs",
    "make_default_transport",
    "publish_recognition",
    "redact_token",
    "render_dataset_card",
    "resolve_commit_message",
    "resolve_hf_token",
    "summarize_metadata",
    "write_dataset_card",
]
