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
    StagingResult,
    build_recognition_staging,
)
from pd_ocr_synth.publish.summary import (
    ManifestSummary,
    SummaryError,
    format_summary,
    summarize_metadata,
)

__all__ = [
    "CONTENT_SHA_ALGORITHM",
    "CONTENT_SHA_KEY",
    "ContentShaError",
    "DATA_DIRNAME",
    "DatasetCardInputs",
    "HF_TOKEN_ENV_VAR",
    "METADATA_FILENAME",
    "README_FILENAME",
    "REQUIRED_FRONT_MATTER_KEYS",
    "AuthError",
    "ManifestSummary",
    "PreflightError",
    "PreflightReport",
    "ResolvedToken",
    "StagingResult",
    "SummaryError",
    "apply_content_sha_to_readme",
    "assert_staging_publish_ready",
    "build_recognition_staging",
    "check_required_front_matter",
    "compute_content_sha",
    "format_resolution_chain",
    "format_summary",
    "load_card_inputs",
    "render_dataset_card",
    "resolve_hf_token",
    "summarize_metadata",
    "write_dataset_card",
]
