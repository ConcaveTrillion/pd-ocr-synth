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

from pd_ocr_synth.publish.dataset_card import (
    README_FILENAME,
    DatasetCardInputs,
    load_card_inputs,
    render_dataset_card,
    write_dataset_card,
)
from pd_ocr_synth.publish.recognition import (
    DATA_DIRNAME,
    METADATA_FILENAME,
    StagingResult,
    build_recognition_staging,
)

__all__ = [
    "DATA_DIRNAME",
    "DatasetCardInputs",
    "METADATA_FILENAME",
    "README_FILENAME",
    "StagingResult",
    "build_recognition_staging",
    "load_card_inputs",
    "render_dataset_card",
    "write_dataset_card",
]
