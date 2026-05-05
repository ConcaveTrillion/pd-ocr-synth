"""Pydantic v2 models for the recipe YAML schema.

Reference: ``docs/specs/02-recipe-format.md``. Models are frozen so a
loaded recipe is an immutable value through the rest of the pipeline.

Three forms supported wherever the spec calls for a varying value
(scalar, range, or weighted choice) are encoded by the
``Range``/``WeightedChoice`` generics combined with a union — see
``IntRangeOrChoice`` / ``FloatRangeOrChoice`` below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (1,)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Range[T: (int, float)](_Frozen):
    """Inclusive uniform range used by varying scalar fields."""

    min: T
    max: T

    @model_validator(mode="after")
    def _check_order(self) -> Range[T]:
        if self.max < self.min:
            raise ValueError(f"range max ({self.max}) must be >= min ({self.min})")
        return self


class WeightedChoice[T: (int, float)](_Frozen):
    """One option in a discrete weighted choice list."""

    value: T
    weight: float = 1.0

    @field_validator("weight")
    @classmethod
    def _weight_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"weight must be >= 0, got {v}")
        return v


# Convenience aliases. Order in the union matters for pydantic's
# left-to-right matching: scalar first, then mapping (Range), then list
# (weighted choice). Pydantic v2 in default mode tries each alternative
# until one succeeds, which is what we want.
IntRangeOrChoice = int | Range[int] | list[WeightedChoice[int]]
FloatRangeOrChoice = float | Range[float] | list[WeightedChoice[float]]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class OutputBlock(_Frozen):
    format: str
    mode: Literal["recognition", "detection"]
    destination: Path
    count: int = Field(gt=0)
    manifest: str = "manifest.jsonl"


# ---------------------------------------------------------------------------
# Corpus (discriminated union by ``type``)
# ---------------------------------------------------------------------------


class CorpusFilterConfig(_Frozen):
    """Provider-level filter applied per corpus entry.

    Independent of recipe-level ``text_transforms``; this stage cleans
    each provider's *source* data before it joins the pool. See
    ``docs/specs/04-corpus-providers.md``.
    """

    drop_lines_matching: str | None = None
    keep_only_lines_matching: str | None = None
    min_line_chars: int = 0


class _CorpusBase(_Frozen):
    cache: bool = True
    max_chars: int | None = None
    min_word_length: int = 1
    language: str | None = None
    filter: CorpusFilterConfig | None = None


class WebCorpus(_CorpusBase):
    type: Literal["web"]
    url: str
    parser: str | None = None


class LocalCorpus(_CorpusBase):
    type: Literal["local"]
    path: Path


class HFDatasetCorpus(_CorpusBase):
    type: Literal["hf_dataset"]
    name: str
    split: str = "train"
    field: str = "text"


class WikisourceCorpus(_CorpusBase):
    type: Literal["wikisource"]
    language: str
    titles: list[str]


CorpusEntry = Annotated[
    WebCorpus | LocalCorpus | HFDatasetCorpus | WikisourceCorpus,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Text transforms
# ---------------------------------------------------------------------------


class TextTransform(_Frozen):
    """A named text transform with optional keyword-style options.

    The recipe YAML accepts two forms::

        text_transforms:
          - normalize_whitespace             # bare name
          - tironian_et:                     # single-key mapping
              replace_words: ["agus", "and"]
    """

    name: str
    options: dict[str, Any] = Field(default_factory=dict)

    # frozen + dict[str, Any] is fine; the dict is shared but the model
    # itself is immutable (no reassignment of ``options``).
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _coerce_form(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"name": data, "options": {}}
        if isinstance(data, dict):
            if "name" in data:
                return data
            if len(data) == 1:
                ((k, v),) = data.items()
                return {"name": k, "options": v or {}}
        return data


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------


class Font(_Frozen):
    path: Path
    weight: float = 1.0
    license: str | None = None
    source: str | None = None
    optional: bool = False
    features: dict[str, bool] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class ColorSpec(_Frozen):
    r: IntRangeOrChoice
    g: IntRangeOrChoice
    b: IntRangeOrChoice


class Rendering(_Frozen):
    shaping_engine: Literal["harfbuzz", "pillow"] = "harfbuzz"
    font_size_pt: IntRangeOrChoice | FloatRangeOrChoice
    dpi: IntRangeOrChoice
    ink_color: ColorSpec
    background_color: ColorSpec
    antialiasing: bool = True


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class Layout(_Frozen):
    """Layout block.

    Different ``mode`` values legally use different keys. Pydantic only
    accepts known keys (``extra="forbid"``); semantic checks for
    mode-specific consistency live in :mod:`pd_ocr_synth.validation`.
    """

    mode: Literal["word_crops", "lines", "paragraphs", "pages"]
    padding_px: IntRangeOrChoice | None = None
    baseline_jitter_px: IntRangeOrChoice | None = None
    max_width_px: int | None = None
    line_spacing: FloatRangeOrChoice | None = None


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


class DegradationStage(BaseModel):
    """One stage in the degradation pipeline.

    Stage-specific options vary by ``kind`` (blur takes ``sigma``,
    paper_texture takes ``directory``/``blend``/``opacity``, etc.). The
    schema here intentionally allows extras; the validation pass is
    where we enforce known kinds and known keys per kind.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    kind: str
    probability: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------


class HFDatasetPublishConfig(_Frozen):
    repo: str
    private: bool = False
    license: str | None = None
    tags: list[str] = Field(default_factory=list)
    language: list[str] = Field(default_factory=list)
    description_file: Path | None = None


class PublishBlock(_Frozen):
    hf_dataset: HFDatasetPublishConfig | None = None


# ---------------------------------------------------------------------------
# Recipe (top-level)
# ---------------------------------------------------------------------------


class Recipe(_Frozen):
    schema_version: int
    name: str
    description: str | None = None
    seed: int = 0
    output: OutputBlock
    corpus: list[CorpusEntry]
    text_transforms: list[TextTransform] = Field(default_factory=list)
    fonts: list[Font]
    rendering: Rendering
    layout: Layout
    degradation: list[DegradationStage] = Field(default_factory=list)
    publish: PublishBlock | None = None

    # ``source_path`` is set by the loader after model construction via
    # ``model_copy(update=...)``. It is not part of the YAML contract.
    source_path: Path | None = None

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, v: int) -> int:
        if v not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported schema_version {v}; supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        return v

    @field_validator("corpus")
    @classmethod
    def _at_least_one_corpus(cls, v: list[CorpusEntry]) -> list[CorpusEntry]:
        if not v:
            raise ValueError("recipe must declare at least one corpus entry")
        return v

    @field_validator("fonts")
    @classmethod
    def _at_least_one_font(cls, v: list[Font]) -> list[Font]:
        if not v:
            raise ValueError("recipe must declare at least one font")
        return v
