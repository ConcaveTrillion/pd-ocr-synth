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
    # Override the cache key for this entry — spec 04 "Common keys"
    # documents this as an "advanced" knob recipe authors can use to
    # split or merge cache slots across runs without changing the
    # underlying provider options. ``None`` falls through to the
    # provider's ``cache_key(options)`` derivation. Tracked in the
    # iter-N (this commit) per-provider parity audit.
    cache_key: str | None = None
    max_chars: int | None = None
    min_word_length: int = 1
    language: str | None = None
    filter: CorpusFilterConfig | None = None


class WebCorpus(_CorpusBase):
    type: Literal["web"]
    url: str
    parser: str | None = None
    # JSONPath-lite selector applied when ``parser == "json"`` (see spec
    # 04 "Parsers" table). ``None`` returns the whole decoded body. The
    # field is part of the web provider's cache key, so two recipes
    # pulling different sub-trees from the same URL each get their own
    # cache slot.
    field_path: str | None = None
    # HTTP transport options documented in spec 04's ``web`` example.
    # ``user_agent`` overrides the default ``pd-ocr-synth/<version>``
    # UA string set in :mod:`pd_ocr_synth.corpus.http`. ``retries`` and
    # ``timeout_seconds`` plumb the polite-defaults knobs from spec 04
    # ("Polite defaults: 1 req/sec per host, 30s timeout, 3 retries
    # with backoff"). ``respect_robots`` toggles the documented
    # robots.txt honoring (default ``true``). All four were missing
    # from the model pre-iter-N — pydantic's ``extra='forbid'`` was
    # rejecting them at YAML load even though spec 04 advertises them
    # as the canonical surface, so a recipe that copy-pasted the spec
    # example crashed with ValidationError. The runtime side already
    # reads ``retries`` (web.py:_http_get); the other three are
    # forward-looking fields that the model accepts today and the
    # runtime can wire through in follow-up commits without another
    # spec/model drift round.
    user_agent: str | None = None
    retries: int | None = None
    timeout_seconds: float | None = None
    respect_robots: bool | None = None


class LocalCorpus(_CorpusBase):
    type: Literal["local"]
    path: Path
    # Explicit parser override; spec 04 ``local`` block documents that
    # parsers are inferred from extension *or* set explicitly via
    # ``parser:``. The local provider already reads
    # ``options.get("parser")`` (local.py:43), but the model used to
    # reject the key — meaning the documented escape-hatch was
    # unreachable from any recipe.
    parser: str | None = None


class HFDatasetCorpus(_CorpusBase):
    type: Literal["hf_dataset"]
    name: str
    split: str = "train"
    field: str = "text"
    # Per-recipe row cap; spec 04 ``hf_dataset`` block advertises this
    # as the streaming-truncate knob. ``None`` means "stream all
    # available rows", matching the spec's "unlimited" default.
    max_rows: int | None = None


class WikisourceCorpus(_CorpusBase):
    type: Literal["wikisource"]
    language: str
    # Spec 04 ``wikisource`` block documents two YAML examples: one
    # with ``titles:``, one with ``category:``. The pre-iter-N model
    # required ``titles`` even when ``category:`` was the supplied
    # selector, which contradicted the spec. Default to an empty list
    # so the category-only example loads cleanly; the wikisource
    # provider already raises ``ProviderError`` if neither is set
    # (wikisource.py:_fetch_title pre-check).
    titles: list[str] = Field(default_factory=list)
    # Category-based selector + page cap. The provider reads
    # ``options.get("category")`` today (and raises a deferred-feature
    # ``ProviderError``); accepting the field on the model is the
    # forward-looking move so a recipe author following the spec gets
    # the deferred-feature error from the runtime rather than a
    # confusing ``extra_forbidden`` ValidationError at load.
    category: str | None = None
    max_pages: int | None = None

    @model_validator(mode="after")
    def _titles_or_category(self) -> WikisourceCorpus:
        # Spec 04 ``wikisource`` block presents ``titles:`` and
        # ``category:`` as alternative selectors; one of them must be
        # set or there's nothing to fetch. Enforce at load so the
        # error message points at the recipe, not at a runtime
        # ``ProviderError`` deeper in the pipeline.
        if not self.titles and not self.category:
            raise ValueError(
                "wikisource corpus entry must declare at least one of "
                "'titles' or 'category' (spec 04 §wikisource)"
            )
        return self


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
    # Vertical gap *between* paragraphs on a page, expressed as a
    # multiplier of the rendered line height (same units as
    # ``line_spacing``). Only meaningful for ``mode='pages'`` — a
    # ``paragraphs``-mode sample is a single paragraph by construction.
    # See docs/specs/06-rendering.md §pages and the M09 roadmap.
    paragraph_spacing: FloatRangeOrChoice | None = None
    # First-line indent (in pixels) applied to the first line of every
    # paragraph on a page. Only meaningful for ``mode='pages'`` —
    # ``paragraphs`` mode is a single paragraph and an indent there
    # would just shift the whole sample's first line, which is what
    # padding is for. Spec 06 § paragraphs advertises an em-based
    # ``paragraph_indent_em``; we expose px directly to keep the layout
    # field deterministic (no font-size dependence). Default ``None``
    # preserves the existing un-indented output bit-for-bit. See
    # docs/roadmap/09-detection-mode.md § First-line indent.
    paragraph_indent_px: int | None = None
    # Horizontal alignment of each line within the paragraph's bounding
    # box. ``None`` (default) and ``"left"`` both preserve the existing
    # left-aligned output bit-for-bit. ``"center"`` centers each line
    # within ``paragraph_width`` (the width of the longest line, plus
    # any first-line indent contribution); ``"right"`` flushes each
    # line to the right edge of ``paragraph_width``. ``"justify"``
    # distributes the per-line slack (``paragraph_width -
    # line_natural_width``) across the inter-word gaps in that line so
    # the line's right edge sits flush with ``paragraph_width``. Per
    # standard book-typesetting practice, the **last line** of a
    # paragraph and **single-word** lines fall back to left alignment
    # (a justified single-word line would be a stretched glyph run, not
    # a justified line; a justified last line typically looks
    # awkwardly stretched). Only meaningful for ``mode in {paragraphs,
    # pages}`` — recognition-mode samples are tight crops where
    # alignment is undefined. See docs/roadmap/09-detection-mode.md §
    # Paragraph alignment.
    paragraph_alignment: Literal["left", "center", "right", "justify"] | None = None
    # Explicit fixed page canvas size in pixels (width, height). When
    # set, ``render_page`` produces output of *exactly* this size by
    # rendering content at its natural extent and then padding the
    # remaining canvas with the sampled background colour. Content is
    # placed top-left; all bbox annotations remain inside the natural-
    # content rectangle. If natural content exceeds ``page_size_px`` in
    # either dimension, a :class:`RenderError` is raised — silent
    # truncation would corrupt the per-word/per-line annotations the
    # detection trainer consumes. ``None`` (default) preserves the
    # historical auto-sized canvas. Only meaningful for ``mode='pages'``;
    # ``paragraphs`` mode is by definition a tight single-paragraph crop
    # with no notion of a "page". See docs/specs/06-rendering.md
    # §pages and docs/roadmap/09-detection-mode.md § Explicit page_size_px.
    page_size_px: tuple[int, int] | None = None

    @field_validator("page_size_px")
    @classmethod
    def _check_page_size_px_positive(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        if v is None:
            return v
        w, h = v
        if w <= 0 or h <= 0:
            raise ValueError(f"page_size_px must be positive (width, height); got ({w}, {h})")
        return v


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
    # Named groups of degradation stages that can be expanded inline
    # via a ``- preset: <name>`` entry in ``degradation``. The loader
    # performs the expansion at load time, so by the time runtime code
    # walks ``degradation`` there are no ``preset`` entries left. This
    # block is preserved on the model purely for round-tripping /
    # introspection (``pd-ocr-synth describe``).
    degradation_presets: dict[str, list[DegradationStage]] = Field(default_factory=dict)
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
