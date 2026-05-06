"""Cross-project contract tests against ``pd-ocr-trainer`` readers.

This is the M09 residual: locks the contract that ``pd-ocr-synth``'s
recognition + detection writer outputs are loadable by the readers
``pd-ocr-trainer`` actually drives — `doctr.datasets.RecognitionDataset`
and `doctr.datasets.DetectionDataset` (the trainer wires both directly,
see ``pd_ocr_trainer/train_recog.py`` and ``train_detect.py``).

Two layers
----------

1. **Always-on shape contract tests.** ``doctr`` isn't a runtime dep
   of pd-ocr-synth (and shouldn't be — synth produces, doesn't read).
   So we re-implement the *exact* shape checks the doctr readers do,
   as plain ``json.load + assertions``, against tiny synthetic
   outputs we build in-process. These run under ``make ci`` on every
   commit and catch ``labels.json`` schema drift the moment it
   happens. The shape predicates are inlined from
   ``doctr/datasets/recognition.py`` and ``doctr/datasets/detection.py``
   v1.x as of 2026-05; if doctr changes its reader contract, this
   file is the first place to update.

2. **Opt-in live integration tests.** Gated on
   ``PD_OCR_SYNTH_TRAINER_E2E=1``. These import
   ``doctr.datasets.RecognitionDataset`` / ``DetectionDataset``
   directly and feed them the synthetic output. Skipped under default
   ``make ci`` (doctr isn't installed in synth's env), and on
   developer machines where the trainer's venv is reachable they
   provide the strongest possible drift signal: the actual
   trainer-side reader instantiates without error and yields the
   expected sample count.

Env-var contract
~~~~~~~~~~~~~~~~

``PD_OCR_SYNTH_TRAINER_E2E``
    Master switch. Truthy enables the integration tests. The test
    *also* requires ``doctr`` to be importable; we treat
    ``ModuleNotFoundError`` as a skip rather than a failure so a
    misconfigured env doesn't look like the test broke.

Convention follows ``tests/integration/test_publish_live_hf.py``
(M08): ``tests/integration/`` for opt-in tests, ``@pytest.mark.integration``,
``pytest.mark.skipif`` gating, always-on sanity tests for the gating
helpers.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from pd_ocr_synth.output import DetectionWriter, page_filename
from pd_ocr_synth.output.detection import LABELS_FILENAME
from pd_ocr_synth.recipe import load_recipe

# ---------------------------------------------------------------------------
# Gating helpers (mirrors test_publish_live_hf.py)
# ---------------------------------------------------------------------------


_TRUTHY = frozenset({"1", "true", "True", "TRUE", "yes", "on"})


def _e2e_enabled() -> bool:
    """True iff the master switch is set AND ``doctr`` is importable.

    We require *both* because either alone is a misconfiguration: an
    unset switch means "skip" by design; a set switch in an env
    without ``doctr`` would fail at import and look like a test break
    rather than a missing optional dep.
    """

    if os.environ.get("PD_OCR_SYNTH_TRAINER_E2E", "") not in _TRUTHY:
        return False
    return importlib.util.find_spec("doctr") is not None


# ---------------------------------------------------------------------------
# Fixtures: tiny recognition + detection outputs
# ---------------------------------------------------------------------------


def _write_recognition_output(local: Path) -> tuple[list[str], dict[str, str]]:
    """Materialize a 2-sample recognition layout.

    Returns ``(image_filenames, labels_dict)`` so callers can assert on
    the same data the readers will see. Mirrors the helper in
    ``tests/test_publish_orchestrator.py`` / ``test_publish_live_hf.py``
    — kept inline here so the integration test stays self-contained
    and the recipe pipeline doesn't gate the contract test.
    """

    images = local / "images"
    images.mkdir(parents=True, exist_ok=True)

    labels: dict[str, str] = {}
    image_names: list[str] = []
    for idx, text in enumerate(["Séadna", "agus"]):
        name = f"{idx:07d}.png"
        Image.new("RGB", (32, 32), color=(200, 200, 200)).save(images / name, format="PNG")
        labels[name] = text
        image_names.append(name)

    (local / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return image_names, labels


_DETECTION_RECIPE_TEMPLATE = """\
schema_version: 1
name: trainer-contract-detection
seed: 7
output:
  format: pd-ocr-trainer/v1
  mode: detection
  destination: ./out
  count: 2
corpus:
  - type: local
    path: ./seed-words.txt
fonts:
  - path: {font}
    weight: 1.0
rendering:
  font_size_pt: 14
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 240, b: 240 }}
layout:
  mode: paragraphs
  padding_px: 4
"""


def _build_detection_recipe(tmp_path: Path) -> Path:
    font = tmp_path / "fake.otf"
    font.write_bytes(b"\x00\x01\x02fake")
    seed = tmp_path / "seed-words.txt"
    seed.write_text("alpha beta gamma\n", encoding="utf-8")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_DETECTION_RECIPE_TEMPLATE.format(font=font), encoding="utf-8")
    return rp


def _line(text: str, bbox: tuple[int, int, int, int]) -> SimpleNamespace:
    return SimpleNamespace(text=text, bbox=bbox)


def _word(text: str, bbox: tuple[int, int, int, int]) -> SimpleNamespace:
    return SimpleNamespace(text=text, bbox=bbox)


def _fake_page_sample(
    *,
    line_boxes: tuple[Any, ...],
    word_boxes: tuple[Any, ...],
    size: tuple[int, int] = (200, 100),
) -> SimpleNamespace:
    img = Image.new("RGB", size, color=(240, 240, 240))
    return SimpleNamespace(
        text="\n".join(getattr(lb, "text", "") for lb in line_boxes),
        image=img,
        font_path=Path("/fake/font.otf"),
        font_size_pt=14.0,
        dpi=300,
        ink_color=(10, 10, 10),
        background_color=(240, 240, 240),
        size=size,
        bbox=(0, 0, size[0], size[1]),
        glyph_runs=(),
        line_boxes=line_boxes,
        word_boxes=word_boxes,
    )


def _write_detection_output(tmp_path: Path) -> tuple[Path, list[str]]:
    """Drive ``DetectionWriter`` on two synthetic pages and return the
    output dir + ordered list of page filenames written."""

    rp = _build_detection_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "det_out"

    pages: list[str] = []
    with DetectionWriter.open(recipe, out, seed=recipe.seed) as writer:
        for idx, (line_text, word_specs) in enumerate(
            [
                ("alpha beta", [("alpha", (10, 12, 50, 28)), ("beta", (60, 12, 110, 28))]),
                ("gamma", [("gamma", (10, 12, 70, 28))]),
            ]
        ):
            line_box = _line(line_text, (10, 10, max(w[1][2] for w in word_specs), 30))
            wbs = tuple(_word(t, b) for t, b in word_specs)
            sample = _fake_page_sample(
                line_boxes=(line_box,),
                word_boxes=wbs,
                size=(200, 100),
            )
            writer.write_rendered(idx, sample)
            pages.append(page_filename(idx, width=7))

    return out, pages


# ---------------------------------------------------------------------------
# Always-on shape contract tests (no doctr import — locks drift any time)
# ---------------------------------------------------------------------------


def test_recognition_labels_match_doctr_recognitiondataset_shape(tmp_path: Path) -> None:
    """``labels.json`` is a dict of ``{img_name: text}`` and every
    image referenced exists on disk under ``images/``.

    This is the exact shape ``doctr.datasets.RecognitionDataset``
    asserts at construction time — see
    ``site-packages/doctr/datasets/recognition.py``::

        for img_name, label in labels.items():
            if not os.path.exists(os.path.join(self.root, img_name)):
                raise FileNotFoundError(...)

    We hold the line at the synth side so a regression in our writer
    surfaces here, not at trainer-launch time.
    """

    local = tmp_path / "rec"
    image_names, labels = _write_recognition_output(local)

    on_disk = json.loads((local / "labels.json").read_text(encoding="utf-8"))
    assert isinstance(on_disk, dict), "doctr expects labels.json to be a JSON object"
    assert set(on_disk.keys()) == set(image_names), (
        "every label key must correspond to an image we wrote; orphan "
        "keys would FileNotFoundError in RecognitionDataset.__init__"
    )
    for name, text in on_disk.items():
        assert isinstance(name, str) and name, "label keys must be non-empty filenames"
        assert isinstance(text, str), (
            f"RecognitionDataset stores label as ``str`` per its data tuple; got {type(text)!r}"
        )
        assert (local / "images" / name).exists(), (
            f"image {name!r} referenced in labels.json must exist in images/ "
            f"(doctr resolves via os.path.join(root, img_name))"
        )

    assert on_disk == labels, "in-memory labels and on-disk labels must match"


def test_detection_labels_match_doctr_detectiondataset_shape(tmp_path: Path) -> None:
    """``labels.json`` is a dict of ``{img_name: {"polygons": [...], ...}}``
    where ``polygons`` is a list of 4-corner polygons (or a dict of
    class → polygon-list), each polygon is shape (4, 2), and every
    referenced image exists under ``images/``.

    This is the exact shape ``doctr.datasets.DetectionDataset``
    asserts — see ``site-packages/doctr/datasets/detection.py``
    ``__init__`` + ``format_polygons``. The reader does
    ``np.asarray(polygons, dtype=np.float32)`` then either treats
    the result as ``(N, 4, 2)`` rotated boxes or
    ``concatenate((min(axis=1), max(axis=1)))`` to extract straight
    bboxes. Either path requires uniform 4×2 polygon shape.
    """

    out, pages = _write_detection_output(tmp_path)

    on_disk = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(on_disk, dict), "doctr expects labels.json to be a JSON object"
    assert set(on_disk.keys()) == set(pages), (
        "every page filename in labels.json must be a real image we wrote"
    )

    for name, entry in on_disk.items():
        assert isinstance(entry, dict), (
            f"detection entry for {name!r} must be a dict (got {type(entry)!r}); "
            "doctr accesses ``label['polygons']`` directly"
        )
        assert "polygons" in entry, (
            f"entry for {name!r} missing required 'polygons' key — "
            "DetectionDataset.__init__ does ``label['polygons']``"
        )

        polygons = entry["polygons"]
        # doctr accepts both list (single class) and dict (multiclass)
        # — see format_polygons. We write the list shape today.
        if isinstance(polygons, list):
            polygon_iter = polygons
        elif isinstance(polygons, dict):
            polygon_iter = [p for v in polygons.values() for p in v]
        else:
            raise AssertionError(
                f"polygons must be list or dict (got {type(polygons)!r}); "
                "doctr.format_polygons raises TypeError otherwise"
            )

        for poly in polygon_iter:
            assert isinstance(poly, list) and len(poly) == 4, (
                f"each polygon must have exactly 4 corners; got {poly!r}. "
                "doctr stacks them as np.float32 array of shape (N, 4, 2)"
            )
            for corner in poly:
                assert isinstance(corner, list) and len(corner) == 2, (
                    f"each corner must be [x, y]; got {corner!r}"
                )
                for coord in corner:
                    assert isinstance(coord, (int, float)), (
                        f"polygon coords must be numeric (int|float) for "
                        f"np.float32 cast; got {coord!r}"
                    )

        assert (out / "images" / name).exists(), (
            f"image {name!r} referenced in labels.json must exist in images/"
        )


def test_detection_polygon_array_is_float32_castable(tmp_path: Path) -> None:
    """Belt-and-braces: feed the polygons through the same
    ``np.asarray(polygons, dtype=np.float32)`` path doctr's
    ``format_polygons`` does. If the writer ever emits ragged polygons
    (mixed corner counts) or non-numeric coords, this fails the way
    the trainer would — minus the dataset class machinery."""

    np = pytest.importorskip("numpy")  # synth runtime dep, present in CI
    out, _pages = _write_detection_output(tmp_path)
    on_disk = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))

    for name, entry in on_disk.items():
        polygons = entry["polygons"]
        # Cast list-form (the form we write) — dict-form would be
        # iterated per-class first, but we don't emit that today.
        assert isinstance(polygons, list), (
            f"writer is currently expected to emit list-form polygons; "
            f"entry {name!r} had {type(polygons)!r}"
        )
        if not polygons:
            # Empty pages are legal (a page with no detected lines);
            # np.asarray([]) succeeds. Skip the shape assertion.
            continue
        arr = np.asarray(polygons, dtype=np.float32)
        assert arr.shape[1:] == (4, 2), (
            f"polygons must stack to (N, 4, 2); got {arr.shape!r} for {name!r}. "
            "doctr.format_polygons concatenates min/max on axis=1."
        )


# ---------------------------------------------------------------------------
# Opt-in live integration tests (require doctr installed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not _e2e_enabled(),
    reason=(
        "trainer cross-project E2E is opt-in: set PD_OCR_SYNTH_TRAINER_E2E=1 "
        "AND have ``doctr`` importable in the test env to run"
    ),
)
def test_recognition_dataset_loads_synth_output(tmp_path: Path) -> None:
    """Drive ``doctr.datasets.RecognitionDataset`` on synth output.

    Locks the strongest possible cross-project contract: the actual
    reader pd-ocr-trainer constructs (``train_recog.py:483``)
    instantiates without error and yields the right sample count.
    """

    from doctr.datasets import RecognitionDataset

    local = tmp_path / "rec"
    image_names, labels = _write_recognition_output(local)

    ds = RecognitionDataset(
        img_folder=str(local / "images"),
        labels_path=str(local / "labels.json"),
    )

    assert len(ds) == len(labels), (
        f"RecognitionDataset should report {len(labels)} samples; got {len(ds)}"
    )
    # ds.data is list[tuple[str, str]] of (img_name, label).
    seen = dict(ds.data)
    assert seen == labels, (
        f"RecognitionDataset.data should round-trip labels.json; got {seen!r} vs {labels!r}"
    )
    # Every image we wrote should be addressable by the trainer.
    assert set(seen.keys()) == set(image_names)


@pytest.mark.integration
@pytest.mark.skipif(
    not _e2e_enabled(),
    reason=(
        "trainer cross-project E2E is opt-in: set PD_OCR_SYNTH_TRAINER_E2E=1 "
        "AND have ``doctr`` importable in the test env to run"
    ),
)
def test_detection_dataset_loads_synth_output(tmp_path: Path) -> None:
    """Drive ``doctr.datasets.DetectionDataset`` on synth output.

    Locks the contract for the trainer's detection reader
    (``train_detect.py:455``). Asserts both straight-bbox mode
    (``use_polygons=False``, the default) and rotated mode
    (``use_polygons=True``) — the trainer flips this based on
    ``--rotation`` so synth output must be valid for either.
    """

    from doctr.datasets import DetectionDataset

    out, pages = _write_detection_output(tmp_path)

    # Straight bboxes — what train_detect.py:455 does by default.
    ds_straight = DetectionDataset(
        img_folder=str(out / "images"),
        label_path=str(out / "labels.json"),
        use_polygons=False,
    )
    assert len(ds_straight) == len(pages)

    # Rotated polygons — the --rotation path.
    ds_rotated = DetectionDataset(
        img_folder=str(out / "images"),
        label_path=str(out / "labels.json"),
        use_polygons=True,
    )
    assert len(ds_rotated) == len(pages)

    # Class names default to the single 'words' class doctr uses for
    # list-form polygons. The trainer reads this for model head
    # construction (train_detect.py:356).
    assert ds_straight.class_names == ds_rotated.class_names
    assert len(ds_straight.class_names) >= 1


# ---------------------------------------------------------------------------
# Collection sanity (always runs, exercises the gating helpers)
# ---------------------------------------------------------------------------


def test_e2e_disabled_without_env() -> None:
    """The gating helper returns False under default ``make ci``.

    Save & restore the env so a developer running with the variable
    set locally still gets their live run after this test.
    """

    saved = os.environ.get("PD_OCR_SYNTH_TRAINER_E2E")
    try:
        os.environ.pop("PD_OCR_SYNTH_TRAINER_E2E", None)
        assert _e2e_enabled() is False, (
            "with the master switch unset, the integration test must skip"
        )

        # Switch on but require doctr to actually be importable. If
        # doctr happens to be installed in the dev's pd-ocr-synth env
        # (rare; not a runtime dep), the helper returns True. We don't
        # assert a specific value here because both branches are valid;
        # we only assert the truthy-set codepath is reached.
        os.environ["PD_OCR_SYNTH_TRAINER_E2E"] = "1"
        result = _e2e_enabled()
        expected = importlib.util.find_spec("doctr") is not None
        assert result is expected, (
            f"with switch on, helper should mirror doctr availability; "
            f"got {result!r} (doctr present: {expected})"
        )

        # Falsy values stay disabled regardless of doctr.
        for falsy in ("", "0", "false", "no"):
            os.environ["PD_OCR_SYNTH_TRAINER_E2E"] = falsy
            assert _e2e_enabled() is False, f"value {falsy!r} should disable"
    finally:
        if saved is None:
            os.environ.pop("PD_OCR_SYNTH_TRAINER_E2E", None)
        else:
            os.environ["PD_OCR_SYNTH_TRAINER_E2E"] = saved


def test_truthy_set_matches_publish_live_helper() -> None:
    """The truthy set is identical to the one in
    ``test_publish_live_hf.py``. Drift between the two would be
    confusing — a developer who sets ``PD_OCR_SYNTH_*=yes`` for one
    suite would expect the other to honor it too."""

    from tests.integration.test_publish_live_hf import _TRUTHY as publish_truthy

    assert _TRUTHY == publish_truthy, (
        "integration tests should agree on what counts as a truthy "
        "env-var value; update both helpers together"
    )
