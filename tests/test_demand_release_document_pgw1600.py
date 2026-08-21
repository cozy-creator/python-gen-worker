"""pgw#1600 acceptance (b): the release document carries the formula and a
worst case a READER FINDS rather than a number a human wrote.

This goes through the REAL `release derive` CLI on a REAL declaration — the
same path that produces the document tensorhub decodes. What is asserted is
the whole chain: an author's `lane(request=...)` -> the serialized term list
-> the advertised envelope derived from the payload's own bucket table -> the
derived worst case.

The subject is sdxl's declaration SHAPE (`demand_sdxl_shaped_endpoint`),
because sdxl's payload is the hard case: it dimensions itself through an
aspect-ratio ENUM and carries no `width` field at all. The tree traced against
is the tiny SD15-class one — this repository ships no sdxl checkpoint and a
mint on a dev box is banned — which is a fixture detail, not a gap in what is
proven: the declaration -> document path reads no weight.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
import gen_worker._vendor.torchcg  # noqa: E402,F401

from gen_worker.demand import MiB, RequestShape, evaluate  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"

LOCK = (
    'version = 1\n'
    '\n[[package]]\nname = "torch"\nversion = "2.13.0"\n'
    '\n[[package]]\nname = "triton"\nversion = "3.7.1"\n'
    '\n[[package]]\nname = "nvidia-cublas"\nversion = "13.1.1.3"\n'
    '\n[[package]]\nname = "diffusers"\nversion = "0.39.0"\n'
)


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))
    return tiny_tree.save_config_only(tmp_path_factory.mktemp("demand-tree"))


@pytest.fixture(scope="module")
def document(tree: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    from gen_worker.cli import main

    out = tmp_path_factory.mktemp("demand-derive") / "release.json"
    lockfile = out.parent / "uv.lock"
    lockfile.write_text(LOCK)
    rc = main([
        "release", "derive",
        "--dir", str(FIXTURES),
        "--module", "demand_sdxl_shaped_endpoint",
        "--checkpoint", str(tree),
        "--lockfile", str(lockfile),
        "--out", str(out),
    ])
    assert rc == 0
    return json.loads(out.read_text())


def _demand(document: dict) -> dict:
    (entry,) = document["lane_contracts"].values()
    assert "demand" in entry, (
        "the release document carries no demand block — a reader has nothing "
        "to size a pod from but a hand-written number"
    )
    return entry["demand"]


def test_the_lane_contract_carries_the_serialized_FORMULA(document: dict) -> None:
    demand = _demand(document)
    rows = {row["term"]: row for row in demand["terms"]}
    assert set(rows) == {"const", "mp_batch"}
    assert rows["const"]["bytes"] == MiB(1155)
    assert rows["mp_batch"]["bytes"] == MiB(220)
    assert demand["algebra"].startswith("C + sum(")
    assert demand["arena"] == "request"


def test_every_coefficient_carries_its_PROVENANCE(document: dict) -> None:
    """No magic constants: a number in a published document says where it
    came from, and the formula as a whole claims only what its weakest term
    supports."""

    rows = {row["term"]: row for row in _demand(document)["terms"]}
    assert rows["const"]["basis"] == "measured"
    assert "tcg#80" in rows["const"]["source"]
    assert "n=1" in rows["const"]["source"]
    assert rows["mp_batch"]["basis"] == "uncalibrated"
    assert _demand(document)["basis"] == "uncalibrated"


def test_the_envelope_comes_from_the_payloads_OWN_bucket_table(
    document: dict,
) -> None:
    """The platform cannot dimension an aspect-ratio enum, and does not guess.

    `Shape(pixels=_BUCKETS)` hands it the endpoint's own dict, so width and
    height are DECLARED — with the source recorded — rather than defaulted.
    """

    envelope = _demand(document)["envelope"]
    assert envelope["width"]["source"] == "declared"
    assert envelope["height"]["source"] == "declared"
    assert envelope["width"]["value"] == 1536
    assert envelope["height"]["value"] == 1536


def test_the_batch_axis_is_a_STATED_default_that_names_its_own_falsifier(
    document: dict,
) -> None:
    """sdxl's CFG pair is a decision the checkpoint's config makes, so no
    payload field can advertise it. The document says so instead of
    pretending, and points at the instrument that will catch it."""

    batch = _demand(document)["envelope"]["batch"]
    assert batch["source"] == "default"
    assert batch["value"] == 1
    assert "demand_miss" in batch["why"]


def test_the_worst_case_is_DERIVED_and_reproduces_from_the_document_alone(
    document: dict,
) -> None:
    """THE PROPERTY THE HUB DEPENDS ON. A reader with only these bytes — no
    Python, no import — re-derives the same integer from the term list and
    the worst-case shape."""

    demand = _demand(document)
    shape = RequestShape(**demand["worst_case_shape"])
    assert evaluate(demand["terms"], shape) == demand["worst_case_request_bytes"]
    # 1024x1024 is the largest-area bucket; 1536x1536 is not a bucket at all.
    assert (shape.width, shape.height) == (1024, 1024)
    assert demand["worst_case_request_bytes"] == MiB(1155) + MiB(220) * 1024 * 1024 // 1_000_000


def test_the_document_says_what_the_number_is_NOT(document: dict) -> None:
    """The weight arena is never declared (pgw#1598 §2). A reader that adds
    manifest bytes gets the pod-buy worst case; a reader that treats this as
    the whole answer buys a card too small, so the block states its scope."""

    demand = _demand(document)
    assert "manifest weight bytes" in demand["note"]
    assert demand["note"].startswith("REQUEST ARENA ONLY")


def test_the_document_carries_the_VOCABULARY_a_go_reader_validates_against(
    document: dict,
) -> None:
    demand = _demand(document)
    assert demand["vocabulary"]["mp_batch"]["scale"] == 1_000_000
    assert demand["vocabulary"]["mp_batch"]["inputs"] == ["width", "height", "batch"]
    assert demand["vocabulary_version"] >= 1
    assert demand["shape_bounds"]["width"] == 1 << 16
