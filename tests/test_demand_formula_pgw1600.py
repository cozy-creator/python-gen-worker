"""pgw#1600: the demand formula is DATA — algebra, envelope, serialization.

Acceptance (a) is the CROSS-LANGUAGE half and it has two legs: this file
proves the corpus is exactly what this repository's evaluator computes (so it
cannot be a table of numbers a human typed), and tensorhub's Go evaluator runs
the same corpus on its side. A corpus with hand-written expectations would
prove that a human and Go agree, which is not the property anybody wants.

Nothing here imports torch. Acceptance (b)'s end-to-end derive lives in
`test_demand_release_document_pgw1600.py`, which does.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import msgspec
import pytest

from gen_worker.contracts import read_contract
from gen_worker.demand import (
    Basis,
    DemandDeclarationError,
    DemandEvaluationError,
    GiB,
    MAX_RESULT_BYTES,
    MiB,
    RequestShape,
    SHAPE_BOUNDS,
    Shape,
    ShapeDeclarationError,
    TERM_VOCABULARY,
    VOCABULARY_VERSION,
    const,
    evaluate,
    per_batch,
    per_frame,
    per_frame_squared,
    per_latent_token,
    per_mp,
    per_mp_batch,
    vocabulary_document,
)
from gen_worker.demand_envelope import (
    advertised_envelope,
    demand_document,
    request_shape,
)

ROOT = Path(__file__).resolve().parents[1]


def _corpus() -> dict[str, Any]:
    return json.loads(read_contract("demand_vectors.json"))


# --------------------------------------------------------------------------
# ACCEPTANCE (a), python half: the corpus IS the evaluator's own answer
# --------------------------------------------------------------------------


def test_every_corpus_case_reproduces_through_the_real_evaluator() -> None:
    corpus = _corpus()
    assert corpus["cases"], "an empty corpus proves nothing"
    for case in corpus["cases"]:
        shape = msgspec.convert(case["shape"], RequestShape)
        assert evaluate(case["terms"], shape) == case["bytes"], case["name"]


def test_every_corpus_refusal_refuses_here_too() -> None:
    for case in _corpus()["refusals"]:
        shape = msgspec.convert(case["shape"], RequestShape)
        with pytest.raises(DemandEvaluationError):
            evaluate(case["terms"], shape)


def test_the_corpus_carries_the_vocabulary_it_was_generated_against() -> None:
    """A Go reader validates its own term table against these rows.

    Without this the two languages could hold different SCALES for the same
    term name and agree on every case that happens to divide evenly.
    """

    corpus = _corpus()
    assert corpus["vocabulary_version"] == VOCABULARY_VERSION
    assert corpus["vocabulary"] == vocabulary_document()
    assert corpus["shape_bounds"] == dict(SHAPE_BOUNDS)


def test_the_corpus_gate_goes_red_when_the_evaluator_MOVES() -> None:
    """The guard is falsified by moving the source of truth, not by prose.

    Runs the real gate script against a corpus whose expected bytes were
    edited by one, which is exactly what a silent algebra change looks like
    from the outside.
    """

    import tempfile

    corpus = _corpus()
    corpus["cases"][0]["bytes"] += 1
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demand_vectors.json"
        path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n")
        digest = Path(tmp) / "DEMAND_VECTORS_DIGEST"
        digest.write_text("0" * 64 + "\n")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_demand_corpus_digest.py"),
                "--corpus", str(path), "--digest", str(digest),
            ],
            capture_output=True, text=True,
        )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "not what this repository's evaluator produces" in result.stdout


def test_the_overflow_case_is_actually_an_overflow_case() -> None:
    """The corpus's ceiling row must be one that breaks a 64-bit reader.

    THIS ROW HAS ALREADY EARNED ITS KEEP. The first Go evaluator did a split
    division (`q*v + (r*v)//scale`) on the premise that the shape bounds kept
    both products inside int64; `r*v` is 1.3e19 here, and Go wrapped by
    exactly 2**64/1e6 on the corpus's first run. Python's bignums could not
    have surfaced that in a thousand green test runs — which is the entire
    argument for having a corpus a second language executes.
    """

    ceiling = RequestShape(
        width=SHAPE_BOUNDS["width"], height=SHAPE_BOUNDS["height"],
        batch=SHAPE_BOUNDS["batch"], frames=SHAPE_BOUNDS["frames"],
        latent_tokens=SHAPE_BOUNDS["latent_tokens"],
    )
    naive = GiB(1) * ceiling.width * ceiling.height * ceiling.batch
    assert naive > MAX_RESULT_BYTES, "the ceiling row no longer overflows int64"
    # The ANSWER still fits, which is what makes this a conformance case rather
    # than a refusal case.
    assert (const(GiB(64)) + per_mp_batch(GiB(1))).evaluate(ceiling) < MAX_RESULT_BYTES


def test_a_result_past_int64_REFUSES_rather_than_wrapping() -> None:
    """The bound is on the ANSWER, and it exists because Go is the other reader.

    Python could carry this number forever; the wire form and the Go evaluator
    cannot. A wrapped answer is a SMALL number, which is the direction that
    buys too little card, so the refusal is loud instead.
    """

    huge = RequestShape(latent_tokens=SHAPE_BOUNDS["latent_tokens"])
    with pytest.raises(DemandEvaluationError, match="64-bit"):
        evaluate([{"term": "latent_tokens", "bytes": 1 << 33}], huge)
    # One below the bound is fine — the refusal is a ceiling, not a taste.
    assert evaluate([{"term": "latent_tokens", "bytes": 1 << 31}], huge) == 1 << 62


# --------------------------------------------------------------------------
# The algebra
# --------------------------------------------------------------------------


def test_the_sdxl_reference_shape_evaluates_to_the_number_in_the_comment() -> None:
    formula = const(GiB(1.2)) + per_mp_batch(MiB(220))
    shape = RequestShape(width=1024, height=1024, batch=2)
    assert formula.evaluate(shape) == 1772275304
    assert round(formula.evaluate(shape) / MiB(1)) == 1690


def test_evaluation_is_FLOOR_not_round_and_not_float() -> None:
    """A float divide answers differently here; Go must not be allowed to."""

    terms = [{"term": "megapixels", "bytes": 1_000_003}]
    shape = RequestShape(width=1062, height=1099)
    product = 1_000_003 * 1062 * 1099
    exact = product // 1_000_000
    assert evaluate(terms, shape) == exact == 1_167_141
    assert round(product / 1_000_000) == 1_167_142, (
        "this row no longer discriminates floor from round"
    )


def test_the_vocabulary_is_closed_at_evaluation_too() -> None:
    with pytest.raises(DemandEvaluationError, match="CLOSED"):
        evaluate([{"term": "teraflops", "bytes": 1}], RequestShape())


def test_a_shape_past_the_declared_domain_REFUSES() -> None:
    """The refusal direction matters: a wrapped worst case is a SMALL number."""

    over = RequestShape(width=SHAPE_BOUNDS["width"] + 1, height=8)
    with pytest.raises(DemandEvaluationError, match="exceeds the declared"):
        const(1).evaluate(over)


def test_a_negative_shape_scalar_REFUSES() -> None:
    with pytest.raises(DemandEvaluationError, match="negative"):
        const(1).evaluate(RequestShape(width=-1))


def test_every_constructor_produces_a_term_in_the_vocabulary() -> None:
    made = {
        const(1), per_mp(1), per_mp_batch(1), per_batch(1),
        per_frame(1), per_frame_squared(1), per_latent_token(1),
    }
    names = {term.name for demand in made for term in demand.terms}
    assert names == set(TERM_VOCABULARY), (
        "a vocabulary entry with no constructor is unreachable, and a "
        "constructor with no entry is a term Go cannot evaluate"
    )


def test_a_negative_coefficient_is_refused_as_a_TECHNIQUE_not_a_term() -> None:
    with pytest.raises(DemandDeclarationError, match="REDUCES demand"):
        const(-1)


# --------------------------------------------------------------------------
# Provenance — no magic constants
# --------------------------------------------------------------------------


def test_the_default_basis_is_the_honest_one() -> None:
    assert const(MiB(1)).terms[0].basis is Basis.UNCALIBRATED
    assert const(MiB(1)).weakest_basis() is Basis.UNCALIBRATED


def test_claiming_a_measurement_without_naming_it_REFUSES() -> None:
    with pytest.raises(DemandDeclarationError, match="magic constant"):
        const(MiB(1155), basis=Basis.MEASURED)
    with pytest.raises(DemandDeclarationError, match="magic constant"):
        per_mp_batch(MiB(220), basis="ledger")


def test_a_formula_is_only_as_calibrated_as_its_WORST_term() -> None:
    mixed = (
        const(MiB(1155), basis=Basis.MEASURED, source="tcg#80 n=1")
        + per_mp_batch(MiB(220))
    )
    assert mixed.weakest_basis() is Basis.UNCALIBRATED
    assert mixed.bases() == {
        "const": Basis.MEASURED, "mp_batch": Basis.UNCALIBRATED,
    }


def test_summing_two_terms_of_DIFFERENT_provenance_refuses() -> None:
    """The sum would inherit one claim and launder the other."""

    with pytest.raises(DemandDeclarationError, match="DIFFERENT provenance"):
        const(MiB(1), basis=Basis.MEASURED, source="a run") + const(MiB(2))


def test_the_serialized_document_carries_every_basis() -> None:
    document = (
        const(MiB(1155), basis=Basis.MEASURED, source="tcg#80 n=1")
        + per_mp_batch(MiB(220))
    ).formula_document()
    rows = {row["term"]: row for row in document["terms"]}
    assert rows["const"]["basis"] == "measured"
    assert rows["const"]["source"] == "tcg#80 n=1"
    assert rows["mp_batch"]["basis"] == "uncalibrated"
    assert "source" not in rows["mp_batch"]
    assert document["arena"] == "request"


# --------------------------------------------------------------------------
# The envelope
# --------------------------------------------------------------------------


class _Bounded(msgspec.Struct):
    width: Annotated[int, msgspec.Meta(ge=64, le=2048)] = 1024
    height: Annotated[int, msgspec.Meta(ge=64, le=2048)] = 1024
    num_images: Annotated[int, msgspec.Meta(ge=1, le=4)] = 1


class _Unbounded(msgspec.Struct):
    width: int = 1024
    height: int = 1024


_TABLE = {"wide": (1536, 640), "square": (1024, 1024), "tall": (640, 1536)}


class _Enumerated(msgspec.Struct):
    aspect: Annotated[str, Shape(pixels=_TABLE)] = "square"


def test_a_bounded_payload_advertises_its_ceiling() -> None:
    envelope = advertised_envelope(_Bounded)
    assert envelope.axes["width"].value == 2048
    assert envelope.axes["batch"].value == 4
    assert envelope.axes["batch"].source == "advertised"
    assert set(envelope.advertised_axes) == {"width", "height", "batch"}


def test_an_UNBOUNDED_field_is_a_stated_default_not_a_guess() -> None:
    envelope = advertised_envelope(_Unbounded)
    assert envelope.axes["width"].source == "default"
    assert "NO upper bound" in envelope.axes["width"].why


def test_an_axis_nothing_advertises_says_WHY_and_names_the_falsifier() -> None:
    envelope = advertised_envelope(_Bounded)
    assert envelope.axes["frames"].source == "default"
    assert envelope.axes["frames"].value == 0
    batch = advertised_envelope(_Enumerated).axes["batch"]
    assert batch.source == "default" and "demand_miss" in batch.why


def test_the_bucket_table_beats_a_per_axis_max_on_a_COUPLED_envelope() -> None:
    """max(width) * max(height) is 1536x1536 — a bucket that does not exist.

    The worst case is the argmax over the REAL buckets, and taking the
    per-axis maximum would buy a card for a request the API cannot express.
    """

    formula = const(0) + per_mp_batch(MiB(220))
    envelope = advertised_envelope(_Enumerated)
    worst_bytes, worst_shape = envelope.worst_case(formula)
    assert (worst_shape.width, worst_shape.height) == (1024, 1024)
    assert worst_bytes == formula.evaluate(RequestShape(width=1024, height=1024))
    assert worst_bytes < formula.evaluate(RequestShape(width=1536, height=1536))


def test_request_shape_reads_a_bucket_table_at_SERVE_time_too() -> None:
    assert request_shape(_Enumerated(aspect="wide")) == RequestShape(
        width=1536, height=640, batch=1,
    )
    assert request_shape(_Bounded(width=512, height=512, num_images=3)) == (
        RequestShape(width=512, height=512, batch=3)
    )


def test_a_payload_with_no_shape_axis_yields_the_zero_shape() -> None:
    class _Blind(msgspec.Struct):
        prompt: str = ""

    assert request_shape(_Blind()) == RequestShape(batch=1)
    assert (const(MiB(7)) + per_mp(GiB(4))).evaluate(request_shape(_Blind())) == MiB(7)


def test_a_shape_annotation_that_states_both_or_neither_REFUSES() -> None:
    with pytest.raises(ShapeDeclarationError, match="never both and never neither"):
        Shape()
    with pytest.raises(ShapeDeclarationError, match="never both and never neither"):
        Shape("width", pixels=_TABLE)


def test_a_shape_annotation_naming_an_axis_outside_the_closed_set_REFUSES() -> None:
    with pytest.raises(ShapeDeclarationError, match="CLOSED"):
        Shape("gigapixels")


def test_the_demand_block_derives_its_number_and_says_what_it_is_NOT() -> None:
    formula = const(MiB(1155), basis=Basis.MEASURED, source="tcg#80 n=1")
    block = demand_document(formula, advertised_envelope(_Enumerated))
    assert block["worst_case_request_bytes"] == MiB(1155)
    assert block["worst_case_shape"]["width"] == 1024
    assert block["arena"] == "request"
    assert "manifest weight bytes" in block["note"]
    assert block["envelope"]["width"]["source"] == "declared"
