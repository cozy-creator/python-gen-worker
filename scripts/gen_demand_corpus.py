#!/usr/bin/env python3
"""Regenerate the shared demand-formula conformance corpus.

pgw#1600 acceptance (a). The corpus is the CONTRACT between this repository's
evaluator and tensorhub's Go one: a table of (serialized formula, request
shape) -> bytes, plus the refusals both sides must produce. Go reads the same
bytes and must return the same integers.

The cases are chosen to break a naive implementation, not to look tidy:

* coefficients that are NOT multiples of their term's scale, so a float divide
  and a floor divide disagree;
* a coefficient SMALLER than its scale, which is the pure-remainder arm of the
  split division;
* the domain ceiling on every axis at once with a large coefficient, which is
  where `coefficient * value` would overflow int64 if either side formed the
  product before dividing;
* the tcg#80 sdxl reference shape, so the corpus carries the one payload a
  human can check against a measurement.

Run: `uv run python scripts/gen_demand_corpus.py` then record the digest with
`scripts/check_demand_corpus_digest.py --write`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gen_worker.demand import (  # noqa: E402
    GiB,
    KiB,
    MiB,
    RequestShape,
    SHAPE_BOUNDS,
    VOCABULARY_VERSION,
    evaluate,
    vocabulary_document,
)

CORPUS = ROOT / "src" / "gen_worker" / "contracts" / "demand_vectors.json"


def _case(name: str, terms: list[dict[str, object]], shape: RequestShape,
          why: str = "") -> dict[str, object]:
    row: dict[str, object] = {
        "name": name,
        "terms": terms,
        "shape": shape.as_document(),
        "bytes": evaluate(terms, shape),
    }
    if why:
        row["why"] = why
    return row


def _terms(*pairs: tuple[str, int]) -> list[dict[str, object]]:
    return [{"term": name, "bytes": coefficient} for name, coefficient in pairs]


def build() -> dict[str, object]:
    ceiling = RequestShape(
        width=SHAPE_BOUNDS["width"],
        height=SHAPE_BOUNDS["height"],
        batch=SHAPE_BOUNDS["batch"],
        frames=SHAPE_BOUNDS["frames"],
        latent_tokens=SHAPE_BOUNDS["latent_tokens"],
    )
    cases = [
        _case(
            "a const-only formula ignores the shape entirely",
            _terms(("const", GiB(1.2))),
            RequestShape(width=1024, height=1024, batch=2),
        ),
        _case(
            "the sdxl reference shape",
            _terms(("const", GiB(1.2)), ("mp_batch", MiB(220))),
            RequestShape(width=1024, height=1024, batch=2),
            why="1024x1024 with the CFG pair — the shape pgw#1577 measured "
                "and the one tcg#80 re-ran compiled on sm_89",
        ),
        _case(
            "a coefficient that is not a multiple of its scale FLOORS",
            _terms(("megapixels", 1_000_003)),
            RequestShape(width=1023, height=1021),
            why="1_000_003 * 1_044_483 / 1_000_000 is not an integer; a float "
                "divide and a floor divide answer differently here",
        ),
        _case(
            "a coefficient SMALLER than its scale is the pure-remainder arm",
            _terms(("megapixels", 7)),
            RequestShape(width=999, height=997),
            why="quotient is 0, so the whole answer comes out of "
                "(remainder * value) // scale",
        ),
        _case(
            "a sub-megapixel request rounds DOWN to zero on a small coefficient",
            _terms(("mp_batch", KiB(1))),
            RequestShape(width=64, height=64, batch=1),
        ),
        _case(
            "every video term at once",
            _terms(
                ("const", MiB(512)),
                ("frames", MiB(3)),
                ("frames_squared", KiB(64)),
                ("batch", MiB(11)),
            ),
            RequestShape(width=1280, height=720, batch=1, frames=241),
        ),
        _case(
            "latent tokens, the transformer-shaped term",
            _terms(("const", MiB(700)), ("latent_tokens", 8192)),
            RequestShape(width=1024, height=1024, batch=1, latent_tokens=4096),
        ),
        _case(
            "the zero shape: every non-const term contributes nothing",
            _terms(
                ("const", MiB(64)),
                ("megapixels", GiB(4)),
                ("mp_batch", GiB(4)),
                ("batch", GiB(4)),
                ("frames", GiB(4)),
                ("frames_squared", GiB(4)),
                ("latent_tokens", GiB(4)),
            ),
            RequestShape(width=0, height=0, batch=0, frames=0, latent_tokens=0),
            why="a lane whose request advertises no shape axis is not a lane "
                "with an unbounded demand",
        ),
        _case(
            "the domain ceiling on every axis with a large coefficient",
            _terms(
                ("const", GiB(64)),
                ("megapixels", GiB(1)),
                ("mp_batch", GiB(1)),
                ("batch", MiB(1)),
                ("frames", MiB(1)),
                ("frames_squared", 1),
                ("latent_tokens", 1),
            ),
            ceiling,
            why="THE OVERFLOW CASE. Forming coefficient*value before dividing "
                "overflows int64 here in both languages; the split division "
                "does not, and both sides must return this exact integer",
        ),
        _case(
            "the term ORDER in the document does not change the sum",
            _terms(("mp_batch", MiB(220)), ("const", GiB(1.2))),
            RequestShape(width=1024, height=1024, batch=2),
            why="same answer as the sdxl reference case above",
        ),
    ]
    refusals = [
        {
            "name": "an unknown term name is refused, never skipped",
            "terms": _terms(("gigaflops", 1024)),
            "shape": RequestShape(width=1024, height=1024).as_document(),
            "reason": "unknown_term",
            "why": "a formula evaluated without one of its terms is a LOW "
                   "number, and low is the direction that buys too little card",
        },
        {
            "name": "a shape past the declared domain is refused",
            "terms": _terms(("mp_batch", MiB(220))),
            "shape": {
                "width": SHAPE_BOUNDS["width"] + 1, "height": 1024,
                "batch": 1, "frames": 0, "latent_tokens": 0,
            },
            "reason": "shape_out_of_domain",
        },
        {
            "name": "a result past int64 is refused, never wrapped",
            "terms": _terms(("latent_tokens", 1 << 33)),
            "shape": RequestShape(latent_tokens=SHAPE_BOUNDS["latent_tokens"]).as_document(),
            "reason": "result_out_of_range",
            "why": "THE ROW THE GO SIDE FOUND. Python's bignums cannot notice "
                   "an int64 overflow, so the shared constraint is on the "
                   "ANSWER: 2**33 bytes per latent token at 2**31 tokens is "
                   "2**64, and a wrapped answer is a SMALL number",
        },
        {
            "name": "a negative shape scalar is refused",
            "terms": _terms(("const", MiB(1))),
            "shape": {
                "width": -1, "height": 1024, "batch": 1,
                "frames": 0, "latent_tokens": 0,
            },
            "reason": "shape_out_of_domain",
        },
    ]
    return {
        "version": 1,
        "vocabulary_version": VOCABULARY_VERSION,
        "algebra": "C + sum(coefficient_i * value_i / scale_i), floor, int64",
        "vocabulary": vocabulary_document(),
        "shape_bounds": dict(SHAPE_BOUNDS),
        "cases": cases,
        "refusals": refusals,
    }


def main() -> int:
    CORPUS.write_text(
        json.dumps(build(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {CORPUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
