"""pgw#1606 — a SERVE LANE with no top-level dtype is unselectable, so say so
about the vendored set rather than discovering it one endpoint at a time.

**Why this file exists.** `sdxl.diffusers-nvfp4-flat@1` was authored (tensorfs
PR #144) with per-tensor `dtypes` and **no top-level `dtype`**. That document
is a serve lane, and a serve lane with no dtype cannot be used at all:

* pgw#1599 refuses it at CLASS DEFINITION — `serving/model.py:149 lane_dtype`
  turns the read's raise into a `ModelDeclarationError`, and the waiver set
  `DTYPELESS_UPSTREAM_LANES` is empty and documented to stay that way;
* this issue's ladder would independently reject it — `dtype_body(None)` is
  `""`, which is `REJECT_UNKNOWN_DTYPE`.

Failing closed twice is correct. What is NOT correct is finding out when an
endpoint author writes the lane. The generator that produced that document
emits per-tensor dtypes without a top-level one, so the hazard is a PROPERTY OF
THE GENERATOR and the next document off it carries the same hole.

The precedent this asserts is already settled by the vendored set itself: the
top-level dtype names **the serve lane's quantization, not a uniform tensor
dtype**. `sdxl.diffusers-fp8-rowwise@1` is a mixed tree — 36 UNET-only fp8 rows
out of 257, bf16 text encoders — and declares `float8_e4m3fn` anyway.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

#: Documents that are FRAGMENTS — a shared block or component spelling that a
#: `lanes=` header never names on its own, so they never reach a dtype read.
#: The same three pgw#1599's `DTYPELESS_UPSTREAM_LANES` comment names.
FRAGMENTS = frozenset({
    "dit.blocks-fused-qkv",
    "sdxl.clip-g-fused-qkv",
    "sdxl.clip-g-split-qkv",
})

#: Non-safetensors packagings whose dtype is carried in the container, not the
#: contract (a GGUF quant type is per-tensor by construction).
CONTAINER_TYPED = ("gguf",)

#: Serve lanes KNOWINGLY vendored dtype-less, each against a named upstream
#: issue. Same discipline as pgw#1599's `DTYPELESS_UPSTREAM_LANES`: a waiver is
#: a fact about the vendored document, never a preference, so it cannot outlive
#: its reason — `test_the_pending_dtype_waiver_deletes_itself` below FAILS the
#: moment the document grows a dtype, and the entry must then be removed.
#:
#: `sdxl.diffusers-nvfp4-flat` — authored by tensorfs#130 with per-tensor
#: `dtypes` and no top-level one. The coordinator sequenced pgw#1157 to merge
#: AS IS (it unblocks nine other classes; nvfp4 was undeclarable either way)
#: with the dtype fix riding a second mini-bump behind. This entry exists ONLY
#: to keep that window from false-reddening a fence that is otherwise correct.
#: It does NOT make the lane usable: pgw#1599 still refuses it at class
#: definition and pgw#1606's ladder still answers `unknown_dtype`.
PENDING_DTYPE = frozenset({"sdxl.diffusers-nvfp4-flat"})

_CONTRACTS = (
    Path(__file__).resolve().parents[1]
    / "src" / "gen_worker" / "_vendor" / "tensorfs" / "_contracts"
)


def _documents() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for path in sorted(_CONTRACTS.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        name = str(doc.get("name") or "")
        if not name:
            continue
        rows.append((name, doc))
    return rows


def test_the_vendored_set_is_actually_there() -> None:
    """Read the COUNT, not the verdict. An empty glob would make every
    assertion below vacuously true, which is the failure mode this whole file
    is about."""
    rows = _documents()
    assert len(rows) >= 20, (
        f"only {len(rows)} contract documents found under {_CONTRACTS} — the "
        f"vendored set moved or the glob broke. Every dtype assertion below "
        f"passes vacuously on an empty set"
    )


def test_every_serve_lane_declares_a_top_level_dtype() -> None:
    """A serve lane with no dtype is refused at declaration (pgw#1599) and
    rejected by the ladder (pgw#1606). Catch it in the vendored set, once,
    instead of in an endpoint author's class header, repeatedly."""
    missing = [
        name for name, doc in _documents()
        if name not in FRAGMENTS
        and name not in PENDING_DTYPE
        and not any(tag in name for tag in CONTAINER_TYPED)
        and doc.get("dtype") in (None, "")
    ]
    assert not missing, (
        f"these vendored SERVE LANES declare no top-level `dtype`: {missing}. "
        f"Such a lane cannot be declared (pgw#1599 `lane_dtype` refuses at "
        f"class definition) and cannot be selected (pgw#1606's ladder answers "
        f"`unknown_dtype`), so it ships and is unusable. The top-level dtype "
        f"names the SERVE LANE'S QUANTIZATION, not a uniform tensor dtype — "
        f"`sdxl.diffusers-fp8-rowwise@1` is a mixed tree and declares "
        f"`float8_e4m3fn` anyway. If one of these is really a FRAGMENT, add it "
        f"to FRAGMENTS above with the reason"
    )


def test_every_declared_dtype_is_one_the_capability_table_knows() -> None:
    """An unknown dtype answers floor 0 — silently. That is deliberate in
    `capability_floor_for_dtype` (an invented floor would be a placement claim
    with nothing behind it), and it is exactly why the vendored set must not
    contain one: a quantized lane silently floored at 0 would be offered to
    every card in the fleet."""
    from gen_worker.models.tensor_layout_contract import DTYPE_MIN_SM

    unknown = sorted({
        str(doc["dtype"]) for _, doc in _documents()
        if doc.get("dtype") and str(doc["dtype"]) not in DTYPE_MIN_SM
    })
    assert not unknown, (
        f"vendored documents declare dtypes the capability table has never "
        f"seen: {unknown}. `capability_floor_for_dtype` answers 0 for these — "
        f"no floor — so a quantized lane would be offered to a card that "
        f"cannot run it. Teach `DTYPE_MIN_SM` the dtype in the same change "
        f"that introduces it"
    )


def test_the_quantized_lanes_derive_a_real_floor() -> None:
    """The fp8 documents must derive 89 and any 4-bit one must derive 100.
    This is the arithmetic acceptance (a)'s ladder rests on: it is what makes
    an Ampere card reject fp8 and an Ada card reject nvfp4."""
    from gen_worker.models.tensor_layout_contract import capability_floor_for_dtype

    floors = {
        name: capability_floor_for_dtype(doc.get("dtype"))
        for name, doc in _documents() if doc.get("dtype")
    }
    fp8 = {n: f for n, f in floors.items() if "fp8" in n}
    assert fp8, "the vendored set must contain at least one fp8 lane"
    assert all(f == 89 for f in fp8.values()), fp8
    four_bit = {n: f for n, f in floors.items()
                if "nvfp4" in n or "fp4" in n or "w4a4" in n}
    assert all(f == 100 for f in four_bit.values()), four_bit


def test_the_pending_dtype_waiver_deletes_itself() -> None:
    """A waiver may not outlive its reason.

    pgw#1599 learned this the expensive way and wrote it down: it briefly held
    `minimax.h3-dit-diffusers@1` while that document was dtype-less, tensorfs
    gave it `bfloat16`, and the entry had to go. The mechanism that made that
    automatic was a test exactly like this one. Without it a waiver becomes a
    permanent exemption nobody re-reads, which is the same guard-goes-green-by
    -construction shape this file exists to catch.
    """
    # A waiver may be PRE-REGISTERED for a document that is not vendored yet:
    # this one was written while pgw#1157 was still open, precisely so the
    # sequenced window (bump merges as-is, dtype follows) does not false-red a
    # fence that is otherwise correct. "Not present" is therefore legal and is
    # NOT what this test guards.
    #
    # What it guards is the one state that must never persist: the document is
    # here AND it has a dtype, so the waiver's reason is gone and the entry is
    # now hiding a lane from `test_every_serve_lane_declares_a_top_level_dtype`.
    fixed = sorted(
        name for name, doc in _documents()
        if name in PENDING_DTYPE and doc.get("dtype")
    )
    assert not fixed, (
        f"upstream landed the top-level dtype for {fixed} — DELETE those "
        f"entries from PENDING_DTYPE. The waiver's whole reason is gone, and "
        f"`test_every_serve_lane_declares_a_top_level_dtype` should now be "
        f"guarding these documents like every other serve lane"
    )


@pytest.mark.skipif(
    not any(
        "nvfp4" in n and d.get("dtype") for n, d in _documents()
    ),
    reason="sdxl.diffusers-nvfp4-flat@1 is not vendored WITH A DTYPE yet "
           "(pgw#1157 merges as-is; the dtype rides a second mini-bump). When "
           "it lands this runs and pins the floor the nvfp4 rung needs, and "
           "`test_the_pending_dtype_waiver_deletes_itself` fails until the "
           "waiver above is removed — so this cannot be forgotten",
)
def test_the_nvfp4_lane_is_declarable_and_floors_at_blackwell() -> None:
    """The re-proof this lane owes acceptance (a): once the real document is
    vendored, the nvfp4 rung must be expressible and must floor at sm100."""
    from gen_worker.models.tensor_layout_contract import capability_floor_for_dtype
    from gen_worker.serving.lane_ladder import dtype_body

    (name, doc), = [(n, d) for n, d in _documents() if "nvfp4" in n]
    dtype = doc.get("dtype")
    assert dtype, (
        f"{name} declares no top-level dtype, so no Model class can name it "
        f"as a lane — see this file's module docstring"
    )
    assert capability_floor_for_dtype(dtype) == 100
    assert dtype_body(dtype) == "nvfp4-w4a4-static"
