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

#: ⛔ `CONTAINER_TYPED` and `PENDING_DTYPE` BOTH DELETED 2026-08-20 by
#: tensorfs#130's follow-up (`51adc50`), and both deletions were FORCED by the
#: tests below rather than remembered.
#:
#: `PENDING_DTYPE` waived `sdxl.diffusers-nvfp4-flat` while it was dtype-less.
#: The document now declares `float4_e2m1fn`, `test_the_pending_dtype_waiver_
#: deletes_itself` went red naming the entry, and the entry is gone — which is
#: the entire behaviour that waiver was built to have.
#:
#: `CONTAINER_TYPED` exempted GGUF on this lane's reasoning that a block-quant
#: container "has no torch spelling and therefore no declarable dtype". **That
#: reasoning was wrong**, and the tensorfs#130 lane priced it before diverging:
#: dtype-less left `QwenMtpModel` UNDECLARABLE under the always-required
#: `lanes=`, i.e. the same waiver-shaped failure this file exists to catch, one
#: model over. The field never meant "a torch spelling" — it means THE LANE'S
#: QUANTIZATION, which is why `sdxl.diffusers-fp8-rowwise@1` declares
#: `float8_e4m3fn` over 257 declarations of which only 36 are fp8. `q4_k` is
#: the ggml type name and is exactly as legitimate.
#:
#: There is now NO exemption of any kind here. If one is ever needed again it
#: must arrive with a self-deleting test, as the last one did.

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
        if name not in FRAGMENTS and doc.get("dtype") in (None, "")
    ]
    assert not missing, (
        f"these vendored SERVE LANES declare no top-level `dtype`: {missing}. "
        f"Such a lane cannot be declared (pgw#1599 `lane_dtype` refuses at "
        f"class definition) and cannot be selected (pgw#1606's ladder answers "
        f"`unknown_dtype`), so it ships and is unusable. The top-level dtype "
        f"names the SERVE LANE'S QUANTIZATION, not a uniform tensor dtype — "
        f"`sdxl.diffusers-fp8-rowwise@1` is a mixed tree and declares "
        f"`float8_e4m3fn` anyway. If one of these is really a FRAGMENT, add it "
        f"to FRAGMENTS above with the reason — and note that 'it is a "
        f"block-quant container' is NOT a reason: `q4_k` is the ggml type name "
        f"and names its lane's quantization exactly as `float8_e4m3fn` does"
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


def test_there_is_no_exemption_left_to_outlive_its_reason() -> None:
    """Both exemptions this file once carried are gone, and neither was
    removed because someone remembered.

    `PENDING_DTYPE` deleted itself: the test that watched it went red the hour
    `sdxl.diffusers-nvfp4-flat` grew its dtype, named the entry, and the entry
    was removed in the same change. `CONTAINER_TYPED` was deleted because its
    premise was falsified upstream — a GGUF lane DOES declare its quantization.

    This test is what is left of both: it asserts the module carries no
    exemption set at all, so re-introducing one is a visible, deliberate act
    rather than a name quietly added to a tuple nobody re-reads.
    """
    import tests.test_lane_dtype_fence_pgw1606 as mod

    leftovers = [
        name for name in ("PENDING_DTYPE", "CONTAINER_TYPED", "WAIVED", "SKIP")
        if getattr(mod, name, None)
    ]
    assert not leftovers, (
        f"this module grew an exemption set again: {leftovers}. That is "
        f"allowed — but it must ship WITH a test that deletes it when its "
        f"reason expires, the way PENDING_DTYPE did. An exemption that cannot "
        f"expire is not a waiver, it is a permanent hole nobody re-reads"
    )


def test_a_floor_losing_spelling_is_refused_by_name_not_silently_priced() -> None:
    """`torch.float4_e2m1fn_x2` EXISTS and `torch.float4_e2m1fn` does not, so
    the wrong spelling is the one that looks right — and it resolves, and
    `DTYPE_MIN_SM` does not know it, so the lane would silently lose its sm100
    floor and be placed on Ampere.

    The nvfp4 document asks in its own description not to be 'fixed' this way.
    A comment cannot enforce that; this can. Note the choice: the spelling is
    REFUSED rather than aliased to 100. Aliasing would also work, and would
    make the wrong spelling spread.
    """
    from gen_worker.models.tensor_layout_contract import (
        DTYPE_MIN_SM,
        FLOOR_LOSING_SPELLINGS,
        capability_floor_for_dtype,
    )

    assert "float4_e2m1fn_x2" in FLOOR_LOSING_SPELLINGS
    # The trap is real: it prices at 0, which is Ampere-and-anything.
    assert capability_floor_for_dtype("float4_e2m1fn_x2") == 0
    assert "float4_e2m1fn_x2" not in DTYPE_MIN_SM, (
        "aliasing the packed-pair spelling to 100 would hide the mistake "
        "instead of refusing it, and would let the wrong spelling spread"
    )
    # ...and the spelling that must be used prices correctly.
    assert capability_floor_for_dtype("float4_e2m1fn") == 100

    offenders = sorted({
        str(doc["dtype"]) for _, doc in _documents()
        if str(doc.get("dtype") or "") in FLOOR_LOSING_SPELLINGS
    })
    assert not offenders, (
        f"vendored documents declare FLOOR-LOSING spellings {offenders}: "
        + "; ".join(f"{k} — {v}" for k, v in FLOOR_LOSING_SPELLINGS.items()
                    if k in offenders)
    )


@pytest.mark.skipif(
    not any(
        "nvfp4" in n and d.get("dtype") for n, d in _documents()
    ),
    reason="no nvfp4 document with a dtype is vendored (it landed in "
           "tensorfs 51adc50; if this ever skips again, the document REGRESSED "
           "and `test_every_serve_lane_declares_a_top_level_dtype` is the "
           "test that will say so)",
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
