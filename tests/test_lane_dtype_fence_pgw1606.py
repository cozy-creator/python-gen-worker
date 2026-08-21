"""The lane FENCE over the vendored corpus — what still bites after pgw#1621,
and what the v2 cut made INEXPRESSIBLE rather than merely unguarded.

**Why this file existed.** `sdxl.diffusers-nvfp4-flat@1` was authored (tensorfs
PR #144) with per-tensor `dtypes` and **no top-level `dtype`**. Under v1 a serve
lane was a per-lane DOCUMENT and that field was optional, so a dtype-less lane
shipped, cleared a `runtime_checkable` Protocol that never invokes its own
getter, and died at load on a rented pod. The generator that produced it emitted
per-tensor dtypes without a top-level one, so the hazard was a PROPERTY OF THE
GENERATOR and the next document off it carried the same hole. This file
enumerated the vendored set so that hole was found once, here, rather than one
endpoint author at a time.

**Three of its five assertions are now inexpressible. Each is recorded below
with what it was protecting against and why the hazard is structural now.**

* *(a) every serve lane declares a top-level dtype.* **Inexpressible.** There is
  no per-lane document to omit a field from. A lane is `(topology, quant)` and
  the dtype is `declared_dtype` on the QUANT RULE, which the v2 rule schema
  REQUIRES; there are eight rules for the whole fleet rather than one document
  per lane, and the corpus is checked against `spec/v2/vectors/digests.json`
  (`test_lane_contracts.py`). The waiver set this cost (`DTYPELESS_UPSTREAM_LANES`)
  and the self-deleting-waiver test that watched it are both deleted with their
  subject.

* *(b) every declared dtype is one `DTYPE_MIN_SM` knows.* **Inexpressible, and
  the table it guarded is gone.** `DTYPE_MIN_SM` was keyed on a dtype SPELLING
  and answered a silent, permissive `0` for anything it had not heard of — so a
  quantized lane whose dtype was spelled unfamiliarly was offered to every card
  in the fleet. It could not tell its own two zeros apart ("measured: no floor"
  vs "never heard of it"). `capability_floor_for_rule` takes a RULE HANDLE and
  REFUSES an unknown one; the refusal is proved below rather than assumed.

* *(d) `float4_e2m1fn_x2` is refused BY NAME.* **Inexpressible.** That guard
  existed because `torch.float4_e2m1fn_x2` (the packed-pair container type)
  EXISTS while `torch.float4_e2m1fn` does not, so the wrong spelling was the one
  that looked right — it resolved, `DTYPE_MIN_SM` did not know it, and the lane
  silently lost its sm100 floor onto Ampere. **There is no dtype string left to
  mis-spell.** The handle is `cozy.nvfp4-flat@1`, the floor is a field on that
  document, and a handle the corpus does not carry refuses. Same for the `q4_k`
  trap (tensorfs#130): a GGUF lane's floor would be whatever its rule document
  says, not whatever a spelling table happened to know.

**What still bites, and is asserted below:**

* every ratified rule derives the floor its own document states;
* **every ratified rule has a `_RULE_BODY` row in `lane_ladder`** — NEW, and
  the direct successor of the old `q4_k` guard's property: it fails the moment
  tensorfs ratifies a ninth rule, i.e. the day it lands rather than the day a
  lane is mis-placed;
* **`cozy.fp8-storage@1` and `cozy.fp8-rowwise@1` execute in DIFFERENT lanes**
  — NEW, and a real defect the pgw#1621 re-key exposed. Both declare
  `float8_e4m3fn`; a dtype-keyed table answered one body for both;
* the unknown-handle REFUSALS, proved;
* (e) no exemption is left to outlive its reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gen_worker.models.tensor_layout_contract import (
    LayoutDeclarationError,
    capability_floor_for_rule,
    known_quant_rules,
    quant_rules,
    rule_dtype,
)
from gen_worker.serving import lane_ladder as L

#: ⛔ THERE IS NO EXEMPTION SET HERE, and there has not been since 2026-08-20.
#:
#: `PENDING_DTYPE` waived `sdxl.diffusers-nvfp4-flat` while it was dtype-less;
#: it DELETED ITSELF — the document gained a dtype, the watching test went red
#: naming the entry, and the entry went in the same change. `CONTAINER_TYPED`
#: exempted GGUF on the reasoning that a block-quant container "has no torch
#: spelling and therefore no declarable dtype"; that reasoning was FALSIFIED
#: upstream (the field never meant a uniform tensor dtype — it means the LANE'S
#: QUANTIZATION, which is why `cozy.fp8-rowwise@1` declares `float8_e4m3fn`
#: over a mixed tree). `FRAGMENTS` waived three v1 component documents that a
#: `lanes=` header never named on its own; v2 has no per-lane documents, so it
#: has no fragments either — `sdxl.clip-g-fused@1` is a TOPOLOGY now, and a
#: topology carries no dtype by design.
#:
#: If an exemption is ever needed again it must arrive with a self-deleting
#: test, as PENDING_DTYPE did.

_RULES = (
    Path(__file__).resolve().parents[1]
    / "src" / "gen_worker" / "_vendor" / "tensorfs" / "spec" / "v2" / "rules"
)


def _documents() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for path in sorted(_RULES.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows.append((f"{doc['name']}@{doc['version']}", doc))
    return rows


def test_the_vendored_set_is_actually_there() -> None:
    """Read the COUNT, not the verdict."""
    rows = _documents()
    assert len(rows) == 8, (
        f"only {len(rows)} quant-rule documents found under {_RULES} — the "
        f"vendored corpus moved, the glob broke, or upstream ratified a new "
        f"rule. Every assertion below passes vacuously on an empty set."
    )
    # ...and the reader this repo actually uses agrees with the raw glob.
    assert set(known_quant_rules()) == {handle for handle, _ in rows}


def test_every_ratified_rule_derives_the_floor_its_document_states() -> None:
    """The arithmetic the ladder rests on: it is what makes an Ampere card
    reject fp8 and an Ada card reject nvfp4.

    Two halves, and both matter. First the reader agrees with the DOCUMENT for
    every rule — so this cannot pass by the reader and the corpus sharing one
    wrong opinion. Then the specific numbers the fleet depends on are named,
    because a floor that silently became 0 would agree with itself perfectly.
    """
    floors = {handle: capability_floor_for_rule(handle)
              for handle, _ in _documents()}
    for handle, doc in _documents():
        assert floors[handle] == int(doc["capability_floor_sm"]), handle

    fp8 = {h: f for h, f in floors.items() if "fp8" in h}
    assert len(fp8) == 3, fp8
    assert all(f == 89 for f in fp8.values()), fp8

    four_bit = {h: f for h, f in floors.items() if "fp4" in h}
    assert len(four_bit) == 2, four_bit
    assert all(f == 100 for f in four_bit.values()), four_bit

    # `plain.f32@1`'s 0 is a MEASURED zero — written in the document, not the
    # "never heard of it" zero the deleted dtype table could not tell it from.
    assert floors["plain.f32@1"] == 0
    assert floors["plain.f16@1"] == 70
    assert floors["plain.bf16@1"] == 80


def test_every_ratified_rule_has_a_serving_body() -> None:
    """NEW (pgw#1621), and the direct successor of the deleted `q4_k` guard.

    `lane_ladder._RULE_BODY` maps a ratified rule to the lane body it executes
    as, and `rule_body()` answering `""` becomes `REJECT_UNKNOWN_RULE` — a
    named rejection, so a rule the fleet has no executor for is passed over
    loudly rather than crashing a boot. That is the correct RUNTIME behaviour
    and it is exactly why this BUILD-TIME check has to exist: a lane silently
    rejected as "unknown rule" on every card in the fleet looks, from a pod
    log, like a lane nobody declared.

    **This fails the moment tensorfs ratifies a ninth rule** — the day it
    lands, not the day a lane is mis-placed. That is the property the old
    guard had and the reason it is rebuilt rather than dropped.
    """
    missing = [h for h in known_quant_rules() if not L.rule_body(h)]
    assert not missing, (
        f"these RATIFIED quant rules have no row in `lane_ladder._RULE_BODY`: "
        f"{missing}. A re-vendor brought in a rule the fleet has no executor "
        f"for, so every lane naming it will be rejected `unknown_quant_rule` "
        f"on every card — visible only as an absent lane. Add the row in the "
        f"SAME change that re-vendors, or say in `_RULE_BODY` why the rule is "
        f"deliberately unexecutable here."
    )
    # ...and every row is a body the fleet's ranked lane table actually knows,
    # so a typo in `_RULE_BODY` cannot invent an unrankable lane.
    from gen_worker.models.execution_lanes import known_execution_lane_bodies

    bodies = set(known_execution_lane_bodies())
    for handle in known_quant_rules():
        assert L.rule_body(handle) in bodies, handle

    # The table has EXACTLY the ratified rules — no row for a rule the corpus
    # does not carry, which would be an executor with nothing to execute.
    assert set(L._RULE_BODY) == set(known_quant_rules())


def test_the_two_cozy_fp8_rules_do_NOT_share_a_lane_body() -> None:
    """THE DEFECT THE pgw#1621 RE-KEY EXPOSED, pinned so it cannot come back.

    `cozy.fp8-storage@1` and `cozy.fp8-rowwise@1` both declare
    `float8_e4m3fn`. They execute in DIFFERENT lanes:

      * `cozy.fp8-rowwise@1` emits an F32 `[out]` `weight_scale` beside each
        weight (`"scale": "per_channel_out"`) and is consumed by the w8a8
        GEMM — body `fp8-w8a8-dynamic`.
      * `cozy.fp8-storage@1` is SCALE-FREE (`"scale": "none"`) and its own
        conventions say `"consumption": "diffusers layerwise cast to bf16"` —
        fp8 bytes resident, **bf16 compute**. Body `bf16-w16a16`.

    A table keyed on the DTYPE cannot see that difference, so it answered
    `fp8-w8a8-dynamic` for both — offering a scale-free tree to a GEMM that
    multiplies by scales which do not exist, and flooring the lane at sm89 for
    arithmetic it never performs. Both failures are SILENT: one produces
    plausible-looking wrong numbers, the other loses a card it could have run
    on. The dtype names the ELEMENT; the rule names the EXECUTOR, and only one
    of those is what a lane body is.
    """
    storage, rowwise = "cozy.fp8-storage@1", "cozy.fp8-rowwise@1"

    # The premise: one dtype, two rules. If this ever stops being true the
    # test below is proving nothing, so it is asserted rather than assumed.
    assert rule_dtype(storage) == rule_dtype(rowwise) == "float8_e4m3fn", (
        "these two rules no longer share a declared dtype, so a dtype-keyed "
        "table could tell them apart after all — re-read this test's premise"
    )

    assert L.rule_body(rowwise) == "fp8-w8a8-dynamic", (
        f"{rowwise} emits a per-output-channel `weight_scale` and feeds the "
        f"w8a8 GEMM; it got body {L.rule_body(rowwise)!r}"
    )
    assert L.rule_body(storage) == "bf16-w16a16", (
        f"{storage} is SCALE-FREE and consumed by a diffusers layerwise cast "
        f"to bf16 — the arithmetic that runs is bf16, so its body is the "
        f"baseline one. It got {L.rule_body(storage)!r}. If that is "
        f"`fp8-w8a8-dynamic`, the table has been re-keyed on the DTYPE again: "
        f"both rules declare float8_e4m3fn, so a dtype key cannot distinguish "
        f"a tree that carries scales from one that does not, and the w8a8 "
        f"kernel would multiply by scales that are not in the file."
    )
    assert L.rule_body(storage) != L.rule_body(rowwise)

    # ...and the conventions the bodies are derived FROM are still what the
    # documents say, so this test cannot pass against a corpus that changed
    # its mind about either rule.
    rules = quant_rules()
    assert rules[storage].conventions["scale"] == "none"
    assert "cast to bf16" in rules[storage].conventions["consumption"]
    assert rules[rowwise].conventions["scale"] == "per_channel_out"

    # The baseline body is what decides a lane needs no kernel gate, so the
    # storage lane must actually LAND there rather than merely be spelled it.
    assert L.is_baseline(L.rule_body(storage)) is True
    assert L.is_baseline(L.rule_body(rowwise)) is False


def test_an_unknown_rule_handle_REFUSES_rather_than_answering_a_floor() -> None:
    """The permissive direction is the one that puts an nvfp4 lane on Ampere.

    `DTYPE_MIN_SM` returned 0 — "no floor" — for anything it had not heard of,
    and could not distinguish that from a measured 0. Both readers now refuse,
    and the refusal is PROVED here rather than assumed: a guard nobody has
    watched fail is a guard nobody knows works.
    """
    for handle in ("q4_k", "float4_e2m1fn_x2", "cozy.nvfp4-flat@2", "", None):
        with pytest.raises(LayoutDeclarationError):
            capability_floor_for_rule(handle)
        with pytest.raises(LayoutDeclarationError):
            rule_dtype(handle)

    # The refusal NAMES the corpus, because "not registered" with no list is
    # the message that sends an author to grep.
    with pytest.raises(LayoutDeclarationError) as caught:
        capability_floor_for_rule("cozy.q4-k@1")
    message = str(caught.value)
    for known in known_quant_rules():
        assert known in message

    # ...and a version bump is a DIFFERENT rule, not the same one: `@2` above
    # refuses even though `@1` is ratified.
    assert capability_floor_for_rule("cozy.nvfp4-flat@1") == 100


def test_there_is_no_exemption_left_to_outlive_its_reason() -> None:
    """(e), kept as-is in spirit and widened by one name.

    Every exemption this file has ever carried is gone, and none was removed
    because someone remembered. `PENDING_DTYPE` deleted itself when the
    document it waived grew its dtype. `CONTAINER_TYPED` was deleted because
    its premise was falsified upstream. `FRAGMENTS` — three v1 component
    documents a `lanes=` header never named alone — lost its subject entirely:
    v2 has no per-lane documents, so it has no fragments.

    This test is what is left of all three: it asserts the module carries no
    exemption set at all, so re-introducing one is a visible, deliberate act
    rather than a name quietly added to a tuple nobody re-reads.
    """
    leftovers = [
        name for name in
        ("PENDING_DTYPE", "CONTAINER_TYPED", "FRAGMENTS", "WAIVED", "SKIP")
        if globals().get(name)
    ]
    assert not leftovers, (
        f"this module grew an exemption set again: {leftovers}. That is "
        f"allowed — but it must ship WITH a test that deletes it when its "
        f"reason expires, the way PENDING_DTYPE did. An exemption that cannot "
        f"expire is not a waiver, it is a permanent hole nobody re-reads"
    )


def test_the_deleted_dtype_vocabulary_is_GONE_not_merely_unused() -> None:
    """A table that still imports is a table something can still key on.

    Named one by one rather than checked as a group: each of these was a live
    producer of a floor, and a re-import of any single one restores the
    spelling-keyed hazard on its own.
    """
    from gen_worker.models import tensor_layout_contract as tlc

    for dead in ("DTYPE_MIN_SM", "FLOOR_LOSING_SPELLINGS",
                 "capability_floor_for_dtype", "KNOWN_CONTRACTS",
                 "DecodeDimensions", "ContractDecoder", "implements_contract",
                 "contract_decoders_of"):
        assert not hasattr(tlc, dead), (
            f"`{dead}` is back in tensor_layout_contract. The sm floor is a "
            f"property of the RULE (Paul, 2026-08-18) and a table keyed on a "
            f"dtype SPELLING silently loses it — see this module's docstring "
            f"for the two traps that cost."
        )

    from gen_worker.serving import model as model_module

    assert not hasattr(model_module, "DTYPELESS_UPSTREAM_LANES")
    assert not hasattr(L, "dtype_body")
    assert not hasattr(L, "REJECT_UNKNOWN_DTYPE")
    assert L.REJECT_UNKNOWN_RULE == "unknown_quant_rule"


def test_the_nvfp4_lanes_are_declarable_and_floor_at_blackwell() -> None:
    """The re-proof pgw#1606's acceptance (a) owed, now for BOTH nvfp4 rules.

    v1 had one nvfp4 document and it was the dtype-less one. v2 has two, and
    they are two rules with two digests precisely because one is LOW-nibble
    with flat scales and the other HIGH-nibble pre-swizzled — reading one as
    the other measured LPIPS 1.11 (te#151). Under v1 that difference lived on
    a side axis beside a shared handle; here it IS the handle.
    """
    for handle in ("cozy.nvfp4-flat@1", "bfl.nvfp4-preswizzled@1"):
        assert capability_floor_for_rule(handle) == 100, handle
        assert rule_dtype(handle) == "float4_e2m1fn", handle
        assert L.rule_body(handle) == "nvfp4-w4a4-static", handle

    # Two rules, two digests — never one handle with a side note.
    rules = quant_rules()
    assert (rules["cozy.nvfp4-flat@1"].digest
            != rules["bfl.nvfp4-preswizzled@1"].digest)
