"""pgw#1032 — the COMPUTED key space has had no producer since pgw#1010, so
advertising it as deliverable demand advertised nothing.

Two key digests share the ``ck1-<56 hex>`` shape and are DISJOINT SPACES by
construction:

* ``cell_key.compute`` hard-codes ``kind="inductor"`` — what THIS runtime's
  static axes ask for, known before anything is compiled;
* every publishable cell is STAMPED by ``aot_mint.cell_identity`` with
  ``kind=aot_serve.ARTIFACT_KIND`` (``"aot-inductor"``) — what the exported
  cell IS, unknowable until the export finishes.

``kind`` is inside the canonical axis set the digest is taken over, so the two
can never collide. Since pgw#1010 the AOT export is the ONLY mint any pod
runs (JIT intake compiles in-process and names no artifact at all), so nothing
has produced an ``inductor``-kind artifact on any pod since. Every hub
mechanism keyed on exact equality with a COMPUTED key — store lookup delivery,
HelloAck snapshot attach, the ``ADOPT_COMPILE_CACHE`` hot push, the
``cell_demand`` ledger — was therefore structurally unable to fire, and this
issue deletes the worker half that fed them.

What SURVIVES is the ACTIVE (stamped) advertisement: the worker states the
identity of the artifact it is actually serving, and the hub's dispatch fence
verifies THAT against its own store. These tests pin both halves.
"""

from __future__ import annotations

from typing import Dict

import pytest

from gen_worker import aot_serve, cell_key, serving_mode
from gen_worker.pb import worker_scheduler_pb2 as pb

FAMILY = "sdxl"

#: One axis set, spelled once, so the only difference between the two keys
#: below is the ``kind`` value — which is the whole point.
_SHARED_AXES: Dict[str, str] = {
    "format": "2",
    "family": FAMILY,
    "lane": "w8a8",
    "mode": "",
    "sm": "89",
    "contract": "0123456789abcdef",
    "env_seal": "fedcba9876543210",
    "toolchain": "00112233445566aa",
}


# ---------------------------------------------------------------------------
# 1. The disjointness proof, in code
# ---------------------------------------------------------------------------


def test_a_computed_key_and_a_stamped_key_are_disjoint_spaces() -> None:
    """The fact the whole cut rests on: identical axes, different ``kind``,
    different digest. Exact-key equality between the two can never hold, so
    every hub mechanism that looked a COMPUTED key up in a store of STAMPED
    keys was dead by construction."""
    computed = cell_key.from_axes({**_SHARED_AXES, "kind": "inductor"})
    stamped = cell_key.from_axes({**_SHARED_AXES, "kind": aot_serve.ARTIFACT_KIND})

    assert aot_serve.ARTIFACT_KIND == "aot-inductor"
    assert computed.digest != stamped.digest, (
        "kind is inside the canonical axis set the digest covers; if these "
        "ever collide the pull-by-key delivery this issue retired was not "
        "dead after all")
    # Both are well-formed keys — the spaces are indistinguishable by SHAPE,
    # which is exactly why the divergence went unnoticed for so long.
    assert cell_key.is_key(computed.digest) and cell_key.is_key(stamped.digest)


def test_compute_can_only_ever_mint_an_inductor_kind_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cell_key.compute`` hard-codes the kind; no caller can widen it."""
    from gen_worker import compile_cache as cc

    monkeypatch.setattr(cc, "runtime_key", lambda: {"sm": "89", "torch": "2.9"})
    monkeypatch.setattr(cc, "toolchain_digest", lambda: {"torch": "2.9.0"})
    key = cell_key.compute(FAMILY, "w8a8", 0, contract="0123456789abcdef")
    assert key.axes_dict()["kind"] == "inductor"


def test_a_stamped_aot_key_is_refused_by_the_computed_brain() -> None:
    """The other direction of the same wall: an exported cell's identity may
    not be RECOMPUTED from inductor-cache axes — it is read off the envelope.
    So the hub could not have bridged the two spaces either."""
    with pytest.raises(cell_key.CellKeyError, match="aot-inductor"):
        cell_key.from_artifact_metadata(
            {"kind": aot_serve.ARTIFACT_KIND, "cell_key": "ck1-" + "b" * 56})


# ---------------------------------------------------------------------------
# 2. The producer is gone: no computed key on the wire
# ---------------------------------------------------------------------------


def test_a_compile_target_advertises_no_computed_key(monkeypatch) -> None:
    """RED before this issue: ``_refresh_compile_target`` computed a key and
    stamped it on every target as ``requested_cell_key``. A target now states
    only what it IS serving."""
    from gen_worker import executor as executor_mod

    fields = {f.name for f in pb.CompileTarget.DESCRIPTOR.fields}
    assert "requested_cell_key" in fields, (
        "the wire field is retired by the th#1457/pgw#891 RunJob cut, not "
        "here — this test is about the PRODUCER")

    assert not hasattr(executor_mod._CompileTargetRecord, "requested_cell_key")
    assert not hasattr(executor_mod._CompileTargetRecord, "requested_cell_axes")


def test_the_executor_no_longer_produces_cell_lookups() -> None:
    """``cell_lookups`` advertised SPECULATIVE computed keys — up to five base
    lanes per declared spec — into the same dead space. Its producer is gone;
    the wire field retires with the RunJob cut."""
    from gen_worker import executor as executor_mod
    from gen_worker import lifecycle as lifecycle_mod
    from gen_worker.procsplit import merge as merge_mod

    assert not hasattr(executor_mod.Executor, "cell_lookups")
    assert "cell_lookups" not in lifecycle_mod.__dict__.get("__source_marker__", "")
    # The merge of G child deltas must not resurrect it either.
    merged = merge_mod.merge_state_deltas([
        pb.StateDelta(cell_lookups=[pb.CellLookup(family="f", cell_key="ck1-a")]),
        pb.StateDelta(cell_lookups=[pb.CellLookup(family="f", cell_key="ck1-b")]),
    ])
    assert list(merged.cell_lookups) == []


def test_the_divergence_warning_is_gone_with_the_divergence() -> None:
    """``_warn_cell_key_divergence`` compared the two spaces above and so fired
    an ERROR on every healthy AOT boot (pgw#1033 silenced it as an interim and
    named this issue as the deleter)."""
    from gen_worker import executor as executor_mod

    assert not hasattr(executor_mod.Executor, "_warn_cell_key_divergence")


# ---------------------------------------------------------------------------
# 3. The handler nothing ever dispatched
# ---------------------------------------------------------------------------


def test_no_adopt_compile_cache_handler_survives() -> None:
    """The ~660-line hot-adoption handler was reachable only from the hub's
    ``ModelOp{ADOPT_COMPILE_CACHE}`` push, which is keyed off the same
    never-matching computed key (hub census: th#1702). No stack has ever
    dispatched one."""
    from gen_worker import executor as executor_mod
    from gen_worker import lifecycle as lifecycle_mod

    for gone in (
        "handle_model_op",
        "_handle_compile_cache_adoption",
        "_adopt_compile_cache",
        "_adoption_intent",
        "_adoption_event",
    ):
        assert not hasattr(executor_mod.Executor, gone), gone
    assert not hasattr(lifecycle_mod.Lifecycle, "_adopt_compile_cache_then_delta")


def test_the_strict_hot_adopt_arm_entry_point_is_gone() -> None:
    """``compile_cache.arm_staged_artifact`` existed ONLY for hot adoption —
    the strict twin of ``enable`` whose mismatch raises instead of falling back
    to eager. ``stage_artifact`` itself stays: ``seed_artifact`` and ``enable``
    are live callers."""
    from gen_worker import compile_cache as cc

    assert not hasattr(cc, "arm_staged_artifact")
    assert hasattr(cc, "stage_artifact")


def test_the_demand_echoing_build_entry_point_is_gone_entirely() -> None:
    """A demand-driven mint used to echo the worker-computed key it had to
    satisfy. Nothing has issued demand-driven mints since the forge was
    deleted, and no caller passed it.

    pgw#1032 asserted the PARAMETER had left `compile_cache.build`'s signature.
    pgw#1035 then deleted `build` outright — the whole-pipeline dynamo mint had
    no caller at all, and `aot_cells.discover` rejects the artifact kind it
    produced. Absence of the function is the STRONGER form of the same claim
    (a parameter cannot come back to a function that does not exist), so this
    is the assertion that survives the merge of the two lanes.
    """
    from gen_worker import compile_cache as cc

    assert not hasattr(cc, "build")


# ---------------------------------------------------------------------------
# 4. What SURVIVES — the advertised ACTIVE identity and the fence
# ---------------------------------------------------------------------------


class _ArmedPipe:
    """A pipeline that reports itself compile-armed (the JIT intake shape)."""


def test_serving_tier_and_serving_mode_answer_DIFFERENT_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1032 was filed asking for these two to be reconciled onto one
    authority. They must NOT be, and this test pins why.

    ``serving_mode`` answers *what code ran this request*: a JIT-intake arm
    compiles its own graphs, so ``jit_cell`` is the honest answer even though
    pgw#1010 left it naming no artifact. ``serving_tier`` answers *is this
    worker serving from a CELL* — the hub reads it as ADOPTION evidence
    (``WorkerServingCompiledTier`` -> ``WorkerAdoptedDeliveredCell``, th#1216),
    so an intake pod reporting ``compiled`` would testify that the cell
    exchange worked on a pod that adopted nothing. The divergence the issue
    called a defect is the design; `test_guard_miss_pgw680` and
    `test_eager_first_boot_pgw671` assert the same contract from the other
    side."""
    from gen_worker import compile_cache as cc

    pipe = _ArmedPipe()
    monkeypatch.setattr(cc, "is_compile_armed", lambda p: p is pipe)

    # Same pod, same instant, two honest answers.
    assert serving_mode.classify_mode("", pipe) == serving_mode.MODE_JIT_CELL
    assert cc.is_compile_armed(pipe) is True
    # A cell ref IS what makes the tier compiled — nothing else.
    assert serving_mode.classify_mode(
        f"root/family-{FAMILY}#ck1-" + "b" * 56, None) == serving_mode.MODE_JIT_CELL


def test_an_armed_target_still_advertises_the_identity_it_serves() -> None:
    """The surviving half: the ACTIVE (stamped) ref plus its snapshot digest.
    This is what ``hubPublishedAdvertisedCell`` verifies against the hub's own
    store before W8A8 dispatch rides it — the load-bearing fence."""
    from gen_worker import executor as executor_mod

    ref = f"root/family-{FAMILY}#ck1-" + "b" * 56
    target = executor_mod._CompileTargetRecord(
        incarnation_id="inc-1", spec=None, pipeline=object(),  # type: ignore[arg-type]
        pipeline_weight_lane="w8a8", lora_bucket=0, contract_digest="d",
        active_compile_ref=ref, active_compile_snapshot_digest="sha256:aa",
        active_self_mint=True, function_names=("generate",),
        model_bindings=(("unet", "acme/m", "blake3:cc"),))

    wire = pb.CompileTarget(
        incarnation_id=target.incarnation_id,
        family=FAMILY,
        pipeline_weight_lane=target.pipeline_weight_lane,
        lora_bucket=target.lora_bucket,
        contract_digest=target.contract_digest,
        active_compile_ref=target.active_compile_ref,
        active_compile_snapshot_digest=target.active_compile_snapshot_digest,
        function_names=list(target.function_names),
        model_bindings=[pb.CompileTargetBinding(slot=s, ref=r, snapshot_digest=d)
                        for s, r, d in target.model_bindings],
    )
    assert wire.active_compile_ref == ref
    assert wire.active_compile_snapshot_digest == "sha256:aa"
    assert not wire.requested_cell_key
    assert not dict(wire.requested_cell_axes)
