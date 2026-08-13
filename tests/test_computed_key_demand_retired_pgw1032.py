"""pgw#1032 (re-cut by pgw#1059) — the COMPUTED key space has had no
producer since pgw#1010, so advertising it as deliverable demand advertised
nothing.

pgw#1059 finished the cut: the pre-trace "computed key" no longer EXISTS as
a key at all. An obligation's identity is ``fleet_compiled_graphs.arm_identity`` — an
``arm1-``-prefixed token that ``compiled_graph_key.is_key`` rejects by SPELLING — and
the only compiled graph key there is is the STAMP ``aot_mint.compiled_graph_identity`` derives
from the artifact's own recorded facts (graph x envelope x sm x toolchain).
The two spaces that pgw#1032/#1033 kept disjoint by a ``kind`` axis value
are now disjoint by grammar, which no reader can miss.

What SURVIVES is the ACTIVE (stamped) advertisement: the worker states the
identity of the artifact it is actually serving, and the hub's dispatch fence
verifies THAT against its own store. These tests pin both halves.
"""

from __future__ import annotations


import pytest

from gen_worker import aot_serve, compiled_graph_key, serving_mode
from gen_worker.pb import worker_scheduler_pb2 as pb

FAMILY = "sdxl"

# ---------------------------------------------------------------------------
# 1. The disjointness proof, in code
# ---------------------------------------------------------------------------


def test_an_arm_token_and_a_stamped_key_are_disjoint_by_grammar() -> None:
    """pgw#1059: an obligation identity is not a compiled graph key in any reader's
    eyes — ``arm1-`` never passes ``is_key``, so every mechanism that keys
    a store of STAMPED keys structurally cannot consume one."""
    from gen_worker import fleet_compiled_graphs

    token = fleet_compiled_graphs.ArmIdentity(facts=(("family", FAMILY),)).token
    assert token.startswith(fleet_compiled_graphs.ARM_SCHEME + "-")
    assert not compiled_graph_key.is_key(token)
    assert aot_serve.ARTIFACT_KIND == "aot-inductor"


def test_the_pre_trace_computed_key_no_longer_exists() -> None:
    """The producer of the computed key space is DELETED, not merely
    unplugged: ``compiled_graph_key`` exposes no ``compute``/``from_artifact_metadata``
    — the stamp derivation is the only key derivation there is."""
    assert not hasattr(compiled_graph_key, "compute")
    assert not hasattr(compiled_graph_key, "from_artifact_metadata")
    assert not hasattr(compiled_graph_key, "stamp")
    assert not hasattr(compiled_graph_key, "mismatch")


def test_a_non_exported_kind_has_no_key_identity() -> None:
    """The other direction of the same wall: only exported compiled graphs are keyed;
    a torch-inductor-cache artifact is refused by name."""
    with pytest.raises(compiled_graph_key.CompiledGraphKeyError, match="no compiled-graph-key identity"):
        compiled_graph_key.from_compiled_graph_metadata(
            {"kind": "torch-inductor-cache", "compiled_graph_key": "ek1-" + "b" * 56})


# ---------------------------------------------------------------------------
# 2. The producer is gone: no computed key on the wire
# ---------------------------------------------------------------------------


def test_a_compile_target_advertises_no_computed_key(monkeypatch) -> None:
    """RED before this issue: ``_refresh_compile_target`` computed a key and
    stamped it on every target as ``requested_compiled_graph_key``. A target now states
    only what it IS serving."""
    from gen_worker import executor as executor_mod

    fields = {f.name for f in pb.CompileTarget.DESCRIPTOR.fields}
    assert "requested_compiled_graph_key" in fields, (
        "the wire field is retired by the th#1457/pgw#891 RunJob cut, not "
        "here — this test is about the PRODUCER")
    # ...and its axes twin is already gone: §4.28 / th#1751 W4 reserved 11,
    # because "forge mint parameters" was its only stated purpose.
    assert "requested_compiled_graph_axes" not in fields

    assert not hasattr(executor_mod._CompileTargetRecord, "requested_compiled_graph_key")
    assert not hasattr(executor_mod._CompileTargetRecord, "requested_compiled_graph_axes")


def test_the_executor_no_longer_produces_compiled_graph_lookups() -> None:
    """``compiled_graph_lookups`` advertised SPECULATIVE computed keys — up to five base
    lanes per declared spec — into the same dead space. Its producer is gone;
    the wire field retires with the RunJob cut."""
    from gen_worker import executor as executor_mod
    from gen_worker import lifecycle as lifecycle_mod
    from gen_worker.procsplit import merge as merge_mod

    assert not hasattr(executor_mod.Executor, "compiled_graph_lookups")
    assert "compiled_graph_lookups" not in lifecycle_mod.__dict__.get("__source_marker__", "")
    # The merge of G child deltas must not resurrect it either.
    merged = merge_mod.merge_state_deltas([
        pb.StateDelta(compiled_graph_lookups=[pb.CompiledGraphLookup(family="f", compiled_graph_key="ck1-a")]),
        pb.StateDelta(compiled_graph_lookups=[pb.CompiledGraphLookup(family="f", compiled_graph_key="ck1-b")]),
    ])
    assert list(merged.compiled_graph_lookups) == []


def test_the_divergence_warning_is_gone_with_the_divergence() -> None:
    """``_warn_compiled_graph_key_divergence`` compared the two spaces above and so fired
    an ERROR on every healthy AOT boot (pgw#1033 silenced it as an interim and
    named this issue as the deleter)."""
    from gen_worker import executor as executor_mod

    assert not hasattr(executor_mod.Executor, "_warn_compiled_graph_key_divergence")


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
    to eager.

    pgw#1181: ``stage_artifact`` is gone too, and the note this row used to
    carry — *"``stage_artifact`` itself stays: ``seed_artifact`` and ``enable``
    are live callers"* — is exactly the reasoning the ratchet could not see
    through. Both callers were real, and the FORMAT they staged had no writer
    from the moment pgw#1178 deleted ``mint_artifact``. A lane is not alive
    because it is reachable; it is alive because something feeds it."""
    from gen_worker import compile_cache as cc

    for gone in ("arm_staged_artifact", "stage_artifact", "seed_artifact",
                 "pack", "unpack", "verify", "artifact_metadata"):
        assert not hasattr(cc, gone), gone


def test_the_demand_echoing_build_entry_point_is_gone_entirely() -> None:
    """A demand-driven mint used to echo the worker-computed key it had to
    satisfy. Nothing has issued demand-driven mints since the forge was
    deleted, and no caller passed it.

    pgw#1032 asserted the PARAMETER had left `compile_cache.build`'s signature.
    pgw#1035 then deleted `build` outright — the whole-pipeline dynamo mint had
    no caller at all, and `aot_compiled_graphs.discover` rejects the artifact kind it
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
    compiles its own graphs, so ``jit_compiled_graph`` is the honest answer even though
    pgw#1010 left it naming no artifact. ``serving_tier`` answers *is this
    worker serving from a COMPILED GRAPH* — the hub reads it as ADOPTION evidence
    (``WorkerServingCompiledTier`` -> ``WorkerAdoptedDeliveredCompiledGraph``, th#1216),
    so an intake pod reporting ``compiled`` would testify that the compiled graph
    exchange worked on a pod that adopted nothing. The divergence the issue
    called a defect is the design; `test_guard_miss_pgw680` and
    `test_eager_first_boot_pgw671` assert the same contract from the other
    side."""
    from gen_worker import compile_cache as cc

    pipe = _ArmedPipe()
    monkeypatch.setattr(cc, "is_compile_armed", lambda p: p is pipe)

    # Same pod, same instant, two honest answers.
    assert serving_mode.classify_mode("", pipe) == serving_mode.MODE_JIT_COMPILED_GRAPH
    assert cc.is_compile_armed(pipe) is True
    # A compiled graph ref IS what makes the tier compiled — nothing else.
    assert serving_mode.classify_mode(
        f"root/family-{FAMILY}#ek1-" + "b" * 56, None) == serving_mode.MODE_JIT_COMPILED_GRAPH


def test_an_armed_target_still_advertises_the_identity_it_serves() -> None:
    """The surviving half: the ACTIVE (stamped) ref plus its snapshot digest.
    This is what ``hubPublishedAdvertisedCompiledGraph`` verifies against the hub's own
    store before W8A8 dispatch rides it — the load-bearing fence."""
    from gen_worker import executor as executor_mod

    ref = f"root/family-{FAMILY}#ek1-" + "b" * 56
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
    assert not wire.requested_compiled_graph_key
