"""The layout converter registry (§1.33), and the fence that keeps conversion
out of cell identity.

What is proven here, in the order the ruling puts it:

1. **re-quantization is structurally unregistrable** — the admission proof, run
   at registration against a REAL safetensors shard, refuses a lossy transform
   and names the rung it belongs on instead;
2. a lossless topology edge registers, both directions, and cannot have touched
   payload bytes;
3. the ladder's four rungs come out of ONE relation — membership, then lossless
   reachability, then production reachability — with no format knowledge in it;
4. the accepted set is a FILTER: neither the planner nor the declaration lets
   its order become a preference;
5. the derived-artifact identity is stable across PROCESSES, which is the whole
   basis of convert-once-into-the-CAS;
6. **no cell re-keys**: no cell-key axis can read the layout vocabulary
   (structural, via the shipped gate) and a slot's demand does not move the
   compile contract (behavioural, through `extract_specs`).

Run: uv run pytest tests/test_layout_converter_registry_pgw1143.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import msgspec
import pytest

from gen_worker import RequestContext, Slot, endpoint
from gen_worker.api.decorators import Compile
from gen_worker.api.derive import contract_delta, override_delta
from gen_worker.api.export_contract import Dim, Fork, GraphClass
from gen_worker.convert.layout_converters import (
    ConversionCase,
    ConversionProofError,
    CorpusTensor,
    LayoutProduction,
    LayoutRung,
    QuantRepack,
    TopologyConversion,
    classify_layout,
    conversion_provenance,
    derived_artifact_identity,
    materialize_case,
    plan_layout_conversions,
    register_layout_conversion,
    register_layout_production,
    registered_layout_conversions,
    reset_layout_conversions,
    run_layout_conversion,
)
from gen_worker.convert.repack_spec import DeclarationError, RenameRule
from gen_worker.convert.writer import (
    rewrite_safetensors_keys,
    shard_payload_digests,
    shard_tensor_entries,
)
from gen_worker.families.base import GenerationDefaults
from gen_worker.models.tensor_layout_contract import (
    CONTRACT_COZY_FP8_ROWWISE,
    CONTRACT_HF_FP8_BLOCKWISE,
    CONTRACT_PLAIN_BF16,
    TOPOLOGY_COMFY_SPLITFILES,
    TOPOLOGY_DIFFUSERS_MULTIFILE,
    TOPOLOGY_DIFFUSERS_SINGLEFILE,
    LayoutId,
    normalize_layout_demand,
    parse_layout_id,
)
from gen_worker.registry import extract_specs

REPO = Path(__file__).resolve().parents[1]

# A real DiT shard's key shape, at header granularity (§4.25 corpus method).
DIT_CASE = ConversionCase(
    name="dit-block0",
    tensors={
        "model.diffusion_model.blocks.0.attn.to_q.weight":
            CorpusTensor(dtype="BF16", shape=(8, 8)),
        "model.diffusion_model.blocks.0.attn.to_k.weight":
            CorpusTensor(dtype="BF16", shape=(8, 8)),
        "model.diffusion_model.blocks.0.ff.net.0.proj.weight":
            CorpusTensor(dtype="BF16", shape=(16, 8)),
    },
    metadata={"format": "pt"},
)


@pytest.fixture(autouse=True)
def _bare_registry() -> Iterator[None]:
    """The wheel ships this registry EMPTY; every test declares its
    own edges the way an endpoint does, and leaves it bare again."""
    reset_layout_conversions()
    yield
    reset_layout_conversions()


def _comfy_to_diffusers(version: int = 1) -> TopologyConversion:
    return TopologyConversion(
        from_id=TOPOLOGY_COMFY_SPLITFILES,
        to_id=TOPOLOGY_DIFFUSERS_MULTIFILE,
        version=version,
        rules=(RenameRule(
            kind="prefix", pairs=(("model.diffusion_model.", "transformer."),)),),
        inverse_rules=(RenameRule(
            kind="prefix", pairs=(("transformer.", "model.diffusion_model."),)),),
        corpus=(DIT_CASE,),
        why="ComfyUI names the DiT under model.diffusion_model.*",
    )


# ── 1. the admission bar: re-quantization cannot enter this registry ─────────


def _bf16_to_fp8(io: object) -> None:
    """A REAL re-quantization: drop the low byte of every BF16 element.

    Deliberately the cheapest honest one — it is not a good fp8 cast, it is a
    transform that loses information, which is the property the admission bar
    is about.
    """
    for key in io.keys():  # type: ignore[attr-defined]
        spec = io.spec(key)  # type: ignore[attr-defined]
        raw = io.read(key)  # type: ignore[attr-defined]
        io.emit(  # type: ignore[attr-defined]
            key, dtype="F8_E4M3", shape=spec.shape, payload=raw[1::2])


def _fp8_to_bf16(io: object) -> None:
    """The inverse anyone would write: widen back, zero-filling what was lost."""
    for key in io.keys():  # type: ignore[attr-defined]
        spec = io.spec(key)  # type: ignore[attr-defined]
        raw = io.read(key)  # type: ignore[attr-defined]
        widened = bytearray()
        for byte in raw:
            widened.extend((0, byte))
        io.emit(  # type: ignore[attr-defined]
            key, dtype="BF16", shape=spec.shape, payload=bytes(widened))


def _always_equivalent(_a: Path, _b: Path) -> None:
    """An author's optimistic dequant check — passes unconditionally.

    Present on purpose: the round trip must refuse the cast even when the
    mapping's OWN equivalence obligation says everything is fine. An admission
    bar that only refuses what the author already admits is lossy is no bar.
    """


def test_a_re_quantization_cannot_be_registered_as_a_converter() -> None:
    with pytest.raises(ConversionProofError) as excinfo:
        register_layout_conversion(QuantRepack(
            from_id=CONTRACT_PLAIN_BF16,
            to_id=CONTRACT_COZY_FP8_ROWWISE,
            version=1,
            forward=_bf16_to_fp8,
            inverse=_fp8_to_bf16,
            equivalence=_always_equivalent,
            corpus=(DIT_CASE,),
        ))
    message = str(excinfo.value)
    assert "round trip did not recover" in message
    # The refusal is USEFUL: it names the rung this transform belongs on and
    # the call that declares it, so the author's next move is written down.
    assert "PRODUCIBLE" in message
    assert "register_layout_production()" in message
    assert registered_layout_conversions() == (), (
        "a mapping that failed its admission proof must leave NO edge behind — "
        "the registry cannot hold a lossy transform, which is what makes the "
        "CONVERTIBLE rung mean something")


def test_the_same_transform_is_declarable_as_a_production_and_rung_producible() -> None:
    """The rung a refused converter belongs on — priced, offered, never a
    transform: `LayoutProduction` carries a recipe NAME and a quality gate."""
    register_layout_production(LayoutProduction(
        axis="quant",
        from_id=CONTRACT_PLAIN_BF16,
        to_id=CONTRACT_COZY_FP8_ROWWISE,
        recipe="quantize-fp8-rowwise",
        quality_gate="numerics-parity",
    ))
    verdict = classify_layout(
        LayoutId(quant=CONTRACT_PLAIN_BF16),
        [LayoutId(quant=CONTRACT_COZY_FP8_ROWWISE)],
    )
    assert verdict.rung is LayoutRung.PRODUCIBLE
    assert verdict.productions[0].recipe == "quantize-fp8-rowwise"
    assert not hasattr(verdict.productions[0], "apply"), (
        "a production must carry no transform; an `apply` here is how a "
        "priced quality decision becomes automatic")


def test_a_production_must_name_the_gate_that_will_judge_it() -> None:
    with pytest.raises(DeclarationError, match="quality_gate is empty"):
        register_layout_production(LayoutProduction(
            axis="quant", from_id=CONTRACT_PLAIN_BF16,
            to_id=CONTRACT_COZY_FP8_ROWWISE,
            recipe="quantize-fp8-rowwise", quality_gate=""))


# ── 2. a lossless topology edge: data, both directions, bytes untouched ──────


def test_a_topology_rename_registers_both_directions() -> None:
    register_layout_conversion(_comfy_to_diffusers())
    edges = {(e.from_id, e.to_id) for e in registered_layout_conversions()}
    assert edges == {
        (TOPOLOGY_COMFY_SPLITFILES, TOPOLOGY_DIFFUSERS_MULTIFILE),
        (TOPOLOGY_DIFFUSERS_MULTIFILE, TOPOLOGY_COMFY_SPLITFILES),
    }, ("a lossless mapping is invertible and the inverse was proven in the "
        "same pass; withholding it would be an edge we know is safe and refuse "
        "to use")


def test_a_topology_edge_with_no_inverse_is_refused_at_declaration() -> None:
    with pytest.raises(DeclarationError, match="inverse_rules="):
        TopologyConversion(
            from_id=TOPOLOGY_COMFY_SPLITFILES,
            to_id=TOPOLOGY_DIFFUSERS_MULTIFILE,
            version=1,
            rules=(RenameRule(kind="prefix", pairs=(("a.", "b."),)),),
            inverse_rules=(),
            corpus=(DIT_CASE,),
        )


def test_a_rename_that_collides_two_keys_is_refused() -> None:
    """An unmapped key is a refusal and an invented key is worse — and a rename
    that maps two source keys onto one is the silent-data-loss version of both."""
    with pytest.raises(ConversionProofError, match="not injective|source keys"):
        register_layout_conversion(TopologyConversion(
            from_id=TOPOLOGY_COMFY_SPLITFILES,
            to_id=TOPOLOGY_DIFFUSERS_MULTIFILE,
            version=1,
            rules=(RenameRule(
                kind="substring", pairs=(("to_q", "to_x"), ("to_k", "to_x"))),),
            inverse_rules=(RenameRule(
                kind="substring", pairs=(("to_x", "to_q"),)),),
            corpus=(DIT_CASE,),
        ))


def test_the_byte_rewriter_refuses_a_non_injective_map(tmp_path: Path) -> None:
    """`rewrite_safetensors_keys` is the ENGINE, and th#1809 T6 / cozy-local
    will drive it WITHOUT the registration proof around it. So its own refusals
    are tested directly — the delete-the-call-site experiment showed the proof's
    key-count check was masking this one, which would have left the public
    primitive's only safety check untested.
    """
    source = materialize_case(DIT_CASE, tmp_path / "src.safetensors")
    keys = [name for name, _ in shard_tensor_entries(source)]
    collide = {k: "one.key.for.all" for k in keys}
    with pytest.raises(ValueError, match="not injective"):
        rewrite_safetensors_keys(source, tmp_path / "out.safetensors", collide)


def test_the_byte_rewriter_refuses_a_partial_map(tmp_path: Path) -> None:
    """An unmapped key is a REFUSAL, never a silent passthrough: a partial
    rename produces a file that loads as neither layout."""
    source = materialize_case(DIT_CASE, tmp_path / "src.safetensors")
    keys = [name for name, _ in shard_tensor_entries(source)]
    partial = {keys[0]: "renamed.only.this"}
    with pytest.raises(ValueError, match="no mapping"):
        rewrite_safetensors_keys(source, tmp_path / "out.safetensors", partial)


def test_the_conversion_engine_never_touches_a_payload_byte(tmp_path: Path) -> None:
    register_layout_conversion(_comfy_to_diffusers())
    source = materialize_case(DIT_CASE, tmp_path / "src.safetensors")
    plan = plan_layout_conversions(
        LayoutId(topology=TOPOLOGY_COMFY_SPLITFILES),
        [LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE)],
    )[0]
    result = run_layout_conversion(
        plan, source, tmp_path / "out.safetensors",
        source_digest="sha256:source", produced_by="test")

    before = sorted(shard_payload_digests(source).values())
    after = sorted(shard_payload_digests(result.path).values())
    assert before == after
    assert [name for name, _ in shard_tensor_entries(result.path)] == [
        "transformer.blocks.0.attn.to_q.weight",
        "transformer.blocks.0.attn.to_k.weight",
        "transformer.blocks.0.ff.net.0.proj.weight",
    ]


def test_a_converted_artifact_carries_its_own_chain(tmp_path: Path) -> None:
    """Artifacts self-describe: anyone can re-derive the identity from the file
    without asking the hub (th#1721 extension 4 stage 3)."""
    register_layout_conversion(_comfy_to_diffusers())
    source = materialize_case(DIT_CASE, tmp_path / "src.safetensors")
    plan = plan_layout_conversions(
        LayoutId(topology=TOPOLOGY_COMFY_SPLITFILES),
        [LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE)],
    )[0]
    result = run_layout_conversion(
        plan, source, tmp_path / "out.safetensors",
        source_digest="sha256:source", produced_by="local_conversion")

    chain = conversion_provenance(result.path)
    assert chain is not None
    assert chain["identity"] == result.identity
    assert chain["produced_by"] == "local_conversion"
    assert [(h["from"], h["to"]) for h in chain["chain"]] == [
        (TOPOLOGY_COMFY_SPLITFILES, TOPOLOGY_DIFFUSERS_MULTIFILE)]
    assert derived_artifact_identity(
        "sha256:source",
        [h["converter_digest"] for h in chain["chain"]],
        LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE),
    ) == result.identity


def test_a_conversion_that_cannot_say_where_it_ran_is_refused(
    tmp_path: Path,
) -> None:
    """§4.28's never-upload fence needs `produced_by` to exist on every derived
    artifact; a conversion that omits it produces bytes the publish path cannot
    judge."""
    register_layout_conversion(_comfy_to_diffusers())
    source = materialize_case(DIT_CASE, tmp_path / "src.safetensors")
    plan = plan_layout_conversions(
        LayoutId(topology=TOPOLOGY_COMFY_SPLITFILES),
        [LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE)],
    )[0]
    with pytest.raises(DeclarationError, match="produced_by"):
        run_layout_conversion(
            plan, source, tmp_path / "out.safetensors",
            source_digest="sha256:source", produced_by="")


# ── 3. the relation: composition, the cost cap, the four rungs ───────────────


def _singlefile_edge() -> TopologyConversion:
    return TopologyConversion(
        from_id=TOPOLOGY_DIFFUSERS_MULTIFILE,
        to_id=TOPOLOGY_DIFFUSERS_SINGLEFILE,
        version=1,
        rules=(RenameRule(kind="prefix", pairs=(("transformer.", "model."),)),),
        inverse_rules=(RenameRule(kind="prefix", pairs=(("model.", "transformer."),)),),
        corpus=(ConversionCase(
            name="multifile-block0",
            tensors={"transformer.blocks.0.attn.to_q.weight":
                     CorpusTensor(dtype="BF16", shape=(8, 8))},
        ),),
    )


def test_two_edges_compose_into_one_plan() -> None:
    """Composition is what makes N+M registrations cover what a composite
    vocabulary would need N*M for — and lossless∘lossless is lossless by
    construction, so it needs no separate proof."""
    register_layout_conversion(_comfy_to_diffusers())
    register_layout_conversion(_singlefile_edge())
    plans = plan_layout_conversions(
        LayoutId(topology=TOPOLOGY_COMFY_SPLITFILES),
        [LayoutId(topology=TOPOLOGY_DIFFUSERS_SINGLEFILE)],
    )
    assert len(plans) == 1
    assert [(h.from_id, h.to_id) for h in plans[0].hops] == [
        (TOPOLOGY_COMFY_SPLITFILES, TOPOLOGY_DIFFUSERS_MULTIFILE),
        (TOPOLOGY_DIFFUSERS_MULTIFILE, TOPOLOGY_DIFFUSERS_SINGLEFILE),
    ]


def test_a_three_hop_need_is_refused_rather_than_silently_paid_for() -> None:
    """The cap is a COST bound: each hop is a full weight rewrite, so a 3-hop
    need means a missing direct edge, which the planner says rather than
    charging for."""
    register_layout_conversion(_comfy_to_diffusers())
    register_layout_conversion(_singlefile_edge())
    register_layout_conversion(TopologyConversion(
        from_id=TOPOLOGY_DIFFUSERS_SINGLEFILE,
        to_id="transformers.native@1",
        version=1,
        rules=(RenameRule(kind="prefix", pairs=(("model.", "text_model."),)),),
        inverse_rules=(RenameRule(kind="prefix", pairs=(("text_model.", "model."),)),),
        corpus=(ConversionCase(
            name="singlefile-block0",
            tensors={"model.blocks.0.attn.to_q.weight":
                     CorpusTensor(dtype="BF16", shape=(8, 8))},
        ),),
    ))
    verdict = classify_layout(
        LayoutId(topology=TOPOLOGY_COMFY_SPLITFILES),
        [LayoutId(topology="transformers.native@1")],
    )
    assert verdict.rung is LayoutRung.INCOMPATIBLE
    assert "no registered lossless mapping reaches" in verdict.reason


@pytest.mark.parametrize("supply, accepts, rung", [
    # in the set
    (LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE),
     [LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE)], LayoutRung.COMPATIBLE),
    # a registered lossless edge reaches the set
    (LayoutId(topology=TOPOLOGY_COMFY_SPLITFILES),
     [LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE)], LayoutRung.CONVERTIBLE),
    # nothing reaches it
    (LayoutId(topology="gguf.native@1"),
     [LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE)], LayoutRung.INCOMPATIBLE),
    # an UNDECLARED demand is not "accepts everything"
    (LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE), [], LayoutRung.INCOMPATIBLE),
])
def test_the_ladder(supply: LayoutId, accepts: list, rung: LayoutRung) -> None:
    register_layout_conversion(_comfy_to_diffusers())
    assert classify_layout(supply, accepts).rung is rung


def test_the_axes_are_compared_field_wise_not_as_one_string() -> None:
    """A whole-string compare is exact and still cannot express "topology
    differs, quant matches", which is the CONVERTIBLE rung."""
    register_layout_conversion(_comfy_to_diffusers())
    verdict = classify_layout(
        LayoutId(topology=TOPOLOGY_COMFY_SPLITFILES, quant=CONTRACT_PLAIN_BF16),
        [LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE, quant=CONTRACT_PLAIN_BF16)],
    )
    assert verdict.rung is LayoutRung.CONVERTIBLE
    # The quant axis needed no hop; only the topology one did.
    assert verdict.plans[0].quant == ()
    assert len(verdict.plans[0].topology) == 1


def test_a_quant_mismatch_is_not_convertible_on_a_topology_edge() -> None:
    register_layout_conversion(_comfy_to_diffusers())
    verdict = classify_layout(
        LayoutId(topology=TOPOLOGY_COMFY_SPLITFILES, quant=CONTRACT_PLAIN_BF16),
        [LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE,
                  quant=CONTRACT_HF_FP8_BLOCKWISE)],
    )
    assert verdict.rung is LayoutRung.INCOMPATIBLE


def test_an_axis_neither_side_declares_is_reported_not_assumed() -> None:
    """UNDECLARED is a different fact from agnostic, and the verdict says which
    axes it did not decide instead of reading silence as agreement."""
    verdict = classify_layout(
        LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE),
        [LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE)],
    )
    assert verdict.rung is LayoutRung.COMPATIBLE
    assert verdict.unevaluated_axes == ("quant",)


def test_the_relation_holds_no_handle_literal() -> None:
    """§1.33's extensibility invariant: contract CONTENTS evolve, the
    compatibility RELATION never does. Enforced by the shipped gate, asserted
    here so the invariant has a test and not only a script."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "lint_compiled_graph_key_layout_fence.py")],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no contract-handle literal in the relation" in proc.stdout


# ── 4. the accepted set is a FILTER; its order is not a preference ───────────


def test_the_declared_set_is_canonicalized_so_no_reader_can_recover_an_order(
) -> None:
    """§1.33 point 2 as AMENDED: the set is a compatibility filter and
    preference has exactly ONE authority — the (GPU, lane) ladder. Two authors
    who spell the same set differently state the SAME demand, and storing the
    written order would be a second ordering that can disagree with the first.
    """
    first = normalize_layout_demand(
        {"*": (CONTRACT_HF_FP8_BLOCKWISE, CONTRACT_PLAIN_BF16)}, where="a")
    second = normalize_layout_demand(
        {"*": (CONTRACT_PLAIN_BF16, CONTRACT_HF_FP8_BLOCKWISE)}, where="b")
    assert first == second


def test_the_planner_returns_every_reachable_target_and_ranks_none() -> None:
    register_layout_conversion(_comfy_to_diffusers())
    register_layout_conversion(_singlefile_edge())
    supply = LayoutId(topology=TOPOLOGY_COMFY_SPLITFILES)
    accepts = [LayoutId(topology=TOPOLOGY_DIFFUSERS_SINGLEFILE),
               LayoutId(topology=TOPOLOGY_DIFFUSERS_MULTIFILE)]
    forward = plan_layout_conversions(supply, accepts)
    reversed_ = plan_layout_conversions(supply, list(reversed(accepts)))
    assert len(forward) == 2
    assert [p.target for p in forward] == [p.target for p in reversed_], (
        "the plan order must not follow the demand's order — that is the second "
        "ordering §1.33 point 2 forbids")


# ── 5. one identity function, stable across processes ────────────────────────


def test_the_derived_identity_is_identical_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    """Convert-once-into-the-CAS is worth nothing if the hub and a laptop
    compute different identities for the same conversion — which is exactly
    what an mtime- or path-dependent code digest would do."""
    script = tmp_path / "identity.py"
    script.write_text(textwrap.dedent(
        """
        import json
        from gen_worker.convert.layout_converters import (
            ConversionCase, CorpusTensor, TopologyConversion,
            register_layout_conversion, registered_layout_conversions)
        from gen_worker.convert.repack_spec import RenameRule
        from gen_worker.models.tensor_layout_contract import (
            TOPOLOGY_COMFY_SPLITFILES, TOPOLOGY_DIFFUSERS_MULTIFILE)

        register_layout_conversion(TopologyConversion(
            from_id=TOPOLOGY_COMFY_SPLITFILES,
            to_id=TOPOLOGY_DIFFUSERS_MULTIFILE,
            version=1,
            rules=(RenameRule(kind="prefix",
                   pairs=(("model.diffusion_model.", "transformer."),)),),
            inverse_rules=(RenameRule(kind="prefix",
                   pairs=(("transformer.", "model.diffusion_model."),)),),
            corpus=(ConversionCase(
                name="dit-block0",
                tensors={
                  "model.diffusion_model.blocks.0.attn.to_q.weight":
                      CorpusTensor(dtype="BF16", shape=(8, 8)),
                  "model.diffusion_model.blocks.0.attn.to_k.weight":
                      CorpusTensor(dtype="BF16", shape=(8, 8)),
                  "model.diffusion_model.blocks.0.ff.net.0.proj.weight":
                      CorpusTensor(dtype="BF16", shape=(16, 8)),
                },
                metadata={"format": "pt"}),),
        ))
        print(json.dumps({
            (e.from_id + "->" + e.to_id): e.digest
            for e in registered_layout_conversions()}))
        """
    ), encoding="utf-8")

    register_layout_conversion(_comfy_to_diffusers())
    here = {f"{e.from_id}->{e.to_id}": e.digest
            for e in registered_layout_conversions()}

    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True,
        timeout=300, cwd=str(REPO))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout.strip()) == here


@pytest.mark.parametrize("rendered", [
    "plain.bf16@1",                          # bare handle = the quant axis
    "+plain.bf16@1",
    "diffusers.multifile@1+plain.bf16@1",
    "diffusers.multifile@1+",
    "any+plain.bf16@1",                      # a DECLARED agnostic, not an inference
])
def test_a_layout_id_survives_its_own_rendering(rendered: str) -> None:
    """`render()` is what `derived_artifact_identity` digests, so a rendering
    that does not parse back to the same pair would let two different LayoutIds
    address one CAS object — silently, on the axis nobody printed."""
    parsed = parse_layout_id(rendered, where="t")
    assert parse_layout_id(parsed.render(), where="t") == parsed


def test_a_version_bump_moves_the_identity(tmp_path: Path) -> None:
    """Bumping `version` changes the converter digest, which changes every
    derived artifact's identity — bytes never silently change under a name."""
    register_layout_conversion(_comfy_to_diffusers(version=1))
    v1 = {(e.from_id, e.to_id): e.digest for e in registered_layout_conversions()}
    reset_layout_conversions()
    register_layout_conversion(_comfy_to_diffusers(version=2))
    v2 = {(e.from_id, e.to_id): e.digest for e in registered_layout_conversions()}
    assert set(v1) == set(v2)
    assert all(v1[k] != v2[k] for k in v1)


# ── 6. the §1.33 point-5 fence: conversion is upstream of compute ────────────


def _fence_module():
    spec = importlib.util.spec_from_file_location(
        "_fence", REPO / "scripts" / "lint_compiled_graph_key_layout_fence.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_fence_covers_every_module_that_computes_a_compiled_graph_key() -> None:
    """The fenced set is DERIVED from who calls an axis producer, not
    hand-listed — the failure mode of a hand-maintained list is that the one
    file which violates the rule is the one nobody added."""
    fenced = {p.name for p in _fence_module().fenced_modules()}
    # pgw#1277: the key's DEFINITION left this tree for
    # torchcg.identity, so no module here defines it and the
    # fence covers exactly the CALL SITES plus the axis-input seeds.
    # pgw#1327: `boot_key.py` dropped OUT because the boot fold moved to
    # `keyset/fold.py` — the fence is derived from who calls an axis producer,
    # so it followed the arithmetic without anyone editing a list. That is the
    # property this test exists to keep.
    assert {"fleet_cells.py", "fold.py", "aot_mint.py",
            "compile_cache.py"} <= fenced
    assert "boot_key.py" not in fenced, (
        "the tracer states declarations; it no longer folds a key")
    assert "aot_serve.py" not in fenced, (
        "runtime admission consumes the TCG key; it must not derive one")


def _run_fence(*argv: str) -> subprocess.CompletedProcess:
    """The gate exactly as CI runs it: the script, through its own `main()`."""
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "lint_compiled_graph_key_layout_fence.py"),
         *argv],
        capture_output=True, text=True, timeout=180)


def test_the_fence_fires_on_a_key_axis_that_reads_the_layout(
    tmp_path: Path,
) -> None:
    """THE GATE'S OWN RED, through the ENTRY POINT CI invokes.

    Deliberately not a call to `_violations()`. The delete-the-call-site
    experiment on this very file found that disconnecting the detector from
    `main()` left every test green — the th#1820 shape, where a function has
    a dozen tests and its only call site has none. So this drives the SCRIPT
    against a tree that violates the fence, and a disconnected detector fails
    here.
    """
    (tmp_path / "graph_facts.py").write_text(textwrap.dedent(
        """
        from gen_worker.convert.layout_converters import LayoutId


        def axis(block, layout: LayoutId):
            # calls an axis producer, so the fence COVERS this module by
            # derivation rather than by being named in the seed list
            return toolchain_axis_digest({"v": 1, "layout": layout.render()})
        """
    ), encoding="utf-8")
    result = _run_fence("--src", str(tmp_path))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "fence BROKEN" in result.stdout
    assert "layout_converters" in result.stdout
    # The refusal must tell the author what to do instead of widening the key.
    assert "Conversion is UPSTREAM of compute" in result.stdout


def test_the_fence_fires_on_a_deferred_import_too(tmp_path: Path) -> None:
    """A string handed to `import_module` is the obvious way around an import
    check, so it is checked as a string."""
    (tmp_path / "graph_facts.py").write_text(textwrap.dedent(
        """
        import importlib


        def axis(block):
            mod = importlib.import_module("gen_worker.convert.layout_converters")
            return toolchain_axis_digest({"v": 1, "chain": mod.conversion_provenance})
        """
    ), encoding="utf-8")
    result = _run_fence("--src", str(tmp_path))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "string 'gen_worker.convert.layout_converters'" in result.stdout


def test_the_fence_does_not_fire_on_prose(tmp_path: Path) -> None:
    """Docstrings are excluded on purpose: a vocabulary gate that reds on the
    word appearing in an explanation teaches lanes to stop explaining
    themselves, and has already cost this repo a lane-day."""
    (tmp_path / "graph_facts.py").write_text(textwrap.dedent(
        '''
        """The cell key. Deliberately blind to Slot.layouts and to any
        LayoutId or conversion chain — see classify_layout for why."""


        def axis(block):
            return toolchain_axis_digest({"v": 1})
        '''
    ), encoding="utf-8")
    result = _run_fence("--src", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_shipped_tree_passes_the_fence() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "lint_compiled_graph_key_layout_fence.py")],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no cell re-keys on a conversion" in proc.stdout


# ── the behavioural half of point 5: a demand does not move the contract ─────


class _Vae:
    pass


class _TextEncoder:
    pass


class _Unet:
    pass


class FakePipeline:
    def __init__(self, vae: _Vae, text_encoder: _TextEncoder, unet: _Unet):
        self.vae = vae
        self.text_encoder = text_encoder
        self.unet = unet

    @classmethod
    def _get_signature_keys(cls, _obj: object) -> tuple:
        return {"vae", "text_encoder", "unet"}, set()


class _In(msgspec.Struct):
    prompt: str = ""
    model: str = ""


class _Out(msgspec.Struct):
    ok: bool = True


class _Defaults(GenerationDefaults):
    pass


def _compile_declaration() -> Compile:
    return Compile(
        family="layoutfence",
        targets=("unet",),
        text_len=512,
        dims=(Dim("H", carried_by=(("hidden_states", 2),), multiple_of=2),
              Dim("B", carried_by=(("hidden_states", 0),))),
        forks=(Fork("cfg", served=(False,), unserved=(True,),
                     reason="default_value"),),
        classes=(GraphClass(dims={"H": 90, "B": 1}, fork={"cfg": False}),),
        shape_strategy="dynamic-collapse",
        warm_changes_key=False,
    )


def _spec_for(slot: Slot):
    @endpoint(models={"pipeline": slot}, compile=_compile_declaration())
    class Endpoint:
        def setup(self, pipeline: FakePipeline) -> None:
            self.pipeline = pipeline

        def generate(self, ctx: RequestContext[_Defaults], p: _In) -> _Out:
            return _Out()

    return extract_specs(Endpoint)[0]


def test_declaring_a_layout_demand_does_not_move_the_compile_contract() -> None:
    """§1.33 point 5, behaviourally: the demand is upstream of compute, so
    adding it to a slot must not move ANY cell-key input. The compile contract
    digest is what an endpoint's declaration actually reaches, and this goes RED
    the moment someone folds `layouts` into `contract_facts()` — which is the
    placement §1.33 point 5 rules out and the reason the demand lives on `Slot`
    and not on `Compile`.
    """
    bare = _spec_for(Slot(FakePipeline, selected_by="model"))
    declared = _spec_for(Slot(
        FakePipeline, selected_by="model",
        layouts={"*": (CONTRACT_PLAIN_BF16,),
                 "text_encoder": (CONTRACT_HF_FP8_BLOCKWISE, CONTRACT_PLAIN_BF16)},
    ))
    assert declared.slots["pipeline"].layouts is not None, (
        "the fixture must actually declare something, or this test passes for "
        "the wrong reason")
    assert bare.slots["pipeline"].layouts is None
    assert bare.compile is not None and declared.compile is not None
    assert declared.compile.contract_axes() == bare.compile.contract_axes()
    assert contract_delta(bare.compile, declared.compile) == {}
    assert override_delta(bare.compile, declared.compile) == {}
