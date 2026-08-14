"""pgw#1215 step 4 (th#1834 Phase 3): the serving parent supervises its own
compile children.

Successor to `test_mint_delegate_pgw784.py`. pgw#784's three claims survive
the reroute verbatim — they were never about the middle tier — and are kept
here against the process that now makes them. What is NOT ported is the
`build_cell` retry loop's own coverage: that loop is deleted, and its
replacement (the accretion loop) has the opposite property, which is asserted
below as `test_a_retry_CONSUMES_what_the_first_attempt_already_packed`.

pgw#784's original header, still the frame:

Three claims, each with the fact that would break it:

1. **The live pipeline is never armed for a delegated mint.** In the
   in-process shape the serving pipe carries guarded wrappers, LoRA branch
   containers and a process-global ``TORCHINDUCTOR_CACHE_DIR`` move for the
   whole mint; delegated, it carries none of that and keeps serving plain
   eager. (The in-process shape is gone; delegated is the only shape a mint
   has.)
2. **The child exports the declaration the PARENT stated.** The parent states
   the compile contract on the wire because the class-scoped unions live on
   the spec, not the decorator: a child re-deriving from ``@endpoint`` alone
   would export a different declaration than the parent asked for. (This is NOT
   key parity: the child computes no key; the parent stamps one from the
   returned envelope. What the wire owes is the DECLARED EXPORT, and that is
   what is checked below.)
3. **Failure inversion.** A dead mint process is a FAILED MINT reported by a
   LIVE worker. Every branch returns a typed result, the retry is bounded and
   class-driven, and a card with no room for a co-resident child DECLINES
   without spawning anything.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

from gen_worker import activity as activity_mod
from gen_worker import aot_compile_pool, aot_mint
from gen_worker import compile_cache as cc
from gen_worker import fleet_cells, mint_supervisor, mint_workers
from gen_worker.api.decorators import DynamicDim
from gen_worker.registry import CompileCell
from gen_worker.cell_adopt import AdoptOutcome

GIB = 1 << 30
def _cfg() -> CompileCell:
    return CompileCell(
        shapes=((1024, 1024), (832, 1216)),
        targets=("unet",),
        family="sdxl",
        regional=False,
        text_len=77,
        dynamic=(DynamicDim(dim="batch", min=2, max=8),),
        lora_bucket=64,
        guidance_scales=(5.0, 7.5),
        text_lens=(77, 226),
    )


def _fake_card(
    monkeypatch: pytest.MonkeyPatch, *, total_gib: float, resident_gib: float,
) -> None:
    total, resident = int(total_gib * GIB), int(resident_gib * GIB)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda dev=0: (total - resident, total))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev=0: resident)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev=0: resident)
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated",
        lambda dev=0: resident + (1 * GIB))


# ------------------------------------------- 1. the live pipe is untouched

def test_the_supervised_arm_never_touches_the_live_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    armed: List[str] = []
    monkeypatch.setattr(
        cc, "arm_jit_intake", lambda *a, **k: armed.append("arm_jit_intake"))
    # WHICH recipe a miss runs has its own coverage; a test box
    # registers no export declarations, so state the answer this test is about.
    monkeypatch.setattr(
        fleet_cells, "mint_recipe", lambda *a, **k: fleet_cells.RECIPE_AOT)
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda *a, **k: AdoptOutcome.miss("no_cell"))
    monkeypatch.setattr(cc, "has_compile_target", lambda *a, **k: True)
    monkeypatch.setattr(cc, "mandatory_serving", lambda pipe: False)
    monkeypatch.setattr(cc, "apply_lora_execution_lane", lambda pipe, bucket, **_kw: True)
    monkeypatch.setattr(cc, "drop_lora_execution_lane", lambda pipe: True)
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "fp8")
    monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)
    # No CUDA on this box, so the real sm axis is unavailable; the key itself
    # is not what this test is about.
    monkeypatch.setattr(
        fleet_cells, "arm_identity",
        lambda *a, **k: SimpleNamespace(
            token="arm1-test", facts_dict=lambda: {}))

    prior = dict(os.environ)
    outcome = fleet_cells.enable_compiled(
        SimpleNamespace(), _cfg(), tmp_path, None, None, delegate=True)

    assert armed == [], "a delegated mint must not arm the serving pipeline"
    assert not outcome.armed, (
        "armed=False is the honest answer — this pipe serves EAGER while the "
        "child compiles")
    pending = outcome.self_mint
    assert pending is not None and pending.delegated
    assert pending.arm_token and pending.target.name.endswith(".tar.gz")
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        assert os.environ.get(key) == prior.get(key), (
            "the process-global inductor cache dir must not move: the child "
            "owns the capture (gw#608's root cause, avoided by construction)")


# --------------------------------------------- 2. one key, stated by one side

def test_the_wire_form_preserves_the_declared_export_exactly() -> None:
    """The load-bearing check: the ``ExportSpec`` the CHILD derives from the
    wire form must equal the one the parent's own ``CompileCell`` derives.

    That — not ``contract_facts()`` — is what the wire owes. The child's single
    consumer of the cfg is ``fleet_cells.aot_export_spec`` (plus
    ``resolve_targets``, which reads ``targets``), and a dropped union or a
    lost shape row silently exports a different declaration than the parent
    asked for. pgw#1034 also asserts the converse: a field with no child reader
    does not cross the wire.
    """
    parent = _cfg()
    wire = mint_supervisor.cfg_spec(parent)
    pipe = SimpleNamespace()

    assert (fleet_cells.aot_export_spec(pipe, wire)
            == fleet_cells.aot_export_spec(pipe, parent))
    assert cc.resolve_targets(pipe, wire) == cc.resolve_targets(pipe, parent)

    # No child reads these, so they do not ride.
    for dead in ("regional", "text_len", "dynamic"):
        assert not hasattr(wire, dead), dead


# --------------------------------------------------- 3. failure inversion


def _task(tmp_path: Path, **over: Any) -> mint_supervisor.MintTask:
    pending = fleet_cells.PendingSelfMint(
        family="sdxl", arm_token="ck1-abc",
        ref="root/family-sdxl#cg-key-v1-abc", cfg=_cfg(),
        target=tmp_path / "cell.tar.gz",
        mint_root=tmp_path / "root", publisher=None, cache_dir=tmp_path)
    pending.mint_root.mkdir(parents=True, exist_ok=True)
    fields: Dict[str, Any] = dict(
        pending=pending, pipe=SimpleNamespace(), function="gen",
        modules=("harness.toy_endpoints",), weight_lane="fp8", device=0,
        handler_proof="resident warm forward 'gen' (real weights)")
    fields.update(over)
    return mint_supervisor.MintTask(**fields)


class _Act:
    """Just enough Activity to record what the hub would have seen."""

    def __init__(self) -> None:
        self.phases: List[str] = []
        self.notes: List[str] = []

    def phase(self, phase: str, step: int = 0, total: int = 0) -> None:
        self.phases.append(phase)

    def note(self, detail: str) -> None:
        self.notes.append(detail)

    def heartbeat(self) -> None:
        pass


def _events(monkeypatch: pytest.MonkeyPatch) -> List[tuple]:
    seen: List[tuple] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, phase="", **_kw: seen.append((kind, phase, detail)))
    monkeypatch.setattr(
        fleet_cells.activity_mod, "emit_event",
        lambda kind, detail, phase="", **_kw: seen.append((kind, phase, detail)))
    return seen


def _stub_parent_gates(
    monkeypatch: pytest.MonkeyPatch, *, declared: int = 2,
) -> None:
    """The three reads the supervisor makes off the LIVE pipeline.

    A test box composes no pipeline, so the enumeration is stated. The gates
    themselves have their own tests below; this is the seam every driver test
    needs and none of them is about.
    """
    monkeypatch.setattr(
        mint_supervisor, "assert_family_mintable", lambda family: None)
    monkeypatch.setattr(
        fleet_cells, "aot_export_spec",
        lambda pipe, cfg: SimpleNamespace(family="sdxl", strict=True,
                                          lora_bucket=0))
    monkeypatch.setattr(
        mint_supervisor, "export_declaration", lambda family: object())
    monkeypatch.setattr(
        aot_mint, "declared_class_rows",
        lambda pipe, spec, decl: [object()] * declared)


def _artifact(path: Path, name: str, key: str) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub-graph-bytes")
    return aot_mint.MintedArtifact(
        key=key, entry=name, artifact=path,
        metadata={"cell_key": key, "entry": {"name": name}})


def test_a_packed_graph_class_is_adopted_through_the_delivered_cell_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_parent_gates(monkeypatch, declared=2)
    adopted: List[Path] = []

    def _mint(template: Any, **kw: Any) -> Any:
        out = Path(kw["workdir"]).parent / "graphs"
        return aot_mint.MintResult(
            entries=(_artifact(out / "a.tar.gz", "cls-a", "cg-key-v1-" + "a" * 56),
                     _artifact(out / "b.tar.gz", "cls-b", "cg-key-v1-" + "b" * 56)),
            manifest="m", timings={})

    monkeypatch.setattr(aot_mint, "mint_graph_classes", _mint)

    def _adopt(pipe: Any, pending: Any, artifacts: Any) -> Any:
        # The adopt takes the SET, one artifact per graph class. A double
        # taking a single Path models a call production does not make.
        rows = [Path(a) for a in artifacts]
        adopted.extend(rows)
        return fleet_cells.SelfMint(
            family="sdxl", compiled_graph_key="cg-key-v1-abc", ref="r#k",
            snapshot_digest="blake3:x", artifact=rows[0])

    monkeypatch.setattr(fleet_cells, "adopt_delegated_mint", _adopt)
    act = _Act()
    result = asyncio.run(mint_supervisor.supervise(_task(tmp_path), act=act))
    assert result.status == mint_supervisor.ADOPTED and result.ok
    assert result.attempts == 1 and result.covered == 2
    assert len(adopted) == 2, (
        "every packed graph class is adopted on its own — pgw#1176's whole "
        "point is that a class that refuses costs itself")
    # No new protocol: the supervision phases land on the SAME activity the
    # hub already reads for its minting classification.
    assert "load" in act.phases and "seal_publish" in act.phases


def test_the_pool_peak_reaches_the_parent_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The current supervisor consumes the child phase table directly."""

    _stub_parent_gates(monkeypatch, declared=1)
    mint_workers._ENTRY_RSS_PEAKS.clear()
    peak = 6 * GIB

    def _mint(template: Any, **kw: Any) -> Any:
        row = _artifact(
            Path(kw["workdir"]).parent / "graph.tar.gz",
            "cls-a",
            "cg-key-v1-" + "a" * 56,
        )
        row.metadata["mint_phases"] = {
            "pool": {"peak_child_rss_bytes": peak, "pool_workers": 1},
        }
        return aot_mint.MintResult(
            entries=(row,), manifest="m", timings={})

    monkeypatch.setattr(aot_mint, "mint_graph_classes", _mint)
    monkeypatch.setattr(
        fleet_cells,
        "adopt_delegated_mint",
        lambda *_a, **_kw: fleet_cells.SelfMint(
            family="sdxl",
            compiled_graph_key="cg-key-v1-" + "a" * 56,
            ref="r#k",
            snapshot_digest="blake3:x",
            artifact=tmp_path / "graph.tar.gz",
        ),
    )

    result = asyncio.run(mint_supervisor.supervise(_task(tmp_path), act=_Act()))

    assert result.ok
    assert mint_workers.compiled_graph_peak_rss("sdxl", "fp8") == peak
    measured = aot_compile_pool.entry_workers(
        2,
        vcpus=16,
        available_bytes=64 * GIB,
        device_lock=True,
        peak_rss_bytes=peak,
    )
    assert measured.per_entry_rss_basis == "measured"


def test_a_named_refusal_is_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running a deterministic refusal buys a second billed compile for the
    same sentence (ie#576/th#1288: every retry is a billed pod)."""
    _stub_parent_gates(monkeypatch)
    _events(monkeypatch)
    calls: List[int] = []

    def _mint(template: Any, **kw: Any) -> Any:
        calls.append(1)
        raise aot_mint.MintRefused("the declaration cannot be exported")

    monkeypatch.setattr(aot_mint, "mint_graph_classes", _mint)
    result = asyncio.run(mint_supervisor.supervise(
        _task(tmp_path), act=_Act(), max_attempts=3))
    assert result.status == mint_supervisor.FAILED
    assert result.attempts == 1 and len(calls) == 1, (
        "a refusal must not consume the retry budget")


def test_a_resource_shortfall_gets_exactly_one_more_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first shortfall may have been the tenant\'s peak, which has since
    passed — so retry, but only after RE-SIZING K, and only within budget."""
    _stub_parent_gates(monkeypatch)
    _events(monkeypatch)
    calls: List[int] = []

    def _mint(template: Any, **kw: Any) -> Any:
        calls.append(1)
        raise aot_mint.MintResourceExhausted("the card could not hold K")

    monkeypatch.setattr(aot_mint, "mint_graph_classes", _mint)
    result = asyncio.run(mint_supervisor.supervise(
        _task(tmp_path), act=_Act(), max_attempts=2))
    assert result.status == mint_supervisor.FAILED
    assert result.attempts == 2 and len(calls) == 2


def test_a_retry_CONSUMES_what_the_first_attempt_already_packed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE PRICED-RESUME FIX (pgw#1215 step 4).

    The deleted ``mint_delegate.build_cell`` gave every attempt a fresh
    ``child-N`` workdir, so attempt 2 of a 36-class mint re-traced and
    re-compiled 35 finished classes to retry one — at a measured 156-509 s
    each. The supervisor accretes into an attempt-STABLE directory and hands
    the children the names it already holds, so a retry pays for the residue
    and nothing else.

    Asserted at BOTH ends, because either alone is satisfiable by an accident:
    the second attempt is told to SKIP the packed class (``have_classes``),
    and the packed artifact is still in the result it adopts (``held``).
    """
    _stub_parent_gates(monkeypatch, declared=2)
    seen_have: List[tuple] = []
    seen_held: List[int] = []

    def _mint(template: Any, **kw: Any) -> Any:
        seen_have.append(tuple(template.have_classes))
        seen_held.append(len(kw.get("held") or ()))
        out = Path(template.out_dir)
        if len(seen_have) == 1:
            # Attempt 1 packs one of two classes and then dies on the second.
            _artifact(out / "a.tar.gz", "cls-a", "cg-key-v1-" + "a" * 56)
            raise aot_mint.MintResourceExhausted("class b OOMed")
        return aot_mint.MintResult(
            entries=(
                _artifact(out / "a.tar.gz", "cls-a", "cg-key-v1-" + "a" * 56),
                _artifact(out / "b.tar.gz", "cls-b", "cg-key-v1-" + "b" * 56)),
            manifest="m", timings={})

    monkeypatch.setattr(aot_mint, "mint_graph_classes", _mint)
    monkeypatch.setattr(
        mint_supervisor, "held_graph_classes",
        lambda out_dir: [
            aot_mint.MintedArtifact(
                key="cg-key-v1-" + "a" * 56, entry="cls-a",
                artifact=path, metadata={"entry": {"name": "cls-a"}})
            for path in sorted(Path(out_dir).glob("a.tar.gz"))])
    monkeypatch.setattr(
        fleet_cells, "adopt_delegated_mint",
        lambda pipe, pending, artifacts: fleet_cells.SelfMint(
            family="sdxl", compiled_graph_key="k", ref="r",
            snapshot_digest="d",
            artifact=Path(list(artifacts)[0])))

    result = asyncio.run(mint_supervisor.supervise(
        _task(tmp_path), act=_Act(), max_attempts=2))
    assert result.status == mint_supervisor.ADOPTED
    assert seen_have == [(), ("cls-a",)], (
        "attempt 2 re-compiled a class this pod already had packed — that is "
        "the priced-resume regression back on the tree")
    assert seen_held == [0, 1], (
        "the already-packed class must JOIN the result, or the retry reports "
        "1 of 2 classes for a pod that holds both")


def test_abandonment_is_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adopt-on-arm, vacate and shutdown abandon a mint; none of them is a
    broken worker, so none of them may look like one.

    And the signal must reach the CHILDREN, not merely the task: the
    supervisor drives the pool from a worker thread, so a cancelled task would
    leave K compile children running on a card the pod just gave back. That is
    what ``should_abandon`` is polled for, and this test drives it through the
    real predicate.
    """
    _stub_parent_gates(monkeypatch)
    _events(monkeypatch)

    def _mint(template: Any, **kw: Any) -> Any:
        should = kw["should_abandon"]
        for _ in range(500):
            if should():
                raise aot_compile_pool.EntryCompileAbandoned("supervisor said stop")
            time.sleep(0.01)
        raise AssertionError("the abandon signal never reached the pool")

    monkeypatch.setattr(aot_mint, "mint_graph_classes", _mint)

    async def _go() -> mint_supervisor.SupervisedResult:
        stop = asyncio.Event()
        task = asyncio.ensure_future(mint_supervisor.supervise(
            _task(tmp_path), act=_Act(), abandon=stop))
        await asyncio.sleep(0.2)
        stop.set()
        return await task

    result = asyncio.run(_go())
    assert result.status == mint_supervisor.ABANDONED
    assert not result.ok


def test_a_declared_blocker_REFUSES_on_the_parent_and_never_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1115, re-homed by the reroute.

    The gate was ``mint_child._assert_family_mintable`` — a fail-closed check
    in the middle tier. The middle tier is gone, so it moved to the process
    that now decides to mint. A reroute that dropped it would publish compiled
    graphs for a class set the declaration says it cannot yet claim.

    The name matters as much as the gate: ``MintRefused`` is live in the
    declaration collector for *"every declared class refused"*, so reusing it
    would make the two indistinguishable at the terminus with nothing going red.
    """
    _events(monkeypatch)
    monkeypatch.setattr(
        mint_supervisor, "assert_family_mintable",
        lambda family: (_ for _ in ()).throw(
            mint_supervisor.DeclaredBlockerRefusal("sdxl declares 1 blocker")))

    def _mint(template: Any, **kw: Any) -> Any:
        raise AssertionError("a blocked family must not reach the pool")

    monkeypatch.setattr(aot_mint, "mint_graph_classes", _mint)
    result = asyncio.run(mint_supervisor.supervise(_task(tmp_path), act=_Act()))
    assert result.status == mint_supervisor.FAILED
    assert result.reason == "declared_blocker"
    assert not issubclass(
        mint_supervisor.DeclaredBlockerRefusal, aot_mint.MintRefused), (
        "the declared-blocker refusal and the every-class-refused refusal are "
        "different verdicts and must stay distinguishable at the terminus")


def test_the_only_bank_that_SIZES_anything_is_the_host_rss_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K = f(cores, ONE measured child RSS), and no device number takes part.

    §4.33 / pgw#1175 deleted five peak banks whose arithmetic declined four
    families at 49-113 GiB. This fence keeps that deletion — but it states the
    PROPERTY rather than a list of forbidden names.

    WHY IT CHANGED (pgw#1205, coordinator ruling 2026-08-13). It used to assert
    `not hasattr(mint_workers, "record_entry_device_peak")`, and §4.33's own
    follow-up then asked for exactly that function back. A banked MEASUREMENT
    is not the thing §4.33 deleted; a banked measurement that SIZES something
    is. Naming symbols could not tell those apart.

    So: prohibition by STRUCTURE, not by vocabulary.
    """
    mint_workers._ENTRY_RSS_PEAKS.clear()
    assert mint_workers.compiled_graph_peak_rss("sdxl", "w8a8") == 0
    mint_workers.record_compiled_graph_peak_rss("sdxl", "w8a8", 5 * GIB)
    mint_workers.record_compiled_graph_peak_rss("sdxl", "w8a8", 2 * GIB)
    assert mint_workers.compiled_graph_peak_rss("sdxl", "w8a8") == 5 * GIB, (
        "the bank is not monotone — a lucky run talked the ask down")

    # The two whose names are history: they had consumers, and a consumer is
    # what turns a reading into a floor.
    assert not hasattr(mint_workers, "record_child_peak")
    assert not hasattr(mint_workers, "record_adopt_peak")

    reader = re.compile(r"(?<!record_)entry_device_peak\s*\(")
    src_root = Path(mint_workers.__file__).resolve().parent
    consumers = [
        path.name for path in sorted(src_root.rglob("*.py"))
        if path.name != "mint_workers.py" and reader.search(path.read_text())
    ]
    assert consumers == [], (
        f"{consumers} reads the device census — that is `mint_budget`\'s "
        f"arithmetic returning, and §4.33 deleted it on measured evidence")

    # ...and nothing device-shaped crosses to a compile child. The ONE
    # measurement handed down is the host RSS; `vram_cap_bytes` died with
    # pgw#1175.
    handed_down = set(aot_compile_pool.EntryJob.__struct_fields__)
    assert not [
        f for f in handed_down
        if ("vram" in f or "device" in f) and f.endswith("_bytes")
    ], "a device budget is being handed down on the child job again"


def test_supervision_is_unconditional_and_has_no_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1010: the in-process shape is DELETED, not disabled.

    It existed only to capture and pack a dynamo cell, so
    ``GEN_WORKER_MINT_IN_PROCESS`` selected a state this worker can no longer
    be in — and pgw#995 named that env the last deletable behaviour switch."""
    monkeypatch.setenv("GEN_WORKER_MINT_IN_PROCESS", "1")
    assert mint_supervisor.delegation_refusal() == ""
    assert not hasattr(mint_supervisor, "ENV_IN_PROCESS")
    assert not hasattr(mint_supervisor, "delegated")


def test_the_middle_mint_TIER_is_gone_from_the_serving_path() -> None:
    """th#1834 Phase 3's deletion, asserted structurally.

    ``mint_delegate`` is deleted outright; ``executor`` reaches no one-shot
    mint child. Read from the SOURCE rather than from behaviour, because a
    behavioural assertion passes while a second, unreached caller survives
    somewhere else in the module.
    """
    import importlib

    with pytest.raises(ImportError):
        importlib.import_module("gen_worker.mint_delegate")

    from gen_worker import executor as _ex

    src = Path(_ex.__file__).read_text()
    for gone in ("mint_delegate", "mint_process", "run_mint", "build_cell"):
        assert gone not in src, (
            f"executor.py still reaches `{gone}` — the serving parent is "
            f"supposed to drive compile children DIRECTLY")


# ------------------------------- the child half of the accretion, functionally


def test_a_HELD_class_is_dropped_from_the_share_BEFORE_it_is_exported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other end of ``have_classes``, and the one that saves the money.

    A skip applied at PACK time would still pay the 76-93 s export and the
    108-140 s compile; the whole point is that the row never reaches
    ``_export_entry``. Asserted by counting exports, not by reading source.

    The filter runs AFTER the shard, deliberately: ``rows[i::K]`` must stay the
    same partition of the same order, or a skipped class moves its siblings
    between children and the pool's whole-set proof stops meaning anything.
    """
    from gen_worker import aot_declaration as _decl

    rows = [(f"plan-{i}", True) for i in range(4)]
    exported: List[str] = []
    monkeypatch.setattr(
        aot_mint, "declared_class_rows", lambda pipe, spec, decl: list(rows))
    monkeypatch.setattr(_decl, "plan_entry_name", lambda plan: str(plan))
    monkeypatch.setattr(aot_mint, "_arm_branches", lambda pipe, bucket: None)
    monkeypatch.setattr(aot_mint, "_disarm_branches", lambda pipe: None)
    monkeypatch.setattr(aot_mint, "keying_block", lambda p, f, s: {})

    def _export(pipe: Any, spec: Any, plan: Any, decl: Any, **kw: Any) -> Any:
        exported.append(str(plan))
        return SimpleNamespace(
            program=SimpleNamespace(graph_module=SimpleNamespace(
                graph=SimpleNamespace(nodes=()))),
            ingress=object(), spec=spec, timings={})

    monkeypatch.setattr(aot_mint, "_export_entry", _export)
    spec = SimpleNamespace(lora_bucket=0)

    list(aot_mint.trace_for_key(
        object(), spec, object(), have_classes=("plan-1", "plan-3")))
    assert exported == ["plan-0", "plan-2"], (
        "a class this pod already holds was re-EXPORTED — the priced-resume "
        "regression is back, and it costs the trace as well as the compile")


def test_the_whole_SET_proof_counts_held_classes_beside_packed_ones() -> None:
    """pgw#1089's proof at the compile seam, re-based on accretion.

    The union of the shares must be the whole declared class set — that is
    what stops a short cell verifying and arming with a class missing. A
    retry compiles the residue, so the proof has to count coverage rather than
    this attempt's work; counting only the packed rows would refuse every
    successful retry, which is a fence that fires on the fix.
    """
    pool = aot_compile_pool.EntryCompilePool.__new__(
        aot_compile_pool.EntryCompilePool)
    declared = {"share-000": 3}
    packed = {"cls-c": object()}
    pool._assert_shares_whole(declared, packed, 1, have=2)
    with pytest.raises(aot_compile_pool.EntryCompileFailed) as exc:
        pool._assert_shares_whole(declared, packed, 1, have=0)
    assert "would be short" in str(exc.value)
