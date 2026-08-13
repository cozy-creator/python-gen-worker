"""pgw#784: the serving side of an out-of-process mint.

Three claims, each with the fact that would break it:

1. **The live pipeline is never armed for a delegated mint.** In the
   in-process shape the serving pipe carries guarded wrappers, LoRA branch
   containers and a process-global ``TORCHINDUCTOR_CACHE_DIR`` move for the
   whole mint; delegated, it carries none of that and keeps serving plain
   eager. (pgw#1010 deleted the in-process shape outright, so this is now the
   only shape a mint has — and the restriction it carried is gone with it.)
2. **The child exports the declaration the PARENT stated.** The parent states
   the compile contract on the wire because the class-scoped unions live on
   the spec, not the decorator: a child re-deriving from ``@endpoint`` alone
   would export a different declaration than the parent asked for. (pgw#1034:
   this used to be phrased as key parity — "the child computes the same key" —
   which stopped being true at pgw#758. The child computes no key; the parent
   stamps one from the returned envelope. What the wire owes is the DECLARED
   EXPORT, and that is what is checked below.)
3. **Failure inversion.** A dead mint process is a FAILED MINT reported by a
   LIVE worker. Every branch returns a typed result, the retry is bounded and
   class-driven, and a card with no room for a co-resident child DECLINES
   without spawning anything.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

from gen_worker import compile_cache as cc
from gen_worker import fleet_compiled_graphs, mint_delegate, mint_workers
from gen_worker import mint_process as mp
from gen_worker.api.binding import ModelRef
from gen_worker.api.decorators import DynamicDim
from gen_worker.registry import CompileCompiledGraph
from gen_worker.compiled_graph_adopt import AdoptOutcome

GIB = 1 << 30
STUB_MODULE = "harness.mint_child_stub"


def _cfg() -> CompileCompiledGraph:
    return CompileCompiledGraph(
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


@pytest.fixture(autouse=True)
def _stub_child(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(mp, "MINT_CHILD_MODULE", STUB_MODULE)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(root / "src"), str(root / "tests")]))


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

def test_the_delegated_arm_never_touches_the_live_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    armed: List[str] = []
    monkeypatch.setattr(
        cc, "arm_jit_intake", lambda *a, **k: armed.append("arm_jit_intake"))
    # pgw#1010: WHICH recipe a miss runs has its own coverage; a test box
    # registers no export declarations, so state the answer this test is about.
    monkeypatch.setattr(
        fleet_compiled_graphs, "mint_recipe", lambda *a, **k: fleet_compiled_graphs.RECIPE_AOT)
    monkeypatch.setattr(
        fleet_compiled_graphs.provision, "enable_compiled",
        lambda *a, **k: AdoptOutcome.miss("no_compiled_graph"))
    monkeypatch.setattr(cc, "has_compile_target", lambda *a, **k: True)
    monkeypatch.setattr(cc, "mandatory_serving", lambda pipe: False)
    monkeypatch.setattr(cc, "apply_lora_execution_lane", lambda pipe, bucket, **_kw: True)
    monkeypatch.setattr(cc, "drop_lora_execution_lane", lambda pipe: True)
    monkeypatch.setattr(
        fleet_compiled_graphs.loading, "pipeline_weight_lane", lambda pipe: "fp8")
    monkeypatch.setattr(fleet_compiled_graphs, "_cuda_ready", lambda: True)
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)
    # No CUDA on this box, so the real sm axis is unavailable; the key itself
    # is not what this test is about.
    monkeypatch.setattr(
        fleet_compiled_graphs, "arm_identity",
        lambda *a, **k: SimpleNamespace(
            token="arm1-test", facts_dict=lambda: {}))

    prior = dict(os.environ)
    outcome = fleet_compiled_graphs.enable_compiled(
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
    wire form must equal the one the parent's own ``CompileCompiledGraph`` derives.

    That — not ``contract_facts()`` — is what the wire owes. The child's single
    consumer of the cfg is ``fleet_compiled_graphs.aot_export_spec`` (plus
    ``resolve_targets``, which reads ``targets``), and a dropped union or a
    lost shape row silently exports a different declaration than the parent
    asked for. pgw#1034 also asserts the converse: a field with no child reader
    does not cross the wire.
    """
    parent = _cfg()
    wire = mint_delegate.cfg_spec(parent)
    pipe = SimpleNamespace()

    assert (fleet_compiled_graphs.aot_export_spec(pipe, wire)
            == fleet_compiled_graphs.aot_export_spec(pipe, parent))
    assert cc.resolve_targets(pipe, wire) == cc.resolve_targets(pipe, parent)

    # No child reads these, so they do not ride (pgw#1034).
    for dead in ("regional", "text_len", "dynamic"):
        assert not hasattr(wire, dead), dead


def test_the_request_carries_the_execution_lane_and_the_effective_config(
    tmp_path: Path,
) -> None:
    """Both steer the warm forwards, so both must be the PARENT's values —
    a child warming at different config traces different graphs and the
    parent's own proof then misses."""
    pending = SimpleNamespace(
        family="sdxl", arm_token="ck1-abc", cfg=_cfg(),
        target=tmp_path / "compiled_graph.tar.gz", mint_root=tmp_path)
    task = mint_delegate.MintTask(
        pending=pending, pipe=object(), function="gen",
        modules=("app",), slots={"pipeline": mp.MintSlot(
            ref=ModelRef(source="tensorhub", path="harness/sdxl",
                         tag="prod"), path="/cas/sdxl")},
        execution_lane="fp8-w8a16", configs={"gen": {"steps": 28}}, device=3)
    req = mint_delegate.build_request(task, workdir=tmp_path / "w")
    assert req.execution_lane == "fp8-w8a16"
    assert req.configs == {"gen": {"steps": 28}}
    assert req.slots["pipeline"].path == "/cas/sdxl"
    assert req.device == 3
    # pgw#1010: the child's WORK ROOT — the tree it actually writes into, and
    # the byte-growth half of the parent's progress evidence. It used to be
    # the inductor capture dir, which an AOT mint never touched.
    assert req.work_root == str(tmp_path / "w")


# --------------------------------------------------- 3. failure inversion

def _task(tmp_path: Path, **over: Any) -> mint_delegate.MintTask:
    pending = fleet_compiled_graphs.PendingSelfMint(
        family="sdxl", arm_token="ck1-abc",
        ref="root/family-sdxl#ek1-abc", cfg=_cfg(),
        target=tmp_path / "compiled_graph.tar.gz",
        mint_root=tmp_path / "root", publisher=None, cache_dir=tmp_path)
    fields: Dict[str, Any] = dict(
        pending=pending, pipe=SimpleNamespace(), function="gen",
        modules=("harness.toy_endpoints",), weight_lane="fp8", device=0)
    fields.update(over)
    return mint_delegate.MintTask(**fields)


class _Act:
    """Just enough Activity to record what the hub would have seen."""

    def __init__(self) -> None:
        self.phases: List[str] = []
        self.notes: List[str] = []

    def phase(self, phase: str, step: int = 0, total: int = 0) -> None:
        self.phases.append(phase)

    def note(self, detail: str) -> None:
        self.notes.append(detail)


def _events(monkeypatch: pytest.MonkeyPatch) -> List[tuple]:
    seen: List[tuple] = []
    monkeypatch.setattr(
        mint_delegate.activity_mod, "emit_event",
        lambda kind, detail, phase="", **_kw: seen.append((kind, phase, detail)))
    monkeypatch.setattr(
        fleet_compiled_graphs.activity_mod, "emit_event",
        lambda kind, detail, phase="", **_kw: seen.append((kind, phase, detail)))
    return seen


def test_a_minted_child_is_adopted_through_the_delivered_compiled_graph_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINT_STUB_MODE", "minted")
    _fake_card(monkeypatch, total_gib=80, resident_gib=6)
    adopted: List[Path] = []

    def _adopt(pipe: Any, pending: Any, artifacts: Any) -> Any:
        # pgw#1176: the adopt takes the SET the child produced, one artifact
        # per graph class. A double taking a single Path models a call
        # production does not make.
        rows = [Path(a) for a in artifacts]
        adopted.extend(rows)
        return fleet_compiled_graphs.SelfMint(
            family="sdxl", compiled_graph_key="ek1-abc", ref="r#k",
            snapshot_digest="blake3:x", artifact=rows[0])

    monkeypatch.setattr(fleet_compiled_graphs, "adopt_delegated_mint", _adopt)
    act = _Act()
    result = asyncio.run(mint_delegate.build_compiled_graph(_task(tmp_path), act=act))
    assert result.status == mint_delegate.ADOPTED and result.ok
    assert result.attempts == 1
    assert adopted and adopted[0].read_bytes() == b"stub-compiled_graph-bytes"
    # No new protocol: the child's phases land on the SAME activity the hub
    # already reads for its minting classification.
    assert "load" in act.phases and "seal_publish" in act.phases


def test_a_dead_mint_process_is_a_failed_mint_not_a_dead_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """th#1299 inverted: the mint dies, the worker does not. No exception
    escapes, the reason rides the wire typed, and serving is untouched."""
    monkeypatch.setenv("MINT_STUB_MODE", "sigkill")
    _fake_card(monkeypatch, total_gib=80, resident_gib=6)
    seen = _events(monkeypatch)
    result = asyncio.run(mint_delegate.build_compiled_graph(
        _task(tmp_path), act=_Act(), max_attempts=1))
    assert result.status == mint_delegate.FAILED
    assert not result.ok and result.attempts == 1
    aborts = [e for e in seen if e[0] == "self_mint_abort"]
    assert aborts, "a failed mint must be wire-visible, not a pod-log line"
    kind, phase, detail = aborts[0]
    assert phase == "delegated_crashed"
    assert "kept serving eager" in detail and "SIGKILL" in detail


def test_a_named_refusal_is_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running a deterministic refusal buys a second billed compile for the
    same sentence (ie#576/th#1288: every retry is a billed pod)."""
    monkeypatch.setenv("MINT_STUB_MODE", "refused")
    _fake_card(monkeypatch, total_gib=80, resident_gib=6)
    _events(monkeypatch)
    result = asyncio.run(mint_delegate.build_compiled_graph(
        _task(tmp_path), act=_Act(), max_attempts=3))
    assert result.status == mint_delegate.FAILED
    assert result.attempts == 1, "a refusal must not consume the retry budget"


def test_a_resource_shortfall_gets_exactly_one_more_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first shortfall may have been the tenant's peak, which has since
    passed — so retry, but only after RE-BUDGETING, and only once."""
    monkeypatch.setenv("MINT_STUB_MODE", "resource")
    _fake_card(monkeypatch, total_gib=80, resident_gib=6)
    _events(monkeypatch)
    result = asyncio.run(mint_delegate.build_compiled_graph(
        _task(tmp_path), act=_Act(), max_attempts=2))
    assert result.status == mint_delegate.FAILED
    assert result.attempts == 2


def test_abandonment_is_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adopt-on-arm, vacate and shutdown abandon a mint; none of them is a
    broken worker, so none of them may look like one."""
    monkeypatch.setenv("MINT_STUB_MODE", "silent")
    monkeypatch.setenv("MINT_STUB_SECONDS", "120")
    _fake_card(monkeypatch, total_gib=80, resident_gib=6)
    _events(monkeypatch)

    async def _go() -> mint_delegate.DelegatedResult:
        stop = asyncio.Event()
        task = asyncio.ensure_future(mint_delegate.build_compiled_graph(
            _task(tmp_path), act=_Act(), abandon=stop))
        await asyncio.sleep(1.0)
        stop.set()
        return await task

    result = asyncio.run(_go())
    assert result.status == mint_delegate.ABANDONED
    assert not result.ok


def test_a_mint_spawns_a_child_with_no_pre_flight_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§4.33 / pgw#1175: THE ATTEMPT IS THE BUDGET.

    A `mint_budget.co_residency` gate used to stand at the top of
    `build_compiled_graph` and return DECLINED without spawning anything — on a `need`
    whose leading term was `allocated`, the PARENT's resident weights, already
    excluded from the `free_bytes` it was compared against. This card is the
    wan-2.2 shape verbatim (80 GiB total, 54 GiB resident) and the mint now
    RUNS: a weight-free compile child does not re-hold the parent's weights,
    and if the card really cannot take it the child dies in its own process
    and says so.
    """
    monkeypatch.setenv("MINT_STUB_MODE", "minted")
    _fake_card(monkeypatch, total_gib=80, resident_gib=54)
    monkeypatch.setattr(
        fleet_compiled_graphs, "adopt_delegated_mint",
        lambda pipe, pending, artifacts: fleet_compiled_graphs.SelfMint(
            family="sdxl", compiled_graph_key="k", ref="r", snapshot_digest="d",
            artifact=Path(list(artifacts)[0])))
    result = asyncio.run(mint_delegate.build_compiled_graph(
        _task(tmp_path, weight_lane="w8a8"), act=_Act()))
    assert result.status == mint_delegate.ADOPTED, (
        "a 54-GiB-resident pod refused to even TRY — that is the retracted "
        "double-count back on the tree")
    assert not hasattr(mint_delegate, "DECLINED"), (
        "the DECLINED vocabulary survived its only producer")


def test_the_only_bank_that_SIZES_anything_is_the_host_rss_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K = f(cores, ONE measured child RSS), and no device number takes part.

    §4.33 / pgw#1175 deleted five peak banks whose arithmetic declined four
    families at 49-113 GiB. This fence keeps that deletion — but it now states
    the PROPERTY rather than a list of forbidden names.

    WHY IT CHANGED (pgw#1205, coordinator ruling 2026-08-13). It used to assert
    `not hasattr(mint_workers, "record_compiled_graph_device_peak")`, and §4.33's own
    follow-up then asked for exactly that function back: *"Bank, per GRAPH
    CLASS … Provenance on every row — this is the point … Monotone per (class,
    card, toolchain, lane), as `record_child_peak` already was."* A banked
    MEASUREMENT is not the thing §4.33 deleted; a banked measurement that SIZES
    something is. Naming symbols could not tell those apart, and this session
    watched the same form fail in both directions — a required gate stayed
    green while naming a symbol that had been deleted, and then a legitimate
    reintroduction went red on a name while the protected invariant was
    stronger than before.

    So: prohibition by STRUCTURE, not by vocabulary.

    * `record_child_peak` and `record_adopt_peak` stay banned BY NAME. They had
      consumers; their names are load-bearing history and their return would be
      the arithmetic itself coming back.
    * The device census that DOES exist may not be read outside its own module
      — asserted here, and asserted more fully by
      `test_device_peak_bank_pgw1205.test_no_width_or_placement_decision_reads_the_bank`.
    * Nothing device-shaped is handed down on the request.
    """
    mint_workers._COMPILED_GRAPH_RSS_PEAKS.clear()
    assert mint_workers.compiled_graph_peak_rss("sdxl", "w8a8") == 0
    mint_workers.record_compiled_graph_peak_rss("sdxl", "w8a8", 5 * GIB)
    mint_workers.record_compiled_graph_peak_rss("sdxl", "w8a8", 2 * GIB)
    assert mint_workers.compiled_graph_peak_rss("sdxl", "w8a8") == 5 * GIB, (
        "the bank is not monotone — a lucky run talked the ask down")

    # The two whose names are history: they had consumers, and a consumer is
    # what turns a reading into a floor.
    assert not hasattr(mint_workers, "record_child_peak")
    assert not hasattr(mint_workers, "record_adopt_peak")

    # The invariant itself. A device reading may be BANKED (it is a
    # measurement) and may not be READ by anything that decides a width, a
    # placement or an admission.
    reader = re.compile(r"(?<!record_)compiled_graph_device_peak\s*\(")
    src_root = Path(mint_workers.__file__).resolve().parent
    consumers = [
        path.name for path in sorted(src_root.rglob("*.py"))
        if path.name != "mint_workers.py" and reader.search(path.read_text())
    ]
    assert consumers == [], (
        f"{consumers} reads the device census — that is `mint_budget`'s "
        f"arithmetic returning, and §4.33 deleted it on measured evidence")

    # ...and nothing device-shaped crosses to the child, which is the other
    # half of the sentence this fence used to carry. `compiled_graph_peak_rss_bytes` is
    # the one measurement handed down; `vram_cap_bytes` died with pgw#1175.
    handed_down = set(mp.MintRequest.__struct_fields__)
    assert "compiled_graph_peak_rss_bytes" in handed_down
    assert not [
        f for f in handed_down
        if ("vram" in f or "device" in f) and f.endswith("_bytes")
    ], "a device budget is being handed down on the request again"


def test_delegation_is_unconditional_and_has_no_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1010: the in-process shape is DELETED, not disabled.

    It existed only to capture and pack a dynamo compiled graph, so
    ``GEN_WORKER_MINT_IN_PROCESS`` selected a state this worker can no longer
    be in — and pgw#995 named that env the last deletable behaviour switch,
    blocked only on the ten test sites that forced the shape. Setting it must
    now do nothing at all; the env must not be read by anybody."""
    monkeypatch.setenv("GEN_WORKER_MINT_IN_PROCESS", "1")
    assert mint_delegate.delegation_refusal() == ""
    assert not hasattr(mint_delegate, "ENV_IN_PROCESS")
    # pgw#1030: the `delegated()` bool wrapper is deleted with the switch it
    # once negated; `delegation_refusal` is the predicate.
    assert not hasattr(mint_delegate, "delegated")
