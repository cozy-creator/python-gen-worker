"""pgw#784: the executor wiring — the delegated route actually runs.

Everything else in pgw#784 is reachable in isolation; this is the file that
proves the serving worker CALLS it, and that the exception contract the
existing ``_background_mint`` wrapper depends on is preserved exactly.

Four claims, and one of them is a bug this wiring had to fix:

1. A DELEGATED pending is recorded even though the arm reports ``armed=False``.
   That is not a detail — the recording gate was `if armed and selection is not
   None`, so a delegated pending was silently DROPPED and no mint ever ran. The
   pipe serves eager, so nothing is armed, so `armed` is honestly False; the
   obligation is still real, it is just owed to a child process.
2. `_BackgroundMint` carries the two facts a child cannot rediscover — the
   declaring module(s) to walk, and the already-materialized local snapshot
   path per slot (a mint is compute; the child never touches the network).
3. `_delegated_mint_run` drives a child, adopts, publishes on gw#612's
   sibling-coverage rule, and advertises through the SAME phase-4 code the
   in-process route uses.
4. Its failure modes map onto the wrapper's existing vocabulary:
   `_MintAbandoned` (stop asked) or plain `Exception` (a failed mint).
   Serving continues in either. (pgw#1175 deleted `_MintDeclined`: nothing
   refuses a mint in advance any more — §4.33.)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

import gen_worker.executor as executor_mod
from gen_worker import fleet_compiled_graphs, mint_delegate, mint_workers
from gen_worker.api.binding import ModelRef
from gen_worker import mint_process as mp
from gen_worker.executor import (
    Executor,
    ModelStore,
    _BackgroundMint,
    _ClassRecord,
    _CompileTargetRecord,
    _MintAbandoned,
    _delegated_pendings,
    _mint_modules,
)
from gen_worker.registry import CompileCompiledGraph, extract_specs
from gen_worker.compiled_graph_adopt import AdoptOutcome

GIB = 1 << 30
STUB_MODULE = "harness.mint_child_stub"
SDXL_REF = ModelRef(source="tensorhub", path="harness/sdxl", tag="prod")


def _slot(path: str) -> mp.MintSlot:
    """One resolved slot. ``ref`` has no default (pgw#974), so a test cannot
    describe bytes without saying whose they are either."""
    return mp.MintSlot(ref=SDXL_REF, path=path)


class _Pipe:
    pass


def _cfg() -> CompileCompiledGraph:
    return CompileCompiledGraph(
        shapes=((1024, 1024),), targets=("unet",), family="sdxl",
        regional=False, text_len=77, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=())


@pytest.fixture(autouse=True)
def _stub_child(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(mp, "MINT_CHILD_MODULE", STUB_MODULE)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(root / "src"), str(root / "tests")]))
    monkeypatch.setenv("MINT_STUB_MODE", "minted")


def _pending(tmp_path: Path) -> Any:
    return fleet_compiled_graphs.PendingSelfMint(
        family="sdxl", arm_token="ck1-abc", ref="root/family-sdxl#ek1-abc",
        cfg=_cfg(), target=tmp_path / "compiled_graph.tar.gz",
        mint_root=tmp_path / "root", publisher=None, cache_dir=tmp_path)


# ------------------------------------------------- 1. the recording gate

def test_a_delegated_pending_is_recorded_even_though_nothing_is_armed() -> None:
    """The bug this wiring fixes.

    `_delegated_pendings` is what the executor now asks instead of trusting
    `armed`. A delegated arm reports armed=False *truthfully* — the live pipe
    carries no wrappers and serves eager — so a gate keyed on `armed` drops the
    obligation and the compiled graph is never minted at all.
    """
    delegated = SimpleNamespace(delegated=True)
    plain = SimpleNamespace()          # a finalized SelfMint has no such field
    assert _delegated_pendings({1: delegated})
    assert _delegated_pendings({1: plain, 2: delegated})
    assert not _delegated_pendings({1: plain})
    assert not _delegated_pendings({})


def test_the_arm_returns_armed_false_with_a_delegated_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restating it from the other side, so the two halves cannot drift: the
    arming brain really does hand back armed=False plus an obligation."""
    monkeypatch.setattr(
        fleet_compiled_graphs.provision, "enable_compiled",
        lambda *a, **k: AdoptOutcome.miss("no_compiled_graph"))
    monkeypatch.setattr(fleet_compiled_graphs.cc, "has_compile_target", lambda *a, **k: True)
    monkeypatch.setattr(fleet_compiled_graphs.cc, "mandatory_serving", lambda pipe: False)
    monkeypatch.setattr(fleet_compiled_graphs.cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(fleet_compiled_graphs, "_cuda_ready", lambda: True)
    monkeypatch.setattr(
        fleet_compiled_graphs.loading, "pipeline_weight_lane", lambda pipe: "fp8")
    monkeypatch.setattr(
        fleet_compiled_graphs, "arm_identity",
        lambda *a, **k: SimpleNamespace(
            token="arm1-wired", facts_dict=lambda: {}))
    monkeypatch.setattr(
        fleet_compiled_graphs, "mint_recipe", lambda *a, **k: fleet_compiled_graphs.RECIPE_AOT)

    outcome = fleet_compiled_graphs.enable_compiled(_Pipe(), _cfg(), tmp_path)
    assert not outcome.armed
    assert _delegated_pendings({1: outcome.self_mint})


def test_delegation_is_the_policy_not_a_caller_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`enable_compiled`'s signature is unchanged for callers.

    Learned the hard way: threading a `delegate=` flag from the executor broke
    every existing arming double in the suite (18 tests, `TypeError:
    _fake_enable_compiled() got an unexpected keyword argument`). The decision
    belongs to the arming brain — one place, no caller churn — and the
    parameter survives only so tests can force either shape.
    """
    import inspect

    params = inspect.signature(fleet_compiled_graphs.enable_compiled).parameters
    assert params["delegate"].default is None, (
        "delegate must default to None = 'ask the policy', never to a value a "
        "caller is expected to supply")


# --------------------------------- 2. what the child cannot rediscover

def test_mint_modules_derives_from_the_declaring_module() -> None:
    from harness import toy_endpoints

    (spec,) = [
        s for s in extract_specs(toy_endpoints.Basics) if s.name == "echo"]
    assert _mint_modules(spec) == ("harness.toy_endpoints",)
    assert _mint_modules(SimpleNamespace(module="")) == ()


def test_background_mint_carries_the_modules_and_the_resolution(
    tmp_path: Path,
) -> None:
    bg = _BackgroundMint(
        spec=SimpleNamespace(name="gen"), instance=None, snapshots=None,
        pendings={}, pipes={},
        modules=("app",), slots={"pipeline": _slot("/cas/sdxl")})
    assert bg.modules == ("app",)
    assert bg.slots["pipeline"].path == "/cas/sdxl"
    assert bg.slots["pipeline"].ref == SDXL_REF
    # Defaulted, so the in-process route is untouched by their existence.
    plain = _BackgroundMint(
        spec=SimpleNamespace(name="gen"), instance=None, snapshots=None,
        pendings={}, pipes={})
    assert plain.modules == () and plain.slots == {}


# ------------------------------------- 3 + 4. the route and its outcomes

def _executor(tmp_path: Path) -> Executor:
    async def _send(msg: Any) -> None:
        pass

    store = ModelStore(_send, cache_dir=tmp_path / "cas")
    return Executor([], _send, store=store)


class _Act:
    def __init__(self) -> None:
        self.phases: List[str] = []

    def phase(self, phase: str, step: int = 0, total: int = 0) -> None:
        self.phases.append(phase)

    def note(self, detail: str) -> None:
        pass


def _wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, pipes: int = 1,
) -> tuple:
    """An executor + record + delegated _BackgroundMint over ``pipes`` objects
    that SHARE one pending (the qwen edit shape: two lanes, one family compiled graph)."""
    ex = _executor(tmp_path)
    spec = SimpleNamespace(
        name="gen", module="harness.toy_endpoints", lora_bucket=0,
        instance_key="k", compile=object())
    pending = _pending(tmp_path)
    objs = [_Pipe() for _ in range(pipes)]
    rec = _ClassRecord(cls=_Pipe)
    for i, pipe in enumerate(objs):
        rec.compile_targets[f"t{i}"] = _CompileTargetRecord(
            incarnation_id=f"t{i}", spec=spec, pipeline=pipe,
            pipeline_weight_lane="fp8", lora_bucket=0, contract_digest="cd")
    bg = _BackgroundMint(
        spec=spec, instance=None, snapshots=None,
        pendings={id(p): pending for p in objs},
        pipes={id(p): p for p in objs},
        modules=("harness.toy_endpoints",),
        slots={"pipeline": _slot(str(tmp_path / "snap"))})
    monkeypatch.setattr(ex, "_served_execution_lane", lambda s, instructed="": "fp8-w8a16")
    monkeypatch.setattr(ex, "_effective_config", lambda s, run=None: {"steps": 28})
    monkeypatch.setattr(
        executor_mod.mint_workers, "device_of", lambda pipe: 0)
    assert mint_workers.device_of is not None
    return ex, rec, bg, pending, objs


def test_the_delegated_route_mints_in_a_child_adopts_and_advertises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    adopted: List[Path] = []
    published: List[Any] = []

    def _adopt(pipe: Any, pending: Any, artifacts: Any) -> Any:
        # pgw#1176: the adopt takes the SET the child produced, one artifact
        # per graph class. A double taking a single Path models a call
        # production does not make — and `Path(a_tuple)` is how that surfaces.
        rows = [Path(a) for a in artifacts]
        adopted.extend(rows)
        return fleet_compiled_graphs.SelfMint(
            family="sdxl", compiled_graph_key="ek1-abc",
            ref="root/family-sdxl#ek1-abc", snapshot_digest="blake3:aa",
            artifact=rows[0])

    monkeypatch.setattr(fleet_compiled_graphs, "adopt_delegated_mint", _adopt)
    monkeypatch.setattr(
        fleet_compiled_graphs, "publish_self_mint", lambda p: published.append(p))
    ex, rec, bg, pending, (pipe,) = _wired(tmp_path, monkeypatch)
    monkeypatch.setattr(ex, "_refresh_compile_target", lambda t: None)
    monkeypatch.setattr(ex, "_bind_compile_guard", lambda r, t: True)

    act = _Act()
    asyncio.run(ex._delegated_mint_run(rec, bg, act))

    # A real child produced real bytes, and they were adopted through the
    # delivered-compiled graph path rather than trusted.
    assert adopted and adopted[0].read_bytes() == b"stub-compiled_graph-bytes"
    # gw#612: every sharer is covered, so the compiled graph ships.
    assert published == [pending]
    # Phase 4, shared with the in-process route: the target now advertises the
    # worker's OWN key (th#910's self-attested fence).
    target = rec.compile_targets["t0"]
    assert target.active_compile_ref == "root/family-sdxl#ek1-abc"
    assert target.active_compile_snapshot_digest == "blake3:aa"
    assert target.active_self_mint is True
    assert "finalize" in act.phases


def test_shared_holders_mint_one_compiled_graph_between_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two pipes, one pending, ONE child — and only the ARMED pipe advertises.

    pgw#1113 flips the second half of this assertion. One child is still the
    point and still holds: a per-pipe child would compile the same compiled graph twice
    on one card. But `build_compiled_graph` is handed ONE pipeline and
    `adopt_delegated_mint` installs the compiled graph on that ONE pipeline, so the
    sibling target never had those bytes — advertising a compiled ref for it
    was a claim about what it serves that was false at the moment it was made,
    and only `_bind_compile_guard`'s incidental `False` ("advertising eager")
    kept it off the wire.
    """
    spawns: List[Any] = []
    real = mint_delegate.build_compiled_graph

    async def _counting(task: Any, **kw: Any) -> Any:
        spawns.append(task)
        return await real(task, **kw)

    monkeypatch.setattr(mint_delegate, "build_compiled_graph", _counting)
    monkeypatch.setattr(
        fleet_compiled_graphs, "adopt_delegated_mint",
        lambda pipe, pending, artifacts: fleet_compiled_graphs.SelfMint(
            family="sdxl", compiled_graph_key="k", ref="r", snapshot_digest="d",
            artifact=Path(list(artifacts)[0])))
    monkeypatch.setattr(fleet_compiled_graphs, "publish_self_mint", lambda p: None)
    ex, rec, bg, _pending, objs = _wired(tmp_path, monkeypatch, pipes=2)
    monkeypatch.setattr(ex, "_refresh_compile_target", lambda t: None)
    monkeypatch.setattr(ex, "_bind_compile_guard", lambda r, t: True)

    asyncio.run(ex._delegated_mint_run(rec, bg, _Act()))
    assert len(spawns) == 1, "one shared compiled_graph must mean one child process"
    armed_pipe = spawns[0].pipe
    advertised = {
        name: t.active_compile_ref for name, t in rec.compile_targets.items()}
    assert [
        ref for name, ref in advertised.items()
        if rec.compile_targets[name].pipeline is armed_pipe] == ["r"]
    assert [
        ref for name, ref in advertised.items()
        if rec.compile_targets[name].pipeline is not armed_pipe] == [""], (
        "a pipe the mint never armed must advertise nothing: the compiled_graph was "
        "installed on one pipeline and only that pipeline serves it")


# pgw#1175 / §4.33: `test_a_decline_raises_MintDeclined_so_the_tier_stays_eager`
# is DELETED with the exception it named. `_MintDeclined` existed so the
# wrapper could read "the card had no room" off an EXCEPTION TYPE — a verdict
# reached before any child ran, from `mint_budget.co_residency`, whose leading
# term charged a weight-free child for the parent's resident weights. There is
# no advance verdict any more, so there is no outcome to distinguish: a mint
# either runs or it fails, and a failed one is a plain `Exception` covered by
# `test_a_child_that_dies_leaves_the_worker_serving` below.


def test_an_abandon_raises_MintAbandoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adopt-on-arm, vacate and shutdown all abandon a mint. None of them is a
    broken worker, so none may look like one."""
    monkeypatch.setenv("MINT_STUB_MODE", "silent")
    monkeypatch.setenv("MINT_STUB_SECONDS", "120")
    ex, rec, bg, _pending, _objs = _wired(tmp_path, monkeypatch)

    async def _go() -> None:
        task = asyncio.ensure_future(ex._delegated_mint_run(rec, bg, _Act()))
        await asyncio.sleep(1.0)
        bg.abandon.set()
        await task

    with pytest.raises(_MintAbandoned):
        asyncio.run(_go())


def test_a_dead_child_raises_a_plain_exception_and_never_advertises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """th#1299 inverted at the wiring level: the mint dies, the worker keeps
    serving, and nothing unproven is ever advertised (gw#586)."""
    monkeypatch.setenv("MINT_STUB_MODE", "sigkill")
    ex, rec, bg, _pending, _objs = _wired(tmp_path, monkeypatch)
    with pytest.raises(Exception) as err:
        asyncio.run(ex._delegated_mint_run(rec, bg, _Act()))
    assert not isinstance(err.value, _MintAbandoned)
    assert rec.compile_targets["t0"].active_compile_ref == ""
    assert rec.compile_targets["t0"].active_self_mint is False


def test_the_mint_driver_has_exactly_one_route_and_it_is_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1010: there is no in-process route left to fall through to.

    The branch this test used to police (`_background_mint_run` -> delegated vs
    seed/drain/prove/pack in the serving process, th#1299) is gone because the
    in-process half is deleted — it only ever built a dynamo compiled graph. The
    invariant survives as a stronger one: the driver reaches the child, and
    `_warmup_plan` — the first thing the in-process route did — is not
    reachable from it at all.
    """
    ex, rec, bg, _pending, _objs = _wired(tmp_path, monkeypatch)
    took: List[str] = []

    async def _delegated(r: Any, b: Any, a: Any) -> None:
        took.append("delegated")

    def _plan_must_not_run(spec: Any, r: Any) -> Any:
        took.append("in-process")
        return [], []

    monkeypatch.setattr(ex, "_delegated_mint_run", _delegated)
    monkeypatch.setattr(ex, "_warmup_plan", _plan_must_not_run)
    asyncio.run(ex._background_mint_run(rec, bg, _Act()))
    assert took == ["delegated"], (
        "the mint driver ran a compile in the serving process — the loop that "
        "carries the beat (th#1299)")
