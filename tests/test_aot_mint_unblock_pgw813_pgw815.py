"""pgw#813 + pgw#815 — the two walls between the fleet and its first AOT cell.

**pgw#813** (measured, gen-worker 0.80.0, real L4, chaos, pod `149ku1h1pgjq7q`):

    aot_cell_discovery  miss                     family=sdxl lane=w8a8-lora64
    self_mint_skipped   aot_requires_delegation  "out-of-process minting is
                        disabled (GEN_WORKER_MINT_IN_PROCESS or eager-first
                        off) and an AOTI export has no eager tier..."
    self_mint_started   dynamo                   ... armed an in-process capture

Neither named cause was true on that pod — no env was set. The operative
refusal was `fleet_cells.delegatable` reading `mandatory_serving(pipe)` as
"cannot serve eager". It cannot: `_Fp8ScaledLinear.forward` is a complete
`torch._scaled_mm` forward, the fleet's cold-boot ladder measures w8a8 eager
serving, and pgw#672/#673 already made mandatory lanes DEGRADE to eager loudly
instead of raising. With the plain lane held on dynamo by #730, that left NO
lane on which a serving pod could mint an AOT cell — which is why
`aot_mint_phases` has zero rows platform-wide.

A second, independent blocker sat one layer up: `_eager_first_eligible`
demanded a hot-swap ROUTER on every pending pipe. A delegated pending never
has one (nothing is armed on its pipe, by construction), so every delegated
mint failed the test and was discarded — pgw#784's out-of-process route could
not run on ANY lane, quantized or not.

**pgw#815** (same pod): a 24m22s mint walked `seal_publish -> finalize
completed` and produced zero cells, zero receipts, no local arm, no
`self_mint_publish`, no abort, no error. Three of those are structural and are
pinned here: no success event exists at any publish terminus, `publish_self_mint`
and `withhold_self_mint_publish` both return BARE when nothing was packed, and
a boot can reach readiness with a mint obligation that touched no terminus at
all.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import compile_cache, fleet_cells, mint_delegate
from gen_worker.api.decorators import Compile, Dim, GraphClass, Input
from gen_worker.api.export_contract import (
    register_export_declaration,
    reset_export_declarations,
)
from gen_worker import config as gw_config
from gen_worker.cell_adopt import AdoptOutcome

FAMILY = "sdxl"


class _Pipe:
    pass


@dataclass
class _Cfg:
    family: str = FAMILY
    lora_bucket: int = 64
    shapes: Tuple[Tuple[int, int], ...] = ((1024, 1024),)
    targets: Tuple[str, ...] = ("unet",)
    text_lens: Tuple[int, ...] = (77,)
    guidance_scales: Tuple[float, ...] = (1.0, 5.0)
    regional: bool = False


class _Publisher:
    """A publish sink that behaves like the hub: it keeps a store."""

    base_url = "http://hub.invalid"

    def __init__(self, fail: BaseException | None = None) -> None:
        self.store: Dict[str, Dict[str, Any]] = {}
        self.fail = fail
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def enabled(self) -> bool:
        return True

    def worker_jwt(self) -> str:
        return "jwt"

    def publish(self, family: str, artifact: Path, meta: dict,
                mint_duration_ms: int = 0) -> str:
        self.started.set()
        self.release.wait(timeout=30)
        if self.fail is not None:
            raise self.fail
        key = str(meta.get("cell_key") or "")
        self.store[key] = {
            "family": family,
            "bytes": Path(artifact).stat().st_size,
            "meta": dict(meta),
            "mint_duration_ms": mint_duration_ms,
        }
        return f"chk-{key[:8]}"


def _declaration(family: str = FAMILY) -> Compile:
    return Compile(
        family=family,
        targets=("unet",),
        text_len=77,
        shapes=((1024, 1024),),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", 4, 128, 128), dtype="bfloat16"),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    )


@pytest.fixture(autouse=True)
def _clean_declarations() -> Any:
    reset_export_declarations()
    yield
    reset_export_declarations()


@pytest.fixture()
def _events(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str, str]]:
    seen: List[Tuple[str, str, str]] = []

    def _sink(kind: str, detail: str, phase: str = "",
              duration_ms: int = 0) -> None:
        seen.append((kind, phase, detail))

    monkeypatch.setattr(fleet_cells.activity_mod, "emit_event", _sink)
    monkeypatch.setattr(mint_delegate.activity_mod, "emit_event", _sink)
    return seen


def _phases(events: List[Tuple[str, str, str]], kind: str) -> List[str]:
    return [phase for k, phase, _ in events if k == kind]


@pytest.fixture()
def _w8a8_miss(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The measured pod, reduced: prefer_aot armed, a real w8a8 lane, a
    resolvable compile target, CUDA + toolchain present, and no cell.

    `mandatory_serving` is deliberately NOT stubbed — the w8a8 weight-lane
    stamp really does classify this pipe as mandatory, and that is the whole
    point: mandatory must stop meaning "cannot serve eager".
    """
    from gen_worker import aot_cells

    monkeypatch.setenv("GEN_WORKER_PREFER_AOT", "1")
    monkeypatch.delenv("GEN_WORKER_MINT_IN_PROCESS", raising=False)
    monkeypatch.delenv("GEN_WORKER_EAGER_FIRST_BOOT", raising=False)
    gw_config.reload_for_test()
    monkeypatch.setattr(aot_cells, "discover", lambda *a, **k: None)
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: AdoptOutcome.miss("no_cell"))
    monkeypatch.setattr(fleet_cells.cc, "has_compile_target", lambda p, c: True)
    monkeypatch.setattr(fleet_cells.cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(fleet_cells.cc, "delivered_cell_seeded", lambda: False)
    monkeypatch.setattr(fleet_cells.cc, "apply_lora_lane", lambda p, b: None)
    monkeypatch.setattr(fleet_cells.cc, "drop_lora_lane", lambda p: None)
    monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
    monkeypatch.setattr(fleet_cells, "_PENDING", {})
    monkeypatch.setattr(
        fleet_cells.cell_key, "compute",
        lambda *a, **k: type("_K", (), {"digest": "ck5-" + "a" * 56})())
    monkeypatch.setattr(
        fleet_cells.cc, "begin_fleet_mint", lambda p, c, capture: None)
    # THE lane: sdxl's mixed fp8 checkpoint stamps `w8a8-lora64` (pgw#686 cell
    # identity), which is what `mandatory_serving` falls back to without hub
    # lane evidence — exactly the measured pod's state.
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane",
        lambda pipe: "w8a8-lora64")
    # The torch-VERSION floor for the lifted-LoRA fork is not what these tests
    # measure and torch is not importable on a CPU dev box; `lane_admitted`
    # (the #730 hold — the OTHER half of "no lane can mint") stays REAL.
    from gen_worker import aot_mint

    monkeypatch.setattr(aot_mint, "lifted_torch_gap", lambda spec: "")
    yield
    gw_config.reload_for_test()


def _arm(**kw: Any) -> Any:
    return fleet_cells.enable_compiled(
        _Pipe(), _Cfg(), publisher=_Publisher(), **kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# pgw#813 — the w8a8 lane is eager-serveable, therefore delegatable
# ---------------------------------------------------------------------------


def test_a_w8a8_pipeline_has_an_eager_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The predicate that replaces the category error.

    `mandatory_serving` answers "is the COMPILED tier the intended production
    tier"; it is NOT the answer to "can this object answer a forward with
    nothing armed". A w8a8 pipeline can — that is what the fleet does all day.
    """
    pipe = _Pipe()
    from gen_worker.models import loading as loading_mod

    monkeypatch.setattr(
        loading_mod, "pipeline_weight_lane", lambda p: "w8a8-lora64")

    assert compile_cache.mandatory_serving(pipe) is True
    assert compile_cache.eager_tier_available(pipe) is True


def test_the_w8a8_lane_is_delegatable(monkeypatch: pytest.MonkeyPatch) -> None:
    """RED at HEAD: `delegatable` refused `mandatory_serving(pipe)`, so the
    lane Paul ruled AOT-first was the one lane that could never get a
    delegated minter."""
    from gen_worker.models import loading as loading_mod

    monkeypatch.setattr(
        loading_mod, "pipeline_weight_lane", lambda p: "w8a8-lora64")
    assert fleet_cells.delegation_refusal(_Pipe(), _Cfg()) == ""
    assert fleet_cells.delegatable(_Pipe(), _Cfg()) is True


def test_an_armed_non_eager_backend_still_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The genuine sub-case, scoped to itself: when an AOTI export or a TRT
    engine has REPLACED the forward there is no eager tier to serve from."""
    from gen_worker import aot_serve

    monkeypatch.setattr(aot_serve, "is_armed", lambda p: True)
    assert compile_cache.eager_tier_available(_Pipe()) is False
    assert (fleet_cells.delegation_refusal(_Pipe(), _Cfg())
            == fleet_cells.REFUSAL_NO_EAGER_TIER)


def test_a_w8a8_miss_mints_AOT_and_not_dynamo(
    _w8a8_miss: None, _events: List[Tuple[str, str, str]],
) -> None:
    """THE issue. RED at HEAD: `self_mint_skipped aot_requires_delegation`
    followed by a dynamo cell AOT discovery can never adopt."""
    register_export_declaration(_declaration())

    outcome = _arm()

    pending = outcome.self_mint
    assert pending is not None, "the miss produced no mint at all"
    assert pending.recipe == fleet_cells.RECIPE_AOT, (
        "a w8a8 miss with prefer_aot armed and a declaration registered must "
        "mint an AOT cell; a dynamo artifact will never satisfy AOT discovery")
    assert pending.delegated is True
    assert "aot_requires_delegation" not in _phases(_events, "self_mint_skipped")
    assert ("self_mint_started", "aot") in [(k, p) for k, p, _ in _events]


def test_delegation_declines_name_their_TRUE_cause(
    _w8a8_miss: None, _events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED at HEAD: every refusal declined under one phase carrying a
    hand-written either/or sentence, so the wire could not tell an operator
    kill switch from a pipeline classification."""
    register_export_declaration(_declaration())

    monkeypatch.setenv("GEN_WORKER_MINT_IN_PROCESS", "1")
    _arm()
    assert "aot_mint_forced_in_process" in _phases(_events, "self_mint_skipped")

    _events.clear()
    monkeypatch.delenv("GEN_WORKER_MINT_IN_PROCESS")
    monkeypatch.setenv("GEN_WORKER_EAGER_FIRST_BOOT", "0")
    fleet_cells._PENDING.clear()
    _arm()
    assert "aot_eager_first_disabled" in _phases(_events, "self_mint_skipped")

    # pgw#846: `Compile.regional` is the dynamo/JIT per-block knob (ie#381)
    # and the AOT mint ignores it — regional EXPORT is retired, the recipe is
    # always whole-graph. A family that declares it must neither decline
    # delegation nor change the mint shape.
    _events.clear()
    monkeypatch.delenv("GEN_WORKER_EAGER_FIRST_BOOT")
    fleet_cells._PENDING.clear()
    fleet_cells.enable_compiled(
        _Pipe(), _Cfg(regional=True), publisher=_Publisher())  # type: ignore[arg-type]
    assert "aot_regional_targets" not in _phases(_events, "self_mint_skipped")
    assert fleet_cells.delegation_refusal(_Pipe(), _Cfg(regional=True)) == ""


def test_mint_delegate_names_its_own_refusals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEN_WORKER_MINT_IN_PROCESS", raising=False)
    monkeypatch.delenv("GEN_WORKER_EAGER_FIRST_BOOT", raising=False)
    assert mint_delegate.delegation_refusal() == ""
    monkeypatch.setenv("GEN_WORKER_EAGER_FIRST_BOOT", "0")
    assert (mint_delegate.delegation_refusal()
            == mint_delegate.REFUSAL_EAGER_FIRST_DISABLED)
    monkeypatch.setenv("GEN_WORKER_MINT_IN_PROCESS", "1")
    assert (mint_delegate.delegation_refusal()
            == mint_delegate.REFUSAL_IN_PROCESS_FORCED)


# ---------------------------------------------------------------------------
# pgw#813 second blocker — a delegated pending has no router, by construction
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    pipeline: Any
    slots: Tuple[str, ...] = ()


@dataclass
class _Inj:
    compile_objects: List[Any] = field(default_factory=list)
    pending_self_mints: Dict[int, Any] = field(default_factory=dict)
    active_compile_artifacts: Dict[int, Any] = field(default_factory=dict)


def _executor(tmp_path: Path) -> Any:
    from gen_worker.executor import Executor, ModelStore

    async def _send(msg: Any) -> None:
        pass

    return Executor([], _send, store=ModelStore(_send, cache_dir=tmp_path / "cas"))


def test_eager_first_admits_a_DELEGATED_pending_with_no_router(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED at HEAD: `router is None -> return False`, and a delegated pending
    can never have a router because nothing is armed on its pipe. Every
    delegated mint therefore failed this predicate and was discarded a few
    lines later — pgw#784 could not run on any lane."""
    from gen_worker.models import loading as loading_mod

    monkeypatch.delenv("GEN_WORKER_EAGER_FIRST_BOOT", raising=False)
    monkeypatch.setattr(
        loading_mod, "pipeline_weight_lane", lambda p: "w8a8-lora64")

    ex = _executor(tmp_path)
    pipe = _Pipe()
    cfg = _Cfg()
    spec = type("_S", (), {
        "cls": None, "compile_cell": lambda self: cfg, "models": {},
        "name": "generate",
    })()
    pending = fleet_cells.PendingSelfMint(
        family=FAMILY, cell_key="ck5-" + "a" * 56, ref="r", cfg=cfg,
        target=tmp_path / "c.tar.gz", capture_dir=tmp_path / "cap",
        mint_root=tmp_path, publisher=None, delegated=True,
        recipe=fleet_cells.RECIPE_AOT)
    inj = _Inj(compile_objects=[_Candidate(pipe)],
               pending_self_mints={id(pipe): pending})

    assert ex._eager_first_eligible(spec, inj) is True  # type: ignore[arg-type]


def test_eager_first_still_requires_a_router_for_an_IN_PROCESS_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-process shape is unchanged: it serves eager THROUGH the pgw#622
    router, so no router means no eager tier."""
    from gen_worker.models import loading as loading_mod

    monkeypatch.delenv("GEN_WORKER_EAGER_FIRST_BOOT", raising=False)
    monkeypatch.setattr(loading_mod, "pipeline_weight_lane", lambda p: "")

    ex = _executor(tmp_path)
    pipe = _Pipe()
    cfg = _Cfg()
    spec = type("_S", (), {
        "cls": None, "compile_cell": lambda self: cfg, "models": {},
        "name": "generate",
    })()
    pending = fleet_cells.PendingSelfMint(
        family=FAMILY, cell_key="ck5-" + "b" * 56, ref="r", cfg=cfg,
        target=tmp_path / "c.tar.gz", capture_dir=tmp_path / "cap",
        mint_root=tmp_path, publisher=None, delegated=False)
    inj = _Inj(compile_objects=[_Candidate(pipe)],
               pending_self_mints={id(pipe): pending})

    assert ex._eager_first_eligible(spec, inj) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# pgw#815 — every publish terminus is typed
# ---------------------------------------------------------------------------


def _finalized_pending(tmp_path: Path, publisher: Any) -> Any:
    """A pending that has been packed — a real file, a real key, real bytes."""
    key = "ck5-" + "c" * 56
    target = tmp_path / "cell.tar.gz"
    target.write_bytes(b"x" * 4096)
    pending = fleet_cells.PendingSelfMint(
        family=FAMILY, cell_key=key, ref=f"root/family-{FAMILY}#{key}",
        cfg=_Cfg(), target=target, capture_dir=tmp_path / "cap",
        mint_root=tmp_path / "root", publisher=publisher)
    pending.mint_root.mkdir(parents=True, exist_ok=True)
    pending._state["minted"] = fleet_cells.SelfMint(
        family=FAMILY, cell_key=key, ref=pending.ref,
        snapshot_digest="blake3:deadbeef", artifact=target)
    pending._state["meta"] = {"cell_key": key, "family": FAMILY}
    return pending


def test_a_SUCCESSFUL_publish_is_a_typed_event_and_reaches_the_store(
    tmp_path: Path, _events: List[Tuple[str, str, str]],
) -> None:
    """RED at HEAD: no `self_mint_publish` event exists anywhere in the tree.
    A publish that worked and a publish thread killed mid-upload when the pod
    retired were the SAME observation — silence — which is exactly how a
    24-minute L4 mint left no artifact and no explanation."""
    publisher = _Publisher()
    pending = _finalized_pending(tmp_path, publisher)

    fleet_cells.publish_self_mint(pending)
    _join_publishes()

    assert pending.cell_key in publisher.store, "the cell never reached the store"
    phases = _phases(_events, "self_mint_publish")
    assert "started" in phases, "an upload beginning must be on the wire"
    assert "published" in phases, "a successful publish must be on the wire"
    assert fleet_cells.terminus_of(pending) == fleet_cells.TERMINUS_PUBLISHING


def test_a_publish_in_flight_is_an_observable_fact(
    tmp_path: Path, _events: List[Tuple[str, str, str]],
) -> None:
    """RED at HEAD: the publish was a fire-and-forget daemon thread with no
    registry, so 'still uploading' and 'never happened' were indistinguishable
    from outside the process."""
    publisher = _Publisher()
    publisher.release.clear()
    pending = _finalized_pending(tmp_path, publisher)

    fleet_cells.publish_self_mint(pending)
    assert publisher.started.wait(timeout=10)
    assert pending.cell_key in fleet_cells.publishes_in_flight()
    publisher.release.set()
    _join_publishes()
    assert pending.cell_key not in fleet_cells.publishes_in_flight()


def test_a_publish_gate_with_nothing_packed_is_NAMED(
    tmp_path: Path, _events: List[Tuple[str, str, str]],
) -> None:
    """RED at HEAD: a bare `return`. The executor's publish gate runs only for
    pendings it believes it packed, so reaching it with nothing packed is a
    real defect and must not be a no-op."""
    pending = fleet_cells.PendingSelfMint(
        family=FAMILY, cell_key="ck5-" + "d" * 56, ref="r", cfg=_Cfg(),
        target=tmp_path / "c.tar.gz", capture_dir=tmp_path / "cap",
        mint_root=tmp_path / "root2", publisher=_Publisher())

    fleet_cells.publish_self_mint(pending)

    assert "nothing_to_publish" in _phases(
        _events, "self_mint_publish_withheld")


def test_a_withhold_with_nothing_packed_is_NAMED(
    tmp_path: Path, _events: List[Tuple[str, str, str]],
) -> None:
    pending = fleet_cells.PendingSelfMint(
        family=FAMILY, cell_key="ck5-" + "e" * 56, ref="r", cfg=_Cfg(),
        target=tmp_path / "c.tar.gz", capture_dir=tmp_path / "cap",
        mint_root=tmp_path / "root3", publisher=_Publisher())

    fleet_cells.withhold_self_mint_publish(pending, "sibling never exercised")

    assert "nothing_to_publish" in _phases(
        _events, "self_mint_publish_withheld")
    assert fleet_cells.terminus_of(pending) == fleet_cells.TERMINUS_ABANDONED


def test_a_failed_publish_still_names_the_key(
    tmp_path: Path, _events: List[Tuple[str, str, str]],
) -> None:
    publisher = _Publisher(fail=RuntimeError("hub 502"))
    pending = _finalized_pending(tmp_path, publisher)

    fleet_cells.publish_self_mint(pending)
    _join_publishes()

    failures = [d for k, p, d in _events if k == "self_mint_publish_failed"]
    assert failures and pending.cell_key in failures[0]
    assert pending.cell_key not in fleet_cells.publishes_in_flight()


def test_a_boot_that_resolves_NOTHING_confesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE pgw#815 assertion: `finalize completed` must be unreachable while a
    mint obligation is unresolved. On the measured pod the publish gate lived
    entirely inside `if proves_inductor or proves_exported:`, so a boot that
    answered "nothing proves by FX or export" walked past every terminus and
    said nothing at all."""
    seen: List[Tuple[str, str, str]] = []
    monkeypatch.setattr(
        fleet_cells.activity_mod, "emit_event",
        lambda kind, detail, phase="", duration_ms=0: seen.append(
            (kind, phase, detail)))
    import gen_worker.executor as executor_mod

    monkeypatch.setattr(
        executor_mod.activity_mod, "emit_event",
        lambda kind, detail, phase="", duration_ms=0: seen.append(
            (kind, phase, detail)))

    ex = _executor(tmp_path)
    spec = type("_S", (), {"name": "generate"})()
    pending = fleet_cells.PendingSelfMint(
        family=FAMILY, cell_key="ck5-" + "f" * 56, ref="r", cfg=_Cfg(),
        target=tmp_path / "c.tar.gz", capture_dir=tmp_path / "cap",
        mint_root=tmp_path / "root4", publisher=_Publisher())
    pending.mint_root.mkdir(parents=True, exist_ok=True)

    ex._assert_mint_termini(spec, [pending])  # type: ignore[arg-type]

    assert ("self_mint_abort", "no_terminus") in [(k, p) for k, p, _ in seen]
    assert fleet_cells.terminus_of(pending) == fleet_cells.TERMINUS_ABANDONED


def test_a_resolved_mint_does_not_trip_the_assertion(
    tmp_path: Path, _events: List[Tuple[str, str, str]],
) -> None:
    publisher = _Publisher()
    pending = _finalized_pending(tmp_path, publisher)
    fleet_cells.publish_self_mint(pending)
    _join_publishes()

    ex = _executor(tmp_path)
    spec = type("_S", (), {"name": "generate"})()
    _events.clear()
    ex._assert_mint_termini(spec, [pending])  # type: ignore[arg-type]

    assert "no_terminus" not in _phases(_events, "self_mint_abort")


def _join_publishes(deadline_s: float = 30.0) -> None:
    """Wait on the publish threads themselves — never a fixed sleep."""
    end = time.monotonic() + deadline_s
    for t in threading.enumerate():
        if t.name != "cell-publish":
            continue
        t.join(timeout=max(0.0, end - time.monotonic()))
    assert not [t for t in threading.enumerate()
                if t.name == "cell-publish" and t.is_alive()]
