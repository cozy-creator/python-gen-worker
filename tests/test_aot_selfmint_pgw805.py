"""pgw#805 — an AOT cell-discovery MISS must start a mint, or refuse BY NAME.

The defect this pins, measured on five real 0.78.0 L4 pods (hub-dispatched,
`GEN_WORKER_PREFER_AOT` armed release-scoped, `Compile(family="sdxl",
targets=("unet",))` declared, compile target advertised, discovery working
through a 200 listing, `cell_mint_hold_granted` on every pod, trickle traffic
earning pgw#677 background turns):

    self_mint_compile  load            running
    aot_cell_discovery miss            completed   family=sdxl lane=lora64
    self_mint_compile  warmup_forward  running
    self_mint_compile  warmup_forward  completed
    (nothing further, ever)

No mint, no `self_mint_abort`, no `self_mint_skipped` — no refusal of ANY
kind. Three separate wires were missing, and each gets a test here:

1. **No producer.** `aot_mint.mint` was reachable only from
   `python -m gen_worker.aot_mint`; nothing on the serving path imported it.
   A miss could only fall through to the DYNAMO self-mint, whose artifact
   kind `aot_cells._candidates` rejects — so the next pod missed identically,
   forever.
2. **No declaration.** `aot_mint.mint` refuses a family with no registered
   export declaration, and registration only happened when a mint REQUEST
   named a declaration module. A serving pod loads its endpoint and nothing
   else.
3. **Silence.** Every not-mint exit was a `logger.info`, and a serve pod
   exposes no logs (pgw#760).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import aot_cells, aot_mint, fleet_cells, mint_delegate
from gen_worker.api.decorators import Compile, Dim, GraphClass, Input
from gen_worker.api.export_contract import (
    export_declaration, register_export_declaration, reset_export_declarations,
)
from gen_worker.config import get_settings

FAMILY = "sdxl"


# ---------------------------------------------------------------------------
# Doubles — the arming brain's real entry point, everything around it stubbed
# ---------------------------------------------------------------------------


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
    base_url = "http://hub.invalid"

    def enabled(self) -> bool:
        return True

    def worker_jwt(self) -> str:
        return "jwt"


def _declaration(family: str = FAMILY) -> Compile:
    """A minimal but REAL export declaration (the pgw#739 vocabulary)."""
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

    def _sink(kind: str, detail: str, phase: str = "", duration_ms: int = 0) -> None:
        seen.append((kind, phase, detail))

    monkeypatch.setattr(fleet_cells.activity_mod, "emit_event", _sink)
    monkeypatch.setattr(mint_delegate.activity_mod, "emit_event", _sink)
    return seen


@pytest.fixture()
def _miss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A real AOT discovery MISS on a mint-capable pod.

    Everything the five measured pods had: the flag armed, a publisher, a
    resolvable compile target, CUDA and a toolchain — and no cell.
    """
    monkeypatch.setenv("GEN_WORKER_PREFER_AOT", "1")
    get_settings.cache_clear()
    monkeypatch.setattr(aot_cells, "discover", lambda *a, **k: None)
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: False)
    monkeypatch.setattr(fleet_cells.cc, "has_compile_target", lambda p, c: True)
    monkeypatch.setattr(fleet_cells.cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(fleet_cells.cc, "delivered_cell_seeded", lambda: False)
    monkeypatch.setattr(fleet_cells.cc, "apply_lora_lane", lambda p, b: None)
    monkeypatch.setattr(fleet_cells.cc, "drop_lora_lane", lambda p: None)
    monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
    monkeypatch.setattr(fleet_cells, "_PENDING", {})
    # No GPU on a dev box: the ck5 `sm` axis is a real-runtime fact, and this
    # test is about the RECIPE decision, not key computation.
    monkeypatch.setattr(
        fleet_cells.cell_key, "compute",
        lambda *a, **k: type("_K", (), {"digest": "ck5-" + "a" * 56})())
    # w8a8 is the migration's first lane (pgw#704 parity); its mandatory
    # serving refusal is a separate policy, exercised in its own test.
    monkeypatch.setattr(fleet_cells.cc, "mandatory_serving", lambda p: False)
    monkeypatch.setattr(
        fleet_cells.cc, "begin_fleet_mint", lambda p, c, capture: None)
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "w8a8")
    yield
    get_settings.cache_clear()


def _arm(**kw: Any) -> Any:
    return fleet_cells.enable_compiled(
        _Pipe(), _Cfg(), publisher=_Publisher(), **kw)  # type: ignore[arg-type]


def _phases(events: List[Tuple[str, str, str]], kind: str) -> List[str]:
    return [phase for k, phase, _ in events if k == kind]


# ---------------------------------------------------------------------------
# 1. THE MISSING WIRE — a miss must enqueue an AOT mint
# ---------------------------------------------------------------------------


def test_aot_discovery_miss_enqueues_an_aot_mint(
    _miss: None, _events: List[Tuple[str, str, str]],
) -> None:
    """RED at HEAD: the pending's recipe was always the dynamo capture, whose
    artifact kind AOT discovery rejects — so a fleet with prefer_aot armed
    could never produce the cell it was looking for."""
    register_export_declaration(_declaration())

    outcome = _arm(delegate=True)

    pending = outcome.self_mint
    assert pending is not None, "a discovery miss produced no mint at all"
    assert pending.recipe == fleet_cells.RECIPE_AOT
    assert pending.delegated is True, (
        "an AOTI export holds the GPU with no router to yield through; it "
        "must never run in the serving process")
    assert not outcome.armed, "the live pipe serves EAGER while the child mints"
    assert ("self_mint_started", "aot") in [(k, p) for k, p, _ in _events]


def test_the_mint_recipe_reaches_the_child_process(
    _miss: None, _events: List[Tuple[str, str, str]], tmp_path: Path,
) -> None:
    """The recipe must survive the process boundary: the child never guesses
    the artifact kind."""
    register_export_declaration(_declaration())
    pending = _arm(delegate=True).self_mint
    assert pending is not None

    task = mint_delegate.MintTask(
        pending=pending, pipe=_Pipe(), function="generate",
        modules=("sdxl.main",), weight_lane="w8a8")
    request = mint_delegate.build_request(
        task, workdir=tmp_path, cap_bytes=1 << 30)
    assert request.recipe == "aot"


# ---------------------------------------------------------------------------
# 2. NO DECLARATION — refused by name, never silently
# ---------------------------------------------------------------------------


def test_a_family_with_no_export_declaration_refuses_by_name(
    _miss: None, _events: List[Tuple[str, str, str]],
) -> None:
    outcome = _arm(delegate=True)

    assert export_declaration(FAMILY) is None
    pending = outcome.self_mint
    assert pending is not None and pending.recipe == fleet_cells.RECIPE_DYNAMO
    assert "no_export_declaration" in _phases(_events, "self_mint_skipped")


def test_the_endpoints_own_compile_block_registers_the_declaration() -> None:
    """The declaration travels with the endpoint that owns it (pgw#805).

    Before this, `export_declaration` resolved only when a MINT REQUEST named
    a declaration module — a concept a serving pod has no access to.
    """
    from gen_worker.registry import register_declared_exports

    decl = _declaration()
    spec = type("_S", (), {"compile": decl, "cls": object})()
    assert register_declared_exports([spec]) == (FAMILY,)  # type: ignore[list-item]
    assert export_declaration(FAMILY) is decl
    # Idempotent: a second collection pass of the same endpoint is a no-op,
    # never a conflicting-registration raise that would kill endpoint walking.
    assert register_declared_exports([spec]) == ()  # type: ignore[list-item]


def test_a_compile_block_without_graph_classes_is_not_a_declaration() -> None:
    """Every endpoint declares `compile=`; only the ones carrying the pgw#739
    class vocabulary are export declarations."""
    from gen_worker.registry import register_declared_exports

    plain = Compile(family=FAMILY, targets=("unet",), shapes=((1024, 1024),),
                    text_len=77)
    spec = type("_S", (), {"compile": plain, "cls": object})()
    assert register_declared_exports([spec]) == ()  # type: ignore[list-item]
    assert export_declaration(FAMILY) is None


# ---------------------------------------------------------------------------
# 3. THE HELD LANES — #730's dynamo hold is a decision, and it must be audible
# ---------------------------------------------------------------------------


def test_a_lane_held_on_dynamo_declines_by_name(
    _miss: None, _events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This is the five L4 pods' ACTUAL state: `pipeline_weight_lane` was the
    plain lane, which #730 holds on dynamo (6.9-7.0% slower under AOTI). The
    hold is right; being unable to say so is the defect."""
    register_export_declaration(_declaration())
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "")

    pending = _arm(delegate=True).self_mint

    assert pending is not None and pending.recipe == fleet_cells.RECIPE_DYNAMO
    assert "aot_lane_regressed" in _phases(_events, "self_mint_skipped")


def test_an_in_process_only_pod_declines_the_aot_recipe_by_name(
    _miss: None, _events: List[Tuple[str, str, str]],
) -> None:
    register_export_declaration(_declaration())

    outcome = _arm(delegate=False)

    pending = outcome.self_mint
    assert pending is not None and pending.recipe == fleet_cells.RECIPE_DYNAMO
    # pgw#813 sharpened the vocabulary: the generic `aot_requires_delegation`
    # carried a hand-written either/or sentence and could not distinguish an
    # operator kill switch from a pipeline classification. Each cause now
    # declines under its own phase; this arm is the forced in-process one.
    assert "aot_mint_forced_in_process" in _phases(
        _events, "self_mint_skipped")


# ---------------------------------------------------------------------------
# 4. SILENCE IS THE DEFECT CLASS — every eager degrade names itself
# ---------------------------------------------------------------------------


def test_degrading_to_eager_is_never_silent(
    _miss: None, _events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED at HEAD: `_fail_closed`'s plain-lane arm was a bare `logger.info`,
    and a serve pod exposes no logs (pgw#760)."""
    monkeypatch.setattr(fleet_cells.cc, "toolchain_present", lambda: False)

    outcome = _arm(delegate=True)

    assert not outcome.armed and outcome.self_mint is None
    skipped = [
        detail for kind, phase, detail in _events
        if kind == "self_mint_skipped" and phase == "mint_unavailable"]
    assert skipped, "a pod that mints nothing and refuses nothing is unreadable"
    assert "no C compiler" in skipped[0]


# ---------------------------------------------------------------------------
# 5. TELEMETRY — aot_mint_phases finally gets a producer
# ---------------------------------------------------------------------------


def test_the_parent_reemits_the_childs_aot_phase_table(
    _events: List[Tuple[str, str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`aot_mint` emits `aot_mint_phases` from the mint CHILD, which holds no
    orchestrator session — so those events reached nothing and the hub table
    has been empty on both stacks since th#1322 shipped the column."""
    rows: List[Tuple[str, str, int]] = []
    monkeypatch.setattr(
        aot_mint.activity_mod if hasattr(aot_mint, "activity_mod") else aot_mint,
        "__name__", aot_mint.__name__)  # no-op; keeps the import explicit

    from gen_worker import activity as activity_mod

    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, phase="", duration_ms=0: rows.append(
            (kind, phase, duration_ms)))

    table = {
        "v": 1, "n_entries": 2,
        "totals": {"total_s": 900.0, "export_s": 60.0, "compile_s": 800.0},
        "phases": {}, "autotune": {},
        "entries": {
            "unet/cfg": {"export_s": 30.0, "compile_s": 400.0},
            "unet/nocfg": {"export_s": 30.0, "compile_s": 400.0},
        },
    }
    aot_mint.emit_phase_events(family=FAMILY, lane="w8a8-lora64", table=table)

    kinds = {(k, p) for k, p, _ in rows}
    assert ("aot_mint_phases", "minted") in kinds
    assert ("aot_mint_phases", "entry:unet/cfg") in kinds
    assert ("aot_mint_phases", "entry:unet/nocfg") in kinds
    # Durations are NUMERIC (th#1322), never interpolated prose.
    assert dict(((k, p), ms) for k, p, ms in rows)[
        ("aot_mint_phases", "minted")] == 900_000


def test_a_mint_that_produced_no_cell_still_reports_its_seconds(
    _events: List[Tuple[str, str, str]],
) -> None:
    from gen_worker import mint_process
    from gen_worker.mint_process import MintOutcome, MintReport

    outcome = MintOutcome(
        status=mint_process.CRASHED, detail="boom", elapsed_s=120.0,
        report=MintReport(status="failed", elapsed_s=120.0, recipe="aot"))
    assert not outcome.minted
    mint_delegate._emit_aot_phases(outcome, family=FAMILY, lane="w8a8")

    assert ("aot_mint_phases", "aborted") in [(k, p) for k, p, _ in _events]


# ---------------------------------------------------------------------------
# 6. THE CHILD — the AOT recipe runs the exporter, not the FX capture
# ---------------------------------------------------------------------------


def test_the_child_runs_the_exporter_for_the_aot_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker import mint_child

    target = tmp_path / "cell.tar.gz"
    packed = tmp_path / "aot" / "ck5-abc.tar.gz"
    seen: Dict[str, Any] = {}

    def _fake_mint(pipe: Any, spec: Any, out_dir: Path, **kw: Any) -> Any:
        seen["family"] = spec.family
        seen["lane"] = spec.weight_lane
        seen["lifted"] = spec.lifted_inputs
        packed.parent.mkdir(parents=True, exist_ok=True)
        packed.write_bytes(b"packed-cell")
        return aot_mint.MintResult(
            artifact=packed,
            metadata={"cell_key": "ck5-abc", "entries": {"unet/cfg": {}},
                      "mint_phases": {"totals": {"total_s": 1.0}}},
            timings={"total_s": 1.0})

    monkeypatch.setattr(aot_mint, "mint", _fake_mint)
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "w8a8")

    from gen_worker.mint_process import MintRequest
    from gen_worker.mint_delegate import cfg_spec

    request = MintRequest(
        function="generate", modules=("sdxl.main",), family=FAMILY,
        cell_key="ck5-parent", target=str(target),
        capture=str(tmp_path / "cap"), report=str(tmp_path / "report.json"),
        cfg=cfg_spec(_Cfg()), recipe="aot")
    report = mint_child._mint_aot(
        request, _Pipe(), _Cfg(), target,
        started=0.0, blake3_file=lambda p: "deadbeef")

    assert report.status == "minted"
    assert report.recipe == "aot"
    assert report.cell_key == "ck5-abc"
    assert report.mint_phases == {"totals": {"total_s": 1.0}}
    assert target.read_bytes() == b"packed-cell"
    # The spec the exporter got describes the LIVE pipeline, not a re-compose.
    assert seen == {"family": FAMILY, "lane": "w8a8",
                    "lifted": ("lora_a", "lora_b")}


def test_a_named_export_refusal_is_a_refusal_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent must not retry a named refusal — re-running it buys a
    second billed compile for the same sentence."""
    from gen_worker import mint_child

    def _refuse(*a: Any, **k: Any) -> Any:
        raise aot_mint.MintRefused("lane held on dynamo")

    monkeypatch.setattr(aot_mint, "mint", _refuse)
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "w8a8")

    from gen_worker.mint_process import MintRequest
    from gen_worker.mint_delegate import cfg_spec

    request = MintRequest(
        function="generate", modules=("sdxl.main",), family=FAMILY,
        cell_key="k", target=str(tmp_path / "cell.tar.gz"),
        capture=str(tmp_path / "cap"), report=str(tmp_path / "r.json"),
        cfg=cfg_spec(_Cfg()), recipe="aot")
    with pytest.raises(mint_child.MintChildRefused, match="lane held on dynamo"):
        mint_child._mint_aot(
            request, _Pipe(), _Cfg(), tmp_path / "cell.tar.gz",
            started=0.0, blake3_file=lambda p: "x")


# ---------------------------------------------------------------------------
# 7. ADOPTION — a self-minted .pt2 arms through the AOT gates
# ---------------------------------------------------------------------------


def test_a_self_minted_aot_cell_arms_through_the_aot_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED at HEAD: `adopt_delegated_mint` always called `compile_cache.enable`
    — the inductor seed path — so an exported `.pt2` this pod just built could
    never arm, and `provision.enable_compiled` is not reusable here because
    its pgw#709 receipts gate drops an artifact the hub has not countersigned
    yet."""
    calls: List[str] = []
    monkeypatch.setattr(
        fleet_cells.provision, "arm_aot",
        lambda *a, **k: (calls.append("aot"), True)[1])
    monkeypatch.setattr(
        fleet_cells.cc, "enable",
        lambda *a, **k: (calls.append("dynamo"), True)[1])
    monkeypatch.setattr(
        fleet_cells, "_packed_metadata",
        lambda artifact: {"cell_key": "ck5-real", "kind": "aot-inductor"})
    monkeypatch.setattr(fleet_cells, "blake3_file", lambda p: "beef")
    monkeypatch.setattr(fleet_cells, "_unregister", lambda p: None)

    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"cell")
    pending = fleet_cells.PendingSelfMint(
        family=FAMILY, cell_key="ck5-handle", ref="root/family-sdxl#ck5-handle",
        cfg=_Cfg(), target=artifact, capture_dir=tmp_path / "cap",
        mint_root=tmp_path, publisher=None, delegated=True,
        recipe=fleet_cells.RECIPE_AOT)

    minted = fleet_cells.adopt_delegated_mint(_Pipe(), pending, artifact)

    assert calls == ["aot"]
    assert minted is not None
    # The REAL key comes off the packed envelope — an AOT key folds the
    # combined graph hash and cannot be known before the export runs.
    assert minted.cell_key == "ck5-real"
