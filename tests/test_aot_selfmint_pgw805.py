"""pgw#805 — an AOT compiled graph-discovery MISS must start a mint, or refuse BY NAME.

The defect this pins, measured on five real 0.78.0 L4 pods (hub-dispatched,
`Compile(family="sdxl",
targets=("unet",))` declared, compile target advertised, discovery working
through a 200 listing, `compiled_graph_mint_hold_granted` on every pod, trickle traffic
earning pgw#677 background turns):

    self_mint_compile  load            running
    aot_compiled_graph_discovery miss            completed   family=sdxl lane=lora64
    self_mint_compile  warmup_forward  running
    self_mint_compile  warmup_forward  completed
    (nothing further, ever)

No mint, no `self_mint_abort`, no `self_mint_skipped` — no refusal of ANY
kind. Three separate wires were missing, and each gets a test here:

1. **No producer.** `aot_mint.mint` was reachable only from
   `python -m gen_worker.aot_mint`; nothing on the serving path imported it.
   A miss could only fall through to the DYNAMO self-mint, whose artifact
   kind `aot_compiled_graphs._candidates` rejects — so the next pod missed identically,
   forever.
2. **No declaration.** `aot_mint.mint` refuses a family with no registered
   export declaration, and registration only happened when a mint REQUEST
   named a declaration module. A serving pod loads its endpoint and nothing
   else.
3. **Silence.** Every not-mint exit was a `logger.info`, and a serve pod
   exposes no logs (pgw#760).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import aot_mint, fleet_compiled_graphs, mint_delegate
from gen_worker.api.decorators import Compile, Dim, GraphClass, Input
from gen_worker.api.export_contract import (
    export_declaration, register_export_declaration, reset_export_declarations,
)
from gen_worker import config as gw_config
from gen_worker.compiled_graph_adopt import AdoptOutcome
from gen_worker.models import loading

FAMILY = "sdxl"


# ---------------------------------------------------------------------------
# Doubles — the arming brain's real compiled graph point, everything around it stubbed
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

    def _sink(kind: str, detail: str, phase: str = "", duration_ms: int = 0, **_kw) -> None:
        seen.append((kind, phase, detail))

    monkeypatch.setattr(fleet_compiled_graphs.activity_mod, "emit_event", _sink)
    monkeypatch.setattr(mint_delegate.activity_mod, "emit_event", _sink)
    return seen


@pytest.fixture()
def _miss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A real AOT discovery MISS on a mint-capable pod.

    Everything the five measured pods had: the flag armed, a publisher, a
    resolvable compile target, CUDA and a toolchain — and no compiled graph.
    """
    gw_config.reload_for_test()
    monkeypatch.setattr(
        fleet_compiled_graphs.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: AdoptOutcome.miss("no_compiled_graph"))
    monkeypatch.setattr(fleet_compiled_graphs.cc, "has_compile_target", lambda p, c, **_kw: True)
    monkeypatch.setattr(fleet_compiled_graphs.cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(fleet_compiled_graphs.cc, "apply_lora_execution_lane", lambda p, b, **_kw: None)
    monkeypatch.setattr(fleet_compiled_graphs.cc, "drop_lora_execution_lane", lambda p: None)
    monkeypatch.setattr(fleet_compiled_graphs, "_cuda_ready", lambda: True)
    monkeypatch.setattr(fleet_compiled_graphs, "_PENDING", {})
    # No GPU on a dev box: the `sm` axis is a real-runtime fact, and this
    # test is about the RECIPE decision, not key computation.
    monkeypatch.setattr(
        fleet_compiled_graphs, "arm_identity",
        lambda *a, **k: type("_A", (), {
            "token": "arm1-" + "a" * 56,
            "facts_dict": lambda self: {}})())
    # w8a8 is the migration's first lane (pgw#704 parity); its mandatory
    # serving refusal is a separate policy, exercised in its own test.
    monkeypatch.setattr(fleet_compiled_graphs.cc, "mandatory_serving", lambda p: False)
    monkeypatch.setattr(
        fleet_compiled_graphs.cc, "arm_jit_intake", lambda p, c, **_kw: None)
    monkeypatch.setattr(
        fleet_compiled_graphs.loading, "pipeline_weight_lane", lambda pipe: "w8a8")
    yield
    gw_config.reload_for_test()


def _arm(**kw: Any) -> Any:
    return fleet_compiled_graphs.enable_compiled(
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
    artifact kind AOT discovery rejects — so a fleet
    could never produce the compiled graph it was looking for."""
    register_export_declaration(_declaration())

    outcome = _arm(delegate=True)

    pending = outcome.self_mint
    assert pending is not None, "a discovery miss produced no mint at all"
    # pgw#1010: a pending IS an AOT mint — the dynamo recipe opens none at all,
    # so the recipe axis this used to assert cannot disagree any more.
    assert pending.delegated is True, (
        "an AOTI export holds the GPU with no router to yield through; it "
        "must never run in the serving process")
    assert not outcome.armed, "the live pipe serves EAGER while the child mints"
    assert ("self_mint_started", "aot") in [(k, p) for k, p, _ in _events]


def test_the_child_is_handed_one_artifact_kind_and_cannot_choose(
    _miss: None, _events: List[Tuple[str, str, str]], tmp_path: Path,
) -> None:
    """pgw#1010: the recipe stopped travelling because there is nothing to
    choose. The child exports; that is the only artifact it can produce, so a
    `recipe` field on the wire could only ever disagree with the truth."""
    register_export_declaration(_declaration())
    pending = _arm(delegate=True).self_mint
    assert pending is not None

    task = mint_delegate.MintTask(
        pending=pending, pipe=_Pipe(), function="generate",
        modules=("sdxl.main",), weight_lane="w8a8")
    request = mint_delegate.build_request(
        task, workdir=tmp_path)
    assert not hasattr(request, "recipe")
    assert request.work_root == str(tmp_path)


# ---------------------------------------------------------------------------
# 2. NO DECLARATION — refused by name, never silently
# ---------------------------------------------------------------------------


def test_a_family_with_no_export_declaration_refuses_by_name(
    _miss: None, _events: List[Tuple[str, str, str]],
) -> None:
    """...and, since pgw#1010, mints NOTHING: the JIT intake arm serves this
    pod and opens no obligation, because a dynamo compiled graph has no consumer."""
    outcome = _arm(delegate=True)

    assert export_declaration(FAMILY) is None
    assert outcome.self_mint is None, "a JIT intake arm owes no compiled_graph"
    assert outcome.armed, "intake still SERVES — it compiles in this process"
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
# 3. THE LANE IS AN INPUT — pgw#850/#879: the mint compiles what it is handed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base_execution_lane", loading.STAMPABLE_BASE_EXECUTION_LANES)
def test_every_execution_lane_a_pod_can_serve_reaches_the_aot_recipe(
    base_execution_lane: str,
    _miss: None, _events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#850's fix, proven over the WHOLE lane vocabulary rather than the
    one lane the allowlist named.

    The composition this replaces, measured on `origin/dev` 2026-08-03:
    `aot_mint.PARITY_LANES` admitted exactly `"w8a8"`, and tensorhub's lane
    table makes `fp8-w8a8-dynamic` compiled-only — so the hub withholds it
    from AUTO until a compiled graph exists (th#1123 `applyCompileCompiledGraphAvailability`,
    th#1127 `applyPublishMintObligation`), and tensorhub's own comment says
    "only a worker's own self-mint can discharge it". The single admitted
    lane was the single lane no AUTO pod could ever be on, and the four lanes
    a pod COULD be on were each declined `aot_lane_regressed`, quoting a
    6.9-7.0% AOTI regression measured on sdxl's lanes alone. Zero fleet
    families reached the mint gate, on any card.

    The lane is an INPUT now. `loading.STAMPABLE_BASE_LANES` is every lane a
    loader can leave a pipeline on (pgw#918, mechanically checked), so this
    parametrisation IS the fleet: each one must mint.
    """
    register_export_declaration(_declaration())
    monkeypatch.setattr(
        fleet_compiled_graphs.loading, "pipeline_weight_lane", lambda pipe: base_execution_lane)

    pending = _arm(delegate=True).self_mint

    assert pending is not None, (
        f"lane {base_execution_lane!r} did not reach the AOT recipe — a "
        f"mint-side lane judgement has grown back")
    assert not _phases(_events, "self_mint_skipped"), (
        f"lane {base_execution_lane!r} was declined: "
        f"{[d for k, _p, d in _events if k == 'self_mint_skipped']}")


def test_the_bucketed_lora_form_of_a_execution_lane_mints_too(
    _miss: None, _events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stamp a pod carries in production is usually the BUCKETED form
    (`w8a8-lora64`, `fp8-hooks-lora32`). It decomposes to a base lane and
    must not be treated as an unknown string."""
    register_export_declaration(_declaration())
    monkeypatch.setattr(
        fleet_compiled_graphs.loading, "pipeline_weight_lane",
        lambda pipe: "fp8-hooks-lora64")

    pending = _arm(delegate=True).self_mint

    assert pending is not None
    assert not _phases(_events, "self_mint_skipped")


def test_the_mint_holds_no_execution_lane_predicate() -> None:
    """The deletion itself. `lane_admitted`/`PARITY_LANES` were a SECOND
    opinion about a lane the hub's resolution tree had already chosen; two
    opinions in two repos is what composed into the total block. Every
    surviving mint check answers "can this compile physically run"."""
    assert not hasattr(aot_mint, "lane_admitted")
    assert not hasattr(aot_mint, "PARITY_LANES")
    assert "allow_regressed_lanes" not in inspect.signature(
        aot_mint.mint).parameters


def test_an_in_process_only_pod_declines_the_aot_recipe_by_name(
    _miss: None, _events: List[Tuple[str, str, str]],
) -> None:
    register_export_declaration(_declaration())

    outcome = _arm(delegate=False)

    assert outcome.self_mint is None, (
        "a pod that cannot delegate cannot mint an AOT compiled_graph — and pgw#1010 "
        "leaves it nothing else to mint, so it serves JIT intake")
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
    and a serve pod exposes no logs (pgw#760).

    pgw#824 tightened the assertion: the phase is the CLASSIFIED cause, not the
    constant `mint_unavailable` that all nine `_fail_closed` exits used to
    share. The cause lived only in the free-text detail, so counting "how much
    of this fleet is eager for want of a C++ compiler" meant substring-matching
    a sentence. The token is also what the request row's `fallback_reason`
    carries, so the event stream and the request table join on one string.
    """
    monkeypatch.setattr(fleet_compiled_graphs.cc, "toolchain_present", lambda: False)

    outcome = _arm(delegate=True)

    assert not outcome.armed and outcome.self_mint is None
    skipped = [
        detail for kind, phase, detail in _events
        if kind == "self_mint_skipped" and phase == "no_toolchain"]
    assert skipped, "a pod that mints nothing and refuses nothing is unreadable"
    assert "no C compiler" in skipped[0]
    # and the arm carries the same token out, for the request path to report
    assert outcome.eager_reason == "no_toolchain"


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
        lambda kind, detail, phase="", duration_ms=0, **_kw: rows.append(
            (kind, phase, duration_ms)))

    table = {
        "v": 1, "n_compiled_graphs": 2,
        "totals": {"total_s": 900.0, "export_s": 60.0, "compile_s": 800.0},
        "phases": {}, "autotune": {},
        "compiled_graphs": {
            "unet/cfg": {"export_s": 30.0, "compile_s": 400.0},
            "unet/nocfg": {"export_s": 30.0, "compile_s": 400.0},
        },
    }
    aot_mint.emit_phase_events(family=FAMILY, execution_lane="w8a8-lora64", table=table)

    kinds = {(k, p) for k, p, _ in rows}
    assert ("aot_mint_phases", "minted") in kinds
    assert ("aot_mint_phases", "compiled_graph:unet/cfg") in kinds
    assert ("aot_mint_phases", "compiled_graph:unet/nocfg") in kinds
    # Durations are NUMERIC (th#1322), never interpolated prose.
    assert dict(((k, p), ms) for k, p, ms in rows)[
        ("aot_mint_phases", "minted")] == 900_000


def test_a_mint_that_produced_no_compiled_graph_still_reports_its_seconds(
    _events: List[Tuple[str, str, str]],
) -> None:
    from gen_worker import mint_process
    from gen_worker.mint_process import MintOutcome, MintReport

    outcome = MintOutcome(
        status=mint_process.CRASHED, detail="boom", elapsed_s=120.0,
        report=MintReport(status="failed", elapsed_s=120.0))
    assert not outcome.minted
    mint_delegate._emit_aot_phases(outcome, family=FAMILY, execution_lane="w8a8")

    assert ("aot_mint_phases", "aborted") in [(k, p) for k, p, _ in _events]


# ---------------------------------------------------------------------------
# 6. THE CHILD — the AOT recipe runs the exporter, not the FX capture
# ---------------------------------------------------------------------------


def test_the_child_runs_the_exporter_for_the_aot_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker import mint_child

    target = tmp_path / "compiled_graph.tar.gz"
    # pgw#1176: a `ck1` key names a 36-compiled graph all-or-nothing compiled graph, which this
    # runtime cannot arm at all — `compiled_graph_key.is_key` refuses the prefix
    # deliberately, so a fixture keyed that way tests a shape nothing produces.
    key = "ek1-" + "a" * 56
    packed = tmp_path / "aot" / f"{key}.tar.gz"
    seen: Dict[str, Any] = {}

    def _fake_mint(pipe: Any, spec: Any, out_dir: Path, **kw: Any) -> Any:
        seen["family"] = spec.family
        seen["lane"] = spec.weight_lane
        seen["lifted"] = spec.lifted_inputs
        packed.parent.mkdir(parents=True, exist_ok=True)
        packed.write_bytes(b"packed-compiled-graph")
        # pgw#1176: a mint returns N independently keyed compiled graph artifacts plus a
        # manifest digest, never "a compiled graph". This declaration traces one class.
        return aot_mint.MintResult(
            compiled_graphs=(aot_mint.MintedArtifact(
                key=key, compiled_graph="unet/cfg", artifact=packed,
                metadata={"compiled_graph_key": key,
                          "mint_phases": {"totals": {"total_s": 1.0}}}),),
            manifest="c0ffee0000000000",
            timings={"total_s": 1.0})

    monkeypatch.setattr(aot_mint, "mint", _fake_mint)
    monkeypatch.setattr(
        fleet_compiled_graphs.loading, "pipeline_weight_lane", lambda pipe: "w8a8")

    from gen_worker.mint_process import MintRequest
    from gen_worker.mint_delegate import cfg_spec

    request = MintRequest(
        function="generate", modules=("sdxl.main",), family=FAMILY,
        arm_token="arm1-parent", target=str(target),
        work_root=str(tmp_path), report=str(tmp_path / "report.json"),
        cfg=cfg_spec(_Cfg()))
    report = mint_child._mint_aot(
        request, _Pipe(), _Cfg(), target,
        started=0.0, sha256_file=lambda p: "deadbeef")

    assert report.status == "minted"
    # pgw#1176: the child moves EVERY compiled graph it packed into the parent's
    # directory, one file per graph class NAMED BY ITS OWN `ek1` key — so the
    # parent addresses each by identity rather than by position, and `target`
    # names the directory rather than the single file it used to be. The
    # one-element unpack asserts this declaration's arity.
    (moved_key, moved_path, _sha), = report.compiled_graphs
    assert moved_key == key
    assert Path(moved_path) == target.parent / f"{key}.tar.gz"
    assert Path(moved_path).read_bytes() == b"packed-compiled-graph"
    assert report.mint_phases == {"totals": {"total_s": 1.0}}
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
        fleet_compiled_graphs.loading, "pipeline_weight_lane", lambda pipe: "w8a8")

    from gen_worker.mint_process import MintRequest
    from gen_worker.mint_delegate import cfg_spec

    request = MintRequest(
        function="generate", modules=("sdxl.main",), family=FAMILY,
        arm_token="k", target=str(tmp_path / "compiled_graph.tar.gz"),
        work_root=str(tmp_path), report=str(tmp_path / "r.json"),
        cfg=cfg_spec(_Cfg()))
    with pytest.raises(mint_child.MintChildRefused, match="lane held on dynamo"):
        mint_child._mint_aot(
            request, _Pipe(), _Cfg(), tmp_path / "compiled_graph.tar.gz",
            started=0.0, sha256_file=lambda p: "x")


# ---------------------------------------------------------------------------
# 7. ADOPTION — a self-minted .pt2 arms through the AOT gates
# ---------------------------------------------------------------------------


def test_a_self_minted_aot_compiled_graph_arms_through_the_aot_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED at HEAD: `adopt_delegated_mint` always called `compile_cache.enable`
    — the inductor seed path — so an exported `.pt2` this pod just built could
    never arm, and `provision.enable_compiled` is not reusable here because
    its pgw#709 receipts gate drops an artifact the hub has not countersigned
    yet."""
    calls: List[str] = []
    monkeypatch.setattr(
        fleet_compiled_graphs.provision, "arm_aot",
        lambda *a, **k: (calls.append("aot"), AdoptOutcome.hit())[1])
    monkeypatch.setattr(
        fleet_compiled_graphs.cc, "enable",
        lambda *a, **k: (calls.append("dynamo"), True)[1])
    # pgw#1010: `cc.enable` is not merely unused on this path — the branch that
    # called it is deleted, so a "dynamo" compiled graph below would mean an inductor
    # seed adopt has grown back for an exported compiled graph.
    monkeypatch.setattr(
        fleet_compiled_graphs, "_packed_metadata",
        lambda artifact: {"compiled_graph_key": "ck1-real", "kind": "aot-inductor"})
    monkeypatch.setattr(fleet_compiled_graphs, "sha256_file", lambda p: "beef")
    monkeypatch.setattr(fleet_compiled_graphs, "_unregister", lambda p: None)

    # pgw#1098: a READABLE envelope. This used to be `b"compiled graph"` and worked only
    # because `try_read_metadata` swallowed the error into `None`; an
    # unreadable envelope is now its own refusal before the arm, so a test
    # about the ARM has to supply one the adopt can read.
    import io as _io
    import json as _json
    import tarfile as _tarfile

    artifact = tmp_path / "compiled_graph.tar.gz"
    _payload = _json.dumps(
        {"compiled_graph_key": "ck1-real", "kind": "aot-inductor"}).encode()
    with _tarfile.open(artifact, mode="w:gz") as _tar:
        _info = _tarfile.TarInfo("metadata.json")
        _info.size = len(_payload)
        _tar.addfile(_info, _io.BytesIO(_payload))
    pending = fleet_compiled_graphs.PendingSelfMint(
        family=FAMILY, arm_token="ck1-handle", ref="root/family-sdxl#ek1-handle",
        cfg=_Cfg(), target=artifact,
        mint_root=tmp_path, publisher=None)

    minted = fleet_compiled_graphs.adopt_delegated_mint(_Pipe(), pending, [artifact])

    assert calls == ["aot"]
    assert minted is not None
    # The REAL key comes off the packed envelope — an AOT key folds the
    # combined graph hash and cannot be known before the export runs.
    assert minted.compiled_graph_key == "ck1-real"
