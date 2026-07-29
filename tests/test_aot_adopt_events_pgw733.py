"""pgw#733 (arm half): every AOT adopt/arm outcome is a TYPED wire event.

The verdict lane's blocker (tracker CP5, 2026-07-29): a cross-pod adopt
fails inside stage/bind/arm with a classified ``AdoptError`` reason, and
v0.76.5 reduces ALL of them to ``logger.warning`` — hub-spawned workers
expose no stdout, so the reason is structurally invisible. These tests are
the red half: each classified refusal must ride ``aot_serve.ADOPT_EVENT``
(one event class, ``phase`` = the reason), a successful arm must too, and
the fleet-level F1 consumer must bind the outcome to the DISCOVERED
candidate's identity (key + ref) on the same event class.

The wire transport itself (ActivityUpdate -> hub row, th#1250) is captured
at ``activity._emit`` — the exact envelope the stream sink sends — so the
assertion covers kind/phase/detail truncation as wired, not a test double
of the event API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gen_worker import activity, aot_cells, aot_serve, fleet_cells
from gen_worker.pb import worker_scheduler_pb2 as pb

FAMILY = "sdxl"
RUNTIME = {"sku": "l4", "torch": "2.13.0+cu130", "cuda": "13.0"}
KEY = "ck5-" + "a" * 56

INPUTS = [
    {"name": "sample", "position": 0, "dtype": "bfloat16",
     "shape": [2, 4, 128, 128]},
]
CONSTANTS = [
    {"fqn": "conv_in.weight", "source": aot_serve.SOURCE_STATE_DICT,
     "dtype": "bfloat16", "shape": [320, 4, 3, 3]},
]
ENTRY = "unet/g"


class FakeTensor:
    def __init__(self, shape: Any, dtype: str = "torch.bfloat16") -> None:
        self.shape = tuple(shape)
        self.dtype = dtype


class FakePackage:
    """Stands in for ``AOTICompiledModel`` (no dlopen of a real .pt2)."""

    def __init__(self, fqns: Any = ("conv_in.weight",)) -> None:
        self._fqns = list(fqns)
        self.loaded: Optional[Dict[str, Any]] = None

    def get_constant_fqns(self) -> List[str]:
        return list(self._fqns)

    def load_constants(self, values: Dict[str, Any],
                       check_full_update: bool = False) -> None:
        if check_full_update and set(values) != set(self._fqns):
            raise RuntimeError("partial constant update refused by torch")
        self.loaded = dict(values)

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return "ARTIFACT_OUTPUT"


class FakeModule:
    def __init__(self, weights: Any = ("conv_in.weight",)) -> None:
        self.device = "cpu"
        self._weights = {name: FakeTensor([1]) for name in weights}

    def state_dict(self) -> Dict[str, Any]:
        return dict(self._weights)

    def forward(self, *args: Any, **kwargs: Any) -> str:
        return "EAGER_OUTPUT"


class FakePipeline:
    def __init__(self) -> None:
        self.unet = FakeModule()


class Cfg:
    family = FAMILY
    lora_bucket = 0
    targets = ("unet",)
    regional = False


def _entry(**over: Any) -> Dict[str, Any]:
    e: Dict[str, Any] = {
        "target": "unet", "fork": [], "class_dims": [],
        "inputs": [dict(r) for r in INPUTS], "symbols": {},
        "constants": [dict(r) for r in CONSTANTS], "graph": {},
    }
    e.update(over)
    try:
        e["range_digest"] = aot_serve.range_digest(e)
    except (ValueError, TypeError):
        e.setdefault("range_digest", "")
    e["class_hash"] = aot_serve.class_hash(e, strict=True, lora_bucket=0)
    return e


def _meta(**over: Any) -> Dict[str, Any]:
    entries = over.pop("entries", None) or {ENTRY: _entry()}
    m: Dict[str, Any] = {
        "format": aot_serve.ARTIFACT_FORMAT, "kind": aot_serve.ARTIFACT_KIND,
        **RUNTIME, "family": FAMILY, "precision": "w8a8",
        "cell_key": KEY, "entries": entries,
        "strict_export": True, "lora_bucket": 0,
        "package_constants_in_so": False,
        "source_ref": "", "source_digest": "",
    }
    m["combined_graph_hash"] = aot_serve.combined_graph_hash(
        str((b or {}).get("class_hash") or "")
        for b in entries.values() if isinstance(b, dict))
    m.update(over)
    return m


def _tar(tmp_path: Path, meta: Optional[Dict[str, Any]] = None,
         name: str = "cell.tar.gz") -> Path:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / aot_serve.PACKAGE_NAME).write_bytes(b"\x00not-a-real-pt2")
    return aot_serve.pack(work, tmp_path / name, meta or _meta())


@pytest.fixture()
def events(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    captured: List[Any] = []
    monkeypatch.setattr(activity, "_emit", captured.append)
    return captured


@pytest.fixture()
def stub_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: dict(RUNTIME))


def _adopt_rows(events: List[Any]) -> List[Any]:
    return [e for e in events if e.kind == aot_serve.ADOPT_EVENT]


# ---------------------------------------------------------------------------
# aot_serve.enable — the classified inner reason, success AND failure
# ---------------------------------------------------------------------------


def test_successful_arm_emits_armed(
    tmp_path: Path, events: List[Any], stub_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        aot_serve, "_load_package", lambda path, entry: FakePackage())
    assert aot_serve.enable(FakePipeline(), Cfg(), artifact=_tar(tmp_path))
    rows = _adopt_rows(events)
    assert [e.phase for e in rows] == ["armed"]
    assert KEY in rows[0].detail and FAMILY in rows[0].detail
    assert rows[0].state == pb.ActivityState.ACTIVITY_STATE_COMPLETED


def test_key_mismatch_named_on_the_wire(
    tmp_path: Path, events: List[Any], stub_runtime: None,
) -> None:
    art = _tar(tmp_path, _meta(sku="a100"))
    assert not aot_serve.enable(FakePipeline(), Cfg(), artifact=art)
    rows = _adopt_rows(events)
    assert [e.phase for e in rows] == ["key_mismatch"]
    assert KEY in rows[0].detail  # the refusal names the candidate cell


def test_host_isa_unsupported_named_on_the_wire(
    tmp_path: Path, events: List[Any], stub_runtime: None,
) -> None:
    art = _tar(tmp_path, _meta(host_isa={"machine": "sparc64", "level": ""}))
    assert not aot_serve.enable(FakePipeline(), Cfg(), artifact=art)
    assert [e.phase for e in _adopt_rows(events)] == ["host_isa_unsupported"]


def test_artifact_invalid_named_on_the_wire(
    tmp_path: Path, events: List[Any], stub_runtime: None,
) -> None:
    art = tmp_path / "corrupt.tar.gz"
    art.write_bytes(b"\x00definitely not a tarball")
    assert not aot_serve.enable(FakePipeline(), Cfg(), artifact=art)
    rows = _adopt_rows(events)
    assert [e.phase for e in rows] == ["artifact_invalid"]
    # Identity is best-effort: an unreadable artifact still names its file.
    assert "corrupt.tar.gz" in rows[0].detail


def test_malformed_entries_named_on_the_wire(
    tmp_path: Path, events: List[Any], stub_runtime: None,
) -> None:
    art = _tar(tmp_path, _meta(entries={ENTRY: {"target": "unet"}}))
    assert not aot_serve.enable(FakePipeline(), Cfg(), artifact=art)
    rows = _adopt_rows(events)
    assert [e.phase for e in rows] == ["artifact_invalid"]
    assert "declares no inputs" in rows[0].detail


def test_constants_refusal_named_on_the_wire(
    tmp_path: Path, events: List[Any], stub_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        aot_serve, "_load_package",
        lambda path, entry: FakePackage(fqns=("conv_in.weight",
                                              "not.in.state_dict")))
    assert not aot_serve.enable(FakePipeline(), Cfg(), artifact=_tar(tmp_path))
    rows = _adopt_rows(events)
    assert len(rows) == 1
    assert rows[0].phase.startswith("constants_")


# ---------------------------------------------------------------------------
# fleet_cells F1 consumer — outcome bound to the DISCOVERED candidate
# ---------------------------------------------------------------------------


class _StubPublisher:
    base_url = "http://hub.invalid"

    def enabled(self) -> bool:
        return True

    def worker_jwt(self) -> str:
        return "jwt"


@dataclass
class _FleetCfg:
    family: str = FAMILY
    lora_bucket: int = 0


class _FleetPipe:
    pass


@pytest.fixture()
def _f1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """Flag on + a discovered candidate; provision seam is per-test."""
    from gen_worker import compile_cache as cc
    from gen_worker.config import get_settings

    monkeypatch.setenv("GEN_WORKER_PREFER_AOT", "1")
    get_settings.cache_clear()
    monkeypatch.setattr(cc, "has_compile_target", lambda pipe, cfg: True)
    art = tmp_path / "cell.tar.gz"
    art.write_bytes(b"artifact")
    adopted = aot_cells.AdoptedAotCell(
        family=FAMILY, cell_key=KEY, ref=f"root/family-{FAMILY}#{KEY}",
        snapshot_digest="blake3:" + "5" * 64, artifact=art)
    monkeypatch.setattr(aot_cells, "discover", lambda *a, **k: adopted)
    yield adopted
    get_settings.cache_clear()


def test_fleet_success_is_one_armed_row(
    monkeypatch: pytest.MonkeyPatch, events: List[Any], _f1: Any,
) -> None:
    def _enable(pipe: Any, cfg: Any, cache_dir: Any, artifact: Any) -> bool:
        pipe._cozy_aot = {"state": {"failed": False}}
        activity.emit_event(aot_serve.ADOPT_EVENT, "k", phase="armed")
        return True

    monkeypatch.setattr(fleet_cells.provision, "enable_compiled", _enable)
    outcome = fleet_cells.enable_compiled(
        _FleetPipe(), _FleetCfg(), publisher=_StubPublisher())  # type: ignore[arg-type]
    assert outcome.armed and outcome.self_mint is _f1
    # exactly the inner "armed" row — the fleet layer adds no duplicate
    assert [e.phase for e in _adopt_rows(events)] == ["armed"]


def test_fleet_did_not_arm_row_names_the_candidate(
    monkeypatch: pytest.MonkeyPatch, events: List[Any], _f1: Any,
    tmp_path: Path,
) -> None:
    delivered = tmp_path / "delivered.tar.gz"
    delivered.write_bytes(b"dynamo")
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: artifact == delivered)
    outcome = fleet_cells.enable_compiled(
        _FleetPipe(), _FleetCfg(), artifact=delivered,
        publisher=_StubPublisher())  # type: ignore[arg-type]
    assert outcome.armed and outcome.self_mint is None
    rows = _adopt_rows(events)
    assert [e.phase for e in rows] == ["did_not_arm"]
    assert KEY in rows[0].detail and _f1.ref in rows[0].detail


def test_fleet_armed_other_path_never_advertises_aot(
    monkeypatch: pytest.MonkeyPatch, events: List[Any], _f1: Any,
) -> None:
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: True)  # armed, marker absent
    outcome = fleet_cells.enable_compiled(
        _FleetPipe(), _FleetCfg(), publisher=_StubPublisher())  # type: ignore[arg-type]
    assert outcome.armed and outcome.self_mint is None
    rows = _adopt_rows(events)
    assert [e.phase for e in rows] == ["armed_other_path"]
    assert KEY in rows[0].detail


def test_fleet_lane_unavailable_row_names_the_candidate(
    monkeypatch: pytest.MonkeyPatch, events: List[Any], _f1: Any,
    tmp_path: Path,
) -> None:
    from gen_worker import compile_cache as cc

    delivered = tmp_path / "delivered.tar.gz"
    delivered.write_bytes(b"dynamo")
    calls: List[Any] = []

    def _enable(pipe: Any, cfg: Any, cache_dir: Any, artifact: Any) -> bool:
        calls.append(artifact)
        if len(calls) == 1:
            raise cc.CompiledLaneUnavailableError("no cell for w8a8")
        return True

    monkeypatch.setattr(fleet_cells.provision, "enable_compiled", _enable)
    outcome = fleet_cells.enable_compiled(
        _FleetPipe(), _FleetCfg(), artifact=delivered,
        publisher=_StubPublisher())  # type: ignore[arg-type]
    assert outcome.armed
    rows = _adopt_rows(events)
    assert [e.phase for e in rows] == ["lane_unavailable"]
    assert KEY in rows[0].detail
