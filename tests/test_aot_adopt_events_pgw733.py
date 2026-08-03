"""pgw#733 (arm half) + pgw#923: every AOT adopt/arm outcome is TYPED and MEASURED.

The verdict lane's blocker (tracker CP5, 2026-07-29): a cross-pod adopt fails
inside stage/bind/arm with a classified ``AdoptError`` reason, and v0.76.5
reduced ALL of them to ``logger.warning`` — hub-spawned workers expose no
stdout, so the reason was structurally invisible. pgw#733 fixed that with a
free-text ``aot_adopt`` activity event.

pgw#923 replaced that spelling. The event carried a reason and no numbers,
while the MEASURED lane it duplicated (``compile_cache_adopt``, fed by
``ModelEvent{ADOPTED}``) had zero rows on both live stacks because its only
sender was a hub operation nothing dispatches. So the arm now RETURNS its
outcome, the arming policy MEASURES it, and one ledger — ``ArmOutcome.
adoptions`` — carries every attempt with its identity, its classified reason
and its wall time to the executor, which puts it on the wire the hub already
stores.

The acceptance is unchanged and strictly stronger: every classified refusal is
still named, and now it is also timed and bound to the candidate's ref+digest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gen_worker import activity, aot_cells, aot_serve, fleet_cells
from gen_worker.cell_adopt import AdoptOutcome

#: The arm is INDUCED to take this long, and the floor asserted against it is a
#: share of that induced quantity rather than a bare constant (pgw#795). This is
#: a LOWER bound on work the test itself produced: a slow runner only raises the
#: measured value, so nothing here can fail because the machine was busy.
_INDUCED_ARM_S = 0.02
_MEASURED_ARM_FLOOR_MS = int(_INDUCED_ARM_S * 1000 * 0.75)

FAMILY = "sdxl"
RUNTIME = {"sku": "l4", "sm": "sm_89", "torch": "2.13.0+cu130",
           "cuda": "13.0"}
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


def _reasons(outcome: Any) -> List[str]:
    """The classified reason of every adoption ATTEMPT, in order."""
    return [row.reason for row in outcome.adoptions]


# ---------------------------------------------------------------------------
# aot_serve.enable — the classified inner reason, success AND failure
# ---------------------------------------------------------------------------


def test_successful_arm_returns_an_armed_outcome_naming_the_cell(
    tmp_path: Path, stub_runtime: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        aot_serve, "_load_package", lambda path, entry: FakePackage())
    out = aot_serve.enable(FakePipeline(), Cfg(), artifact=_tar(tmp_path))
    assert out.armed and out.reason == ""
    assert KEY in out.identity and FAMILY in out.identity


def test_key_mismatch_named_on_the_wire(
    tmp_path: Path, events: List[Any], stub_runtime: None,
) -> None:
    # pgw#765: an sm mismatch is the key mismatch; a SKU difference alone is
    # adopted (same-sm cross-SKU cells are the point of the pgw#691 collapse).
    art = _tar(tmp_path, _meta(sm="sm_80"))
    out = aot_serve.enable(FakePipeline(), Cfg(), artifact=art)
    assert not out.armed and out.reason == "key_mismatch"
    assert KEY in out.detail  # the refusal names the candidate cell


def test_host_isa_unsupported_named_on_the_wire(
    tmp_path: Path, events: List[Any], stub_runtime: None,
) -> None:
    art = _tar(tmp_path, _meta(host_isa={"machine": "sparc64", "level": ""}))
    out = aot_serve.enable(FakePipeline(), Cfg(), artifact=art)
    assert not out.armed and out.reason == "host_isa_unsupported"


def test_artifact_invalid_named_on_the_wire(
    tmp_path: Path, events: List[Any], stub_runtime: None,
) -> None:
    art = tmp_path / "corrupt.tar.gz"
    art.write_bytes(b"\x00definitely not a tarball")
    out = aot_serve.enable(FakePipeline(), Cfg(), artifact=art)
    assert not out.armed and out.reason == "artifact_invalid"
    # Identity is best-effort: an unreadable artifact still names its file.
    assert "corrupt.tar.gz" in out.detail

    # Malformed entries classify the same, with the reason in the detail.
    malformed = _tar(tmp_path, _meta(entries={ENTRY: {"target": "unet"}}))
    out = aot_serve.enable(FakePipeline(), Cfg(), artifact=malformed)
    assert not out.armed and out.reason == "artifact_invalid"
    assert "declares no inputs" in out.detail


def test_constants_refusal_named_on_the_wire(
    tmp_path: Path, events: List[Any], stub_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        aot_serve, "_load_package",
        lambda path, entry: FakePackage(fqns=("conv_in.weight",
                                              "not.in.state_dict")))
    out = aot_serve.enable(FakePipeline(), Cfg(), artifact=_tar(tmp_path))
    assert not out.armed
    assert out.reason.startswith("constants_")


# ---------------------------------------------------------------------------
# discovery — pre-clamp (unstamped host_isa) cells are retired by name
# ---------------------------------------------------------------------------


def test_candidates_retire_unstamped_preclamp_cells(
    stub_runtime: None,
) -> None:
    """Live-proven 2026-07-29 (pod 3cjmd3ohuk98a5): an unstamped pre-clamp
    cell passes every metadata gate, gets downloaded, then refuses at stage
    (`host_isa_unsupported`, torch package stamp). Discovery must retire the
    whole class instead of shipping doomed candidates."""
    from gen_worker import host_isa

    stamped = _meta(host_isa=host_isa.stamp())
    unstamped = _meta()
    assert "host_isa" not in unstamped
    items = [
        {"checkpoint_id": "ck-old-preclamp",
         "updated_at": "2026-07-30T00:00:00Z", "metadata": unstamped},
        {"checkpoint_id": "ck-stamped",
         "updated_at": "2026-07-28T00:00:00Z", "metadata": stamped},
    ]
    rows = aot_cells._candidates(items, FAMILY, "")
    assert [r[1] for r in rows] == ["ck-stamped"]


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
    from gen_worker import config as gw_config

    monkeypatch.setenv("GEN_WORKER_PREFER_AOT", "1")
    gw_config.reload_for_test()
    monkeypatch.setattr(cc, "has_compile_target", lambda pipe, cfg: True)
    art = tmp_path / "cell.tar.gz"
    art.write_bytes(b"artifact")
    adopted = aot_cells.AdoptedAotCell(
        family=FAMILY, cell_key=KEY, ref=f"root/family-{FAMILY}#{KEY}",
        snapshot_digest="blake3:" + "5" * 64, artifact=art)
    monkeypatch.setattr(aot_cells, "discover", lambda *a, **k: adopted)
    yield adopted
    gw_config.reload_for_test()


def test_fleet_success_is_one_measured_adoption_bound_to_the_candidate(
    monkeypatch: pytest.MonkeyPatch, _f1: Any,
) -> None:
    def _enable(pipe: Any, cfg: Any, cache_dir: Any, artifact: Any) -> Any:
        pipe._cozy_aot = {"state": {"failed": False}}
        time.sleep(_INDUCED_ARM_S)  # a real arm costs time; a measured one records it
        return AdoptOutcome.hit(f"family={FAMILY} key={KEY}")

    monkeypatch.setattr(fleet_cells.provision, "enable_compiled", _enable)
    outcome = fleet_cells.enable_compiled(
        _FleetPipe(), _FleetCfg(), publisher=_StubPublisher())  # type: ignore[arg-type]
    assert outcome.armed and outcome.self_mint is _f1
    assert len(outcome.adoptions) == 1
    row = outcome.adoptions[0]
    assert row.armed and row.reason == ""
    # The identity the hub fences an adoption on, and the number the whole
    # measurement lane exists for. `arm_ms == 0` is what shipped for a year.
    assert row.ref == _f1.ref and row.snapshot_digest == _f1.snapshot_digest
    assert row.artifact_kind == aot_serve.ARTIFACT_KIND
    assert row.arm_ms >= _MEASURED_ARM_FLOOR_MS, (
        "the adoption recorded no time at all")


def test_fleet_did_not_arm_is_a_measured_refusal_naming_the_candidate(
    monkeypatch: pytest.MonkeyPatch, _f1: Any, tmp_path: Path,
) -> None:
    delivered = tmp_path / "delivered.tar.gz"
    delivered.write_bytes(b"dynamo")
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: (
            AdoptOutcome.hit() if artifact == delivered
            else AdoptOutcome.miss("key_mismatch", f"key={KEY}")))
    outcome = fleet_cells.enable_compiled(
        _FleetPipe(), _FleetCfg(), artifact=delivered,
        publisher=_StubPublisher(),  # type: ignore[arg-type]
        delivered_ref="root/family-sdxl#ck5-delivered",
        delivered_digest="blake3:" + "7" * 64)
    assert outcome.armed and outcome.self_mint is None
    # BOTH attempts are on the ledger, in order: the discovered cell that
    # refused, then the delivered cell that armed. The old vocabulary recorded
    # the refusal and nothing at all for the adoption that actually happened.
    assert _reasons(outcome) == ["key_mismatch", ""]
    assert outcome.adoptions[0].ref == _f1.ref
    assert not outcome.adoptions[0].armed
    assert outcome.adoptions[1].armed
    assert outcome.adoptions[1].ref == "root/family-sdxl#ck5-delivered"


def test_fleet_armed_other_path_never_advertises_aot(
    monkeypatch: pytest.MonkeyPatch, _f1: Any,
) -> None:
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: AdoptOutcome.hit())  # marker absent
    outcome = fleet_cells.enable_compiled(
        _FleetPipe(), _FleetCfg(), publisher=_StubPublisher())  # type: ignore[arg-type]
    assert outcome.armed and outcome.self_mint is None
    assert _reasons(outcome) == ["armed_other_path"]
    row = outcome.adoptions[0]
    assert not row.armed, "the DISCOVERED cell is not what armed this pipe"
    assert row.ref == _f1.ref and KEY in row.detail


def test_fleet_lane_unavailable_is_recorded_against_the_candidate(
    monkeypatch: pytest.MonkeyPatch, _f1: Any, tmp_path: Path,
) -> None:
    from gen_worker import compile_cache as cc

    delivered = tmp_path / "delivered.tar.gz"
    delivered.write_bytes(b"dynamo")
    calls: List[Any] = []

    def _enable(pipe: Any, cfg: Any, cache_dir: Any, artifact: Any) -> Any:
        calls.append(artifact)
        if len(calls) == 1:
            raise cc.CompiledLaneUnavailableError("no cell for w8a8")
        return AdoptOutcome.hit()

    monkeypatch.setattr(fleet_cells.provision, "enable_compiled", _enable)
    outcome = fleet_cells.enable_compiled(
        _FleetPipe(), _FleetCfg(), artifact=delivered,
        publisher=_StubPublisher())  # type: ignore[arg-type]
    assert outcome.armed
    assert _reasons(outcome) == ["lane_unavailable", ""]
    assert outcome.adoptions[0].ref == _f1.ref
    assert KEY in outcome.adoptions[0].detail


# ---------------------------------------------------------------------------
# nested (added_cond_kwargs) input resolution at bind (arm-events lane)
# ---------------------------------------------------------------------------


def _nested_contract() -> Any:
    entry = _entry(inputs=[
        {"name": "sample", "position": 0, "dtype": "bfloat16",
         "shape": [2, 4, 128, 128]},
        {"name": "text_embeds", "position": 3, "dtype": "bfloat16",
         "shape": [2, 1280]},
    ])
    return aot_serve.contract_from_meta(entry)


def test_missing_nested_input_still_refuses_by_name() -> None:
    """Live refusal (pod ae2uc81yub0gyq): 'text_embeds' declared positional
    by the export but passed NESTED in added_cond_kwargs by every diffusers
    caller. The resolve half lives in test_aot_adapter_fork_pgw790's nested
    marshal test; the refusal half — nested container present, leaf absent —
    must still name the input."""
    contract = _nested_contract()
    with pytest.raises(aot_serve.IngressContractError) as exc:
        aot_serve.bind_call_inputs(
            contract, (FakeTensor([2, 4, 128, 128]),),
            {"added_cond_kwargs": {"time_ids": FakeTensor([2, 6])}})
    assert exc.value.reason == "input_missing"
