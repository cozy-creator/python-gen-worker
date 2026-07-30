"""pgw#784: the serving side of an out-of-process mint.

Three claims, each with the fact that would break it:

1. **The live pipeline is never armed for a delegated mint.** In the
   in-process shape the serving pipe carries guarded wrappers, LoRA branch
   containers and a process-global ``TORCHINDUCTOR_CACHE_DIR`` move for the
   whole mint; delegated, it carries none of that and keeps serving plain
   eager. That is also why the "one live capture per process" restriction
   lifts — the capture is the child's.
2. **The child computes the SAME key the parent will demand.** The parent
   states the compile contract on the wire because the parent owns the key
   (the ck2 ``contract`` axis digests ``CompileCell.contract_facts()``, and
   the class-scoped unions live on the spec, not the decorator). If the child
   re-derived it, the parent would refuse its own artifact on an axis nobody
   changed.
3. **Failure inversion.** A dead mint process is a FAILED MINT reported by a
   LIVE worker. Every branch returns a typed result, the retry is bounded and
   class-driven, and a card with no room for a co-resident child DECLINES
   without spawning anything.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

from gen_worker import compile_cache as cc
from gen_worker import fleet_cells, mint_budget, mint_delegate
from gen_worker import mint_process as mp
from gen_worker.api.decorators import DynamicDim
from gen_worker.registry import CompileCell

GIB = 1 << 30
STUB_MODULE = "harness.mint_child_stub"


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
        cc, "begin_fleet_mint",
        lambda *a, **k: armed.append("begin_fleet_mint"))
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled", lambda *a, **k: False)
    monkeypatch.setattr(cc, "has_compile_target", lambda *a, **k: True)
    monkeypatch.setattr(cc, "mandatory_serving", lambda pipe: False)
    monkeypatch.setattr(cc, "apply_lora_lane", lambda pipe, bucket: True)
    monkeypatch.setattr(cc, "drop_lora_lane", lambda pipe: True)
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "fp8")
    monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)
    # gw#608's seeded-cell gate exists only because an IN-PROCESS capture
    # moves the process-global inductor dir. Assert it does not block a
    # delegated mint, whose capture lives in the child.
    monkeypatch.setattr(cc, "delivered_cell_seeded", lambda: True)
    # No CUDA on this box, so the real sm axis is unavailable; the key itself
    # is not what this test is about.
    monkeypatch.setattr(
        fleet_cells.cell_key, "compute",
        lambda *a, **k: SimpleNamespace(digest="ck5-test"))

    prior = dict(os.environ)
    outcome = fleet_cells.enable_compiled(
        SimpleNamespace(), _cfg(), tmp_path, None, None, delegate=True)

    assert armed == [], "a delegated mint must not arm the serving pipeline"
    assert not outcome.armed, (
        "armed=False is the honest answer — this pipe serves EAGER while the "
        "child compiles")
    pending = outcome.self_mint
    assert pending is not None and pending.delegated
    assert pending.cell_key and pending.target.name.endswith(".tar.gz")
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        assert os.environ.get(key) == prior.get(key), (
            "the process-global inductor cache dir must not move: the child "
            "owns the capture (gw#608's root cause, avoided by construction)")


# --------------------------------------------- 2. one key, stated by one side

def test_the_wire_form_preserves_the_contract_facts_exactly(
    tmp_path: Path,
) -> None:
    """The load-bearing key-parity check: flatten -> ship -> rebuild must be a
    round trip on ``contract_facts()``, because that dict IS the ck2 contract
    axis. A dropped union or a lost dynamic row silently forks the key."""
    from gen_worker import mint_child

    parent = _cfg()
    rebuilt = mint_child.compile_cfg(mint_delegate.cfg_spec(parent))
    assert rebuilt.contract_facts() == parent.contract_facts()
    assert rebuilt.contract_text_lens() == parent.contract_text_lens()
    assert rebuilt.guidance_scales == parent.guidance_scales
    assert rebuilt.lora_bucket == parent.lora_bucket == 64


def test_the_request_carries_the_lane_and_the_effective_config(
    tmp_path: Path,
) -> None:
    """Both steer the warm forwards, so both must be the PARENT's values —
    a child warming at different config traces different graphs and the
    parent's own proof then misses."""
    pending = SimpleNamespace(
        family="sdxl", cell_key="ck5-abc", cfg=_cfg(),
        capture_dir=tmp_path / "capture", target=tmp_path / "cell.tar.gz",
        mint_root=tmp_path)
    task = mint_delegate.MintTask(
        pending=pending, pipe=object(), function="gen",
        modules=("app",), snapshots={"pipeline": "/cas/sdxl"},
        lane="fp8-w8a16", configs={"gen": {"steps": 28}}, device=3)
    req = mint_delegate.build_request(
        task, workdir=tmp_path / "w", cap_bytes=7 * GIB)
    assert req.lane == "fp8-w8a16"
    assert req.configs == {"gen": {"steps": 28}}
    assert req.snapshots == {"pipeline": "/cas/sdxl"}
    assert req.device == 3 and req.vram_cap_bytes == 7 * GIB
    assert req.capture == str(tmp_path / "capture")


# --------------------------------------------------- 3. failure inversion

def _task(tmp_path: Path, **over: Any) -> mint_delegate.MintTask:
    pending = fleet_cells.PendingSelfMint(
        family="sdxl", cell_key="ck5-abc",
        ref="root/family-sdxl#ck5-abc", cfg=_cfg(),
        target=tmp_path / "cell.tar.gz", capture_dir=tmp_path / "capture",
        mint_root=tmp_path / "root", publisher=None, cache_dir=tmp_path,
        delegated=True)
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
        lambda kind, detail, phase="": seen.append((kind, phase, detail)))
    monkeypatch.setattr(
        fleet_cells.activity_mod, "emit_event",
        lambda kind, detail, phase="": seen.append((kind, phase, detail)))
    return seen


def test_a_dead_mint_process_is_a_failed_mint_not_a_dead_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """th#1299 inverted: the mint dies, the worker does not. No exception
    escapes, the reason rides the wire typed, and serving is untouched."""
    monkeypatch.setenv("MINT_STUB_MODE", "sigkill")
    _fake_card(monkeypatch, total_gib=80, resident_gib=6)
    seen = _events(monkeypatch)
    result = asyncio.run(mint_delegate.build_cell(
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
    result = asyncio.run(mint_delegate.build_cell(
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
    result = asyncio.run(mint_delegate.build_cell(
        _task(tmp_path), act=_Act(), max_attempts=2))
    assert result.status == mint_delegate.FAILED
    assert result.attempts == 2


def test_no_room_for_a_co_resident_child_declines_without_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wan-2.2 shape. A decline is an OUTCOME (pgw#737): eager serving,
    cell absent, typed self_mint_skipped — and crucially no child is ever
    started, so nothing can OOM the tenant."""
    monkeypatch.setenv("MINT_STUB_MODE", "minted")
    _fake_card(monkeypatch, total_gib=80, resident_gib=54)
    spawned: List[Any] = []
    monkeypatch.setattr(
        mp, "run_mint",
        lambda *a, **k: spawned.append(a) or asyncio.sleep(0))
    seen = _events(monkeypatch)
    result = asyncio.run(mint_delegate.build_cell(_task(tmp_path), act=_Act()))
    assert result.declined and result.status == mint_delegate.DECLINED
    assert not spawned
    skips = [e for e in seen if e[0] == "self_mint_skipped"]
    assert skips and skips[0][1] == "insufficient_vram"
    assert "needed~=" in skips[0][2] and "headroom=" in skips[0][2]


def test_a_child_peak_is_banked_for_the_next_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NO MAGIC NUMBERS: the child measures its own peak and the next ask on
    this pod is that fact, not this module's arithmetic."""
    monkeypatch.setenv("MINT_STUB_MODE", "minted")
    monkeypatch.setenv("MINT_STUB_PEAK", str(11 * GIB))
    _fake_card(monkeypatch, total_gib=80, resident_gib=6)
    monkeypatch.setattr(
        fleet_cells, "adopt_delegated_mint",
        lambda pipe, pending, artifact: fleet_cells.SelfMint(
            family="sdxl", cell_key="k", ref="r", snapshot_digest="d",
            artifact=Path(artifact)))
    monkeypatch.setattr(mint_budget, "_CHILD_PEAKS", {})
    asyncio.run(mint_delegate.build_cell(
        _task(tmp_path, weight_lane="w8a8"), act=_Act()))
    assert mint_budget.child_peak("sdxl", "w8a8") == 11 * GIB


def test_delegation_is_the_default_and_the_in_process_shape_is_a_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-process shape stays reachable only to RED-VERIFY that it
    violates the liveness contract — never as a supported mode."""
    monkeypatch.delenv(mint_delegate.ENV_IN_PROCESS, raising=False)
    assert mint_delegate.delegated()
    monkeypatch.setenv(mint_delegate.ENV_IN_PROCESS, "1")
    assert not mint_delegate.delegated()
