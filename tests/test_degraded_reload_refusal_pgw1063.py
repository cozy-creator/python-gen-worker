"""pgw#1063: a degraded reload that cannot fit its own staging REFUSES.

ie#615, pod ``nzlrzusxjl8tm1`` (233.76 GiB cgv1 ceiling, gen-worker 0.96.0),
verbatim off the postmortem dial:

1. a request died CUDA OOM mid-inference (an endpoint defect, fixed there);
2. the ladder engaged ``rung=off->model_offload``, quarantined the instance
   "for a clean offloaded reload", and the retry was re-admitted **on the
   same worker**;
3. that reload re-staged the 105 GB set in the same process whose prior
   staging anon was still resident. At the ceiling every fault went through
   direct reclaim: ``read_bytes`` reached **1.578 TB — a 15x re-read of a
   105 GB set** — rss_anon 232.9 GiB, for **37 minutes**, until the kernel
   OOM-killed the child. $2+ of billed H100 to reach a death that was
   arithmetically certain at minute zero.

Three defects, one per section below:

* the offload rung was charged pgw#1026's per-component discount, which is
  the loader's promise that each component LEAVES the host for the card — a
  promise an offloaded pipeline cannot keep, since offloading IS keeping the
  weights on the host;
* the degrade was never priced at decision time, so a reload that could not
  fit was prescribed anyway;
* the load dial counted re-reads as progress (1.578 TB of "advancement" for
  a 105 GB set), so nothing — not the stall clock, not the next component's
  admission — could see a load that had stopped making any.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gen_worker.api.binding import wire_ref
from gen_worker.capability import HostRamCapacityError
from gen_worker.models import load_progress as lp
from gen_worker.models import loading as loading_mod
from gen_worker.models.loading import (
    _admit_component_staging,
    decide_streamed_hydration,
    plan_streamed_hydration,
)
from gen_worker.models.memory import (
    OFFLOAD_LADDER,
    keeps_weights_in_host_ram,
)
from gen_worker.models.residency import HostRamHeadroom

from harness.modular_endpoint import build_base_tree

_GIB = 1024 ** 3


def _ram(total_gb: float, available_gb: float):
    from gen_worker.models.memory import HostRam

    return HostRam(
        total_gb=total_gb, available_gb=available_gb,
        meminfo_total_gb=total_gb, meminfo_available_gb=available_gb,
        cgroup_limit_gb=total_gb, source="cgroup",
    )


def _h3(**over: Any):
    """ie#615's measured shape: 134.1 GiB tree, 46 GiB largest component,
    116.4 GiB host, a card set that holds the tree."""
    kwargs: Dict[str, Any] = dict(
        tree_bytes=int(134.1 * _GIB),
        largest_unit_bytes=int(46.0 * _GIB),
        unit_count=6,
        host_total_bytes=int(116.4 * _GIB),
        device_free_bytes=int(141.0 * _GIB),
    )
    kwargs.update(over)
    return decide_streamed_hydration(**kwargs)


# ---------------------------------------------------------------------------
# 1. the discount is the loader's promise, and an offload rung cannot keep it
# ---------------------------------------------------------------------------


def test_an_offload_rung_never_takes_the_per_component_discount() -> None:
    assert _h3().engaged, "the resident shape must still engage"
    for rung in OFFLOAD_LADDER:
        plan = _h3(placement_mode=rung)
        assert not plan.engaged, rung
        assert "host RAM" in plan.reason and rung in plan.reason


def test_the_resident_rungs_are_untouched() -> None:
    for rung in ("", "auto", "off", "vae_only"):
        assert _h3(placement_mode=rung).engaged, rung
        assert not keeps_weights_in_host_ram(rung)


def test_the_rung_travels_with_the_plan(tmp_path, monkeypatch) -> None:
    """The measuring wrapper carries it, so the executor and the loader
    cannot reach different verdicts from the same tree."""
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    monkeypatch.setattr(loading_mod, "probe_host_ram", lambda: _ram(64.0, 60.0))

    plan = plan_streamed_hydration(
        tree, device_free_bytes=80 * _GIB, placement_mode="group_offload")

    assert plan.placement_mode == "group_offload"
    assert not plan.engaged
    assert "group_offload" in plan.summary()


# ---------------------------------------------------------------------------
# 2. the admission charges an offloaded reload its WHOLE TREE
# ---------------------------------------------------------------------------


def _sparse_h3_tree(root: Path) -> Path:
    """A real modular tree whose component dirs measure H3's sizes; the
    files are sparse, so ``st_size`` is real and the blocks are not."""
    root.mkdir(parents=True, exist_ok=True)
    sizes = {
        "text_encoder": int(60.0 * _GIB),
        "transformer": int(46.0 * _GIB),
        "transformer_2": int(26.0 * _GIB),
        "vae": int(2.1 * _GIB),
    }
    index: Dict[str, Any] = {
        "_class_name": "TinyModularPipeline",
        "_blocks_class_name": "TinyBlocks",
    }
    for name, nbytes in sizes.items():
        d = root / name
        d.mkdir()
        with open(d / "model.safetensors", "wb") as f:
            f.truncate(nbytes)
        index[name] = ["diffusers", "UNet2DConditionModel", {
            "pretrained_model_name_or_path": "upstream/h3",
            "subfolder": name, "variant": None, "revision": None,
        }]
    (root / "modular_model_index.json").write_text(json.dumps(index))
    return root


def _modular_spec():
    from gen_worker.registry import extract_specs

    from harness.modular_endpoint import ModularEndpoint

    return extract_specs(ModularEndpoint)[0]


def _executor(spec):
    from gen_worker.executor import Executor

    async def _send(_msg: Any) -> None:
        return None

    return Executor([spec], _send)


async def _admit(ex, spec, paths):
    await ex._ensure_host_ram_for(spec, paths)


def _host_of(monkeypatch, total_gb: float, avail_gb: float,
             device_free_gb: float) -> None:
    monkeypatch.setattr(
        loading_mod, "probe_host_ram", lambda: _ram(total_gb, avail_gb))
    monkeypatch.setattr(
        "gen_worker.models.memory.probe_host_ram",
        lambda **_: _ram(total_gb, avail_gb))
    monkeypatch.setattr(
        loading_mod, "get_available_vram_gb", lambda *a, **k: device_free_gb)


def test_a_degraded_ref_is_charged_its_tree_and_refuses(
    tmp_path, monkeypatch,
) -> None:
    """The incident's admission: 134.1 GiB of tree on a 116.4 GiB host with
    a card that holds it. Resident, that is a boot (pgw#1026). On the rung
    the OOM degrade just learned, it is a structural refusal — and the
    refusal is what did not happen."""
    tree = _sparse_h3_tree(tmp_path / "h3")
    _host_of(monkeypatch, 116.4, 110.0, 141.0)
    spec = _modular_spec()
    ex = _executor(spec)
    ex.degraded_floor[wire_ref(spec.models["pipeline"])] = "model_offload"

    with pytest.raises(HostRamCapacityError) as exc:
        asyncio.run(_admit(ex, spec, {"pipeline": str(tree)}))
    assert exc.value.required_bytes > exc.value.total_bytes


def test_without_the_degraded_floor_the_same_tree_still_boots(
    tmp_path, monkeypatch,
) -> None:
    """pgw#1026 is untouched on the resident path — this issue may only turn
    a boot into a refusal where the rung makes the discount a lie."""
    tree = _sparse_h3_tree(tmp_path / "h3")
    _host_of(monkeypatch, 116.4, 110.0, 141.0)
    spec = _modular_spec()

    asyncio.run(_admit(_executor(spec), spec, {"pipeline": str(tree)}))


# ---------------------------------------------------------------------------
# 3. the degrade is PRICED before it is prescribed
# ---------------------------------------------------------------------------


def _degrade_executor(monkeypatch, *, tree_bytes: int, headroom: HostRamHeadroom):
    spec = _modular_spec()
    ex = _executor(spec)
    res = ex.store.residency
    monkeypatch.setattr(res, "local_path", lambda ref: Path("/nonexistent"))
    monkeypatch.setattr(res, "host_ram_headroom", lambda needed: headroom)
    monkeypatch.setattr(
        loading_mod.disk_gc, "tree_bytes", lambda p: tree_bytes)
    monkeypatch.setattr(
        "gen_worker.executor.disk_gc.tree_bytes", lambda p: tree_bytes)
    sent: List[Any] = []

    async def _record(refs, error):
        sent.append((list(refs), error))

    monkeypatch.setattr(ex, "_record_host_ram_failure", _record)
    return ex, spec, sent


_INCIDENT_SET = int(105e9)


def _headroom(available_gb: float, total_gb: float) -> HostRamHeadroom:
    floor = int(8 * _GIB)
    return HostRamHeadroom(
        available_bytes=int(available_gb * _GIB),
        floor_bytes=floor,
        required_bytes=_INCIDENT_SET + floor,
        total_bytes=int(total_gb * _GIB),
    )


def test_a_structurally_unfittable_offload_reload_is_refused(
    monkeypatch,
) -> None:
    """No eviction, and no identically-sized pod, can hold it: report the
    hardware axis, disable the function here, do not reload."""
    ex, spec, sent = _degrade_executor(
        monkeypatch, tree_bytes=_INCIDENT_SET,
        headroom=_headroom(available_gb=20.0, total_gb=64.0))

    verdict = asyncio.run(ex._refuse_unfittable_offload(
        spec, [("hub/h3:prod", "", "model_offload", 0.0)]))

    assert verdict and "host RAM" in verdict
    assert len(sent) == 1
    assert isinstance(sent[0][1], HostRamCapacityError)


def test_a_reload_that_does_not_fit_RIGHT_NOW_blocks_the_ref_here(
    monkeypatch,
) -> None:
    """The incident's own shape: it fits a 233.76 GiB host, just not this
    one while the quarantined instance's staging is still resident. The
    rung is still learned; what must not happen is the retry landing back
    here mid-reload."""
    ex, spec, sent = _degrade_executor(
        monkeypatch, tree_bytes=_INCIDENT_SET,
        headroom=_headroom(available_gb=100.0, total_gb=233.76))

    verdict = asyncio.run(ex._refuse_unfittable_offload(
        spec, [("hub/h3:prod", "", "model_offload", 0.0)]))

    assert verdict == ""  # the ladder proceeds
    assert len(sent) == 1
    assert sent[0][1].reason == "insufficient_host_ram"


def test_an_affordable_reload_says_nothing(monkeypatch) -> None:
    ex, spec, sent = _degrade_executor(
        monkeypatch, tree_bytes=_INCIDENT_SET,
        headroom=_headroom(available_gb=200.0, total_gb=233.76))

    assert asyncio.run(ex._refuse_unfittable_offload(
        spec, [("hub/h3:prod", "", "model_offload", 0.0)])) == ""
    assert sent == []


def test_a_resident_rung_is_not_priced_as_an_offload(monkeypatch) -> None:
    """Only a rung that keeps weights on the host owes this price."""
    ex, spec, sent = _degrade_executor(
        monkeypatch, tree_bytes=_INCIDENT_SET,
        headroom=_headroom(available_gb=1.0, total_gb=64.0))

    assert asyncio.run(ex._refuse_unfittable_offload(
        spec, [("hub/h3:prod", "", "vae_only", 0.0)])) == ""
    assert sent == []


# ---------------------------------------------------------------------------
# 4. re-reads are not progress
# ---------------------------------------------------------------------------


def _reporter(monkeypatch, *, total_bytes: int, limit_gb: float,
              events: Optional[List] = None) -> lp.LoadProgressReporter:
    if events is not None:
        monkeypatch.setattr(
            lp.activity_mod, "emit_event",
            lambda kind, detail="", phase="", duration_ms=0: events.append(
                (kind, detail)))
    monkeypatch.setattr(
        "gen_worker.models.memory.cgroup_memory_limit_bytes",
        lambda *a, **k: int(limit_gb * _GIB))
    rep = lp.LoadProgressReporter(
        "h3:hub/h3:prod", total_bytes, marker_path=Path("/dev/null"))
    # What `start()` does, without its thread: the load began at read 0, so
    # the tick's own delta arithmetic is the production one.
    rep._io0 = 0
    return rep


def _drive(rep, monkeypatch, *, read: int, anon: int) -> None:
    monkeypatch.setattr(lp, "_proc_read_bytes", lambda: read)
    monkeypatch.setattr(lp, "_proc_rss_anon_kb", lambda: anon // 1024)
    rep._tick()


def test_the_incident_is_named_while_the_worker_is_still_alive(
    tmp_path, monkeypatch,
) -> None:
    """1.578 TB read for a 105 GB set, anon against the ceiling, and the
    staging has not produced one copy of it."""
    events: List = []
    rep = _reporter(
        monkeypatch, total_bytes=_INCIDENT_SET, limit_gb=233.76, events=events)
    _drive(rep, monkeypatch, read=0, anon=int(130 * _GIB))
    rep.set_phase("hydrate:transformer", int(46 * _GIB))

    _drive(rep, monkeypatch, read=int(1.578e12), anon=int(232.9 * _GIB))

    assert rep.thrash, "the crawl must name itself"
    assert "re-reading" in rep.thrash
    assert any(k == lp.EVENT_PHASE_THRASH for k, _ in events)


def test_re_reads_stop_advancing_the_counter(tmp_path, monkeypatch) -> None:
    """The stall clock runs on counter advancement; crediting a 15x re-read
    as progress is what made a stalled load look busy for 37 minutes."""
    rep = _reporter(monkeypatch, total_bytes=_INCIDENT_SET, limit_gb=233.76)
    _drive(rep, monkeypatch, read=0, anon=0)
    rep.set_phase("hydrate:transformer", int(46 * _GIB))

    _drive(rep, monkeypatch, read=_INCIDENT_SET, anon=0)
    once = rep._staged
    _drive(rep, monkeypatch, read=int(1.578e12), anon=0)

    assert rep._staged == once <= _INCIDENT_SET


def test_a_healthy_cold_load_never_trips_it(tmp_path, monkeypatch) -> None:
    """One pass over the bytes, anon growing with it: the ordinary shape of
    every load this worker runs."""
    rep = _reporter(monkeypatch, total_bytes=_INCIDENT_SET, limit_gb=233.76)
    _drive(rep, monkeypatch, read=0, anon=0)
    rep.set_phase("hydrate:transformer", int(46 * _GIB))

    for frac in (0.25, 0.5, 0.75, 1.0):
        staged = int(frac * 46 * _GIB)
        _drive(rep, monkeypatch, read=staged, anon=staged)
        assert not rep.thrash, frac


def test_a_big_slow_load_far_from_the_ceiling_never_trips_it(
    tmp_path, monkeypatch,
) -> None:
    """Re-reads alone are not the threat — reclaim pressure is. A host with
    room is allowed to read as much as it likes."""
    rep = _reporter(monkeypatch, total_bytes=_INCIDENT_SET, limit_gb=233.76)
    _drive(rep, monkeypatch, read=0, anon=0)
    rep.set_phase("hydrate:transformer", int(46 * _GIB))

    _drive(rep, monkeypatch, read=int(10 * 46 * _GIB), anon=int(20 * _GIB))

    assert not rep.thrash


def test_growing_anon_at_the_ceiling_is_the_incident_not_an_excuse(
    tmp_path, monkeypatch,
) -> None:
    """ie#615's anon DID keep growing (130 -> 232.9 GiB) — the estimate was
    simply wrong about how much the set weighed. Staging more, at the
    ceiling, while re-reading, is the crawl itself; treating growth as proof
    of progress would exempt exactly the incident this issue is about."""
    rep = _reporter(monkeypatch, total_bytes=_INCIDENT_SET, limit_gb=233.76)
    _drive(rep, monkeypatch, read=0, anon=int(180 * _GIB))
    rep.set_phase("hydrate:transformer", int(46 * _GIB))

    _drive(rep, monkeypatch, read=int(10 * 46 * _GIB), anon=int(230 * _GIB))

    assert rep.thrash


def test_the_measured_verdict_outranks_the_estimate(monkeypatch) -> None:
    """Admission arithmetic said yes; the process has been measured saying
    otherwise. Nothing further is admitted into the crawl."""
    monkeypatch.setattr(loading_mod, "probe_host_ram", lambda: _ram(233.76, 100.0))
    monkeypatch.setattr(
        loading_mod.load_progress, "thrash_verdict",
        lambda: "hydrate:transformer: read 1.5 TiB for a 46 GiB set")

    with pytest.raises(HostRamCapacityError) as exc:
        _admit_component_staging("vae", int(2 * _GIB))
    assert "re-read crawl" in str(exc.value)


def test_without_a_verdict_the_arithmetic_still_decides(monkeypatch) -> None:
    monkeypatch.setattr(loading_mod, "probe_host_ram", lambda: _ram(233.76, 200.0))

    _admit_component_staging("vae", int(2 * _GIB))
