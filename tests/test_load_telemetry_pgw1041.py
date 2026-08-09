"""pgw#1041: the load path reports progress, staging is admitted against the
cgroup budget per component, and a SIGKILL'd load leaves a postmortem
breadcrumb — the ie#615 attempt-4 defects (94 silent minutes, an
unattributable kill on a cgroup-v1 host) each pinned by a test.

Integration-style: the hydration tests run REAL diffusers modular pipelines
(the pgw#1036 harness trees); the v1 postmortem tests read a real on-disk
cgroup-v1 layout (the exact files measured on the AP-JP-1 H100 host)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from gen_worker import postmortem
from gen_worker import progress as progress_mod
from gen_worker.capability import (
    HostRamCapacityError,
    InsufficientHostRamError,
)
from gen_worker.models import load_progress, provision
from gen_worker.models.loading import load_from_pretrained
from gen_worker.models.memory import HostRam

from harness.modular_endpoint import TinyModularPipeline, build_base_tree


# ---------------------------------------------------------------------------
# cgroup v1 postmortem (the attempt-4 blindness: memory.max=unlimited,
# memory.events={} on a host whose v1 limit was 233.8GiB)
# ---------------------------------------------------------------------------


def _v1_tree(tmp_path: Path) -> Path:
    v1 = tmp_path / "memory"
    v1.mkdir()
    (v1 / "memory.limit_in_bytes").write_text("250999996416\n")
    (v1 / "memory.usage_in_bytes").write_text("250053111808\n")
    (v1 / "memory.max_usage_in_bytes").write_text("250999996416\n")
    (v1 / "memory.oom_control").write_text(
        "oom_kill_disable 0\nunder_oom 0\noom_kill 3\n")
    return v1


def test_container_limits_v1_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(postmortem, "_deepest", lambda name: None)
    monkeypatch.setattr(postmortem, "_V1_MEM", _v1_tree(tmp_path))
    limits = postmortem.container_limits()
    assert limits["cgroup_flavor"] == "v1"
    assert limits["memory_max_bytes"] == 250999996416
    assert limits["memory_current_bytes"] == 250053111808
    assert limits["memory_peak_bytes"] == 250999996416
    assert limits["memory_events"]["oom_kill"] == 3


def test_oom_kill_count_v1(tmp_path, monkeypatch):
    monkeypatch.setattr(postmortem, "_deepest", lambda name: None)
    monkeypatch.setattr(postmortem, "_V1_MEM", _v1_tree(tmp_path))
    assert postmortem.oom_kill_count() == 3


def test_v1_unlimited_sentinel_reads_none(tmp_path, monkeypatch):
    v1 = tmp_path / "memory"
    v1.mkdir()
    (v1 / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    monkeypatch.setattr(postmortem, "_deepest", lambda name: None)
    monkeypatch.setattr(postmortem, "_V1_MEM", v1)
    limits = postmortem.container_limits()
    assert limits["cgroup_flavor"] == "v1"
    assert limits["memory_max_bytes"] is None


def test_format_detail_names_v1_oom(tmp_path, monkeypatch):
    """The attempt-4 report said 'memory.max=unlimited memory.events={}';
    on the same host the v1 fallback must name the ceiling and the kill."""
    monkeypatch.setattr(postmortem, "_deepest", lambda name: None)
    monkeypatch.setattr(postmortem, "_V1_MEM", _v1_tree(tmp_path))
    detail = postmortem.format_detail(
        phase="compute_process_exit",
        verdict={"signaled": True, "signal": 9, "signal_name": "SIGKILL",
                 "exit_code": 137},
        limits=postmortem.container_limits(),
        oom_kill_delta=1,
    )
    assert "memory.max=233.76GiB" in detail
    assert "THE KERNEL OOM-KILLED US" in detail


# ---------------------------------------------------------------------------
# load-progress breadcrumbs + counter
# ---------------------------------------------------------------------------


def test_reporter_ticks_counter_and_breadcrumb(tmp_path):
    marker = tmp_path / "load-progress.json"
    rep = load_progress.LoadProgressReporter(
        "pipeline:test/ref", 1000, marker_path=marker, interval_s=0.2)
    rep.start()
    try:
        rep.set_phase("hydrate:unet")
        time.sleep(0.5)
        record = json.loads(marker.read_text())
        assert record["phase"] == "hydrate:unet"
        assert record["label"] == "pipeline:test/ref"
        assert record["staged_bytes"] >= 0
        snaps = {s.name: s for s in progress_mod.snapshot()}
        assert load_progress.COUNTER_NAME in snaps
    finally:
        rep.stop(clean=True)
    assert not marker.exists()
    assert load_progress.COUNTER_NAME not in {
        s.name for s in progress_mod.snapshot()}


def test_unclean_stop_leaves_breadcrumb_for_death_attribution(tmp_path):
    marker = tmp_path / "load-progress.json"
    rep = load_progress.LoadProgressReporter(
        "pipeline:test/ref", 1000, marker_path=marker, interval_s=0.2)
    rep.start()
    rep.set_phase("hydrate:transformer")
    rep.stop(clean=False)
    assert marker.exists()

    extra = postmortem.attribute_signal_death(
        signal_name="SIGKILL",
        inflight_path=tmp_path / "no-inflight.json",
        registry_path=tmp_path / "no-registry.json",
        dump_path=tmp_path / "no-dump.txt",
        load_progress_path=marker,
    )
    assert extra["last_load_progress"]["phase"] == "hydrate:transformer"
    assert not marker.exists()  # consumed


# ---------------------------------------------------------------------------
# per-component staging admission (pgw#1026's minimal per-stage form)
# ---------------------------------------------------------------------------


def _ram(total_gb: float, available_gb: float) -> HostRam:
    return HostRam(
        total_gb=total_gb, available_gb=available_gb,
        meminfo_total_gb=total_gb, meminfo_available_gb=available_gb,
        cgroup_limit_gb=total_gb, source="cgroup",
    )


def test_component_admission_refuses_transient(tmp_path, monkeypatch):
    """A component that cannot fit CURRENT headroom refuses typed with the
    measured numbers instead of dying silently to the kernel."""
    from gen_worker.models import loading as loading_mod

    tree = build_base_tree(tmp_path / "base", fill=1.0)
    monkeypatch.setattr(
        loading_mod, "probe_host_ram", lambda: _ram(64.0, 0.001))
    with pytest.raises(InsufficientHostRamError) as exc:
        load_from_pretrained(TinyModularPipeline, tree)
    assert "modular component" in str(exc.value)
    assert exc.value.required_bytes > exc.value.available_before_bytes


def test_component_admission_refuses_structural(tmp_path, monkeypatch):
    """A component bigger than the whole cgroup is the pgw#752 hardware
    verdict (disable the function here), not a retry."""
    from gen_worker.models import loading as loading_mod

    tree = build_base_tree(tmp_path / "base", fill=1.0)
    monkeypatch.setattr(
        loading_mod, "probe_host_ram", lambda: _ram(2.0, 2.0))
    monkeypatch.setattr(
        loading_mod.disk_gc, "tree_bytes", lambda p: 10 * 1024**3)
    with pytest.raises(HostRamCapacityError):
        load_from_pretrained(TinyModularPipeline, tree)


def test_hydration_reports_per_component_phases(tmp_path, monkeypatch):
    """The 94-minute black box: hydration must announce each component."""
    from gen_worker.models import loading as loading_mod

    phases: list[str] = []
    monkeypatch.setattr(
        loading_mod.load_progress, "set_phase",
        lambda phase: phases.append(phase))
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    pipe = load_from_pretrained(TinyModularPipeline, tree)
    assert pipe.unet is not None
    hydrated = [p.split(":", 1)[1] for p in phases if p.startswith("hydrate:")]
    assert "unet" in hydrated and "vae" in hydrated


def test_load_slot_clears_breadcrumb_on_success(tmp_path, monkeypatch):
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    marker = marker_dir / "load-progress.json"
    monkeypatch.setattr(postmortem, "LOAD_PROGRESS_PATH", marker)
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    sl = provision.load_slot(
        TinyModularPipeline, str(tree), slot="pipeline",
        ref="test/tiny:latest", mode="auto", device="cpu")
    assert sl.obj is not None
    assert not marker.exists()
    assert load_progress.COUNTER_NAME not in {
        s.name for s in progress_mod.snapshot()}
