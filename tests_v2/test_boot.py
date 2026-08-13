"""Boot & lifecycle scenarios (design domains 1 + the boot half of 2).

Scenario shape: each test is a WALK that asserts many behaviors of one real
boot — an in-process Worker over the hub double, or a real
``python -m gen_worker.entrypoint`` subprocess dialing a live scheduler
socket. Refusals are first-class: the matrix at the bottom pins the typed,
named, hub-visible failure of every boot gate.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest

import gen_worker.lifecycle as lifecycle_mod
from gen_worker import boot_phases, entrypoint, env_seal
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.hardware_report_hub import recording_hub
from harness.hub_double import is_ready
from harness.subprocess_runner import (
    assert_no_unhandled_crash,
    cpu_manifest_entry,
    gpu_manifest_entry,
    run_entrypoint,
    startup_phase_lines,
)

from tests_v2 import catalog
from tests_v2.conftest import manifest_entry, spawn_entrypoint, standalone_scheduler


def _boot_rows(conn) -> List[pb.BootPhase]:
    return [m.boot_phase for m in list(conn.received)
            if m.WhichOneof("msg") == "boot_phase"]


def _phase_names(phases: List[Dict]) -> List[str]:
    return [p.get("phase") for p in phases]


# ---------------------------------------------------------------------------
# Scenario 1 — cold boot to READY over the hub double: identity, discovery,
# gating, the boot span ladder, and the app-level heartbeat.
# ---------------------------------------------------------------------------


def test_cold_boot_walks_to_ready_and_heartbeats(hub, monkeypatch) -> None:
    monkeypatch.setattr(lifecycle_mod, "HEARTBEAT_INTERVAL_MS", 100)
    with hub(worker_id="v2-boot") as (scheduler, harness):
        conn = scheduler.wait_connection(0)

        # Hello/HelloAck: protocol + identity + declared heartbeat cadence.
        assert conn.hello is not None
        assert conn.hello.protocol_version == pb.PROTOCOL_VERSION_CURRENT
        assert conn.hello.worker_id == "v2-boot"
        assert conn.hello.worker_session_id
        assert conn.hello.lifecycle_snapshot.full_replace
        assert conn.hello.heartbeat_interval_ms == 100

        # Function discovery: every model-free catalog row is advertised at
        # READY; hub-bound rows are GATED (loading, never available) until
        # residency arrives — a worker must not advertise what it cannot serve.
        ready = conn.wait_for(is_ready).state_delta
        for name in ("echo", "stream3", "slow-stream", "sleepy",
                     "staged-generate", "small-usage"):
            assert name in ready.available_functions, name
        assert "hot-echo" not in ready.available_functions
        assert "hot-echo" in ready.loading_functions

        # Boot span ladder, read OFF THE WIRE from the real boot: the hello
        # milestone and the boot close both emit, the close never precedes
        # hello, and cumulative milestones are never some span's child.
        conn.wait_for(
            lambda m: m.WhichOneof("msg") == "boot_phase"
            and m.boot_phase.phase == boot_phases.PHASE_FIRST_REQUEST_SERVABLE
            and m.boot_phase.terminal
        )
        rows = _boot_rows(conn)
        hello_rows = [r for r in rows if r.phase == boot_phases.PHASE_HELLO and r.terminal]
        servable_rows = [
            r for r in rows
            if r.phase == boot_phases.PHASE_FIRST_REQUEST_SERVABLE and r.terminal
        ]
        assert hello_rows and servable_rows
        assert servable_rows[0].ordinal > hello_rows[0].ordinal
        assert (servable_rows[0].process_uptime_ms
                >= hello_rows[0].process_uptime_ms)
        assert all(r.parent_ordinal == 0 for r in rows if r.cumulative)

        # Heartbeat: with nothing changing, force-sent byte-identical deltas
        # keep flowing (the app-level liveness signal, th#965).
        conn.wait_for_count(lambda m: m.WhichOneof("msg") == "state_delta", 4)
        deltas = [m.state_delta.SerializeToString(deterministic=True)
                  for m in conn.received if m.WhichOneof("msg") == "state_delta"]
        assert any(a == b for a, b in zip(deltas, deltas[1:])), (
            "no two consecutive identical deltas — the beat is not force-sending"
        )

        # Clean shutdown is part of the walk: stop() exits 0, never wedges.
        assert harness.stop() == 0


# ---------------------------------------------------------------------------
# Scenario 2 — the env seal: erase-and-impose, canonical config effective,
# deterministic digest, typed knob refusal, point-of-use drift refusal.
# ---------------------------------------------------------------------------


def test_env_seal_imposes_canonical_config_and_refuses_drift(monkeypatch) -> None:
    import torch

    saved_env = {k: v for k, v in os.environ.items()
                 if k.startswith(env_seal.SCRUB_PREFIXES)}
    saved_flags = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    )
    try:
        # Hostile AND merely-informational base-image vars: ERASED, never fatal
        # (the 0.70.3 allowlist killed every fleet pod on PYTORCH_VERSION).
        monkeypatch.setenv("PYTORCH_VERSION", "2.13.0")
        monkeypatch.setenv("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
        monkeypatch.setenv("TRITON_PTXAS_PATH", "/definitely/wrong")
        monkeypatch.setenv("OMP_NUM_THREADS", "5")
        seal = entrypoint._establish_env_seal()  # must not raise
        for var in ("PYTORCH_VERSION", "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
                    "TRITON_PTXAS_PATH", "OMP_NUM_THREADS"):
            assert var not in os.environ, f"{var} survived the scrub"

        # The canonical surface is EFFECTIVE, not merely recorded — and it IS
        # the ratified serving posture (TF32 on), so mint==serve.
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cudnn.benchmark is False
        from gen_worker import settings_authority as sa

        assert sa.DECLARED_TORCH.items() <= seal["config"].items()

        # The digest is the env_seal cell-key axis: deterministic.
        digest = env_seal.seal_digest(seal)
        assert len(digest) == 16
        assert digest == env_seal.seal_digest(entrypoint._establish_env_seal())

        # Refusal 1: an undeclared knob refuses BY NAME (one-way door).
        with pytest.raises(sa.SettingsImpositionError, match="not_a_knob"):
            sa.impose_torch(overrides={"not_a_knob": "1"})

        # Refusal 2: point-of-use drift refuses, naming the fact and both
        # values — endpoint code mutating config behind our back is a named
        # error, never a silently different graph.
        torch.backends.cudnn.benchmark = True
        with pytest.raises(env_seal.EnvSealError, match="cudnn_benchmark"):
            env_seal.assert_seal_unchanged("tests_v2")
    finally:
        precision, matmul_tf32, cudnn_tf32, benchmark = saved_flags
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.benchmark = benchmark
        for k, v in saved_env.items():
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Scenario 3 — a REAL entrypoint subprocess: seal ordering in the live phase
# stream, dial-in over a real socket, discovery on the wire, SIGUSR2 stack
# dump forensics, clean terminate.
# ---------------------------------------------------------------------------


def test_real_entrypoint_seals_dials_and_dumps_stacks(tmp_path: Path) -> None:
    with standalone_scheduler() as (scheduler, port):
        proc = spawn_entrypoint(
            tmp_path,
            functions=[manifest_entry(name="echo")],
            env_overrides={
                "ORCHESTRATOR_PUBLIC_ADDR": f"127.0.0.1:{port}",
                "WORKER_ID": "v2-subprocess",
            },
        )
        scheduler.worker_alive = lambda: proc.alive
        try:
            conn = scheduler.wait_connection(0)
            assert conn.hello is not None
            assert conn.hello.worker_id == "v2-subprocess"

            # The startup phase stream: the seal is established AFTER settings
            # (boot precedes it) and BEFORE the cache preflight / any CUDA
            # touch; an accelerator-free manifest never probes CUDA at all.
            proc.wait_for_output(
                lambda text: "cache_preflight_ok" in text, "cache preflight")
            phases = proc.phases()
            names = _phase_names(phases)
            assert names.index("boot") < names.index("env_seal")
            assert names.index("env_seal") < names.index("manifest_loaded")
            assert names.index("env_seal") < names.index("cache_preflight_ok")
            assert "cuda_probe_ok" not in names
            seal_phase = next(p for p in phases if p.get("phase") == "env_seal")
            assert seal_phase.get("digest"), "the seal phase must carry its digest"

            # the seal's COST, off the wire. The startup phase lines
            # above prove the seal ran and in what order; they say nothing
            # about what it cost, and "expect ms; prove it" was the issue's own
            # instruction. The library-digest memo nests inside it, so the
            # memo's saving is a subtraction between two real rows. This is the
            # ONLY boot shape that produces these — `env_seal.establish` is an
            # entrypoint/mint-child call, not something an embedded worker
            # does, which is why `boot_phases.SHAPE_ENTRYPOINT` is a shape of
            # its own.
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "boot_phase"
                and m.boot_phase.phase == boot_phases.PHASE_ENV_ESTABLISH
                and m.boot_phase.terminal)
            rows = _boot_rows(conn)
            est = next(r for r in rows
                       if r.phase == boot_phases.PHASE_ENV_ESTABLISH and r.terminal)
            memo = next(r for r in rows
                        if r.phase == boot_phases.PHASE_LIB_MEMO and r.terminal)
            assert memo.parent_ordinal == est.ordinal, (
                "lib_memo must nest inside env_establish or the two "
                "double-count the same seconds")
            assert memo.reason in ("hit", "miss", "partial", "no_libs")
            assert memo.outcome == boot_phases.OUTCOME_OK, (
                "a memo MISS is the expensive branch of a successful phase, "
                "never a refusal")
            assert est.duration_ms >= memo.duration_ms

            # Discovery over the wire: the catalog module baked into the
            # manifest is what the worker advertises.
            ready = conn.wait_for(is_ready).state_delta
            assert "echo" in ready.available_functions
            assert "stream3" in ready.available_functions

            # SIGUSR2 dumps every thread's stack to stderr
            # and the worker keeps running — a wedged pod is diagnosable from
            # any exec channel without killing it.
            proc.send_signal(signal.SIGUSR2)
            proc.wait_for_output(
                lambda text: "Current thread" in text or "Thread 0x" in text,
                "SIGUSR2 stack dump")
            assert proc.alive, "SIGUSR2 must never kill the worker"
        finally:
            proc.terminate_and_wait()


# ---------------------------------------------------------------------------
# Scenario 4 — a torchless image is a first-class boot: the absence is SEALED
# as a fact and the worker still dials the hub and advertises.
# ---------------------------------------------------------------------------


def test_torchless_boot_seals_the_absence_and_still_dials(tmp_path: Path) -> None:
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "torch.py").write_text(
        "raise ImportError('torchless image (tests_v2)')\n")
    repo = Path(__file__).resolve().parents[1]
    with standalone_scheduler() as (scheduler, port):
        proc = spawn_entrypoint(
            tmp_path,
            functions=[manifest_entry(name="echo")],
            env_overrides={
                "ORCHESTRATOR_PUBLIC_ADDR": f"127.0.0.1:{port}",
                "WORKER_ID": "v2-torchless",
                "PYTHONPATH": os.pathsep.join(
                    [str(shim), str(repo), str(repo / "src")]),
            },
        )
        scheduler.worker_alive = lambda: proc.alive
        try:
            conn = scheduler.wait_connection(0)
            assert conn.hello is not None and conn.hello.worker_id == "v2-torchless"
            proc.wait_for_output(lambda text: '"phase":"env_seal"' in text
                                 or '"phase": "env_seal"' in text, "env_seal phase")
            seal_phase = next(
                p for p in proc.phases() if p.get("phase") == "env_seal")
            config = seal_phase.get("config") or {}
            # The seal records the ABSENCE as a keyable fact instead of dying
            # at phase=env_seal (the 0.70.3 torchless regression class).
            assert config.get("torch") == "absent", config
            assert "cuda_matmul_allow_tf32" not in config
            ready = conn.wait_for(is_ready).state_delta
            assert "echo" in ready.available_functions
        finally:
            proc.terminate_and_wait()


# ---------------------------------------------------------------------------
# Scenario 5 — the boot refusal matrix. Every gate's fail-closed half: typed,
# named, structured, and (where the contract says so) delivered to the hub.
# ---------------------------------------------------------------------------


def test_gpu_boot_refusal_is_typed_and_reaches_the_hub(tmp_path: Path) -> None:
    """A GPU-required manifest on this driver-incompatible host must exit
    typed AND dial the hub with the HardwareUnsuitable report — the exact
    wire path that ended the th#986 silent pod_exited blindness."""
    with recording_hub() as (servicer, addr):
        result = run_entrypoint(
            tmp_path, functions=[gpu_manifest_entry()],
            env_overrides={
                "ORCHESTRATOR_PUBLIC_ADDR": addr,
                "WORKER_ID": "v2-gpu-refusal",
            },
        )
        combined = result.stdout + result.stderr
        phases = startup_phase_lines(combined)
        assert_no_unhandled_crash(result, phases)
        assert result.returncode == 1

        names = _phase_names(phases)
        # The seal came first — refusing later gates under a sealed env.
        assert names.index("env_seal") < names.index("cache_preflight_ok")
        fatal = next(p for p in phases if p.get("phase") == "worker_fatal")
        assert fatal.get("phase_context") == "cuda_probe"
        assert fatal.get("exit_code") == 1
        assert "Starting worker..." not in combined

        # the probe fails in the compute child, which holds no
        # credential — it hands the typed report to the parent (boot_fatal,
        # relayed=true) and the PARENT delivers it to the hub before exiting 1.
        report = next(
            p for p in phases if p.get("phase") == "cuda_probe_boot_fatal")
        assert report.get("relayed") is True, report
        servicer.wait_for_message()
        reports = [
            m.hardware_unsuitable for m in servicer.received
            if m.WhichOneof("msg") == "hardware_unsuitable"
        ]
        hw = next(
            (r for r in reports
             if r.reason_class in ("cuda_unavailable", "driver_too_old")),
            None,
        )
        assert hw is not None, [r.reason_class for r in reports]
        assert hw.worker_id == "v2-gpu-refusal"
        assert hw.detail


@pytest.mark.parametrize("case", ["import", "bad_kind", "empty", "absent_manifest"])
def test_boot_refusals_are_structured_never_crashes(tmp_path: Path, case: str) -> None:
    """Every remaining boot gate refuses structured: named phase, nonzero
    exit, no raw-traceback crash, and the absent thing is NAMED."""
    if case == "import":
        result = run_entrypoint(tmp_path, functions=[cpu_manifest_entry()])
    elif case == "bad_kind":
        result = run_entrypoint(
            tmp_path,
            functions=[{"name": "gen", "kind": "not-a-real-kind", "module": "nope"}],
        )
    elif case == "empty":
        result = run_entrypoint(tmp_path, functions=[])
    else:
        result = run_entrypoint(
            tmp_path, functions=[cpu_manifest_entry()],
            env_overrides={"ENDPOINT_LOCK_PATH": str(tmp_path / "missing.lock")},
        )
    combined = result.stdout + result.stderr
    phases = startup_phase_lines(combined)
    assert_no_unhandled_crash(result, phases)
    assert result.returncode != 0

    names = _phase_names(phases)
    if case == "import":
        fatal = next(p for p in phases if p.get("phase") == "worker_fatal")
        assert fatal.get("phase_context") == "import"
        # accelerator=none never touches CUDA.
        assert "cuda_probe_ok" not in names
        assert "GEN_WORKER_CUDA_PROBE_FAILED" not in combined
    elif case == "absent_manifest":
        loaded = next(p for p in phases if p.get("phase") == "manifest_loaded")
        assert loaded.get("status") == "error"
        assert loaded.get("reason") == "missing_or_invalid_manifest"
        assert str(tmp_path / "missing.lock") in (loaded.get("manifest_path") or "")
        # The remedy is named for the operator, not implied.
        assert "ENDPOINT_LOCK_PATH" in combined


def test_seal_refusal_exits_typed_with_settings_loaded(monkeypatch) -> None:
    """The 0.70.3 regression: a seal refusal must exit typed AND carry the
    loaded settings, or the fatal cannot dial the hub and every fleet pod
    dies as a silent pod_exited. Observed at the real _run_main seams."""
    fatal: list = []
    settings = SimpleNamespace(endpoint_lock_path="")
    monkeypatch.setattr(entrypoint, "_install_stack_dump_handler", lambda: None)
    monkeypatch.setattr(entrypoint, "_bootstrap_configuration", lambda: settings)

    def _refuse() -> dict:
        raise env_seal.EnvSealError("config freeze failed: HOSTILE_FACT")

    monkeypatch.setattr(entrypoint, "_establish_env_seal", _refuse)
    monkeypatch.setattr(
        entrypoint, "_log_worker_fatal",
        lambda phase, exc, **kw: fatal.append((phase, str(exc), kw)))

    assert entrypoint._run_main() == 1
    (phase, message, kw), = fatal
    assert phase == "env_seal"
    assert "HOSTILE_FACT" in message
    assert kw.get("settings") is settings, (
        "the env_seal fatal no longer carries settings — the hub dial "
        "precondition 0.70.3 broke"
    )


def test_seal_order_is_settings_then_seal_then_probe(monkeypatch) -> None:
    """The control-flow half of the same contract: settings load FIRST (so a
    refusal can dial typed), the seal SECOND, the CUDA probe only after."""
    order: list = []
    monkeypatch.setattr(entrypoint, "_install_stack_dump_handler", lambda: None)
    monkeypatch.setattr(
        entrypoint, "_bootstrap_configuration",
        lambda: order.append("settings") or SimpleNamespace(endpoint_lock_path=""))
    monkeypatch.setattr(
        entrypoint, "_establish_env_seal", lambda: order.append("seal") or {})
    monkeypatch.setattr(entrypoint, "load_manifest", lambda path: {})
    monkeypatch.setattr(entrypoint, "_preflight_cache_dirs", lambda: None)

    class _ProbeReached(Exception):
        pass

    def _probe(manifest: object) -> bool:
        order.append("probe")
        raise _ProbeReached

    monkeypatch.setattr(entrypoint, "should_probe_cuda", _probe)
    with pytest.raises(_ProbeReached):
        entrypoint._run_main()
    assert order == ["settings", "seal", "probe"]


def test_torchless_declared_knob_refuses_by_name(torchless) -> None:
    """Every canonical knob is a torch flag; honouring one on a torchless
    worker would silently fork cell identity — refuse, naming the knob."""
    from gen_worker import settings_authority as sa

    with pytest.raises(sa.SettingsImpositionError, match="cudnn_benchmark"):
        sa.impose_torch(overrides={"cudnn_benchmark": "False"})
    with pytest.raises(sa.SettingsImpositionError, match="TORCHLESS"):
        sa.impose_torch(overrides={"cudnn_benchmark": "False"})
    # The torchless seal itself still stands: absence is a keyable fact.
    cfg = env_seal.effective_config()
    assert cfg.get("torch") == "absent"
    assert "cuda_matmul_allow_tf32" not in cfg
