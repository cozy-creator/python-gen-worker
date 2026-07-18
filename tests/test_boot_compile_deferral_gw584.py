"""gw#584: compile-declared endpoints defer from eager boot setup.

The ie#501 run-17 churn: ``worker.py`` starts ``lifecycle.startup()`` and
``transport.run()`` concurrently, so boot-time ``ensure_setup`` could race
ahead of HelloAck's ``apply_model_resolutions`` rebind and reach
``_fetch_compile_snapshot`` with bare authored refs and ``snapshots=None`` —
a silent ``None`` (no cell selected) while materialization later followed the
resolved w8a8 lane, fail-closing ``enable()`` generically. A compile cell,
exactly like a Slot pick (pgw#532), can only arrive via hub delivery, so boot
must defer these functions the same way.

Covered here, over the REAL ``Lifecycle.startup()`` / ``Executor`` machinery
(no mocks of dispatch/setup; only the network download primitive is faked):
  1. boot: a compile-declared function with locally present weights (the
     exact pre-fix eager-setup precondition) is NOT set up at boot — no
     ``ensure_setup``, no snapshot-less ``_fetch_compile_snapshot``; it
     reports loading (awaiting hub delivery, the ie#455-visible state), never
     failed. A ``Slot`` function's pgw#532 deferral holds.
  2. hub delivery (DesiredResidency-equivalent): after the HelloAck rebind,
     ``ensure_desired_instance`` with resolved w8a8 bindings + snapshots
     selects the delivered Forge cell — selection and materialization derive
     from one resolved state.
  3. a w8a8-resolved setup with NO cell in the snapshot map fails LOUD
     (CompiledLaneUnavailableError naming the missing cell), never the
     silent boot-path bail.
  4. the full w8a8 serve chain over deferral: desired-state warm mints the
     compile target, then a RunJob carrying ``required_compile`` for that
     live incarnation executes (the th#868 fence holds end to end).
  5. a PLAIN-lane compile function (no w8a8 fence) deferred at boot is set
     up cold by its first RunJob — a deferred compile function is never
     orphaned.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

import msgspec
import pytest

import gen_worker
from gen_worker import Compile
from gen_worker import compile_cache as cc
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.api.slot import Slot
from gen_worker.config.settings import Settings
from gen_worker.executor import Executor
from gen_worker.families.base import FamilyDefaults, family
from gen_worker.lifecycle import Lifecycle
from gen_worker.models import provision
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


@family("gw584-testfam")
class _Fam(FamilyDefaults):
    steps: int = 7


class _StubPipeline:
    """Slot compat class only — setup() is str-annotated so the executor
    injects local paths (no torch machinery)."""


class _Denoiser:
    def forward(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        return None


class _Pipe:
    def __init__(self) -> None:
        self.transformer = _Denoiser()


class _CompilePipe(_Pipe):
    """Self-loaded w8a8 pipeline stub (the pgw#517 arming-seam shape)."""

    def __init__(self) -> None:
        super().__init__()
        self._cozy_weight_lane = "w8a8"


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    pipeline_path: str = ""


FAMILY = "gw584-fam"
AUTHORED = Hub("acme/qwen-image", tag="prod")           # bare authored binding
AUTHORED_REF = "acme/qwen-image:prod"
RESOLVED_REF = "acme/qwen-image:prod#fp8-w8a8"          # HelloAck ladder pick
CELL_REF = f"_system/family-{FAMILY}#inductor-rtx-4090-torch2.9-w8a8"
PLAIN_CELL_REF = f"_system/family-{FAMILY}#inductor-rtx-4090-torch2.9"


def _compile_spec(setup_calls: List[str]) -> EndpointSpec:
    class Endpoint:
        def setup(self, pipeline: str) -> None:
            self.pipeline_path = pipeline
            self.pipe = _CompilePipe()
            self.armed = gen_worker.arm_compile(self.pipe)
            setup_calls.append(pipeline)

        def warmup(self) -> None:
            signal = getattr(self.pipe, cc._MARKER_ATTR)["failure_signal"]
            with signal["lock"]:
                signal["successful_calls"] += 1
                signal["cache_hits"] += 1

        def generate(self, ctx: Any, payload: _In) -> _Out:
            return _Out(pipeline_path=self.pipeline_path)

    return EndpointSpec(
        name="generate", method=Endpoint.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="generate", models={"pipeline": AUTHORED},
        compile=Compile(shapes=((768, 768),), family=FAMILY),
    )


def _slot_spec(setup_calls: List[str]) -> EndpointSpec:
    class Endpoint:
        def setup(self, pipeline: str) -> None:
            setup_calls.append(pipeline)

        def generate(self, ctx: Any, payload: _In) -> _Out:
            return _Out()

    default = Hub("acme/slotted-default", tag="prod")
    slots = {"pipeline": Slot(
        _StubPipeline, selected_by="model",
        default_checkpoint=default, default_config=_Fam(),
    )}
    return EndpointSpec(
        name="slotted", method=Endpoint.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="generate", models={"pipeline": default}, slots=slots,
        slot_family={"pipeline": "gw584-testfam"},
    )


def _snapshot(digest: str) -> pb.Snapshot:
    return pb.Snapshot(digest=digest, files=[pb.SnapshotFile(
        path="model.safetensors", size_bytes=5, blake3="cd" * 32,
        url="http://r2.invalid/presigned")])


def _cell_artifact(tmp_path: Path) -> Path:
    """A real packed Forge cell tarball (what find_artifact must discover)."""
    cap = tmp_path / "cap"
    (cap / "inductor" / "g").mkdir(parents=True)
    (cap / "inductor" / "g" / "code.py").write_text("x")
    (cap / "triton").mkdir()
    cfg = Compile(shapes=((768, 768),), family=FAMILY)
    signature, weight_contract = cc.execution_contract(_Pipe(), cfg)
    meta = cc.artifact_metadata(
        family=FAMILY, shapes=cfg.shapes, targets=cfg.targets,
        graph_signature=signature, weight_contract=weight_contract,
    )
    out = tmp_path / "minted"
    out.mkdir(exist_ok=True)
    return cc.pack(cap, out / "inductor-rtx-4090-torch2.9-w8a8.tar.gz", meta)


def _harness(tmp_path: Path, monkeypatch, specs: List[EndpointSpec]):
    """Real Executor over the real ModelStore/setup/selection machinery; only
    the network download primitive and the torch.compile leaf
    (``enable_compiled``) are faked."""
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor(specs, _send)
    artifact = _cell_artifact(tmp_path)

    async def _fake_download(ref: str, **kwargs: Any) -> Path:
        p = tmp_path / ref.replace("/", "_").replace(":", "_").replace("#", "_")
        p.mkdir(parents=True, exist_ok=True)
        if ref.startswith("_system/"):
            shutil.copy(artifact, p / artifact.name)
        return p

    import gen_worker.executor as ex_mod
    monkeypatch.setattr(ex_mod, "ensure_local", _fake_download)

    enables: List[Tuple[Any, Optional[Path]]] = []

    def _fake_enable(pipe: Any, _cfg: Any, _cache_dir: Any, artifact_path: Any) -> bool:
        enables.append((pipe, artifact_path))
        setattr(pipe, cc._MARKER_ATTR, {
            "failure_signal": {
                "callback": None,
                "lock": threading.Lock(),
                "successful_calls": 0,
                "cache_hits": 0,
                "cache_misses": 0,
            },
            "originals": [],
            "regional_mods": [],
        })
        return True

    monkeypatch.setattr(provision, "enable_compiled", _fake_enable)
    return ex, sent, enables


def _spy_fetch(ex: Executor, monkeypatch) -> List[Tuple[Optional[dict], Any]]:
    calls: List[Tuple[Optional[dict], Any]] = []
    orig = ex._fetch_compile_snapshot

    async def _spy(spec: EndpointSpec, snapshots: Any) -> Any:
        selection = await orig(spec, snapshots)
        calls.append((dict(snapshots) if snapshots else snapshots, selection))
        return selection

    monkeypatch.setattr(ex, "_fetch_compile_snapshot", _spy)
    return calls


def _startup(ex: Executor) -> Lifecycle:
    lc = Lifecycle(Settings(orchestrator_public_addr="localhost:1"), ex)
    lc.hardware = {"gpu_count": 1, "gpu_total_mem": 32 * 1024**3,
                   "gpu_free_mem": 30 * 1024**3, "gpu_sm": "90",
                   "installed_libs": []}
    asyncio.run(lc.startup())
    return lc


def _apply_hello_ack(ex: Executor) -> None:
    ex.apply_model_resolutions({AUTHORED_REF: (RESOLVED_REF, "")})
    assert wire_ref(ex.specs["generate"].models["pipeline"]) == RESOLVED_REF


# ---------------------------------------------------------------------------
# 1. boot: compile-declared function is deferred exactly like a Slot function
# ---------------------------------------------------------------------------


def test_boot_defers_compile_declared_function(tmp_path, monkeypatch, caplog) -> None:
    setup_calls: List[str] = []
    ex, _sent, enables = _harness(
        tmp_path, monkeypatch,
        [_compile_spec(setup_calls), _slot_spec(setup_calls)])
    # Pre-fix eager precondition: the authored weights ARE locally present,
    # so the old `else` branch would have called ensure_setup(snapshots=None).
    local = tmp_path / "seeded"
    local.mkdir()
    ex.store.residency.track_disk(AUTHORED_REF, local)

    fetches = _spy_fetch(ex, monkeypatch)
    ensured: List[str] = []
    orig_setup = ex.ensure_setup

    async def _setup_spy(spec: EndpointSpec, *a: Any, **kw: Any) -> Any:
        ensured.append(spec.name)
        return await orig_setup(spec, *a, **kw)

    monkeypatch.setattr(ex, "ensure_setup", _setup_spy)

    with caplog.at_level(logging.INFO, logger="gen_worker.lifecycle"):
        _startup(ex)

    assert ensured == [], f"boot eagerly set up deferred functions: {ensured}"
    assert fetches == [], "boot reached compile selection with no snapshots"
    assert setup_calls == [] and enables == []
    # Slots stay advertised (per-dispatch serveability); a compile-declared
    # cls function reports loading until hub delivery warms it — the same
    # visible state as the awaiting_hub bucket (ie#455), never failed.
    assert ex.available_functions() == ["slotted"]
    assert ex.loading_functions() == ["generate"]
    assert "generate" not in ex.unavailable and "slotted" not in ex.unavailable
    deferral_logs = [r.message for r in caplog.records if "gw#584" in r.message]
    assert deferral_logs and "generate" in deferral_logs[0]
    assert "slotted" in deferral_logs[0]


# ---------------------------------------------------------------------------
# 2. hub delivery: resolved bindings + snapshots select the delivered cell
# ---------------------------------------------------------------------------


def test_hub_delivery_selects_delivered_cell(tmp_path, monkeypatch) -> None:
    setup_calls: List[str] = []
    ex, _sent, enables = _harness(tmp_path, monkeypatch,
                                  [_compile_spec(setup_calls)])
    _startup(ex)
    _apply_hello_ack(ex)
    fetches = _spy_fetch(ex, monkeypatch)

    snapshots = {RESOLVED_REF: _snapshot("aa" * 32),
                 CELL_REF: _snapshot("bb" * 32)}
    desired = pb.DesiredInstance(
        function_name="generate",
        models=[pb.ModelBinding(slot="pipeline", ref=RESOLVED_REF)],
    )
    asyncio.run(ex.ensure_desired_instance(desired, snapshots))

    assert setup_calls and RESOLVED_REF.replace("/", "_").replace(
        ":", "_").replace("#", "_") in setup_calls[0]
    assert len(fetches) == 1
    seen_snapshots, selection = fetches[0]
    assert seen_snapshots is not None and CELL_REF in seen_snapshots
    assert selection is not None and selection.ref == CELL_REF
    assert selection.snapshot_digest == "bb" * 32
    # Materialization armed the SAME selected artifact (one resolved state).
    assert len(enables) == 1 and enables[0][1] == selection.path
    (target,) = ex.compile_targets()
    assert target.active_compile_ref == CELL_REF
    assert target.pipeline_weight_lane == "w8a8"
    rec = ex._classes[ex.specs["generate"].instance_key]
    assert rec.ready
    assert rec.held_refs == [RESOLVED_REF]
    assert "generate" in ex.available_functions()


# ---------------------------------------------------------------------------
# 3. w8a8 without a delivered cell fails LOUD, never the silent boot bail
# ---------------------------------------------------------------------------


def test_w8a8_setup_without_cell_fails_loud(tmp_path, monkeypatch) -> None:
    setup_calls: List[str] = []
    ex, _sent, _enables = _harness(tmp_path, monkeypatch,
                                   [_compile_spec(setup_calls)])
    _startup(ex)
    _apply_hello_ack(ex)

    desired = pb.DesiredInstance(
        function_name="generate",
        models=[pb.ModelBinding(slot="pipeline", ref=RESOLVED_REF)],
    )
    with pytest.raises(cc.CompiledLaneUnavailableError, match="W8A8"):
        asyncio.run(ex.ensure_desired_instance(
            desired, {RESOLVED_REF: _snapshot("aa" * 32)}))
    assert setup_calls == [], "setup must fail before tenant code runs"


# ---------------------------------------------------------------------------
# 4. full w8a8 chain: desired warm mints the target, fenced RunJob executes
# ---------------------------------------------------------------------------


async def _dispatch(ex: Executor, sent: List[pb.WorkerMessage],
                    run: pb.RunJob) -> pb.JobResult:
    await ex.handle_run_job(run)
    job = ex.jobs[(run.request_id, run.attempt)]
    assert job.task is not None
    await job.task
    results = [m.job_result for m in sent
               if m.WhichOneof("msg") == "job_result"
               and m.job_result.request_id == run.request_id]
    assert results, f"no job_result for {run.request_id}"
    return results[-1]


def test_fenced_runjob_serves_after_desired_warm(tmp_path, monkeypatch) -> None:
    setup_calls: List[str] = []
    ex, sent, _enables = _harness(tmp_path, monkeypatch,
                                  [_compile_spec(setup_calls)])
    _startup(ex)
    assert setup_calls == []  # boot deferred
    _apply_hello_ack(ex)

    snapshots = {RESOLVED_REF: _snapshot("aa" * 32),
                 CELL_REF: _snapshot("bb" * 32)}
    desired = pb.DesiredInstance(
        function_name="generate",
        models=[pb.ModelBinding(slot="pipeline", ref=RESOLVED_REF)],
    )
    asyncio.run(ex.ensure_desired_instance(desired, snapshots))
    assert len(setup_calls) == 1
    (target,) = ex.compile_targets()

    held = {b.slot: b.snapshot_digest for b in target.model_bindings}
    run = pb.RunJob(
        request_id="r1", attempt=1, function_name="generate",
        input_payload=msgspec.msgpack.encode(_In(prompt="a cat")),
        models=[pb.ModelBinding(slot="pipeline", ref=RESOLVED_REF)],
        snapshots={RESOLVED_REF: _snapshot(held["pipeline"]),
                   CELL_REF: _snapshot("bb" * 32)},
        required_compile=pb.RequiredCompileExecution(
            target_incarnation_id=target.incarnation_id,
            cell_ref=target.active_compile_ref,
            cell_snapshot_digest=target.active_compile_snapshot_digest,
            contract_digest=target.contract_digest,
        ),
    )
    res = asyncio.run(_dispatch(ex, sent, run))
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    assert len(setup_calls) == 1, "the warmed instance must serve, not re-setup"


# ---------------------------------------------------------------------------
# 5. plain-lane compile function: first RunJob cold-sets-up the deferred fn
# ---------------------------------------------------------------------------


def _plain_compile_spec(setup_calls: List[str]) -> EndpointSpec:
    class Endpoint:
        def setup(self, pipeline: str) -> None:
            self.pipeline_path = pipeline
            self.pipe = _Pipe()
            self.armed = gen_worker.arm_compile(self.pipe)
            setup_calls.append(pipeline)

        def warmup(self) -> None:
            signal = getattr(self.pipe, cc._MARKER_ATTR)["failure_signal"]
            with signal["lock"]:
                signal["successful_calls"] += 1
                signal["cache_hits"] += 1

        def generate(self, ctx: Any, payload: _In) -> _Out:
            return _Out(pipeline_path=self.pipeline_path)

    return EndpointSpec(
        name="generate", method=Endpoint.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="generate", models={"pipeline": AUTHORED},
        compile=Compile(shapes=((768, 768),), family=FAMILY),
    )


def test_plain_lane_runjob_cold_setup_after_deferral(tmp_path, monkeypatch) -> None:
    setup_calls: List[str] = []
    ex, sent, _enables = _harness(tmp_path, monkeypatch,
                                  [_plain_compile_spec(setup_calls)])
    _startup(ex)
    assert setup_calls == []  # boot deferred
    fetches = _spy_fetch(ex, monkeypatch)

    run = pb.RunJob(
        request_id="r1", attempt=1, function_name="generate",
        input_payload=msgspec.msgpack.encode(_In(prompt="a cat")),
        models=[pb.ModelBinding(slot="pipeline", ref=AUTHORED_REF)],
        snapshots={AUTHORED_REF: _snapshot("aa" * 32),
                   PLAIN_CELL_REF: _snapshot("bb" * 32)},
    )
    res = asyncio.run(_dispatch(ex, sent, run))
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    assert len(setup_calls) == 1
    assert fetches and fetches[0][1] is not None
    assert fetches[0][1].ref == PLAIN_CELL_REF
    (target,) = ex.compile_targets()
    assert target.active_compile_ref == PLAIN_CELL_REF
