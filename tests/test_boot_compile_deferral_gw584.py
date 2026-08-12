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
from gen_worker.families.base import GenerationDefaults, family
from gen_worker.lifecycle import Lifecycle
from gen_worker.models import provision
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec
from gen_worker.cell_adopt import AdoptOutcome


@family("gw584-testfam")
class _Fam(GenerationDefaults):
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
AUTHORED_REF = "acme/qwen-image"
#: pgw#1148/§1.32(d): a HelloAck pick can no longer RE-ADDRESS a binding —
#: the `#flavor` that used to is deleted and th#1803's digest pin resolves
#: the same address. The pick's carried facts are the cast and the lane, and
#: the w8a8 lane below is stated the way production states it.
RESOLVED_REF = AUTHORED_REF
RESOLVED_LANE = "fp8-w8a8-dynamic+compiled"
CELL_REF = f"root/family-{FAMILY}#inductor-rtx-4090-torch2.9-w8a8"
PLAIN_CELL_REF = f"root/family-{FAMILY}#inductor-rtx-4090-torch2.9"


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
    # SDK v2: Slot(default_config=...) is deleted — the config schema is the
    # handler's derived defaults type (ctx: RequestContext[_Fam]), carried on
    # the spec as defaults_type.
    slots = {"pipeline": Slot(
        _StubPipeline, selected_by="model", default_checkpoint=default,
    )}
    return EndpointSpec(
        name="slotted", method=Endpoint.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="generate", models={"pipeline": default}, slots=slots,
        slot_family={"pipeline": "gw584-testfam"}, defaults_type=_Fam,
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
        if ref.startswith("root/"):
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
        return AdoptOutcome.hit()

    monkeypatch.setattr(provision, "enable_compiled", _fake_enable)
    return ex, sent, enables


def _mint_enable(cell_ref: str, digest: str, artifact_path: Path):
    """pgw#904: an advertised identity now comes from the worker's OWN mint
    (delivered-cell selection is deleted); stamp it the way the live fleet
    policy does — an ArmOutcome carrying the finalized SelfMint."""
    from gen_worker import fleet_cells

    def _enable(pipe, cfg, cache_dir=None, artifact=None, publisher=None,
                delegate=None, delivered_ref="", delivered_digest=""):
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
        return fleet_cells.ArmOutcome(armed=True, self_mint=fleet_cells.SelfMint(
            family=FAMILY, cell_key=cell_ref.rsplit("#", 1)[-1], ref=cell_ref,
            snapshot_digest=digest, artifact=artifact_path))

    return _enable


def _startup(ex: Executor) -> Lifecycle:
    lc = Lifecycle(Settings(orchestrator_public_addr="localhost:1"), ex)
    lc.hardware = {"gpu_count": 1, "gpu_total_mem": 32 * 1024**3,
                   "gpu_free_mem": 30 * 1024**3, "gpu_sm": "90",
                   "installed_libs": []}
    asyncio.run(lc.startup())
    return lc


def _apply_hello_ack(ex: Executor) -> None:
    ex.apply_model_resolutions(
        {AUTHORED_REF: (RESOLVED_REF, "", RESOLVED_LANE)})
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

    ensured: List[str] = []
    orig_setup = ex.ensure_setup

    async def _setup_spy(spec: EndpointSpec, *a: Any, **kw: Any) -> Any:
        ensured.append(spec.name)
        return await orig_setup(spec, *a, **kw)

    monkeypatch.setattr(ex, "ensure_setup", _setup_spy)

    with caplog.at_level(logging.INFO, logger="gen_worker.lifecycle"):
        _startup(ex)

    assert ensured == [], f"boot eagerly set up deferred functions: {ensured}"
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
# 3. w8a8 without a delivered cell still fails LOUD when the arm is unproven
# ---------------------------------------------------------------------------


def test_w8a8_setup_without_cell_degrades_to_explicit_eager(
    tmp_path, monkeypatch,
) -> None:
    """gw#587 moved the refusal to the proof gate; pgw#672 changed its
    POSTURE: an armed compile object that cannot PROVE itself on the warmup
    no longer kills the boot — it degrades to EXPLICIT eager (the tier is
    on the wire, never silent), the function stays dispatchable, and no
    compiled identity is ever advertised."""
    setup_calls: List[str] = []
    ex, _sent, _enables = _harness(tmp_path, monkeypatch,
                                   [_compile_spec(setup_calls)])
    _startup(ex)
    _apply_hello_ack(ex)

    desired = pb.DesiredInstance(
        function_name="generate",
        models=[pb.ModelBinding(slot="pipeline", ref=RESOLVED_REF)],
    )
    asyncio.run(ex.ensure_desired_instance(
        desired, {RESOLVED_REF: _snapshot("aa" * 32)}))
    # The load/setup ran (the self-mint's precondition); the unprovable
    # lane degraded at the proof gate: dispatchable at eager tier, nothing
    # advertised as compiled.
    assert len(setup_calls) == 1
    assert "generate" in ex.available_functions()
    assert ex.serving_tiers()["generate"] == "eager"
    assert all(
        not t.active_compile_ref for t in ex.compile_targets()
    )


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
    from gen_worker import fleet_cells as fleet_cells_mod
    monkeypatch.setattr(
        fleet_cells_mod, "enable_compiled",
        _mint_enable(CELL_REF, "bb" * 32, tmp_path / "cell.tar.gz"))
    _startup(ex)
    assert setup_calls == []  # boot deferred
    _apply_hello_ack(ex)

    snapshots = {RESOLVED_REF: _snapshot("aa" * 32)}
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


def test_plain_execution_lane_runjob_cold_setup_after_deferral(tmp_path, monkeypatch) -> None:
    setup_calls: List[str] = []
    ex, sent, _enables = _harness(tmp_path, monkeypatch,
                                  [_plain_compile_spec(setup_calls)])
    from gen_worker import fleet_cells as fleet_cells_mod
    monkeypatch.setattr(
        fleet_cells_mod, "enable_compiled",
        _mint_enable(PLAIN_CELL_REF, "bb" * 32, tmp_path / "cell.tar.gz"))
    _startup(ex)
    assert setup_calls == []  # boot deferred

    run = pb.RunJob(
        request_id="r1", attempt=1, function_name="generate",
        input_payload=msgspec.msgpack.encode(_In(prompt="a cat")),
        models=[pb.ModelBinding(slot="pipeline", ref=AUTHORED_REF)],
        snapshots={AUTHORED_REF: _snapshot("aa" * 32)},
    )
    res = asyncio.run(_dispatch(ex, sent, run))
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    assert len(setup_calls) == 1
    (target,) = ex.compile_targets()
    assert target.active_compile_ref == PLAIN_CELL_REF


def test_conversion_kind_never_reports_loading(tmp_path, monkeypatch):
    """ie#522 (live, 2026-07-21): a conversion-kind function that has never
    been dispatched must read as available, never "loading". _warmup_plan
    already never schedules non-inference kinds for boot warmup (spec.kind
    != "inference", see the compile-target setup path); available_functions
    / loading_functions must agree, or a declared-but-idle conversion
    function (one release can bundle many; only a few are ever dispatched)
    sits in loading_functions() forever. The hub's th#965 layer-3 stall
    watchdog takes that at face value and, after 10 minutes of "no open
    activity" on a function nothing was ever going to warm, kills the WHOLE
    pod — including an unrelated, actively-progressing conversion job on the
    same worker. Reproduced live: a wan-2.2 conversion release bundling
    z-image-w8a8-quantization alongside clone-huggingface got its pod killed
    at 10m15s three times running, on three different hosts, mid a
    legitimate multi-GB clone-huggingface transfer."""
    class Endpoint:
        def convert(self, ctx: Any, payload: _In) -> _Out:
            return _Out()

    spec = EndpointSpec(
        name="z-image-w8a8-quantization", method=Endpoint.convert,
        kind="conversion", payload_type=_In, output_mode="single",
        cls=Endpoint, attr_name="convert", models={"pipeline": AUTHORED},
    )
    ex, _sent, _enables = _harness(tmp_path, monkeypatch, [spec])
    assert ex.available_functions() == ["z-image-w8a8-quantization"]
    assert ex.loading_functions() == []
