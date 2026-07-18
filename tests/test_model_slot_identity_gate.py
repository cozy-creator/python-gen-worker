"""gw#583: worker model-slot identity gate (the ie#518 silence).

Live incident (2026-07-18, ltx-video-2.3 audio-reactive, 3 completed
requests): the hub dispatched ``pipeline=tensorhub/ltx-2.3-distilled`` to a
function whose ``@endpoint`` declared ``pipeline=tensorhub/ltx-2.3-audio-
reactive``, plus an undeclared ``lora`` model param — the worker loaded the
wrong repo and dropped the extra param with zero refusal, warning, or
ModelEvent.

Covered here, over the REAL ``_slot_dispatch_binding``/``_effective_spec``
dispatch machinery (``gen_worker.executor``), never a mock of it:
  1. a FIXED slot (no ``selected_by=``) dispatched a DIFFERENT repo refuses,
     naming the slot and both refs (revert-turns-red on the gate itself).
  2. the SAME repo with a hub-resolved flavor/tag pick still serves.
  3. a ``selected_by=`` catalog slot's pick of a different repo is a
     legitimate explicit surface, not a mismatch — still serves.
  4. a lora overlay riding the correctly-declared repo is not a mismatch.
  5. an undeclared model slot in the dispatch map logs a warning naming the
     slot and does not block the job.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import msgspec

from gen_worker.api.binding import Hub, wire_ref
from gen_worker.api.errors import ModelSlotIdentityError
from gen_worker.api.slot import Slot
from gen_worker.executor import Executor
from gen_worker.families.base import FamilyDefaults, family
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


@family("gw583-testfam")
class _Fam(FamilyDefaults):
    steps: int = 7


class _StubPipeline:
    """Slot compat class only — setup() takes a str-annotated path so the
    executor injects the local materialized path (no torch machinery)."""


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    slot_ref: str
    pipeline_path: str


DECLARED = Hub("acme/declared-repo", tag="prod")
WRONG_REPO = "acme/different-repo:prod"
SAME_REPO_OTHER_FLAVOR = "acme/declared-repo:prod#fp8"
CATALOG_DEFAULT = Hub("acme/catalog-default", tag="prod")
CATALOG_PICK = "acme/catalog-pick:prod"
LORA_REF = "acme/some-lora:prod"


def _fixed_spec(name: str, setup_calls: List[Tuple[str, str]]) -> EndpointSpec:
    class Endpoint:
        def setup(self, pipeline: str) -> None:
            self.pipeline_path = pipeline
            setup_calls.append((name, pipeline))

        def generate(self, ctx: Any, payload: _In) -> _Out:
            resolved = ctx.slots["pipeline"]
            ref = resolved.ref
            return _Out(
                slot_ref=f"{ref.source}:{ref.path}:{ref.tag}#{ref.flavor}",
                pipeline_path=self.pipeline_path,
            )

    slots = {
        "pipeline": Slot(_StubPipeline, default_checkpoint=DECLARED, default_config=_Fam()),
    }
    return EndpointSpec(
        name=name, method=Endpoint.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="generate", models={"pipeline": DECLARED}, slots=slots,
        slot_family={"pipeline": "gw583-testfam"},
    )


def _catalog_spec(name: str, setup_calls: List[Tuple[str, str]]) -> EndpointSpec:
    class Endpoint:
        def setup(self, pipeline: str) -> None:
            self.pipeline_path = pipeline
            setup_calls.append((name, pipeline))

        def generate(self, ctx: Any, payload: _In) -> _Out:
            resolved = ctx.slots["pipeline"]
            ref = resolved.ref
            return _Out(
                slot_ref=f"{ref.source}:{ref.path}:{ref.tag}#{ref.flavor}",
                pipeline_path=self.pipeline_path,
            )

    slots = {
        "pipeline": Slot(
            _StubPipeline, selected_by="model",
            default_checkpoint=CATALOG_DEFAULT, default_config=_Fam(),
        ),
    }
    return EndpointSpec(
        name=name, method=Endpoint.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="generate", models={"pipeline": CATALOG_DEFAULT}, slots=slots,
        slot_family={"pipeline": "gw583-testfam"},
    )


def _harness(tmp_path: Path, monkeypatch, spec: EndpointSpec):
    sent: List[pb.WorkerMessage] = []
    downloads: List[str] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor([spec], _send)

    async def _fake_download(ref: str, **kwargs: Any) -> Path:
        downloads.append(ref)
        p = tmp_path / ref.replace("/", "_").replace(":", "_").replace("#", "_")
        p.mkdir(parents=True, exist_ok=True)
        return p

    import gen_worker.executor as ex_mod
    monkeypatch.setattr(ex_mod, "ensure_local", _fake_download)
    return ex, sent, downloads


def _snapshot(digest: str) -> pb.Snapshot:
    return pb.Snapshot(digest=digest, files=[pb.SnapshotFile(
        path="model.safetensors", size_bytes=5, blake3="cd" * 32,
        url="http://r2.invalid/presigned")])


def _run_job(rid: str, *, models: List[pb.ModelBinding],
             snapshots: Dict[str, pb.Snapshot] = {}) -> pb.RunJob:
    return pb.RunJob(
        request_id=rid, attempt=1, function_name="generate",
        input_payload=msgspec.msgpack.encode(_In(prompt="a cat")),
        models=models, snapshots=snapshots,
    )


async def _dispatch(ex: Executor, sent: List[pb.WorkerMessage], run: pb.RunJob) -> pb.JobResult:
    await ex.handle_run_job(run)
    job = ex.jobs[(run.request_id, run.attempt)]
    assert job.task is not None
    await job.task
    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"
               and m.job_result.request_id == run.request_id]
    assert results, f"no job_result for {run.request_id}"
    return results[-1]


# ---------------------------------------------------------------------------
# 1. fixed slot, different repo dispatched -> refuses naming slot + both refs
# ---------------------------------------------------------------------------


def test_wrong_repo_dispatch_refuses_naming_slot_and_refs(tmp_path, monkeypatch) -> None:
    setup_calls: List[Tuple[str, str]] = []
    spec = _fixed_spec("generate", setup_calls)
    ex, sent, downloads = _harness(tmp_path, monkeypatch, spec)

    import asyncio
    res = asyncio.run(_dispatch(ex, sent, _run_job(
        "r1", models=[pb.ModelBinding(slot="pipeline", ref=WRONG_REPO)],
        snapshots={WRONG_REPO: _snapshot("aa" * 32)},
    )))

    assert res.status != pb.JOB_STATUS_OK
    assert "pipeline" in res.safe_message
    assert "acme/declared-repo" in res.safe_message
    assert "acme/different-repo" in res.safe_message
    assert setup_calls == [], "a refused dispatch must never reach setup()"
    assert downloads == [], "a refused dispatch must never download"
    assert "generate" not in ex.unavailable, (
        "a per-request refusal must not disable the function"
    )


def test_slot_identity_error_carries_typed_fields() -> None:
    exc = ModelSlotIdentityError(
        "generate", "pipeline",
        declared_ref="acme/declared-repo:prod",
        dispatched_ref="acme/different-repo:prod",
    )
    assert exc.slot == "pipeline"
    assert exc.declared_ref == "acme/declared-repo:prod"
    assert exc.dispatched_ref == "acme/different-repo:prod"
    assert "pipeline" in str(exc)
    assert "acme/declared-repo:prod" in str(exc)
    assert "acme/different-repo:prod" in str(exc)


# ---------------------------------------------------------------------------
# 2. same repo, hub-resolved flavor pick -> not a mismatch, serves
# ---------------------------------------------------------------------------


def test_same_repo_flavor_resolution_serves(tmp_path, monkeypatch) -> None:
    setup_calls: List[Tuple[str, str]] = []
    spec = _fixed_spec("generate", setup_calls)
    ex, sent, downloads = _harness(tmp_path, monkeypatch, spec)

    import asyncio
    res = asyncio.run(_dispatch(ex, sent, _run_job(
        "r1", models=[pb.ModelBinding(slot="pipeline", ref=SAME_REPO_OTHER_FLAVOR)],
        snapshots={SAME_REPO_OTHER_FLAVOR: _snapshot("aa" * 32)},
    )))

    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    out = msgspec.msgpack.decode(res.inline, type=_Out)
    assert out.slot_ref == "tensorhub:acme/declared-repo:prod#fp8"
    assert setup_calls == [("generate", out.pipeline_path)]


# ---------------------------------------------------------------------------
# 3. selected_by= catalog slot's different-repo pick is a legitimate surface
# ---------------------------------------------------------------------------


def test_catalog_slot_pick_is_not_a_mismatch(tmp_path, monkeypatch) -> None:
    setup_calls: List[Tuple[str, str]] = []
    spec = _catalog_spec("generate", setup_calls)
    ex, sent, downloads = _harness(tmp_path, monkeypatch, spec)

    import asyncio
    res = asyncio.run(_dispatch(ex, sent, _run_job(
        "r1", models=[pb.ModelBinding(slot="pipeline", ref=CATALOG_PICK)],
        snapshots={CATALOG_PICK: _snapshot("aa" * 32)},
    )))

    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    out = msgspec.msgpack.decode(res.inline, type=_Out)
    assert out.slot_ref == "tensorhub:acme/catalog-pick:prod#"
    assert setup_calls == [("generate", out.pipeline_path)]


# ---------------------------------------------------------------------------
# 4. a lora overlay riding the declared repo is not a mismatch
# ---------------------------------------------------------------------------


def test_lora_overlay_on_declared_repo_is_not_a_mismatch(tmp_path, monkeypatch) -> None:
    # Full lora weight application (real pipeline injection) is exercised by
    # tests/test_executor_lora.py; this exercises the REAL identity-gate
    # entry point (executor._effective_spec) directly, over a ModelBinding
    # that rides a lora overlay on the correctly-declared repo — a lora must
    # never be mistaken for a repo mismatch (they're independent proto
    # fields: ModelBinding.ref vs ModelBinding.loras).
    setup_calls: List[Tuple[str, str]] = []
    spec = _fixed_spec("generate", setup_calls)
    ex, _sent, _downloads = _harness(tmp_path, monkeypatch, spec)

    run = _run_job("r1", models=[pb.ModelBinding(
        slot="pipeline", ref="acme/declared-repo:prod",
        loras=[pb.LoraOverlay(ref=LORA_REF, weight=0.8)],
    )])
    effective = ex._effective_spec(spec, run)
    assert wire_ref(effective.models["pipeline"]) == "acme/declared-repo:prod"


# ---------------------------------------------------------------------------
# 5. undeclared model slot in the dispatch map warns, never blocks the job
# ---------------------------------------------------------------------------


def test_undeclared_model_slot_warns_and_serves(tmp_path, monkeypatch, caplog) -> None:
    setup_calls: List[Tuple[str, str]] = []
    spec = _fixed_spec("generate", setup_calls)
    ex, sent, downloads = _harness(tmp_path, monkeypatch, spec)

    import asyncio
    with caplog.at_level(logging.WARNING, logger="gen_worker.executor"):
        res = asyncio.run(_dispatch(ex, sent, _run_job(
            "r1", models=[
                pb.ModelBinding(slot="pipeline", ref="acme/declared-repo:prod"),
                pb.ModelBinding(slot="lora", ref=LORA_REF),
            ],
            snapshots={"acme/declared-repo:prod": _snapshot("aa" * 32)},
        )))

    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                and "UNDECLARED_MODEL_SLOT" in r.getMessage()]
    assert warnings, f"no undeclared-slot warning logged: {caplog.records}"
    assert any("lora" in r.getMessage() for r in warnings)
    assert setup_calls, "the undeclared param must not block the declared slot's setup"
