"""pgw#1078: a WORKER-loaded slot that only becomes compile-capable during
`setup()` must end the boot with an INSTALLED compile target.

The live failure (ie#632, minimax-h3 0.4.2 on the master stack): the endpoint
declares `models={"pipeline": Slot(MiniMaxH3ModularPipeline)}` — class
annotated, so the executor materializes it — but `ModularPipeline` hydrates its
weight-bearing components lazily. At injection `pipeline.transformer is None`,
`has_compile_target` is False, the automatic branch skips it and the arm
declines `no_compile_target`. setup() then hydrates and calls
`gen_worker.arm_compile(pipeline)`, which ARMS.

The executor attributed that scope-armed pipeline to `self_loaded_slots` — the
str/Path kwargs — which is EMPTY for a class-annotated slot. So
`_install_compile_targets` saw a candidate owning no slots, computed no
bindings, failed `bindings_valid`, and omitted the target. Nothing was left to
bind the guard to, nothing enabled the hot-swap router, and every request
reported `lane=…+eager  fallback_reason=no_compile_target` while the pipeline
was in fact armed. 257.7 s served against a 131.7 s measured compiled wall.

REVERT-TURNS-RED: both assertions below fail on the pre-fix executor —
`rec.compile_targets` is empty and `rec.eager_posture` reads
`no_compile_target`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import msgspec

import gen_worker
import gen_worker.executor as executor_mod
from gen_worker import Compile, Resources
from gen_worker import compile_cache as cc
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.executor import Executor, ModelStore
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

FAMILY = "minimax-h3"


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    ok: bool = True


class _LazyPipe:
    """A worker-loaded pipeline whose compile target appears only after the
    endpoint hydrates it inside setup()."""

    hydrated = False

    @classmethod
    def from_pretrained(cls, path, **kw):
        return cls()

    def hydrate(self) -> None:
        self.hydrated = True


def _executor(spec: EndpointSpec, tmp_path: Path, sent: list, monkeypatch) -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    store = ModelStore(_send, cache_dir=tmp_path / "cas", vram_budget_bytes=4 << 30)

    async def _fake_ensure_local(ref, **kwargs) -> Path:
        return tmp_path / "snap"

    monkeypatch.setattr(executor_mod, "ensure_local", _fake_ensure_local)
    return Executor([spec], _send, store=store)


def test_a_lazily_hydrated_worker_loaded_slot_installs_its_compile_target(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        cc, "has_compile_target",
        lambda pipeline, cfg: bool(getattr(pipeline, "hydrated", False)))
    monkeypatch.setattr(cc, "apply", lambda *a, **k: True)
    monkeypatch.setattr(cc, "is_compile_armed", lambda pipeline: True)
    # The guard/lane leaves the target install consults — neutral answers so
    # the test addresses OWNERSHIP, which is the defect.
    monkeypatch.setattr(cc, "compile_target_execution_lane_error", lambda *a, **k: "")
    monkeypatch.setattr(cc, "execution_contract_digest", lambda *a, **k: "d")

    class Endpoint:
        def setup(self, pipeline: _LazyPipe) -> None:
            self.pipeline = pipeline
            pipeline.hydrate()
            self.armed = gen_worker.arm_compile(pipeline)

        def warmup(self) -> None:
            """A declared object-level warmup, so the contract has a name to
            attribute — the ownership defect is the subject here, not proof."""

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out()

    spec = EndpointSpec(
        name="generate", method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint, attr_name="run",
        models={"pipeline": Hub("tensorhub/minimax-h3")},
        resources=Resources(gpu=True),
        compile=Compile(family=FAMILY, shapes=((768, 768),), text_len=0),
    )
    sent: list = []

    async def _go() -> None:
        ex = _executor(spec, tmp_path, sent, monkeypatch)
        inst = await ex.ensure_setup(spec, {
            wire_ref(spec.models["pipeline"]): pb.Snapshot(
                digest="blake3:" + "a" * 64),
        })
        assert inst.armed is True, "the in-setup arm must succeed"
        rec = ex._classes[spec.instance_key]

        assert rec.compile_targets, (
            "a scope-armed worker-loaded slot owns ITSELF; with no owning slot "
            "the target is omitted `target_applicability_incomplete` and the "
            "pod serves eager with an armed pipeline"
        )
        target = next(iter(rec.compile_targets.values()))
        assert target.pipeline is inst.pipeline
        assert [slot for slot, _ref, _digest in target.model_bindings] == [
            "pipeline"], "the binding must name the slot the object came from"

        assert rec.eager_posture == "", (
            "the injection-time `no_compile_target` decline ran BEFORE setup "
            "hydrated the pipeline; a later successful arm disproves it, and "
            "first-token-wins across the two scopes is what put the wrong "
            "cause on every request"
        )

    asyncio.run(_go())
