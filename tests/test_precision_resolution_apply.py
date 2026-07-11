"""th#697 P4: the worker applies hub precision resolutions to its bindings,
and the local (no-orchestrator) ladder rebinding walks the same spec."""

from __future__ import annotations

from typing import List

import msgspec

from gen_worker.api.binding import Hub, wire_ref
from gen_worker.api.decorators import Resources
from gen_worker.executor import Executor
from gen_worker.models.hub_client import WorkerResolvedFlavor, WorkerResolvedRepo
from gen_worker.models.hub_policy import TensorhubWorkerCapabilities
from gen_worker.models.ladder import resolve_local_bindings
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _In(msgspec.Struct):
    prompt: str = ""


class _Fake:
    def setup(self, pipeline) -> None:  # pragma: no cover - never run here
        self.pipeline = pipeline

    def generate(self, ctx, payload: _In) -> dict:  # pragma: no cover
        return {}


def _spec() -> EndpointSpec:
    return EndpointSpec(
        name="generate", method=_Fake.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=_Fake,
        models={"pipeline": Hub("acme/z-image")},
        resources=Resources(gpu=True),
    )


def _executor() -> Executor:
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    return Executor([_spec()], _send)


def test_apply_model_resolutions_rebinds_flavor_and_cast() -> None:
    ex = _executor()
    spec = ex.specs["generate"]
    base_ref = wire_ref(spec.models["pipeline"])
    assert base_ref == "acme/z-image"

    # Stored-flavor pick: the binding folds the resolved flavor.
    ex.apply_model_resolutions({base_ref: ("acme/z-image#svdq-int4-r128", "")})
    assert wire_ref(spec.models["pipeline"]) == "acme/z-image#svdq-int4-r128"
    assert spec.models["pipeline"].storage_dtype == ""

    # Full-replace: a new map with a cast-only pick reverts the flavor.
    ex.apply_model_resolutions({base_ref: ("acme/z-image", "fp8")})
    assert wire_ref(spec.models["pipeline"]) == "acme/z-image"
    assert spec.models["pipeline"].storage_dtype == "fp8"

    # Empty map reverts to the authored binding.
    ex.apply_model_resolutions({})
    assert wire_ref(spec.models["pipeline"]) == base_ref
    assert spec.models["pipeline"].storage_dtype == ""


def test_apply_model_resolutions_rejects_non_roundtrip() -> None:
    ex = _executor()
    spec = ex.specs["generate"]
    # A resolved ref for a DIFFERENT repo can't round-trip through this
    # binding — the declared ref must be kept.
    ex.apply_model_resolutions({"acme/z-image": ("acme/other#fp8", "")})
    assert wire_ref(spec.models["pipeline"]) == "acme/z-image"


def test_hello_ack_shape_applies() -> None:
    ex = _executor()
    ack = pb.HelloAck(
        keep=["acme/z-image#svdq-int4-r128"],
        resolutions=[pb.ModelResolution(
            ref="acme/z-image", resolved_ref="acme/z-image#svdq-int4-r128", cast="",
        )],
    )
    ex.apply_model_resolutions({r.ref: (r.resolved_ref, r.cast) for r in ack.resolutions})
    assert wire_ref(ex.specs["generate"].models["pipeline"]) == "acme/z-image#svdq-int4-r128"


def _caps(sm: int, libs: tuple = ()) -> TensorhubWorkerCapabilities:
    return TensorhubWorkerCapabilities(
        cuda_version="12.8", gpu_sm=sm, torch_version="2.8.0", installed_libs=libs,
    )


def _resolved_repo() -> WorkerResolvedRepo:
    return WorkerResolvedRepo(
        snapshot_digest="d1", files=[], size_bytes=12_200_000_000,
        sibling_flavors=[
            WorkerResolvedFlavor(flavor="", size_bytes=12_200_000_000),
            WorkerResolvedFlavor(flavor="svdq-int4-r128", size_bytes=3_900_000_000),
        ],
    )


def test_resolve_local_bindings_picks_stored_flavor() -> None:
    bindings = {"pipeline": Hub("acme/z-image")}
    out = resolve_local_bindings(
        bindings, caps=_caps(89, ("nunchaku",)), free_vram_gb=23.0,
        resolver=lambda thref: _resolved_repo(),
    )
    assert wire_ref(out["pipeline"]) == "acme/z-image#svdq-int4-r128"


def test_resolve_local_bindings_cast_rung_on_hopper() -> None:
    bindings = {"pipeline": Hub("acme/z-image")}
    out = resolve_local_bindings(
        bindings, caps=_caps(90), free_vram_gb=78.0,
        resolver=lambda thref: _resolved_repo(),
    )
    assert wire_ref(out["pipeline"]) == "acme/z-image"
    assert out["pipeline"].storage_dtype == "fp8"


def test_resolve_local_bindings_never_touches_pinned_or_failing() -> None:
    pinned = Hub("acme/z-image", flavor="fp8")
    hf_style = Hub("acme/z-image", storage_dtype="fp8")

    def _boom(thref):
        raise RuntimeError("offline")

    out = resolve_local_bindings(
        {"a": pinned, "b": hf_style},
        caps=_caps(89, ("nunchaku",)), free_vram_gb=23.0,
        resolver=lambda thref: _resolved_repo(),
    )
    assert out["a"] is pinned and out["b"] is hf_style

    bare = Hub("acme/z-image")
    out = resolve_local_bindings(
        {"c": bare}, caps=_caps(89), free_vram_gb=23.0, resolver=_boom,
    )
    assert out["c"] is bare


def test_apply_model_resolutions_rehomes_the_instance_group() -> None:
    """spec.instance_key is a live property over spec.models — a precision
    rebind moves it, and the executor's instance-group record must move too.
    Regression (found live, ie#382): the stale key made every later
    self._classes[spec.instance_key] a KeyError, crash-looping the hello
    handler ~1s after HelloAck and churning H100 pods on 60s reap cycles."""
    ex = _executor()
    spec = ex.specs["generate"]
    key_declared = spec.instance_key
    assert key_declared in ex._classes
    rec = ex._classes[key_declared]

    ex.apply_model_resolutions({"acme/z-image": ("acme/z-image", "fp8")})
    key_cast = spec.instance_key
    assert key_cast != key_declared
    # the SAME record (with any live instance) now lives under the new key
    assert ex._classes[key_cast] is rec
    assert key_declared not in ex._classes
    assert spec in rec.specs

    # full-replace back to the authored binding re-homes it again
    ex.apply_model_resolutions({})
    assert spec.instance_key == key_declared
    assert ex._classes[key_declared] is rec
    assert key_cast not in ex._classes
    assert rec.specs == [spec]
