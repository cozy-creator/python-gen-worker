"""pgw#1372: the ship-code-as-is serving layer — eager-first, ctx.compile adopt.

Integration, no mocks: the fixture endpoint under
``tests/fixtures/serving_v2_endpoint`` is shaped exactly like the
serverless-endpoints sdxl ``main_v2.py`` under the pgw#1382 split
(``SdxlModel(Model[SDXL], lanes=…)``, one stateless ``@entrypoint``,
imperative ``self.pipe.unet = ctx.compile(self.pipe.unet)`` marking). It
boots from a CONFIG-ONLY checkpoint (fake weights), serves real requests
end-to-end on CPU, and the adopt path runs real publish-time discovery
output through a real ``LocalGraphStore`` — only the artifact loader is a
stub, because bytes-to-callable is the AOTInductor runtime's job on the
target GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import msgspec
import pytest
import torch

from gen_worker import boot_stages
from gen_worker._vendor.tensorfs import LocalCAS
from gen_worker._vendor.torchcg import EnvironmentMismatch
from gen_worker._vendor.torchcg.discovery import discover_lane
from gen_worker._vendor.torchcg.document import GraphSetDocument
from gen_worker._vendor.torchcg.graph_identity import EnvIdentity, closure_hash
from gen_worker._vendor.torchcg.requirements import RequirementsManifest
from gen_worker._vendor.torchcg.store import LocalGraphStore, StoreError
from gen_worker.serving import (
    Adapter,
    DeployBinding,
    EndpointHost,
    EndpointLoadError,
    ServeDispatchError,
    load_endpoint,
)
from gen_worker.serving.hub_store import HubGraphStore

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_v2_endpoint"
LANE = "sdxl.diffusers-bf16@1"
SM = "sm_89"
INSTALLED = {"torch": torch.__version__}
CLOSURE = closure_hash(INSTALLED)
ENV = EnvIdentity(closure=CLOSURE, sm=SM)

#: Hub per-checkpoint overrides — mutable deploy state, decoded by
#: ``LoadContext.defaults()`` against ``SDXL.Defaults``. Small step knob so
#: the CPU loop stays tiny.
OVERRIDES: dict[str, Any] = {
    "steps": {"default": 2, "lo": 1, "hi": 8, "field": "num_inference_steps"},
    "guidance": {"default": 6.0, "lo": 1.5, "hi": 8.0, "field": "guidance_scale"},
}


@pytest.fixture(autouse=True)
def clean_boot_stages():
    boot_stages.reset_for_tests()
    yield
    boot_stages.reset_for_tests()


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    """Config-only checkpoint: fake weights, real config — weights-free boot."""
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"seed": 7, "scheduler": {"prediction_type": "epsilon"}})
    )
    return root


@pytest.fixture()
def binding(checkpoint: Path) -> DeployBinding:
    return DeployBinding(
        checkpoint_ref="ckpt:tiny@1",
        checkpoint_dir=checkpoint,
        defaults=dict(OVERRIDES),
    )


def fresh_host(binding: DeployBinding, tmp_path: Path) -> EndpointHost:
    loaded = load_endpoint(FIXTURE_DIR)
    return EndpointHost(
        loaded, binding, lane_contract=LANE, output_dir=tmp_path / "outputs")


@pytest.fixture()
def host(binding: DeployBinding, tmp_path: Path) -> EndpointHost:
    booted = fresh_host(binding, tmp_path)
    booted.setup()
    return booted


def fixture_model(host: EndpointHost) -> Any:
    (instance,) = host.instances.values()
    return instance.model


# --- the eager path: standalone, first --------------------------------------


def test_eager_boot_serves_a_request_end_to_end(host: EndpointHost, tmp_path: Path) -> None:
    # load() ran the author's ctx.load against the checkpoint tree under the
    # active lane's dtype, and ctx.defaults() typed the hub overrides.
    model = fixture_model(host)
    assert model.pipe.dtype is torch.float32
    assert model.defaults.steps.default == 2  # hub override beat the platform value
    assert model.defaults.cfg is True  # platform default survived
    # ctx.compile with no adoption source is a transparent pass-through: the
    # marked module carries no swapped forward.
    assert "forward" not in model.pipe.unet.__dict__

    ctx = host.make_context("req-1")
    out = host.dispatch(
        "generate",
        {"prompt": "hello", "aspect_ratio": "1:1", "guidance_scale": 12.0},
        request_id="req-1",
        ctx=ctx,
    )
    assert out.model_used == "ckpt:tiny@1"
    saved = tmp_path / "outputs" / out.image.ref
    assert saved.is_file() and saved.stat().st_size > 0

    # The Knob clamp recorded the caller-visible adjustment (12 -> 8).
    rows = [row for row in ctx.adjustments if row["field"] == "guidance_scale"]
    assert rows and rows[0]["requested"] == "12.0" and rows[0]["applied"] == "8.0"

    # The boot recorded a model_load span for the author's load, and no
    # adopt_pull span — nothing was offered to adopt.
    stages = [span.stage.value for span in boot_stages.recorded()]
    assert "model_load" in stages
    assert "adopt_pull" not in stages


def test_dispatch_refusals_are_typed_before_author_code_runs(host: EndpointHost) -> None:
    with pytest.raises(ServeDispatchError, match="no function 'enhance'"):
        host.dispatch("enhance", {"prompt": "x"}, request_id="r")
    with pytest.raises(msgspec.ValidationError):
        host.dispatch("generate", {"prompt": "x", "num_inference_steps": 999}, request_id="r")
    with pytest.raises(msgspec.ValidationError):
        host.dispatch("generate", {"prompt": "x", "unknown_field": 1}, request_id="r")


def test_the_deployment_decides_the_mode(
    host: EndpointHost, binding: DeployBinding, tmp_path: Path
) -> None:
    """Paul's merge ruling: one entrypoint; adapter bound -> turbo branch."""
    # A bound distillation adapter serves turbo and stamps the recipe name.
    from gen_worker.models import SDXL

    host.rebind(
        DeployBinding(
            checkpoint_ref=binding.checkpoint_ref,
            checkpoint_dir=binding.checkpoint_dir,
            defaults=binding.defaults,
            adapter=Adapter(
                name="lightning-4step", path=tmp_path / "lora",
                defaults=SDXL.Lora.Defaults(),
            ),
        )
    )
    out = host.dispatch("generate", {"prompt": "x"}, request_id="r2")
    assert out.model_used == "ckpt:tiny@1+lightning-4step"
    # Explicit guidance/negatives on a cfg-free serving are IGNORED
    # caller-visibly — ctx.warn rows in the response envelope, never a
    # silent drop and never an aborted request (Paul's warn ruling).
    ctx = host.make_context("r3")
    out = host.dispatch(
        "generate",
        {"prompt": "x", "guidance_scale": 7.0, "negative_prompt": "bad"},
        request_id="r3", ctx=ctx,
    )
    assert out.model_used == "ckpt:tiny@1+lightning-4step"
    assert [w for w in ctx.warnings if "guidance_scale ignored" in w]
    assert [w for w in ctx.warnings if "negative_prompt ignored" in w]


def test_loader_states_the_surface_and_refuses_typed(tmp_path: Path) -> None:
    loaded = load_endpoint(FIXTURE_DIR)
    assert loaded.module_name == "serving_v2_fixture.main"
    assert sorted(loaded.entrypoints) == ["generate"]
    (model_cls,) = loaded.models
    lane = loaded.lane(model_cls, LANE)
    assert lane.contract == LANE
    assert lane.dtype is torch.float32
    # Two declared lanes: the active one is the deploy's pick, never a default.
    with pytest.raises(EndpointLoadError, match="the active lane must be named"):
        loaded.lane(model_cls)
    with pytest.raises(EndpointLoadError, match="no lane 'other.fp8@1'"):
        loaded.lane(model_cls, "other.fp8@1")
    with pytest.raises(EndpointLoadError, match="no endpoint.toml"):
        load_endpoint(tmp_path)


def test_unload_runs_through_drain_then_call(host: EndpointHost) -> None:
    (model_cls,) = host.loaded.models
    assert model_cls in host.instances
    host.teardown()
    assert host.instances == {}


# --- the adopt path: publish-time discovery, store pull, ctx.compile --------


def publish_document(host: EndpointHost) -> GraphSetDocument:
    """The publish-time derive, run for real: discovery hooks the marked
    module on the author's live pipeline and drives the author's own
    entrypoint with schema-enumerated payloads (both aspect-ratio buckets)."""
    from serving_v2_fixture.main import AspectRatio

    model = fixture_model(host)

    def drive() -> None:
        for index, ratio in enumerate(AspectRatio):
            ctx = host.make_context(f"trace-{index}", is_trace=True)
            host.dispatch(
                "generate", {"prompt": "trace", "aspect_ratio": str(ratio)},
                request_id=f"trace-{index}", ctx=ctx,
            )

    lane_graphs = discover_lane(LANE, ("unet",), {"unet": model.pipe.unet}, drive)
    return GraphSetDocument(closure=CLOSURE, lanes=(lane_graphs,))


def manifest() -> RequirementsManifest:
    return RequirementsManifest(include_set=(("torch", torch.__version__),), sm_compiled=SM)


def counting_loader(calls: "list[str]") -> "Callable[[Path, Any], Callable[..., Any]]":
    def load(path: Path, record: Any) -> "Callable[..., Any]":
        def compiled(sample: torch.Tensor) -> torch.Tensor:
            calls.append(record.graph)
            return sample

        return compiled

    return load


def test_adopt_first_boot_swaps_via_ctx_compile_and_hands_ordered_holes(
    host: EndpointHost, binding: DeployBinding, tmp_path: Path
) -> None:
    document = publish_document(host)
    lane_graphs = document.lanes[0]
    assert len(lane_graphs.graphs) == 2  # two buckets -> two graph classes
    hit, hole = lane_graphs.graphs
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    artifact = tmp_path / "minted.so"
    artifact.write_bytes(b"compiled")
    store.publish_artifact(hit.graph, ENV, artifact, manifest())

    calls: list = []
    adopted_host = fresh_host(binding, tmp_path)
    adopted_host.setup(
        store=store, document=document, sm=SM,
        loader=counting_loader(calls),
        artifacts_dir=tmp_path / "adopted",
        installed=INSTALLED,
    )

    # THE handoff for pgw#1371: ordered holes carrying full GraphRecords.
    assert [h.record.graph for h in adopted_host.holes] == [hole.graph]
    assert adopted_host.holes[0].reason == "miss"
    assert adopted_host.holes[0].record.ingress == hole.ingress

    # The armed bucket serves THROUGH the ctx.compile swap (module-forward
    # verified called); the hole bucket stays on the author's eager forward.
    hit_shape = tuple(d for d in hit.ingress.inputs[0].shape)
    hit_ratio = "1:1" if hit_shape == (16, 16) else "3:4"
    hole_ratio = "3:4" if hit_ratio == "1:1" else "1:1"
    adopted_host.dispatch(
        "generate", {"prompt": "x", "aspect_ratio": hit_ratio}, request_id="r1")
    assert calls and set(calls) == {hit.graph}
    swapped = len(calls)
    adopted_host.dispatch(
        "generate", {"prompt": "x", "aspect_ratio": hole_ratio}, request_id="r2")
    assert len(calls) == swapped  # the hole ran eager

    # A late mint arms without a reboot and leaves the hole list empty.
    minted = tmp_path / "late.so"
    minted.write_bytes(b"late")
    adopted_host.adoption.arm(hole, minted)
    adopted_host.dispatch(
        "generate", {"prompt": "x", "aspect_ratio": hole_ratio}, request_id="r3")
    assert hole.graph in calls
    assert adopted_host.holes == ()

    # Telemetry: the adopt_pull span replaced the keyset span in this flow.
    spans = [s for s in boot_stages.recorded() if s.stage.value == "adopt_pull"]
    assert len(spans) == 1
    assert spans[0].attrs["graphs_from"] == "release"
    assert spans[0].attrs["artifact_from_store"] == "1"
    assert spans[0].attrs["artifact_from_eager"] == "1"
    assert not [s for s in boot_stages.recorded() if s.stage.value == "keyset"]


def test_exact_env_mismatch_refuses_loudly_before_author_code(
    host: EndpointHost, binding: DeployBinding, tmp_path: Path
) -> None:
    document = publish_document(host)
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    refused = fresh_host(binding, tmp_path)
    with pytest.raises(EnvironmentMismatch, match="build-system bug"):
        refused.setup(
            store=store, document=document, sm=SM,
            loader=counting_loader([]),
            artifacts_dir=tmp_path / "adopted",
            installed={"torch": "0.0.0-divergent"},
        )
    # The audit fired BEFORE any author model was even instantiated.
    assert refused.instances == {}
    spans = [s for s in boot_stages.recorded() if s.stage.value == "adopt_pull"]
    assert spans and spans[-1].attrs["refusal"] == "environment_mismatch"


def test_eager_permanent_metadata_is_a_clean_noop(
    binding: DeployBinding, tmp_path: Path
) -> None:
    eager = GraphSetDocument(closure=CLOSURE, lanes=())
    booted = fresh_host(binding, tmp_path)
    booted.setup(document=eager, sm=SM)
    assert booted.adoption is None
    assert booted.holes == ()
    spans = [s for s in boot_stages.recorded() if s.stage.value == "adopt_pull"]
    assert [s.attrs["graphs_from"] for s in spans] == ["eager_permanent"]
    # The endpoint still serves — the eager bridge is unconditional.
    out = booted.dispatch("generate", {"prompt": "still serving"}, request_id="r")
    assert out.model_used == "ckpt:tiny@1"


def test_a_store_less_boot_still_forms_the_full_mint_worklist(
    host: EndpointHost, binding: DeployBinding, tmp_path: Path
) -> None:
    document = publish_document(host)
    booted = fresh_host(binding, tmp_path)
    booted.setup(
        document=document, sm=SM, loader=counting_loader([]),
        artifacts_dir=tmp_path / "adopted", installed=INSTALLED,
    )
    # Metadata known, artifacts unreachable: everything is stated mint work.
    assert [h.reason for h in booted.holes] == ["miss", "miss"]
    out = booted.dispatch("generate", {"prompt": "eager"}, request_id="r")
    assert out.model_used == "ckpt:tiny@1"


# --- the hub-backed store: th#2133's answer shape, transport stubbed --------


class StubTransport:
    """The th#2133 route answer, canned; the wiring lands with the route."""

    def __init__(self, answer: Mapping[str, Any], blobs: Mapping[str, bytes]) -> None:
        self.answer = answer
        self.blobs = dict(blobs)
        self.asks = 0

    def release_compiled_graphs(
        self, release_id: str, lane: str, sm: str
    ) -> Mapping[str, Any]:
        self.asks += 1
        return self.answer

    def fetch_blob(self, url: str) -> bytes:
        return self.blobs[url]


def test_hub_store_partial_hit_verifies_digests_and_misses_clean(
    host: EndpointHost, binding: DeployBinding, tmp_path: Path
) -> None:
    import hashlib

    document = publish_document(host)
    hit, hole = document.lanes[0].graphs
    payload = b"presigned-compiled-bytes"
    answer = {
        "document": document.as_dict(),
        "artifacts": {
            hit.graph: {
                "digest": hashlib.sha256(payload).hexdigest(),
                "url": "https://presigned.example/hit",
                "manifest": manifest().as_dict(),
            }
        },
        "misses": [hole.graph],
    }
    transport = StubTransport(answer, {"https://presigned.example/hit": payload})
    store = HubGraphStore(transport, "release-1", LANE, SM)

    calls: list = []
    adopted_host = fresh_host(binding, tmp_path)
    adopted_host.setup(
        store=store, document=store.get_graphs("release-1"), sm=SM,
        loader=counting_loader(calls),
        artifacts_dir=tmp_path / "adopted",
        installed=INSTALLED,
    )
    assert transport.asks == 1  # ONE ask per boot, cached thereafter
    assert [h.record.graph for h in adopted_host.holes] == [hole.graph]
    fetched = tmp_path / "adopted" / ENV.value / f"{hit.graph}.so"
    assert fetched.read_bytes() == payload
    hit_manifest = store.get_manifest(hit.graph, ENV)
    assert hit_manifest is not None and hit_manifest.sm_compiled == SM

    # A lying digest is a StoreError -> a HOLE with the reason stated, never
    # an adopted artifact and never a boot failure (partial-hit everywhere).
    transport.blobs["https://presigned.example/hit"] = b"tampered"
    tampered_store = HubGraphStore(transport, "release-1", LANE, SM)
    tampered_host = fresh_host(binding, tmp_path)
    tampered_host.setup(
        store=tampered_store, document=tampered_store.get_graphs("release-1"),
        sm=SM, loader=counting_loader([]),
        artifacts_dir=tmp_path / "adopted-2",
        installed=INSTALLED,
    )
    reasons = {h.record.graph: h.reason for h in tampered_host.holes}
    assert reasons[hit.graph].startswith("store_error:")
    assert "digest verification" in reasons[hit.graph]

    # The boot-side store is read-only by construction.
    with pytest.raises(StoreError, match="read-only"):
        store.publish_artifact(hit.graph, ENV, tmp_path / "x.so", manifest())
