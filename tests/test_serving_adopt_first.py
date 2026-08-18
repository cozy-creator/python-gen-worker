"""pgw#1372: the ship-code-as-is serving layer, eager-first, adopt as bolt-on.

Integration, no mocks: the fixture endpoint under
``tests/fixtures/serving_v2_endpoint`` is shaped exactly like the
serverless-endpoints sdxl ``main_v2.py`` (lanes + samples declaration,
``setup(ctx)`` loading a pipeline from ``ctx.checkpoint_dir``, handlers on
the serving context surface). It boots from a CONFIG-ONLY checkpoint (fake
weights), serves real requests end-to-end on CPU, and the adopt path runs
real publish-time discovery output through a real ``LocalGraphStore`` — only
the artifact loader is a stub, because bytes-to-callable is the AOTInductor
runtime's job on the target GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import msgspec
import pytest
import torch

from gen_worker import ValidationError, boot_stages
from gen_worker._vendor.tensorfs import LocalCAS
from gen_worker._vendor.torchcg import EnvironmentMismatch
from gen_worker._vendor.torchcg.discovery import discover_lane
from gen_worker._vendor.torchcg.document import GraphSetDocument
from gen_worker._vendor.torchcg.graph_identity import EnvIdentity, closure_hash
from gen_worker._vendor.torchcg.requirements import RequirementsManifest
from gen_worker._vendor.torchcg.store import LocalGraphStore, StoreError
from gen_worker.serving import (
    BoundAdapter,
    DeployBinding,
    EndpointHost,
    EndpointLoadError,
    ServeDispatchError,
    load_endpoint,
)
from gen_worker.serving.hub_store import HubGraphStore

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_v2_endpoint"
SM = "sm_89"
INSTALLED = {"torch": torch.__version__}
CLOSURE = closure_hash(INSTALLED)
ENV = EnvIdentity(closure=CLOSURE, sm=SM)


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
        # Hub per-checkpoint overrides — mutable deploy state, typed by
        # ctx.checkpoint_defaults(TinyDefaults) against the author's schema.
        defaults={"steps": 2, "max_guidance": 8.0},
    )


@pytest.fixture()
def host(binding: DeployBinding, tmp_path: Path) -> EndpointHost:
    loaded = load_endpoint(FIXTURE_DIR)
    host = EndpointHost(loaded, binding, output_dir=tmp_path / "outputs")
    host.setup()
    return host


# --- the eager path: standalone, first --------------------------------------


def test_eager_boot_serves_a_request_end_to_end(host: EndpointHost, tmp_path: Path) -> None:
    # setup() ran the author's own from_pretrained against ctx.checkpoint_dir
    # under ctx.lane.dtype, and typed the hub overrides.
    assert host.instance.pipe.dtype is torch.float32
    assert host.instance.defaults.steps == 2  # hub override beat the schema default
    assert host.instance.defaults.guidance == 6.0  # schema default survived

    ctx = host.make_context("req-1")
    out = host.dispatch(
        "generate",
        {"prompt": "hello", "aspect_ratio": "1:1", "guidance_scale": 12.0, "seed": 3},
        request_id="req-1",
        ctx=ctx,
    )
    assert out.model_used == "ckpt:tiny@1"
    saved = tmp_path / "outputs" / out.image.ref
    assert saved.is_file() and saved.stat().st_size > 0

    # The clamp surface recorded the caller-visible adjustment (12 -> 8).
    rows = [row for row in ctx.adjustments if row["field"] == "guidance_scale"]
    assert rows and rows[0]["requested"] == "12.0" and rows[0]["applied"] == "8.0"

    # The boot recorded a model_load span for the author's setup.
    stages = [span.stage.value for span in boot_stages.recorded()]
    assert "model_load" in stages


def test_dispatch_refusals_are_typed_before_author_code_runs(host: EndpointHost) -> None:
    with pytest.raises(ServeDispatchError, match="no function 'enhance'"):
        host.dispatch("enhance", {"prompt": "x"}, request_id="r")
    with pytest.raises(msgspec.ValidationError):
        host.dispatch("generate", {"prompt": "x", "num_inference_steps": 999}, request_id="r")
    with pytest.raises(msgspec.ValidationError):
        host.dispatch("generate", {"prompt": "x", "unknown_field": 1}, request_id="r")


def test_adapter_gate_and_trace_flag_shape_the_turbo_handler(
    host: EndpointHost, binding: DeployBinding, tmp_path: Path
) -> None:
    # No adapter bound, not a trace: the author's own refusal fires.
    with pytest.raises(ValidationError, match="no distillation adapter"):
        host.dispatch("generate_turbo", {"prompt": "x"}, request_id="r")
    # The trace drive relaxes it (ctx.is_trace) — publish-time discovery runs
    # the same handler with no deploy state.
    ctx = host.make_context("trace", is_trace=True)
    out = host.dispatch("generate_turbo", {"prompt": "x"}, request_id="trace", ctx=ctx)
    assert out.model_used == "ckpt:tiny@1"
    # A bound adapter serves and stamps the recipe name.
    host.rebind(
        DeployBinding(
            checkpoint_ref=binding.checkpoint_ref,
            checkpoint_dir=binding.checkpoint_dir,
            defaults=binding.defaults,
            adapter=BoundAdapter(name="lightning-4step", path=tmp_path / "lora"),
        )
    )
    out = host.dispatch("generate_turbo", {"prompt": "x"}, request_id="r2")
    assert out.model_used == "ckpt:tiny@1+lightning-4step"


def test_loader_states_the_surface_and_refuses_typed(tmp_path: Path) -> None:
    loaded = load_endpoint(FIXTURE_DIR)
    assert loaded.module_name == "serving_v2_fixture.main"
    assert sorted(loaded.handlers) == ["generate", "generate_turbo"]
    lane = loaded.lane()
    assert lane.name == "bf16" and lane.compile == ("unet",)
    assert lane.contract == "plain.fp32@1"
    assert loaded.declaration.samples is not None
    payloads = loaded.declaration.samples()
    assert len(payloads) == 2  # every bucket through the handler
    with pytest.raises(EndpointLoadError, match="no lane named 'fp8'"):
        loaded.lane("fp8")
    with pytest.raises(EndpointLoadError, match="no endpoint.toml"):
        load_endpoint(tmp_path)


# --- the adopt path: publish-time discovery, store pull, swap-in ------------


def publish_document(host: EndpointHost) -> GraphSetDocument:
    """The publish-time derive, run for real: discovery hooks the lane's
    target on the author's live pipeline and drives the author's own
    handlers with the declared sample payloads."""
    lane = host.lane
    sample_fn = host.loaded.declaration.samples
    assert sample_fn is not None
    samples = sample_fn()

    def drive() -> None:
        for index, payload in enumerate(samples):
            ctx = host.make_context(f"trace-{index}", is_trace=True)
            host.dispatch("generate", payload, request_id=f"trace-{index}", ctx=ctx)

    lane_graphs = discover_lane(lane, host.roots(), drive)
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


def test_adopt_swaps_the_module_forward_and_hands_ordered_holes(
    host: EndpointHost, tmp_path: Path
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
    adoption = host.adopt(
        store, document, SM,
        loader=counting_loader(calls),
        artifacts_dir=tmp_path / "adopted",
        installed=INSTALLED,
    )

    # THE handoff for pgw#1371: ordered holes carrying full GraphRecords.
    assert [h.record.graph for h in host.holes] == [hole.graph]
    assert host.holes[0].reason == "miss"
    assert host.holes[0].record.target == "unet"
    assert host.holes[0].record.ingress is hole.ingress

    # The armed bucket serves THROUGH the swap (module-forward verified
    # called); the hole bucket stays on the author's eager forward.
    hit_shape = tuple(d for d in hit.ingress.inputs[0].shape)
    hit_ratio = "1:1" if hit_shape == (16, 16) else "3:4"
    hole_ratio = "3:4" if hit_ratio == "1:1" else "1:1"
    host.dispatch("generate", {"prompt": "x", "aspect_ratio": hit_ratio}, request_id="r1")
    assert calls and set(calls) == {hit.graph}
    swapped = len(calls)
    host.dispatch("generate", {"prompt": "x", "aspect_ratio": hole_ratio}, request_id="r2")
    assert len(calls) == swapped  # the hole ran eager

    # A late mint arms without a reboot and leaves the hole list empty.
    minted = tmp_path / "late.so"
    minted.write_bytes(b"late")
    adoption.arm(hole, minted)
    host.dispatch("generate", {"prompt": "x", "aspect_ratio": hole_ratio}, request_id="r3")
    assert hole.graph in calls
    assert host.holes == ()

    # Telemetry: the adopt_pull span replaced the keyset span in this flow.
    spans = [s for s in boot_stages.recorded() if s.stage.value == "adopt_pull"]
    assert len(spans) == 1
    assert spans[0].attrs["graphs_from"] == "release"
    assert spans[0].attrs["artifact_from_store"] == "1"
    assert spans[0].attrs["artifact_from_eager"] == "1"
    assert not [s for s in boot_stages.recorded() if s.stage.value == "keyset"]


def test_exact_env_mismatch_refuses_loudly(host: EndpointHost, tmp_path: Path) -> None:
    document = publish_document(host)
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    with pytest.raises(EnvironmentMismatch, match="build-system bug"):
        host.adopt(
            store, document, SM,
            loader=counting_loader([]),
            artifacts_dir=tmp_path / "adopted",
            installed={"torch": "0.0.0-divergent"},
        )
    spans = [s for s in boot_stages.recorded() if s.stage.value == "adopt_pull"]
    assert spans and spans[0].attrs["refusal"] == "environment_mismatch"


def test_absent_or_eager_permanent_metadata_is_a_clean_noop(
    host: EndpointHost, tmp_path: Path
) -> None:
    assert host.adopt(
        None, None, SM, loader=counting_loader([]), artifacts_dir=tmp_path
    ) is None
    eager = GraphSetDocument(closure=CLOSURE, lanes=())
    assert host.adopt(
        None, eager, SM, loader=counting_loader([]), artifacts_dir=tmp_path
    ) is None
    assert host.holes == ()
    spans = [s for s in boot_stages.recorded() if s.stage.value == "adopt_pull"]
    assert [s.attrs["graphs_from"] for s in spans] == ["absent", "eager_permanent"]
    # The endpoint still serves — the eager bridge is unconditional.
    out = host.dispatch("generate", {"prompt": "still serving"}, request_id="r")
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
    host: EndpointHost, tmp_path: Path
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
    store = HubGraphStore(transport, "release-1", "bf16", SM)

    calls: list = []
    host.adopt(
        store, document, SM,
        loader=counting_loader(calls),
        artifacts_dir=tmp_path / "adopted",
        installed=INSTALLED,
    )
    assert transport.asks == 1  # ONE ask per boot, cached thereafter
    assert [h.record.graph for h in host.holes] == [hole.graph]
    fetched = tmp_path / "adopted" / ENV.value / f"{hit.graph}.so"
    assert fetched.read_bytes() == payload
    hit_manifest = store.get_manifest(hit.graph, ENV)
    assert hit_manifest is not None and hit_manifest.sm_compiled == SM

    # A lying digest is a StoreError -> a HOLE with the reason stated, never
    # an adopted artifact and never a boot failure (partial-hit everywhere).
    transport.blobs["https://presigned.example/hit"] = b"tampered"
    store_again = HubGraphStore(transport, "release-1", "bf16", SM)
    host2_calls: list = []
    adoption = host.adopt(
        store_again, document, SM,
        loader=counting_loader(host2_calls),
        artifacts_dir=tmp_path / "adopted-2",
        installed=INSTALLED,
    )
    reasons = {h.record.graph: h.reason for h in adoption.holes}
    assert reasons[hit.graph].startswith("store_error:")
    assert "digest verification" in reasons[hit.graph]

    # The boot-side store is read-only by construction.
    with pytest.raises(StoreError, match="read-only"):
        store.publish_artifact(hit.graph, ENV, tmp_path / "x.so", manifest())
