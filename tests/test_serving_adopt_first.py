"""pgw#1372: the ship-code-as-is serving layer — eager-first, ctx.compile adopt.

Integration, no mocks: the fixture endpoint under
``tests/fixtures/serving_v2_endpoint`` is shaped exactly like the
serverless-endpoints sdxl ``main_v2.py`` (``class ...(Endpoint)``, imperative
``self.pipe.unet = ctx.compile(self.pipe.unet)`` marking, handlers on the
serving context surface). It boots from a CONFIG-ONLY checkpoint (fake
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
from gen_worker._vendor.torchcg.lane import Lane
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


def fresh_host(binding: DeployBinding, tmp_path: Path) -> EndpointHost:
    loaded = load_endpoint(FIXTURE_DIR)
    return EndpointHost(loaded, binding, output_dir=tmp_path / "outputs")


@pytest.fixture()
def host(binding: DeployBinding, tmp_path: Path) -> EndpointHost:
    booted = fresh_host(binding, tmp_path)
    booted.setup()
    return booted


# --- the eager path: standalone, first --------------------------------------


def test_eager_boot_serves_a_request_end_to_end(host: EndpointHost, tmp_path: Path) -> None:
    # setup() ran the author's own from_pretrained against ctx.checkpoint_dir
    # under ctx.lane.dtype, and typed the hub overrides.
    assert host.instance.pipe.dtype is torch.float32
    assert host.instance.defaults.steps == 2  # hub override beat the schema default
    assert host.instance.defaults.guidance == 6.0  # schema default survived
    # ctx.compile with no adoption source is a transparent pass-through: the
    # marked module carries no swapped forward.
    assert "forward" not in host.instance.pipe.unet.__dict__

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

    # The boot recorded a model_load span for the author's setup, and no
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
    # setup/teardown are the base-class hooks, never routable handlers.
    assert sorted(loaded.handlers) == ["generate", "generate_turbo"]
    lane = loaded.lane()
    assert lane.name == "bf16"
    assert lane.contract == "plain.fp32@1"
    assert loaded.declaration.samples is not None
    payloads = loaded.declaration.samples()
    assert len(payloads) == 2  # every bucket through the handler
    with pytest.raises(EndpointLoadError, match="no lane named 'fp8'"):
        loaded.lane("fp8")
    with pytest.raises(EndpointLoadError, match="no endpoint.toml"):
        load_endpoint(tmp_path)


def test_the_endpoint_base_class_is_required_and_hooks_are_verified(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rogue"
    (root / "src" / "rogue_ep").mkdir(parents=True)
    (root / "endpoint.toml").write_text('main = "rogue_ep.main"\n')
    (root / "src" / "rogue_ep" / "__init__.py").write_text("")
    (root / "src" / "rogue_ep" / "main.py").write_text(
        "import msgspec\n"
        "class In(msgspec.Struct):\n    prompt: str\n"
        "class Rogue:\n"
        "    def setup(self, ctx):\n        pass\n"
        "    def generate(self, ctx, payload: In):\n        return payload\n"
    )
    # No Endpoint subclass in the module: the base is REQUIRED, named error.
    with pytest.raises(EndpointLoadError, match="Endpoint"):
        load_endpoint(root)
    # A DECORATED class that skips the base is refused by name too — the
    # marker does not exempt anyone from the behavior contract.
    (root / "src" / "rogue_ep" / "main.py").write_text(
        "import msgspec\n"
        "from gen_worker.serving.loader import EndpointDeclaration\n"
        "class In(msgspec.Struct):\n    prompt: str\n"
        "class Rogue:\n"
        "    def setup(self, ctx):\n        pass\n"
        "    def generate(self, ctx, payload: In):\n        return payload\n"
        "Rogue.__cozy_endpoint__ = EndpointDeclaration()\n"
    )
    import importlib
    import rogue_ep.main

    importlib.reload(rogue_ep.main)
    from gen_worker.serving.loader import load_endpoint_module

    with pytest.raises(EndpointLoadError, match="does not inherit"):
        load_endpoint_module("rogue_ep.main")
    # A subclass whose hook breaks the (self, ctx) contract is refused too:
    # framework capabilities arrive via ctx ONLY.
    (root / "src" / "rogue_ep" / "main.py").write_text(
        "import msgspec\n"
        "from gen_worker import Endpoint\n"
        "class In(msgspec.Struct):\n    prompt: str\n"
        "class Rogue(Endpoint):\n"
        "    def setup(self, ctx, extra_toolbox):\n        pass\n"
        "    def generate(self, ctx, payload: In):\n        return payload\n"
    )
    importlib.reload(rogue_ep.main)
    with pytest.raises(EndpointLoadError, match=r"exactly \(self, ctx\)"):
        load_endpoint_module("rogue_ep.main")


def test_teardown_hook_runs_through_the_base_class_contract(
    host: EndpointHost,
) -> None:
    instance = host.instance
    assert not hasattr(instance, "torn_down")
    host.teardown()
    assert instance.torn_down is True
    assert host.instance is None


# --- the adopt path: publish-time discovery, store pull, ctx.compile --------


def publish_document(host: EndpointHost) -> GraphSetDocument:
    """The publish-time derive, run for real: discovery hooks the marked
    module on the author's live pipeline and drives the author's own
    handlers with the declared sample payloads."""
    lane: Lane = host.loaded.lane()
    samples_fn = host.loaded.declaration.samples
    assert samples_fn is not None
    samples = samples_fn()

    def drive() -> None:
        for index, payload in enumerate(samples):
            ctx = host.make_context(f"trace-{index}", is_trace=True)
            host.dispatch("generate", payload, request_id=f"trace-{index}", ctx=ctx)

    lane_graphs = discover_lane(lane, {"unet": host.instance.pipe.unet}, drive)
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
    # The audit fired BEFORE the author's class was even instantiated.
    assert refused.instance is None
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
    store = HubGraphStore(transport, "release-1", "bf16", SM)

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
    tampered_store = HubGraphStore(transport, "release-1", "bf16", SM)
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
