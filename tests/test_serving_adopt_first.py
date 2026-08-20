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
from gen_worker._vendor.torchcg.document import GraphRecord, GraphSetDocument
from gen_worker._vendor.torchcg.graph_identity import EnvIdentity
from gen_worker._vendor.torchcg.requirements import RequirementsManifest
from gen_worker._vendor.torchcg.store import LocalGraphStore, StoreError
from gen_worker.serving import (
    DeployBinding,
    DistillationAdapter,
    EndpointHost,
    EndpointLoadError,
    ServeDispatchError,
    load_endpoint,
)
from gen_worker.serving.hub_store import HubGraphStore

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_v2_endpoint"
LANE = "sdxl.diffusers-bf16@1"
SM = "sm_89"
STACK: tuple[tuple[str, str], ...] = (("torch", torch.__version__),)
ENV = EnvIdentity(stack=STACK, sm=SM)

#: Hub per-checkpoint overrides — mutable deploy state, decoded by
#: ``LoadContext.defaults()`` against ``SDXL.Defaults``. Small step knob so
#: the CPU loop stays tiny.
OVERRIDES: dict[str, Any] = {
    # A hub row narrows [lo, hi] and moves the default; it can NEVER rename
    # the knob (pgw#1377: `name` is stamped by the struct, never wire input),
    # so the caller-visible adjustment row names the KNOB — `guidance`.
    "steps": {"default": 2, "lo": 1, "hi": 8},
    "guidance": {"default": 6.0, "lo": 1.5, "hi": 8.0},
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
        model="sdxl",
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
    assert out.model == "ckpt:tiny@1" and out.loras == []
    saved = tmp_path / "outputs" / out.image.ref
    assert saved.is_file() and saved.stat().st_size > 0

    # The Knob clamp recorded the caller-visible adjustment (12 -> 8).
    rows = [row for row in ctx.adjustments if row["field"] == "guidance"]
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
            model=binding.model,
            defaults=binding.defaults,
            adapter=DistillationAdapter(
                name="lightning-4step", path=tmp_path / "lora",
                defaults=SDXL.Lora.Defaults(), ref="cozy/lightning-4step@1",
            ),
        )
    )
    out = host.dispatch("generate", {"prompt": "x"}, request_id="r2")
    assert out.model == "ckpt:tiny@1"
    assert [(used.ref, used.scale) for used in out.loras] == [("cozy/lightning-4step@1", 1.0)]
    # Explicit guidance/negatives on a cfg-free serving are IGNORED
    # caller-visibly — ctx.warn rows in the response envelope, never a
    # silent drop and never an aborted request (Paul's warn ruling).
    ctx = host.make_context("r3")
    out = host.dispatch(
        "generate",
        {"prompt": "x", "guidance_scale": 7.0, "negative_prompt": "bad"},
        request_id="r3", ctx=ctx,
    )
    assert [used.ref for used in out.loras] == ["cozy/lightning-4step@1"]
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
        # Author code is trace-oblivious (Paul ruling): the derive varies
        # BINDINGS/inputs; it never asks the entrypoint to cooperate.
        for index, ratio in enumerate(AspectRatio):
            host.dispatch(
                "generate", {"prompt": "trace", "aspect_ratio": str(ratio)},
                request_id=f"trace-{index}",
            )

    lane_graphs = discover_lane(LANE, ("unet",), {"unet": model.pipe.unet}, drive)
    return GraphSetDocument(stack=STACK, lanes=(lane_graphs,))


def manifest() -> RequirementsManifest:
    return RequirementsManifest(include_set=(("torch", torch.__version__),), sm_compiled=SM)


def counting_loader(
    calls: "list[str]",
) -> "Callable[[Path, Any, Any], Callable[..., Any]]":
    """The ONE stub: bytes-to-callable needs a real AOTI package and a GPU.

    THREE arguments since pgw#1460/tcg#58, and the third is asserted rather
    than ignored: the production loader binds the compiled graph's constants
    to the LIVE MODULE that claimed the record, so a double that drops the
    module models a loader that cannot exist. That omission is the whole
    reason this stub stayed green over two raw, unservable loaders.
    """

    def load(path: Path, record: Any, module: Any) -> "Callable[..., Any]":
        assert isinstance(module, torch.nn.Module), (
            "the loader is handed the module its constants bind from")

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
    assert len(lane_graphs.graphs) == 2  # two buckets -> two graph specializations
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
        stack=STACK,
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
    with pytest.raises(EnvironmentMismatch, match=r"torch 0.0.0-divergent != stamped"):
        refused.setup(
            store=store, document=document, sm=SM,
            loader=counting_loader([]),
            artifacts_dir=tmp_path / "adopted",
            stack={"torch": "0.0.0-divergent"},
        )
    # The audit fired BEFORE any author model was even instantiated.
    assert refused.instances == {}
    spans = [s for s in boot_stages.recorded() if s.stage.value == "adopt_pull"]
    assert spans and spans[-1].attrs["refusal"] == "environment_mismatch"


def test_eager_permanent_metadata_is_a_clean_noop(
    binding: DeployBinding, tmp_path: Path
) -> None:
    eager = GraphSetDocument(stack=STACK, lanes=())
    booted = fresh_host(binding, tmp_path)
    booted.setup(document=eager, sm=SM)
    assert booted.adoption is None
    assert booted.holes == ()
    spans = [s for s in boot_stages.recorded() if s.stage.value == "adopt_pull"]
    assert [s.attrs["graphs_from"] for s in spans] == ["eager_permanent"]
    # The endpoint still serves — the eager bridge is unconditional.
    out = booted.dispatch("generate", {"prompt": "still serving"}, request_id="r")
    assert out.model == "ckpt:tiny@1"


def test_a_store_less_boot_still_forms_the_full_mint_worklist(
    host: EndpointHost, binding: DeployBinding, tmp_path: Path
) -> None:
    document = publish_document(host)
    booted = fresh_host(binding, tmp_path)
    booted.setup(
        document=document, sm=SM, loader=counting_loader([]),
        artifacts_dir=tmp_path / "adopted", stack=STACK,
    )
    # Metadata known, artifacts unreachable: everything is stated mint work.
    assert [h.reason for h in booted.holes] == ["miss", "miss"]
    out = booted.dispatch("generate", {"prompt": "eager"}, request_id="r")
    assert out.model == "ckpt:tiny@1"


# --- the hub-backed store: th#2133's REAL answer shape ----------------------
#
# This block used to can a `{document, artifacts, misses}` payload that the
# route never emitted — it was written before th#2133 landed, and it made the
# store look tested while `hub_store` could not have parsed one real answer.
# What is canned below is the shape a live hub actually returned on
# `GET /v1/worker/releases/<id>/compiled-graphs?lane=&sm=`: per-graph rows with
# a `status`, the observed `ingress` CONTRACT (not just its digest — torchcg
# dispatches on the rows), and a presigned snapshot manifest under `transport`.


class StubTransport:
    """The th#2133 route answer, canned in the shape the route emits."""

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


def _adopt_answer(
    document: GraphSetDocument,
    hit: GraphRecord,
    hole: GraphRecord,
    payload: bytes,
) -> dict[str, Any]:
    """One th#2133 answer: `hit` minted for this env, `hole` not."""
    import hashlib

    lane = document.lanes[0]

    def row(record: GraphRecord, status: str) -> dict[str, Any]:
        base: dict[str, Any] = {
            "graph_hash": record.graph,
            "graph_specialization": "",
            "module_path": record.target,
            "ingress_digest": record.ingress.digest(),
            # THE CONTRACT ITSELF. Without it the lane document cannot be
            # rebuilt and no artifact can legally arm (th#2134's migration).
            "ingress": record.ingress.as_dict(),
            "status": status,
            "found": status == "hit",
        }
        if status != "hit":
            return base
        base.update({
            "content_digest": hashlib.sha256(payload).hexdigest(),
            "artifact_path": "compiled/graph.so",
            "size_bytes": len(payload),
            "checkpoint_id": "sha256:deadbeef",
            "artifact_kind": "aoti",
            "requirements_manifest": manifest().as_dict(),
            "transport": {
                "snapshot_digest": "sha256:deadbeef",
                "files": [{
                    "path": "compiled/graph.so",
                    "size_bytes": len(payload),
                    "digest": hashlib.sha256(payload).hexdigest(),
                    "url": "https://presigned.example/hit",
                }],
            },
        })
        return base

    return {
        "object": "release_compiled_graphs",
        "release_id": "release-1",
        "binding_generation": 0,
        "env_compile_stack": [list(row) for row in document.stack],
        "lane": lane.contract,
        "lane_stamped": True,
        "lane_contract": {"stamp": lane.contract, "contract_digest": "",
                          "document": None, "requires": ""},
        "sm": SM,
        "empty": False,
        "targets": list(lane.targets),
        "unobserved_targets": list(lane.unobserved_targets),
        "passes": list(lane.passes),
        "graphs": [row(hit, "hit"), row(hole, "miss")],
        "hits": 1,
        "misses": 1,
    }


def test_hub_store_partial_hit_verifies_digests_and_misses_clean(
    host: EndpointHost, binding: DeployBinding, tmp_path: Path
) -> None:
    document = publish_document(host)
    hit, hole = document.lanes[0].graphs
    payload = b"presigned-compiled-bytes"
    answer = _adopt_answer(document, hit, hole, payload)
    transport = StubTransport(answer, {"https://presigned.example/hit": payload})
    store = HubGraphStore(transport, "release-1", LANE, SM)

    # The document is REBUILT from the answer — the hub stores the derive's
    # rows, not the derive's bytes.
    rebuilt = store.get_graphs("release-1")
    assert rebuilt is not None and rebuilt.stack == document.stack
    # pgw#1384: graph order is SEMANTIC, not canonical — the producer states
    # it (default-parameter classes first) and the miner mints holes in that
    # order, so the rebuild preserves the answer's order rather than sorting.
    assert [r.graph for r in rebuilt.lanes[0].graphs] == [hit.graph, hole.graph]
    # The ORDERED hole list pgw#1371's background mint consumes.
    assert store.misses == (hole.graph,)

    calls: list = []
    adopted_host = fresh_host(binding, tmp_path)
    adopted_host.setup(
        store=store, document=rebuilt, sm=SM,
        loader=counting_loader(calls),
        artifacts_dir=tmp_path / "adopted",
        stack=STACK,
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
        stack=STACK,
    )
    reasons = {h.record.graph: h.reason for h in tampered_host.holes}
    assert reasons[hit.graph].startswith("store_error:")
    assert "digest verification" in reasons[hit.graph]

    # The boot-side store is read-only by construction.
    with pytest.raises(StoreError, match="read-only"):
        store.publish_artifact(hit.graph, ENV, tmp_path / "x.so", manifest())


def test_the_adopt_route_is_allowlisted_in_procsplit_pgw1372() -> None:
    """WITHOUT THIS ENTRY THE WHOLE ADOPT BOOT IS DEAD ON EVERY FLEET POD.

    The split parent refuses any path not in the table, `hub_store` treats a
    refusal as a miss (correctly — it must never block a boot), and every pod
    serves eager forever while every test that patches `broker.request` stays
    green. pgw#1353's keyset tier, exactly. So the assertion is on the TABLE.
    """
    from gen_worker.procsplit import actions

    action, query, _ = actions.authorize({
        "method": "GET",
        "path": "/v1/worker/releases/rel-123/compiled-graphs",
        "query": {"lane": LANE, "sm": SM},
    })
    assert action.name == "release.compiled_graphs"
    assert query == {"lane": LANE, "sm": SM}

    # An unlisted query key is a refusal, not an ignored field.
    with pytest.raises(actions.ActionRefused, match="org_id"):
        actions.authorize({
            "method": "GET",
            "path": "/v1/worker/releases/rel-123/compiled-graphs",
            "query": {"lane": LANE, "sm": SM, "org_id": "someone-elses"},
        })

    # The path grammar is pinned: a release id is an identifier, never a path.
    with pytest.raises(actions.ActionRefused, match="not an allowlisted"):
        actions.authorize({
            "method": "GET",
            "path": "/v1/worker/releases/rel-123/extra/compiled-graphs",
            "query": {"lane": LANE, "sm": SM},
        })
