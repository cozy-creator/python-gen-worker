"""pgw#1036: the worker hydrates a ``ModularPipeline`` slot from the LOCAL
tree, refuses an un-repointed spec typed, and delivers th#980/pgw#617
component overrides through the spec routing ``ModularPipeline.__init__``
cannot silently discard.

Integration-style against REAL diffusers 0.39 modular pipelines (real
safetensors on disk, CPU): the harness trees reproduce the mirror disease —
every index spec names an upstream repo id — and the whole module runs under
``HF_HUB_OFFLINE=1``, so any fetch attempt is a loud failure, not a silent
success (the ie#615 hydration-guard hazard, verified rather than assumed).
"""

from __future__ import annotations

import msgspec
import pytest
import torch

from gen_worker.models import provision
from gen_worker.models.loading import (
    ModularHydrationError,
    is_modular_pipeline_class,
    load_from_pretrained,
)
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.blob_host import BlobHost
from harness.hub_double import hub_double, is_ready, is_result_for
from harness.modular_endpoint import (
    MODULAR_DECLARED,
    TinyModularPipeline,
    UPSTREAM_REPO_ID,
    build_base_tree,
    build_override_vae_tree,
    tiny_vae,
    tree_files,
)
from harness.toy_endpoints import EchoIn, EchoOut


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")


def _fill(mod) -> float:
    return float(next(iter(mod.parameters())).flatten()[0])


def test_detection_is_class_shaped() -> None:
    from diffusers import DiffusionPipeline

    assert is_modular_pipeline_class(TinyModularPipeline)
    assert not is_modular_pipeline_class(DiffusionPipeline)
    assert not is_modular_pipeline_class(str)


def test_hydrates_every_component_from_local_tree(tmp_path) -> None:
    """(a) the slot arrives with every needed component non-None, weights
    from OUR tree; (d) no network egress — the index repo id does not exist
    and HF_HUB_OFFLINE=1, so a fetch would fail the test."""
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    pipe = load_from_pretrained(TinyModularPipeline, tree)

    assert pipe.unet is not None and _fill(pipe.unet) == 1.0
    assert pipe.vae is not None and _fill(pipe.vae) == 1.0
    assert pipe.scheduler is not None
    # the unselected partition (config-only dir) stays excluded — the
    # 144 GB bound-checkpoint accounting's assumption, preserved
    assert pipe.vae_ref is None

    # every spec re-pointed local (or neutralized): a later endpoint-side
    # bare load_components() must be equally incapable of fetching
    for name, spec in pipe._component_specs.items():
        if spec.default_creation_method != "from_pretrained":
            continue
        p = spec.pretrained_model_name_or_path
        assert p is None or str(tree) in str(p), (name, p)
        assert p is None or UPSTREAM_REPO_ID not in str(p), (name, p)

    prov = pipe._cozy_modular_hydration
    assert set(prov) == {"unet", "vae", "scheduler"}
    assert all(str(tree) in src for src in prov.values())


def test_unrepointable_spec_refuses_typed(tmp_path) -> None:
    """(b) an index naming a component the local tree does not carry refuses
    typed — never a fetch from the index's repo id."""
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    import shutil

    shutil.rmtree(tree / "unet")  # named by the index, absent locally
    with pytest.raises(ModularHydrationError) as exc:
        load_from_pretrained(TinyModularPipeline, tree)
    assert "unet" in str(exc.value)
    assert UPSTREAM_REPO_ID in str(exc.value)


def test_override_tree_replaces_default_subdir_layout(tmp_path) -> None:
    """(c) a th#980 component override against a modular slot LANDS: the
    loaded module's bytes come from the override tree, not the base."""
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    override = build_override_vae_tree(tmp_path / "ovr", fill=2.0, subdir=True)
    pipe = load_from_pretrained(
        TinyModularPipeline, tree, component_trees={"vae": str(override)})
    assert pipe.vae is not None and _fill(pipe.vae) == 2.0  # override bytes
    assert pipe.unet is not None and _fill(pipe.unet) == 1.0  # base bytes
    assert str(override) in pipe._cozy_modular_hydration["vae"]


def test_override_tree_replaces_default_root_layout(tmp_path) -> None:
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    override = build_override_vae_tree(tmp_path / "ovr", fill=3.0, subdir=False)
    pipe = load_from_pretrained(
        TinyModularPipeline, tree, component_trees={"vae": str(override)})
    assert pipe.vae is not None and _fill(pipe.vae) == 3.0


def test_dtype_lands_on_hydrated_components(tmp_path) -> None:
    """(c) dtype half of the assertion: the binding's declared dtype reaches
    the hydrated modules through load_components' routing."""
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    pipe = load_from_pretrained(TinyModularPipeline, tree, dtype="fp16")
    assert pipe.unet.dtype == torch.float16
    assert pipe.vae.dtype == torch.float16


def test_unknown_override_component_refuses_typed(tmp_path) -> None:
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    override = build_override_vae_tree(tmp_path / "ovr", fill=2.0)
    with pytest.raises(ModularHydrationError, match="text_encoder"):
        load_from_pretrained(
            TinyModularPipeline, tree,
            component_trees={"text_encoder": str(override)})


def test_component_trees_on_non_modular_class_refuses_typed(tmp_path) -> None:
    """The delivery mechanisms must never cross silently — the silent-drop
    class of bug this issue exists to close."""
    class NotModular:
        @classmethod
        def from_pretrained(cls, path, **kw):  # pragma: no cover
            return cls()

    with pytest.raises(ModularHydrationError, match="not a modular"):
        load_from_pretrained(
            NotModular, tmp_path, component_trees={"vae": str(tmp_path)})


def test_load_slot_carries_component_trees(tmp_path) -> None:
    """The production compiled graph the executor calls: provision.load_slot routes
    component_trees through to the hydration guard."""
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    override = build_override_vae_tree(tmp_path / "ovr", fill=4.0)
    sl = provision.load_slot(
        TinyModularPipeline, str(tree), slot="pipeline",
        ref="harness/tiny-modular:prod",
        component_trees={"vae": str(override)}, device="cpu",
    )
    pipe = sl.obj
    assert pipe.vae is not None and _fill(pipe.vae) == 4.0
    assert pipe.unet is not None and _fill(pipe.unet) == 1.0


# ---------------------------------------------------------------------------
# Executor-level: the full dispatch path (real worker/lifecycle/executor over
# the hub double), the acceptance's "modular slot arrives at setup()" bar.
# ---------------------------------------------------------------------------

BASE_REF = "harness/tiny-modular"
OVR_REF = "harness/tiny-modular-vae-fp8"


def _snapshot(blobs: BlobHost, snap_id: str, root) -> pb.Snapshot:
    return blobs.snapshot(snap_id, [
        blobs.file(f"{snap_id}-{i}", data, path_in_snapshot=rel)
        for i, (rel, data) in enumerate(tree_files(root).items())
    ])


def _run(conn, rid: str, models, snapshots) -> pb.JobResult:
    conn.send(run_job=pb.RunJob(
        request_id=rid, attempt=1, function_name="modular-echo",
        input_payload=msgspec.msgpack.encode(EchoIn()),
        models=models, snapshots=snapshots,
    ))
    return conn.wait_for(is_result_for(rid)).job_result


def _decode(data: bytes) -> EchoOut:
    return msgspec.msgpack.decode(data, type=EchoOut)


def test_executor_hydrates_and_substitutes(tmp_path) -> None:
    assert MODULAR_DECLARED  # endpoint module imported, slot declared
    base = build_base_tree(tmp_path / "base", fill=1.0)
    override = build_override_vae_tree(tmp_path / "ovr", fill=2.0)
    blobs = BlobHost(tmp_path / "blobs")
    try:
        base_snap = _snapshot(blobs, "snap-tiny-modular", base)
        ovr_snap = _snapshot(blobs, "snap-tiny-modular-vae", override)
        flat = [pb.ModelBinding(slot="pipeline", ref=BASE_REF)]
        subst = [pb.ModelBinding(
            slot="pipeline", ref=BASE_REF, components={"vae": OVR_REF})]
        with hub_double(modules=(
            "harness.toy_endpoints", "harness.modular_endpoint",
        )) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)

            res = _run(conn, "r-mod-flat", flat, {BASE_REF: base_snap})
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            out = _decode(res.inline).response
            assert "unet=set" in out
            assert "vae_fill=1" in out
            assert "vae_ref=none" in out       # partition stays excluded
            assert "scheduler=set" in out

            res = _run(conn, "r-mod-subst", subst,
                       {BASE_REF: base_snap, OVR_REF: ovr_snap})
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            out = _decode(res.inline).response
            assert "vae_fill=2" in out          # override bytes landed
            assert "unet=set" in out
            assert "prov_vae=" in out and "snap-tiny-modular-vae" in out
    finally:
        blobs.shutdown()


def test_fp8_stored_override_tree_loads_without_upcast_config(tmp_path) -> None:
    """The encoder-trunc-fp8 shape is a transformers FineGrainedFP8 artifact
    that loads natively through its own config — nothing for the worker to
    do. This CPU test covers the nearest loadable analogue: an override tree
    whose stored dtype differs from the base (fp16 vs fp32) keeps its own
    precision fact when no binding dtype narrows it."""
    tree = build_base_tree(tmp_path / "base", fill=1.0)
    ovr_root = tmp_path / "ovr"
    vae = tiny_vae(5.0).to(torch.float16)
    vae.save_pretrained(ovr_root / "vae")
    pipe = load_from_pretrained(
        TinyModularPipeline, tree, component_trees={"vae": str(ovr_root)})
    assert pipe.vae is not None and _fill(pipe.vae) == 5.0
