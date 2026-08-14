"""pgw#1036: the worker hydrates a ``ModularPipeline`` slot from the LOCAL
tree and refuses an un-repointed spec typed. (th#1941 deleted the
override-tree half: a composed manifest has no second tree to route.)

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
# ---------------------------------------------------------------------------
# Executor-level: the full dispatch path (real worker/lifecycle/executor over
# the hub double), the acceptance's "modular slot arrives at setup()" bar.
# ---------------------------------------------------------------------------

BASE_REF = "harness/tiny-modular"


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


def test_executor_hydrates_a_modular_slot(tmp_path) -> None:
    assert MODULAR_DECLARED  # endpoint module imported, slot declared
    base = build_base_tree(tmp_path / "base", fill=1.0)
    blobs = BlobHost(tmp_path / "blobs")
    try:
        base_snap = _snapshot(blobs, "snap-tiny-modular", base)
        flat = [pb.ModelBinding(
            slot="pipeline", ref=BASE_REF, manifest_digest="snap-tiny-modular")]
        with hub_double(modules=(
            "harness.toy_endpoints", "harness.modular_endpoint",
        )) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)

            res = _run(conn, "r-mod-flat", flat,
                       {"snap-tiny-modular": base_snap})
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            out = _decode(res.inline).response
            assert "unet=set" in out
            assert "vae_fill=1" in out
            assert "vae_ref=none" in out       # partition stays excluded
            assert "scheduler=set" in out
    finally:
        blobs.shutdown()
def _EOF_SENTINEL():
    pass
