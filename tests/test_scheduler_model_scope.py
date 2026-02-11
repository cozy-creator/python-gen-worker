import msgspec

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.worker import Worker


class _Payload(msgspec.Struct):
    model: str


def test_scheduler_cannot_widen_manifest_model_scope(monkeypatch) -> None:
    # Tenant-declared scope via baked manifest mapping.
    w = Worker(
        user_module_names=[],
        worker_jwt="dummy-worker-jwt",
        manifest={"models": {"sd15": "cozy:demo/sd15:latest"}},
    )

    # Avoid background download threads in this unit test.
    monkeypatch.setattr(w, "_start_startup_prefetch", lambda *_args, **_kwargs: None)

    # Scheduler tries to widen scope (should be ignored / intersected away).
    msg = pb.WorkerSchedulerMessage(
        release_artifact_config=pb.ReleaseArtifactConfig(
            supported_repo_refs=["cozy:evil/evil:latest"],
            repo_ref_by_key={"evil": "cozy:evil/evil:latest"},
            required_variant_refs=[],
        )
    )
    w._process_message(msg)

    # Worker must not accept new keys/refs outside the tenant manifest mapping.
    assert "evil" not in w._release_model_id_by_key
    assert w._release_allowed_model_ids == {"cozy:demo/sd15:latest"}
