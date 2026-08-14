"""A component OVERRIDE must not also fetch the component it overrides.

The override's own tree is materialized and
handed to ``from_pretrained`` as a constructed object, so the base
composition's copy of that subfolder is downloaded and then never read.
Measured cost on the shape this exists for: ~1.64 GB per SDXL text-encoder
override, per pod.

Same real boundary as ``test_component_bindings_pgw617``: real worker, real
executor, real sha256 CAS downloader against an HTTP blob host. The evidence
is the CAS itself — a blob that was never fetched is not in ``blobs/``.
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import hashlib

from gen_worker.api.binding import wire_ref
from gen_worker.models.cozy_snapshot import snapshot_dir_key
from gen_worker.models.download import select_component_paths
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.blob_host import BlobHost
from harness.hub_double import hub_double, is_ready, is_result_for
from harness.toy_endpoints import COMPOSED_DECLARED, COMPOSED_SETUPS, EchoIn, EchoOut

BASE_REF = wire_ref(COMPOSED_DECLARED)
VAE_REF = "harness/override-vae:prod"

BASE_VAE_BYTES = b"base-vae-payload-that-must-never-be-fetched"
BASE_TRANSFORMER_BYTES = b"base-transformer"


def _payload(**kw: object) -> bytes:
    return msgspec.msgpack.encode(EchoIn(**kw))  # type: ignore[arg-type]


def _base_snapshot(blobs: BlobHost) -> pb.Snapshot:
    model_index = (
        b'{"_class_name": "ToyComposedPipeline",'
        b' "vae": ["harness.toy_endpoints", "ToyVae"],'
        b' "transformer": ["harness.toy_endpoints", "ToyVae"]}'
    )
    return blobs.snapshot("snap-composed-base", [
        blobs.file("mi", model_index, path_in_snapshot="model_index.json"),
        blobs.file("tw", BASE_TRANSFORMER_BYTES,
                   path_in_snapshot="transformer/weights.txt"),
        blobs.file("vw", BASE_VAE_BYTES, path_in_snapshot="vae/weights.txt"),
    ])


def _vae_snapshot(blobs: BlobHost) -> pb.Snapshot:
    return blobs.snapshot("snap-override-vae", [
        blobs.file("ow", b"override-vae", path_in_snapshot="weights.txt"),
    ])


def _run(conn, rid: str, models, snapshots) -> pb.JobResult:
    conn.send(run_job=pb.RunJob(
        request_id=rid, attempt=1, function_name="composed-echo",
        input_payload=_payload(), models=models, snapshots=snapshots,
    ))
    return conn.wait_for(is_result_for(rid)).job_result


def _cas_blob_digests(cache_dir: Path) -> set:
    root = cache_dir / "cas" / "blobs"
    if not root.is_dir():
        return set()
    return {p.name for p in root.rglob("*") if p.is_file()}


def _materialized_base_trees(cache_dir: Path) -> list:
    root = cache_dir / "cas" / "snapshots"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# The invariant, on the real fetch path
# ---------------------------------------------------------------------------


def test_override_never_fetches_the_component_it_replaces(tmp_path) -> None:
    """Cold worker, first dispatch carries a vae override.

    RED at HEAD: the base's ``vae/weights.txt`` blob lands in the CAS (the
    downloader materializes ``snapshot.files`` verbatim) even though
    ``ToyComposedPipeline.from_pretrained`` — a faithful stand-in for
    diffusers' passed-component path — never reads it."""
    COMPOSED_SETUPS.clear()
    cache_dir = tmp_path / "cas"
    cache_dir.mkdir()
    blobs = BlobHost(tmp_path)
    try:
        base_snap = _base_snapshot(blobs)
        vae_snap = _vae_snapshot(blobs)
        subst = [pb.ModelBinding(
            slot="pipeline", ref=BASE_REF, components={"vae": VAE_REF},
        )]
        snaps = {BASE_REF: base_snap, VAE_REF: vae_snap}
        with hub_double(cache_dir=cache_dir) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)

            res = _run(conn, "r-subst", subst, snaps)
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            out = msgspec.msgpack.decode(res.inline, type=EchoOut).response
            # The load itself is unchanged: base bytes read, override injected.
            assert "base=base-transformer" in out
            assert "vae=override-vae" in out
            assert "injected=True" in out

        # THE assertion: the overridden component's bytes were never fetched.
        present = _cas_blob_digests(cache_dir)
        assert hashlib.sha256(BASE_TRANSFORMER_BYTES).hexdigest() in present, (
            "the base's own transformer must still be fetched")
        assert hashlib.sha256(b"override-vae").hexdigest() in present, (
            "the override's tree must still be fetched")
        assert hashlib.sha256(BASE_VAE_BYTES).hexdigest() not in present, (
            "th#1330 B2: the base's vae/ was downloaded and discarded")

        # ...and the narrowed tree is keyed as narrowed, never under the bare
        # digest reserved for a complete snapshot.
        trees = _materialized_base_trees(cache_dir)
        base_tree = [t for t in trees if t.name.startswith("snap-composed-base")]
        assert len(base_tree) == 1, trees
        assert base_tree[0].name != "snap-composed-base", (
            "a partial tree must not occupy the complete snapshot's name")
        assert (base_tree[0] / "transformer" / "weights.txt").exists()
        assert (base_tree[0] / "model_index.json").exists(), (
            "root config files are what validate the override's component name")
        assert not (base_tree[0] / "vae").exists()
    finally:
        blobs.shutdown()


def test_flat_dispatch_after_an_override_still_gets_the_full_base(tmp_path) -> None:
    """The narrowed tree must not poison the complete one: an override run
    followed by a FLAT run of the same base ref loads the base's own vae."""
    COMPOSED_SETUPS.clear()
    cache_dir = tmp_path / "cas"
    cache_dir.mkdir()
    blobs = BlobHost(tmp_path)
    try:
        base_snap = _base_snapshot(blobs)
        vae_snap = _vae_snapshot(blobs)
        subst = [pb.ModelBinding(
            slot="pipeline", ref=BASE_REF, components={"vae": VAE_REF},
        )]
        flat = [pb.ModelBinding(slot="pipeline", ref=BASE_REF)]
        with hub_double(cache_dir=cache_dir) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)

            res = _run(conn, "r-subst", subst,
                       {BASE_REF: base_snap, VAE_REF: vae_snap})
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            assert "vae=override-vae" in msgspec.msgpack.decode(
                res.inline, type=EchoOut).response

            res = _run(conn, "r-flat", flat, {BASE_REF: base_snap})
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            out = msgspec.msgpack.decode(res.inline, type=EchoOut).response
            assert "vae=base-vae-payload" in out, out
            assert "injected=False" in out
    finally:
        blobs.shutdown()


# ---------------------------------------------------------------------------
# Scope arithmetic
# ---------------------------------------------------------------------------


def test_exclude_drops_only_the_named_subfolder() -> None:
    paths = [
        "model_index.json",
        "transformer/weights.safetensors",
        "text_encoder/model.safetensors",
        "text_encoder_2/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    kept = select_component_paths(paths, (), exclude=("text_encoder",))
    assert "text_encoder/model.safetensors" not in kept
    # A prefix collision is not a match: text_encoder_2 is its own component.
    assert "text_encoder_2/model.safetensors" in kept
    assert "model_index.json" in kept
    assert "vae/diffusion_pytorch_model.safetensors" in kept

    # Unchanged when nothing is asked for.
    assert select_component_paths(paths, (), ()) == set(paths)
    # Positive and negative scopes compose.
    both = select_component_paths(
        paths, ("text_encoder", "vae"), exclude=("vae",))
    assert both == {"model_index.json", "text_encoder/model.safetensors"}


def test_narrowed_trees_key_apart() -> None:
    full = snapshot_dir_key("d0", (), ())
    excl = snapshot_dir_key("d0", (), ("text_encoder",))
    excl2 = snapshot_dir_key("d0", (), ("vae",))
    comps = snapshot_dir_key("d0", ("vae",), ())
    assert full == "d0"
    assert len({full, excl, excl2, comps}) == 4
    assert excl.startswith("d0__x")
