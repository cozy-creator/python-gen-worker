"""Content-keyed shared components + transformer lanes (gw#479), against REAL
tiny diffusers pipelines and the REAL executor injection path.

Two Hub-style refs whose ``vae`` files are byte-identical (same blake3 set in
their wire snapshots) must share ONE in-memory vae; each lane's exclusive
``unet`` is an independent residency entry that LRU demote/promote swaps
WITHOUT touching the shared module. A dtype mismatch must NOT share.

CUDA required for the residency-move tests (residency moves real memory,
gw#417); the digest/plan/routing tests run anywhere.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import msgspec
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DModel

from gen_worker.api.binding import HF
from gen_worker.api.decorators import Resources
from gen_worker.api.errors import ValidationError
from gen_worker.executor import Executor, ModelStore
from gen_worker.models.cozy_cas import _blake3_file
from gen_worker.models.residency import Tier
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

_GiB = 1024 ** 3
_MiB = 1024 ** 2

_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="residency moves real memory between VRAM/RAM (gw#417); needs CUDA",
)


class TinyLanePipe(DiffusionPipeline):
    """Real DiffusionPipeline subclass: exclusive ``unet`` + shared ``vae``."""

    def __init__(self, unet, vae, scheduler) -> None:
        super().__init__()
        self.register_modules(unet=unet, vae=vae, scheduler=scheduler)


def _tiny_unet(channels: int, seed: int) -> UNet2DModel:
    torch.manual_seed(seed)
    return UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3, layers_per_block=1,
        block_out_channels=(channels, channels), norm_num_groups=8,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )


def _tiny_vae(seed: int = 7) -> AutoencoderKL:
    torch.manual_seed(seed)
    return AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(32,), layers_per_block=1, norm_num_groups=8,
    )


def _save_lane_repo(path: Path, *, unet_seed: int, channels: int = 384) -> None:
    TinyLanePipe(
        unet=_tiny_unet(channels, unet_seed), vae=_tiny_vae(),
        scheduler=DDPMScheduler(num_train_timesteps=10),
    ).save_pretrained(str(path))


def _snapshot_pb(root: Path) -> pb.Snapshot:
    """Wire snapshot with REAL per-file blake3 digests, like the hub sends."""
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        files.append(pb.SnapshotFile(
            path=str(p.relative_to(root)),
            size_bytes=p.stat().st_size,
            blake3=_blake3_file(p),
        ))
    return pb.Snapshot(digest=f"snap-{root.name}", files=files)


@pytest.fixture(scope="module")
def lane_repos(tmp_path_factory) -> dict:
    """repo-a and repo-b: different unets, BYTE-IDENTICAL vae (copied)."""
    import shutil

    root = tmp_path_factory.mktemp("lane-repos")
    a, b = root / "repo-a", root / "repo-b"
    _save_lane_repo(a, unet_seed=1)
    _save_lane_repo(b, unet_seed=2)
    # Byte-identical shared component: replace b's vae with a's bytes.
    shutil.rmtree(b / "vae")
    shutil.copytree(a / "vae", b / "vae")
    assert (
        _blake3_file(next((a / "vae").glob("*.safetensors")))
        == _blake3_file(next((b / "vae").glob("*.safetensors")))
    )
    return {"a": a, "b": b}


class _In(msgspec.Struct):
    x: str


def _route(payload: "_In"):
    return ("a",) if payload.x == "a" else ("b",)


def _lane_spec(cls: type, models: dict, route=None) -> EndpointSpec:
    return EndpointSpec(
        name="lanes", method=cls.run, kind="inference", payload_type=_In,
        output_mode="single", cls=cls, attr_name="run", models=models,
        resources=Resources(), route=route,
    )


def _executor(specs, tmp_path: Path, budget_bytes: int, sent: list,
              repos: dict) -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    store = ModelStore(_send, cache_dir=tmp_path, vram_budget_bytes=budget_bytes)

    async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
        path = repos[ref]
        store.residency.track_disk(ref, path)
        return path

    store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]
    for ref, path in repos.items():
        store._snapshots[ref] = _snapshot_pb(path)
    return Executor(specs, _send, store=store)


class _Lanes:
    def setup(self, a: TinyLanePipe, b: TinyLanePipe) -> None:
        self.a, self.b = a, b

    def run(self, ctx, payload: _In):  # pragma: no cover
        return payload


# --------------------------------------------------------------------------- #
# Content identity plumbing (no CUDA needed)
# --------------------------------------------------------------------------- #


def test_component_digests_group_by_subfolder(tmp_path, lane_repos) -> None:
    sent: list = []
    repos = {"acme/a": lane_repos["a"], "acme/b": lane_repos["b"]}
    ex = _executor([], tmp_path, 4 * _GiB, sent, repos)

    da = ex.store.component_digests("acme/a")
    db = ex.store.component_digests("acme/b")
    assert set(da) == {"", "unet", "vae", "scheduler"}
    assert da["vae"] == db["vae"]          # byte-identical -> same content id
    assert da["unet"] != db["unet"]        # different weights -> different id
    assert da[""] != "" and "unet" in ex.store.component_sizes("acme/a")
    assert ex.store.component_digests("acme/unknown") == {}


def test_share_plan_only_covers_byte_identical_components(tmp_path, lane_repos) -> None:
    repos = {"acme/a": lane_repos["a"], "acme/b": lane_repos["b"]}
    spec = _lane_spec(_Lanes, {"a": HF("acme/a"), "b": HF("acme/b")})
    ex = _executor([spec], tmp_path, 4 * _GiB, [], repos)

    hints = {"a": TinyLanePipe, "b": TinyLanePipe}
    paths = {"a": str(lane_repos["a"]), "b": str(lane_repos["b"])}
    plan = ex._component_share_plan(spec, paths, hints)
    assert plan is not None
    # vae + scheduler are byte-identical across the repos; unet never shares.
    assert set(plan["a"]) == set(plan["b"]) == {"vae", "scheduler"}
    assert plan["a"]["vae"] == plan["b"]["vae"]

    # dtype mismatch: same bytes, different load facts -> no plan.
    spec2 = _lane_spec(_Lanes, {"a": HF("acme/a", dtype="fp16"), "b": HF("acme/b")})
    assert ex._component_share_plan(spec2, paths, hints) is None


def test_canonical_config_digest_folds_save_era_noise(tmp_path) -> None:
    """gw#479 live lesson (qwen fp8 casts): byte-identical weights, configs
    differing only in save-era serialization must share; real field changes
    must not."""
    import json

    from gen_worker.models.config_identity import canonical_json_digest

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()

    base = {"act_fn": "silu", "torch_dtype": "float32", "sample_size": 32,
            "_diffusers_version": "0.34.0.dev0", "latents_mean": None}
    saved_by_newer = {"act_fn": "silu", "dtype": "float32", "sample_size": 32,
                      "_diffusers_version": "0.36.0.dev0"}
    (a / "cfg.json").write_text(json.dumps(base))
    (b / "cfg.json").write_text(json.dumps(saved_by_newer, indent=2))
    da, db = canonical_json_digest(a / "cfg.json"), canonical_json_digest(b / "cfg.json")
    assert da and da == db                       # provenance/null/dtype-rename fold

    real_change = dict(saved_by_newer, sample_size=64)
    (b / "cfg2.json").write_text(json.dumps(real_change))
    assert canonical_json_digest(b / "cfg2.json") != da   # real field differs

    # transformers config.json: explicit class defaults fold out via AutoConfig.
    gpt_a = {"model_type": "gpt2", "n_layer": 2, "n_head": 2, "n_embd": 8}
    gpt_b = {"model_type": "gpt2", "n_layer": 2, "n_head": 2, "n_embd": 8,
             "resid_pdrop": 0.1, "_name_or_path": "/scratch/x",
             "transformers_version": "4.53.1"}  # resid_pdrop 0.1 IS the default
    (a / "config.json").write_text(json.dumps(gpt_a))
    (b / "config.json").write_text(json.dumps(gpt_b))
    ca, cb = canonical_json_digest(a / "config.json"), canonical_json_digest(b / "config.json")
    assert ca and ca == cb


def test_share_plan_survives_config_provenance_noise(tmp_path, lane_repos) -> None:
    """Repo B's vae config rewritten by a 'newer library' (provenance stamp +
    reindented + explicit null): the vae must STILL share."""
    import json
    import shutil

    src_a, src_b = lane_repos["a"], lane_repos["b"]
    a = tmp_path / "repo-a"
    b = tmp_path / "repo-b"
    shutil.copytree(src_a, a)
    shutil.copytree(src_b, b)
    cfg_path = b / "vae" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["_diffusers_version"] = "9.99.9"
    cfg["latents_mean"] = None
    cfg_path.write_text(json.dumps(cfg, indent=4, sort_keys=True))

    repos = {"acme/a": a, "acme/b": b}
    spec = _lane_spec(_Lanes, {"a": HF("acme/a"), "b": HF("acme/b")})
    ex = _executor([spec], tmp_path / "cache", 4 * _GiB, [], repos)
    hints = {"a": TinyLanePipe, "b": TinyLanePipe}
    paths = {"a": str(a), "b": str(b)}
    plan = ex._component_share_plan(spec, paths, hints)
    assert plan is not None
    assert "vae" in plan["a"] and "vae" in plan["b"]     # canonical digests folded the noise
    assert plan["a"]["vae"] == plan["b"]["vae"]
    assert "unet" not in plan["a"]                        # weights still gate


def test_routed_slots_validation(tmp_path, lane_repos) -> None:
    repos = {"acme/a": lane_repos["a"], "acme/b": lane_repos["b"]}
    spec = _lane_spec(
        _Lanes, {"a": HF("acme/a"), "b": HF("acme/b")}, route=_route)
    ex = _executor([spec], tmp_path, 4 * _GiB, [], repos)

    assert ex._routed_slots(spec, _In(x="a")) == ["a"]
    assert ex._routed_slots(spec, _In(x="edit")) == ["b"]

    bad = _lane_spec(
        _Lanes, {"a": HF("acme/a"), "b": HF("acme/b")},
        route=lambda p: ("nope",))
    with pytest.raises(ValidationError):
        ex._routed_slots(bad, _In(x="a"))
    empty = _lane_spec(
        _Lanes, {"a": HF("acme/a"), "b": HF("acme/b")}, route=lambda p: ())
    with pytest.raises(ValidationError):
        ex._routed_slots(empty, _In(x="a"))
    # No route= -> every slot (unchanged single-lane behavior).
    plain = _lane_spec(_Lanes, {"a": HF("acme/a"), "b": HF("acme/b")})
    assert ex._routed_slots(plain, _In(x="a")) == ["a", "b"]


def test_route_decorator_validation() -> None:
    from gen_worker import Hub, endpoint

    with pytest.raises(ValueError):
        @endpoint(model=Hub("o/r"), route=lambda p: ("m",))
        class OneSlot:
            def setup(self, m: str) -> None: ...
            def run(self, ctx, payload: _In) -> _In: ...

    @endpoint(models={"a": Hub("o/a"), "b": Hub("o/b")}, route=_route)
    class TwoSlots:
        def setup(self, a: str, b: str) -> None: ...
        def run(self, ctx, payload: _In) -> _In: ...

    assert TwoSlots.__gen_worker_endpoint__.route is _route


# --------------------------------------------------------------------------- #
# Real loads: sharing, accounting, lane swap (CUDA)
# --------------------------------------------------------------------------- #


@_cuda
def test_lanes_share_one_vae_instance_and_count_it_once(tmp_path, lane_repos) -> None:
    sent: list = []
    repos = {"acme/a": lane_repos["a"], "acme/b": lane_repos["b"]}
    spec = _lane_spec(
        _Lanes, {"a": HF("acme/a"), "b": HF("acme/b")}, route=_route)

    async def _run() -> None:
        ex = _executor([spec], tmp_path, 4 * _GiB, sent, repos)
        inst = await ex.ensure_setup(spec)
        res = ex.store.residency

        # Object identity: ONE vae instance wired into both lane pipelines.
        assert inst.a.vae is inst.b.vae
        assert inst.a.unet is not inst.b.unet

        stats = res.shared_stats()
        assert stats["hits"] >= 1            # lane b reused lane a's modules
        vae_entries = [e for e in stats["entries"] if "vae" in e["ref"]]
        assert len(vae_entries) == 1         # one shared vae entry, counted once
        assert vae_entries[0]["refcount"] == 2

        # Memory accounting: lane entries carry ~the exclusive unet (148MB
        # fp32), NOT unet+vae; the shared vae is booked once on its own entry.
        assert res.vram_bytes("acme/a") == pytest.approx(148 * _MiB, rel=0.3)
        assert res.vram_bytes("acme/b") == pytest.approx(148 * _MiB, rel=0.3)
        tracked = 4 * _GiB - res.free_vram_bytes()
        vae_bytes = vae_entries[0]["vram_bytes"]
        assert vae_bytes > 0
        assert tracked == pytest.approx(
            res.vram_bytes("acme/a") + res.vram_bytes("acme/b") + vae_bytes
            + sum(e["vram_bytes"] for e in stats["entries"] if "vae" not in e["ref"]),
            rel=0.05,
        )

    asyncio.run(_run())


@_cuda
def test_lane_swap_moves_only_the_exclusive_module(tmp_path, lane_repos) -> None:
    """Tight budget: both 148MB unets cannot be VRAM-booked at once (2GiB
    make_room margin). Setup lands lane b resident + lane a demoted; a routed
    promote of lane a swaps b out. The shared vae NEVER moves."""
    sent: list = []
    repos = {"acme/a": lane_repos["a"], "acme/b": lane_repos["b"]}
    spec = _lane_spec(
        _Lanes, {"a": HF("acme/a"), "b": HF("acme/b")}, route=_route)

    def _dev(module) -> str:
        return next(module.parameters()).device.type

    async def _run() -> None:
        ex = _executor([spec], tmp_path, int(2.3 * _GiB), sent, repos)
        inst = await ex.ensure_setup(spec)
        res = ex.store.residency

        # Swap-mode admission: loading lane b demoted lane a's unet — and
        # ONLY the unet; the shared vae stayed put on cuda.
        assert res.tier("acme/a") is Tier.RAM
        assert res.tier("acme/b") is Tier.VRAM
        assert _dev(inst.a.unet) == "cpu"
        assert _dev(inst.b.unet) == "cuda"
        assert inst.a.vae is inst.b.vae and _dev(inst.a.vae) == "cuda"

        # Routed request for lane a: promote a, LRU-demote b — vae untouched.
        again = await ex.ensure_setup(spec, promote_slots=["a"])
        assert again is inst
        assert res.tier("acme/a") is Tier.VRAM
        assert res.tier("acme/b") is Tier.RAM
        assert _dev(inst.a.unet) == "cuda"
        assert _dev(inst.b.unet) == "cpu"
        assert _dev(inst.a.vae) == "cuda"

        # Swap telemetry (gw#479): counts + durations recorded per lane.
        stats = res.transition_stats()
        assert stats["acme/a"]["demotes"] >= 1
        assert stats["acme/a"]["promotes"] >= 1
        assert stats["acme/b"]["demotes"] >= 1

    asyncio.run(_run())
    # Promote/demote ModelEvents carry duration_ms (the hub-visible signal).
    durs = [m.model_event.duration_ms for m in sent
            if m.WhichOneof("msg") == "model_event"
            and m.model_event.state in (pb.MODEL_STATE_IN_RAM, pb.MODEL_STATE_IN_VRAM)]
    assert durs, "expected residency transition events"


@_cuda
def test_dtype_mismatch_loads_monolithically(tmp_path, lane_repos) -> None:
    sent: list = []
    repos = {"acme/a": lane_repos["a"], "acme/b": lane_repos["b"]}
    spec = _lane_spec(
        _Lanes, {"a": HF("acme/a", dtype="fp16"), "b": HF("acme/b")})

    async def _run() -> None:
        ex = _executor([spec], tmp_path, 4 * _GiB, sent, repos)
        inst = await ex.ensure_setup(spec)
        res = ex.store.residency

        assert inst.a.vae is not inst.b.vae   # no content-key match -> no share
        assert res.shared_stats()["entries"] == []
        # Monolithic entries own whole pipelines (today's behavior).
        assert res.obj("acme/a") is inst.a
        assert res.obj("acme/b") is inst.b

    asyncio.run(_run())


@_cuda
def test_vacate_releases_shared_holds(tmp_path, lane_repos) -> None:
    sent: list = []
    repos = {"acme/a": lane_repos["a"], "acme/b": lane_repos["b"]}
    spec = _lane_spec(
        _Lanes, {"a": HF("acme/a"), "b": HF("acme/b")}, route=_route)

    async def _run() -> None:
        ex = _executor([spec], tmp_path, 4 * _GiB, sent, repos)
        await ex.ensure_setup(spec)
        res = ex.store.residency
        assert res.shared_stats()["entries"]

        rec = ex._classes[spec.instance_key]
        await ex._vacate_record(rec)
        # All shared holds released and unreferenced entries drained.
        assert res.shared_stats()["entries"] == []
        assert rec.shared_keys == []

    asyncio.run(_run())
