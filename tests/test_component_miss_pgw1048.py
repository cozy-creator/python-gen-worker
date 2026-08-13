"""pgw#1048: a composition the tree cannot satisfy is a TYPED refusal, not a
retry-loop on a nameless OSError.

pgw#1047 measured the defect on a paid L40S: the hub's seed path dropped the
sdxl deploy's ``vae`` component override while th#1711 had already
narrowed the outbound snapshot to exclude ``vae/``, so the pod fetched a
vae-less tree, loaded the composition with nothing substituted, and diffusers
raised

    OSError: Error no file named config.json found in directory
             /tmp/tensorhub-cache/cas/snapshots/sha256:32fa2ba6…

— naming neither the component nor the cause. The rotation preloader then
retried that deterministic condition on every desired-set generation until the
hub reaped the pod, nine minutes later.

The root cause was hub-side and is fixed. This is the worker's own half: the
failure SHAPE. Every assertion below runs the REAL load path.

  1. real tiny SDXL (unet = a real ``streaming_w8a8_cast`` product carrying the
     production ``quantization_config`` block), real ``load_component_override``,
     real ``provision.load_slot`` — the pgw#1047 ``repro_1047_w8a8.py`` fixture;
  2. the refusal reaches the hub through the REAL activity sink the worker
     transport binds, not an in-process spy;
  3. the real executor + real ``Preloader`` classify it terminal for the
     dispatched identity — no retry on a new desired-set generation — while an
     ordinary stage failure still retries. The retry decision keys on the
     identity's own bytes, never on a duration (DESIGN-RULINGS §2).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from gen_worker import RequestContext, Slot, activity as activity_mod  # noqa: E402
from gen_worker import endpoint, worker_function  # noqa: E402
from gen_worker.executor import Executor  # noqa: E402
from gen_worker.models import provision  # noqa: E402
from gen_worker.models.loading import (  # noqa: E402
    ComponentSubstitutionError,
    assert_composition_satisfiable,
    load_component_override,
)
from gen_worker.pb import worker_scheduler_pb2 as pb  # noqa: E402
from gen_worker.registry import extract_specs  # noqa: E402

import msgspec  # noqa: E402
from gen_worker.models import store as store_mod

_GiB = 1024 ** 3


# ---------------------------------------------------------------------------
# The pgw#1047 fixture: a production-shaped tiny SDXL, with and without vae/
# ---------------------------------------------------------------------------

UNET_CFG = dict(
    block_out_channels=(32, 64),
    layers_per_block=1,
    sample_size=8,
    in_channels=4,
    out_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    attention_head_dim=(2, 4),
    use_linear_projection=True,
    cross_attention_dim=32,
    norm_num_groups=4,
    addition_embed_type="text_time",
    addition_time_embed_dim=8,
    projection_class_embeddings_input_dim=24,
    transformer_layers_per_block=1,
)


def _tiny_vae() -> Any:
    from diffusers import AutoencoderKL

    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,), layers_per_block=1,
        latent_channels=4, norm_num_groups=4, sample_size=8,
    )


def build_sdxl_tree(dst: Path, *, with_vae: bool) -> Path:
    """A real SDXL-layout snapshot whose unet is a REAL w8a8 artifact — the
    production lane pgw#1047 died on, not the plain from_pretrained one."""
    from diffusers import EulerDiscreteScheduler, UNet2DConditionModel
    from safetensors.torch import save_file
    from transformers import (
        CLIPTextConfig,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )

    from gen_worker.convert.writer import streaming_w8a8_cast

    torch.manual_seed(0)
    unet = UNet2DConditionModel(**UNET_CFG).to(torch.float16)
    src = dst.parent / f"{dst.name}-unet-src"
    src.mkdir(parents=True, exist_ok=True)
    save_file(dict(unet.state_dict()), str(src / "model.safetensors"))
    out = streaming_w8a8_cast(
        src / "model.safetensors", dst / "unet",
        output_stem="diffusion_pytorch_model")
    assert int(out["converted_count"]) > 0, out
    cfg = dict(unet.config)
    cfg["_class_name"] = "UNet2DConditionModel"
    cfg["_diffusers_version"] = "0.39.0"
    # the modelopt block the production unet/config.json carries
    cfg["quantization_config"] = {
        "ignore": ["conv_in", "conv_out"],
        "quant_algo": None,
        "producer": {"name": "modelopt", "version": "0.33.1"},
    }
    (dst / "unet" / "config.json").write_text(json.dumps(cfg))

    if with_vae:
        _tiny_vae().to(torch.float16).save_pretrained(str(dst / "vae"))

    te_cfg = CLIPTextConfig(
        bos_token_id=0, eos_token_id=2, hidden_size=32,
        intermediate_size=64, layer_norm_eps=1e-5,
        num_attention_heads=4, num_hidden_layers=2,
        pad_token_id=1, vocab_size=64, projection_dim=32,
    )
    CLIPTextModel(te_cfg).to(torch.float16).save_pretrained(
        str(dst / "text_encoder"))
    CLIPTextModelWithProjection(te_cfg).to(torch.float16).save_pretrained(
        str(dst / "text_encoder_2"))

    tok_dir = dst.parent / f"{dst.name}-tok-src"
    tok_dir.mkdir(parents=True, exist_ok=True)
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1, "!": 2}
    for i in range(3, 64):
        vocab[f"tok{i}</w>"] = i
    (tok_dir / "vocab.json").write_text(json.dumps(vocab))
    (tok_dir / "merges.txt").write_text("#version: 0.2\n")
    tok = CLIPTokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    tok.save_pretrained(str(dst / "tokenizer"))
    tok.save_pretrained(str(dst / "tokenizer_2"))

    EulerDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        steps_offset=1, timestep_spacing="leading",
    ).save_pretrained(str(dst / "scheduler"))

    (dst / "model_index.json").write_text(json.dumps({
        "_class_name": "StableDiffusionXLPipeline",
        "_diffusers_version": "0.39.0",
        "feature_extractor": [None, None],
        "image_encoder": [None, None],
        "force_zeros_for_empty_prompt": True,
        "scheduler": ["diffusers", "EulerDiscreteScheduler"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "CLIPTokenizer"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    # the non-component root dir the real snapshot carries
    (dst / "degradation_gate").mkdir(exist_ok=True)
    (dst / "degradation_gate" / "report.json").write_text("{}")
    return dst


@pytest.fixture(scope="module")
def trees(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    work = tmp_path_factory.mktemp("pgw1048")
    full = build_sdxl_tree(work / "snap-full", with_vae=True)
    narrowed = build_sdxl_tree(work / "snap-novae", with_vae=False)
    override = work / "override-vae"
    torch.manual_seed(1)
    # the fp16-fix mirror is fp32-stored, ROOT layout
    _tiny_vae().save_pretrained(str(override))
    return {"full": full, "narrowed": narrowed, "override": override}


class _Binding:
    """The dispatched binding shape load_slot reads (sdxl deploy: a vae
    component override)."""

    dtype = ""
    storage_dtype = ""
    flavor = "fp8-w8a8"
    component_overrides = (("vae", "tensorhub/sdxl-vae-fp16-fix:prod"),)


class _HubEvents:
    """The REAL activity sink the worker transport installs, drained after
    the load — these assertions read exactly the ActivityUpdates a hub would
    receive."""

    def __init__(self, loop: Any = None) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self._own_loop = loop is None
        self.loop = loop or asyncio.new_event_loop()

    def __enter__(self) -> "_HubEvents":
        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        activity_mod.bind_sink(_send, self.loop)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._own_loop:
            self.loop.run_until_complete(asyncio.sleep(0.02))
        activity_mod.reset_for_tests()
        if self._own_loop:
            self.loop.close()

    def of_kind(self, kind: str) -> List[pb.ActivityUpdate]:
        return [
            m.activity_update for m in self.sent
            if m.WhichOneof("msg") == "activity_update"
            and m.activity_update.kind == kind
        ]


def _load(base: Path, override: Path | None) -> Any:
    from diffusers import StableDiffusionXLPipeline

    injected = None
    if override is not None:
        injected = {"vae": load_component_override(
            str(base), "vae", str(override), dtype="")}
    return provision.load_slot(
        StableDiffusionXLPipeline, str(base), binding=_Binding(),
        slot="pipeline", ref="tensorhub/cyberrealistic-xl:fp8-linearonly-review",
        mode="auto", components=injected, device="cpu").obj


# ---------------------------------------------------------------------------
# 1. the real load path
# ---------------------------------------------------------------------------


def test_complete_tree_loads(trees: Dict[str, Path]) -> None:
    """Control: the guard must not refuse a composition the tree satisfies."""
    pipe = _load(trees["full"], trees["override"])
    assert type(pipe).__name__ == "StableDiffusionXLPipeline"


def test_narrowed_tree_with_the_override_injected_loads(
    trees: Dict[str, Path],
) -> None:
    """The th#1711 narrowing is CORRECT when the binding carries the override:
    the base's vae/ is absent because the substitute supplies it."""
    pipe = _load(trees["narrowed"], trees["override"])
    assert type(pipe).__name__ == "StableDiffusionXLPipeline"
    assert type(pipe.vae).__name__ == "AutoencoderKL"


def test_narrowed_tree_without_injection_refuses_typed(
    trees: Dict[str, Path],
) -> None:
    """The pgw#1047 production shape: narrowed tree, nothing injected.

    RED on the old code — a bare ``OSError: Error no file named config.json
    found in directory <snapshot root>``, naming neither the component nor the
    cause, and retried forever by the caller."""
    with pytest.raises(ComponentSubstitutionError) as excinfo:
        _load(trees["narrowed"], None)
    exc = excinfo.value
    assert exc.missing == ("vae",)
    assert "vae" in exc.expected and "unet" in exc.expected
    assert exc.injected == ()
    assert exc.tree == str(trees["narrowed"])
    text = str(exc)
    # It names the component, the tree, the injected set, and the suspected
    # narrowing — the four things the raw OSError did not.
    assert "'vae'" in text
    assert str(trees["narrowed"]) in text
    assert "nothing" in text
    assert "th#1711" in text and "th#1715" in text
    # and it is not the old shape
    assert "no file named config.json" not in text


def test_the_refusal_reaches_the_hub(trees: Dict[str, Path]) -> None:
    """Worker-errors-to-hub: the verdict is an ActivityUpdate on the real
    sink, not a logger line the provider has no API to read."""
    with _HubEvents() as hub:
        with pytest.raises(ComponentSubstitutionError):
            _load(trees["narrowed"], None)
    events = hub.of_kind(activity_mod.KIND_COMPONENT_MISS)
    assert len(events) == 1, [e.kind for e in hub.of_kind("")] or hub.sent
    ev = events[0]
    assert ev.phase == "refused"
    assert ev.state == pb.ActivityState.ACTIVITY_STATE_COMPLETED
    assert "missing=vae" in ev.detail
    assert "injected=<nothing>" in ev.detail
    assert str(trees["narrowed"]) in ev.detail
    assert "StableDiffusionXLPipeline" in ev.detail
    assert "tensorhub/cyberrealistic-xl:fp8-linearonly-review" in ev.detail


# ---------------------------------------------------------------------------
# 2. the lanes the guard must NOT judge
# ---------------------------------------------------------------------------


class _DeclaredPipeline:
    """A pipeline class whose signature declares both components, so the guard
    judges the whole index (see ``_pipeline_component_names``)."""

    def __init__(self, unet: Any, vae: Any) -> None:
        self.unet = unet
        self.vae = vae


class _VaeOnlyPipeline:
    """Its signature names ONE component; the index names two."""

    def __init__(self, vae: Any) -> None:
        self.vae = vae


def test_indexless_layouts_are_not_judged(tmp_path: Path) -> None:
    """Single-file checkpoints and transformers trees have no model_index.json,
    so there is no composition to satisfy — and root-layout quantized
    artifacts detect only in the absence of one."""
    tree = tmp_path / "singlefile"
    tree.mkdir()
    (tree / "model.safetensors").write_bytes(b"not a real checkpoint")
    assert_composition_satisfiable(_DeclaredPipeline, tree)  # must not raise


def test_gguf_snapshots_are_not_judged(tmp_path: Path) -> None:
    """A gguf denoiser is a loose ``.gguf`` file that ``load_gguf_pipeline``
    constructs and hands in — the component dir the index names legitimately
    does not exist."""
    tree = tmp_path / "gguf"
    (tree / "vae").mkdir(parents=True)
    (tree / "vae" / "config.json").write_text("{}")
    (tree / "model-Q4_K_S.gguf").write_bytes(b"GGUF")
    (tree / "model_index.json").write_text(json.dumps({
        "_class_name": "FluxPipeline",
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    from gen_worker.models.loading import detect_gguf_snapshot

    assert detect_gguf_snapshot(tree) is not None, "fixture is not a gguf tree"
    assert_composition_satisfiable(_DeclaredPipeline, tree)  # must not raise


def test_an_empty_component_dir_is_a_miss(tmp_path: Path) -> None:
    """A dir the narrowing left behind with no files in it is the same hole as
    an absent one — diffusers reports it identically."""
    tree = tmp_path / "empty-comp"
    (tree / "unet").mkdir(parents=True)
    (tree / "unet" / "config.json").write_text("{}")
    (tree / "vae").mkdir()
    (tree / "model_index.json").write_text(json.dumps({
        "_class_name": "StableDiffusionXLPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    with pytest.raises(ComponentSubstitutionError) as excinfo:
        assert_composition_satisfiable(_DeclaredPipeline, tree)
    assert excinfo.value.missing == ("vae",)


def test_a_component_the_class_does_not_declare_is_not_judged(
    tmp_path: Path,
) -> None:
    """diffusers' own rule: ``from_pretrained`` constructs the parts the
    signature names. A component the index names and the class does not is one
    the load never touches, so refusing on it would refuse a load that works."""
    tree = tmp_path / "extra"
    (tree / "vae").mkdir(parents=True)
    (tree / "vae" / "config.json").write_text("{}")
    (tree / "model_index.json").write_text(json.dumps({
        "_class_name": "ComposedPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    # unet/ is absent, and _VaeOnlyPipeline never asks for it
    assert_composition_satisfiable(_VaeOnlyPipeline, tree)  # must not raise
    # the same tree against a class that DOES declare unet is a miss
    with pytest.raises(ComponentSubstitutionError) as excinfo:
        assert_composition_satisfiable(_DeclaredPipeline, tree)
    assert excinfo.value.missing == ("unet",)


def test_null_index_entries_are_not_components(tmp_path: Path) -> None:
    """``[null, null]`` entries (feature_extractor, image_encoder, safety
    checker) are declared ABSENT by the index itself."""
    tree = tmp_path / "nulls"
    (tree / "unet").mkdir(parents=True)
    (tree / "unet" / "config.json").write_text("{}")
    (tree / "model_index.json").write_text(json.dumps({
        "_class_name": "StableDiffusionXLPipeline",
        "feature_extractor": [None, None],
        "image_encoder": [None, None],
        "unet": ["diffusers", "UNet2DConditionModel"],
    }))
    assert_composition_satisfiable(_DeclaredPipeline, tree)  # must not raise


# ---------------------------------------------------------------------------
# 3. the rotation preloader: terminal for the dispatched identity, no timer
# ---------------------------------------------------------------------------


class _In(msgspec.Struct):
    prompt: str = ""
    model: str = ""


class _Out(msgspec.Struct):
    y: str = "ok"


class ComposedPipeline:
    """Stands in for the diffusers pipeline CLASS only: the guard reads the
    tree's model_index.json, so the class is not what is under test. Its
    from_pretrained would fail on the narrowed tree too — the point is that
    the refusal happens first, and typed."""

    def __init__(self, vae: Any) -> None:
        self.vae = vae

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "ComposedPipeline":
        vae = kwargs.get("vae")
        if vae is None:
            vae = (Path(path) / "vae" / "config.json").read_text()
        return cls(vae)

    def to(self, device: str) -> "ComposedPipeline":
        return self


@endpoint(models={"pipeline": Slot(ComposedPipeline, selected_by="model")})
class _Family:
    def setup(self, pipeline: ComposedPipeline) -> None:
        self.pipe = pipeline

    @worker_function()
    def generate(self, ctx: RequestContext, p: _In) -> _Out:
        return _Out()


_NARROWED_INDEX = json.dumps({
    "_class_name": "ComposedPipeline",
    "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
})


def _narrowed_writer(ref: str, p: Path) -> None:
    """The th#1711 outbound narrowing: the index names vae, the tree has none."""
    (p / "unet").mkdir(exist_ok=True)
    (p / "unet" / "config.json").write_text("{}")
    (p / "model_index.json").write_text(_NARROWED_INDEX)


def _executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
              writer: Any) -> tuple[Executor, List[pb.WorkerMessage]]:
    """The executor and the list its worker->hub sender writes into.

    ``Executor.ensure_setup`` binds THAT sender as the activity sink, so this
    list is the real hub channel — every ActivityUpdate below is one a hub
    would have received off the wire."""
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor(extract_specs(_Family), _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _fake_download(ref: str, **kwargs: Any) -> Path:
        snap = kwargs.get("snapshot")
        digest = str(getattr(snap, "snapshot_digest", "") or "")
        name = (digest.split(":", 1)[-1].strip().lower()
                or ref.replace("/", "_").replace(":", "_"))
        p = tmp_path / name
        p.mkdir(parents=True, exist_ok=True)
        writer(ref, p)
        return p

    import gen_worker.executor as ex_mod

    monkeypatch.setattr(store_mod, "ensure_local", _fake_download)
    ex.store.residency._vram_budget = 64 * _GiB  # both fit -> full background setup
    return ex, sent


def _updates(sent: List[pb.WorkerMessage], kind: str) -> List[pb.ActivityUpdate]:
    return [
        m.activity_update for m in sent
        if m.WhichOneof("msg") == "activity_update"
        and m.activity_update.kind == kind
    ]


def _snapshots(ref: str, digest: str) -> Dict[str, pb.Snapshot]:
    body = _NARROWED_INDEX.encode()
    return {ref: pb.Snapshot(digest=digest, files=[pb.SnapshotFile(
        path="model_index.json", size_bytes=len(body),
        digest="sha256:" + hashlib.sha256(body).hexdigest(),
        url="http://r2.invalid/presigned")])}


def _new_generation(pl: Any, ref: str, snaps: Dict[str, pb.Snapshot],
                    generation: int) -> None:
    """A fresh accepted desired set, exactly as lifecycle delivers it — then
    the background task it spawns is cancelled before it can run, so the test
    drives ``_pass()`` once per generation instead of racing the driver."""
    pl.update_desired([_instance(ref)], snaps, generation)
    task, pl._task = pl._task, None
    if task is not None:
        task.cancel()


def _instance(ref: str) -> pb.DesiredInstance:
    return pb.DesiredInstance(
        function_name="generate",
        models=[pb.ModelBinding(slot="pipeline", ref=ref)],
    )


def test_preloader_refuses_a_composition_miss_terminally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED on the old code: the raw OSError lands in ``self._failed``, which
    ``update_desired`` CLEARS, so every new desired-set generation re-runs a
    stage whose answer cannot change. pgw#1047 watched that burn nine minutes
    of a paid L40S.

    The retry decision keys on the dispatched identity's own bytes — not on a
    duration, and not on an attempt count (DESIGN-RULINGS §2)."""
    ex, sent = _executor(tmp_path, monkeypatch, _narrowed_writer)
    ck = "acme/narrowed-ckpt"
    snaps = _snapshots(ck, "d3" * 16)
    attempts: List[str] = []
    real = ex.ensure_desired_instance

    async def _counting(instance: Any, snapshots: Any) -> None:
        attempts.append(instance.function_name)
        await real(instance, snapshots)

    monkeypatch.setattr(ex, "ensure_desired_instance", _counting)

    async def _run() -> None:
        pl = ex.preloader
        for generation in (1, 2, 3):
            # A fresh desired-set generation: exactly what cleared _failed
            # and restarted the loop.
            _new_generation(pl, ck, snaps, generation)
            await asyncio.wait_for(pl._pass(), timeout=60)
        assert attempts == ["generate"], (
            "a DETERMINISTIC composition miss must be attempted ONCE for "
            f"this identity, not once per generation: {attempts}")
        assert await asyncio.wait_for(pl._pass(), timeout=60) is False
        await asyncio.sleep(0.02)  # let the sink's ship tasks land

        refused = [
            e for e in _updates(sent, activity_mod.KIND_ROTATION_PRELOAD)
            if e.phase == "stage_refused"
        ]
        assert len(refused) == 1, [
            (e.phase, e.detail)
            for e in _updates(sent, activity_mod.KIND_ROTATION_PRELOAD)]
        assert "ComponentSubstitutionError" in refused[0].detail
        assert "terminal for this dispatched identity" in refused[0].detail
        assert "'vae'" in refused[0].detail
        # the typed component_miss event travels the same channel
        misses = _updates(sent, activity_mod.KIND_COMPONENT_MISS)
        assert len(misses) == 1 and misses[0].phase == "refused"
        assert "missing=vae" in misses[0].detail

        # A DIFFERENT dispatched identity is retried on merit — the refusal is
        # keyed on the bytes, so a hub that repairs the binding self-heals with
        # no timer and no manual reset.
        other = "acme/other-ckpt"
        _new_generation(ex.preloader, other, _snapshots(other, "d4" * 16), 4)
        await asyncio.wait_for(ex.preloader._pass(), timeout=60)
        assert attempts == ["generate", "generate"]

    asyncio.run(_run())


def test_preloader_still_retries_an_ordinary_stage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The carve-out is for DETERMINISTIC misses only: a transient failure
    keeps the pgw#760 shape (parked for the generation, retried on the next
    desired set)."""
    ex, _sent = _executor(tmp_path, monkeypatch, _narrowed_writer)
    ck = "acme/flaky-ckpt"
    snaps = _snapshots(ck, "d5" * 16)
    attempts: List[str] = []

    async def _boom(instance: Any, snapshots: Any) -> None:
        attempts.append(instance.function_name)
        raise RuntimeError("transient: the blob store hiccupped")

    monkeypatch.setattr(ex, "ensure_desired_instance", _boom)

    async def _run() -> None:
        pl = ex.preloader
        for generation in (1, 2, 3):
            _new_generation(pl, ck, snaps, generation)
            await asyncio.wait_for(pl._pass(), timeout=60)
        assert attempts == ["generate"] * 3, attempts

    asyncio.run(_run())
