"""Per-SKU TensorRT engine artifacts (#390) — the second producer/consumer
on the compile-cache rails (gw#384 / th#569 / th#567).

One WEIGHT-STRIPPED engine serves every fine-tune of a model family: the
engine captures the optimized graph + tactics (~1% of a weight-full plan);
the consumer REFITs it with the weights of whatever family member is already
resident, then swaps the module's ``forward`` behind a guard. Same trust
model, storage, and delivery as inductor caches — cells live as flavors of
``_system/family-<family>``:

    _system/family-<f>#trt-<sku>-trt<maj.min>-<precision>

Artifact = deterministic ``.tar.gz``::

    metadata.json     key: family, sku, trt (FULL version), cuda, precision,
                      module, batch, shapes, io contract
    engine.plan       weight-stripped serialized engine (STRIP_PLAN | REFIT)
    refit_map.json    torch state_dict key -> engine weight name (+transform)

Key sensitivity: TensorRT plans deserialize ONLY under the exact library
version that built them (major.minor.patch.build) on a matching compute
capability, so ``verify`` exact-matches the FULL trt version + CUDA + SKU.
The flavor label carries maj.min for humans/selection; metadata is the
authority. Weight-stripped refit uses plain ``REFIT`` (``REFIT_IDENTICAL``
is documented undefined-behavior when refit weights differ from build-time
weights — the whole point here is that they differ).

Policy mirrors compile_cache: no verified artifact / key mismatch / missing
tensorrt lib => eager, never a stall. A failing engine call permanently
unwraps to the eager module.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tarfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .compile_cache import AdoptError, _clean_tarinfo, family_from_ref, sku_slug

logger = logging.getLogger(__name__)

METADATA_NAME = "metadata.json"
ENGINE_NAME = "engine.plan"
REFIT_MAP_NAME = "refit_map.json"
ARTIFACT_FORMAT = 1
_MARKER_ATTR = "_cozy_trt"
_MEMBERS = (METADATA_NAME, ENGINE_NAME, REFIT_MAP_NAME)


# ---------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------


def trt_version() -> str:
    try:
        import tensorrt as trt

        return str(trt.__version__)
    except Exception:
        return ""


def runtime_key() -> Dict[str, str]:
    """Consumer-side half of the engine key, probed from this process."""
    key = {"sku": "", "trt": trt_version(), "cuda": ""}
    try:
        import torch

        key["cuda"] = str(torch.version.cuda or "")
        if torch.cuda.is_available():
            key["sku"] = sku_slug(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return key


def trt_maj_min(version: str) -> str:
    parts = str(version or "").split(".")
    if len(parts) < 2 or not parts[0]:
        return ""
    return f"{parts[0]}.{parts[1]}"


def flavor_label(sku: str, version: str, precision: str) -> str:
    """``trt-rtx-4090-trt10.16-fp16``. Full version lives in metadata."""
    mm = trt_maj_min(version)
    if not sku or not mm or not precision:
        return ""
    return f"trt-{sku}-trt{mm}-{precision}"


def is_engine_ref(ref: str, family: str = "") -> bool:
    """True when ``ref`` names a TRT engine cell (optionally of one family)."""
    fam = family_from_ref(ref)
    if not fam or (family and fam != family):
        return False
    flavor = ref.split("#", 1)[1] if "#" in ref else ""
    return flavor.startswith("trt-")


def artifact_metadata(
    *,
    family: str,
    module: str,
    precision: str,
    batch: int,
    shapes: Iterable[Tuple[int, int]],
    inputs: List[Dict[str, Any]],
    source_ref: str = "",
    source_digest: str = "",
) -> Dict[str, Any]:
    return {
        "format": ARTIFACT_FORMAT,
        "kind": "trt-engine",
        **runtime_key(),
        "family": str(family or ""),
        "module": str(module or ""),
        "precision": str(precision or ""),
        "batch": int(batch),
        "shapes": [[int(w), int(h)] for w, h in shapes],
        "inputs": inputs,
        "source_ref": str(source_ref or ""),
        "source_digest": str(source_digest or ""),
    }


def verify(meta: Dict[str, Any], *, family: str = "") -> str:
    """'' when the artifact matches this runtime, else the mismatch reason.

    TRT plans are version-locked: the FULL library version must match, not
    just maj.min (deserialization fails otherwise — fail early and legibly).
    """
    if int(meta.get("format") or 0) != ARTIFACT_FORMAT:
        return f"format {meta.get('format')!r} != {ARTIFACT_FORMAT}"
    if str(meta.get("kind") or "") != "trt-engine":
        return f"kind {meta.get('kind')!r} != trt-engine"
    here = runtime_key()
    if not here["trt"]:
        return "tensorrt not installed"
    for field in ("sku", "trt", "cuda"):
        want, have = str(meta.get(field) or ""), here[field]
        if want != have:
            return f"{field} {want!r} != runtime {have!r}"
    want_fam = str(meta.get("family") or "")
    if family and want_fam and want_fam != family:
        return f"family {want_fam!r} != {family!r}"
    return ""


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------


def pack(content_dir: Path, out_path: Path, metadata: Dict[str, Any]) -> Path:
    """Deterministic artifact from ``content_dir`` holding ``engine.plan`` +
    ``refit_map.json`` (metadata is written from ``metadata``)."""
    content_dir = Path(content_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import gzip

    with open(out_path, "wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                meta_bytes = json.dumps(metadata, sort_keys=True, indent=1).encode()
                ti = _clean_tarinfo(tarfile.TarInfo(METADATA_NAME))
                ti.size = len(meta_bytes)
                tar.addfile(ti, io.BytesIO(meta_bytes))
                for name in (ENGINE_NAME, REFIT_MAP_NAME):
                    p = content_dir / name
                    ti = _clean_tarinfo(tarfile.TarInfo(name))
                    ti.size = p.stat().st_size
                    with open(p, "rb") as f:
                        tar.addfile(ti, f)
    return out_path


def unpack(artifact: Path, dest_root: Path) -> Dict[str, Any]:
    """Extract the fixed member set into ``dest_root``; returns metadata."""
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {}
    with tarfile.open(artifact, mode="r:*") as tar:
        for member in tar:
            name = member.name.lstrip("./")
            if name not in _MEMBERS or not member.isfile():
                raise ValueError(f"unexpected member in trt-engine artifact: {member.name!r}")
            src = tar.extractfile(member)
            assert src is not None
            data = src.read()
            if name == METADATA_NAME:
                meta = json.loads(data.decode())
                continue
            (dest_root / name).write_bytes(data)
    if not meta:
        raise ValueError(f"trt-engine artifact {artifact} has no {METADATA_NAME}")
    return meta


def unpack_metadata(artifact: Path) -> Dict[str, Any]:
    """Read ONLY metadata.json from an artifact (kind sniffing — cheap)."""
    with tarfile.open(artifact, mode="r:*") as tar:
        for member in tar:
            if member.name.lstrip("./") == METADATA_NAME and member.isfile():
                src = tar.extractfile(member)
                assert src is not None
                return json.loads(src.read().decode())
    raise ValueError(f"artifact {artifact} has no {METADATA_NAME}")


def find_artifact(root: Path) -> Optional[Path]:
    """The engine tarball inside a downloaded snapshot dir (or the file)."""
    root = Path(root)
    if root.is_file():
        return root
    return next(iter(sorted(root.rglob("*.tar.gz"))), None)


# ---------------------------------------------------------------------------
# Refit map — torch state_dict <-> engine weight names by VALUE identity
# ---------------------------------------------------------------------------


def _fingerprint(data: bytes, shape: Tuple[int, ...]) -> str:
    return f"{shape}:{hashlib.sha256(data).hexdigest()[:24]}"


def build_refit_map(
    initializers: Dict[str, Any], state_dict: Dict[str, Any]
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Match ONNX initializer names to torch state_dict keys by exact value
    (export renames + occasionally transposes weights; names are NOT a
    contract, bytes are). Returns ``(map_entries, unmatched_initializers)``
    where each entry is ``{"name", "key", "transform"}``.

    ``initializers`` maps engine/ONNX weight name -> numpy array;
    ``state_dict`` maps torch key -> tensor-like exposing ``.cpu().numpy()``.
    """
    import numpy as np

    by_value: Dict[str, str] = {}
    by_value_t: Dict[str, str] = {}
    for key, t in state_dict.items():
        arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
        arr = np.ascontiguousarray(arr)
        by_value.setdefault(_fingerprint(arr.tobytes(), arr.shape), key)
        if arr.ndim == 2:
            at = np.ascontiguousarray(arr.T)
            by_value_t.setdefault(_fingerprint(at.tobytes(), at.shape), key)

    entries: List[Dict[str, str]] = []
    unmatched: List[str] = []
    for name, arr in initializers.items():
        arr = np.ascontiguousarray(arr)
        fp = _fingerprint(arr.tobytes(), arr.shape)
        if fp in by_value:
            entries.append({"name": name, "key": by_value[fp], "transform": ""})
        elif fp in by_value_t:
            entries.append({"name": name, "key": by_value_t[fp], "transform": "transpose"})
        else:
            unmatched.append(name)
    return entries, unmatched


def refit_weights(state_dict: Dict[str, Any], entries: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    """Materialize ``{engine weight name: numpy array}`` from a state_dict
    through a refit map (applying recorded transforms)."""
    import numpy as np

    out: Dict[str, Any] = {}
    for e in entries:
        t = state_dict.get(e["key"])
        if t is None:
            raise AdoptError("refit_missing_key", e["key"])
        arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
        if e.get("transform") == "transpose":
            arr = arr.T
        out[e["name"]] = np.ascontiguousarray(arr)
    return out


# ---------------------------------------------------------------------------
# Consumer — deserialize + refit + guarded module swap
# ---------------------------------------------------------------------------


def _load_engine(plan_path: Path):
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(Path(plan_path).read_bytes())
    if engine is None:
        raise AdoptError("engine_deserialize", str(plan_path))
    return engine


def _refit_engine(engine: Any, weights: Dict[str, Any]) -> None:
    import tensorrt as trt

    refitter = trt.Refitter(engine, trt.Logger(trt.Logger.WARNING))
    needed = list(refitter.get_all_weights())
    missing = [n for n in needed if n not in weights]
    if missing:
        raise AdoptError("refit_incomplete", f"{len(missing)} engine weights unmapped, e.g. {missing[:3]}")
    for name in needed:
        if not refitter.set_named_weights(name, trt.Weights(weights[name])):
            raise AdoptError("refit_set_failed", name)
    if not refitter.refit_cuda_engine():
        raise AdoptError("refit_failed", "refit_cuda_engine returned False")


class TrtModuleRunner:
    """Executes a diffusers denoiser module's forward through a TRT engine.

    Holds one execution context; binds torch CUDA tensors by data_ptr
    (zero-copy). Input binding order/names come from the artifact's ``inputs``
    contract; the single output is returned as a torch tensor. Engines carry
    one optimization profile PER preset shape — the runner selects the
    profile whose ``sample`` range admits the call's shape.
    """

    def __init__(self, engine: Any, meta: Dict[str, Any], device: str = "cuda") -> None:
        import tensorrt as trt  # noqa: F401
        import torch

        self.engine = engine
        self.meta = meta
        self.device = device
        self.context = engine.create_execution_context()
        self._torch = torch
        self._out_name = None
        self._profile = 0
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name).name == "OUTPUT":
                self._out_name = name
        # profile index -> the exact `sample` shape it was built for
        # (min == opt == max: one static profile per preset shape).
        self._profile_sample: Dict[Tuple[int, ...], int] = {}
        try:
            for p in range(engine.num_optimization_profiles):
                mn, _opt, _mx = engine.get_tensor_profile_shape("sample", p)
                self._profile_sample[tuple(mn)] = p
        except Exception:
            pass

    def _select_profile(self, sample_shape: Tuple[int, ...]) -> None:
        idx = self._profile_sample.get(sample_shape)
        if idx is None and self._profile_sample:
            raise RuntimeError(
                f"no optimization profile for sample shape {sample_shape} "
                f"(built: {sorted(self._profile_sample)})"
            )
        if idx is not None and idx != self._profile:
            stream = self._torch.cuda.current_stream().cuda_stream
            self.context.set_optimization_profile_async(idx, stream)
            self._torch.cuda.current_stream().synchronize()
            self._profile = idx

    def __call__(self, feeds: Dict[str, Any]) -> Any:
        torch = self._torch
        ctx = self.context
        if "sample" in feeds:
            self._select_profile(tuple(feeds["sample"].shape))
        for name, tensor in feeds.items():
            t = tensor.contiguous()
            feeds[name] = t
            ctx.set_input_shape(name, tuple(t.shape))
            ctx.set_tensor_address(name, t.data_ptr())
        out_shape = tuple(ctx.get_tensor_shape(self._out_name))
        import tensorrt as trt

        dtype = {
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.FLOAT: torch.float32,
        }[self.engine.get_tensor_dtype(self._out_name)]
        out = torch.empty(out_shape, dtype=dtype, device=self.device)
        ctx.set_tensor_address(self._out_name, out.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        if not ctx.execute_async_v3(stream):
            raise RuntimeError("trt execute_async_v3 failed")
        torch.cuda.current_stream().synchronize()
        return out


def _unet_feeds(meta: Dict[str, Any], args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Map a diffusers UNet/DiT ``forward(sample, timestep, ...)`` call onto
    the engine's input contract."""
    import torch

    sample = args[0] if args else kwargs["sample"]
    timestep = args[1] if len(args) > 1 else kwargs["timestep"]
    ehs = kwargs.get("encoder_hidden_states")
    if ehs is None and len(args) > 2:
        ehs = args[2]
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], dtype=sample.dtype, device=sample.device)
    if timestep.ndim == 0:
        timestep = timestep[None]
    timestep = timestep.expand(sample.shape[0]).to(sample.dtype)
    feeds = {"sample": sample, "timestep": timestep, "encoder_hidden_states": ehs}
    added = kwargs.get("added_cond_kwargs") or {}
    if "text_embeds" in added:
        feeds["text_embeds"] = added["text_embeds"]
    if "time_ids" in added:
        feeds["time_ids"] = added["time_ids"].to(sample.dtype)
    want = {str(i["name"]) for i in meta.get("inputs") or []}
    if want:
        feeds = {k: v for k, v in feeds.items() if k in want}
        missing = want - set(feeds)
        if missing:
            raise RuntimeError(f"engine expects inputs {sorted(missing)} the call did not provide")
    return feeds


def wrap_module(module: Any, runner: TrtModuleRunner, meta: Dict[str, Any]) -> None:
    """Swap ``module.forward`` for the engine behind a fail-soft guard: the
    first engine error permanently unwraps back to eager. The module object
    (config, dtype, device, weights) stays untouched — diffusers pipelines
    read its attributes, and its weights remain the refit source."""
    original = module.forward
    state = {"failed": False}

    def trt_forward(*args: Any, **kwargs: Any) -> Any:
        if state["failed"]:
            return original(*args, **kwargs)
        try:
            feeds = _unet_feeds(meta, args, kwargs)
            out = runner(feeds)
        except Exception as exc:  # noqa: BLE001 — ANY engine problem => eager
            state["failed"] = True
            logger.warning(
                "trt-engine: %s failed (%s: %s); eager for the rest of this process",
                meta.get("module"), type(exc).__name__, exc,
            )
            return original(*args, **kwargs)
        if kwargs.get("return_dict", True):
            from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

            return UNet2DConditionOutput(sample=out)
        return (out,)

    module.forward = trt_forward
    setattr(module, _MARKER_ATTR, {"meta": {k: meta.get(k) for k in ("sku", "trt", "precision", "shapes")}})


def enable(
    pipeline: Any,
    cfg: Any,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> bool:
    """Consumer entry point: verify + unpack a TRT engine artifact, refit it
    with the resident module's weights, and swap the module forward. Returns
    False (staying eager) on ANY miss; raises :class:`AdoptError` only via
    the adopt path (executor catches + classifies)."""
    if artifact is None:
        return False
    try:
        meta = load_and_wrap(pipeline, cfg, Path(artifact), cache_dir=cache_dir)
        logger.info(
            "trt-engine: armed %s (sku=%s trt=%s precision=%s, refit from resident weights)",
            meta.get("module"), meta.get("sku"), meta.get("trt"), meta.get("precision"),
        )
        return True
    except Exception as exc:
        logger.warning("trt-engine: artifact unusable (%s); staying eager", exc)
        return False


def load_and_wrap(
    pipeline: Any, cfg: Any, artifact: Path, cache_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """The adopt-path core: unpack+verify+deserialize+refit+wrap. Raises
    AdoptError with a classified reason on any failure (never half-wraps)."""
    if getattr(pipeline, _MARKER_ATTR, None) is not None:
        return getattr(pipeline, _MARKER_ATTR)["meta"]
    root = (Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gen-worker") / "trt-engine"
    try:
        meta = unpack(Path(artifact), root)
    except AdoptError:
        raise
    except Exception as exc:
        raise AdoptError("artifact_invalid", str(exc)) from exc
    family = str(getattr(cfg, "family", "") or "")
    reason = verify(meta, family=family)
    if reason:
        raise AdoptError("key_mismatch", reason)

    module_name = str(meta.get("module") or "unet")
    module = getattr(pipeline, module_name, None)
    if module is None:
        raise AdoptError("no_target", f"pipeline has no module {module_name!r}")

    t0 = time.monotonic()
    engine = _load_engine(root / ENGINE_NAME)
    entries = json.loads((root / REFIT_MAP_NAME).read_text())
    weights = refit_weights(dict(module.state_dict()), entries)
    _refit_engine(engine, weights)
    runner = TrtModuleRunner(engine, meta, device=str(getattr(module, "device", "cuda")))
    wrap_module(module, runner, meta)
    setattr(pipeline, _MARKER_ATTR, {"meta": meta})
    logger.info("trt-engine: deserialize+refit in %.1fs", time.monotonic() - t0)
    return meta


# ---------------------------------------------------------------------------
# Build (the produce-trt-engine conversion job)
# ---------------------------------------------------------------------------


_SDXL_UNET_INPUTS = ("sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids")


def _export_unet_onnx(pipe: Any, onnx_path: Path, *, batch: int, shape: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Export the pipeline's UNet to ONNX with the SDXL conditioning contract.
    Returns the input contract (names + per-shape dims recorded later)."""
    import torch

    unet = pipe.unet
    dtype = next(unet.parameters()).dtype
    device = next(unet.parameters()).device
    w, h = shape
    lh, lw = h // pipe.vae_scale_factor, w // pipe.vae_scale_factor
    ehs_dim = int(unet.config.cross_attention_dim)
    ehs_len = 77
    add_dim = 1280  # SDXL pooled text embeds

    class _Export(torch.nn.Module):
        def __init__(self, unet: Any) -> None:
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
            return self.unet(
                sample, timestep, encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                return_dict=False,
            )[0]

    sample = torch.randn(batch, unet.config.in_channels, lh, lw, dtype=dtype, device=device)
    timestep = torch.ones(batch, dtype=dtype, device=device)
    ehs = torch.randn(batch, ehs_len, ehs_dim, dtype=dtype, device=device)
    text_embeds = torch.randn(batch, add_dim, dtype=dtype, device=device)
    time_ids = torch.randn(batch, 6, dtype=dtype, device=device)

    dyn = {"sample": {2: "lh", 3: "lw"}}
    torch.onnx.export(
        _Export(unet), (sample, timestep, ehs, text_embeds, time_ids), str(onnx_path),
        input_names=list(_SDXL_UNET_INPUTS), output_names=["out_sample"],
        dynamic_axes=dyn, opset_version=18, do_constant_folding=True, dynamo=False,
    )
    return [{"name": n} for n in _SDXL_UNET_INPUTS]


def _onnx_initializers(onnx_path: Path) -> Dict[str, Any]:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path), load_external_data=True)
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def build(
    model_path: str | Path,
    out_dir: str | Path,
    *,
    shapes: Iterable[Tuple[int, int]],
    module: str = "unet",
    batch: int = 2,
    precision: str = "fp16",
    family: str = "",
    source_ref: str = "",
    source_digest: str = "",
    pipeline_cls: Any = None,
) -> Tuple[Path, Dict[str, Any], Dict[str, float]]:
    """Build a weight-stripped TRT engine for one family on THIS GPU SKU.

    ``batch`` is the CFG regime (ie#345): CFG variants run batch-2 graphs,
    distilled variants batch-1 — one regime per artifact, never both.
    ``shapes`` should derive from the family's payload preset enum; every
    shape becomes one static optimization profile in the ONE engine.
    """
    import tensorrt as trt
    import torch
    from diffusers import DiffusionPipeline

    from .models.loading import load_from_pretrained

    if not torch.cuda.is_available():
        raise RuntimeError("trt-engine build requires CUDA")
    if precision not in ("fp16", "bf16"):
        raise ValueError(f"unsupported precision {precision!r}")

    out_dir = Path(out_dir)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}
    shapes = [(int(w), int(h)) for w, h in shapes]

    t0 = time.monotonic()
    pipe = load_from_pretrained(
        pipeline_cls or DiffusionPipeline, str(model_path), dtype=precision
    )
    pipe.to("cuda")
    mod = getattr(pipe, module)
    timings["load_s"] = round(time.monotonic() - t0, 1)

    t0 = time.monotonic()
    onnx_path = work / "model.onnx"
    if module != "unet":
        raise NotImplementedError(f"pilot exports unet only, got module={module!r}")
    inputs = _export_unet_onnx(pipe, onnx_path, batch=batch, shape=shapes[0])
    timings["onnx_export_s"] = round(time.monotonic() - t0, 1)

    # The builder needs the GPU to itself: tactic timing allocates multi-GB
    # scratch regions, and a resident fp16 pipeline starves it ("region
    # allocation failed" tactic skips => worse engines or a failed build).
    # Everything after export (refit map, self-check) reads CPU tensors.
    pipe.to("cpu")
    mod = getattr(pipe, module)
    torch.cuda.empty_cache()

    t0 = time.monotonic()
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path)):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {errs}")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.STRIP_PLAN)
    config.set_flag(trt.BuilderFlag.REFIT)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)

    scale = pipe.vae_scale_factor
    in_ch = int(mod.config.in_channels)
    for w, h in shapes:
        profile = builder.create_optimization_profile()
        lat = (batch, in_ch, h // scale, w // scale)
        profile.set_shape("sample", lat, lat, lat)
        for i in range(network.num_inputs):
            name = network.get_input(i).name
            if name == "sample":
                continue
            dims = tuple(network.get_input(i).shape)
            fixed = tuple(batch if d == -1 else d for d in dims)
            profile.set_shape(name, fixed, fixed, fixed)
        config.add_optimization_profile(profile)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine build failed (see TRT log)")
    (work / ENGINE_NAME).write_bytes(bytes(plan))
    timings["engine_build_s"] = round(time.monotonic() - t0, 1)

    # Refit map: engine weight names <-> torch keys by value identity, then
    # PROVE completeness against the engine's own refittable-weight list.
    t0 = time.monotonic()
    entries, unmatched = build_refit_map(_onnx_initializers(onnx_path), dict(mod.state_dict()))
    engine = _load_engine(work / ENGINE_NAME)
    import tensorrt as _trt

    refitter = _trt.Refitter(engine, trt_logger)
    needed = set(refitter.get_all_weights())
    mapped = {e["name"] for e in entries}
    missing = sorted(needed - mapped)
    if missing:
        raise RuntimeError(
            f"refit map incomplete: {len(missing)}/{len(needed)} engine weights unmapped "
            f"(e.g. {missing[:5]}); unmatched initializers={len(unmatched)}"
        )
    entries = [e for e in entries if e["name"] in needed]
    (work / REFIT_MAP_NAME).write_text(json.dumps(entries, sort_keys=True, indent=0))
    timings["refit_map_s"] = round(time.monotonic() - t0, 1)

    meta = artifact_metadata(
        family=family, module=module, precision=precision, batch=batch,
        shapes=shapes, inputs=inputs, source_ref=source_ref, source_digest=source_digest,
    )
    label = flavor_label(meta["sku"], meta["trt"], precision)
    artifact = pack(work, out_dir / f"{label}.tar.gz", meta)
    logger.info(
        "trt-engine build: %s (%.1fMB plan, %d refit weights) in %s",
        label, (work / ENGINE_NAME).stat().st_size / 1e6, len(entries), timings,
    )
    return artifact, meta, timings


__all__ = [
    "ARTIFACT_FORMAT",
    "TrtModuleRunner",
    "artifact_metadata",
    "build",
    "build_refit_map",
    "enable",
    "find_artifact",
    "flavor_label",
    "is_engine_ref",
    "load_and_wrap",
    "pack",
    "refit_weights",
    "runtime_key",
    "trt_maj_min",
    "trt_version",
    "unpack",
    "unpack_metadata",
    "verify",
    "wrap_module",
]
