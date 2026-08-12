"""Per-SKU TensorRT engine artifacts (#390) — the second producer/consumer
on the compile-cache rails (gw#384 / th#569 / th#567).

One WEIGHT-STRIPPED engine serves every fine-tune of a model family: the
engine captures the optimized graph + tactics (~1% of a weight-full plan);
the consumer REFITs it with the weights of whatever family member is already
resident, then swaps the module's ``forward`` behind a guard. Same trust
model, storage, and delivery as inductor caches — cells live as flavors of
``root/family-<family>``:

    root/family-<f>#trt-<sku>-trt<maj.min>-<precision>

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

Policy mirrors compile_cache: an optional plain lane stays eager when no exact
artifact is available. A failing optional engine call permanently routes to
the eager module and revokes its compiled-state proof before that fallback.

**THERE IS NO PRODUCER IN THIS REPOSITORY.** ``build`` (ONNX export + engine
build) and ``find_artifact`` were deleted 2026-08-12 under §4.34: nothing on
the pod, in any script, or in any test ever called them. What survives is the
CONSUMER half — stage, verify, refit, wrap, revoke — which reads artifacts some
other producer must write, exactly the shape pgw#1178 found on the adopt lane.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from . import activity as activity_mod
from . import artifact_meta
from .cell_adopt import AdoptOutcome
from .compile_cache import (
    AdoptError,
    CompiledExecutionLaneUnavailableError,
    _clean_tarinfo,
    parse_cell_ref,
    sku_slug,
)
import gzip
from .models.loading import load_from_pretrained

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
    fam, flavor = parse_cell_ref(ref)
    if not fam or (family and fam != family):
        return False
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
    seen: set[str] = set()
    with tarfile.open(artifact, mode="r:*") as tar:
        for member in tar:
            name = member.name
            if name not in _MEMBERS or not member.isfile() or name in seen:
                raise ValueError(f"unexpected member in trt-engine artifact: {member.name!r}")
            seen.add(name)
            src = tar.extractfile(member)
            assert src is not None
            data = src.read()
            if name == METADATA_NAME:
                meta = json.loads(data.decode())
                continue
            (dest_root / name).write_bytes(data)
    missing = set(_MEMBERS) - seen
    if missing:
        raise ValueError(
            f"trt-engine artifact {artifact} is incomplete; missing {sorted(missing)!r}"
        )
    if not meta:
        raise ValueError(f"trt-engine artifact {artifact} has no {METADATA_NAME}")
    return meta


@dataclass
class _StagedEngineArtifact:
    metadata: Dict[str, Any]
    root: Path
    temporary: tempfile.TemporaryDirectory[str]

    def close(self) -> None:
        self.temporary.cleanup()


def stage_artifact(
    artifact: Path, family: str, cache_dir: Optional[Path] = None,
) -> _StagedEngineArtifact:
    """Extract and runtime-verify a complete engine in an isolated tree.

    The live/shared cache and pipeline remain untouched on every rejection.
    Concurrent attempts use distinct trees; a process crash can leave only an
    unreferenced staging directory, never partially published engine files.
    """
    base = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gen-worker"
    base.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="trt-engine-stage-", dir=base)
    root = Path(temporary.name)
    try:
        meta = unpack(Path(artifact), root)
        reason = verify(meta, family=family)
        if reason:
            raise AdoptError("key_mismatch", reason)
        return _StagedEngineArtifact(meta, root, temporary)
    except AdoptError:
        temporary.cleanup()
        raise
    except Exception as exc:
        temporary.cleanup()
        raise AdoptError("artifact_invalid", str(exc)) from exc


def unpack_metadata(artifact: Path) -> Dict[str, Any]:
    """Read ONLY metadata.json from an artifact (kind sniffing — cheap).

    The shared :mod:`artifact_meta` reader; the refusal stays a
    :class:`ValueError` subclass, which is what every caller classifies on.
    """
    return artifact_meta.read_metadata(artifact)


# ---------------------------------------------------------------------------
# Refit map — torch state_dict <-> engine weight names by VALUE identity
# ---------------------------------------------------------------------------


def _fingerprint(data: bytes, shape: Tuple[int, ...]) -> str:
    return f"{shape}:{hashlib.sha256(data).hexdigest()[:24]}"


def build_refit_map(
    initializers: Dict[str, Any], state_dict: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[str]]:
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

    entries: List[Dict[str, Any]] = []
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


def refit_weights(state_dict: Dict[str, Any], entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Materialize ``{engine weight name: numpy array}`` from a state_dict
    through a refit map (applying recorded transforms)."""
    import numpy as np

    out: Dict[str, Any] = {}
    for e in entries:
        if e.get("transform") == "const":
            # Graph constant baked at build time (ONNX Constant-node output
            # or non-weight initializer): fine-tune-invariant, carried in
            # the map itself — no state_dict counterpart exists.
            arr = np.frombuffer(
                base64.b64decode(e["data_b64"]), dtype=np.dtype(e["dtype"])
            ).reshape([int(x) for x in e["shape"]])
            out[e["name"]] = np.ascontiguousarray(arr)
            continue
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


def _load_engine(plan_path: Path) -> Any:
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


def wrap_module(
    module: Any,
    runner: TrtModuleRunner,
    meta: Dict[str, Any],
    *,
    eager_forward: Optional[Callable[..., Any]] = None,
) -> None:
    """Swap ``module.forward`` for an optional engine behind a fail-soft guard.

    The first engine error synchronously revokes scheduler-visible compiled
    proof, then permanently routes this wrapper to eager. The module object
    (config, dtype, device, weights) stays untouched — diffusers pipelines
    read its attributes, and its weights remain the refit source.
    """
    # A different-cell rearm replaces a failed wrapper in one assignment.
    # Preserve the underlying eager callable rather than capturing the old
    # wrapper as the new fallback.
    original = eager_forward or module.forward
    state: Dict[str, Any] = {
        "failed": False,
        "successful_calls": 0,
        "original": original,
        "failure_callback": None,
        "revocation_error": "",
    }

    def trt_forward(*args: Any, **kwargs: Any) -> Any:
        if state["revocation_error"]:
            raise CompiledExecutionLaneUnavailableError(state["revocation_error"])
        if state["failed"]:
            return original(*args, **kwargs)
        try:
            feeds = _unet_feeds(meta, args, kwargs)
            out = runner(feeds)
            state["successful_calls"] += 1
        except Exception as exc:  # noqa: BLE001 — ANY engine problem => eager
            state["failed"] = True
            detail = (
                f"TensorRT target {meta.get('module')} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            callback = state.get("failure_callback")
            if callable(callback):
                try:
                    callback(detail)
                except Exception as callback_exc:
                    state["revocation_error"] = (
                        "compiled-state revocation failed: "
                        f"{type(callback_exc).__name__}: {callback_exc}"
                    )
                    logger.exception(
                        "trt-engine: %s", state["revocation_error"])
                    raise CompiledExecutionLaneUnavailableError(
                        state["revocation_error"]
                    ) from callback_exc
            logger.warning(
                "trt-engine: %s failed (%s: %s); eager for the rest of this process",
                meta.get("module"), type(exc).__name__, exc,
            )
            # pgw#760: a permanent serve-path decision (this module runs
            # eager until the process dies) — countable on the wire, not
            # only in pod logs.
            activity_mod.emit_event(
                activity_mod.KIND_SERVE_DEGRADE,
                f"module={meta.get('module')} sku={meta.get('sku')} "
                f"trt={meta.get('trt')} precision={meta.get('precision')}: "
                f"engine call failed, eager for the rest of this process: "
                f"{type(exc).__name__}: {exc}",
                phase="trt_runtime_failed",
            )
            return original(*args, **kwargs)
        if kwargs.get("return_dict", True):
            from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

            return UNet2DConditionOutput(sample=out)
        return (out,)

    module.forward = trt_forward
    setattr(module, _MARKER_ATTR, {
        "meta": {k: meta.get(k) for k in ("sku", "trt", "precision", "shapes")},
        "state": state,
    })


def enable(
    pipeline: Any,
    cfg: Any,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> AdoptOutcome:
    """Consumer entry point: verify + unpack a TRT engine artifact, refit it
    with the resident module's weights, and swap the module forward. Falsy
    (staying eager) on ANY miss; raises :class:`AdoptError` only via the adopt
    path (executor catches + classifies).

    pgw#923: the outcome is RETURNED, for the same reason the exported arm's
    is. The classified reason used to leave here only as the ``phase`` of a
    free-text ``trt_adopt`` event — a THIRD spelling of "a cell adopted, and
    what it cost", alongside ``aot_adopt`` and the measured
    ``compile_cache_adopt``. The arming policy now measures this arm and the
    executor reports it on the one lane that carries numbers.
    """
    if artifact is None:
        return AdoptOutcome.miss("no_artifact")
    try:
        meta = load_and_wrap(pipeline, cfg, Path(artifact), cache_dir=cache_dir)
    except Exception as exc:
        # pgw#760 (the pgw#733 pattern, TRT half): the classified AdoptError
        # reason must not be reduced to logger.warning — that is structurally
        # invisible on hub-spawned workers. The reason token is the countable
        # fact and it now rides the adoption's own wire event.
        reason = str(getattr(exc, "reason", "") or "") or type(exc).__name__
        logger.warning(
            "trt-engine: artifact unusable (%s: %s); staying eager",
            reason, exc)
        return AdoptOutcome.miss(
            reason,
            f"family={getattr(cfg, 'family', '')} "
            f"artifact={Path(artifact).name}: {type(exc).__name__}: {exc}",
            f"artifact={Path(artifact).name}")
    logger.info(
        "trt-engine: armed %s (sku=%s trt=%s precision=%s, refit from resident weights)",
        meta.get("module"), meta.get("sku"), meta.get("trt"), meta.get("precision"),
    )
    return AdoptOutcome.hit(
        f"family={getattr(cfg, 'family', '')} "
        f"module={meta.get('module')} sku={meta.get('sku')} "
        f"trt={meta.get('trt')} precision={meta.get('precision')}: "
        f"armed {Path(artifact).name}")


def load_and_wrap(
    pipeline: Any, cfg: Any, artifact: Path, cache_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Stage+verify+deserialize+refit, then perform the sole live wrap.

    Raises :class:`AdoptError` with a classified reason on any failure and
    never publishes extracted files into a shared live cache.
    """
    family = str(getattr(cfg, "family", "") or "")
    staged = stage_artifact(Path(artifact), family, cache_dir=cache_dir)
    try:
        meta = staged.metadata
        module_name = str(meta.get("module") or "unet")
        module = getattr(pipeline, module_name, None)
        if module is None:
            raise AdoptError("no_target", f"pipeline has no module {module_name!r}")
        eager_forward: Optional[Callable[..., Any]] = None
        old_marker = getattr(pipeline, _MARKER_ATTR, None)
        if old_marker is not None:
            old_module = old_marker.get("module")
            old_state = old_marker.get("state") or {}
            eager_forward = old_state.get("original")
            if old_module is not module or not callable(eager_forward):
                raise AdoptError(
                    "old_marker_invalid",
                    "existing TRT marker does not retain this module's eager callable",
                )

        t0 = time.monotonic()
        engine = _load_engine(staged.root / ENGINE_NAME)
        entries: list[dict[str, Any]] = json.loads(
            (staged.root / REFIT_MAP_NAME).read_text())
        weights = refit_weights(dict(module.state_dict()), entries)
        _refit_engine(engine, weights)
        runner = TrtModuleRunner(
            engine, meta, device=str(getattr(module, "device", "cuda")))
        # This is the first live mutation: the complete artifact, runtime key,
        # target, engine, and refit weights are already proven above.
        wrap_module(module, runner, meta, eager_forward=eager_forward)
        module_marker = getattr(module, _MARKER_ATTR, {})
        setattr(pipeline, _MARKER_ATTR, {
            "meta": meta,
            "state": module_marker.get("state", {}),
            "module": module,
        })
        logger.info("trt-engine: deserialize+refit in %.1fs", time.monotonic() - t0)
        return meta
    finally:
        staged.close()


def execution_count(pipeline: Any) -> int:
    """Successful engine calls observed on this exact wrapped pipeline."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    return int((marker.get("state") or {}).get("successful_calls", 0))


def set_guard_failure_callback(
    pipeline: Any, callback: Any,
) -> bool:
    """Bind scheduler-state revocation to an armed engine guard."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    state = marker.get("state")
    if not isinstance(state, dict):
        return False
    state["failure_callback"] = callback
    return True


def is_armed(pipeline: Any) -> bool:
    """Whether a TRT engine currently REPLACES this pipeline's forward
    (pgw#813): while it does there is no eager tier to serve from."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    state = marker.get("state")
    if not isinstance(state, dict):
        return False
    return not bool(state.get("failed", False))


def unwrap(pipeline: Any) -> bool:
    """Restore eager forward after an unproven first-time engine adoption."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    module = marker.get("module")
    state = marker.get("state") or {}
    original = state.get("original")
    if module is None or not callable(original):
        return False
    module.forward = original
    try:
        delattr(module, _MARKER_ATTR)
    except AttributeError:
        pass
    try:
        delattr(pipeline, _MARKER_ATTR)
    except AttributeError:
        pass
    return True


__all__ = [
    "ARTIFACT_FORMAT",
    "TrtModuleRunner",
    "artifact_metadata",
    "build_refit_map",
    "enable",
    "execution_count",
    "flavor_label",
    "is_engine_ref",
    "load_and_wrap",
    "pack",
    "refit_weights",
    "runtime_key",
    "set_guard_failure_callback",
    "stage_artifact",
    "unwrap",
    "trt_maj_min",
    "trt_version",
    "unpack",
    "unpack_metadata",
    "verify",
    "wrap_module",
]
