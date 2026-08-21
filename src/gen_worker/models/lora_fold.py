"""Per-request LoRA FOLD: W += B@A*s written IN PLACE into the weights the compiled artifact already points at — it holds raw pointers (load_constants(..., user_managed=True)), so a fresh-tensor swap would be invisible to it, and cudagraph static inputs want pointer stability. Restore is a copy_ back from a saved clone, NEVER a sub_ of the delta: subtracting is not the inverse of adding in bf16 and a serial request stream would accumulate drift. Never folds onto a quantized leaf (fp8/gguf: the grid cannot represent a small delta — those lanes keep the additive branch in .w8a8_lora; refused by name). AOTI no longer runtime-folds (tcg#80 moved the sealed policy to always_keep_tensor_constants=True), so there is no folded constant to go stale and the raw pointer an in-place write moves IS what the artifact reads — the re-arm hazard this docstring was written for is gone. `folded()` still takes `rebind` and still REFUSES a compiled-armed module without one: fail-closed until pgw#1571 measures the seam under the new policy and retires the gate deliberately."""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from ..api.errors import RefCompatibilitySurprise
from . import w8a8_lora

logger = logging.getLogger(__name__)

Adapter = Tuple[Dict[str, Any], float, str]

Rebind = Callable[[Any], Any]


def _plain_leaf(module: Any) -> bool:
    import torch.nn as nn

    from .fp8_storage import structural_base
    from .gguf_torch import is_gguf_leaf
    from .w8a8 import fp8_scaled_linear_class

    if isinstance(module, fp8_scaled_linear_class()) or is_gguf_leaf(module):
        return False
    if structural_base(module) not in (nn.Linear, nn.Conv2d):
        return False
    return not bool(getattr(module, "_cozy_fp8_storage_applied", False))


def adapter_digest(adapters: Sequence[Adapter]) -> str:
    """A stable identity for one ADAPTER SET, order and weights included."""
    h = hashlib.sha256()
    for _sd, weight, ref in adapters:
        h.update(str(ref).encode())
        h.update(b"\0")
        h.update(f"{float(weight):.12g}".encode())
        h.update(b"\0")
    return h.hexdigest()[:32]


def _delta(mod: Any, a: Any, b: Any, scale: float) -> Any:
    import torch
    import torch.nn as nn

    a32 = a.detach().to(torch.float32)
    b32 = b.detach().to(torch.float32)
    if isinstance(mod, nn.Conv2d) or a32.dim() == 4:
        out = torch.einsum("or,rihw->oihw", b32.reshape(b32.shape[0], b32.shape[1]), a32)
    else:
        out = b32 @ a32
    return out.mul_(float(scale))


def fold_targets(pipe: Any) -> Dict[str, Any]:
    """Every component an adapter half can land on: the denoisers, plus the text encoders."""
    from ..component_vocab import text_encoder_components

    targets = dict(w8a8_lora.branch_targets(pipe))
    for name in text_encoder_components():
        module = getattr(pipe, name, None)
        if module is not None and hasattr(module, "named_modules"):
            targets.setdefault(name, module)
    return targets


def _route(
    normalized: Dict[str, Any], targets: Mapping[str, Any],
    denoisers: Sequence[str], *, ref: str,
) -> Dict[str, Dict[str, Any]]:
    from ..utils.lora import te_prefix_to_component

    den, rest = w8a8_lora.split_state_dict(normalized)
    routed: Dict[str, Dict[str, Any]] = {
        comp: dict(keys) for comp, keys
        in w8a8_lora.route_denoiser_keys(den, denoisers, ref=ref).items()
    }
    unrouted: List[str] = []
    for key, tensor in rest.items():
        for prefix, comp in te_prefix_to_component():
            if key.startswith(prefix) and comp in targets:
                routed.setdefault(comp, {})[key[len(prefix):].lstrip(".")] = tensor
                break
        else:
            unrouted.append(key)
    if unrouted:
        raise RefCompatibilitySurprise(
            f"{len(unrouted)} adapter key(s) land on no component this "
            f"pipeline carries (e.g. {', '.join(sorted(unrouted)[:3])}) — it "
            f"has {', '.join(sorted(targets))}. Folding the rest would serve "
            "a partial adapter and say nothing",
            ref=ref, axis="component_missing",
        )
    return routed


def compute_deltas(
    model: Any, adapters: Sequence[Adapter], *, pipe: Any = None,
    keys: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """``module path -> ΔW`` for one component and the whole adapter set."""
    import torch

    mods = w8a8_lora.branch_modules(model)
    deltas: Dict[str, Any] = {}
    for index, (state_dict, weight, ref) in enumerate(adapters):
        if keys is not None:
            slice_ = keys[index]
        else:
            normalized = w8a8_lora.normalize_adapter_state_dict(
                pipe if pipe is not None else model, state_dict, ref=ref)
            slice_, _rest = w8a8_lora.split_state_dict(normalized)
        if not slice_:
            continue
        mapped = w8a8_lora.map_adapter(dict(slice_), model, ref=ref)
        quantized = sorted(p for p in mapped if not _plain_leaf(mods[p]))
        if quantized:
            raise RefCompatibilitySurprise(
                f"{len(quantized)} target module(s) store a QUANTIZED weight "
                f"(e.g. {', '.join(quantized[:3])}) — folding a low-rank delta "
                "onto an fp8/GGML grid rounds it away and serves a silently "
                "weakened adapter; this lane keeps the additive branch",
                ref=ref, axis="state_dict",
            )
        for path, (a, b, alpha_scale) in mapped.items():
            mod = mods[path]
            w = mod.weight
            d = _delta(mod, a.to(w.device), b.to(w.device),
                       alpha_scale * float(weight))
            if tuple(d.shape) != tuple(w.shape):
                raise RefCompatibilitySurprise(
                    f"folded delta for {path!r} is {tuple(d.shape)} but the "
                    f"weight is {tuple(w.shape)}",
                    ref=ref, axis="state_dict",
                )
            prior = deltas.get(path)
            deltas[path] = d if prior is None else prior.add_(d)
    for path in deltas:
        deltas[path] = deltas[path].to(torch.float32)
    return deltas


@dataclass
class FoldScope:
    """One denoiser's live fold: the saved originals that undo it exactly."""

    model: Any
    saved: List[Tuple[Any, Any]] = field(default_factory=list)
    folded_bytes: int = 0

    def restore(self) -> None:
        """Put every mutated weight back, byte for byte."""
        for mod, original in reversed(self.saved):
            try:
                mod.weight.data.copy_(original)
            except Exception:  # noqa: BLE001 — a restore must finish
                logger.exception(
                    "lora fold: restoring %s failed; this denoiser now carries "
                    "an adapter into later requests", type(mod).__name__)
        self.saved.clear()


def apply_fold(model: Any, deltas: Mapping[str, Any]) -> FoldScope:
    """Write ``deltas`` into the model's weights IN PLACE; return the undo."""
    import torch

    mods = w8a8_lora.branch_modules(model)
    scope = FoldScope(model=model)
    try:
        with torch.no_grad():
            for path in sorted(deltas):
                mod = mods[path]
                w = mod.weight
                scope.saved.append((mod, w.data.detach().clone()))
                scope.folded_bytes += int(w.numel() * w.element_size())
                w.data.copy_((w.data.to(torch.float32) + deltas[path]).to(w.dtype))
    except BaseException:
        scope.restore()
        raise
    return scope


def _compiled_armed(model: Any) -> bool:
    from ..serving import adapter_guard

    if adapter_guard.compiled_armed(model):
        return True
    try:
        from .. import aot_serve
    except ImportError:
        return False
    return bool(aot_serve.serves_compiled(model))


@contextmanager
def folded(
    pipe: Any,
    adapters: Sequence[Adapter],
    *,
    request_id: str = "",
    rebind: Optional[Rebind] = None,
) -> Iterator[Dict[str, Any]]:
    """Serve one request with ``adapters`` folded into the pipeline's weights."""
    targets = fold_targets(pipe)
    if not adapters or not targets:
        yield {}
        return

    denoisers = tuple(w8a8_lora.branch_targets(pipe))
    per_adapter = [
        _route(
            w8a8_lora.normalize_adapter_state_dict(pipe, state_dict, ref=ref),
            targets, denoisers, ref=ref,
        )
        for state_dict, _weight, ref in adapters
    ]

    plan: List[Tuple[Any, Dict[str, Any]]] = []
    for comp, model in targets.items():
        slices = [routed.get(comp, {}) for routed in per_adapter]
        if not any(slices):
            continue
        deltas = compute_deltas(model, adapters, pipe=pipe, keys=slices)
        if deltas:
            plan.append((model, deltas))
    if not plan:
        yield {}
        return

    if rebind is None:
        armed = [type(m).__name__ for m, _d in plan if _compiled_armed(m)]
        if armed:
            raise RefCompatibilitySurprise(
                f"{', '.join(armed)} is serving a COMPILED artifact and no "
                "constant re-arm seam was supplied — an in-place weight fold "
                "is visible to the artifact's user-managed pointers but does "
                "NOT re-run AOTI's runtime constant folding, so any folded "
                "constant would serve stale weights. Pass "
                "rebind=aot_serve.rearm_constants",
                axis="pipeline_load",
            )

    scopes: List[FoldScope] = []
    stats: Dict[str, Any] = {
        "components": len(plan),
        "modules": sum(len(d) for _m, d in plan),
        "digest": adapter_digest(adapters),
    }
    try:
        for model, deltas in plan:
            scopes.append(apply_fold(model, deltas))
            if rebind is not None:
                rebind(model)
        stats["folded_bytes"] = sum(s.folded_bytes for s in scopes)
        logger.info(
            "[request_id=%s] lora folded into weights: %d component(s), "
            "%d module(s), %d saved byte(s), set=%s",
            request_id, stats["components"], stats["modules"],
            stats["folded_bytes"], stats["digest"],
        )
        yield stats
    finally:
        for model, scope in zip((m for m, _d in plan), scopes):
            scope.restore()
            if rebind is not None:
                try:
                    rebind(model)
                except Exception:  # noqa: BLE001 — a restore must finish
                    logger.exception(
                        "lora fold: re-arming %s after restore failed",
                        type(model).__name__)


__all__ = [
    "Adapter",
    "FoldScope",
    "Rebind",
    "adapter_digest",
    "apply_fold",
    "compute_deltas",
    "folded",
]
