"""Per-request LoRA FOLD: the delta goes into the weights, then the pipeline
runs unchanged (pgw#1571, Paul's direct order).

The point is the compiled path. Live adapter ops — peft's ``lora.Linear``
wrappers, or our own additive branch — put extra modules INSIDE the denoiser,
and a compiled denoiser does not execute its own submodules: ``aot_serve.
wrap_module`` replaced ``forward`` with the artifact's dispatch, whose
constants were bound once at arm time. Measured (pgw#1571): a peft adapter on a
compiled-armed unet changes the served tensor by exactly 0.0 — the base model,
bit-identically, with no refusal and no log.

Folding removes the failure mode instead of guarding it. ``W += B @ A * s``
mutates the weight the artifact ALREADY POINTS AT (``load_constants(...,
user_managed=True)`` stores raw pointers), so the traced graph is untouched and
the adapter is simply part of the weights for the duration of the request.

**In place, and the originals are kept as bytes.** Two independent reasons the
mutation must be in place rather than a fresh tensor swapped onto the module:
the artifact holds a POINTER, so a swap would be invisible to it; and cudagraph
static inputs want pointer stability. Restore is therefore a ``copy_`` back from
a saved clone, never a ``sub_`` of the delta — subtracting is not the inverse of
adding in bf16, and a serial request stream would accumulate drift. Bit-exact
restore is by construction here, not by tolerance.

**What this does NOT do.** It does not fold onto a QUANTIZED leaf (fp8/gguf):
an fp8 grid cannot represent a small delta and the fold would be a silent
fidelity loss — those lanes keep the additive branch (:mod:`.w8a8_lora`), and
this module refuses by name. And it does not by itself settle AOTI's runtime
constant folding: ``compiler.py`` compiles with
``use_runtime_constant_folding=True``, the fold runs once on the first ``run()``
and never again, and an in-place weight write resets nothing — so any weight
feeding a ``from_folded`` constant would go stale. :func:`folded` therefore
takes ``rebind``, the caller-supplied seam that re-arms the artifact's constant
table, and REFUSES to run against a compiled-armed module without one.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from ..api.errors import RefCompatibilitySurprise
from . import w8a8_lora

logger = logging.getLogger(__name__)

#: One adapter as the fold takes it: (state dict, request weight, ref).
Adapter = Tuple[Dict[str, Any], float, str]

#: Called with the mutated denoiser after the weights change and again after
#: they are restored. The serve path passes ``aot_serve.rearm_constants``; a
#: pipeline with no compiled artifact passes nothing.
Rebind = Callable[[Any], None]


def _plain_leaf(module: Any) -> bool:
    """True for a leaf whose ``weight`` is the real, unquantized tensor.

    Quantized leaves are excluded rather than handled: an fp8 or GGML grid
    stores the weight through a scale/block code, so ``W += delta`` would round
    the delta away and serve a confidently wrong adapter. Those lanes have the
    additive branch for exactly this reason.
    """
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
    """A stable identity for one ADAPTER SET, order and weights included.

    Keys the fold cache. It hashes refs and weights, never tensor bytes — a
    ``ref`` is already fully pinned (``name@digest``) on every path that
    reaches serving, so hashing gigabytes to learn what the ref already says
    would be the expensive way to get the same answer.
    """
    h = hashlib.sha256()
    for _sd, weight, ref in adapters:
        h.update(str(ref).encode())
        h.update(b"\0")
        h.update(f"{float(weight):.12g}".encode())
        h.update(b"\0")
    return h.hexdigest()[:32]


def _delta(mod: Any, a: Any, b: Any, scale: float) -> Any:
    """``ΔW`` for one module, accumulated in fp32 and shaped like its weight.

    Linear: ``B[out, r] @ A[r, in]``. Conv2d: ``B`` is the 1×1 up half, so the
    contraction is over rank alone and ``A``'s kernel extent rides through —
    which is what makes a LoCon conv pair foldable at all.
    """
    import torch
    import torch.nn as nn

    a32 = a.detach().to(torch.float32)
    b32 = b.detach().to(torch.float32)
    if isinstance(mod, nn.Conv2d) or a32.dim() == 4:
        out = torch.einsum("or,rihw->oihw", b32.reshape(b32.shape[0], b32.shape[1]), a32)
    else:
        out = b32 @ a32
    return out.mul_(float(scale))


def compute_deltas(
    model: Any, adapters: Sequence[Adapter], *, pipe: Any = None,
) -> Dict[str, Any]:
    """``module path -> ΔW`` for one denoiser and the whole adapter set.

    The whole set settles into ONE delta per module before anything is written,
    so a refusal anywhere leaves the weights untouched — the same fail-closed
    ordering :func:`~.w8a8_lora.apply_branch_adapter_set` has.

    Normalization and key resolution are :mod:`.w8a8_lora`'s
    (``normalize_adapter_state_dict`` → ``split_state_dict`` → ``map_adapter``),
    not a second implementation: the kohya/SGM grammars, the alpha/rank scale
    and the typed refusals are already there and must not drift.
    """
    import torch

    mods = w8a8_lora.branch_modules(model)
    deltas: Dict[str, Any] = {}
    for state_dict, weight, ref in adapters:
        normalized = w8a8_lora.normalize_adapter_state_dict(
            pipe if pipe is not None else model, state_dict, ref=ref)
        den, _rest = w8a8_lora.split_state_dict(normalized)
        if not den:
            continue
        mapped = w8a8_lora.map_adapter(den, model, ref=ref)
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
        """Put every mutated weight back, byte for byte. Never raises."""
        for mod, original in reversed(self.saved):
            try:
                mod.weight.data.copy_(original)
            except Exception:  # noqa: BLE001 — a restore must finish
                logger.exception(
                    "lora fold: restoring %s failed; this denoiser now carries "
                    "an adapter into later requests", type(mod).__name__)
        self.saved.clear()


def apply_fold(model: Any, deltas: Mapping[str, Any]) -> FoldScope:
    """Write ``deltas`` into the model's weights IN PLACE; return the undo.

    Saves each original BEFORE the first write, so a failure mid-way is
    undone by the same path as a normal exit. The clone is the exact bytes,
    which is what makes the restore bit-exact rather than merely close.
    """
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
    """Whether a compiled artifact is currently serving this module's forward.

    Read off ``aot_serve``'s own marker rather than a second flag — the
    question "is something other than this module's forward serving" has one
    answer and it lives there.
    """
    from .. import aot_serve

    return aot_serve.serves_compiled(model)


@contextmanager
def folded(
    pipe: Any,
    adapters: Sequence[Adapter],
    *,
    request_id: str = "",
    rebind: Optional[Rebind] = None,
) -> Iterator[Dict[str, Any]]:
    """Serve one request with ``adapters`` folded into the pipeline's weights.

    ::

        with lora_fold.folded(pipe, riding, rebind=aot_serve.rearm_constants):
            pipe(...)

    Every branch-capable denoiser the pipeline carries is folded, and they move
    together: a refusal on the second expert leaves the first restored. Empty
    ``adapters`` is a no-op that still yields, so the call site needs no
    conditional.

    ``rebind`` is MANDATORY when a compiled artifact is armed. The artifact sees
    the in-place write through its user-managed pointer, but AOTI's runtime
    constant folding ran once at the first call and will not re-run on a bare
    mutation — so a weight feeding a folded constant would go stale and serve a
    confidently wrong tensor. Refusing here is the only honest default: the
    alternative is a green request whose numbers are silently wrong, which is
    precisely the defect this module exists to remove.
    """
    targets = w8a8_lora.branch_targets(pipe)
    if not adapters or not targets:
        yield {}
        return

    plan: List[Tuple[Any, Dict[str, Any]]] = []
    for model in targets.values():
        deltas = compute_deltas(model, adapters, pipe=pipe)
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
        "denoisers": len(plan),
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
            "[request_id=%s] lora folded into weights: %d denoiser(s), "
            "%d module(s), %d saved byte(s), set=%s",
            request_id, stats["denoisers"], stats["modules"],
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
