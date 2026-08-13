"""Can this pipeline, AS LOADED, be traced at all?

A pipeline that carries `torch.compiler.disable`d work in its forward path
cannot be exported. `torch.export(strict=True)` does not degrade around it — it
refuses, per graph class, with `Unsupported: Skip inlining
torch.compiler.disable()d function`.

THE CASE THIS EXISTS FOR. Offload hooks. diffusers' group offloading marks
`ModuleGroup.onload_` (verified on diffusers 0.39.0:
`ModuleGroup.onload_._torchdynamo_disable is True`); accelerate's model/
sequential offload marks `CpuOffload.pre_forward`. Either puts disabled work in
a forward path, and then every one of a family's declared graph classes refuses,
one at a time, for the same reason.

**It is not a card-size story.** sdxl refused on a 16 GiB A4000 and z-image on a
**48 GiB A40 holding a 19 GiB model** — so offload here is a pipeline
CONFIGURATION, not a response to memory pressure. This module therefore asks
about HOOKS and never about capacity, which is also what §1.35 requires: every
model runs on every GPU, feasibility is never asked.

WHY A PRE-EXPORT CHECK RATHER THAN LETTING THE PER-ENTRY SKIP HANDLE IT.
The per-entry skip would dutifully skip all 36 and publish nothing: thirty-six
typed refusals and an hour of wall clock to say once what is knowable
before the first export begins. The per-entry skip is for a class that is
individually unexportable; this is the whole PIPELINE being untraceable as
loaded, which is a different fact and deserves its own sentence.

It reads no placement logic and does no card arithmetic — it asks the object in
front of it whether it carries disabled work. That keeps it true for any future
source of such hooks, not just the offload rung that produced the first one.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Tuple

#: The attribute `torch.compiler.disable` stamps on what it wraps. Read rather
#: than inferred from a class name: the marker is torch's own contract and
#: survives diffusers renaming or re-homing its hooks.
DISABLE_MARKER = "_torchdynamo_disable"

#: How far into a hook's own object graph to look. A registered hook does not
#: usually CARRY the disabled function — diffusers' `GroupOffloadingHook` holds
#: a `ModuleGroup` and the marker is on `ModuleGroup.onload_` — so one hop past
#: the hook is required to see it at all. Two hops is where it stops: deeper is
#: someone else's object graph, and an unbounded walk over live model state is
#: how a diagnostic becomes a hang.
_DEPTH = 2


def _is_disabled(obj: Any) -> bool:
    return bool(getattr(obj, DISABLE_MARKER, False))


def _disabled_in(obj: Any, depth: int = _DEPTH) -> str:
    """The name of a `torch.compiler.disable`d callable reachable from ``obj``.

    Empty string when there is none. Checks the object, its class's methods,
    and (one hop at a time) the objects it holds — which is what it takes to
    see `hook.group.onload_` from a registered hook.
    """
    if obj is None or depth < 0:
        return ""
    if _is_disabled(obj):
        return getattr(obj, "__qualname__", None) or type(obj).__name__
    # A bound method's owner, and the class's own methods: the marker lives on
    # the FUNCTION, so it is reached through the type rather than the instance.
    for holder in (obj, type(obj)):
        try:
            names = list(vars(holder))
        except TypeError:
            continue
        for name in names:
            if name.startswith("__"):
                continue
            try:
                attr = getattr(holder, name)
            except Exception:  # noqa: BLE001 — a property that raises is not a hook
                continue
            if _is_disabled(attr):
                owner = getattr(type(obj), "__name__", "?")
                return f"{owner}.{name}"
    if depth == 0:
        return ""
    try:
        held = list(vars(obj).values())
    except TypeError:
        return ""
    for value in held:
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            continue
        found = _disabled_in(value, depth - 1)
        if found:
            return found
    return ""


def _hooks_of(module: Any) -> Iterator[Tuple[str, Any]]:
    """Every hook registered on ``module``, however it was registered.

    Both torch's own dicts and diffusers' `HookRegistry` (`_diffusers_hook`),
    because group offloading uses the registry and a walk that only knew
    torch's dicts would see nothing at all on the case this module exists for.
    """
    for attr in ("_forward_pre_hooks", "_forward_hooks",
                 "_forward_pre_hooks_with_kwargs"):
        table = getattr(module, attr, None)
        if isinstance(table, dict):
            for hook in list(table.values()):
                yield attr, hook
    registry = getattr(module, "_diffusers_hook", None)
    if registry is not None:
        hooks = getattr(registry, "hooks", None)
        if isinstance(hooks, dict):
            for name, hook in list(hooks.items()):
                yield f"_diffusers_hook[{name}]", hook
        elif hooks is not None:
            for hook in list(hooks):
                yield "_diffusers_hook", hook


def untraceable_hooks(pipeline: Any) -> Tuple[Tuple[str, str, str], ...]:
    """``(module path, where the hook was registered, the disabled callable)``
    for every hook on ``pipeline`` that puts `torch.compiler.disable`d work in
    a forward path. Empty means the pipeline is traceable as loaded.

    Walks the pipeline's component modules and their submodules — offload hooks
    are installed per LEAF group, not on the pipeline object.
    """
    try:
        import torch.nn as nn
    except Exception:  # noqa: BLE001 — torch-less: nothing to trace either
        return ()

    found: List[Tuple[str, str, str]] = []
    seen: set[int] = set()
    components = []
    if isinstance(pipeline, nn.Module):
        components.append(("", pipeline))
    for name, value in list(vars(pipeline).items()
                            if hasattr(pipeline, "__dict__") else []):
        if isinstance(value, nn.Module):
            components.append((str(name).lstrip("_"), value))
    for comp_name, comp in components:
        for path, module in comp.named_modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            for where, hook in _hooks_of(module):
                disabled = _disabled_in(hook)
                if not disabled:
                    continue
                full = ".".join(p for p in (comp_name, path) if p) or comp_name
                found.append((full or "<root>", where, disabled))
    return tuple(found)


def untraceable_reason(pipeline: Any) -> str:
    """One sentence naming what makes this pipeline untraceable, or ""."""
    hits = untraceable_hooks(pipeline)
    if not hits:
        return ""
    shown = "; ".join(
        f"{path} ({where} -> {fn})" for path, where, fn in hits[:3])
    more = f" and {len(hits) - 3} more" if len(hits) > 3 else ""
    return (
        f"{len(hits)} module(s) carry `torch.compiler.disable`d work in their "
        f"forward path: {shown}{more}"
    )


__all__ = [
    "DISABLE_MARKER",
    "untraceable_hooks",
    "untraceable_reason",
]
