"""Can this pipeline, AS LOADED, be traced at all? A pipeline that carries `torch.compiler.disable`d work in its forward path cannot be exported."""

from __future__ import annotations

from typing import Any, Iterator, List, Tuple

DISABLE_MARKER = "_torchdynamo_disable"

_DEPTH = 2


def _is_disabled(obj: Any) -> bool:
    return bool(getattr(obj, DISABLE_MARKER, False))


def _disabled_in(obj: Any, depth: int = _DEPTH) -> str:
    if obj is None or depth < 0:
        return ""
    if _is_disabled(obj):
        return getattr(obj, "__qualname__", None) or type(obj).__name__
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
    """``(module path, where the hook was registered, the disabled callable)`` for every hook on ``pipeline`` that puts `torch.compiler.disable`d work in a forward path."""
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
