"""The symbolic-shape facts the export handoff must carry (pgw#998).

THE DEFECT, measured on torch 2.13.0+cu130, CPU, by the micro-mint rig.
``aot_compile_pool`` saves each ``ExportedProgram`` and ``aot_compile_child``
loads it in another interpreter — the process boundary is the pool's whole
point. The round trip does not preserve the ShapeEnv's symbol VALUES in the
form inductor reads them:

    parent   var_to_val   {s11: 32, s37: 32, s18: 16, s57: 16}
             replacements {s11: 2*s18, s37: 2*s57}
    child    var_to_val   {2*s18: 32, 2*s57: 32, 4*s18*s57: 1024}
             replacements {}

The child's map is keyed by the size EXPRESSIONS, not by the free symbols.
``torch.fx.experimental._size_hinting`` resolves an extent by substituting
``shape_env.backed_var_to_val`` into it, so an extent that is literally one of
those keys still resolves — and every other one silently cannot::

    LoweringException: RuntimeError: ('unexpected None!', 512*s18*s57)
      target: aten.addmm.default

**The trigger is a DERIVED symbol, not nonlinearity.** A declaration whose
dims carry ``multiple_of`` exports each axis as ``2*s18``; a matmul M that
multiplies two such axes is ``512*s18*s57``, which appears in no key. Measured
both ways: the same graph with the same product extent and NO ``multiple_of``
compiles fine across the same round trip (its keys are the bare symbols), and
the same declaration WITH ``multiple_of`` fails. Nonlinearity is what makes an
extent stop being a key; the coefficient is what makes the keys wrong.

THE FIX, and why it is shaped this way. The parent's ShapeEnv is the ONE
authority for these values — it is the process that traced the graph. So the
parent sends them (:func:`symbol_values`) and the child restores them
(:func:`restore_symbol_values`); nothing re-derives a value, and nothing
infers one from the serialized expressions. Every save/load site in the mint
uses this module, so the handoff contract has one spelling — the same
invariant pgw#993 and pgw#994 settled for the flattening contract.

:func:`unhinted_extents` is the safety net: if an extent is still unrealizable
after the restore, the mint refuses NAMING THE INPUT, THE AXIS AND THE
DECLARED DIM. ``512*s18*s57`` cost an hour of bisection because it names
nothing an author wrote.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


def shape_env(program: Any) -> Optional[Any]:
    """The ``ShapeEnv`` behind an exported program's placeholders, if any."""
    graph = getattr(getattr(program, "graph_module", None), "graph", None)
    for node in getattr(graph, "nodes", ()) or ():
        if getattr(node, "op", "") != "placeholder":
            continue
        for dim in getattr(node.meta.get("val"), "shape", ()) or ():
            env = getattr(getattr(dim, "node", None), "shape_env", None)
            if env is not None:
                return env
    return None


def symbol_values(program: Any) -> Dict[str, int]:
    """``{symbol name: concrete value}`` for every FREE symbol the parent knows.

    Read off the tracing process's own ``backed_var_to_val`` and keyed by
    symbol NAME, which is what survives serialization: the symbols themselves
    are rebuilt by the deserializer, so an object identity would not carry.
    """
    env = shape_env(program)
    out: Dict[str, int] = {}
    for key, value in _values(env).items():
        name = getattr(key, "name", None)
        if not name:
            # A composite key (`2*s18`) is the child's disease, not a fact
            # worth shipping: the free symbols below carry the same
            # information in the form that resolves.
            continue
        try:
            out[str(name)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def symbol_labels(program: Any) -> Dict[str, str]:
    """``{symbol name: the dim name the AUTHOR wrote}``, best effort.

    ``var_to_sources`` connects each symbol to where it came from:
    ``s18 -> 'H_lat_u'`` for a base symbol (the name the author wrote) and
    ``s11 -> "L['flat_args'][0].size()[1]"`` for the derived alias, whose
    declared spelling (``2*H_lat_u``) is in ``source_name_to_debug_name``.
    Both are DEBUG surfaces and neither survives serialization, which is why
    this is read in the parent and shipped: a refusal in the child that can
    only say ``512*s18*s57`` names nothing an author wrote.
    """
    env = shape_env(program)
    debug = getattr(env, "source_name_to_debug_name", None) or {}
    sources = getattr(env, "var_to_sources", None) or {}
    out: Dict[str, str] = {}
    for symbol, entries in sources.items():
        name = str(getattr(symbol, "name", symbol))
        for source in entries or ():
            # `Source.name` is a method on some torch versions and a plain
            # string on others; both spellings key the same debug map.
            raw = getattr(source, "name", None)
            key = str(raw() if callable(raw) else raw or "")
            # Two spellings, both authored: a DERIVED symbol's source is the
            # input expression and the debug map holds the declared form
            # (`2*H_lat_u`); a BASE symbol's source IS the declared name
            # (`H_lat_u`) with no debug entry. The base symbols are the ones
            # extents are written in, so both are worth carrying.
            label = debug.get(key) or (key if key.isidentifier() else "")
            if label:
                out[name] = str(label)
                break
    return out


def restore_symbol_values(program: Any, values: Mapping[str, int]) -> int:
    """Put the parent's symbol values back into a freshly LOADED program.

    Returns how many symbols were restored.
    """
    if not values:
        return 0
    env = shape_env(program)
    table = getattr(env, "backed_var_to_val", None) if env is not None else None
    if not isinstance(table, dict):
        # No env, or a torch whose table is not a dict: the gate below still
        # refuses by name, so this stays silent rather than guessing.
        return 0
    import sympy

    symbols: Dict[str, Any] = {}
    for key in list(_values(env)):
        for symbol in getattr(key, "free_symbols", ()) or ():
            symbols[str(getattr(symbol, "name", symbol))] = symbol
    for shape in _placeholder_shapes(program):
        for dim in shape:
            expr = getattr(getattr(dim, "node", None), "expr", None)
            for symbol in getattr(expr, "free_symbols", ()) or ():
                symbols[str(getattr(symbol, "name", symbol))] = symbol

    restored = 0
    for name, symbol in symbols.items():
        if name not in values:
            continue
        table[symbol] = sympy.Integer(int(values[name]))
        restored += 1
    if restored:
        logger.info(
            "aot-shape-hints: restored %d symbol value(s) after the export "
            "handoff (pgw#998)", restored)
    return restored


def unhinted_extents(
    program: Any, labels: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Extents inductor could not turn into a number, named for a HUMAN.

    One line per (input, axis) whose size expression still carries a symbol
    with no value — with the DECLARED dim name when the program records one,
    because the symbol name is the thing that names nothing.
    """
    env = shape_env(program)
    if env is None:
        return []
    known = {
        str(getattr(key, "name", ""))
        for key in _values(env)
        if getattr(key, "name", None)
    }
    named = dict(labels or {})
    out: List[str] = []
    for name, shape in _named_placeholder_shapes(program):
        for axis, dim in enumerate(shape):
            expr = getattr(getattr(dim, "node", None), "expr", None)
            free = {
                str(getattr(s, "name", s))
                for s in (getattr(expr, "free_symbols", ()) or ())
            }
            missing = sorted(free - known)
            if not missing:
                continue
            declared = sorted({named[s] for s in missing if s in named})
            said = f", declared dim(s) {declared!r}" if declared else ""
            out.append(
                f"{name}[{axis}] has size {expr} whose symbol(s) "
                f"{missing!r} carry no value{said}")
    return out


def _placeholder_shapes(program: Any) -> List[Tuple[Any, ...]]:
    return [shape for _name, shape in _named_placeholder_shapes(program)]


def _named_placeholder_shapes(program: Any) -> List[Tuple[str, Tuple[Any, ...]]]:
    graph = getattr(getattr(program, "graph_module", None), "graph", None)
    signature = getattr(program, "graph_signature", None)
    user_inputs = {str(n) for n in getattr(signature, "user_inputs", ()) or ()}
    out: List[Tuple[str, Tuple[Any, ...]]] = []
    for node in getattr(graph, "nodes", ()) or ():
        if getattr(node, "op", "") != "placeholder":
            continue
        name = str(getattr(node, "name", ""))
        if user_inputs and name not in user_inputs:
            continue
        shape = tuple(getattr(node.meta.get("val"), "shape", ()) or ())
        if shape:
            out.append((name, shape))
    return out


def _values(env: Any) -> Dict[Any, Any]:
    """The ShapeEnv's backed symbol table. ``var_to_val`` is deprecated on the
    pin and warns; ``backed_var_to_val`` is the same table under the name torch
    now maintains."""
    return dict(getattr(env, "backed_var_to_val", None) or {})


__all__ = [
    "restore_symbol_values",
    "shape_env",
    "symbol_labels",
    "symbol_values",
    "unhinted_extents",
]
