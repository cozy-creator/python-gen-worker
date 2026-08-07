"""The ONE flattening rule, for the mint and the serve side alike (pgw#994).

``torch.export`` does not take a call as written: it FLATTENS it. A container
argument occupies one caller parameter slot and produces one graph input per
pytree leaf. Every part of this SDK that has to line the two views up — the
mint building an ingress contract, the serve path binding a real call to it,
the declared-range gate resolving a declared name against an exported one —
needs the same answer to "which leaf is this", and each one used to compute it
for itself:

* pgw#790: ``input_contract`` zipped caller-side PARAMETER names against
  exported inputs, so sdxl's ``added_cond_kwargs`` shifted every later name by
  one and the recorded contract named the wrong tensors.
* pgw#993: the declared-range gate resolved ``Dim.carried_by``'s input by its
  DECLARED name against a program whose inputs were ``x_0``/``x_1``, which
  made a dim carried by a ``repeat=`` container unsatisfiable by construction.
  Cost: one rented A100.
* pgw#994 (this module): the contract recorded ``position`` as the index among
  FLATTENED inputs while ``bind_call_inputs`` matched it against the caller's
  PRE-flattening args, so a container family bound the whole list to element 0
  and shifted every later input.

Three instances of one defect, so the rule lives in exactly one place and
every consumer reads it from here.

MEASURED, on torch 2.13.0+cu130, because each of these had been assumed wrong
at least once:

* A sequence leaf is ``<param>_<index>``, for EVERY arity — a one-element
  container is ``x_0``, never ``x``.
* A mapping leaf is ``<param>_<key>`` (``added_cond_kwargs_text_embeds``).
* Nesting composes: ``img_shapes_0_0``.
* **Mappings flatten in INSERTION order, not sorted order.** The pre-pgw#994
  walk sorted keys "because that is what torch's pytree does"; it does not.
  ``{"zeta", "alpha", "mid"}`` exports as ``d_zeta, d_alpha, d_mid`` with the
  matching shapes, so a sorted walk pairs every leaf with another leaf's dtype
  and shape. sdxl escaped only because ``text_embeds`` < ``time_ids``.
* Non-tensor leaves still occupy a flat slot (``user_inputs`` carries the
  constants themselves: ``['hidden_states', 1, 4, 6, False, 'extra']``), so a
  walk that skipped them would desynchronise from the exported order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple, Union

#: One step into a container: a sequence index or a mapping key.
PathStep = Union[int, str]


@dataclass(frozen=True)
class Leaf:
    """One exported graph input, and where it came from in the CALL.

    :attr:`param` + :attr:`path` is the leaf's IDENTITY — the only thing that
    survives the round trip from the mint's example feed to a real serve call,
    because it describes the call's structure rather than a spelling of it.
    :attr:`name` is the serve-facing spelling and is what the ingress contract
    has always been keyed by; it is derived here so it cannot drift from the
    identity it names.
    """

    param: str
    #: Index of :attr:`param` in the traced call's parameter order — how the
    #: serve side finds it when the pipeline passes it positionally.
    param_position: int
    path: Tuple[PathStep, ...]
    value: Any = None

    @property
    def name(self) -> str:
        """The serve-facing name (pgw#790's spelling, kept deliberately).

        A mapping leaf takes its BARE KEY because that is the keyword the
        pipeline's own forward uses; a sequence leaf takes ``<param>.<index>``.
        This is the string the published contracts are keyed by and the one
        ``contract_digest`` folds into the key, so it is fixed — pgw#994 adds the
        identity next to it rather than renaming 144 live checkpoints.
        """
        name = self.param
        for step in self.path:
            # A mapping level REPLACES the name with the bare key; a sequence
            # level appends `.index`. Thread-through, which is pgw#790's walk
            # restated over the identity rather than a second copy of it.
            name = step if isinstance(step, str) else f"{name}.{step}"
        return name

    @property
    def exported_name(self) -> str:
        """The name ``torch.export`` gives this leaf's placeholder."""
        return exported_name(self.param, self.path)

    @property
    def trivial(self) -> bool:
        """True when the leaf IS its parameter — no container in between."""
        return not self.path


def exported_name(param: str, path: Sequence[PathStep] = ()) -> str:
    """``torch.export``'s placeholder name for one leaf of one parameter.

    ``('x', (0,)) -> 'x_0'``; ``('added_cond_kwargs', ('text_embeds',)) ->
    'added_cond_kwargs_text_embeds'``; ``('img_shapes', (0, 0)) ->
    'img_shapes_0_0'``. All three measured, not inferred.
    """
    return "_".join([str(param), *(str(step) for step in path)])


def flatten_call(
    param_names: Sequence[str], args: Sequence[Any], kwargs: Mapping[str, Any],
) -> Tuple[Leaf, ...]:
    """The call's leaves, in the order ``torch.export`` flattens them.

    ``param_names`` is the traced call's parameter order (positional first,
    then the keywords actually passed) — ``aot_mint._input_names``' output.
    """
    values = list(args) + [
        kwargs[name] for name in param_names[len(args):] if name in kwargs
    ]
    out: List[Leaf] = []
    for position, (param, value) in enumerate(zip(param_names, values)):
        _walk(param, position, (), value, out)
    return tuple(out)


def _walk(
    param: str, position: int, path: Tuple[PathStep, ...], value: Any,
    out: List[Leaf],
) -> None:
    if isinstance(value, Mapping):
        # INSERTION order: torch's pytree does not sort, and sorting here
        # pairs each leaf with another leaf's tensor (measured; see module
        # docstring).
        for key in value:
            _walk(param, position, path + (str(key),), value[key], out)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(param, position, path + (int(index),), item, out)
    else:
        out.append(Leaf(param=param, param_position=position, path=path,
                        value=value))


def resolve_leaf(
    param: str, param_position: int, path: Sequence[PathStep],
    args: Sequence[Any], kwargs: Mapping[str, Any],
) -> Tuple[bool, Any]:
    """``(found, value)`` for one leaf identity against a REAL call.

    The serve-side half of :func:`flatten_call`: the mint recorded where a
    leaf lives, this replays it. Keyword first, then the parameter's own
    position — never a search, because a search is what made sdxl's dict case
    pass by luck while z-image's list case could not pass at all.
    """
    if param in kwargs:
        container: Any = kwargs[param]
    elif 0 <= param_position < len(args):
        container = args[param_position]
    else:
        return False, None
    for step in path:
        if isinstance(step, str):
            if not isinstance(container, Mapping) or step not in container:
                return False, None
            container = container[step]
        else:
            if isinstance(container, (str, bytes)) or \
                    not isinstance(container, (list, tuple)):
                return False, None
            if step >= len(container):
                return False, None
            container = container[step]
    return True, container


__all__ = [
    "Leaf",
    "PathStep",
    "exported_name",
    "flatten_call",
    "resolve_leaf",
]
