"""Reading one declared scheduler block: the parsers, shared by every solver.

``recipe_v1`` records a scheduler as a NAME and a block of finite JSON scalars.
These are the readers that turn that block into typed fields, and every one of
them REFUSES rather than coerces — each value they read changes the sigma
ladder, and a ladder that changes silently is the failure mode this whole
package exists to remove.

Lifted out of ``model/scheduler.py`` (where pgw#1346 B2 wrote them) so the
solver modules can share them without importing the module that re-exports the
solvers. ``scheduler.py`` imports these back, so there is still exactly one
definition of each.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..errors import ModelError, ModelRefusal

#: What a scheduler block's values may be — ``recipe_v1``'s finite JSON scalars.
SchedulerValue = bool | int | float | str
SchedulerBlock = Mapping[str, SchedulerValue]


def refuse(name: str, wanted: str, value: object) -> ModelError:
    return ModelError(
        ModelRefusal.SCHEDULER_INVALID,
        f"scheduler parameter {name!r} must be {wanted}, got {value!r}",
    )


def flag(block: SchedulerBlock, name: str, default: bool) -> bool:
    """Read one declared boolean.

    Checked as ``bool`` and not as ``int`` even though ``bool`` IS an ``int`` in
    Python: a block that said ``use_karras_sigmas: 1`` means something the
    author did not write, and accepting it is how a schedule silently changes.
    """

    value = block.get(name, default)
    if not isinstance(value, bool):
        raise refuse(name, "a boolean", value)
    return value


def count(block: SchedulerBlock, name: str, default: int) -> int:
    """Read one declared integer. ``bool`` is refused, for the reason above."""

    value = block.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise refuse(name, "an integer", value)
    return value


def real(block: SchedulerBlock, name: str, default: float) -> float:
    """Read one declared real. An integer literal is a legal spelling of one."""

    value = block.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise refuse(name, "a real number", value)
    return float(value)


def choice(block: SchedulerBlock, name: str, default: str, allowed: tuple[str, ...]) -> str:
    """Read one declared string out of a CLOSED set.

    A misspelled spacing, objective or algorithm is the kind of parameter that
    changes the ladder silently rather than loudly, so the set is checked here
    rather than at the first step that reads it.
    """

    value = block.get(name, default)
    if not isinstance(value, str) or value not in allowed:
        raise refuse(name, f"one of {list(allowed)!r}", value)
    return value


def only(block: SchedulerBlock, known: tuple[str, ...]) -> SchedulerBlock:
    """Refuse a block carrying a parameter this scheduler does not read.

    The failure this prevents is silent and specific: ``EulerAncestralDiscrete``
    has no ``final_sigmas_type`` and ``Ddim`` has no ``final_sigmas_type``
    either, so a per-sampler block copied from the euler one keeps a key that
    changes nothing — the declaration says one thing and the ladder does
    another, with no error anywhere. Every reader below is total over its own
    parameters, so an unknown key can only be a mistake (pgw#1346 K10).
    """

    unknown = sorted(set(block) - set(known))
    if unknown:
        raise ModelError(
            ModelRefusal.SCHEDULER_INVALID,
            f"scheduler parameter {unknown[0]!r} is not read by this scheduler; it reads "
            f"{list(known)!r}",
        )
    return block


__all__ = [
    "SchedulerBlock",
    "SchedulerValue",
    "choice",
    "count",
    "flag",
    "only",
    "real",
    "refuse",
]
