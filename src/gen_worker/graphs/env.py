"""Where an artifact sits for THIS worker: the compile stack and the card.

torchcg addresses an artifact by ONE key over
``(graph x env x layout x policy x sm)`` and hands that key back from the mint.
This is the other half of the same fact, and it is pgw's: a serving worker holds
a release document that states a compile stack, sits on one card, and needs to
say whether the artifact the hub is offering was built for the machine it is
standing on.

It is deliberately NOT a second key. It never addresses anything -- the hub
route is keyed by GRAPH, and this is the local consistency guard that refuses an
answer minted for a different stack, plus a short handle to name it in the
refusal. Making it an address again is how the two-level scheme grew.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .._vendor.torchcg.identity import compile_stack
from .._vendor.torchcg.refuse import IdentityError

_SM_RE = re.compile(r"(sm_[0-9]{2,3}|cpu(-[a-z0-9_]+)?)\Z")


class EnvError(ValueError):
    """An artifact environment cannot state itself."""


@dataclass(frozen=True, slots=True)
class ArtifactEnv:
    """One (compile stack, card) position.

    ``stack`` is carried as the VERSIONS THEMSELVES rather than a digest, for
    the reason a refusal that can say ``torch 2.13.0 != 2.14.0`` is worth more
    than one saying two hashes differ -- and there is nothing to look up to
    expand it.
    """

    stack: tuple[tuple[str, str], ...]
    sm: str

    def __post_init__(self) -> None:
        try:
            selected = compile_stack(dict(self.stack))
        except IdentityError as exc:
            raise EnvError(str(exc)) from exc
        object.__setattr__(self, "stack", tuple(sorted(selected.items())))
        if not isinstance(self.sm, str) or _SM_RE.fullmatch(self.sm) is None:
            raise EnvError(
                f"env sm must be a concrete 'sm_NN' or cpu capability, got {self.sm!r}"
            )

    @property
    def block(self) -> dict[str, str]:
        """The env fingerprint block that feeds ``torchcg.artifact_key(env=)``.

        The HOST ISA facts are absent on purpose: ``torchcg.mint`` imposes them
        from the machine it runs on, so a key derived here matches only if this
        host is ISA-compatible with the one that minted -- which is the
        fail-closed behaviour we want.
        """

        return dict(self.stack)

    @property
    def value(self) -> str:
        payload = json.dumps(
            {"sm": self.sm, "stack": [list(row) for row in self.stack]},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return "cg-env-v5-" + hashlib.sha256(payload).hexdigest()[:32]

    def describe(self) -> str:
        rows = dict(self.stack)
        head = [f"{n} {rows[n]}" for n in ("torch", "triton") if n in rows]
        others = len(rows) - len(head)
        if others:
            head.append(f"+{others} more")
        return f"{', '.join(head)} @ {self.sm}"

    def __str__(self) -> str:
        return self.value


def require_env(
    stack: Mapping[str, str] | Sequence[tuple[str, str]], sm: str
) -> ArtifactEnv:
    rows = tuple(stack.items()) if isinstance(stack, Mapping) else tuple(stack)
    return ArtifactEnv(stack=rows, sm=sm)


__all__ = ["ArtifactEnv", "EnvError", "require_env"]
