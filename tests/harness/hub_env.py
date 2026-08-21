from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

WITHHELD_UNDECLARED = "undeclared_by_release"
WITHHELD_RESERVED = "reserved_name"

RESERVED_PREFIXES: Tuple[str, ...] = (
    "WORKER_", "TENSORHUB_", "ORCHESTRATOR_", "HF_HOME", "RUNPOD_",
    "GEN_WORKER_C2PA_", "GEN_WORKER_PROCESS_SPLIT", "GEN_WORKER_COMPUTE_CHILD",
    "GEN_WORKER_CHILD_",
)


@dataclass(frozen=True)
class Withheld:
    """One entry the operator set that this pod will not receive."""

    name: str
    reason: str
    detail: str = ""


@dataclass
class ReleaseEnvDeclarations:
    """What THIS release's worker functions declare, per `release_env_declarations`."""

    names: Tuple[str, ...] = ()

    @classmethod
    def of(cls, *names: str) -> "ReleaseEnvDeclarations":
        return cls(tuple(names))


@dataclass
class EndpointEnvEntries:
    """What the operator set, per `endpoint_env_entries`."""

    values: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Delivery:
    """The outcome of one pod launch's env resolution."""

    env: Dict[str, str]
    withheld: Tuple[Withheld, ...]

    def withheld_names(self) -> List[str]:
        return [w.name for w in self.withheld]


def is_reserved(name: str) -> bool:
    return any(name.startswith(p) for p in RESERVED_PREFIXES)


def resolve(
    declarations: ReleaseEnvDeclarations,
    entries: EndpointEnvEntries,
) -> Delivery:
    """The hub's rule: an entry reaches the pod only if the release declares it."""
    declared = set(declarations.names)
    env: Dict[str, str] = {}
    withheld: List[Withheld] = []
    for name in sorted(entries.values):
        if name not in declared:
            withheld.append(Withheld(
                name, WITHHELD_UNDECLARED,
                f"release declares {len(declared)} env name(s), not this one"))
            continue
        if is_reserved(name):
            withheld.append(Withheld(
                name, WITHHELD_RESERVED, "platform-reserved namespace"))
            continue
        env[name] = entries.values[name]
    return Delivery(env=env, withheld=tuple(withheld))


def pod_environ(
    base: Mapping[str, str],
    delivery: Delivery,
    *,
    strip: Iterable[str] = (),
) -> Dict[str, str]:
    """The environment a pod actually boots with: image env + delivered entries."""
    out = {k: v for k, v in base.items() if k not in set(strip)}
    out.update(delivery.env)
    return out


def declared_by(function_env: Sequence[str]) -> ReleaseEnvDeclarations:
    """Build declarations the way a build does: from the function's own list."""
    return ReleaseEnvDeclarations(tuple(function_env))
