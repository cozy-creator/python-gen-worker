"""pgw#995 — the hub's env-delivery chain, modelled so local tests can see it.

THE BLIND SPOT THIS CLOSES. The local mint rig (pgw#978, since deleted) ran the
whole mint machinery on this box and was why a change could be proven before
PyPI. But it **constructed its own environment**: the mint child gets `mint_process.child_env`
plus a few rig keys, and the adopting process gets `dict(os.environ)`. Neither
resembles how a production pod is given its env, so the chain below was
invisible to every test in this repo:

    worker function declares env  ->  build schema  ->  release_env_declarations
    operator sets a value         ->  endpoint_env_entries (+ Vault)
    pod launch                    ->  EndpointEnvService.Resolve  ->  pod env
                                                                       |
                                                     config.loader ----+--> Settings

That chain is exactly what took `GEN_WORKER_PREFER_AOT` dark. The flag was
declared by the worker function and set on the endpoint; a release rebuild
stopped declaring the name, the hub withheld every matching entry **silently**,
and three pod attempts went by before anyone noticed the AOT path was off. Every
component was individually correct. The DELIVERY was what broke, and delivery
was the one thing nothing tested.

WHAT THIS IS. A faithful model of `EndpointEnvService.Resolve`'s contract — NOT
a reimplementation of the hub. It carries the one rule that matters (an entry
reaches the pod only if the release DECLARES its name) and the reserved-name
defence, and it reports withholdings with the same typed vocabulary the hub uses
(th#1650). It is deliberately small: a big fake hub would drift from the real
one and start certifying its own behaviour.

WHAT IT IS NOT. It does not model Vault, `applies_to` version/tag matching, or
the mTLS resolve path. Those belong to the full HelloAck-shaped boot filed on
pgw#995; this is the seam plus the first regression it makes visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

#: Mirrors tensorhub `internal/api/endpoint_env_withheld.go` (th#1650). Kept
#: verbatim so a log line from a real pod and a failure from this harness use
#: the same word for the same thing.
WITHHELD_UNDECLARED = "undeclared_by_release"
WITHHELD_RESERVED = "reserved_name"

#: Mirrors the hub's `reservedEnvPrefixes` narrowly — only what a worker test
#: can meaningfully trip. The hub is the authority; this exists so a test that
#: sets a platform name gets the platform answer rather than a false pass.
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
    """What THIS release's worker functions declare, per `release_env_declarations`.

    Per-RELEASE, which is the property that makes the postmortem possible: a
    rebuild produces a new declaration set while the operator's entries are
    per-ENDPOINT and long-lived.
    """

    names: Tuple[str, ...] = ()

    @classmethod
    def of(cls, *names: str) -> "ReleaseEnvDeclarations":
        return cls(tuple(names))


@dataclass
class EndpointEnvEntries:
    """What the operator set, per `endpoint_env_entries`. Long-lived."""

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
    """The hub's rule: an entry reaches the pod only if the release declares it.

    Returns the delivered map AND every withholding, because a delivery that
    reports only what arrived is the exact shape that hid the defect — "nothing
    arrived" and "nothing was ever configured" have to be distinguishable.
    """
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
    """The environment a pod actually boots with: image env + delivered entries.

    `strip` removes names the launching process happens to carry — without it a
    rig running on a developer box would let an ambient export stand in for a
    hub-delivered value and prove nothing. That substitution IS the blind spot;
    a rig mode that allowed it would be decoration.
    """
    out = {k: v for k, v in base.items() if k not in set(strip)}
    out.update(delivery.env)
    return out


def declared_by(function_env: Sequence[str]) -> ReleaseEnvDeclarations:
    """Build declarations the way a build does: from the function's own list.

    Real builds read this off the function schema payload. A worker function
    that stops listing a name produces a release that does not declare it —
    with no error anywhere, because an empty `env` list is legal.
    """
    return ReleaseEnvDeclarations(tuple(function_env))
