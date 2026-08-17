"""pgw#1347 — the pod mint-rig: rent a pod, run one named command, bank the row.

This package is the primitive pgw#1331 named as missing and pgw#1346's matrix
lanes are built on: *rent a pod by (gpu type, image), ship a workload, run a
named command, capture the artifacts, tear down with double verification, and
emit one machine-readable row.*

It is deliberately NOT another bespoke lane script. `research/RIG-ENV.md` §5
records what bespoke costs: two false verdicts and ~$11 of rental, both from a
rig built by copying an older rig's `setup.sh` forward. The layers here are
separated so a new lane writes a :class:`~mint_rig.workload.Workload` — a data
value — and never a new pod driver.

    from mint_rig import Rig, Rail, Workload, cards

    rig = Rig(rail=Rail(max_usd=2.0), lane="pgw1331-clip")
    row = rig.run(cards.pick("a40"), Workload.mint_family(...))

THE FOUR RULES THIS PACKAGE ENFORCES RATHER THAN DOCUMENTS

1. **A spend rail is mandatory.** :class:`~mint_rig.rail.Rail` cannot be
   constructed without a dollar cap and :class:`~mint_rig.driver.Rig` cannot be
   constructed without a Rail. There is no default, because a default is the
   number nobody chose.
2. **The kill-set is written before the pod exists.** See
   :mod:`mint_rig.killset`: the record lands on disk *before* the POST, keyed by
   the name we are about to ask for, so a host crash inside the create call
   still leaves a stoppable pod.
3. **No magic timeouts.** Every wait is a :class:`~mint_rig.progress.Gate` over
   a goal predicate and a progress token. Stuck means "the token stopped
   advancing", never "the clock ran out". The only wall-clock bound in the
   package is the money one, and it is the operator's declared dollars divided
   by the pod's OBSERVED rate.
4. **Teardown is verified three ways** — DELETE, then GET 404, then absent from
   the account listing — and all three verdicts land in the row.
"""

from __future__ import annotations

from .driver import Rig
from .killset import KillSet
from .progress import Gate, Observation, Stuck
from .rail import Rail, RailTripped
from .row import Artifact, RigRow, Teardown
from .runpod import PodApi, RunpodRest
from .transport import SshTransport, Transport
from .workload import Upload, Workload

__all__ = [
    "Artifact",
    "Gate",
    "KillSet",
    "Observation",
    "PodApi",
    "Rail",
    "RailTripped",
    "Rig",
    "RigRow",
    "RunpodRest",
    "SshTransport",
    "Stuck",
    "Teardown",
    "Transport",
    "Upload",
    "Workload",
]
