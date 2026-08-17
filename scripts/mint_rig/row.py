"""The row — one rental, reduced to a record a matrix can hold.

pgw#1346's W2 lanes need a table of (family, card, lane) cells; pgw#1331 needs
one row that says a real compile happened and what it produced. This is that
row, and the fields are chosen so nobody has to re-derive anything from prose:

  * **asked vs observed.** `asked_gpu` is the plan, `observed_gpu` and
    `observed_sm` come from the pod's own `nvidia-smi`. e2e's matrix records the
    same split after finding rows that measured an intention.
  * **the driver path.** `native` or `compat` — a wall measured through a
    forward-compat libcuda is a fact about the report (RIG-ENV §3c), not a
    detail, so it is a column.
  * **cost from runtime x rate**, with the rate the create call actually
    returned. This PRICES a rental; it never reconciles a bill. e2e's
    `privatedeploy/cost.go` states the same boundary, and for the same reason:
    the ledger's own settlement figures are the charge.
  * **three teardown verdicts**, not one boolean.
  * **digests**, so "which code ran" is answerable a month later.

`verdict` is a small closed vocabulary rather than a bool, because the
interesting failures are not "the command exited non-zero":

  green     the command printed its done marker
  red       the command failed — a marker, a non-zero rc, a refusal
  stuck     progress stopped (:class:`~mint_rig.progress.Stuck`)
  railed    the spend rail tripped first
  reroll    the host's driver has no usable path to the fleet CUDA line
  refused   the rig would not start (no rail, no podguard, no key)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Artifact:
    remote: str
    local: str = ""
    bytes: int = 0
    sha256: str = ""
    fetched: bool = False
    note: str = ""


@dataclass
class Teardown:
    """Three independent verdicts. Any one of them alone has lied before."""

    #: The DELETE call did not raise. Weakest of the three: RunPod answers 200
    #: to a delete whose pod is still winding down.
    delete_issued: bool = False
    #: GET /pods/<id> answered 404. The pod's own record is gone.
    get_404: bool = False
    #: The id is absent from GET /pods. Catches the case where the individual
    #: GET is cached or eventually-consistent but the account listing is not.
    absent_from_list: bool = False
    attempts: int = 0
    note: str = ""

    @property
    def confirmed(self) -> bool:
        return self.get_404 and self.absent_from_list


@dataclass
class RigRow:
    """One rental. Serialise with :meth:`dumps`; a matrix is a list of these."""

    lane: str
    issue: str = ""
    verdict: str = "refused"
    failed_stage: str = ""
    detail: str = ""

    pod_id: str = ""
    pod_name: str = ""
    image: str = ""

    asked_gpu: str = ""
    asked_gpu_type_ids: list[str] = field(default_factory=list)
    observed_gpu: str = ""
    observed_sm: str = ""
    sm_expected: str = ""
    driver_version: str = ""
    cuda_path: str = ""  # native | compat | reroll

    command: str = ""
    workload_name: str = ""
    workload_digest: str = ""
    uploads: list[dict[str, str]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    env_line: list[str] = field(default_factory=list)

    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    pod_seconds: float = 0.0
    rate_per_hr: float = 0.0
    est_cost_usd: float = 0.0
    rail_usd: float = 0.0
    rail_tripped: bool = False

    teardown: Teardown = field(default_factory=Teardown)
    killset_path: str = ""
    stage_trail: list[dict[str, Any]] = field(default_factory=list)

    def stage(self, name: str, note: str = "") -> None:
        self.stage_trail.append({"stage": name, "at": round(time.time(), 3), "note": note})

    def price(self, now: float | None = None) -> None:
        """Runtime x the rate the provider actually charged. Never a bill."""
        self.ended_at = time.time() if now is None else now
        self.pod_seconds = max(0.0, self.ended_at - self.started_at)
        self.est_cost_usd = round(self.rate_per_hr * self.pod_seconds / 3600.0, 4)

    def record(self) -> dict[str, Any]:
        return asdict(self)

    def dumps(self) -> str:
        return json.dumps(self.record(), indent=2, sort_keys=True) + "\n"

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.dumps())
        return path

    def one_line(self) -> str:
        gpu = self.observed_gpu or self.asked_gpu
        return (
            f"{self.verdict.upper():8s} {self.lane:24s} {gpu:22s} sm{self.observed_sm or '?':4s} "
            f"{self.pod_seconds / 60:6.1f}min ${self.est_cost_usd:6.3f} "
            f"teardown={'ok' if self.teardown.confirmed else 'UNCONFIRMED'} "
            f"{self.failed_stage or self.workload_name}"
        )
