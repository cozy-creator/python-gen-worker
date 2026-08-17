"""The kill-set: how to stop this pod, written down BEFORE the pod exists.

THE FAILURE THIS EXISTS FOR. A pod id is born inside the create call's response.
If the box dies between the POST leaving and the response landing — an agent
session killed, a network drop, an OOM — a pod is running on the account and
nothing anywhere names it. podguard's th#1323 incident is the same shape one
step later (four 4090s, four unattended hours, $12.42) and it fixed the case
where the renter dies *after* the id is known. This module closes the window
before that.

The mechanism is a two-phase record keyed on the pod NAME, which we choose:

  1. Before the POST, write ``{"state": "PENDING", "pod_name": ...}`` and fsync.
     The name is unique per invocation, so `kill_by_name` can find the pod on
     the account even though nobody ever saw its id.
  2. After the response, rewrite with the id and the direct `kill` command.
  3. On verified teardown, rewrite with ``"state": "RELEASED"``.

A record left PENDING or LIVE is the alarm. `python -m mint_rig sweep` lists
them, and every record carries its own literal kill command as a string, so
recovery is copy-and-paste rather than reconstruction.

This is deliberately SEPARATE from podguard's lease record even though the two
overlap: podguard's record is written from the create RESPONSE (`Lease` is
constructed from `pod.get("id")`), so it has the same blind spot. The rig arms
podguard as well — both layers, not either.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: The command a record names as the way to stop its pod, as an absolute path.
ENTRYPOINT = Path(__file__).resolve().parents[1] / "pod_rig.py"

#: Queryable records, not someone remembering to look. Same home shape as
#: podguard's lease directory so an operator has two places, not twelve.
KILLSET_DIR = Path(
    os.environ.get("MINT_RIG_KILLSET_DIR", Path.home() / ".cache" / "cozy-mint-rig" / "killset")
)


@dataclass
class KillSet:
    """One invocation's stoppability record."""

    lane: str
    pod_name: str
    state: str = "PENDING"  # PENDING | LIVE | RELEASED | LEAKED
    pod_id: str = ""
    gpu: str = ""
    image: str = ""
    rate_per_hr: float = 0.0
    rail_usd: float = 0.0
    opened_at: float = field(default_factory=time.time)
    closed_at: float = 0.0
    renter_pid: int = field(default_factory=os.getpid)
    note: str = ""
    # A FACTORY, not a value: `KILLSET_DIR` is read when a record is made, so a
    # test (or an operator's `MINT_RIG_KILLSET_DIR`) redirecting it is honoured
    # by every record rather than only by the ones made after the import.
    root: Path = field(default_factory=lambda: KILLSET_DIR, compare=False, repr=False)

    # ---- the literal commands, as strings, so recovery is paste-not-reconstruct
    #
    # ABSOLUTE, deliberately. A record is read by whoever finds a pod still
    # billing — possibly a different agent, in a different checkout, hours
    # later — and a relative command is a puzzle at exactly the wrong moment.
    @property
    def kill_by_name(self) -> str:
        return f"python3 {ENTRYPOINT} terminate --name {self.pod_name}"

    @property
    def kill_by_id(self) -> str:
        return f"python3 {ENTRYPOINT} terminate --pod {self.pod_id}" if self.pod_id else ""

    @property
    def path(self) -> Path:
        return self.root / f"{self.pod_name}.json"

    def record(self) -> dict[str, Any]:
        body = {k: v for k, v in asdict(self).items() if k != "root"}
        body["kill_by_name"] = self.kill_by_name
        body["kill_by_id"] = self.kill_by_id
        return body

    def save(self) -> "KillSet":
        """Write durably. `fsync` is not ceremony here: the whole point is to
        survive a host that stops between this line and the next one."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w") as fh:
            json.dump(self.record(), fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(self.path)
        return self

    def arm(self, pod_id: str, *, rate_per_hr: float = 0.0) -> "KillSet":
        self.pod_id, self.rate_per_hr, self.state = pod_id, rate_per_hr, "LIVE"
        return self.save()

    def close(self, *, confirmed_dead: bool, note: str = "") -> "KillSet":
        self.state = "RELEASED" if confirmed_dead else "LEAKED"
        self.closed_at = time.time()
        self.note = note
        return self.save()

    # ---- the sweep
    @classmethod
    def open_records(cls, root: Path | None = None) -> list[dict[str, Any]]:
        """Every record that is not RELEASED — the alarm list.

        A PENDING record with no id is the create-window leak and is reported
        first, because it is the only one no other tool in the workspace can
        see.
        """
        base = root or KILLSET_DIR
        out: list[dict[str, Any]] = []
        for path in sorted(base.glob("*.json")):
            try:
                body = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if body.get("state") != "RELEASED":
                body["record_path"] = str(path)
                out.append(body)
        out.sort(key=lambda r: (r.get("state") != "PENDING", r.get("opened_at", 0.0)))
        return out
