"""Parent-side authorization for child-requested hub actions (pgw#763 delta 1).

The worker JWT is the pod's signing identity: it authenticates the gRPC stream,
mints per-job capability tokens, publishes cells, and authorizes the platform
C2PA signing oracle. Before this it was handed to the compute child on every
rotation (the deleted ``T_TOKEN`` frame) and read straight out of the child's
environment — which is to say it was handed to tenant endpoint code, since that
code is imported into the same process. th#1311's credential laundering had all
the material it needed.

The child now holds no credential at all. When it needs a hub call that requires
worker identity, it ASKS: the parent decides whether to make the call, chooses
the base URL, attaches the JWT, and returns only the response.

Treat this table as the authorization surface it is:

* **The child never names a host.** ``base_url`` is the parent's, taken from the
  HelloAck it relayed. A child that could name the host could point the pod's
  credential at an attacker's server (th#1312's base-URL item).
* **The child never names a free-form path.** Every request must match one entry
  below by method AND full-path regex.
* **The child never names a header.** Authorization is the parent's to add.
* **Query and body keys are enumerated per entry**, so an allowlisted path
  cannot be used to smuggle a parameter the action was never meant to carry.
* Semantic narrowing that needs parent state (is this request actually in
  flight? is this capability token scoped to it?) lives in ``parent.py``, which
  owns the in-flight table — see delta 4.

Anything not in the table is refused and reported. A refusal is a security
event, not a 404.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class ActionRefused(Exception):
    """The parent refuses to perform the action the child asked for."""


@dataclass(frozen=True)
class HubAction:
    """One allowlisted hub call the parent will make on the child's behalf."""

    name: str
    method: str
    path: re.Pattern
    query: frozenset
    body: frozenset
    # Longest a parent will hold its own control loop for this call.
    timeout_s: float
    # Actions that carry per-job authority get parent-side semantic checks in
    # parent.py; flagged here so the dispatcher cannot forget to run them.
    scoped_to_job: bool = False


def _a(
    name: str,
    method: str,
    pattern: str,
    *,
    query: Tuple[str, ...] = (),
    body: Tuple[str, ...] = (),
    timeout_s: float = 30.0,
    scoped_to_job: bool = False,
) -> HubAction:
    return HubAction(
        name=name,
        method=method,
        path=re.compile(pattern),
        query=frozenset(query),
        body=frozenset(body),
        timeout_s=timeout_s,
        scoped_to_job=scoped_to_job,
    )


# A repo segment: owner/name, both restricted. Deliberately no "%", no dots and
# no slashes beyond the single separator — a path allowlist that admits ".." or
# an encoded slash is not an allowlist.
_REPO = r"[A-Za-z0-9._-]{1,64}/[A-Za-z0-9._-]{1,64}"

ACTIONS: Dict[str, HubAction] = {
    a.name: a
    for a in (
        # Per-job capability renewal (th#639). The parent additionally proves
        # the (request_id, attempt) is one it actually dispatched and applies
        # the delta-4 token policy to what comes back.
        _a(
            "capability.renew",
            "POST",
            r"^/v1/worker/capability/renew$",
            body=("request_id", "attempt", "capability_token"),
            scoped_to_job=True,
        ),
        # th#1307: the platform signing oracle. The child sends claim bytes and
        # receives a signature; the key is the hub's and the credential that
        # authenticates the ask is the parent's. See delta 5.
        _a(
            "c2pa.sign",
            "POST",
            r"^/v1/worker/c2pa/sign$",
            body=("alg", "claim_b64"),
        ),
        # pgw#709 cell-receipt gate: fetch one receipt, fetch the revocation
        # list. Read-only, and the gate fails closed without them.
        # ``artifact_digest`` is ONE algorithm-tagged digest (pgw#1034 collapsed
        # the multi-algorithm ceremony; the bare ``blake3`` key had already gone
        # with the v1 publish protocol). The repeat handling below still accepts
        # a list — it is parse-side input validation on an untrusted seam, not a
        # promise to any particular caller.
        _a(
            "cells.receipt",
            "GET",
            r"^/v1/worker/cells/receipt$",
            query=("cell_key", "artifact_digest"),
        ),
        _a(
            "cells.revocations",
            "GET",
            r"^/v1/worker/cells/revocations$",
        ),
        # Self-mint publish (gw#587/th#910). publish-intent returns a
        # key-pinned capability token — a short-TTL, least-authority grant, the
        # one credential shape the child is explicitly allowed to hold.
        # The body enumerations are the LIVE payloads `fleet_cells.publish`
        # sends, and nothing else. An unlisted key is an ActionRefused, not a
        # silent drop — so a publisher that grows a field the table does not
        # know refuses the whole publish under the split (the only execution
        # model since pgw#783). `identity_axes`/`mint_duration_ms` (th#1355)
        # were exactly that gap; `status`/`detail`/`axes` were names on
        # `publish-complete` that no caller and no hub route ever had.
        _a(
            "cells.publish_intent",
            "POST",
            r"^/v1/worker/cells/publish-intent$",
            body=("family", "cell_key", "axes", "identity_axes",
                  "mint_duration_ms"),
            timeout_s=60.0,
        ),
        # pgw#1090/th#1750 (§4.29): pull-by-key. The body carries the family
        # and the derived cell key and NOTHING else — every entitlement input
        # is resolved hub-side from the live session, and a body naming one is
        # a named 400 (`cell_resolve_client_supplied_field`), not an ignored
        # field. So this enumeration is not a convention, it is the request:
        # an action table that admitted one more key would make the whole
        # resolve refuse.
        _a(
            "cells.resolve",
            "POST",
            r"^/v1/worker/cells/resolve$",
            body=("family", "cell_key"),
            timeout_s=30.0,
        ),
        _a(
            "cells.publish_complete",
            "POST",
            r"^/v1/worker/cells/publish-complete$",
            body=("family", "cell_key", "checkpoint_id", "ok", "error"),
            timeout_s=60.0,
        ),
        # AOT cell discovery: list a system repo's checkpoints, resolve one
        # manifest. The manifest's artifact URL is PRESIGNED, so the bytes are
        # fetched by the child directly — the seam carries control, not data.
        _a(
            "repo.checkpoints",
            "GET",
            rf"^/api/v1/repos/{_REPO}/checkpoints$",
            query=("limit",),
        ),
        _a(
            "repo.resolve",
            "GET",
            rf"^/api/v1/repos/{_REPO}/resolve$",
            query=("digest",),
        ),
    )
}

# Non-HTTP action: the child asks the parent to dial the hub with a typed
# worker report (gw#640's HardwareUnsuitable carrier). The child cannot open
# that gRPC stream itself any more — it has no credential — and it should not:
# a HardwareUnsuitable claim is a FLEET-WIDE verdict key (th#1310), so it is
# worth strictly more coming from the process that does not run tenant code.
ACTION_REPORT_DETAIL = "report.detail"

_MAX_STR = 8192
# Repeats per query key. No action sends a repeated key today (pgw#1034 made
# the receipt fetch's ``artifact_digest`` a scalar); this bounds the shape the
# PARSER must accept from a process that runs tenant code, so it stays.
_MAX_QUERY_REPEATS = 8
# The seam's control-body cap. pgw#783's job-result seam accountant
# (procsplit/seam.py) reuses THIS constant as its ceiling so the two halves of
# "the seam carries CONTROL, not DATA" — the delta-1 action broker and the
# fan-in relay — cannot drift apart.
_MAX_JSON_BYTES = 256 * 1024
CONTROL_BODY_CEILING_BYTES = _MAX_JSON_BYTES


#: pgw#980: the two actions that WRITE a cell into a shared family namespace.
#: Every other action in the table reads, renews or reports.
PUBLISH_ACTIONS = frozenset({"cells.publish_intent", "cells.publish_complete"})

#: pgw#980: the env that marks a pod a LIVE-EDIT PROBE. A probe runs code that
#: was rsync'd onto it — code that is, by construction, not any released
#: version and not any built image. Its mints are therefore stamped with a
#: `gen_worker` version that lies (the dist-info does not move with the source)
#: and a `code_closure` axis no other pod can reproduce. A probe publishing into
#: the shared family namespace poisons every pod that later adopts from it.
_PROBE_ENV = "GEN_WORKER_PROBE"

#: And the deliberate, explicit arming. Two names rather than one truthy flag:
#: "this is a probe" and "this probe may write to the store" are different
#: decisions, and the second must never be reachable by forgetting the first.
_PROBE_PUBLISH_ARM_ENV = "GEN_WORKER_PROBE_PUBLISH_ARMED"


def probe_pod() -> bool:
    """Whether this pod is a live-edit probe (pgw#980)."""
    return str(os.environ.get(_PROBE_ENV, "")).strip().lower() in ("1", "true", "yes")


def publish_disarmed() -> bool:
    """Whether cell publish is disarmed for this pod.

    Read in the PARENT, which is the point. The compute child is where the
    swapped code runs and where a tenant endpoint executes; the parent holds
    the credential and owns this table, and nothing rsync'd into the child's
    site-packages can reach either. Disarming here is structural — a probe
    cannot publish by editing the code it is running.
    """
    return probe_pod() and (
        str(os.environ.get(_PROBE_PUBLISH_ARM_ENV, "")).strip().lower()
        not in ("1", "true", "yes"))


def authorize(req: Dict[str, Any]) -> Tuple[HubAction, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Validate one child action request against the table.

    Returns ``(action, query, json_body)`` ready for the parent to execute, or
    raises :class:`ActionRefused` naming exactly what was wrong. The caller adds
    the base URL and the credential; neither is representable in ``req``.
    """
    method = str(req.get("method") or "").upper()
    path = str(req.get("path") or "")
    if not method or not path:
        raise ActionRefused("action request carries no method/path")
    if len(path) > 1024 or "\\" in path or ".." in path:
        raise ActionRefused(f"malformed path {path[:120]!r}")

    match: Optional[HubAction] = None
    for action in ACTIONS.values():
        if action.method == method and action.path.fullmatch(path):
            match = action
            break
    if match is None:
        raise ActionRefused(
            f"{method} {path[:200]} is not an allowlisted parent-mediated action "
            "(the compute child holds no credential and may not name arbitrary "
            "hub calls)"
        )
    if match.name in PUBLISH_ACTIONS and publish_disarmed():
        # Refused, not dropped: the parent counts, logs and dials every
        # ActionRefused, so a probe whose operator MEANT to publish learns it
        # from a named refusal rather than from a cell that never appeared.
        raise ActionRefused(
            f"{match.name} is disarmed on this pod: {_PROBE_ENV} marks it a "
            f"live-edit probe running rsync'd code, whose cells carry a "
            f"`gen_worker` version that does not describe them and a "
            f"`code_closure` no other pod can reproduce. Set "
            f"{_PROBE_PUBLISH_ARM_ENV}=1 to arm it deliberately (pgw#980)."
        )

    raw_query = req.get("query") or {}
    if not isinstance(raw_query, dict):
        raise ActionRefused(f"{match.name}: query must be a mapping")
    query: Dict[str, Any] = {}
    for key, value in raw_query.items():
        key = str(key)
        if key not in match.query:
            raise ActionRefused(f"{match.name}: query parameter {key!r} is not permitted")
        if isinstance(value, (list, tuple)):
            # A repeated query key (requests encodes a list as repeated
            # params). Still an allowlisted key, still scalar items, still
            # size-capped — a list is not a hole in the enumeration.
            items: List[str] = []
            for item in value:
                text = "" if item is None else str(item)
                if len(text) > _MAX_STR:
                    raise ActionRefused(
                        f"{match.name}: query parameter {key!r} is oversized")
                items.append(text)
            if len(items) > _MAX_QUERY_REPEATS:
                raise ActionRefused(
                    f"{match.name}: query parameter {key!r} repeats too often")
            query[key] = items
            continue
        text = "" if value is None else str(value)
        if len(text) > _MAX_STR:
            raise ActionRefused(f"{match.name}: query parameter {key!r} is oversized")
        query[key] = text

    raw_json = req.get("json")
    body: Optional[Dict[str, Any]] = None
    if raw_json is not None:
        if not isinstance(raw_json, dict):
            raise ActionRefused(f"{match.name}: body must be a mapping")
        body = {}
        for key, value in raw_json.items():
            key = str(key)
            if key not in match.body:
                raise ActionRefused(f"{match.name}: body key {key!r} is not permitted")
            body[key] = value
        import json as _json

        try:
            size = len(_json.dumps(body))
        except (TypeError, ValueError) as exc:
            raise ActionRefused(f"{match.name}: body is not JSON-encodable: {exc}") from exc
        if size > _MAX_JSON_BYTES:
            # The seam carries CONTROL, not DATA. A body this size is either a
            # bug or an attempt to route the data plane through the parent's
            # single interpreter — the exact thing that would evaporate the
            # multi-GPU win one layer up.
            raise ActionRefused(
                f"{match.name}: body of {size} bytes exceeds the control-plane "
                f"limit of {_MAX_JSON_BYTES}"
            )
    if body is None and match.method == "POST":
        body = {}
    return match, query, body


__all__ = [
    "ACTIONS",
    "ACTION_REPORT_DETAIL",
    "ActionRefused",
    "HubAction",
    "authorize",
]
