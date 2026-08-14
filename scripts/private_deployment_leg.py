#!/usr/bin/env python3
"""A mint leg on a tensorhub PRIVATE DEPLOYMENT: create -> invoke -> read -> stop.

    python scripts/private_deployment_leg.py --hub URL --org O --endpoint org/name \\
        --release <id> --pair a40:bf16-w16a16+compiled --confirm
    python scripts/private_deployment_leg.py --dry --state merged     # no hub, no money

WHY THIS EXISTS (pgw#1250, for tensorhub th#1929 / epic th#1925)

The retired local micro-mint rig spawned a child interpreter on this box and
compiled there. Paul's 2026-08-10 hard cut removed that ground: **all mints and
compiles run on remote pods only**, with no carve-out. This private-deployment
leg is its sole replacement.

Before private deployments the answer was ugly, and it is written down in this
repo's own tracker as a THREE-STEP TEARDOWN PROTOCOL: cancel your own demand by
request id, then QUERY `tensorhub.endpoint_tags` to find out whether the release
you just used is tag-routable, and force-terminate the pod ONLY if that query
comes back empty — because a force-terminate against a tag-routable release can
take production capacity with it. Plus tagging a release in the first place to
manufacture the demand that buys the pod, and hoping the platform spend cap did
not count your rig against production.

**All of that collapses into `stop`.** A private deployment is an owner-paid
rental of (endpoint, PINNED release, ONE (GPU, lane) pair) that no capacity pass
can see and no reaper can touch:

  * the release does not need a routing tag, so there is no tag to delete and
    no demand to cancel — an UNTAGGED release is rentable by design;
  * the pod is attributable to one deployment id, so the kill set is that id
    and the `endpoint_tags` gate has nothing left to protect against;
  * the spend is the owner's, bounded by the rental's own `spend_limit_micros`,
    and never lands on the platform's concurrent-spend cap.

THE SEAM, AND WHAT IS NOT MERGED YET

The resource surface (tensorhub th#1926) is merged. The INVOKE route
(`POST /v1/private-deployments/:id/:function`, th#1927's follow-up) and
per-second settlement (th#1928) are not. Both are modelled here against the
settled contract, and a leg run against a hub that lacks them reports a typed
BLOCKED phase naming the issue -- never a skipped assertion and never a green.
`--dry` runs the whole leg against an in-process model of the contract, with
`--state resource|fences|merged` selecting which slices have "merged", so the
report shows exactly what is owed today.

There is no local fallback. A missing private-deployment route is a typed
BLOCKED result, never permission to resurrect the second mint path.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

# ---------------------------------------------------------------------------
# Vocabulary. A copy across a repo boundary of tensorhub's
# internal/orchestrator/privatedeployment and its schema CHECKs, so the
# assertions below check the invariants rather than trusting the copy.
# ---------------------------------------------------------------------------

STATE_ACTIVE = "active"
STATE_STOPPING = "stopping"
STATE_STOPPED = "stopped"

STOP_REASON_NONE = ""
STOP_REASON_OWNER = "owner_stop"
STOP_REASON_ADMIN = "admin_stop"

ACCESS_OWNER = "owner"
ACCESS_ORG = "org"

ON_POD_FAILURE_REPLACE = "replace"
ON_POD_FAILURE_STOP = "stop"

CHOOSER_ORCHESTRATOR = "orchestrator"
CHOOSER_PRIVATE_DEPLOYMENT = "private_deployment"

HISTORY_SOURCE_CREATE = "create"
HISTORY_SOURCE_STOP = "stop"

#: worker_pods states that do NOT hold a deployment in `stopping`.
DEAD_POD_STATES = frozenset({"terminated", "startup_error", "startup_error_claimed"})
#: worker_pods states that can take a dispatch.
READY_POD_STATES = frozenset({"ready", "connected", "running", "serving"})

BLOCKER_INVOKE_ROUTE = "th#1927 follow-up (POST /v1/private-deployments/:id/:function)"
BLOCKER_RECONCILER = "th#1927 (provisioning reconciler)"
BLOCKER_SETTLEMENT = "th#1928 (per-second settlement)"
BLOCKER_COMPILED_GRAPH_INVENTORY = (
    "th#1910 (GET /v1/admin/compiled-graphs) is absent from this hub"
)
BLOCKER_ACTIVITY_EVENTS = "th#1839 (GET /v1/admin/worker-activity-events) is absent from this hub"

OK = "ok"
FAILED = "failed"
BLOCKED = "blocked"
SKIPPED = "skipped"

_RANK = {OK: 0, SKIPPED: 1, BLOCKED: 2, FAILED: 3}


def _worse(a: str, b: str) -> str:
    return b if _RANK[b] > _RANK[a] else a


def parse_pair(raw: str) -> Tuple[str, str]:
    """Read the wire form "gpu:lane".

    A rental may NOT name a bare lane: "any card" is not a choice a per-second
    bill can be attributed to, so the lane-only spelling a release ladder
    accepts is a refusal here.
    """
    text = raw.strip().lower()
    if not text:
        raise ValueError("empty pair")
    if ":" not in text:
        raise ValueError(f"pair {raw!r} names no GPU; a rental pins one card, and {raw!r} is a lane on its own")
    gpu, lane = (part.strip() for part in text.split(":", 1))
    if not gpu:
        raise ValueError(f"pair {raw!r}: the GPU half before ':' is empty")
    if not lane:
        raise ValueError(f"pair {raw!r}: names no execution lane")
    return gpu, lane


def sku_slug(gpu_name: str) -> str:
    """Mirror tensorhub's `compilecache.SKUSlug`.

    THE FIELD TRAP THIS EXISTS FOR. `worker_pods.gpu_class` and the compiled-
    graph row's `sku` look like the same fact and are in DIFFERENT vocabularies:
    tensorhub's `observedGPUClass` writes the provider's catalogue id
    ("NVIDIA A40"); the graph store keys on the compilecache SKU slug ("a40").
    Comparing them raw is ALWAYS FALSE — a vacuously RED assertion, which is
    exactly as useless as a vacuously green one and harder to notice, because a
    red looks like it is working.
    """
    text = gpu_name.lower()
    for noise in ("nvidia", "geforce"):
        text = text.replace(noise, " ")
    out = "".join(c if c.isascii() and c.isalnum() else "-" for c in text).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out


def decoded_pixel_digest(mode: str, width: int, height: int, raw_pixels: bytes) -> str:
    """Bind decoded pixels, not encoder-dependent PNG/JPEG transport bytes.

    v1 is unambiguous: a 32-bit big-endian UTF-8 mode length, the mode bytes,
    64-bit big-endian width and height, then the row-major raw pixel bytes.
    Changing shape or mode therefore changes the identity even if the raw byte
    sequence happens to be the same.
    """
    mode_bytes = mode.encode("utf-8")
    framed = struct.pack(">I", len(mode_bytes)) + mode_bytes
    framed += struct.pack(">QQ", width, height) + raw_pixels
    return "sha256:" + hashlib.sha256(framed).hexdigest()


def compiled_graph_inventory_path(release: str) -> str:
    query = urllib.parse.urlencode(
        {"view": "compiled_graphs", "release": release, "limit": 200}
    )
    return "/v1/admin/compiled-graphs?" + query


def coherence_error(state: str, stop_reason: str, stopped_at: Optional[str]) -> Optional[str]:
    """Mirror the two schema CHECKs, and name the violation when one fails.

        (state = 'active')       = (stop_reason = '' AND stopped_at IS NULL)
        (stopped_at IS NOT NULL) = (state = 'stopped')

    Checked on every observation rather than assumed: this driver's job is to be
    the outside witness that a money-bearing row never lies.
    """
    quiet = not stop_reason.strip() and stopped_at is None
    if (state == STATE_ACTIVE) != quiet:
        return f"coherence: state={state!r} with stop_reason={stop_reason!r} stopped_at={stopped_at!r}"
    if (stopped_at is not None) != (state == STATE_STOPPED):
        return f"coherence: stopped_at={stopped_at!r} with state={state!r}"
    return None


# ---------------------------------------------------------------------------
# The API seam
# ---------------------------------------------------------------------------


class ApiError(Exception):
    """A typed hub refusal. `code` is the hub's refusal vocabulary when it named one."""

    def __init__(self, method: str, path: str, status: int, code: str, detail: str) -> None:
        super().__init__(f"{method} {path}: HTTP {status} {code or 'unnamed'}: {detail}")
        self.method = method
        self.path = path
        self.status = status
        self.code = code
        self.detail = detail

    @property
    def route_missing(self) -> bool:
        """True when this is "the hub does not serve that route".

        A hub that HAS the route answers a typed code, so "no such route" and
        "no such rental" stay distinguishable — which is what lets a report say
        BLOCKED honestly instead of claiming a resource vanished.
        """
        return (self.status == 404 and not self.code) or self.status == 405


class DeploymentAPI(Protocol):
    """Everything a leg needs. HttpDeploymentAPI speaks to a real hub; the tests
    drive an in-memory model of the same contract."""

    def create(self, org: str, body: Mapping[str, Any]) -> Dict[str, Any]: ...
    def get(self, org: str, deployment_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]: ...
    def stop(self, org: str, deployment_id: str, reason: str) -> Tuple[Dict[str, Any], bool]: ...
    def usage(self, org: str, deployment_id: str) -> Dict[str, Any]: ...
    def config_history(self, org: str, deployment_id: str) -> List[Dict[str, Any]]: ...
    def invoke(self, deployment_id: str, function: str, payload: Mapping[str, Any]) -> str: ...
    def request(self, request_id: str) -> Dict[str, Any]: ...
    def admin_compiled_graphs(self, release: str) -> List[Dict[str, Any]]: ...
    def admin_activity_events(self, release: str, state: str) -> List[Dict[str, Any]]: ...


class HttpDeploymentAPI:
    """The real surface. Every call carries an explicit timeout — a rig that can
    hang forever is a rig that leaves a pod running."""

    def __init__(self, base: str, token: str, timeout_s: float = 60.0) -> None:
        self.base = base.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    def _call(self, method: str, path: str, body: Optional[Mapping[str, Any]] = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", "Bearer " + self.token)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            code, detail = "", raw.decode(errors="replace")[:600].strip()
            try:
                envelope = json.loads(raw)
                inner = envelope.get("error", envelope)
                code = str(inner.get("code", ""))
                detail = str(inner.get("message", detail))
            except (ValueError, AttributeError):
                pass
            raise ApiError(method, path, exc.code, code, detail) from None
        except urllib.error.URLError as exc:
            raise ApiError(method, path, 0, "", f"transport: {exc.reason}") from None
        return json.loads(raw) if raw else {}

    def _base(self, org: str) -> str:
        return "/v1/orgs/" + urllib.parse.quote(org, safe="") + "/private-deployments"

    def create(self, org: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        out = self._call("POST", self._base(org), body)
        return dict(out.get("deployment", {}))

    def get(self, org: str, deployment_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        out = self._call("GET", f"{self._base(org)}/{urllib.parse.quote(deployment_id)}")
        return dict(out.get("deployment", {})), list(out.get("pods", []))

    def stop(self, org: str, deployment_id: str, reason: str) -> Tuple[Dict[str, Any], bool]:
        out = self._call("POST", f"{self._base(org)}/{urllib.parse.quote(deployment_id)}/stop",
                         {"reason": reason})
        return dict(out.get("deployment", {})), bool(out.get("already_stopped", False))

    def usage(self, org: str, deployment_id: str) -> Dict[str, Any]:
        return dict(self._call("GET", f"{self._base(org)}/{urllib.parse.quote(deployment_id)}/usage"))

    def config_history(self, org: str, deployment_id: str) -> List[Dict[str, Any]]:
        out = self._call("GET", f"{self._base(org)}/{urllib.parse.quote(deployment_id)}/config/history")
        return list(out.get("history", []))

    def invoke(self, deployment_id: str, function: str, payload: Mapping[str, Any]) -> str:
        path = ("/v1/private-deployments/" + urllib.parse.quote(deployment_id)
                + "/" + urllib.parse.quote(function))
        out = self._call("POST", path, payload)
        request_id = str(out.get("request_id") or out.get("id") or "")
        if not request_id:
            raise ApiError("POST", path, 200, "", "answered no request id")
        return request_id

    def request(self, request_id: str) -> Dict[str, Any]:
        return dict(self._call("GET", "/v1/requests/" + urllib.parse.quote(request_id)))

    def admin_compiled_graphs(self, release: str) -> List[Dict[str, Any]]:
        """Read durable compiled graphs from their own rows, not demand joins."""
        out = self._call("GET", compiled_graph_inventory_path(release))
        return list(out.get("compiled_graphs", []))

    def admin_activity_events(self, release: str, state: str) -> List[Dict[str, Any]]:
        query = urllib.parse.urlencode({"release": release, "state": state, "limit": 200})
        out = self._call("GET", "/v1/admin/worker-activity-events?" + query)
        return list(out.get("events", []))


# ---------------------------------------------------------------------------
# The leg
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    ident: str
    status: str
    detail: str = ""
    blocker: str = ""


@dataclass
class Phase:
    name: str
    status: str = OK
    findings: List[Finding] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class Leg:
    """One rental: ONE (GPU, lane), one release, N invokes."""

    org: str
    endpoint: str
    release_id: str
    pair: Tuple[str, str]
    function: str = "generate"
    payload: Mapping[str, Any] = field(default_factory=dict)
    invocations: int = 1
    pod_count: int = 1
    access_mode: str = ACCESS_OWNER
    # A mint leg measures ONE machine: replacing a dead pod mid-leg would
    # silently change the producer, which is the whole subject of the proof.
    on_pod_failure: str = ON_POD_FAILURE_STOP
    spend_limit_usd: float = 1.0
    reason: str = "pgw#1250 mint leg on a private deployment"
    #: A leg whose PRODUCT is published compiled graphs rather than latency
    #: numbers. It relaxes nothing; it adds the product check below.
    mint_proof: bool = True

    def validate(self) -> None:
        if not self.org.strip():
            raise ValueError("leg: org is required (it is the payer)")
        if not self.endpoint.strip():
            raise ValueError("leg: endpoint is required")
        if not self.release_id.strip():
            raise ValueError(
                "leg: release_id is required — a rental pins its release and never follows a tag")
        if not self.pair[0] or not self.pair[1]:
            raise ValueError("leg: the pair must name both a GPU and a lane")
        if self.pod_count < 1:
            raise ValueError("leg: pod_count must be at least 1")
        if self.access_mode not in (ACCESS_OWNER, ACCESS_ORG):
            raise ValueError(f"leg: access_mode {self.access_mode!r} must be owner or org")
        if self.on_pod_failure not in (ON_POD_FAILURE_REPLACE, ON_POD_FAILURE_STOP):
            raise ValueError(f"leg: on_pod_failure {self.on_pod_failure!r} must be replace or stop")
        if self.spend_limit_usd < 0:
            raise ValueError("leg: spend_limit_usd cannot be negative")


@dataclass
class LegResult:
    leg: Leg
    deployment_id: str = ""
    phases: List[Phase] = field(default_factory=list)
    deployment: Dict[str, Any] = field(default_factory=dict)
    pods: List[Dict[str, Any]] = field(default_factory=list)
    requests: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        out = OK
        for phase in self.phases:
            out = _worse(out, phase.status)
        return out

    @property
    def findings(self) -> List[Finding]:
        return [f for phase in self.phases for f in phase.findings]

    @property
    def blockers(self) -> List[str]:
        return sorted({f.blocker for f in self.findings if f.status == BLOCKED and f.blocker})

    def to_json(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "status": self.status,
            "blockers": self.blockers,
            "phases": [
                {"name": p.name, "status": p.status, "seconds": round(p.seconds, 3),
                 "findings": [vars(f) for f in p.findings]}
                for p in self.phases
            ],
            "deployment": self.deployment,
            "pods": self.pods,
            "requests": self.requests,
            "usage": self.usage,
        }


def _live(pods: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [p for p in pods if str(p.get("state", "")) not in DEAD_POD_STATES]


def _ready(pods: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [p for p in pods if str(p.get("state", "")) in READY_POD_STATES]


class _Run:
    """Driver state. Give-up is PROGRESS STALENESS plus the caller's own budget —
    never a fixed duration standing in for a lifecycle decision (the standing
    no-magic-timeouts rule). Whatever happens, the rental is stopped."""

    def __init__(self, api: DeploymentAPI, leg: Leg, *, poll_s: float,
                 stall_budget_s: float, log: Callable[[str], None],
                 clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
        self.api = api
        self.leg = leg
        self.poll_s = poll_s
        self.stall_budget_s = stall_budget_s
        self.log = log
        self.clock = clock
        self.sleep = sleep
        self.result = LegResult(leg=leg)
        self.phase = Phase(name="init")
        self.last_progress = clock()
        self.fingerprint = ""

    # -- phase plumbing ----------------------------------------------------
    def begin(self, name: str) -> None:
        self.phase = Phase(name=name)
        self.phase.seconds = self.clock()
        self.result.phases.append(self.phase)

    def end(self) -> None:
        self.phase.seconds = self.clock() - self.phase.seconds

    def find(self, ident: str, status: str, detail: str, blocker: str = "") -> None:
        self.phase.findings.append(Finding(ident, status, detail, blocker))
        self.phase.status = _worse(self.phase.status, status)
        self.log(f"[{self.phase.name}] {status} {ident}: {detail}")

    def check(self, ident: str, ok: bool, detail: str) -> bool:
        self.find(ident, OK if ok else FAILED, detail)
        return ok

    def note_progress(self, fingerprint: str) -> None:
        if fingerprint != self.fingerprint:
            self.fingerprint = fingerprint
            self.last_progress = self.clock()

    def stalled(self) -> bool:
        return (self.clock() - self.last_progress) >= self.stall_budget_s

    # -- legs --------------------------------------------------------------
    def create(self) -> bool:
        self.begin("create")
        body: Dict[str, Any] = {
            "endpoint": self.leg.endpoint,
            "release_id": self.leg.release_id,
            "pair": f"{self.leg.pair[0]}:{self.leg.pair[1]}",
            "pod_count": self.leg.pod_count,
            "access_mode": self.leg.access_mode,
            "on_pod_failure": self.leg.on_pod_failure,
            "reason": self.leg.reason,
        }
        if self.leg.spend_limit_usd > 0:
            body["spend_limit_usd"] = self.leg.spend_limit_usd
        try:
            dep = self.api.create(self.leg.org, body)
        except ApiError as exc:
            if exc.route_missing:
                self.find("create.route", BLOCKED,
                          f"the hub does not serve the resource surface: {exc}",
                          "th#1926 (private-deployment resource)")
            else:
                self.find("create.accepted", FAILED, f"create refused: {exc}")
            self.end()
            return False
        self.result.deployment_id = str(dep.get("id", ""))
        self.result.deployment = dep
        self.check("create.accepted", bool(self.result.deployment_id),
                   f"deployment {self.result.deployment_id} created")
        self.check("create.state_active", dep.get("state") == STATE_ACTIVE,
                   f"state={dep.get('state')!r}")
        err = coherence_error(str(dep.get("state", "")), str(dep.get("stop_reason", "")),
                              dep.get("stopped_at"))
        self.check("create.row_coherent", err is None, err or "state/stop_reason/stopped_at agree")
        self.check("create.release_pinned", dep.get("release_id") == self.leg.release_id,
                   f"release_id={dep.get('release_id')!r} (asked {self.leg.release_id!r})")
        self.check("create.pair_roundtrip",
                   dep.get("gpu_slug") == self.leg.pair[0] and dep.get("execution_lane") == self.leg.pair[1],
                   f"pair={dep.get('pair')!r}")
        self.check("create.generation_genesis", dep.get("config_generation") == 1,
                   f"config_generation={dep.get('config_generation')}")
        self.check("create.owner_recorded", bool(str(dep.get("created_by", "")).strip()),
                   f"created_by={dep.get('created_by')!r}")
        self.end()
        return True

    def provision(self) -> None:
        self.begin("provision")
        pods = self._wait_pods()
        self.result.pods = list(pods)
        live = _live(pods)
        if not live:
            self.find("provision.pods", BLOCKED,
                      "no pod was provisioned for the rental", BLOCKER_RECONCILER)
            self.end()
            return
        self.check("provision.pod_count", len(live) >= self.leg.pod_count,
                   f"{len(live)} live pod(s) for pod_count={self.leg.pod_count}")
        for pod in live:
            pod_id = str(pod.get("pod_id", "?"))
            self.check(f"provision.chooser[{pod_id}]",
                       pod.get("placement_chooser") == CHOOSER_PRIVATE_DEPLOYMENT,
                       f"placement_chooser={pod.get('placement_chooser')!r}")
            self.check(f"provision.attributed[{pod_id}]",
                       pod.get("private_deployment_id") == self.result.deployment_id,
                       f"private_deployment_id={pod.get('private_deployment_id')!r}")
            # The provider figure is READ, never re-derived: the ledger column
            # is already the WHOLE-POD rate (compute AND disk), and re-deriving
            # a disk rate on top of it double-bills the owner.
            self.check(f"provision.rate_recorded[{pod_id}]",
                       int(pod.get("cost_micros_per_hour", 0)) > 0,
                       f"cost_micros_per_hour={pod.get('cost_micros_per_hour')} (whole pod)")
        self.end()

    def _wait_pods(self) -> List[Dict[str, Any]]:
        pods: List[Dict[str, Any]] = []
        while True:
            try:
                dep, pods = self.api.get(self.leg.org, self.result.deployment_id)
            except ApiError as exc:
                self.log(f"[provision] read failed, retrying: {exc}")
            else:
                self.result.deployment = dep
                self.note_progress(json.dumps(
                    [dep.get("state"), dep.get("config_generation"),
                     sorted(f"{p.get('pod_id')}={p.get('state')}" for p in pods)], sort_keys=True))
                if dep.get("state") != STATE_ACTIVE:
                    self.find("provision.still_active", FAILED,
                              f"the rental left active before serving: state={dep.get('state')!r} "
                              f"stop_reason={dep.get('stop_reason')!r}")
                    return pods
                if len(_ready(pods)) >= self.leg.pod_count:
                    self.find("provision.ready", OK, f"{len(_ready(pods))} pod(s) ready")
                    return pods
                if not _live(pods) and self.stalled():
                    return pods
            if self.stalled():
                self.find("provision.ready", FAILED,
                          f"no observable progress for {self.stall_budget_s:.0f}s; "
                          f"{len(_live(pods))} live / {len(_ready(pods))} ready")
                return pods
            self.sleep(self.poll_s)

    def invoke(self) -> None:
        self.begin("invoke")
        if self.leg.invocations <= 0:
            self.find("invoke.workload", SKIPPED, "leg declares no invocations (lifecycle only)")
            self.end()
            return
        for index in range(self.leg.invocations):
            started = self.clock()
            try:
                request_id = self.api.invoke(self.result.deployment_id, self.leg.function, self.leg.payload)
            except ApiError as exc:
                self.result.requests.append({"index": index, "error": str(exc)})
                if exc.route_missing:
                    self.find("invoke.route", BLOCKED,
                              f"the hub does not serve the private invoke route yet: {exc}",
                              BLOCKER_INVOKE_ROUTE)
                else:
                    self.find("invoke.accepted", FAILED, f"invocation {index} refused: {exc}")
                self.end()
                return
            self.note_progress("invoke:" + request_id)
            record = self._wait_request(request_id)
            record["index"] = index
            record["seconds"] = round(self.clock() - started, 3)
            self.result.requests.append(record)
            self.check(f"invoke.completed[{index}]", record.get("status") == "completed",
                       f"request {request_id} status={record.get('status')!r} in {record['seconds']}s")
        self.end()

    def _wait_request(self, request_id: str) -> Dict[str, Any]:
        last = ""
        while True:
            try:
                record = self.api.request(request_id)
            except ApiError as exc:
                self.log(f"[invoke] request read failed, retrying: {exc}")
            else:
                status = str(record.get("status", ""))
                if status != last:
                    last = status
                    self.note_progress(f"request:{request_id}:{status}")
                if status in ("completed", "failed", "canceled", "cancelled"):
                    return dict(record)
            if self.stalled():
                return {"request_id": request_id, "status": last,
                        "error": f"no observable progress for {self.stall_budget_s:.0f}s"}
            self.sleep(self.poll_s)

    def read(self) -> None:
        self.begin("read")
        try:
            usage = self.api.usage(self.leg.org, self.result.deployment_id)
        except ApiError as exc:
            self.find("read.usage", FAILED, f"usage read failed: {exc}")
        else:
            self.result.usage = usage
            self.find("read.usage", OK,
                      f"pod_rows={usage.get('pod_rows')} pod_seconds={usage.get('pod_seconds')}")
            self._assert_settlement(usage)
        try:
            history = self.api.config_history(self.leg.org, self.result.deployment_id)
        except ApiError as exc:
            self.find("read.history", FAILED, f"config history read failed: {exc}")
            self.end()
            return
        self.result.history = history
        genesis = [row for row in history if row.get("config_generation") == 1]
        if not genesis:
            self.find("read.history_genesis", FAILED,
                      f"no generation-1 row among {len(history)} history entries")
        else:
            self.check("read.history_genesis", genesis[0].get("source") == HISTORY_SOURCE_CREATE,
                       f"genesis source={genesis[0].get('source')!r}")
            self.check("read.history_actor", bool(str(genesis[0].get("actor", "")).strip()),
                       f"genesis actor={genesis[0].get('actor')!r}")
        # The mint leg's PRODUCT is published compiled graphs. A completed
        # request is not evidence of one: a worker can serve eagerly, or mint
        # and fail to seal, or seal and fail to publish, and every one of those
        # completes the request. Runs BEFORE the stop, because a stopped
        # rental's pods are gone and this reads what they left behind.
        if self.leg.mint_proof:
            # ONLY when something actually ran. A leg that never invoked — the
            # invoke route is absent, or the workload was refused — cannot have
            # minted anything, and asserting an empty store there is a FALSE
            # RED: it would report "no graph was published" as a defect of the
            # mint when the real answer is upstream, already named by the
            # invoke phase's own blocker.
            completed = [r for r in self.result.requests if r.get("status") == "completed"]
            if not completed:
                self.find("seal.workload", SKIPPED,
                          "no invocation completed, so no compiled graph could have been minted; "
                          "the invoke phase says why")
            else:
                self._assert_sealed()
        self.end()

    def _assert_sealed(self) -> None:
        completed = [r for r in self.result.requests if r.get("status") == "completed"]
        evidences: List[Dict[str, Any]] = []
        for index, record in enumerate(completed):
            evidence = self._assert_request_evidence(record, index)
            if evidence is not None:
                evidences.append(evidence)

        # The card ACTUALLY rented, from the pod row — never the pair asked
        # for. A create can land on an sm-equivalent sibling of the SKU it
        # named, and an assertion against the ask is measuring an intention.
        observed = ""
        for pod in self.result.pods:
            if str(pod.get("gpu_class", "")).strip():
                observed = str(pod["gpu_class"])
                break
        try:
            compiled_graphs = self.api.admin_compiled_graphs(self.leg.release_id)
        except ApiError as exc:
            if exc.route_missing:
                self.find("seal.route", BLOCKED,
                          f"cannot read the compiled-graph store: {exc}",
                          BLOCKER_COMPILED_GRAPH_INVENTORY)
            else:
                self.find("seal.rows", FAILED, f"compiled-graph inventory read failed: {exc}")
            return
        mine = [g for g in compiled_graphs
                if g.get("minted_for_release_id") == self.leg.release_id]
        if not self.check("seal.rows", bool(mine),
                          f"{len(mine)} compiled graph(s) minted for the pinned release "
                          f"{self.leg.release_id} ({len(compiled_graphs)} row(s) returned)"):
            return
        if not observed:
            self.find("seal.sku_matches_rented_card", FAILED,
                      f"no pod row recorded a gpu_class, so there is no rented card to match "
                      f"{len(mine)} graph(s) against")
            return
        want = sku_slug(observed)
        matched = [g for g in mine if g.get("sku") == want]
        self.check("seal.sku_matches_rented_card", bool(matched),
                   f"{len(matched)} graph(s) on sku={want!r} (from pod gpu_class {observed!r}); "
                   f"the store holds {sorted({str(g.get('sku', '')) for g in mine})} for this "
                   f"release [note: the pod view exposes no gpu_class provenance, so a "
                   f"provider-reported card and an echoed ask are indistinguishable from here]")
        if not matched:
            return
        # Sealed means an ARTIFACT exists, not merely that a key was computed.
        sealed = [g for g in matched if str(g.get("artifact_digest", "")).strip()]
        self.check("seal.artifact_ref", len(sealed) == len(matched),
                   f"{len(sealed)}/{len(matched)} matched graph(s) carry an artifact ref")
        by_key = {str(g.get("compiled_graph_key", "")): g for g in matched}
        evidence_keys = [str(e.get("compiled_graph_key", "")) for e in evidences]
        self.check("seal.graph_keys_exact",
                   bool(evidence_keys) and all(key in by_key for key in evidence_keys),
                   f"request keys={evidence_keys!r}; inventory keys={sorted(by_key)!r}")
        self.check(
            "seal.artifact_refs_exact",
            bool(evidences) and all(
                str(by_key.get(str(e.get("compiled_graph_key", "")), {}).get(
                    "artifact_digest", "")) == str(e.get("artifact_ref", ""))
                for e in evidences
            ),
            "every request artifact ref equals its durable inventory row",
        )
        self.check(
            "seal.worker_versions_exact",
            bool(evidences) and all(
                str(by_key.get(str(e.get("compiled_graph_key", "")), {}).get(
                    "gen_worker_version", "")) == str(e.get("gen_worker_version", ""))
                for e in evidences
            ),
            "every request worker version equals its durable inventory row",
        )
        quarantined = [g for g in matched if g.get("quarantined_at")]
        self.check("seal.not_quarantined", not quarantined,
                   f"{len(quarantined)} matched graph(s) quarantined")
        self.check("seal.sm_recorded", bool(str(matched[0].get("sm", "")).strip()),
                   f"sm={matched[0].get('sm')!r} on "
                   f"{matched[0].get('compiled_graph_key')!r}")
        try:
            failures = self.api.admin_activity_events(self.leg.release_id, "failed")
        except ApiError as exc:
            if exc.route_missing:
                self.find("seal.no_failed_publish", BLOCKED,
                          f"cannot read the publish timeline: {exc}", BLOCKER_ACTIVITY_EVENTS)
            else:
                self.find("seal.no_failed_publish", FAILED, f"publish timeline read failed: {exc}")
            return
        named = [f"{e.get('kind')}/{e.get('phase')}({e.get('error') or e.get('detail')})"
                 for e in failures]
        self.check("seal.no_failed_publish", not named,
                   f"{len(named)} failed worker phase(s) in the leg window"
                   + (": " + "; ".join(named[:5]) if named else ""))

    def _assert_request_evidence(
            self, record: Mapping[str, Any], index: int) -> Optional[Dict[str, Any]]:
        ident = f"[{index}]"
        raw = record.get("compiled_graph_evidence")
        if not isinstance(raw, dict):
            self.find(f"evidence.present{ident}", FAILED,
                      "completed request carries no compiled_graph_evidence object")
            return None
        evidence = dict(raw)
        required = {
            "status", "outcome", "refusal", "compiled_graph_key", "artifact_ref",
            "receipt_ref", "gen_worker_version", "torch_compiled_graphs_version",
            "hashrepo_version", "hashrepo", "compile_child", "serving_pid_before",
            "serving_pid_after", "serving_compile_count", "compile_count",
            "child_spawn_count", "bind_fqns", "bind_call_count", "runner_call_count",
            "decoded_pixel",
        }
        self.check(f"evidence.schema{ident}", set(evidence) == required,
                   f"fields={sorted(evidence)}")
        self.check(f"evidence.status{ident}", evidence.get("status") == "completed",
                   f"status={evidence.get('status')!r}")
        outcome = evidence.get("outcome")
        self.check(f"evidence.outcome{ident}", outcome in ("published", "reused"),
                   f"outcome={outcome!r}")
        self.check(f"evidence.refusal{ident}", evidence.get("refusal") == "",
                   f"refusal={evidence.get('refusal')!r}")
        self.check(f"evidence.graph_key{ident}",
                   bool(str(evidence.get("compiled_graph_key", "")).strip()),
                   f"compiled_graph_key={evidence.get('compiled_graph_key')!r}")
        self.check(f"evidence.artifact_ref{ident}",
                   bool(str(evidence.get("artifact_ref", "")).strip()),
                   f"artifact_ref={evidence.get('artifact_ref')!r}")
        self.check(f"evidence.receipt_ref{ident}",
                   bool(str(evidence.get("receipt_ref", "")).strip()),
                   f"receipt_ref={evidence.get('receipt_ref')!r}")
        versions = [
            str(evidence.get("gen_worker_version", "")).strip(),
            str(evidence.get("torch_compiled_graphs_version", "")).strip(),
            str(evidence.get("hashrepo_version", "")).strip(),
        ]
        self.check(f"evidence.versions{ident}", all(versions), f"versions={versions!r}")

        before = evidence.get("serving_pid_before")
        after = evidence.get("serving_pid_after")
        self.check(f"evidence.serving_pid{ident}",
                   isinstance(before, int) and not isinstance(before, bool) and before > 0
                   and before == after,
                   f"serving_pid_before={before!r} serving_pid_after={after!r}")
        self.check(f"evidence.no_serving_compile{ident}",
                   evidence.get("serving_compile_count") == 0,
                   f"serving_compile_count={evidence.get('serving_compile_count')!r}")

        hashrepo = evidence.get("hashrepo")
        if not isinstance(hashrepo, dict):
            hashrepo = {}
        hashrepo_required = {
            "manifest_ref", "object_refs", "materialized_root",
            "local_ref_count_before", "local_object_count_before",
            "reference_count", "object_count", "materialized_object_count",
        }
        self.check(f"evidence.hashrepo_schema{ident}", set(hashrepo) == hashrepo_required,
                   f"fields={sorted(hashrepo)}")
        self.check(f"evidence.hashrepo_manifest_ref{ident}",
                   bool(str(hashrepo.get("manifest_ref", "")).strip()),
                   f"manifest_ref={hashrepo.get('manifest_ref')!r}")
        self.check(f"evidence.hashrepo_materialized_root{ident}",
                   bool(str(hashrepo.get("materialized_root", "")).strip()),
                   f"materialized_root={hashrepo.get('materialized_root')!r}")
        self.check(f"evidence.empty_cache{ident}",
                   hashrepo.get("local_ref_count_before") == 0
                   and hashrepo.get("local_object_count_before") == 0,
                   "local ref/object counts before resolve are "
                   f"{hashrepo.get('local_ref_count_before')!r}/"
                   f"{hashrepo.get('local_object_count_before')!r}")
        object_refs = hashrepo.get("object_refs")
        refs = object_refs if isinstance(object_refs, list) else []
        refs_valid = bool(refs) and all(isinstance(ref, str) and ref.strip() for ref in refs)
        self.check(f"evidence.hashrepo_refs{ident}",
                   refs_valid and hashrepo.get("reference_count") == len(refs),
                   f"reference_count={hashrepo.get('reference_count')!r} refs={refs!r}")
        unique_objects = len(set(refs))
        self.check(f"evidence.hashrepo_objects{ident}",
                   refs_valid and hashrepo.get("object_count") == unique_objects
                   and hashrepo.get("materialized_object_count") == unique_objects,
                   f"object_count={hashrepo.get('object_count')!r} "
                   f"materialized={hashrepo.get('materialized_object_count')!r} "
                   f"unique_refs={unique_objects}")

        bind_fqns = evidence.get("bind_fqns")
        fqns = bind_fqns if isinstance(bind_fqns, list) else []
        self.check(f"evidence.bind_fqns{ident}",
                   bool(fqns) and fqns == sorted(set(fqns))
                   and all(isinstance(fqn, str) and fqn for fqn in fqns),
                   f"bind_fqns={fqns!r}")
        self.check(f"evidence.bind_calls{ident}", evidence.get("bind_call_count") == 1,
                   f"bind_call_count={evidence.get('bind_call_count')!r}")
        self.check(f"evidence.runner_calls{ident}", evidence.get("runner_call_count") == 1,
                   f"runner_call_count={evidence.get('runner_call_count')!r}")

        pixel = evidence.get("decoded_pixel")
        if not isinstance(pixel, dict):
            pixel = {}
        mode, width, height = pixel.get("mode"), pixel.get("width"), pixel.get("height")
        self.check(f"evidence.pixel_shape{ident}",
                   set(pixel) == {"mode", "width", "height", "sha256"}
                   and isinstance(mode, str) and bool(mode)
                   and isinstance(width, int) and width > 0
                   and isinstance(height, int) and height > 0,
                   f"decoded_pixel={pixel!r}")
        source = record.get("decoded_pixel_raw_base64")
        try:
            decoded = base64.b64decode(str(source), validate=True)
            if not isinstance(width, int) or not isinstance(height, int):
                raise TypeError("decoded pixel dimensions are not integers")
            expected_digest = decoded_pixel_digest(str(mode), width, height, decoded)
        except (ValueError, TypeError):
            expected_digest = ""
        self.check(f"evidence.pixel_digest{ident}",
                   bool(expected_digest) and pixel.get("sha256") == expected_digest,
                   f"reported={pixel.get('sha256')!r} recomputed={expected_digest!r}")
        if isinstance(record, dict):
            record.pop("decoded_pixel_raw_base64", None)

        if outcome == "reused":
            self.check(f"evidence.reuse_compile_count{ident}", evidence.get("compile_count") == 0,
                       f"compile_count={evidence.get('compile_count')!r}")
            self.check(f"evidence.reuse_spawn_count{ident}",
                       evidence.get("child_spawn_count") == 0,
                       f"child_spawn_count={evidence.get('child_spawn_count')!r}")
            self.check(f"evidence.reuse_no_child{ident}", evidence.get("compile_child") is None,
                       f"compile_child={evidence.get('compile_child')!r}")
        elif outcome == "published":
            self.check(f"evidence.publisher_compile_count{ident}",
                       evidence.get("compile_count") == 1,
                       f"compile_count={evidence.get('compile_count')!r}")
            self.check(f"evidence.publisher_spawn_count{ident}",
                       evidence.get("child_spawn_count") == 1,
                       f"child_spawn_count={evidence.get('child_spawn_count')!r}")
            child = evidence.get("compile_child")
            if not isinstance(child, dict):
                child = {}
            start, finish = child.get("start"), child.get("finish")
            self.check(
                f"evidence.compile_child{ident}",
                set(child) == {"pid", "ppid", "role", "start", "finish"}
                and isinstance(child.get("pid"), int) and child["pid"] > 0
                and child.get("pid") != before
                and isinstance(child.get("ppid"), int) and child["ppid"] > 0
                and child.get("role") == "compile_child"
                and isinstance(start, (int, float)) and not isinstance(start, bool)
                and isinstance(finish, (int, float)) and not isinstance(finish, bool)
                and finish >= start,
                f"compile_child={child!r} serving_pid={before!r}",
            )
        return evidence

    def _assert_settlement(self, usage: Mapping[str, Any]) -> None:
        settlement = dict(usage.get("settlement", {}))
        if not settlement.get("available", False):
            reason = str(settlement.get("reason", "")).strip()
            if not reason:
                self.find("settle.honest_absence", FAILED,
                          "settlement is unavailable and states no reason")
                return
            self.find("settle.honest_absence", OK, f"settlement unavailable, stated: {reason}")
            figures = [settlement.get(key, 0) for key in
                       ("captured_micros", "settled_micros", "held_micros")]
            if any(figures):
                # A zero here reads as "nothing was owed" rather than "nothing
                # has settled", and a figure is worse.
                self.find("settle.no_phantom_figures", FAILED,
                          f"unavailable settlement carries figures: {figures}")
                return
            self.find("settle.windows", BLOCKED,
                      "no settlement window exists for this rental", BLOCKER_SETTLEMENT)
            return
        windows = int(settlement.get("windows", 0))
        if not self.check("settle.windows", windows > 0, f"{windows} settlement window(s)"):
            return
        self.check("settle.provider_micros", int(settlement.get("provider_micros", 0)) > 0,
                   f"provider_micros={settlement.get('provider_micros')} (the ledger's own figure)")
        self.check("settle.no_quarantine", int(settlement.get("quarantined_windows", 0)) == 0,
                   f"quarantined_windows={settlement.get('quarantined_windows')}")
        margin_at_create = int(usage.get("rate_margin_bps_at_create", 0))
        for window in settlement.get("recent_windows", []):
            start = window.get("window_start")
            ident = f"settle.window[{start}]"
            self.check(f"{ident}.margin_frozen", int(window.get("margin_bps", -1)) == margin_at_create,
                       f"window margin_bps={window.get('margin_bps')}, rental recorded {margin_at_create}")
            if window.get("status") == "settled":
                provider = int(window.get("provider_micros", 0))
                amount = int(window.get("amount_micros", 0))
                want = provider + provider * int(window.get("margin_bps", 0)) // 10000
                self.check(f"{ident}.margin_applied", abs(amount - want) <= 1,
                           f"amount={amount} provider={provider} expected≈{want}")
                held = int(window.get("held_micros", 0))
                self.check(f"{ident}.clamped_to_hold", held == 0 or amount <= held,
                           f"amount={amount} held={held}")
        limit = usage.get("spend_limit_micros")
        if limit is not None:
            spent = int(settlement.get("captured_micros", 0)) + int(settlement.get("held_micros", 0))
            self.check("settle.within_spend_limit", spent <= int(limit),
                       f"captured+held={spent} against spend_limit_micros={limit}")

    def stop(self) -> None:
        """End the rental. THIS is the whole teardown — no demand to cancel, no
        `endpoint_tags` query, no conditional force-terminate. The kill set is
        the deployment id."""
        self.begin("stop")
        try:
            dep, already = self.api.stop(self.leg.org, self.result.deployment_id, self.leg.reason)
        except ApiError as exc:
            self.find("stop.accepted", FAILED, f"stop refused: {exc}")
            self.end()
            return
        self.result.deployment = dep
        self.check("stop.accepted", not already, f"stop accepted (already_stopped={already})")
        self.check("stop.state", dep.get("state") in (STATE_STOPPING, STATE_STOPPED),
                   f"state={dep.get('state')!r}")
        self.check("stop.reason", dep.get("stop_reason") in (STOP_REASON_OWNER, STOP_REASON_ADMIN),
                   f"stop_reason={dep.get('stop_reason')!r}")
        err = coherence_error(str(dep.get("state", "")), str(dep.get("stop_reason", "")),
                              dep.get("stopped_at"))
        self.check("stop.row_coherent", err is None, err or "state/stop_reason/stopped_at agree")
        try:
            repeat, already_again = self.api.stop(self.leg.org, self.result.deployment_id, self.leg.reason)
        except ApiError as exc:
            self.find("stop.idempotent", FAILED, f"second stop refused: {exc}")
            return_early = True
        else:
            return_early = False
            self.check("stop.idempotent",
                       already_again and repeat.get("config_generation") == dep.get("config_generation"),
                       f"second stop already_stopped={already_again} "
                       f"config_generation {dep.get('config_generation')}->{repeat.get('config_generation')}")
        if not return_early:
            try:
                history = self.api.config_history(self.leg.org, self.result.deployment_id)
            except ApiError:
                history = []
            if history:
                self.result.history = history
                newest = max(history, key=lambda row: int(row.get("config_generation", 0)))
                self.check("stop.history_source", newest.get("source") == HISTORY_SOURCE_STOP,
                           f"newest history row source={newest.get('source')!r}")
        self.end()


def run_leg(api: DeploymentAPI, leg: Leg, *, poll_s: float = 5.0,
            stall_budget_s: float = 1800.0,
            log: Callable[[str], None] = lambda _line: None,
            clock: Callable[[], float] = time.monotonic,
            sleep: Callable[[float], None] = time.sleep) -> LegResult:
    """create -> invoke -> read -> stop, with the rental stopped on every path."""
    leg.validate()
    run = _Run(api, leg, poll_s=poll_s, stall_budget_s=stall_budget_s,
               log=log, clock=clock, sleep=sleep)
    if not run.create():
        return run.result
    try:
        run.provision()
        run.invoke()
        run.read()
    finally:
        run.stop()
    return run.result


# ---------------------------------------------------------------------------
# The in-process model, so the driver is testable before the routes exist
# ---------------------------------------------------------------------------


class ContractModel:
    """A MODEL of the contract, with one switch per unmerged slice.

    A green run here proves the DRIVER reads the contract correctly; it is never
    evidence about tensorhub. That is precisely why every phase the model cannot
    reach is reported BLOCKED rather than skipped.
    """

    def __init__(self, *, provisioning: bool = False, invoke_route: bool = False,
                 settlement: bool = False, ready_after_reads: int = 1,
                 margin_bps: int = 2000, pod_rate_micros_per_hour: int = 440_000,
                 window_s: int = 300, advance_per_read_s: int = 120,
                 actor: str = "admin:rig@cozy", break_invariant: str = "",
                 seal_evidence: bool = True, pod_gpu_class: str = "NVIDIA A40") -> None:
        self.provisioning = provisioning
        self.invoke_route = invoke_route
        self.settlement = settlement
        self.ready_after_reads = ready_after_reads
        self.margin_bps = margin_bps
        self.pod_rate = pod_rate_micros_per_hour
        self.window_s = window_s
        self.advance = advance_per_read_s
        self.actor = actor
        self.break_invariant = break_invariant
        #: th#1355 + th#1839 are on tensorhub master, so these routes exist by
        #: default; turned off only to prove the driver reports absence as
        #: BLOCKED rather than passing quietly.
        self.seal_evidence = seal_evidence
        #: What a REAL hub records: the provider's catalogue id, not a slug.
        #: Keeping it faithful is what makes sku_slug load-bearing in the tests
        #: instead of an identity function over an already-slugged value.
        self.pod_gpu_class = pod_gpu_class
        self.compiled_graphs: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.now = 0
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.requests: Dict[str, Dict[str, Any]] = {}
        self._seq = 0

    def _id(self) -> str:
        self._seq += 1
        return f"00000000-0000-4000-8000-{self._seq:012d}"

    def create(self, org: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        gpu, lane = parse_pair(str(body["pair"]))
        if not str(body.get("release_id", "")).strip():
            raise ApiError("POST", "/create", 400, "invalid_request", "release_id is required")
        deployment_id = self._id()
        pods: List[Dict[str, Any]] = []
        if self.provisioning:
            chooser = (CHOOSER_ORCHESTRATOR if self.break_invariant == "chooser"
                       else CHOOSER_PRIVATE_DEPLOYMENT)
            for index in range(int(body.get("pod_count", 1))):
                pods.append({"pod_id": f"pod-{deployment_id[-4:]}-{index}", "state": "provisioning",
                             "gpu_class": self.pod_gpu_class, "cost_micros_per_hour": self.pod_rate,
                             "placement_chooser": chooser, "private_deployment_id": deployment_id})
        row = {
            "id": deployment_id, "org_id": self._id(), "created_by": self.actor,
            "release_id": body["release_id"], "pair": f"{gpu}:{lane}",
            "gpu_slug": gpu, "execution_lane": lane,
            "pod_count": int(body.get("pod_count", 1)),
            "access_mode": body.get("access_mode") or ACCESS_OWNER,
            "on_pod_failure": body.get("on_pod_failure") or ON_POD_FAILURE_REPLACE,
            "spend_limit_micros": (int(float(body["spend_limit_usd"]) * 1_000_000)
                                   if body.get("spend_limit_usd") else None),
            "rate_margin_bps_at_create": self.margin_bps,
            "state": STATE_ACTIVE, "stop_reason": STOP_REASON_NONE,
            "config_generation": 1, "stopped_at": None,
            "_pods": pods, "_reads": 0, "_started": self.now, "_ended": None,
            "_history": [{"config_generation": 1, "source": HISTORY_SOURCE_CREATE,
                          "actor": self.actor, "reason": body.get("reason", ""),
                          "classes": ["lifecycle"]}],
        }
        self.rows[deployment_id] = row
        return self._view(row)

    @staticmethod
    def _view(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in row.items() if not key.startswith("_")}

    def _row(self, deployment_id: str) -> Dict[str, Any]:
        row = self.rows.get(deployment_id)
        if row is None:
            raise ApiError("GET", "/get", 404, "private_deployment_not_found", "no such rental")
        return row

    def get(self, org: str, deployment_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        row = self._row(deployment_id)
        self.now += self.advance
        row["_reads"] += 1
        if self.provisioning and row["state"] == STATE_ACTIVE and row["_reads"] >= self.ready_after_reads:
            for pod in row["_pods"]:
                if pod["state"] == "provisioning":
                    pod["state"] = "ready"
        return self._view(row), list(row["_pods"])

    def stop(self, org: str, deployment_id: str, reason: str) -> Tuple[Dict[str, Any], bool]:
        row = self._row(deployment_id)
        if row["state"] != STATE_ACTIVE:
            if self.break_invariant == "stop_not_idempotent":
                row["config_generation"] += 1
                row["_history"].append({"config_generation": row["config_generation"],
                                        "source": HISTORY_SOURCE_STOP, "actor": self.actor})
            return self._view(row), True
        self.now += self.advance
        row["stop_reason"] = STOP_REASON_ADMIN
        row["config_generation"] += 1
        row["_ended"] = self.now
        if not [p for p in row["_pods"] if p["state"] not in DEAD_POD_STATES]:
            row["state"] = STATE_STOPPED
            row["stopped_at"] = f"t+{self.now}"
        else:
            row["state"] = STATE_STOPPING
            for pod in row["_pods"]:
                pod["state"] = "terminated"
            if self.break_invariant == "coherence":
                row["stopped_at"] = f"t+{self.now}"
        row["_history"].append({"config_generation": row["config_generation"],
                                "source": HISTORY_SOURCE_STOP, "actor": self.actor,
                                "reason": reason})
        return self._view(row), False

    def config_history(self, org: str, deployment_id: str) -> List[Dict[str, Any]]:
        return list(self._row(deployment_id)["_history"])

    def usage(self, org: str, deployment_id: str) -> Dict[str, Any]:
        row = self._row(deployment_id)
        self.now += self.advance
        end = row["_ended"] if row["_ended"] is not None else self.now
        pods = len(row["_pods"])
        pod_seconds = max(0, end - row["_started"]) * pods
        return {"deployment_id": deployment_id, "state": row["state"],
                "pod_rows": pods, "live_pods": len([p for p in row["_pods"]
                                                    if p["state"] not in DEAD_POD_STATES]),
                "pod_seconds": pod_seconds,
                "rate_margin_bps_at_create": row["rate_margin_bps_at_create"],
                "spend_limit_micros": row["spend_limit_micros"],
                "settlement": self._settlement(row, pod_seconds)}

    def _settlement(self, row: Mapping[str, Any], pod_seconds: int) -> Dict[str, Any]:
        pods = max(1, len(row["_pods"]))
        if not self.settlement or pod_seconds < self.window_s * pods:
            out: Dict[str, Any] = {"available": False,
                                   "reason": "no settlement window has opened for this rental yet"}
            if self.break_invariant == "phantom_figures":
                out["captured_micros"] = 12345
            return out
        margin = row["rate_margin_bps_at_create"]
        if self.break_invariant == "margin_drift":
            margin += 500
        windows: List[Dict[str, Any]] = []
        captured = provider_total = billed = 0
        start = int(row["_started"])
        remaining = pod_seconds
        while remaining >= self.window_s * pods:
            seconds = self.window_s * pods
            provider = self.pod_rate * seconds // 3600
            amount = provider + provider * margin // 10000
            windows.append({"window_start": f"t+{start}", "window_end": f"t+{start + self.window_s}",
                            "status": "settled", "kind": "capture", "margin_bps": margin,
                            "held_micros": amount, "amount_micros": amount,
                            "provider_micros": provider, "pod_seconds": seconds})
            captured += amount
            provider_total += provider
            billed += seconds
            remaining -= seconds
            start += self.window_s
        return {"available": True, "windows": len(windows), "open_windows": 0,
                "pending_windows": 0, "quarantined_windows": 0,
                "settled_micros": captured, "captured_micros": captured, "held_micros": 0,
                "provider_micros": provider_total, "billed_pod_seconds": billed,
                "recent_windows": windows}

    def invoke(self, deployment_id: str, function: str, payload: Mapping[str, Any]) -> str:
        if not self.invoke_route:
            raise ApiError("POST", f"/v1/private-deployments/{deployment_id}/{function}",
                           404, "", "404 page not found")
        row = self._row(deployment_id)
        if row["state"] != STATE_ACTIVE:
            raise ApiError("POST", "/invoke", 409, "private_deployment_stopped", "this rental is stopped")
        if not [p for p in row["_pods"] if p["state"] in READY_POD_STATES]:
            raise ApiError("POST", "/invoke", 409, "private_deployment_stopped", "no ready pod")
        request_id = self._id()
        self.requests[request_id] = {"id": request_id, "status": "queued", "_reads": 0,
                                     "_deployment_id": deployment_id}
        return request_id

    def request(self, request_id: str) -> Dict[str, Any]:
        record = self.requests.get(request_id)
        if record is None:
            raise ApiError("GET", "/v1/requests", 404, "request_not_found", "no such request")
        self.now += self.advance
        record["_reads"] += 1
        if record["_reads"] >= 2 and record["status"] != "completed":
            self._mint_graph(record["_deployment_id"], record)
        record["status"] = "completed" if record["_reads"] >= 2 else "running"
        return {key: value for key, value in record.items() if not key.startswith("_")}

    @staticmethod
    def _digest(seed: str) -> str:
        return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()

    def _mint_graph(self, deployment_id: str, request: Dict[str, Any]) -> None:
        """What a worker leaves behind when it seals and publishes a graph."""
        row = self.rows[deployment_id]
        pod = row["_pods"][0] if row["_pods"] else {}
        sku = sku_slug(str(pod.get("gpu_class", "")))
        if self.break_invariant == "seal_sku_mismatch":
            sku = "some-other-card"
        release = row["release_id"]
        if self.break_invariant == "seal_wrong_release":
            release = "a-release-this-rental-did-not-pin"
        existing = next(
            (graph for graph in self.compiled_graphs
             if graph.get("minted_for_release_id") == release),
            None,
        )
        outcome = "reused" if existing is not None else "published"
        if existing is None:
            token = self._id()
            graph_key = "cg-key-v1-" + hashlib.sha224(token.encode()).hexdigest()
            artifact_ref = self._digest("artifact:" + token)
            if self.break_invariant == "seal_no_artifact":
                artifact_ref = ""
            existing = {
                "compiled_graph_key": graph_key,
                "family": "sdxl",
                "lane": row["execution_lane"],
                "sm": "sm86",
                "sku": sku,
                "artifact_digest": artifact_ref,
                "gen_worker_version": "0.116.0",
                "minted_for_release_id": release,
                "minted_by_pod_id": pod.get("pod_id", ""),
                "publisher_tier": "platform",
            }
            self.compiled_graphs.append(existing)
        graph_key = str(existing["compiled_graph_key"])
        artifact_ref = str(existing["artifact_digest"])
        receipt_ref = self._digest("receipt:" + graph_key)
        object_refs = [self._digest("manifest:" + graph_key), artifact_ref]
        raw_pixels = bytes((0, 17, 34, 51, 68, 85))
        pixel = {
            "mode": "RGB",
            "width": 2,
            "height": 1,
            "sha256": decoded_pixel_digest("RGB", 2, 1, raw_pixels),
        }
        serving_pid = 4242
        compile_child: Optional[Dict[str, Any]] = None
        if outcome == "published":
            compile_child = {
                "pid": 5252,
                "ppid": serving_pid,
                "role": "compile_child",
                "start": 1000.0,
                "finish": 1001.0,
            }
        evidence: Dict[str, Any] = {
            "status": "completed",
            "outcome": outcome,
            "refusal": "",
            "compiled_graph_key": graph_key,
            "artifact_ref": artifact_ref,
            "receipt_ref": receipt_ref,
            "gen_worker_version": "0.116.0",
            "torch_compiled_graphs_version": "0.3.0",
            "hashrepo_version": "0.3.1",
            "hashrepo": {
                "manifest_ref": self._digest("hashrepo:" + graph_key),
                "object_refs": object_refs,
                "materialized_root": f"/var/lib/gen-worker/compiled-graphs/{graph_key}",
                "local_ref_count_before": 0,
                "local_object_count_before": 0,
                "reference_count": len(object_refs),
                "object_count": len(set(object_refs)),
                "materialized_object_count": len(set(object_refs)),
            },
            "compile_child": compile_child,
            "serving_pid_before": serving_pid,
            "serving_pid_after": serving_pid,
            "serving_compile_count": 0,
            "compile_count": 1 if outcome == "published" else 0,
            "child_spawn_count": 1 if outcome == "published" else 0,
            "bind_fqns": ["transformer.bias", "transformer.weight"],
            "bind_call_count": 1,
            "runner_call_count": 1,
            "decoded_pixel": pixel,
        }
        self._break_evidence(evidence, outcome)
        request["compiled_graph_evidence"] = evidence
        request["decoded_pixel_raw_base64"] = base64.b64encode(raw_pixels).decode("ascii")
        self.events.append({"kind": "aot_mint_phases", "phase": "minted", "state": "completed",
                            "release_id": row["release_id"], "pod_id": pod.get("pod_id", "")})
        if self.break_invariant == "seal_failed_publish":
            self.events.append({"kind": "aot_mint_phases", "phase": "pack_failed",
                                "state": "failed", "release_id": row["release_id"],
                                "error": "artifact pack refused: short write",
                                "pod_id": pod.get("pod_id", "")})

    def _break_evidence(self, evidence: Dict[str, Any], outcome: str) -> None:
        broken = self.break_invariant
        if broken == "evidence_status":
            evidence["status"] = "running"
        elif broken == "evidence_outcome":
            evidence["outcome"] = "unknown"
        elif broken == "evidence_refusal":
            evidence["refusal"] = "compile_refused"
        elif broken == "evidence_graph_key":
            evidence["compiled_graph_key"] = ""
        elif broken == "evidence_artifact_ref":
            evidence["artifact_ref"] = ""
        elif broken == "evidence_receipt_ref":
            evidence["receipt_ref"] = ""
        elif broken == "evidence_versions":
            evidence["torch_compiled_graphs_version"] = ""
        elif broken == "evidence_serving_pid":
            evidence["serving_pid_after"] = 4343
        elif broken == "evidence_serving_compile":
            evidence["serving_compile_count"] = 1
        elif broken == "evidence_manifest_ref":
            evidence["hashrepo"]["manifest_ref"] = ""
        elif broken == "evidence_materialized_root":
            evidence["hashrepo"]["materialized_root"] = ""
        elif broken == "evidence_nonempty_cache":
            evidence["hashrepo"]["local_object_count_before"] = 1
        elif broken == "evidence_ref_count":
            evidence["hashrepo"]["reference_count"] += 1
        elif broken == "evidence_object_count":
            evidence["hashrepo"]["materialized_object_count"] += 1
        elif broken == "evidence_bind_fqns":
            evidence["bind_fqns"] = ["transformer.weight", "transformer.bias"]
        elif broken == "evidence_bind_calls":
            evidence["bind_call_count"] = 0
        elif broken == "evidence_runner_calls":
            evidence["runner_call_count"] = 0
        elif broken == "evidence_pixel_shape":
            evidence["decoded_pixel"]["width"] = 0
        elif broken == "evidence_pixel_digest":
            evidence["decoded_pixel"]["sha256"] = "sha256:" + "0" * 64
        elif broken == "evidence_publish_compile" and outcome == "published":
            evidence["compile_count"] = 0
        elif broken == "evidence_publish_spawn" and outcome == "published":
            evidence["child_spawn_count"] = 0
        elif broken == "evidence_compile_child" and outcome == "published":
            evidence["compile_child"]["role"] = "serving"
        elif broken == "evidence_reuse_compile" and outcome == "reused":
            evidence["compile_count"] = 1
        elif broken == "evidence_reuse_spawn" and outcome == "reused":
            evidence["child_spawn_count"] = 1
        elif broken == "evidence_reuse_child" and outcome == "reused":
            evidence["compile_child"] = {
                "pid": 5252, "ppid": 4242, "role": "compile_child",
                "start": 1000.0, "finish": 1001.0,
            }

    def admin_compiled_graphs(self, release: str) -> List[Dict[str, Any]]:
        if not self.seal_evidence:
            raise ApiError("GET", "/v1/admin/compiled-graphs", 404, "", "404 page not found")
        return [graph for graph in self.compiled_graphs
                if not release or graph["minted_for_release_id"] == release]

    def admin_activity_events(self, release: str, state: str) -> List[Dict[str, Any]]:
        if not self.seal_evidence:
            raise ApiError("GET", "/v1/admin/worker-activity-events", 404, "", "404 page not found")
        return [e for e in self.events
                if (not release or e["release_id"] == release) and (not state or e["state"] == state)]


def _main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hub", default="", help="hub base URL")
    parser.add_argument("--token", default="", help="bearer token (or TENSORHUB_TOKEN in the env)")
    parser.add_argument("--org", default="tensorhub")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--release", default="", help="release id — PINNED; untagged is fine")
    parser.add_argument("--pair", default="", help="gpu:lane, e.g. a40:bf16-w16a16+compiled")
    parser.add_argument("--function", default="generate")
    parser.add_argument("--payload", default="{}")
    parser.add_argument("--invocations", type=int, default=1)
    parser.add_argument("--spend-limit-usd", type=float, default=1.0)
    parser.add_argument("--json", default="", help="write the JSON report here")
    parser.add_argument("--confirm", action="store_true", help="required: this buys a real pod")
    parser.add_argument("--dry", action="store_true", help="run against the in-process model; no hub, no money")
    parser.add_argument("--state", choices=("resource", "fences", "merged"), default="resource",
                        help="--dry only: which slices have merged")
    args = parser.parse_args(argv)

    if args.dry:
        model = ContractModel(
            provisioning=args.state in ("fences", "merged"),
            invoke_route=args.state == "merged",
            settlement=args.state == "merged",
        )
        leg = Leg(org="tensorhub", endpoint="tensorhub/sdxl", release_id="dry-run-release",
                  pair=("a40", "bf16-w16a16+compiled"), invocations=1)
        # MODEL seconds, not wall seconds: the model clock advances per read,
        # so this is "give up after ~15 reads with nothing changing".
        result = run_leg(model, leg, poll_s=0.0, stall_budget_s=1800.0, log=print,
                         clock=lambda: float(model.now), sleep=lambda _s: None)
    else:
        if not args.hub or not args.endpoint or not args.release or not args.pair:
            parser.error("--hub, --endpoint, --release and --pair are required without --dry")
        token = args.token or os.environ.get("TENSORHUB_TOKEN", "")
        if not token:
            parser.error("a bearer token is required (--token or TENSORHUB_TOKEN)")
        leg = Leg(org=args.org, endpoint=args.endpoint, release_id=args.release,
                  pair=parse_pair(args.pair), function=args.function,
                  payload=json.loads(args.payload), invocations=args.invocations,
                  spend_limit_usd=args.spend_limit_usd)
        leg.validate()
        if not args.confirm:
            print("this buys a real pod on a real card; re-run with --confirm once the leg is approved",
                  file=sys.stderr)
            return 2
        result = run_leg(HttpDeploymentAPI(args.hub, token), leg, log=print)

    print(f"\nleg -> {result.status} (deployment {result.deployment_id})")
    for blocker in result.blockers:
        print(f"  blocked on {blocker}")
    for finding in result.findings:
        if finding.status == FAILED:
            print(f"  FAILED {finding.ident}: {finding.detail}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result.to_json(), handle, indent=2)
    return 0 if result.status in (OK, BLOCKED) else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
