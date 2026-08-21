"""Parent-side AUTHORIZATION for child-requested hub actions. The compute child holds no credential; when it needs an identity-bearing hub call it ASKS, and the parent decides, chooses the base URL, attaches the JWT and returns only the response. Treat this table as the authorization surface it is: the child never names a host; never a free-form path (every request must match an entry by method AND full-path regex); never a header (Authorization is the parent's to add); query and body keys are ENUMERATED per entry so an allowlisted path cannot smuggle a parameter. Anything not in the table is refused and reported — a refusal is a security event, not a 404."""

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
    timeout_s: float
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


_REPO = r"[A-Za-z0-9._-]{1,64}/[A-Za-z0-9._-]{1,64}"

ACTIONS: Dict[str, HubAction] = {
    a.name: a
    for a in (
        _a(
            "capability.renew",
            "POST",
            r"^/v1/worker/capability/renew$",
            body=("request_id", "attempt", "capability_token"),
            scoped_to_job=True,
        ),
        _a(
            "c2pa.sign",
            "POST",
            r"^/v1/worker/c2pa/sign$",
            body=("alg", "claim_b64"),
        ),
        _a(
            "compiled_graphs.receipt",
            "GET",
            r"^/v1/worker/compiled-graphs/receipt$",
            query=("compiled_graph_key", "artifact_digest"),
        ),
        _a(
            "compiled_graphs.revocations",
            "GET",
            r"^/v1/worker/compiled-graphs/revocations$",
        ),
        _a(
            "compiled_graphs.publish_intent",
            "POST",
            r"^/v1/worker/compiled-graphs/publish-intent$",
            body=("family", "axes", "entries"),
            timeout_s=60.0,
        ),
        _a(
            "compiled_graphs.resolve",
            "POST",
            r"^/v1/worker/compiled-graphs/resolve$",
            body=("family", "keys"),
            timeout_s=30.0,
        ),
        _a(
            "keysets.fetch",
            "GET",
            r"^/v1/worker/keysets/[0-9a-f]{32}$",
            timeout_s=10.0,
        ),
        _a(
            "keysets.publish",
            "PUT",
            r"^/v1/worker/keysets/[0-9a-f]{32}$",
            body=("schema", "version", "closures"),
            timeout_s=10.0,
        ),
        _a(
            "compiled_graphs.publish_complete",
            "POST",
            r"^/v1/worker/compiled-graphs/publish-complete$",
            body=("family", "compiled_graph_key", "checkpoint_id", "ok", "error"),
            timeout_s=60.0,
        ),
        _a(
            "release.compiled_graphs",
            "GET",
            r"^/v1/worker/releases/[A-Za-z0-9._+-]{1,128}/compiled-graphs$",
            query=("lane", "sm"),
            timeout_s=30.0,
        ),
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

ACTION_REPORT_DETAIL = "report.detail"

ACTION_VIEWER_IDENTITY = "identity.viewer"

_MAX_STR = 8192
_MAX_QUERY_REPEATS = 8
_MAX_JSON_BYTES = 256 * 1024
CONTROL_BODY_CEILING_BYTES = _MAX_JSON_BYTES


PUBLISH_ACTIONS = frozenset({"compiled_graphs.publish_intent", "compiled_graphs.publish_complete"})

_PROBE_ENV = "GEN_WORKER_PROBE"

_PROBE_PUBLISH_ARM_ENV = "GEN_WORKER_PROBE_PUBLISH_ARMED"


def probe_pod() -> bool:
    return str(os.environ.get(_PROBE_ENV, "")).strip().lower() in ("1", "true", "yes")


def publish_disarmed() -> bool:
    """Whether compiled graph publish is disarmed for this pod."""
    return probe_pod() and (
        str(os.environ.get(_PROBE_PUBLISH_ARM_ENV, "")).strip().lower()
        not in ("1", "true", "yes"))


def authorize(req: Dict[str, Any]) -> Tuple[HubAction, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Validate one child action request against the table."""
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
        raise ActionRefused(
            f"{match.name} is disarmed on this pod: {_PROBE_ENV} marks it a "
            f"live-edit probe running rsync'd code, whose compiled graphs carry a "
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
