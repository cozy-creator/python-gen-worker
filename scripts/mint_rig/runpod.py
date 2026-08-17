"""The RunPod REST layer, behind a Protocol so the driver can be tested at $0.

Everything the rig needs from RunPod is these five calls. They are a
:class:`PodApi` Protocol rather than a concrete class for one reason: the whole
driver — bring-up gating, stall detection, rail arithmetic, teardown's three
verdicts — is then unit-testable against a fake that answers scripted
responses, so the only thing a real pod proves is that the wire matches.

:class:`RunpodRest` is the real implementation. It carries the two things
pod_run.py learned the hard way and a third that file has not yet learned:

  * a `User-Agent` that is not urllib's default (RunPod's edge answers 403 to
    some defaults);
  * `podguard.assert_armed` on the POST path, so this module physically cannot
    create an unguarded pod — the th#1327 gate;
  * every HTTP call carries an explicit connect/read bound, which is a
    TRANSPORT bound (a socket that has produced no bytes) and not a bound on
    the work — the distinction pgw's own `lint_http_timeouts` guard exists for.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

REST_BASE = "https://rest.runpod.io/v1"

#: A socket that has produced no bytes in this long is broken, not slow. This
#: bounds the TRANSPORT, never the work: pod bring-up and compiles are gated by
#: mint_rig.progress, which has no clock at all.
HTTP_TIMEOUT_S = 120.0


class PodNotFound(LookupError):
    """GET /pods/<id> answered 404 — the pod is gone."""


class RestError(RuntimeError):
    def __init__(self, status: int, method: str, path: str, body: str) -> None:
        super().__init__(f"HTTP {status} {method} {path}: {body[:600]}")
        self.status, self.method, self.path, self.body = status, method, path, body


@runtime_checkable
class PodApi(Protocol):
    """What the rig needs from a pod provider. Five calls, no more.

    THERE IS NO CAPACITY QUERY, and that is a fact about the provider rather
    than a gap here. `rest.runpod.io/v1` (openapi.json, read 2026-08-17)
    exposes pods, endpoints, templates, network volumes, container-registry
    auth and billing — and NO gpu-type or availability path at all. An earlier
    draft of this Protocol had a `gpu_types()` call; the real wire answered
    HTTP 400 *"that path does not exist in the specification"*, which is the
    kind of thing a fake cannot tell you.

    So availability is discovered the only way the API allows: ASK for a card
    SET and read the create call's answer. `HTTP 500 "This machine does not
    have the resources to deploy your pod"` is the out-of-capacity signal, and
    it costs nothing — no pod is created, and the rig's name sweep confirms it.
    """

    def create(self, body: Mapping[str, Any]) -> dict[str, Any]: ...

    def get(self, pod_id: str) -> dict[str, Any]:
        """Raise :class:`PodNotFound` on 404 — that is a teardown VERDICT, not
        an error, so it must be distinguishable from a 500."""

    def list_pods(self) -> list[dict[str, Any]]: ...

    def delete(self, pod_id: str) -> None: ...

    def registry_auth(self, name: str, username: str, password: str) -> str: ...


def dotenv(names: Sequence[str], path: Path | None = None) -> dict[str, str]:
    """Read credentials from the workspace's own `.env`, then the environment.

    The order matters and is the workspace convention (`podguard.creds`,
    `pod_run.dotenv`): a value already exported wins nothing over the committed
    operator file, because a stale shell export is how a lane spends money on
    the wrong account.
    """
    envf = path or Path(os.environ.get("COZY_RIG_ENV", Path.home() / "cozy" / "e2e" / ".env"))
    out: dict[str, str] = {}
    if envf.exists():
        for line in envf.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, _, value = line.partition("=")
                if key.strip() in names:
                    out[key.strip()] = value.strip().strip('"').strip("'")
    for name in names:
        if not out.get(name) and os.environ.get(name):
            out[name] = os.environ[name]
    return out


def load_podguard() -> Any:
    """Import podguard from the tracker checkout, or explain why we refuse.

    The rig REFUSES to rent without it. An unguarded pod is exactly th#1323 and
    the whole point of a reusable primitive is that no lane has to remember.
    """
    directory = Path(
        os.environ.get(
            "COZY_PODGUARD_DIR",
            Path.home() / "cozy" / "cozy-creator-tracker" / "scripts" / "podguard",
        )
    )
    if not (directory / "podguard.py").is_file():
        raise RuntimeError(
            f"pgw#1347: podguard not found at {directory}. The rig will not create a pod "
            "without it (th#1327: renting a pod and arming its teardown are one operation). "
            "Set COZY_PODGUARD_DIR."
        )
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    import podguard  # noqa: PLC0415 — deliberately late: an optional workspace peer

    return podguard


class RunpodRest:
    """The real wire."""

    def __init__(self, api_key: str, base: str = REST_BASE, *, guard: Any | None = None) -> None:
        if not api_key:
            raise ValueError("pgw#1347: no RUNPOD_API_KEY (env or ~/cozy/e2e/.env)")
        self._key, self._base = api_key, base.rstrip("/")
        self._guard = guard

    # ---- transport
    def _call(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{self._base}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                # Not decoration: RunPod's edge 403s some default agents.
                "User-Agent": "curl/8.5.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as fh:
                raw = fh.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if exc.code == 404:
                raise PodNotFound(f"{method} {path}: {detail[:200]}") from None
            raise RestError(exc.code, method, path, detail) from None
        return json.loads(raw) if raw else {}

    # ---- PodApi
    def create(self, body: Mapping[str, Any]) -> dict[str, Any]:
        guard = self._guard if self._guard is not None else load_podguard()
        # The th#1327 gate, on OUR path. podguard installs it on its own
        # `rest()`; a driver with its own HTTP client walks around that unless
        # it calls the assertion itself, which pod_run.py does and this does.
        guard.assert_armed(dict(body))
        result = self._call("POST", "/pods", body)
        return dict(result if isinstance(result, dict) else {})

    def get(self, pod_id: str) -> dict[str, Any]:
        result = self._call("GET", f"/pods/{pod_id}")
        return dict(result if isinstance(result, dict) else {})

    def list_pods(self) -> list[dict[str, Any]]:
        result = self._call("GET", "/pods")
        rows = result if isinstance(result, list) else (result or {}).get("data", [])
        return [dict(r) for r in rows or []]

    def delete(self, pod_id: str) -> None:
        try:
            self._call("DELETE", f"/pods/{pod_id}")
        except PodNotFound:
            # Already gone is the outcome we wanted. The 404 CHECK is what
            # decides the teardown verdict, not this call's status.
            return

    def registry_auth(self, name: str, username: str, password: str) -> str:
        result = self._call(
            "POST", "/containerregistryauth", {"name": name, "username": username, "password": password}
        )
        return str((result or {}).get("id", ""))

