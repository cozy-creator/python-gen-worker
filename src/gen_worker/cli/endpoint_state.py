"""Where a resident endpoint announces itself, and how the client verbs find it.

pgw#1491. ``up`` starts ONE daemon and writes a handle; ``down`` and ``run``
read that handle and talk to it. The handle is the whole coupling — there is no
second way to discover a running endpoint, because a second way is how two
execution paths get built (``run`` executes no inference itself; it is a client
of ``up``'s daemon, and refuses by name when nothing is up).

Layout, one endpoint per directory::

    ~/.cache/cozy/resident/<slug>-<digest8>/
        endpoint.json   the handle: pid, listen address, functions, state
        endpoint.sock   the NDJSON socket the daemon accepts on
        up.log          the detached daemon's stdout+stderr

The key is derived from the endpoint directory's RESOLVED path, so two
checkouts of the same endpoint are two endpoints and the same checkout reached
by two spellings is one. The slug is carried alongside the digest purely so a
human reading ``ls`` can tell which directory a state dir belongs to.

Liveness is PROVEN, never assumed: :func:`read_handle` signal-0s the recorded
pid and treats a handle whose process is gone as absent (and removes it). A
stale handle that read as "up" would make ``run`` hang against a socket nobody
is listening on — the exact shape of failure this module exists to prevent.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

#: Per-user root for RESIDENT-endpoint state. Config key
#: ``COZY_ENDPOINT_STATE`` (Settings field ``endpoint_state_root``), never
#: guessed from the environment at the call site (§1.18).
#:
#: NOT ``~/.cache/cozy/endpoints`` — that directory is already taken, by
#: cozy-local, for installed endpoint SOURCE and venvs
#: (``paths.EndpointsDir()``). Two different things under one name is how a
#: `down` ends up deleting a checkout. "endpoints" is where they are
#: INSTALLED; "resident" is which of them are UP right now.
DEFAULT_ENDPOINT_STATE = Path.home() / ".cache" / "cozy" / "resident"

#: The daemon writes this the moment it is accepting requests. Its presence is
#: not enough — see :func:`read_handle`.
HANDLE_NAME = "endpoint.json"
SOCKET_NAME = "endpoint.sock"
LOG_NAME = "up.log"

#: Handle document version. Bump when a field changes meaning; ``run`` refuses
#: a handle it cannot read rather than guessing.
HANDLE_VERSION = 1

_SLUG = re.compile(r"[^a-z0-9]+")


class EndpointStateError(RuntimeError):
    """The state directory or handle could not be read/written."""


def state_root() -> Path:
    from .. import config

    settings = config.current_or(config.Settings())
    return Path(getattr(settings, "endpoint_state_root", "") or DEFAULT_ENDPOINT_STATE)


def endpoint_key(endpoint_dir: Path) -> str:
    """``<slug>-<digest8>`` for a resolved endpoint directory.

    The digest is over the resolved path so the key is stable across cwd and
    symlink spellings; the slug is a human affordance and carries no meaning.
    """
    resolved = Path(endpoint_dir).resolve()
    digest = hashlib.blake2b(str(resolved).encode("utf-8"), digest_size=4).hexdigest()
    slug = _SLUG.sub("-", resolved.name.lower()).strip("-") or "endpoint"
    return f"{slug}-{digest}"


@dataclass(frozen=True, slots=True)
class EndpointHandle:
    """The paths one resident endpoint owns. Pure addressing — no I/O."""

    endpoint_dir: Path
    key: str
    state_dir: Path

    @property
    def handle_path(self) -> Path:
        return self.state_dir / HANDLE_NAME

    @property
    def socket_path(self) -> Path:
        return self.state_dir / SOCKET_NAME

    @property
    def log_path(self) -> Path:
        return self.state_dir / LOG_NAME


def handle_for(endpoint_dir: str | Path) -> EndpointHandle:
    resolved = Path(endpoint_dir).resolve()
    key = endpoint_key(resolved)
    return EndpointHandle(endpoint_dir=resolved, key=key, state_dir=state_root() / key)


def pid_alive(pid: int) -> bool:
    """Is this pid a live process we may signal?

    ``ESRCH`` is the only answer that means "gone". ``EPERM`` means the pid is
    taken by a process this user may not signal, which is still a live pid and
    therefore still a reason not to claim the socket.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def write_handle(handle: EndpointHandle, document: Dict[str, Any]) -> None:
    """Atomically publish the handle. Rename, never truncate-then-write: a
    reader that catches a half-written handle would refuse a live endpoint."""
    handle.state_dir.mkdir(parents=True, exist_ok=True)
    document = dict(document)
    document.setdefault("handle_version", HANDLE_VERSION)
    tmp = handle.handle_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, default=str), encoding="utf-8")
    tmp.replace(handle.handle_path)


def read_handle(handle: EndpointHandle) -> Optional[Dict[str, Any]]:
    """The live handle, or ``None`` — with a dead one removed on the way out.

    "Live" is a signal-0 against the recorded pid, not the presence of the
    file. A crashed daemon leaves both file and socket behind; treating either
    as evidence is how ``run`` ends up connecting to nothing.
    """
    try:
        raw = handle.handle_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise EndpointStateError(f"{handle.handle_path}: {exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        clear_handle(handle)
        return None
    if not isinstance(document, dict):
        clear_handle(handle)
        return None
    if int(document.get("handle_version", 0)) != HANDLE_VERSION:
        raise EndpointStateError(
            f"{handle.handle_path} states handle_version="
            f"{document.get('handle_version')!r}; this gen-worker speaks "
            f"{HANDLE_VERSION}. `gen-worker down` and `up` again."
        )
    if not pid_alive(int(document.get("pid", 0) or 0)):
        clear_handle(handle)
        return None
    return document


def clear_handle(handle: EndpointHandle) -> None:
    """Remove the handle and its socket. Idempotent; never raises."""
    for path in (handle.handle_path, handle.socket_path):
        try:
            path.unlink()
        except OSError:
            pass


def wait_for_handle(
    handle: EndpointHandle,
    *,
    still_running: Callable[[], bool],
    poll_s: float = 0.1,
) -> Dict[str, Any]:
    """Block until the daemon publishes a READY handle, or it dies.

    NO TIMEOUT, deliberately: a cold boot legitimately spends minutes pulling
    weights and arming graphs, and there is no elapsed-time threshold that
    distinguishes that from a hang. The loop's terminating condition is the
    CHILD — if it exits without publishing, that is a boot failure and the log
    says why. Progress is observed, not clocked.

    ``still_running`` is a CALLABLE, not a pid, and that is the whole fix for
    pgw#1523. Signalling a pid cannot answer this question for one's OWN child:
    an exited-but-unreaped child is a ZOMBIE, and `os.kill(pid, 0)` on a zombie
    SUCCEEDS. So a pid-based check reports a child that died in its first
    second as alive forever, and this loop — correctly having no timeout —
    spins until the caller gives up. Measured: `gen-worker up -d` against an
    endpoint whose boot refused immediately hung for the full 10-minute test
    timeout with the refusal already sitting in its log. The caller passes
    `Popen.poll`, which REAPS and therefore reports the truth.
    """
    while True:
        document = read_handle(handle)
        if document is not None and document.get("state") == "ready":
            return document
        if not still_running():
            tail = ""
            try:
                tail = "\n".join(
                    handle.log_path.read_text(encoding="utf-8").splitlines()[-40:]
                )
            except OSError:
                pass
            raise EndpointStateError(
                f"the endpoint exited during boot without becoming ready.\n"
                f"  log: {handle.log_path}\n"
                + (f"  last lines:\n{tail}\n" if tail else "")
            )
        time.sleep(poll_s)


def terminate(pid: int, *, poll_s: float = 0.1) -> bool:
    """SIGTERM ``pid`` and wait for it to go. Returns False if it was not live.

    Unbounded by design, same reason as :func:`wait_for_handle`: a teardown
    that frees 8 GB of VRAM and calls the author's shutdown takes as long as it
    takes, and a timer here would leave a half-torn-down process holding the
    card. The caller (``down``) reports what it is waiting on.
    """
    if not pid_alive(pid):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        raise
    while pid_alive(pid):
        time.sleep(poll_s)
    return True


__all__ = [
    "DEFAULT_ENDPOINT_STATE",
    "EndpointHandle",
    "EndpointStateError",
    "HANDLE_VERSION",
    "clear_handle",
    "endpoint_key",
    "handle_for",
    "pid_alive",
    "read_handle",
    "state_root",
    "terminate",
    "wait_for_handle",
    "write_handle",
]
