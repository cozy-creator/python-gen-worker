"""Where a resident endpoint announces itself, and how the client verbs find it."""

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

# NOT ~/.cache/cozy/endpoints — cozy-local already uses that for installed endpoint SOURCE and venvs; "resident" is which of them are UP. Two things under one name is how a `down` ends up deleting a checkout. Config key COZY_ENDPOINT_STATE.
DEFAULT_ENDPOINT_STATE = Path.home() / ".cache" / "cozy" / "resident"

HANDLE_NAME = "endpoint.json"
SOCKET_NAME = "endpoint.sock"
LOG_NAME = "up.log"

HANDLE_VERSION = 1

_SLUG = re.compile(r"[^a-z0-9]+")


class EndpointStateError(RuntimeError):
    """The state directory or handle could not be read/written."""


def state_root() -> Path:
    from .. import config

    settings = config.current_or(config.Settings())
    return Path(getattr(settings, "endpoint_state_root", "") or DEFAULT_ENDPOINT_STATE)


def endpoint_key(endpoint_dir: Path) -> str:
    """``<slug>-<digest8>`` for a resolved endpoint directory."""
    resolved = Path(endpoint_dir).resolve()
    digest = hashlib.blake2b(str(resolved).encode("utf-8"), digest_size=4).hexdigest()
    slug = _SLUG.sub("-", resolved.name.lower()).strip("-") or "endpoint"
    return f"{slug}-{digest}"


@dataclass(frozen=True, slots=True)
class EndpointHandle:
    """The paths one resident endpoint owns."""

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
    """Is this pid a live process we may signal? ``ESRCH`` is the only answer that means "gone"."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def write_handle(handle: EndpointHandle, document: Dict[str, Any]) -> None:
    """Atomically publish the handle."""
    handle.state_dir.mkdir(parents=True, exist_ok=True)
    document = dict(document)
    document.setdefault("handle_version", HANDLE_VERSION)
    tmp = handle.handle_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, default=str), encoding="utf-8")
    tmp.replace(handle.handle_path)


def read_handle(handle: EndpointHandle) -> Optional[Dict[str, Any]]:
    """The live handle, or ``None`` — with a dead one removed on the way out."""
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
    """Remove the handle and its socket."""
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
    """Block until the daemon publishes a READY handle, or it dies."""
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
    """SIGTERM ``pid`` and wait for it to go."""
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
