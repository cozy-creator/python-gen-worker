"""Listen/connect address parsing for serve + invoke.

The default transport is a Unix domain socket (same-host / same-container). For
the Docker / cross-process story, ``serve --listen tcp://0.0.0.0:PORT`` (and
``invoke --socket tcp://host:PORT``) speak the same NDJSON protocol over TCP so
``docker run -p PORT:PORT`` works without exec / bind-mounts (#347).

Address forms:
  * ``tcp://host:port``  -> TCP
  * ``unix:///abs/path`` -> Unix domain socket
  * bare path (default)  -> Unix domain socket
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import NamedTuple

class Address(NamedTuple):
    """("unix", path, 0) | ("tcp", host, port)."""

    scheme: str
    host: str  # unix socket path when scheme == "unix"
    port: int = 0


def parse_addr(spec: str) -> Address:
    s = (spec or "").strip()
    if s.startswith("tcp://"):
        hostport = s[len("tcp://"):]
        host, sep, port = hostport.rpartition(":")
        if not sep or not port.isdigit():
            raise ValueError(f"bad tcp address {spec!r}; expected tcp://host:port")
        return Address("tcp", host or "0.0.0.0", int(port))
    if s.startswith("unix://"):
        return Address("unix", s[len("unix://"):])
    return Address("unix", s)


def is_unix(spec: str) -> bool:
    return parse_addr(spec).scheme == "unix"


def display(spec: str) -> str:
    addr = parse_addr(spec)
    return addr.host if addr.scheme == "unix" else f"tcp://{addr.host}:{addr.port}"


def create_listener(spec: str, backlog: int = 8) -> socket.socket:
    """Bind + listen on ``spec``. Removes a stale unix socket first."""
    addr = parse_addr(spec)
    if addr.scheme == "unix":
        path = Path(addr.host)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        path.parent.mkdir(parents=True, exist_ok=True)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(str(path))
    else:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((addr.host, addr.port))
    s.listen(backlog)
    return s


def cleanup_listener(spec: str) -> None:
    """Remove the unix socket file (no-op for TCP)."""
    addr = parse_addr(spec)
    if addr.scheme == "unix":
        try:
            Path(addr.host).unlink()
        except OSError:
            pass


def create_client(spec: str, connect_timeout: float) -> socket.socket:
    """Connect to ``spec`` (raises OSError/FileNotFoundError on failure)."""
    addr = parse_addr(spec)
    if addr.scheme == "unix":
        if not Path(addr.host).exists():
            raise FileNotFoundError(addr.host)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(connect_timeout)
        s.connect(str(addr.host))
    else:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(connect_timeout)
        s.connect((addr.host or "127.0.0.1", addr.port))
    return s
