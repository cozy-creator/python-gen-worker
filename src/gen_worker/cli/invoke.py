"""``gen-worker invoke <function-name> <payload>`` — fire one request at a
running ``gen-worker serve``.

Address by FUNCTION NAME — the unique routable id declared on
its routable name. The server resolves which handler hosts it, so
there is no ``--class`` / ``--method`` to specify.

``<payload>`` accepts the three usual forms:

* inline JSON string: ``gen-worker invoke marco_polo '{"text":"marco"}'``
* ``@file.json`` (curl convention): ``gen-worker invoke marco_polo @req.json``
* ``-`` / piped stdin: ``echo '{"text":"marco"}' | gen-worker invoke marco_polo -``

Connects to the running serve via ``./.gen-worker.sock`` (override ``--socket``),
sends one NDJSON request ``{"function": ..., "payload": ...}``, reads one NDJSON
response, and prints the result to stdout. Exits non-zero with a clear stderr
message if no serve is running or the request errored.

The full design lives in ``progress.json`` issue #340.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any, List, Optional

from . import run as run_mod
from . import transport
from .serve import DEFAULT_SOCKET_PATH


# --------------------------------------------------------------------------
# Client-side cancellation — two-stage Ctrl-C
# --------------------------------------------------------------------------

class _ClientCanceler:
    """Two-stage SIGINT for a request in flight against a warm ``serve``.

    1st Ctrl-C: open a FRESH connection and send a ``{"cancel":{request_id}}``
    control frame, then keep waiting — the server trips ``ctx._cancel()`` for
    THIS request and the (canceled) response comes back on the original socket;
    the server stays running. 2nd Ctrl-C within 2s: detach hard (exit 130).
    Cancelling the request and killing the worker are deliberately separate
    (#352): this only ever exits the CLIENT.
    """

    def __init__(self, sock_path: Path, request_id: str) -> None:
        self._sock_path = sock_path
        self._request_id = request_id
        self._last_at = 0.0
        self._installed = False
        self._prev: Any = signal.SIG_DFL

    def install(self) -> None:
        try:
            self._prev = signal.signal(signal.SIGINT, self._on_sigint)
            self._installed = True
        except (ValueError, OSError):
            self._installed = False

    def restore(self) -> None:
        if not self._installed:
            return
        try:
            signal.signal(signal.SIGINT, self._prev)
        except Exception:
            pass
        self._installed = False

    def _send_cancel(self) -> None:
        try:
            c = transport.create_client(str(self._sock_path), 5.0)
            c.sendall(
                (json.dumps({"cancel": {"request_id": self._request_id}}) + "\n").encode("utf-8")
            )
            try:
                c.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            c.close()
        except OSError:
            pass

    def _on_sigint(self, _signum: int, _frame: Optional[types.FrameType]) -> None:
        now = time.monotonic()
        if self._last_at and (now - self._last_at) < 2.0:
            sys.stderr.write("\ngen-worker invoke: detaching (exit 130); the "
                             "request may still finish server-side.\n")
            sys.stderr.flush()
            os._exit(run_mod.EXIT_SIGINT)
        self._last_at = now
        sys.stderr.write(
            "\ngen-worker invoke: canceling request on server "
            "(Ctrl-C again within 2s to detach)\n"
        )
        sys.stderr.flush()
        self._send_cancel()


# --------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------

def add_subparser(sub: argparse._SubParsersAction[Any]) -> None:
    """Register the ``invoke`` subcommand on the top-level parser."""
    p = sub.add_parser(
        "invoke",
        help="Fire one request at a running 'gen-worker serve'.",
        description=(
            "Connect to a running 'gen-worker serve' over its Unix domain "
            "socket, send one NDJSON request for <function-name> with the "
            "given JSON payload, and print the result to stdout. Payload "
            "accepts inline JSON, @file.json (curl convention), or - (stdin)."
        ),
    )
    p.add_argument(
        "function_name",
        help="The @inference.function(name=...) to invoke (e.g. marco_polo).",
    )
    p.add_argument(
        "args",
        nargs="*",
        help=(
            "Either a single JSON payload (inline string, @file.json, or - for "
            "stdin), OR ergonomic tokens: 'field=value' (coerced), 'field:=<json>', "
            "'field@path', or a bare value for the primary field. "
            "E.g. gen-worker invoke generate \"a cat\" seed=42"
        ),
    )
    p.add_argument(
        "--config", dest="config_path", default=None,
        help="Path to the endpoint's pyproject.toml (for schema-aware coercion of ergonomic args).",
    )
    p.add_argument(
        "--module", dest="module", default=None,
        help="Python module to import for schema (overrides [tool.gen_worker] main).",
    )
    p.add_argument(
        "--socket", dest="socket_path", default=DEFAULT_SOCKET_PATH,
        help=(
            "Unix domain socket of the running serve "
            f"(default: {DEFAULT_SOCKET_PATH})."
        ),
    )
    p.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the result with newlines + 2-space indent.",
    )
    p.add_argument(
        "--stream", action="store_true",
        help=(
            "Stream events as they are produced (one JSON line each) instead of "
            "buffering until the request completes — useful for generator "
            "endpoints (#344)."
        ),
    )
    p.add_argument(
        "--timeout", type=float, default=0.0,
        help=(
            "Seconds to wait for the response (0 = wait forever, the "
            "default -- a first request per model may cold-load weights "
            "for minutes)."
        ),
    )
    p.set_defaults(_handler=_handle_invoke)


# --------------------------------------------------------------------------
# Payload reading (inline / @file / - stdin)
# --------------------------------------------------------------------------

def _read_payload(spec: str) -> Any:
    """Resolve a single JSON-blob payload arg into a decoded JSON value."""
    if spec == "-":
        raw = sys.stdin.read()
    elif spec.startswith("@"):
        path = Path(spec[1:])
        if not path.exists():
            raise run_mod._UsageError(f"payload file not found: {path}")
        raw = path.read_text(encoding="utf-8")
    else:
        raw = spec
    raw = raw.strip() or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise run_mod._UsageError(f"payload is not valid JSON: {e}") from e


def _schema_for(function_name: str, config_path: Any, module: Any) -> Any:
    """Best-effort payload Struct for ``function_name`` (None if unimportable).

    Importing the endpoint module is only attempted when ergonomic tokens are
    used; the plain ``invoke fn '<json>'`` path never imports it (stays thin).
    """
    try:
        _root, mod = run_mod.load_endpoint_module(config_path=config_path, module=module)
        for c in run_mod.discover_candidates(mod):
            if c.fn_name == function_name:
                return c.payload_type
    except Exception:
        return None
    return None


def _resolve_payload(args: argparse.Namespace) -> Any:
    """Turn invoke's positional ``args`` into a decoded JSON payload.

    A single JSON/@file/- blob takes the thin path. Otherwise the tokens are
    ergonomic ``field=value`` args coerced against the function's schema (#350).
    """
    from .args import ArgError, build_payload, looks_like_field_token

    tokens: List[str] = list(getattr(args, "args", []) or [])
    if not tokens:
        return {}

    # Single blob (JSON / @file / -) -> thin path, no module import.
    if len(tokens) == 1 and not looks_like_field_token(tokens[0]):
        only = tokens[0]
        if only == "-" or only.startswith("@"):
            return _read_payload(only)
        try:
            return json.loads(only)  # explicit JSON blob
        except json.JSONDecodeError:
            pass  # a bare primary value like "a cat" -> fall through to ergonomic

    struct_type = _schema_for(args.function_name, args.config_path, args.module)
    try:
        return build_payload(tokens, struct_type)
    except ArgError as e:
        raise run_mod._UsageError(str(e)) from e


# --------------------------------------------------------------------------
# Socket client
# --------------------------------------------------------------------------

_CONNECT_TIMEOUT_SECONDS = 10.0
_WAIT_NOTICE_SECONDS = 15.0


def _send_request(
    sock_path: Path, request: dict, timeout: float = 0.0, on_frame: Any = None,
) -> dict:
    """Connect, send one NDJSON request, read the NDJSON response(s).

    With ``on_frame`` (streaming, #344): each streamed event frame
    (``{"event",...}``) is passed to ``on_frame`` as it arrives; the TERMINAL
    envelope (``{"ok":...}``) is returned. Without it, the single response
    envelope is returned (default).

    The CONNECT is always bounded (a healthy serve accepts instantly). The
    RESPONSE wait is unbounded by default (``timeout`` <= 0): a first request
    per model may legitimately cold-load weights for minutes, and a client-side
    deadline firing while the server works correctly misleads the developer
    into debugging a healthy system. While waiting without a deadline, a
    one-time notice is printed so a long cold load is distinguishable from a
    hang. A crashed serve closes the socket, which ends the wait immediately.
    """
    spec = str(sock_path)
    canceler: Optional[_ClientCanceler] = None
    s: Optional[socket.socket] = None
    try:
        try:
            s = transport.create_client(spec, _CONNECT_TIMEOUT_SECONDS)
        except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
            raise run_mod._UsageError(
                f"could not connect to serve at {transport.display(spec)}: {e}; "
                "is 'gen-worker serve' running? (start it with `gen-worker serve` "
                "or pass --socket PATH / tcp://host:port)"
            ) from e
        line = (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")
        s.sendall(line)
        try:
            s.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        # Ctrl-C while waiting cancels THIS request (by request_id) and leaves
        # the server running; a second Ctrl-C detaches. No-op if the request
        # carries no id.
        rid = request.get("request_id")
        if rid:
            canceler = _ClientCanceler(sock_path, rid)
            canceler.install()
        buf = bytearray()
        terminal: Any = None
        deadline = (time.monotonic() + timeout) if timeout and timeout > 0 else None
        notified = False
        while terminal is None:
            # Drain any complete lines already buffered.
            while b"\n" in buf:
                raw_line, _, rest = bytes(buf).partition(b"\n")
                buf = bytearray(rest)
                ln = raw_line.strip()
                if not ln:
                    continue
                try:
                    frame = json.loads(ln.decode("utf-8"))
                except json.JSONDecodeError as e:
                    raise run_mod._UsageError(f"serve returned invalid JSON: {e}") from e
                if on_frame is not None and isinstance(frame, dict) and "event" in frame:
                    on_frame(frame)  # streamed event; keep reading
                    continue
                terminal = frame  # terminal envelope
                break
            if terminal is not None:
                break
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise run_mod._UsageError(
                        f"no response from serve within --timeout={timeout:g}s "
                        f"(the request may still be running server-side)"
                    )
                s.settimeout(min(remaining, _WAIT_NOTICE_SECONDS))
            else:
                s.settimeout(_WAIT_NOTICE_SECONDS)
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                if deadline is None and not notified:
                    sys.stderr.write(
                        "gen-worker invoke: still waiting (a cold model load "
                        "can take minutes; Ctrl-C to abort, or pass "
                        "--timeout SECONDS)\n"
                    )
                    sys.stderr.flush()
                    notified = True
                continue
            if not chunk:
                break
            buf.extend(chunk)
    finally:
        if canceler is not None:
            canceler.restore()
        if s is not None:
            s.close()
    if terminal is None:
        raise run_mod._UsageError("serve closed the connection with no response")
    return terminal


# --------------------------------------------------------------------------
# Handler
# --------------------------------------------------------------------------

def _print_value(value: Any, pretty: bool) -> None:
    if pretty:
        sys.stdout.write(json.dumps(value, indent=2, default=str) + "\n")
    else:
        sys.stdout.write(json.dumps(value, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()


def _handle_invoke(args: argparse.Namespace) -> int:
    stream = bool(getattr(args, "stream", False))
    try:
        payload = _resolve_payload(args)
        # Unix paths resolve to absolute; a tcp://host:port spec passes through.
        if transport.is_unix(args.socket_path):
            sock_path: Any = str(
                Path(transport.parse_addr(args.socket_path).host).resolve())
        else:
            sock_path = args.socket_path
        request = {
            "request_id": uuid.uuid4().hex,
            "function": args.function_name,
            "payload": payload,
        }
        if stream:
            request["stream"] = True
        on_frame = (lambda ev: _print_value(ev.get("value"), args.pretty)) if stream else None
        resp = _send_request(
            sock_path, request,
            timeout=float(getattr(args, "timeout", 0.0) or 0.0),
            on_frame=on_frame,
        )
    except run_mod._UsageError as e:
        sys.stderr.write(f"gen-worker invoke: {e}\n")
        return run_mod.EXIT_USAGE
    except run_mod._ModelResolutionError as e:
        sys.stderr.write(f"gen-worker invoke: {e}\n")
        return run_mod.EXIT_MODEL_RESOLUTION

    if not resp.get("ok", False):
        err = resp.get("error") or {}
        kind = err.get("kind", "error")
        msg = err.get("message", "unknown error")
        sys.stderr.write(f"gen-worker invoke: {kind}: {msg}\n")
        # Map server-side error kind to the same exit-code matrix as `run`.
        if kind == "usage" or kind == "not_found":
            return run_mod.EXIT_USAGE
        if kind == "model_resolution":
            return run_mod.EXIT_MODEL_RESOLUTION
        if kind == "canceled":
            return run_mod.EXIT_SIGINT
        return run_mod.EXIT_USER_EXCEPTION

    events: List[dict] = resp.get("events") or []
    # Print the result value(s). Yields stream first, then the final result —
    # mirroring `run`'s stdout event stream.
    for ev in events:
        value = ev.get("value")
        if args.pretty:
            line = json.dumps(value, indent=2, default=str)
        else:
            line = json.dumps(value, separators=(",", ":"), default=str)
        sys.stdout.write(line + "\n")
    sys.stdout.flush()
    return run_mod.EXIT_OK
