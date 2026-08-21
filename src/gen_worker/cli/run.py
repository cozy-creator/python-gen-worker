"""``gen-worker run`` — one request against the endpoint that is already up."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List

from . import endpoint_state, sockaddr
from .args import ArgError, build_payload
from .endpoint_state import EndpointStateError

CONNECT_TIMEOUT_S = 10.0


class NotUp(RuntimeError):
    """Nothing is resident for this endpoint."""


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    parser = sub.add_parser(
        "run",
        help="Send one request to the endpoint that is up, and print the result.",
        description=(
            "A client of `gen-worker up`. Requires an endpoint to already be "
            "up; refuses (it never auto-starts one). Payload fields use the "
            "ergonomic grammar: a bare value fills the primary field, "
            "field=value sets one, field:=json for lists/objects, field@path "
            "to read one from a file."
        ),
    )
    parser.add_argument("tokens", nargs="*", metavar="FIELD=VALUE",
                        help="payload tokens; a bare value fills the primary field")
    parser.add_argument("-C", "--endpoint-dir", default=".",
                        help="endpoint directory (default: .)")
    parser.add_argument("--function", default="",
                        help="which entrypoint to call (default: the only one)")
    parser.add_argument("--payload", default="",
                        help="a full JSON payload; FIELD=VALUE tokens merge over it")
    parser.add_argument("--json", dest="json_output", action="store_true",
                        help="print the raw response envelope instead of a summary")
    parser.set_defaults(_handler=run_run)


def _require_up(endpoint_dir: Path) -> Dict[str, Any]:
    handle = endpoint_state.handle_for(endpoint_dir)
    document = endpoint_state.read_handle(handle)
    if document is None:
        raise NotUp(
            f"nothing is up for {endpoint_dir}.\n"
            f"  `run` is a client — it never starts an endpoint itself, "
            f"because a second boot path is a second answer to what actually "
            f"served.\n"
            f"  Start one first:  gen-worker up -d\n"
            f"  Then:             gen-worker run \"<prompt>\""
        )
    return document


def _pick_function(document: Dict[str, Any], requested: str) -> str:
    functions: List[str] = list(document.get("functions") or ())
    if requested:
        if requested not in functions:
            raise NotUp(
                f"this endpoint serves no function {requested!r}; it serves: "
                f"{', '.join(functions) or '(none)'}"
            )
        return requested
    if len(functions) == 1:
        return functions[0]
    if not functions:
        raise NotUp("the endpoint that is up advertises no functions")
    raise NotUp(
        f"this endpoint serves several functions ({', '.join(functions)}); "
        f"name one with --function"
    )


def request(document: Dict[str, Any], frame: Dict[str, Any]) -> Dict[str, Any]:
    """One NDJSON round trip."""
    listen = str(document.get("socket") or "")
    if not listen:
        raise NotUp("the handle names no socket; `gen-worker down` and up again")
    try:
        conn = sockaddr.create_client(listen, CONNECT_TIMEOUT_S)
    except (OSError, FileNotFoundError) as exc:
        raise NotUp(
            f"the endpoint's handle says it is up (pid {document.get('pid')}) "
            f"but its socket {listen} did not accept: {exc}.\n"
            f"  `gen-worker down` clears it."
        ) from exc
    try:
        conn.sendall((json.dumps(frame) + "\n").encode("utf-8"))
        conn.settimeout(None)
        buf = bytearray()
        while b"\n" not in buf:
            chunk = conn.recv(65536)
            if not chunk:
                break
            if len(buf) + len(chunk) > sockaddr.MAX_NDJSON_LINE_BYTES:
                raise NotUp("the endpoint's response exceeded the frame bound")
            buf.extend(chunk)
    except socket.timeout as exc:  # pragma: no cover - timeout is disabled
        raise NotUp(f"the endpoint stopped responding: {exc}") from exc
    finally:
        conn.close()
    line = bytes(buf).split(b"\n", 1)[0].strip()
    if not line:
        raise NotUp(
            "the endpoint closed the connection without answering — it may "
            "have crashed; check its log with `gen-worker down` and `up`"
        )
    return dict(json.loads(line.decode("utf-8")))


def run_run(args: argparse.Namespace) -> int:
    endpoint_dir = Path(args.endpoint_dir).resolve()
    try:
        document = _require_up(endpoint_dir)
        function = _pick_function(document, args.function)
    except (NotUp, EndpointStateError) as exc:
        sys.stderr.write(f"gen-worker run: {exc}\n")
        return 1

    base: Dict[str, Any] = {}
    if args.payload:
        try:
            base = dict(json.loads(args.payload))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            sys.stderr.write(f"gen-worker run: --payload is not a JSON object: {exc}\n")
            return 2
    primary = (document.get("primary_fields") or {}).get(function)
    try:
        payload = build_payload(list(args.tokens), None, base=base, primary=primary)
    except ArgError as exc:
        sys.stderr.write(f"gen-worker run: {exc}\n")
        return 2

    try:
        envelope = request(document, {"function": function, "payload": payload})
    except NotUp as exc:
        sys.stderr.write(f"gen-worker run: {exc}\n")
        return 1

    if args.json_output:
        sys.stdout.write(json.dumps(envelope) + "\n")
        return 0 if envelope.get("ok") else 1

    if not envelope.get("ok"):
        error = envelope.get("error") or {}
        sys.stderr.write(
            f"gen-worker run: {error.get('kind', 'error')}: "
            f"{error.get('message', '')}\n"
        )
        _print_dispatch(envelope)
        return 1

    for warning in envelope.get("warnings") or ():
        sys.stderr.write(f"warn: {warning}\n")
    _print_dispatch(envelope)
    _print_result(envelope.get("result"))
    return 0


def _print_dispatch(envelope: Dict[str, Any]) -> None:
    facts = envelope.get("dispatch")
    if not isinstance(facts, dict):
        return
    armed = int(facts.get("armed_modules", 0) or 0)
    compiled = int(facts.get("compiled_graph_calls", 0) or 0)
    eager = int(facts.get("eager_calls", 0) or 0)
    displaced = facts.get("displaced_modules") or []
    if displaced:
        sys.stderr.write(
            f"dispatch: DISPLACED on {', '.join(displaced)} — served eager\n"
        )
    elif armed == 0:
        sys.stderr.write("dispatch: eager (no compiled graph armed)\n")
    else:
        sys.stderr.write(
            f"dispatch: compiled_graph_calls={compiled} eager_calls={eager}\n"
        )


def _print_result(result: Any) -> None:
    paths = sorted(_local_paths(result))
    if paths:
        for path in paths:
            sys.stdout.write(f"{path}\n")
        return
    sys.stdout.write(json.dumps(result, default=str) + "\n")


def _local_paths(node: Any) -> List[str]:
    found: List[str] = []
    if isinstance(node, dict):
        value = node.get("local_path")
        if isinstance(value, str) and value:
            found.append(value)
        for child in node.values():
            found.extend(_local_paths(child))
    elif isinstance(node, list):
        for child in node:
            found.extend(_local_paths(child))
    return found


__all__ = ["NotUp", "add_subparser", "request", "run_run"]
