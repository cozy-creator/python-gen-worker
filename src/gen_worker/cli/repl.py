"""``gen-worker repl`` — interactive single-endpoint session.

Loads the endpoint + model ONCE (in-process, resident for the session) and loops
over typed requests — the interactive sibling of ``serve --stdin``, reusing the
same ``_Endpoint`` engine plus a prompt, the ergonomic ``field=value`` grammar
(#350), and meta-commands. ``Ctrl-C`` cancels the in-flight request (via
``ctx.cancel()``) and returns to the prompt; ``Ctrl-D`` / ``:quit`` exits and
runs ``shutdown()``.

This is the THIN, single-endpoint REPL (the rich multi-endpoint console lives in
cozy-local). The full design lives in ``progress.json`` issue #348.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sys
import uuid
from typing import Any, List, Optional

from . import run as run_mod
from . import serve as serve_mod


def add_subparser(sub: argparse._SubParsersAction[Any]) -> None:
    """Register the ``repl`` subcommand on the top-level parser."""
    p = sub.add_parser(
        "repl",
        help="Interactive session: load the endpoint once, submit many requests.",
        description=(
            "Load the endpoint + model ONCE and loop over typed requests "
            "(field=value or raw JSON), keeping the model resident. Ctrl-C "
            "cancels the in-flight request; Ctrl-D or ':quit' exits. "
            "Meta-commands: :use <fn>, :schema, :functions, :help, :quit."
        ),
    )
    p.add_argument("--config", dest="config_path", default=None,
                   help="Path to endpoint.toml (defaults to ./endpoint.toml).")
    p.add_argument("--module", dest="module", default=None,
                   help="Python module to import (overrides endpoint.toml `main`).")
    p.add_argument("--function", dest="functions", action="append", default=None,
                   metavar="NAME", help="Boot only the class(es) hosting NAME (repeatable).")
    p.add_argument("--offline", action="store_true",
                   help="Use only the local CAS; fail instead of fetching weights.")
    p.add_argument("--device", dest="device", default=None,
                   help="Override the torch device (e.g. 'cuda:0', 'cpu').")
    p.add_argument("--allow-publish", action="store_true",
                   help="Allow ConversionContext producer RPCs to hit real tensorhub.")
    p.add_argument("--pretty", action="store_true",
                   help="Pretty-print results with newlines + 2-space indent.")
    p.set_defaults(_handler=_handle_repl)


def _handle_repl(args: argparse.Namespace) -> int:
    try:
        return _repl_inner(args)
    except run_mod._UsageError as e:
        sys.stderr.write(f"gen-worker repl: {e}\n")
        return run_mod.EXIT_USAGE
    except run_mod._ModelResolutionError as e:
        sys.stderr.write(f"gen-worker repl: model resolution failed: {e}\n")
        return run_mod.EXIT_MODEL_RESOLUTION


def _repl_inner(args: argparse.Namespace) -> int:
    _root, mod = run_mod.load_endpoint_module(
        config_path=args.config_path, module=args.module,
    )
    candidates = run_mod.discover_candidates(mod)
    candidates = serve_mod._filter_candidates_by_function(
        candidates, getattr(args, "functions", None),
    )
    if args.device:
        os.environ["GEN_WORKER_LOCAL_DEVICE"] = args.device

    endpoint = serve_mod._Endpoint(
        offline=bool(args.offline), allow_publish=bool(args.allow_publish),
    )
    # Load once: eager setup so the model is resident when the prompt appears.
    endpoint.boot(candidates, eager=True)
    names = endpoint.function_names()
    active: Optional[str] = names[0] if len(names) == 1 else None

    sys.stderr.write(
        f"gen-worker repl — {getattr(mod, '__name__', '?')} "
        f"({len(names)} function(s): {', '.join(names)})\n"
        "type ':help' for commands, ':quit' to exit\n"
    )
    if active:
        sys.stderr.write(f"active function: {active}\n")
    sys.stderr.flush()

    try:
        _loop(endpoint, names, active, args)
    finally:
        endpoint.cancel_all()
        endpoint.shutdown()
    return run_mod.EXIT_OK


_HELP = (
    "commands:\n"
    "  <field=value ...> | <raw-json>   submit a request to the active function\n"
    "  :use <name>                      switch the active function\n"
    "  :functions                       list functions\n"
    "  :schema                          show the active function's input schema\n"
    "  :help                            this help\n"
    "  :quit                            exit (or Ctrl-D)\n"
)


def _loop(endpoint: "serve_mod._Endpoint", names: List[str], active: Optional[str],
          args: argparse.Namespace) -> None:
    while True:
        # Prompt to stderr so stdout stays clean (results only). Read from
        # sys.stdin directly so piped input + tests work the same as a TTY.
        sys.stderr.write(f"{active or '(no function)'}> ")
        sys.stderr.flush()
        try:
            raw = sys.stdin.readline()
        except KeyboardInterrupt:
            sys.stderr.write("\n  (Ctrl-C; ':quit' or Ctrl-D to exit)\n")
            continue
        if raw == "":  # EOF (Ctrl-D / end of piped input)
            sys.stderr.write("\n")
            return
        line = raw.strip()
        if not line:
            continue
        if line in (":quit", ":q", ":exit"):
            return
        if line in (":help", ":h"):
            sys.stderr.write(_HELP)
            continue
        if line in (":functions", ":fns"):
            sys.stderr.write("  " + "\n  ".join(names) + "\n")
            continue
        if line == ":schema":
            _print_schema(endpoint, active)
            continue
        if line.startswith(":use"):
            cand = line[len(":use"):].strip()
            if cand in names:
                active = cand
                sys.stderr.write(f"active function: {active}\n")
            else:
                sys.stderr.write(f"unknown function {cand!r}; available: {names}\n")
            continue
        if active is None:
            sys.stderr.write(f"no active function; ':use <name>' first ({names})\n")
            continue
        _run_one(endpoint, active, line, args)


def _print_schema(endpoint: "serve_mod._Endpoint", active: Optional[str]) -> None:
    if active is None:
        sys.stderr.write("no active function\n")
        return
    from .describe import _function_input_schema

    pt = endpoint.functions[active].selected.payload_type
    sys.stderr.write(json.dumps(_function_input_schema(pt), indent=2) + "\n")


def _run_one(endpoint: "serve_mod._Endpoint", fn: str, line: str,
             args: argparse.Namespace) -> None:
    from .args import ArgError, build_payload

    payload_type = endpoint.functions[fn].selected.payload_type
    stripped = line.strip()

    payload: Any = None
    # A raw JSON object/array blob (detected before shlex, which would strip its
    # quotes). Otherwise shell-style tokens for the ergonomic grammar — multi-word
    # values must be quoted ("a cat" steps=5), exactly like a shell.
    if stripped[:1] in ("{", "["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"  invalid JSON: {e}\n")
            return
    if payload is None:
        try:
            tokens = shlex.split(line)
        except ValueError as e:
            sys.stderr.write(f"  parse error: {e}\n")
            return
        try:
            payload = build_payload(tokens, payload_type)
        except ArgError as e:
            sys.stderr.write(f"  {e}\n")
            return

    rid = uuid.uuid4().hex

    def _on_sigint(_signum: int, _frame: Any) -> None:
        endpoint.interrupt_request(rid)
        sys.stderr.write("  (canceling…)\n")
        sys.stderr.flush()

    def _emit(ev: dict) -> None:
        value = ev.get("value")
        if args.pretty:
            sys.stdout.write(json.dumps(value, indent=2, default=str) + "\n")
        else:
            sys.stdout.write(json.dumps(value, separators=(",", ":"), default=str) + "\n")
        sys.stdout.flush()

    prev: Any = None
    try:
        prev = signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):
        prev = None
    try:
        terminal = endpoint.dispatch(fn, payload, request_id=rid, on_event=_emit)
    finally:
        if prev is not None:
            try:
                signal.signal(signal.SIGINT, prev)
            except Exception:
                pass

    if not terminal.get("ok"):
        err = terminal.get("error") or {}
        sys.stderr.write(f"  {err.get('kind', 'error')}: {err.get('message', '')}\n")
