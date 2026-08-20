"""``gen-worker up [-d]`` / ``gen-worker down`` — the resident endpoint.

pgw#1491, docker-compose semantics by Paul's ruling: bare ``up`` is
FOREGROUND — logs stream to the terminal and Ctrl-C stops the endpoint — and
``up -d`` detaches, leaving it resident for ``run`` clients until ``down``.

Foreground is the default because it is the introspection-friendly mode: you
see the boot ladder, the adopt report, the per-request compiled-vs-eager
counter and the teardown, live, in the shell you typed into. ``-d`` is the
daemon mode a client talks to.

``up`` is also where the checkpoint lands. Its boot pulls the refs its runtime
config names through the SAME download path ``gen-worker download`` uses, and
it does not accept a request until they are on disk — which is the whole reason
``run`` may require ``up`` instead of parking a multi-GB fetch inside a user
request (th#2204's invariant, holding by construction rather than by protocol).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional

from . import endpoint_state
from .daemon import BootError, BootSpec, serve
from .endpoint_state import EndpointStateError

#: Hidden flag the detaching parent passes to the child it re-execs. Not part
#: of the user surface; a user who types it gets exactly what the parent would
#: have run, which is the honest behavior for an internal flag.
DETACHED_CHILD_FLAG = "--detached-child"


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    up = sub.add_parser(
        "up",
        help="Bring an endpoint up resident in memory (foreground; -d detaches).",
        description=(
            "Load the endpoint, materialize its configured checkpoint(s), "
            "adopt compiled graphs, and serve requests warm. Foreground by "
            "default: logs stream here and Ctrl-C stops it. With -d it "
            "detaches and `gen-worker run` talks to it until `gen-worker down`."
        ),
    )
    up.add_argument("endpoint_dir", nargs="?", default=".",
                    help="directory holding endpoint.toml (default: .)")
    up.add_argument("-d", "--detach", action="store_true",
                    help="run in the background and return immediately")
    up.add_argument("--checkpoint-ref", action="append", default=[], metavar="REF",
                    help="checkpoint ref to serve (repeatable). Default: what "
                         "this machine's runtime config names.")
    up.add_argument("--checkpoint", default="", metavar="DIR",
                    help="an already-materialized checkpoint tree, bypassing "
                         "ref resolution entirely")
    up.add_argument("--model", default="",
                    help="the hub row's model classification (e.g. sdxl)")
    up.add_argument("--defaults", default="{}",
                    help="per-checkpoint override JSON")
    up.add_argument("--lane", default="", help="active lane contract handle")
    up.add_argument("--sm", default="", help="this GPU's sm (e.g. sm_89); "
                                             "required to adopt compiled graphs")
    up.add_argument("--graph-store", default="", metavar="DIR",
                    help="compiled-graph CAS to adopt from. Default: the box "
                         "graph CAS when it exists; absent = serve eager.")
    # pgw#1526: NO default here. Unset means "ask `cli/workspace.py`", which
    # is the one place that answers it; a default spelled here would be a
    # second answer that can drift from the daemon's.
    up.add_argument("--artifacts-dir", default="", metavar="DIR",
                    help="where compiled artifacts are built and adopted "
                         "from. Default: the box cache "
                         "(~/.cache/cozy/compiled-graphs).")
    up.add_argument("--output-dir", default="outputs",
                    help="where saved images/media land")
    up.add_argument("--compile", dest="compile_policy", default="auto",
                    choices=("auto", "off"),
                    help="auto: fill this boot's compiled-graph holes in the "
                         "background, never blocking a request. off: never "
                         "compile (`gen-worker compile` stays the front door).")
    up.add_argument("--idle-timeout", type=float, default=0.0, metavar="SECONDS",
                    help="self-stop after SECONDS with no request, freeing the "
                         "card. 0 (default) = stay up until `down`.")
    up.add_argument("--env-lockfile", default="",
                    help="uv.lock stating this boot's compile stack "
                         "(default: the endpoint's own)")
    up.add_argument(DETACHED_CHILD_FLAG, dest="detached_child",
                    action="store_true", help=argparse.SUPPRESS)
    up.set_defaults(_handler=run_up)

    down = sub.add_parser(
        "down",
        help="Stop a resident endpoint and free its card.",
        description=(
            "Signal the endpoint brought up by `gen-worker up -d` and wait for "
            "it to tear down (author shutdown, VRAM released, socket removed)."
        ),
    )
    down.add_argument("endpoint_dir", nargs="?", default=".",
                      help="directory holding endpoint.toml (default: .)")
    down.set_defaults(_handler=run_down)


# --------------------------------------------------------------------------
# up
# --------------------------------------------------------------------------


def _artifacts_dir(args: argparse.Namespace) -> Path:
    """STATED wins; otherwise the box cache (pgw#1526).

    Resolved HERE and passed to the detached child as an absolute path, not
    re-defaulted there: `up -d` re-execs with a different cwd, so a relative
    or re-derived answer would put the child's artifacts somewhere the parent
    never looks — a build under one address and a lookup under another, which
    reads as an endless cache miss rather than as an error.
    """
    from . import workspace

    stated = str(getattr(args, "artifacts_dir", "") or "").strip()
    return Path(stated).resolve() if stated else workspace.artifacts_root()


def _spec(args: argparse.Namespace) -> BootSpec:
    from . import workspace
    from .download import runtime_config_refs

    endpoint_dir = Path(args.endpoint_dir).resolve()
    refs = tuple(args.checkpoint_ref) or (
        () if args.checkpoint else runtime_config_refs(endpoint_dir)
    )
    sm = args.sm or workspace.host_sm()
    graph_store: Optional[Path] = None
    if args.graph_store:
        # STATED explicitly. If the sm cannot be resolved, adoption is
        # impossible and refusing is right — the user asked for a store.
        graph_store = Path(args.graph_store)
    else:
        default = workspace.graph_cas_root()
        # DEFAULTED. Two defaults have to agree, and they did not: the store
        # defaulted ON whenever the box CAS existed while the sm defaulted to
        # empty, so `_adoption_source` refused and a bare `gen-worker up`
        # could not bring ANY endpoint up on a machine that had ever compiled
        # anything (pgw#1523, measured on the echo example: "--sm is required
        # to adopt"). A default that makes the common invocation refuse is not
        # a default. With no sm there is nothing to adopt FOR, so the eager
        # bridge is the honest answer and the log says so.
        if default.is_dir() and sm:
            graph_store = default
        elif default.is_dir():
            print(
                "gen-worker up: a compiled-graph store exists at "
                f"{default} but this host's sm could not be determined, so "
                "nothing can be adopted for it — serving EAGER. Pass --sm to "
                "adopt.",
                file=sys.stderr,
            )
    return BootSpec(
        endpoint_dir=endpoint_dir,
        checkpoint_refs=refs,
        checkpoint_dir=Path(args.checkpoint) if args.checkpoint else None,
        checkpoint_ref_label=refs[0] if refs else "local/checkpoint",
        model=args.model,
        defaults=args.defaults,
        lane=args.lane,
        sm=sm,
        graph_store=graph_store,
        artifacts_dir=_artifacts_dir(args),
        output_dir=Path(args.output_dir),
        compile_policy=args.compile_policy,
        idle_timeout_s=float(args.idle_timeout or 0.0),
        env_lockfile=Path(args.env_lockfile) if args.env_lockfile else None,
    )


def run_up(args: argparse.Namespace) -> int:
    endpoint_dir = Path(args.endpoint_dir).resolve()
    handle = endpoint_state.handle_for(endpoint_dir)

    if not args.detached_child:
        try:
            live = endpoint_state.read_handle(handle)
        except EndpointStateError as exc:
            sys.stderr.write(f"gen-worker up: {exc}\n")
            return 1
        if live is not None:
            sys.stderr.write(
                f"gen-worker up: {live.get('module') or endpoint_dir} is already "
                f"up (pid {live.get('pid')}, socket {live.get('socket')}).\n"
                f"  `gen-worker run ...` talks to it; `gen-worker down` stops "
                f"it.\n"
            )
            return 1

    if args.detach and not args.detached_child:
        return _detach(args, handle)

    try:
        spec = _spec(args)
    except Exception as exc:  # noqa: BLE001 — arg assembly refusals are usage
        sys.stderr.write(f"gen-worker up: {exc}\n")
        return 2
    try:
        return serve(spec, handle)
    except BootError as exc:
        sys.stderr.write(f"gen-worker up: {exc}\n")
        endpoint_state.clear_handle(handle)
        return 1
    except KeyboardInterrupt:
        endpoint_state.clear_handle(handle)
        return 130


def _detach(args: argparse.Namespace, handle: Any) -> int:
    """Re-exec this same command as a detached child and wait for READY.

    A re-exec rather than a fork: the child must not inherit a half-initialized
    CUDA context or an argparse namespace this process mutated, and re-running
    the identical argv means the detached endpoint is booted by exactly the
    code path the foreground one is — not a second boot function that could
    drift from it.
    """
    handle.state_dir.mkdir(parents=True, exist_ok=True)
    argv = _child_argv(args)
    log = handle.log_path.open("ab", buffering=0)
    try:
        child = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=os.getcwd(),
        )
    finally:
        log.close()
    sys.stderr.write(
        f"gen-worker up: booting {Path(args.endpoint_dir).resolve()} "
        f"detached (pid {child.pid}); logs: {handle.log_path}\n"
    )
    sys.stderr.flush()
    try:
        # `child.poll` REAPS; a bare pid check would see the zombie of a child
        # that already refused and wait forever (pgw#1523).
        document = endpoint_state.wait_for_handle(
            handle, still_running=lambda: child.poll() is None
        )
    except EndpointStateError as exc:
        sys.stderr.write(f"gen-worker up: {exc}\n")
        return 1
    sys.stderr.write(
        f"gen-worker up: {document.get('module')} ready — "
        f"functions: {', '.join(document.get('functions') or ())}\n"
        f"gen-worker up: `gen-worker run \"<prompt>\"` to send a request, "
        f"`gen-worker down` to stop.\n"
    )
    return 0


def _child_argv(args: argparse.Namespace) -> List[str]:
    """This invocation, minus ``-d``, plus the detached-child marker."""
    argv = [sys.executable, "-m", "gen_worker.cli", "up",
            str(Path(args.endpoint_dir).resolve()), DETACHED_CHILD_FLAG]
    for ref in args.checkpoint_ref:
        argv += ["--checkpoint-ref", ref]
    for flag, value in (
        ("--checkpoint", args.checkpoint),
        ("--model", args.model),
        ("--lane", args.lane),
        ("--sm", args.sm),
        ("--graph-store", args.graph_store),
        ("--env-lockfile", args.env_lockfile),
    ):
        if value:
            argv += [flag, str(value)]
    argv += [
        "--defaults", args.defaults,
        "--artifacts-dir", str(_artifacts_dir(args)),
        "--output-dir", str(Path(args.output_dir).resolve()),
        "--compile", args.compile_policy,
        "--idle-timeout", str(args.idle_timeout),
    ]
    return argv


# --------------------------------------------------------------------------
# down
# --------------------------------------------------------------------------


def run_down(args: argparse.Namespace) -> int:
    endpoint_dir = Path(args.endpoint_dir).resolve()
    handle = endpoint_state.handle_for(endpoint_dir)
    try:
        document = endpoint_state.read_handle(handle)
    except EndpointStateError as exc:
        sys.stderr.write(f"gen-worker down: {exc}\n")
        return 1
    if document is None:
        sys.stderr.write(
            f"gen-worker down: nothing is up for {endpoint_dir}\n"
        )
        return 0
    pid = int(document.get("pid", 0) or 0)
    sys.stderr.write(
        f"gen-worker down: stopping {document.get('module') or endpoint_dir} "
        f"(pid {pid}) — waiting for teardown\n"
    )
    sys.stderr.flush()
    endpoint_state.terminate(pid)
    endpoint_state.clear_handle(handle)
    sys.stderr.write("gen-worker down: stopped\n")
    return 0


__all__ = ["DETACHED_CHILD_FLAG", "add_subparser", "run_down", "run_up"]
