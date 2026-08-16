"""``gen-worker job run <name> --payload file.json`` — the author dev loop.

The hub-less half of :mod:`gen_worker.jobs`: same registry walk, same schema
decode, same ``execute_job`` harness, same :class:`JobContext` — only the
hub-backed accessors are stubbed (``LocalJobContext``). So an author's local
run exercises the body the pod will run, including the ``publishes=True``
refusal, rather than a lookalike.

**GPU jobs are REFUSED on a control box**, by declaration and before anything
loads: ``Resources(gpu=True)`` means this job needs a card, and this repo's
standing rule is that no real-model inference runs here. ``--allow-gpu`` exists
for a machine that legitimately has one; it is not a way to get a GPU job to
start on a laptop. CPU-shaped ``plan-*`` jobs are the ones this loop is for.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import msgspec

from ..jobs import DEFAULT_PHASE_BUDGET_S, JobDispatch, execute_job
from ..registry import JobSpec, collect_jobs
from .local_context import _stderr_emitter, build_local_context

EXIT_OK = 0
EXIT_USER_EXCEPTION = 1
EXIT_USAGE = 2
EXIT_SIGINT = 130


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    p = sub.add_parser(
        "job",
        help="list or run this package's @job functions locally",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    job_sub = p.add_subparsers(dest="job_command", metavar="<subcommand>")
    job_sub.required = False

    run_p = job_sub.add_parser("run", help="run ONE job against a payload file")
    run_p.add_argument("name", help="job name (as published in the jobs manifest)")
    run_p.add_argument(
        "--payload", metavar="FILE",
        help="path to a JSON payload file ('-' reads stdin)",
    )
    run_p.add_argument(
        "--config", dest="config_path", default=None,
        help="path to pyproject.toml (default: discovered from cwd)",
    )
    run_p.add_argument(
        "--phase-budget-s", type=float, default=DEFAULT_PHASE_BUDGET_S,
        help=(
            "how long a phase may report the same position before the run is "
            "judged stalled (default: %(default)s)"
        ),
    )
    run_p.add_argument(
        "--allow-publish", action="store_true",
        help="let the publisher surface reach a real tensorhub",
    )
    run_p.add_argument(
        "--allow-gpu", action="store_true",
        help="permit a job declaring Resources(gpu=True) to run on this host",
    )
    run_p.set_defaults(_handler=_handle_job_run)

    list_p = job_sub.add_parser("list", help="list this package's jobs")
    list_p.add_argument(
        "--config", dest="config_path", default=None,
        help="path to pyproject.toml (default: discovered from cwd)",
    )
    list_p.set_defaults(_handler=_handle_job_list)

    p.set_defaults(_handler=_handle_job_run)


def _collect(config_path: Optional[str]) -> List[JobSpec]:
    # Reuse `run`'s project resolution + sys.path priming verbatim: two
    # spellings of "where does this package live" is how a CLI and a build
    # start disagreeing about what a release contains.
    from .run import _ensure_sys_path, _load_project_main

    root, main_module = _load_project_main(config_path)
    _ensure_sys_path(root)
    return collect_jobs([main_module.split(".", 1)[0]])


def _read_payload(path: Optional[str]) -> bytes:
    if not path:
        return b"{}"
    if path == "-":
        return sys.stdin.buffer.read()
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        print(f"error: --payload not found: {p}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)
    return p.read_bytes()


def _handle_job_list(args: argparse.Namespace) -> int:
    specs = _collect(getattr(args, "config_path", None))
    rows = [
        {
            "name": s.name,
            "python_name": s.python_name,
            "module": s.module,
            "resources": msgspec.to_builtins(s.resources),
            "env": list(s.env),
            "resumable": s.resumable,
            "visibility": s.visibility,
            "publishes": s.publishes,
        }
        for s in sorted(specs, key=lambda s: s.name)
    ]
    print(json.dumps(rows, indent=2, sort_keys=True))
    return EXIT_OK


def _handle_job_run(args: argparse.Namespace) -> int:
    name = getattr(args, "name", None)
    if not name:
        print(
            "usage: gen-worker job run <name> --payload file.json",
            file=sys.stderr,
        )
        return EXIT_USAGE

    specs = {s.name: s for s in _collect(getattr(args, "config_path", None))}
    spec = specs.get(str(name))
    if spec is None:
        print(
            f"error: unknown job {name!r} (this package declares: "
            f"{sorted(specs) or 'none'})",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if spec.needs_gpu and not getattr(args, "allow_gpu", False):
        print(
            f"error: job {spec.name!r} declares Resources(gpu=True) and this "
            "is a control box — GPU jobs run on a pod, never here. Run a "
            "CPU-shaped plan-* job locally, or pass --allow-gpu on a host "
            "that genuinely has a card to spare.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    raw = _read_payload(getattr(args, "payload", None))
    try:
        payload = msgspec.msgpack.encode(msgspec.json.decode(raw))
    except msgspec.DecodeError as exc:
        print(f"error: payload is not valid JSON: {exc}", file=sys.stderr)
        return EXIT_USAGE

    ctx = build_local_context(
        kind="job",
        allow_publish=bool(getattr(args, "allow_publish", False)),
        emitter=_stderr_emitter,
        publishes=spec.publishes,
    )
    dispatch = JobDispatch(
        job_name=spec.name,
        payload=payload,
        job_id=str(ctx.request_id),
        phase_budget_s=float(getattr(args, "phase_budget_s", DEFAULT_PHASE_BUDGET_S)),
    )
    try:
        outcome = execute_job(
            dispatch, jobs={spec.name: spec}, ctx=ctx, reraise=True)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_SIGINT
    except msgspec.ValidationError as exc:
        print(f"error: payload does not match the job's schema: {exc}",
              file=sys.stderr)
        return EXIT_USAGE
    except Exception:
        import traceback

        traceback.print_exc()
        return EXIT_USER_EXCEPTION

    result: Dict[str, Any] = msgspec.msgpack.decode(outcome.result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return EXIT_OK


__all__ = ["add_subparser"]
