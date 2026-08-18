"""``gen-worker release`` -- publish-time commands (pgw#1370).

``release derive`` runs the instrumented derive INSIDE the release env:
weights-free (config-only checkpoint tree), CPU-only, byte-reproducible.
The document goes to ``--out`` (or stdout); the summary -- per-lane graph
counts and the document digest -- goes to stderr, so a pipeline can capture
the bytes and a human can read the log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def add_subparser(sub: Any) -> None:
    parser = sub.add_parser(
        "release",
        help="publish-time release commands (derive graph metadata)",
        description=__doc__,
    )
    release_sub = parser.add_subparsers(dest="release_command", metavar="<command>")
    release_sub.required = True

    derive = release_sub.add_parser(
        "derive",
        help="run the instrumented derive; emit the release metadata document",
    )
    derive.add_argument(
        "--dir",
        default=".",
        help="endpoint project root (primed onto sys.path; default: cwd)",
    )
    derive.add_argument(
        "--module",
        required=True,
        help="endpoint main module to import, e.g. 'sdxl.main_v2'",
    )
    derive.add_argument(
        "--checkpoint",
        required=True,
        help="CONFIG-ONLY checkpoint tree the author's load path resolves "
        "(subset snapshot; weights never transit the derive)",
    )
    derive.add_argument(
        "--lockfile",
        default=None,
        help="uv.lock to hash as the env-identity closure (default: the "
        "installed distribution set of THIS env)",
    )
    derive.add_argument(
        "--out",
        default=None,
        help="write the document bytes here (default: stdout)",
    )
    derive.set_defaults(_handler=_run_derive)


def _run_derive(args: argparse.Namespace) -> int:
    import importlib

    from ..release.derive import DeriveError, derive_release
    from .run import _ensure_sys_path

    root = Path(args.dir).resolve()
    if not root.is_dir():
        print(f"error: --dir {root} is not a directory", file=sys.stderr)
        return 2
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_dir():
        print(f"error: --checkpoint {checkpoint} is not a directory", file=sys.stderr)
        return 2
    _ensure_sys_path(root)
    try:
        module = importlib.import_module(args.module)
    except Exception as exc:  # noqa: BLE001 - author import errors are the check
        print(f"error: failed to import {args.module!r}: {exc}", file=sys.stderr)
        return 1

    try:
        result = derive_release(
            module,
            checkpoint_dir=checkpoint,
            lockfile=Path(args.lockfile).resolve() if args.lockfile else None,
        )
    except DeriveError as exc:
        print(f"derive error: {exc}", file=sys.stderr)
        return 1

    if args.out:
        Path(args.out).write_bytes(result.document)
    else:
        sys.stdout.buffer.write(result.document + b"\n")

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if result.eager_permanent:
        print(
            f"{result.endpoint}: no lanes -- explicit eager-permanent document",
            file=sys.stderr,
        )
    for lane_name, hashes in result.lane_graphs.items():
        print(f"lane {lane_name}: {len(hashes)} graph class(es)", file=sys.stderr)
        for graph in hashes:
            print(f"  {graph}", file=sys.stderr)
    print(f"document sha256: {result.digest}", file=sys.stderr)
    return 0


__all__ = ["add_subparser"]
