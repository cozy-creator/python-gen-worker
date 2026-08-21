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
        action="append",
        default=[],
        metavar="[SLOT=]PATH",
        help="CONFIG-ONLY checkpoint tree the author's load path resolves "
        "(subset snapshot; weights never transit the derive). Repeat as "
        "`slot=path` to give a SECONDARY model slot its own tree (pgw#1508); "
        "the bare form is the primary model's.",
    )
    # pgw#1489: a derive STATES the compile stack it traced under, and reads it
    # from the endpoint's own `uv.lock` — the one file the derive, the mint and
    # every serving pod share by construction. This is not a flag that switches
    # the source of an identity (the defect pgw#1472 named): there IS no other
    # source any more, and a derive without a lock refuses by name.
    derive.add_argument(
        "--lockfile",
        default=None,
        help="the endpoint's uv.lock, whose torch/triton/nvidia-* rows are "
        "this document's compile stack — the env half of every artifact key "
        "(default: the uv.lock beside --dir)",
    )
    derive.add_argument(
        "--graph-cas",
        default=None,
        help="tensorfs CAS root to store each discovered graph's SERIALIZED "
        "ExportedProgram in (the miner downloads the graph and runs inductor; "
        "it never re-traces). The digests travel in the document.",
    )
    derive.add_argument(
        "--out",
        default=None,
        help="write the document bytes here (default: stdout)",
    )
    # pgw#1599: `--dynamic-axes` is DELETED. Which axis is worth collapsing is
    # a MEASURED, per-model question (pgw#1548), so it is declared on the model
    # class that measured it (`shapes={"aspect": DYNAMIC}`) — a CLI flag
    # re-keyed every graph in the fleet from one word and left no record on the
    # model of what was chosen or why.
    derive.set_defaults(_handler=_run_derive)


def _run_derive(args: argparse.Namespace) -> int:
    import importlib

    from ..discovery.discover import prime_sys_path
    from ..release.derive import DeriveError, derive_release

    root = Path(args.dir).resolve()
    if not root.is_dir():
        print(f"error: --dir {root} is not a directory", file=sys.stderr)
        return 2
    trees: dict[str, Path] = {}
    for raw in args.checkpoint:
        slot, _sep, rest = raw.partition("=")
        if not (_sep and slot.isidentifier()):
            slot, rest = "", raw
        if slot in trees:
            owner = f"slot {slot!r}" if slot else "the primary model"
            print(f"error: --checkpoint given twice for {owner}", file=sys.stderr)
            return 2
        resolved = Path(rest).resolve()
        if not resolved.is_dir():
            print(f"error: --checkpoint {resolved} is not a directory", file=sys.stderr)
            return 2
        trees[slot] = resolved
    if "" not in trees:
        print(
            "error: --checkpoint names only secondary slots; the primary "
            "model's tree is the bare form",
            file=sys.stderr,
        )
        return 2
    checkpoint = trees.pop("")
    prime_sys_path(root)
    try:
        module = importlib.import_module(args.module)
    except Exception as exc:  # noqa: BLE001 - author import errors are the check
        print(f"error: failed to import {args.module!r}: {exc}", file=sys.stderr)
        return 1

    try:
        from ..env_identity import lockfile_beside

        lockfile = (
            Path(args.lockfile).resolve() if args.lockfile else lockfile_beside(root)
        )
        result = derive_release(
            module,
            checkpoint_dir=checkpoint,
            lockfile=lockfile,
            graph_cas=Path(args.graph_cas).resolve() if args.graph_cas else None,
            slot_checkpoints=trees,
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
    # pgw#1449: named, and named SEPARATELY from the warnings — an entrypoint
    # with no traced coverage is a property of the document a reader has to
    # be able to see without parsing prose.
    for name, reason in result.unenumerable_entrypoints:
        print(f"entrypoint {name}: NOT enumerated -- {reason}", file=sys.stderr)
    # pgw#1527: named separately from the warnings for the same reason — a
    # payload with no graphs is a property of the document, not prose.
    for skipped in result.unservable_payloads:
        print(f"payload SKIPPED (unservable): {skipped}", file=sys.stderr)
    if result.eager_permanent:
        # pgw#1392: two different reasons reach "no graphs" and the log must
        # not conflate them — a model that MARKS nothing (it declares real
        # lanes like every model does; the absent `ctx.compile` mark is the
        # whole statement), or NO MODEL AT ALL (a weightless endpoint).
        why = (
            "no model class -- weightless endpoint, nothing to compile"
            if result.weightless
            else "no graphs -- load() marks no ctx.compile target"
        )
        print(f"{result.endpoint}: {why}", file=sys.stderr)
    for lane_name, hashes in result.lane_graphs.items():
        print(f"lane {lane_name}: {len(hashes)} graph specialization(es)", file=sys.stderr)
        for graph in hashes:
            print(f"  {graph}", file=sys.stderr)
    print(f"document sha256: {result.digest}", file=sys.stderr)
    return 0


__all__ = ["add_subparser"]
