"""Operator commands for inspecting and materializing compiled graphs."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from tensorfs import LocalCAS

from .artifact import read_metadata, unpack_artifact
from .engine import Engine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torchcg")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="print an artifact's validated metadata")
    inspect.add_argument("artifact", type=Path)

    verify = commands.add_parser("verify", help="fully verify an artifact and its AOTI package")
    verify.add_argument("artifact", type=Path)

    resolve = commands.add_parser(
        "resolve", help="unpack one exact key from the local tensorfs store"
    )
    resolve.add_argument("key")
    resolve.add_argument("destination", type=Path)
    resolve.add_argument("--cas-root", type=Path, required=True)
    return parser


def _print(metadata: dict[str, object]) -> None:
    print(json.dumps(metadata, sort_keys=True, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        _print(read_metadata(args.artifact))
        return 0
    if args.command == "verify":
        with tempfile.TemporaryDirectory(prefix="torchcg-cli-") as raw:
            _print(unpack_artifact(args.artifact, Path(raw) / "artifact"))
        return 0
    if args.command == "resolve":
        compiled_graph = Engine(LocalCAS(args.cas_root)).resolve(args.key, args.destination)
        if compiled_graph is None:
            parser.error(f"no local compiled graph for {args.key}")
        _print(compiled_graph.metadata)
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")
