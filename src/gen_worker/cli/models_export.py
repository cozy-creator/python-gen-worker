"""``gen-worker models`` — the model-typed serving-defaults surface (pgw#1377).

``export`` emits the ``{names, schemas}`` document generated from the
``gen_worker.models`` Defaults structs — the mechanical artifact the hub's
recognized-name guard (th#2140) and write-time validation (th#2141) consume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..models.defaults_export import export_document


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    p = sub.add_parser(
        "models",
        help="Model-typed serving defaults (the pgw#1376 vocabulary).",
        description=(
            "Operate on the gen_worker.models model-type vocabulary: names, "
            "Defaults structs, and their exported JSON Schemas."
        ),
    )
    models_sub = p.add_subparsers(dest="models_command", metavar="<command>")
    models_sub.required = True

    export = models_sub.add_parser(
        "export",
        help="Emit the {names, schemas} JSON document from the Defaults structs.",
    )
    export.add_argument(
        "--out",
        default="-",
        help="Destination file, or '-' for stdout (default).",
    )
    export.set_defaults(_handler=_handle_export)


def _handle_export(args: argparse.Namespace) -> int:
    text = json.dumps(export_document(), indent=2, sort_keys=True) + "\n"
    out = str(args.out or "-")
    if out == "-":
        sys.stdout.write(text)
        return 0
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    sys.stderr.write(f"wrote {dest}\n")
    return 0
