"""``gen-worker families`` — the pgw#520 family-vocabulary CLI surface.

``export-schemas <dir>`` writes one ``<family>.schema.json`` per registered
:class:`~gen_worker.families.FamilyDefaults` subclass — the JSON Schema
(draft 2020-12, ``additionalProperties: false``) tensorhub validates repo
inference-defaults metadata against at PUT time (th#767c).

Only the SHIPPED families (``gen_worker.families``) are registered by
default; an endpoint project with its own third-party families should
``--module`` its main package first so those subclasses import (and
therefore register) before export.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    p = sub.add_parser(
        "families",
        help="Inspect / export the per-family inference-defaults vocabulary.",
        description=(
            "Operate on the registered gen_worker.families.FamilyDefaults "
            "vocabulary (pgw#520): each family's JSON Schema is the contract "
            "tensorhub validates repo inference-defaults metadata against."
        ),
    )
    fam_sub = p.add_subparsers(dest="families_command", metavar="<command>")
    fam_sub.required = True

    export = fam_sub.add_parser(
        "export-schemas",
        help="Write <family>.schema.json for every registered family into a directory.",
    )
    export.add_argument("out_dir", help="Directory to write <family>.schema.json files into.")
    export.add_argument(
        "--module", dest="module", default=None,
        help=(
            "Import this module first so its FamilyDefaults subclasses "
            "register (e.g. an endpoint project's main package). Ships-only "
            "families export without this flag."
        ),
    )
    export.set_defaults(_handler=_handle_export_schemas)


def _handle_export_schemas(args: argparse.Namespace) -> int:
    if args.module:
        try:
            importlib.import_module(args.module)
        except Exception as e:
            sys.stderr.write(f"gen-worker families export-schemas: failed to import {args.module!r}: {e}\n")
            return 2

    from ..families import export_all_schemas

    schemas = export_all_schemas()
    if not schemas:
        sys.stderr.write("gen-worker families export-schemas: no families registered\n")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, schema in schemas.items():
        dest = out_dir / f"{name}.schema.json"
        dest.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        sys.stderr.write(f"wrote {dest}\n")
    return 0
