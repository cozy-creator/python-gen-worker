"""``gen-worker family`` — export a declaration and generate its bindings.

Two commands, deliberately separate, because they have different requirements
and different cadences:

``export``    runs ``torch.export`` over a family DECLARATION with fake tensors
              and writes ``<family>.export.json``. Needs torch and the model
              library; needs no GPU, no weights and no network. Run it when the
              declaration changes.
``generate``  turns that document into a typed binding module. Pure, fast, and
              needs NOTHING but the document — no torch, no diffusers. Run it in
              CI to prove the committed binding is what the export implies.

The split is what makes the fence cheap. ``generate --check`` is a byte
comparison a two-minute job can afford, and it catches the case that actually
bites: an export regenerated and committed while the binding beside it was not.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

from ..family.codegen import render_module
from ..family.errors import FamilyError
from ..family.snapshot import FamilyExport
from ..family.spec import GraphFamily


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    parser = sub.add_parser(
        "family",
        help="Export a family declaration and generate its typed bindings.",
        description=(
            "The typed Family SDK's build steps (pgw#1332). A declaration is "
            "exported with fake tensors into family_export_v1; the bindings "
            "generate from that document and from nothing else."
        ),
    )
    commands = parser.add_subparsers(dest="family_command", metavar="<command>")
    commands.required = True

    export = commands.add_parser(
        "export",
        help="Fake-tensor export a GraphFamily declaration into <family>.export.json.",
    )
    export.add_argument(
        "target",
        help="Dotted path to the declaration, e.g. gen_worker.family.catalog.sdxl:SDXL",
    )
    export.add_argument(
        "--out-dir",
        default="",
        help="Directory to write into. Defaults beside the declaration's package.",
    )
    export.set_defaults(_handler=_handle_export)

    generate = commands.add_parser(
        "generate",
        help="Render the typed binding module for a committed <family>.export.json.",
    )
    generate.add_argument("document", help="Path to a <family>.export.json.")
    generate.add_argument(
        "--out",
        default="",
        help="Path to write. Defaults to <family>.py beside the document.",
    )
    generate.add_argument(
        "--spec",
        default="",
        help=(
            "Dotted path to the declaration the bindings should lazily import, "
            "e.g. gen_worker.family.catalog.sdxl:SDXL. Omit for a binding that "
            "carries no declaration at all."
        ),
    )
    generate.add_argument(
        "--check",
        action="store_true",
        help="Do not write: exit 1 when the file on disk is not what would be written.",
    )
    generate.set_defaults(_handler=_handle_generate)


def _load(target: str) -> tuple[str, str, Any]:
    module_name, _, attr = str(target).partition(":")
    if not module_name or not attr:
        raise ValueError(
            f"{target!r} must be <module>:<attribute>, e.g. "
            "gen_worker.family.catalog.sdxl:SDXL"
        )
    module = importlib.import_module(module_name)
    return module_name, attr, getattr(module, attr)


def _handle_export(args: argparse.Namespace) -> int:
    try:
        module_name, attr, family = _load(args.target)
    except (ImportError, AttributeError, ValueError) as exc:
        sys.stderr.write(f"gen-worker family export: cannot load {args.target!r}: {exc}\n")
        return 2
    if not isinstance(family, GraphFamily):
        sys.stderr.write(
            f"gen-worker family export: {args.target!r} is a "
            f"{type(family).__name__}, not a GraphFamily; an eager-only Family has no "
            "graph classes to export\n"
        )
        return 2
    from ..family.export import export_family

    try:
        document = export_family(family)
    except FamilyError as exc:
        sys.stderr.write(f"gen-worker family export: {exc}\n")
        return 1
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(importlib.import_module(module_name).__file__ or ".").parent / "_generated"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{document.family}.export.json"
    destination.write_text(document.dumps())
    sys.stderr.write(f"wrote {destination} ({document.digest()}); declared by {attr}\n")
    return 0


def _handle_generate(args: argparse.Namespace) -> int:
    document = Path(args.document)
    try:
        export = FamilyExport.loads(document.read_bytes())
    except (OSError, FamilyError) as exc:
        sys.stderr.write(f"gen-worker family generate: cannot read {document}: {exc}\n")
        return 2
    spec_module, _, spec_attr = str(args.spec).partition(":")
    rendered = render_module(export, spec_module=spec_module, spec_attr=spec_attr)
    destination = Path(args.out) if args.out else document.parent / f"{export.family}.py"
    if args.check:
        current = destination.read_text() if destination.is_file() else ""
        if current == rendered:
            sys.stderr.write(f"{destination} is current ({export.digest()})\n")
            return 0
        sys.stderr.write(
            f"gen-worker family generate --check: {destination} is NOT what "
            f"{document.name} implies. Regenerate it:\n"
            f"    gen-worker family generate {document}"
            + (f" --spec {args.spec}" if args.spec else "")
            + "\n"
        )
        return 1
    destination.write_text(rendered)
    sys.stderr.write(f"wrote {destination} ({export.digest()})\n")
    return 0
