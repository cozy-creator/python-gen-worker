"""``gen-worker family`` — export a declaration, generate bindings, mint classes.

Three commands, deliberately separate, because they have different requirements
and different cadences:

``export``    runs ``torch.export`` over a family DECLARATION with fake tensors
              and writes ``<family>.export.json``. Needs torch and the model
              library; needs no GPU, no weights and no network. Run it when the
              declaration changes.
``generate``  turns that document into a typed binding module. Pure, fast, and
              needs NOTHING but the document — no torch, no diffusers. Run it in
              CI to prove the committed binding is what the export implies.

``mint``      compiles the declaration's graph classes into packed AOTI
              artifacts (pgw#1331). Needs a GPU and a real toolchain; needs no
              weights and no network, because cell identity is checkpoint-free
              (§4.27) and the constants arrive at ARM time from the store.
              **Runs on a pod, never on a shared box** — it is a real compile.

The split is what makes the fence cheap. ``generate --check`` is a byte
comparison a two-minute job can afford, and it catches the case that actually
bites: an export regenerated and committed while the binding beside it was not.
"""

from __future__ import annotations

import argparse
import importlib
import json
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

    mint = commands.add_parser(
        "mint",
        help="Compile a GraphFamily declaration's graph classes into AOTI artifacts.",
        description=(
            "pgw#1331. Traces every declared (runner, bucket, layout) with the "
            "declaration's OWN tracer — so the minted ingress digest is the "
            "committed export's by construction — and compiles each through the "
            "worker's TCG engine. This is a real compile: run it on a pod."
        ),
    )
    mint.add_argument(
        "target",
        help="Dotted path to the declaration, e.g. gen_worker.family.catalog.flux1_dev:FLUX1_DEV",
    )
    mint.add_argument("--out-dir", required=True, help="Directory to write packed artifacts into.")
    mint.add_argument("--work", default="", help="Scratch root. Defaults to <out-dir>/work.")
    mint.add_argument(
        "--cache-root",
        default="",
        help="TCG CAS root. Omit to use the worker's canonical store.",
    )
    mint.add_argument(
        "--runner",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Mint only this runner; repeatable. Omit for every declared runner. "
            "A pod normally mints the cheap classes first."
        ),
    )
    mint.add_argument(
        "--json", default="", help="Write the minted rows to this path as JSON."
    )
    mint.set_defaults(_handler=_handle_mint)


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


def _handle_mint(args: argparse.Namespace) -> int:
    try:
        _, attr, family = _load(args.target)
    except (ImportError, AttributeError, ValueError) as exc:
        sys.stderr.write(f"gen-worker family mint: cannot load {args.target!r}: {exc}\n")
        return 2
    if not isinstance(family, GraphFamily):
        sys.stderr.write(
            f"gen-worker family mint: {args.target!r} is a {type(family).__name__}, not a "
            "GraphFamily; an eager-only Family has no graph classes to mint\n"
        )
        return 2
    from ..family.mint import mint_family

    out_dir = Path(args.out_dir)
    work = Path(args.work) if args.work else out_dir / "work"

    def beat(name: str, done: int, total: int) -> None:
        sys.stderr.write(f"[{done + 1}/{total}] minting {name}\n")
        sys.stderr.flush()

    try:
        minted = mint_family(
            family,
            out_dir=out_dir,
            work=work,
            cache_root=Path(args.cache_root) if args.cache_root else None,
            only=tuple(args.runner),
            on_class=beat,
        )
    except FamilyError as exc:
        sys.stderr.write(f"gen-worker family mint: {exc}\n")
        return 1
    rows = [
        {
            "runner": row.runner,
            "bucket": {name: value for name, value in row.bucket},
            "layout": row.layout,
            "graph_class": row.graph_class,
            "key": row.key,
            "artifact": str(row.artifact),
            "ingress_digest": row.ingress_digest,
            "compile_s": round(row.compile_s, 3),
            "reuse_s": round(row.reuse_s, 3),
            "reused": row.reused,
        }
        for row in minted
    ]
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    for row in rows:
        cost = "reused" if row["reused"] else f"{row['compile_s']}s"
        sys.stderr.write(f"{row['graph_class']}  {row['key']}  {cost}\n")
    sys.stderr.write(f"minted {len(rows)} graph class(es) from {attr}\n")
    return 0
