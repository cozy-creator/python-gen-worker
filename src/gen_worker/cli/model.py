"""``gen-worker model`` — export a declaration, generate bindings, mint classes.

Three commands, deliberately separate, because they have different requirements
and different cadences:

``export``    runs ``torch.export`` over a model DECLARATION with fake tensors
              and writes ``<family>.export.json``. Needs torch and the model
              library; needs no GPU, no weights and no network. Run it when the
              declaration changes. An EAGER declaration has no graph classes to
              trace, so the same command projects it into
              ``<family>.eager.json`` and needs no torch at all (pgw#1346 B5).
``generate``  turns that document into a typed binding module. Pure, fast, and
              needs NOTHING but the document — no torch, no diffusers. Run it in
              CI to prove the committed binding is what the export implies.

``mint``      compiles the declaration's graph classes into packed AOTI
              artifacts (pgw#1331). Needs a GPU and a real toolchain; needs no
              weights and no network, because compiled graph identity is checkpoint-free
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

from ..model.codegen import render_eager_module, render_module
from ..model.errors import ModelError
from ..model.snapshot import EagerExport, ModelExport
from ..model.spec import GraphModelSpec, ModelSpec


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    parser = sub.add_parser(
        "model",
        help="Export a model declaration and generate its typed bindings.",
        description=(
            "The typed Model SDK's build steps (pgw#1332). A declaration is "
            "exported with fake tensors into family_export_v1; the bindings "
            "generate from that document and from nothing else."
        ),
    )
    commands = parser.add_subparsers(dest="model_command", metavar="<command>")
    commands.required = True

    export = commands.add_parser(
        "export",
        help=(
            "Export a declaration: a GraphModelSpec is fake-tensor traced into "
            "<family>.export.json; an eager-only ModelSpec is projected into "
            "<family>.eager.json with no torch at all."
        ),
    )
    export.add_argument(
        "target",
        help="Dotted path to the declaration, e.g. gen_worker.model.catalog.sdxl:SDXL",
    )
    export.add_argument(
        "--out-dir",
        default="",
        help="Directory to write into. Defaults beside the declaration's package.",
    )
    export.set_defaults(_handler=_handle_export)

    generate = commands.add_parser(
        "generate",
        help=(
            "Render the typed binding module for a committed <family>.export.json "
            "or <family>.eager.json."
        ),
    )
    generate.add_argument(
        "document", help="Path to a <family>.export.json or <family>.eager.json."
    )
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
            "e.g. gen_worker.model.catalog.sdxl:SDXL. Omit for a binding that "
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
        help="Compile a GraphModelSpec declaration's graph classes into AOTI artifacts.",
        description=(
            "pgw#1331. Traces every declared (runner, bucket, layout) with the "
            "declaration's OWN tracer — so the minted ingress digest is the "
            "committed export's by construction — and compiles each through the "
            "worker's TCG engine. This is a real compile: run it on a pod."
        ),
    )
    mint.add_argument(
        "target",
        help="Dotted path to the declaration, e.g. gen_worker.model.catalog.flux1_dev:FLUX1_DEV",
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
        "--bucket",
        action="append",
        default=[],
        metavar="AXIS=VALUE",
        help=(
            "Mint only this bucket-axis value; repeatable, and repeatable on one "
            "axis to name several. Omit an axis to mint all of its values. "
            "A gauntlet row proving ONE shape on ONE card wants one value: sdxl's "
            "shape axis carries nine, so omitting this is a nine-fold compile."
        ),
    )
    mint.add_argument(
        "--json", default="", help="Write the minted rows to this path as JSON."
    )
    mint.set_defaults(_handler=_handle_mint)


def _parse_buckets(rows: list[str]) -> dict[str, list[int]]:
    """``["shape=10241024"]`` -> ``{"shape": [10241024]}``. Refuses anything else."""

    parsed: dict[str, list[int]] = {}
    for row in rows:
        axis, sep, value = str(row).partition("=")
        if not sep or not axis.strip():
            raise ValueError(f"--bucket {row!r} must be AXIS=VALUE, e.g. shape=10241024")
        try:
            parsed.setdefault(axis.strip(), []).append(int(value))
        except ValueError:
            raise ValueError(f"--bucket {row!r} value must be an integer") from None
    return parsed


def _load(target: str) -> tuple[str, str, Any]:
    module_name, _, attr = str(target).partition(":")
    if not module_name or not attr:
        raise ValueError(
            f"{target!r} must be <module>:<attribute>, e.g. "
            "gen_worker.model.catalog.sdxl:SDXL"
        )
    module = importlib.import_module(module_name)
    return module_name, attr, getattr(module, attr)


def _handle_export(args: argparse.Namespace) -> int:
    try:
        module_name, attr, family = _load(args.target)
    except (ImportError, AttributeError, ValueError) as exc:
        sys.stderr.write(f"gen-worker model export: cannot load {args.target!r}: {exc}\n")
        return 2
    if not isinstance(family, ModelSpec):
        sys.stderr.write(
            f"gen-worker model export: {args.target!r} is a {type(family).__name__}, not a "
            "model declaration\n"
        )
        return 2
    # An EAGER declaration takes the torch-free path: it has no graph classes,
    # so there is nothing to trace and `eager_model_v1` is what it can honestly
    # state (pgw#1346 B5).
    document: Any
    if isinstance(family, GraphModelSpec):
        from ..model.export import export_model

        try:
            document = export_model(family)
        except ModelError as exc:
            sys.stderr.write(f"gen-worker model export: {exc}\n")
            return 1
        suffix = "export.json"
    else:
        from ..model.export import export_eager_model

        try:
            document = export_eager_model(family)
        except ModelError as exc:
            sys.stderr.write(f"gen-worker model export: {exc}\n")
            return 1
        suffix = "eager.json"
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(importlib.import_module(module_name).__file__ or ".").parent / "_generated"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{document.family}.{suffix}"
    destination.write_text(document.dumps())
    sys.stderr.write(f"wrote {destination} ({document.digest()}); declared by {attr}\n")
    return 0


def _handle_generate(args: argparse.Namespace) -> int:
    document = Path(args.document)
    # The document's own name says which tier it is, so nothing has to be
    # passed twice and a mismatch is impossible rather than merely unlikely.
    eager = document.name.endswith(".eager.json")
    reader = EagerExport.loads if eager else ModelExport.loads
    try:
        export: Any = reader(document.read_bytes())
    except (OSError, ModelError) as exc:
        sys.stderr.write(f"gen-worker model generate: cannot read {document}: {exc}\n")
        return 2
    spec_module, _, spec_attr = str(args.spec).partition(":")
    render = render_eager_module if eager else render_module
    rendered = render(export, spec_module=spec_module, spec_attr=spec_attr)
    destination = Path(args.out) if args.out else document.parent / f"{export.family}.py"
    if args.check:
        current = destination.read_text() if destination.is_file() else ""
        if current == rendered:
            sys.stderr.write(f"{destination} is current ({export.digest()})\n")
            return 0
        sys.stderr.write(
            f"gen-worker model generate --check: {destination} is NOT what "
            f"{document.name} implies. Regenerate it:\n"
            f"    gen-worker model generate {document}"
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
        sys.stderr.write(f"gen-worker model mint: cannot load {args.target!r}: {exc}\n")
        return 2
    if not isinstance(family, GraphModelSpec):
        sys.stderr.write(
            f"gen-worker model mint: {args.target!r} is a {type(family).__name__}, not a "
            "GraphModelSpec; an eager-only ModelSpec has no graph classes to mint\n"
        )
        return 2
    from ..model.mint import mint_model

    out_dir = Path(args.out_dir)
    work = Path(args.work) if args.work else out_dir / "work"

    def beat(name: str, done: int, total: int) -> None:
        sys.stderr.write(f"[{done + 1}/{total}] minting {name}\n")
        sys.stderr.flush()

    try:
        buckets = _parse_buckets(list(args.bucket))
    except ValueError as exc:
        sys.stderr.write(f"gen-worker model mint: {exc}\n")
        return 2
    try:
        minted = mint_model(
            family,
            out_dir=out_dir,
            work=work,
            cache_root=Path(args.cache_root) if args.cache_root else None,
            only=tuple(args.runner),
            buckets=buckets,
            on_class=beat,
        )
    except ModelError as exc:
        sys.stderr.write(f"gen-worker model mint: {exc}\n")
        return 1
    rows = [
        {
            "runner": row.runner,
            "bucket": {name: value for name, value in row.bucket},
            "layout": row.layout,
            # This report's OWN column name for a MintedVariant field, not a
            # read of TCG's metadata block: the value is a handle the mint
            # bridge assigned, and TCG's block key is read once, by name, in
            # aot_compile_child.
            "graph_class": row.graph_class,  # tcg-vocab: this report's column
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
        name = row["graph_class"]  # tcg-vocab: this report's column, see above
        sys.stderr.write(f"{name}  {row['key']}  {cost}\n")
    sys.stderr.write(f"minted {len(rows)} graph class(es) from {attr}\n")
    return 0
