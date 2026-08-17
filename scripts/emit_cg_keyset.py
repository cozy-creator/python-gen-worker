#!/usr/bin/env python3
"""pgw#1327 — emit a ``cg-keyset-v1`` document. MINT-LANE TOOLING.

A serve pod reads its ``cg-key-v1`` set as data. This is how that data is
produced and staged for baking beside ``endpoint.lock``.

Two modes, because there are two honest producers:

``--from-cache DIR`` (default)
    Take the closures a pod ALREADY derived — its own
    ``DIR/cg-keyset-v1.json``, written by ``boot_key.derive`` as a side effect
    of the mint that produced the cells (§4.28) — validate them, and merge them
    into ``--out``. Imports no endpoint code, needs no torch, and is the
    realistic operator flow: mint on a pod, copy one small JSON out, bake it
    into the next image.

``--derive --module M --function F``
    Run the derivation here: structure-only ``torch.export`` traces on fake
    tensors, in child processes. No compile, no ``.so``, no publish — which is
    why it is allowed on a laptop under the standing "mints run on remote
    machines only" rule. Needs the endpoint importable and its slots resolved,
    because a traced graph is a function of the checkpoint's own config.

The document is the SAME schema in both modes and in the pod's own cache: it is
trusted because its closure digest matches what a pod's code would trace, never
because of where it was found.

Where the output goes on a pod: ``/app/.tensorhub/cg-keyset-v1.json``, beside
``endpoint.lock`` (``gen_worker.keyset.store.IMAGE_KEYSET_DIR``), or anywhere
``GEN_WORKER_CG_KEYSET`` points.

Usage::

    python scripts/emit_cg_keyset.py --from-cache ~/.cache/cozy --out build/
    python scripts/emit_cg_keyset.py --derive --module my_endpoint.main \\
        --function generate --slot pipeline=/cas/checkpoint --out build/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _merge(out_path: Path, closures: Dict[str, object]) -> int:
    from gen_worker import keyset
    from gen_worker.keyset import store as keyset_store

    try:
        document = keyset_store.read_document(out_path)
    except keyset.KeySetError:
        document = keyset.empty()
    merged = dict(document.closures)
    merged.update(closures)  # type: ignore[arg-type]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(keyset.encode(keyset.KeySetDocument(
        schema=keyset.KEYSET_SCHEMA, version=keyset.KEYSET_VERSION,
        closures=merged)))  # type: ignore[arg-type]
    return len(merged)


def from_cache(cache: Path, out: Path) -> int:
    from gen_worker import keyset

    path = cache / keyset.KEYSET_FILENAME if cache.is_dir() else cache
    document = keyset.decode(path.read_bytes())
    if not document.closures:
        print(f"{path}: no closures — nothing this pod derived is stageable")
        return 1
    for digest in sorted(document.closures):
        closure = keyset.parse_closure(
            document, keyset.parse_closure_digest(digest))
        print(
            f"  {digest}  family={closure.family} fn={closure.function} "
            f"classes={len(closure.classes)} emitted_by={closure.emitted_by!r}")
    total = _merge(out, dict(document.closures))
    print(f"wrote {out} ({total} closure(s))")
    return 0


def derive(
    modules: Tuple[str, ...], function: str, slots: Dict[str, str],
    out: Path, work: Path,
) -> int:
    from gen_worker import aot_declaration, boot_key, keyset
    from gen_worker.api.export_contract import export_declaration
    from gen_worker.child_contract import CompileSpec, MintSlot
    from gen_worker.keyset import store as keyset_store
    from gen_worker.registry import collect_endpoints

    specs = [s for s in collect_endpoints(list(modules)) if s.name == function]
    if not specs:
        print(f"no endpoint named {function!r} in {list(modules)!r}")
        return 1
    cfg = specs[0].compile_cell()
    if cfg is None:
        print(f"{function!r} declares no compile cell, so it has no key set")
        return 1
    family = str(getattr(cfg, "family", "") or "")
    declaration = export_declaration(family)
    if declaration is None:
        print(f"family {family!r} has no registered export declaration")
        return 1
    spec = CompileSpec(
        shapes=tuple(
            tuple(int(v) for v in row) for row in (getattr(cfg, "shapes", ()) or ())),
        targets=tuple(str(t) for t in (getattr(cfg, "targets", ()) or ())),
        family=family,
        lora_bucket=int(getattr(cfg, "lora_bucket", 0) or 0),
        guidance_scales=tuple(
            float(v) for v in (getattr(cfg, "guidance_scales", ()) or ())),
        text_lens=tuple(int(v) for v in (getattr(cfg, "text_lens", ()) or ())),
    )
    resolved = {name: MintSlot(ref=None, path=path) for name, path in slots.items()}
    cache = work / "cache"
    derived = boot_key.derive(
        function=function,
        modules=modules,
        family=family,
        cfg=spec,
        slots=resolved,
        declared_hint=len(list(aot_declaration.cell_plans(declaration))),
        work_root=work / "trace",
        memo_dir=cache,
        trust_memo=False,
        emitted_by=f"scripts/emit_cg_keyset.py {family}",
    )
    print(
        f"derived {len(derived.entry_keys)} class(es) for {family} in "
        f"{derived.wall_ms} ms (source={derived.source.value}, "
        f"closure={derived.closure})")
    document = keyset_store.read_document(cache / keyset.KEYSET_FILENAME)
    row = document.closures.get(str(derived.closure))
    if row is None:
        print("the derivation recorded no closure — nothing to emit")
        return 1
    _merge(out, {str(derived.closure): row})
    print(f"wrote {out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-cache", default="",
        help="a pod's cache dir (or a cg-keyset-v1.json) to stage")
    ap.add_argument(
        "--derive", action="store_true",
        help="run the structure-only derivation here instead")
    ap.add_argument("--module", action="append", default=[])
    ap.add_argument("--function", default="")
    ap.add_argument(
        "--slot", action="append", default=[], metavar="NAME=PATH",
        help="a resolved setup slot; the traced graph depends on its config")
    ap.add_argument("--out", default="")
    ap.add_argument("--work", default="")
    args = ap.parse_args(argv)

    from gen_worker import keyset

    out = Path(args.out or Path.cwd())
    # A directory (existing or not) means "put the canonical name inside it".
    # Only a path that already NAMES the document is used verbatim, so
    # `--out build/` cannot silently produce a file called `build`.
    if out.name != keyset.KEYSET_FILENAME:
        out = out / keyset.KEYSET_FILENAME

    if args.derive:
        if not args.module or not args.function:
            ap.error("--derive needs --module and --function")
        slots: Dict[str, str] = {}
        for item in args.slot:
            name, _, path = str(item).partition("=")
            if not name or not path:
                ap.error(f"--slot must be NAME=PATH, got {item!r}")
            slots[name] = path
        work = Path(args.work or (out.parent / ".cg-keyset-work"))
        return derive(tuple(args.module), args.function, slots, out, work)

    if not args.from_cache:
        ap.error("pass --from-cache DIR or --derive")
    return from_cache(Path(args.from_cache), out)


if __name__ == "__main__":
    sys.exit(main())
