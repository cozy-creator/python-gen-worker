#!/usr/bin/env python3
"""pgw#1327 — emit a ``cg-keyset-v1`` document. MINT-LANE TOOLING.

A serve pod reads its ``cg-key-v1`` set as data. This is how that data is
produced and staged for baking beside ``endpoint.lock``.

Two modes, because there are two honest producers:

``--from-cache DIR`` (default)
    Take the closures a pod ALREADY derived — its own
    ``DIR/cg-keyset-v1.json``, written by ``boot_key.derive`` as a side effect
    of the mint that produced the compiled graphs (§4.28) — validate them, and merge them
    into ``--out``. Imports no endpoint code, needs no torch, and is the
    realistic operator flow: mint on a pod, copy one small JSON out, bake it
    into the next image.

``--derive --module M --function F``
    Run the derivation here: structure-only ``torch.export`` traces on fake
    tensors, in child processes. No compile, no ``.so``, no publish — which is
    why it is allowed on a laptop under the standing "mints run on remote
    machines only" rule. Needs the endpoint importable and its slots resolved,
    because a traced graph is a function of the checkpoint's own config.

    **Every slot needs BOTH halves**, and the script refuses one without the
    other (pgw#1353): ``--slot NAME=PATH`` says where a checkpoint TREE already
    is — the structure-only build reads ``model_index.json`` and each
    component's config out of it — and ``--slot-ref NAME=REF`` says WHICH
    checkpoint it is. The ref is not bookkeeping: ``keyset.closure`` folds it
    into the closure digest, so a document derived without the ref the serving
    pod will resolve is addressed at a closure no pod ever looks up. This mode
    was unrunnable between pgw#1333 and pgw#1353 — it built
    ``MintSlot(ref=None, path=path)`` against a struct that had gained a
    required ``facts`` and required ``ref``, and raised ``TypeError`` before
    tracing anything.

WHAT THIS SCRIPT CANNOT DO, stated because it was tried (pgw#1353)
------------------------------------------------------------------
Run at IMAGE BUILD to bake a document into ``/app/.tensorhub``. The builder can
state neither half of a slot: the checkpoint tree is not in the image, and the
ref is a deploy-time hub pick (an sdxl-shaped ``Slot(selected_by=...)`` has no
``default_checkpoint``, and the served ref lives in the deploy config). A
document emitted there is addressed at a closure no pod resolves. The document
reaches a fleet pod through the DURABLE ROOT instead —
``keyset.store.durable_root()``, which the pod that traces writes and the next
pod of the endpoint reads.

The document is the SAME schema in both modes and in the pod's own cache: it is
trusted because its closure digest matches what a pod's code would trace, never
because of where it was found.

Where the output goes on a pod: ``/app/.tensorhub/cg-keyset-v1.json``, beside
``endpoint.lock`` (``gen_worker.keyset.store.IMAGE_KEYSET_DIR``), or anywhere
``GEN_WORKER_CG_KEYSET`` points.

``--publish`` — THE MINT LANE FILLS THE HUB (pgw#1353 option (b) / th#2123)
--------------------------------------------------------------------------
The staged file only helps a pod that can read the filesystem it is staged on.
``--publish`` sends every closure this run produced to the hub's key-set store,
``PUT /v1/worker/keysets/<closure_digest>``, which is the one tier an EPHEMERAL
private deployment can reach — no network volume, no baked document, a fresh
container every time. **This is what makes "a serving pod never derives" true**:
the mint lane pays the traces once, at mint time, and every serving pod of that
closure reads them.

Best effort per closure and reported per closure. A 409 means the store already
holds a DIFFERENT document at that address — write-once, first one stands — and
that is worth reading rather than retrying: two runs of the same code against
the same subjects should produce the same class hashes.

Needs a hub and a credential: ``--hub URL`` (or ``$COZY_HUB_URL``) and
``$COZY_WORKER_TOKEN``. Refused by name when either is missing, never silently
skipped — a publish flag that quietly did nothing is how a fleet keeps paying
805 s while a green log line says the document was emitted.

Usage::

    python scripts/emit_cg_keyset.py --from-cache ~/.cache/cozy --out build/
    python scripts/emit_cg_keyset.py --from-cache ~/.cache/cozy --out build/ --publish
    python scripts/emit_cg_keyset.py --derive --module my_endpoint.main \\
        --function generate --slot pipeline=/cas/checkpoint \\
        --slot-ref pipeline=tensorhub/wai-illustrious@prod --out build/ --publish
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

if TYPE_CHECKING:  # pragma: no cover — typing only, keeps the import lazy
    from gen_worker.child_contract import MintSlot


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


#: Where the credential comes from. Env, because it is a SECRET and secrets are
#: config — never a flag, which lands in shell history and in CI logs.
ENV_HUB_URL = "COZY_HUB_URL"
ENV_HUB_TOKEN = "COZY_WORKER_TOKEN"


def hub_tier(url: str) -> "object":
    """The hub to publish to, or a SystemExit naming what is missing.

    Refuses by name rather than degrading to "did not publish". A `--publish`
    that silently does nothing is the failure this whole issue is about: a
    fleet paying 805 s per pod while a green line says the document was
    emitted.
    """
    from gen_worker.keyset.hub import HubTier

    base = (url or os.environ.get(ENV_HUB_URL, "")).strip()
    token = os.environ.get(ENV_HUB_TOKEN, "").strip()
    missing = [name for name, value in (
        (f"--hub or ${ENV_HUB_URL}", base), (f"${ENV_HUB_TOKEN}", token)) if not value]
    if missing:
        raise SystemExit(
            f"--publish needs a hub and a credential; missing: {missing}")
    return HubTier(base_url=base, bearer=token)


def publish(out: Path, tier: "object") -> int:
    """Offer every closure in the staged document to the hub. Returns failures.

    Reads back from the STAGED FILE rather than from whatever the caller has in
    hand, so `--publish` publishes exactly what `--out` says was produced — one
    artifact, one claim, and no way for the two to disagree.
    """
    from gen_worker import keyset
    from gen_worker.keyset import store as keyset_store
    from gen_worker.keyset.hub import publish_closure

    document = keyset_store.read_document(out)
    failed = 0
    for digest_text in sorted(document.closures):
        digest = keyset.parse_closure_digest(digest_text)
        reason = publish_closure(digest, document.closures[digest_text], tier)  # type: ignore[arg-type]
        if reason:
            failed += 1
            print(f"  publish {digest}: REFUSED — {reason}")
        else:
            print(f"  publish {digest}: ok")
    if failed:
        print(f"{failed} closure(s) did not reach the hub")
    return failed


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


def resolved_slots(
    paths: Dict[str, str], refs: Dict[str, str],
) -> Dict[str, "MintSlot"]:
    """Build one COMPLETE ``MintSlot`` per slot, or refuse by name.

    ``MintSlot`` is "present and complete, or absent" by construction, so this
    mirrors that rather than filling a hole: a slot missing either half is named
    in the refusal instead of being built around a ``None`` the traced graph
    would silently depend on. That hole is exactly what broke this mode —
    ``MintSlot(ref=None, path=path)`` against a struct with a required ``ref``
    and (since pgw#1333) a required ``facts``.
    """
    from gen_worker.api.binding import Hub
    from gen_worker.child_contract import MintSlot
    from gen_worker.serving_facts import FactsUnavailable

    incomplete = sorted(set(paths) ^ set(refs))
    if incomplete:
        raise SystemExit(
            f"every slot needs BOTH --slot NAME=PATH and --slot-ref NAME=REF; "
            f"incomplete: {incomplete}")
    out: Dict[str, MintSlot] = {}
    for name in sorted(paths):
        out[name] = MintSlot(
            # `Hub` applies the tensorhub grammar's own validation, including
            # the `@release` split, so a mistyped ref refuses here rather than
            # becoming a wrong SUBJECT in the closure digest.
            ref=Hub(refs[name].strip()),
            path=paths[name],
            # HONEST, not a stub: this producer never asked a catalog. `facts`
            # is deliberately absent from the closure digest, so it cannot move
            # a key; `facts_or_degrade` turns this into a confession naming this
            # script if anything downstream wants the stamp.
            facts=FactsUnavailable(owed_by="scripts/emit_cg_keyset.py"),
        )
    return out


def derive(
    modules: Tuple[str, ...], function: str, slots: Dict[str, str],
    refs: Dict[str, str], out: Path, work: Path,
) -> int:
    from gen_worker import aot_declaration, boot_key, keyset
    from gen_worker.api.export_contract import export_declaration
    from gen_worker.child_contract import CompileSpec
    from gen_worker.keyset import store as keyset_store
    from gen_worker.registry import collect_endpoints

    specs = [s for s in collect_endpoints(list(modules)) if s.name == function]
    if not specs:
        print(f"no endpoint named {function!r} in {list(modules)!r}")
        return 1
    cfg = specs[0].compile_contract()
    if cfg is None:
        print(f"{function!r} declares no compiled graph, so it has no key set")
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
    resolved = resolved_slots(slots, refs)
    cache = work / "cache"
    # The ADDRESS, stated before the traces and independently of them: it is a
    # pure function of code plus the resolved subjects, it costs milliseconds,
    # and having it up front is what lets the document be recovered below even
    # when the derivation's LAST step fails.
    closure = keyset.closure_of(
        family=family, function=function, modules=modules, cfg=spec,
        slots=resolved)
    folded = ""
    try:
        derived = boot_key.derive(
            function=function,
            modules=modules,
            family=family,
            cfg=spec,
            slots=resolved,
            declared_hint=len(list(aot_declaration.compiled_graph_plans(declaration))),
            work_root=work / "trace",
            memo_dir=cache,
            trust_memo=False,
            emitted_by=f"scripts/emit_cg_keyset.py {family}",
        )
        folded = f"{len(derived.entry_keys)} key(s) folded, {derived.wall_ms} ms"
    except Exception as exc:  # noqa: BLE001 — the document may exist regardless
        # THE MINT LANE'S EMITTER DOES NOT NEED A CARD. The document is
        # machine-INDEPENDENT — TCG class hashes, nothing folded — while the
        # tail of `derive` folds this process's `sm` into keys THIS script never
        # uses, and on a cardless box that fold refuses `no_runtime_sm`. The traces
        # are recorded BEFORE the fold (pgw#1353), so look for what landed rather than
        # treating the exception as the answer.
        folded = f"not folded ({type(exc).__name__}: {exc})"
    try:
        document = keyset_store.read_document(cache / keyset.KEYSET_FILENAME)
    except keyset.KeySetError as exc:
        print(f"the derivation recorded no document: {exc}")
        return 1
    row = document.closures.get(str(closure))
    if row is None:
        print(
            f"the derivation recorded no closure {closure} — nothing to emit "
            f"({folded})")
        return 1
    print(f"derived {len(row.classes)} class(es) for {family} — {folded}")
    print(f"closure={closure}")
    _merge(out, {str(closure): row})
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
        help="where a resolved setup slot's checkpoint TREE is; the "
             "structure-only build reads its configs")
    ap.add_argument(
        "--slot-ref", action="append", default=[], metavar="NAME=REF",
        help="WHICH checkpoint that slot is (owner/repo@release); it rides the "
             "closure digest, so a document derived without it is unaddressable")
    ap.add_argument("--out", default="")
    ap.add_argument("--work", default="")
    ap.add_argument(
        "--publish", action="store_true",
        help="also PUT every emitted closure to the hub's key-set store "
             f"(th#2123), so a pod with no volume and no baked document reads "
             f"it instead of tracing. Needs --hub/${ENV_HUB_URL} and "
             f"${ENV_HUB_TOKEN}.")
    ap.add_argument(
        "--hub", default="",
        help=f"the hub base URL for --publish; defaults to ${ENV_HUB_URL}")
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
        refs: Dict[str, str] = {}
        for item in args.slot_ref:
            name, _, ref = str(item).partition("=")
            if not name or not ref:
                ap.error(f"--slot-ref must be NAME=REF, got {item!r}")
            refs[name] = ref
        work = Path(args.work or (out.parent / ".cg-keyset-work"))
        # The tier is built BEFORE the traces, so a missing credential costs a
        # refusal in the first second rather than after a full derive.
        tier = hub_tier(args.hub) if args.publish else None
        status = derive(tuple(args.module), args.function, slots, refs, out, work)
        if status or tier is None:
            return status
        return 1 if publish(out, tier) else 0

    if not args.from_cache:
        ap.error("pass --from-cache DIR or --derive")
    tier = hub_tier(args.hub) if args.publish else None
    status = from_cache(Path(args.from_cache), out)
    if status or tier is None:
        return status
    return 1 if publish(out, tier) else 0


if __name__ == "__main__":
    sys.exit(main())
