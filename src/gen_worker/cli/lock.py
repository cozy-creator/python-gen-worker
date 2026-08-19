"""``gen-worker lock`` — discovery + the graph trace, in one command.

pgw#1466, Paul's verb 1: "go through the code and produce the endpoint.lock".

Two things were separate before and had no business being: discovery
(``python -m gen_worker.discovery``, static, seconds) and the trace
(``gen-worker release derive``, real author code on a real device, ~165s). Both
answer "what does this endpoint consist of", both are invalidated by the same
edit, and a lock carrying one without the other is a lock every later verb has
to finish. So ``lock`` runs both and writes one file.

The expensive half is skipped on a re-run. Paul: "re-runs don't need to
regenerate". The skip key is a digest over the derive's INPUTS — author source,
``uv.lock`` closure, checkpoint ref, trace device, torchcg's format versions —
so it can be computed in milliseconds and answered before paying anything. See
``endpoint_lock`` for why it is inputs and not outputs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

from . import endpoint_lock as el
from . import workspace as ws


def add_subparser(sub: Any) -> None:
    parser = sub.add_parser(
        "lock",
        help="discover entrypoints and trace the graphs; write endpoint.lock",
        description=__doc__,
    )
    parser.add_argument(
        "endpoint_dir",
        nargs="?",
        default=".",
        help="endpoint project root — the directory holding pyproject.toml "
             "and endpoint.toml (default: cwd)",
    )
    parser.add_argument(
        "--checkpoint-ref",
        default="",
        help="owner/name@revision to trace against. Required for an endpoint "
             "that declares a model slot; a weightless endpoint needs none.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="a local checkpoint tree, INSTEAD of resolving --checkpoint-ref "
             "through the weight CAS (for a tree the store does not hold)",
    )
    parser.add_argument(
        "--graph-cas",
        default="",
        help=f"exported-program CAS root (default: {ws.DEFAULT_GRAPH_CAS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-trace even when the saved trace is reusable",
    )
    parser.add_argument(
        "--discovery-only",
        action="store_true",
        help="write the discovery blocks and no [derive] — the bake-time shape",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed lock is current and WRITE NOTHING; exit 1 "
             "on drift. The freshness gate for CI (endpoint.lock is source, "
             "committed beside the endpoint, and a stale one is a wrong build)",
    )
    parser.add_argument(
        "-o", "--out",
        default="",
        help=f"lock path (default: <endpoint_dir>/{el.LOCK_FILENAME})",
    )
    parser.set_defaults(_handler=run_lock)


def _say(message: str) -> None:
    """Progress to stderr, so stdout stays the machine-readable summary."""
    print(message, file=sys.stderr, flush=True)


def _checkpoint_tree(args: argparse.Namespace) -> tuple[Optional[Path], str]:
    """(tree, ref-string). Both may be absent — a weightless endpoint has none."""
    if args.checkpoint:
        tree = Path(args.checkpoint).resolve()
        if not tree.is_dir():
            raise ws.WorkspaceError(f"--checkpoint {tree} is not a directory")
        # A local tree still needs an identity for the skip key. Without one,
        # swapping the tree under a stable path would reuse a stale trace.
        return tree, args.checkpoint_ref or f"local:{tree}"
    if not args.checkpoint_ref:
        return None, ""
    ref = ws.parse_checkpoint_ref(args.checkpoint_ref)
    return ws.resolve_checkpoint(ref), str(ref)


def run_lock(args: argparse.Namespace) -> int:
    from ..discovery.discover import discover_manifest
    from ..discovery.project import load_project_config

    root = Path(args.endpoint_dir).resolve()
    if not root.is_dir():
        _say(f"error: {root} is not a directory")
        return 2
    out = Path(args.out).resolve() if args.out else root / el.LOCK_FILENAME

    try:
        config = load_project_config(root)
    except Exception as exc:  # noqa: BLE001 - the author's config is the check
        _say(f"error: {root}: {exc}")
        return 2

    # ---------------------------------------------------------------- discovery
    started = time.monotonic()
    try:
        manifest = discover_manifest(root)
    except Exception as exc:  # noqa: BLE001 - author import errors are the check
        _say(f"error: discovery failed: {exc}")
        return 1
    entrypoints = manifest.get("entrypoints") or []
    _say(
        f"lock: discovered {len(entrypoints)} entrypoint(s) in "
        f"{config.main} ({time.monotonic() - started:.1f}s)"
    )

    if args.discovery_only:
        el.write_lock(out, manifest, None)
        _say(f"lock: wrote {out} (discovery only, no [derive])")
        print(json.dumps({"lock": str(out), "derive": "skipped:--discovery-only"}))
        return 0

    # ------------------------------------------------------------------- trace
    try:
        tree, ref_text = _checkpoint_tree(args)
    except ws.WorkspaceError as exc:
        _say(f"error: {exc}")
        return 2

    device = ws.trace_device()
    graph_cas_root = Path(args.graph_cas).resolve() if args.graph_cas else ws.graph_cas_root()
    from ..env_identity import describe_lockfile_drift, lockfile_beside

    # pgw#1472: a lockfile is a DIAGNOSTIC here and nothing else. The identity
    # this run stamps is `env_closure_hash()` of the process doing the tracing,
    # which is also what invalidates a saved trace (`inputs_digest`).
    lockfile = lockfile_beside(root)
    if lockfile is not None:
        _say(f"lock: {describe_lockfile_drift(lockfile)}")

    want = el.inputs_digest(
        root=root,
        module_name=config.main,
        checkpoint_ref=ref_text,
        trace_device=device,
    )

    existing: Optional[el.DeriveBlock]
    try:
        existing = el.read_derive_block(out) if out.is_file() else None
    except el.LockError as exc:
        _say(f"lock: ignoring unusable saved trace: {exc}")
        existing = None

    if args.check:
        # endpoint.lock is SOURCE: it is committed beside the endpoint and a
        # stale one builds the wrong thing. The gate is cheap in the common
        # case ON PURPOSE — the inputs digest is the derive's own definition of
        # "would this trace differently", computed in milliseconds, so CI can
        # run it on every push. Only when the inputs MOVED does it pay for a
        # re-derive, and then it compares the produced document against the
        # committed one: an input that changed without changing the output is
        # not drift, and reporting it as drift would train people to ignore
        # this check.
        if existing is None:
            _say(f"error: {out} has no saved trace to check — run `gen-worker lock`")
            return 2
        if existing.inputs_digest == want and existing.trace_device == device:
            _say(f"lock --check: {out} is current (inputs unchanged)")
            print(json.dumps({
                "lock": str(out), "check": "current",
                "document_digest": existing.document_digest,
            }))
            return 0
        _say("lock --check: inputs moved — re-deriving to see whether the "
             "document moved with them")
        return _check_by_rederive(
            config, root, out, existing, tree, lockfile, graph_cas_root, device,
        )

    cas = ws.local_cas(graph_cas_root)
    reuse = el.derive_is_reusable(
        existing,
        want_inputs_digest=want,
        want_trace_device=device,
        cas_has=lambda digest: bool(cas.contains(digest)),
    )
    if reuse.ok and not args.force:
        # The whole point of the verb. Rewrite the lock so freshly-discovered
        # blocks land, but carry the saved [derive] through untouched — the
        # bytes are the document's identity and re-encoding would break the
        # digest that makes it verifiable.
        el.write_lock(out, manifest, reuse.block)
        assert reuse.block is not None
        _say(f"lock: reusing saved trace ({reuse.reason})")
        _say(f"lock: wrote {out}")
        print(json.dumps({
            "lock": str(out),
            "derive": "reused",
            "document_digest": reuse.block.document_digest,
            "trace_device": reuse.block.trace_device,
        }))
        return 0
    if args.force and existing is not None:
        _say("lock: --force, re-tracing despite a saved trace")
    else:
        _say(f"lock: tracing — {reuse.reason}")

    if tree is None:
        # Weightless is legal (pgw#1392) and derive_release handles it, but a
        # model-bearing endpoint reaching here with no tree would trace nothing
        # and write a document that looks eager-permanent. Refuse by name.
        _say(
            "lock: no --checkpoint-ref/--checkpoint. Tracing a weightless "
            "endpoint; if this endpoint declares a model slot the derive will "
            "say so rather than emit an empty document."
        )

    traced = _trace(config, root, tree, lockfile, graph_cas_root, device)
    if traced is None:
        return 1
    result, elapsed = traced

    interface_v, _document_v = el.torchcg_format_versions()
    block = el.DeriveBlock(
        v=el.DERIVE_BLOCK_V,
        interface_v=interface_v,
        inputs_digest=want,
        document_digest=result.digest,
        document=result.document.decode("ascii"),
        trace_device=device,
        endpoint=result.endpoint,
    )
    el.write_lock(out, manifest, block)

    graphs = sum(len(h) for h in result.lane_graphs.values())
    programs = el.program_digests(json.loads(result.document))
    _say_posture(result)
    _say(
        f"lock: traced {graphs} specialization(s), {len(programs)} exported "
        f"program(s) in the CAS at {graph_cas_root} ({elapsed:.1f}s)"
    )
    _say(f"lock: wrote {out}")
    print(json.dumps({
        "lock": str(out),
        "derive": "traced",
        "endpoint": result.endpoint,
        "posture": _posture(result),
        "document_digest": result.digest,
        "trace_device": device,
        "specializations": graphs,
        "programs": len(programs),
        "seconds": round(elapsed, 1),
    }))
    return 0


#: The compile posture of a derived endpoint — ONE WORD, always printed
#: (pgw#1488). Silence used to be a posture: `lanes=()` disabled compilation
#: with no output at all, and a missing contract document refused to trace at
#: all. Both are states now, and both are named.
POSTURE_EAGER_BY_DECLARATION = "eager-by-declaration"
POSTURE_WEIGHTLESS = "weightless"
POSTURE_NO_COMPILE_TARGETS = "traced-no-compile-targets"
POSTURE_TRACED = "traced"


def _posture(result: Any) -> str:
    if result.eager_only:
        return POSTURE_EAGER_BY_DECLARATION
    if result.weightless:
        return POSTURE_WEIGHTLESS
    if not result.lane_graphs:
        return POSTURE_NO_COMPILE_TARGETS
    return POSTURE_TRACED


def _say_posture(result: Any) -> None:
    """State what this endpoint compiles, and why, in the author's words."""

    for lane, hashes in result.lane_graphs.items():
        _say(f"lock: lane {lane}: {len(hashes)} specialization(s)")
    for lane in result.derived_lanes:
        _say(
            f"lock: lane {lane}: identity DERIVED from the model class — this "
            f"endpoint declares no layout contract, so the trace named the "
            f"lane itself. A contract document ATTACHES to these artifacts "
            f"later as fleet metadata; it was never needed to produce them."
        )
    posture = _posture(result)
    if posture == POSTURE_EAGER_BY_DECLARATION:
        _say(
            f"lock: {result.endpoint}: {posture} — eager_only="
            f"{result.eager_only!r}. Nothing was traced because the author "
            f"said so."
        )
    elif posture == POSTURE_WEIGHTLESS:
        _say(
            f"lock: {result.endpoint}: {posture} — no model class, nothing to "
            f"compile"
        )
    elif posture == POSTURE_NO_COMPILE_TARGETS:
        _say(
            f"lock: {result.endpoint}: {posture} — the trace RAN and load() "
            f"marked no module via ctx.compile(), so there is nothing to "
            f"compile (lane(s): "
            f"{', '.join(result.unmarked_lanes) or 'none'}). Mark a module to "
            f"compile it; declare eager_only=\"<why>\" to state that this is "
            f"permanent."
        )


def _trace(
    config: Any,
    root: Path,
    tree: Optional[Path],
    lockfile: Path,
    graph_cas_root: Path,
    device: str,
) -> Optional[tuple[Any, float]]:
    """Import the author's module and derive it. ``None`` = said why, failed."""

    from ..discovery.discover import prime_sys_path
    from ..release.derive import DeriveError, derive_release

    prime_sys_path(root)
    try:
        module = importlib.import_module(config.main)
    except Exception as exc:  # noqa: BLE001
        _say(f"error: failed to import {config.main!r}: {exc}")
        return None

    _say(
        f"lock: trace device is {device}-class"
        + ("" if device == "cuda" else " — CPU-class graphs are NOT servable on "
                                       "a GPU; a cuda mint refuses by name")
    )
    started = time.monotonic()
    try:
        result = derive_release(
            module,
            checkpoint_dir=tree if tree is not None else root,
            graph_cas=graph_cas_root,
        )
    except DeriveError as exc:
        _say(f"error: derive: {exc}")
        return None
    elapsed = time.monotonic() - started
    for warning in result.warnings:
        _say(f"warning: {warning}")
    return result, elapsed


def _check_by_rederive(
    config: Any,
    root: Path,
    out: Path,
    existing: el.DeriveBlock,
    tree: Optional[Path],
    lockfile: Path,
    graph_cas_root: Path,
    device: str,
) -> int:
    """``--check``'s expensive arm: does the committed DOCUMENT still hold?

    Inputs moving is not drift — a comment, a retimed dependency, a new tracer
    that emits the same graphs all move the inputs digest and change nothing
    anyone builds. Only the document decides, so only the document is compared,
    and NOTHING IS WRITTEN either way: `--check` is a question.
    """

    traced = _trace(config, root, tree, lockfile, graph_cas_root, device)
    if traced is None:
        return 1
    result, elapsed = traced
    _say_posture(result)
    drifted = result.digest != existing.document_digest
    if drifted:
        _say(
            f"lock --check: DRIFT — {out} carries document "
            f"{existing.document_digest} and this tree derives "
            f"{result.digest} ({elapsed:.1f}s). The committed lock does not "
            f"describe this endpoint; re-run `gen-worker lock` and commit it."
        )
    else:
        _say(
            f"lock --check: {out} is current — the inputs moved, the document "
            f"did not ({elapsed:.1f}s)"
        )
    print(json.dumps({
        "lock": str(out),
        "check": "drift" if drifted else "current",
        "endpoint": result.endpoint,
        "posture": _posture(result),
        "committed_document_digest": existing.document_digest,
        "derived_document_digest": result.digest,
        "seconds": round(elapsed, 1),
    }))
    return 1 if drifted else 0


__all__ = ["add_subparser", "run_lock"]
