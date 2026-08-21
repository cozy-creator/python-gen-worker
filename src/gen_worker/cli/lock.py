"""``gen-worker lock`` — discovery + the graph trace, in one command.

Discovery (static, seconds) and the instrumented derive (real author code on a
real device) both answer "what does this endpoint consist of" and are
invalidated by the same edits, so ``lock`` runs both and writes one file. The
expensive derive is skipped on a re-run whose input digest (author source,
``uv.lock`` closure, checkpoint ref, trace device, torchcg format versions) is
unchanged — see ``endpoint_lock`` for why the key is inputs, not outputs.
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
        action="append",
        default=[],
        metavar="[SLOT=]OWNER/NAME@REV",
        help="owner/name@revision to trace against. Required for an endpoint "
             "that declares a model slot; a weightless endpoint needs none. "
             "Repeat with `slot=ref` to give a SECONDARY model slot its own "
             "checkpoint; the bare form is the primary model's.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="[SLOT=]PATH",
        help="a local checkpoint tree, INSTEAD of resolving --checkpoint-ref "
             "through the weight CAS (for a tree the store does not hold). "
             "Repeatable as `slot=path`, same rule as --checkpoint-ref.",
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
    print(message, file=sys.stderr, flush=True)


def _split_slot(value: str) -> tuple[str, str]:

    slot, sep, rest = value.partition("=")
    if sep and slot.isidentifier():
        return slot, rest
    return "", value


def _one_tree(checkpoint: str, ref_text: str) -> tuple[Optional[Path], str]:
    if checkpoint:
        tree = Path(checkpoint).resolve()
        if not tree.is_dir():
            raise ws.WorkspaceError(f"--checkpoint {tree} is not a directory")
        return tree, ref_text or f"local:{tree}"
    if not ref_text:
        return None, ""
    ref = ws.parse_checkpoint_ref(ref_text)
    return ws.resolve_checkpoint(ref), str(ref)


def _checkpoint_tree(
    args: argparse.Namespace,
) -> tuple[Optional[Path], str, dict[str, Path], tuple[str, ...]]:

    refs: dict[str, str] = {}
    trees: dict[str, str] = {}
    for raw in args.checkpoint_ref:
        slot, value = _split_slot(raw)
        if slot in refs:
            raise ws.WorkspaceError(
                "--checkpoint-ref given twice for "
                + (f"slot {slot!r}" if slot else "the primary model")
            )
        refs[slot] = value
    for raw in args.checkpoint:
        slot, value = _split_slot(raw)
        if slot in trees:
            raise ws.WorkspaceError(
                "--checkpoint given twice for "
                + (f"slot {slot!r}" if slot else "the primary model")
            )
        trees[slot] = value

    primary_tree, primary_ref = _one_tree(trees.get("", ""), refs.get("", ""))
    slot_trees: dict[str, Path] = {}
    slot_refs: list[str] = []
    for slot in sorted(set(refs) | set(trees)):
        if not slot:
            continue
        tree, ref_text = _one_tree(trees.get(slot, ""), refs.get(slot, ""))
        if tree is None:
            raise ws.WorkspaceError(
                f"slot {slot!r} was named with no resolvable checkpoint"
            )
        slot_trees[slot] = tree
        slot_refs.append(f"{slot}={ref_text}")
    return primary_tree, primary_ref, slot_trees, tuple(slot_refs)


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

    try:
        tree, ref_text, slot_trees, slot_refs = _checkpoint_tree(args)
    except ws.WorkspaceError as exc:
        _say(f"error: {exc}")
        return 2

    device = ws.trace_device()
    graph_cas_root = Path(args.graph_cas).resolve() if args.graph_cas else ws.graph_cas_root()
    from ..env_identity import (
        EnvIdentityError,
        compile_stack_from_lockfile,
        cuda_bucket,
        installed_stack_drift,
        lockfile_beside,
    )

    lockfile = lockfile_beside(root)
    if lockfile is None:
        _say(f"error: no uv.lock beside {root}; a lock states the compile "
             f"stack it traced under, and there is no second source for it")
        return 1
    try:
        stack = compile_stack_from_lockfile(lockfile, bucket=cuda_bucket())
    except EnvIdentityError as exc:
        _say(f"error: {exc}")
        return 1
    _say("lock: compile stack "
         + ", ".join(f"{name} {version}" for name, version in stack[:2])
         + f" (+{max(len(stack) - 2, 0)} more)")
    for row in installed_stack_drift(dict(stack)):
        _say(f"warning: compile-stack drift vs this venv: {row}")

    want = el.inputs_digest(
        root=root,
        module_name=config.main,
        checkpoint_ref=ref_text,
        trace_device=device,
        lockfile=lockfile,
        extra=slot_refs,
    )

    existing: Optional[el.DeriveBlock]
    try:
        existing = el.read_derive_block(out) if out.is_file() else None
    except el.LockError as exc:
        _say(f"lock: ignoring unusable saved trace: {exc}")
        existing = None

    if args.check:
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
            config, root, out, existing, tree, graph_cas_root, device, lockfile,
            slot_trees,
        )

    store = ws.local_graph_store(graph_cas_root)
    reuse = el.derive_is_reusable(
        existing,
        want_inputs_digest=want,
        want_trace_device=device,
        cas_has=lambda graph: bool(store.has_program(graph)),
    )
    if reuse.ok and not args.force:
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
        _say(
            "lock: no --checkpoint-ref/--checkpoint. Tracing a weightless "
            "endpoint; if this endpoint declares a model slot the derive will "
            "say so rather than emit an empty document."
        )

    traced = _trace(
        config, root, tree, graph_cas_root, device, lockfile, slot_trees,
    )
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
    programs = el.graph_identities(json.loads(result.document))
    _say_posture(result)
    for name, reason in result.unenumerable_entrypoints:
        _say(f"lock: entrypoint {name} NOT enumerated -- {reason}")
    for skipped in result.unservable_payloads:
        _say(f"lock: payload SKIPPED (unservable) -- {skipped}")
    _say(
        f"lock: traced {graphs} specialization(s), {len(programs)} exported "
        f"program(s) banked BY GRAPH IDENTITY in the CAS at {graph_cas_root} "
        f"({elapsed:.1f}s) — the bytes stay on this box; the document carries "
        f"identities, never their addresses"
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
        "unenumerable_entrypoints": [
            name for name, _reason in result.unenumerable_entrypoints
        ],
        "unservable_payloads": list(result.unservable_payloads),
        "seconds": round(elapsed, 1),
    }))
    return 0


POSTURE_WEIGHTLESS = "weightless"
POSTURE_NO_COMPILE_TARGETS = "traced-no-compile-targets"
POSTURE_TRACED = "traced"


def _posture(result: Any) -> str:
    if result.weightless:
        return POSTURE_WEIGHTLESS
    if not result.lane_graphs:
        return POSTURE_NO_COMPILE_TARGETS
    return POSTURE_TRACED


def _say_posture(result: Any) -> None:

    for lane, hashes in result.lane_graphs.items():
        _say(f"lock: lane {lane}: {len(hashes)} specialization(s)")
    posture = _posture(result)
    if posture == POSTURE_WEIGHTLESS:
        _say(
            f"lock: {result.endpoint}: {posture} — no model class, nothing to "
            f"compile"
        )
    elif posture == POSTURE_NO_COMPILE_TARGETS:
        _say(
            f"lock: {result.endpoint}: {posture} — the trace RAN and load() "
            f"marked no module via ctx.compile(), so there is nothing to "
            f"compile (lane(s): "
            f"{', '.join(result.unmarked_lanes) or 'none'}). The ABSENT mark "
            f"IS the eager declaration (Paul, 2026-08-20) — there is no "
            f"keyword. Mark a module to compile it."
        )


def _trace(
    config: Any,
    root: Path,
    tree: Optional[Path],
    graph_cas_root: Path,
    device: str,
    lockfile: Optional[Path] = None,
    slot_trees: Optional[dict[str, Path]] = None,
) -> Optional[tuple[Any, float]]:

    from ..discovery.discover import prime_sys_path
    from ..release.derive import DeriveError, derive_release

    prime_sys_path(root)
    try:
        module = importlib.import_module(config.main)
    except Exception as exc:  # noqa: BLE001
        _say(f"error: failed to import {config.main!r}: {exc}")
        return None

    _say(f"lock: trace device is {device}-class")
    from ..env_identity import lockfile_beside

    started = time.monotonic()
    try:
        result = derive_release(
            module,
            checkpoint_dir=tree if tree is not None else root,
            lockfile=lockfile if lockfile is not None else lockfile_beside(root),
            graph_cas=graph_cas_root,
            slot_checkpoints=slot_trees or {},
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
    graph_cas_root: Path,
    device: str,
    lockfile: Optional[Path] = None,
    slot_trees: Optional[dict[str, Path]] = None,
) -> int:

    traced = _trace(
        config, root, tree, graph_cas_root, device, lockfile, slot_trees,
    )
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
