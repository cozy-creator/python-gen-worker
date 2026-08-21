"""``gen-worker download`` — put a checkpoint on this machine."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

logger = logging.getLogger(__name__)

RUNTIME_CONFIG_NAME = "cozy.toml"


class DownloadError(RuntimeError):
    """A ref could not be resolved or materialized."""


class NoRecommendedCheckpoint(DownloadError):
    """Nothing named a checkpoint and no rung of the ladder could answer."""


def runtime_config_refs(endpoint_dir: Path) -> Tuple[str, ...]:
    """Checkpoint refs this machine's runtime config names for the endpoint."""
    import tomllib

    path = Path(endpoint_dir) / RUNTIME_CONFIG_NAME
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise DownloadError(f"{path}: {exc}") from exc
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise DownloadError(f"{path} is not readable TOML: {exc}") from exc
    runtime = document.get("runtime")
    if not isinstance(runtime, dict):
        return ()
    refs = runtime.get("checkpoints")
    if isinstance(refs, str):
        return (refs,)
    if isinstance(refs, list) and all(isinstance(r, str) for r in refs):
        return tuple(refs)
    if refs is None:
        return ()
    raise DownloadError(
        f"{path}: [runtime].checkpoints must be a ref string or a list of them"
    )


def recommended_refs(endpoint_dir: Path) -> Tuple[str, ...]:
    """The author's recommended checkpoint(s), by the ladder in the docstring."""
    configured = runtime_config_refs(endpoint_dir)
    if configured:
        return configured
    raise NoRecommendedCheckpoint(
        f"no checkpoint named, and nothing on this machine says which one.\n"
        f"  Name one:            gen-worker download owner/name@release\n"
        f"  Or configure it:     {Path(endpoint_dir) / RUNTIME_CONFIG_NAME} ->\n"
        f"                         [runtime]\n"
        f"                         checkpoints = [\"owner/name@release\"]\n"
        f"  The hub's author-recommended-checkpoint metadata (the third source)\n"
        f"  is NOT BUILT YET — tensorhub exposes no endpoint->recommended\n"
        f"  checkpoint route today, so this command will not guess one."
    )


def _logging_emit() -> Any:

    async def emit(_message: Any) -> None:
        return None

    return emit


def _resolve_snapshot(ref_text: str) -> Tuple[Any, Any]:
    from ..models.hub_client import HubResolveError, resolve_repo
    from ..models.refs import format_model_ref, parse_model_ref
    from ..wire_snapshots import snapshot_from_resolved_repo

    try:
        parsed = parse_model_ref(ref_text)
    except ValueError as exc:
        raise DownloadError(f"{ref_text}: {exc}") from exc
    ref = format_model_ref(parsed)
    if parsed.tensorhub is None:
        return ref, None
    try:
        resolved = resolve_repo(parsed.tensorhub)
    except HubResolveError as exc:
        raise DownloadError(f"{ref_text}: {exc}") from exc
    return ref, snapshot_from_resolved_repo(resolved)


async def _materialize(refs: Sequence[str]) -> Dict[str, Path]:
    from ..models.store import ModelStore
    from . import workspace

    store = ModelStore(
        _logging_emit(),
        cache_dir=workspace.weights_cas_root(),
    )
    store.bind_loop()
    trees: Dict[str, Path] = {}
    for ref_text in refs:
        ref, snapshot = _resolve_snapshot(ref_text)
        if snapshot is not None:
            store.bank_snapshot(ref, snapshot)
        if await store.announce_resident(ref, snapshot):
            path = store.local_path(ref)
            if path is not None:
                logger.info("checkpoint %s already resident at %s", ref, path)
                trees[ref_text] = Path(path)
                continue
        try:
            path = await store.ensure_local(ref, snapshot)
        except Exception as exc:  # noqa: BLE001 — every failure names its ref
            raise DownloadError(
                f"{ref_text}: {type(exc).__name__}: {exc}"
            ) from exc
        logger.info("checkpoint %s materialized at %s", ref, path)
        trees[ref_text] = Path(path)
    return trees


def materialize_refs(refs: Sequence[str]) -> Dict[str, Path]:
    """Put every ref on disk; return ``{ref: tree}``."""
    if not refs:
        return {}
    return asyncio.run(_materialize(list(refs)))


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    parser = sub.add_parser(
        "download",
        help="Fetch a checkpoint onto this machine.",
        description=(
            "Materialize checkpoint refs into the local weight store, through "
            "the same path a pod's boot uses (integrity gate included). With "
            "no ref, uses what this machine's runtime config names for the "
            "endpoint."
        ),
    )
    parser.add_argument(
        "refs", nargs="*", metavar="REF",
        help="checkpoint ref(s), e.g. owner/name@release. Omit to use the "
             "endpoint's configured/recommended checkpoint(s).",
    )
    parser.add_argument(
        "-C", "--endpoint-dir", default=".",
        help="endpoint directory (default: the current one) — only read when "
             "no REF is given.",
    )
    parser.set_defaults(_handler=run_download)


def run_download(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stderr, force=False,
    )
    refs: Tuple[str, ...] = tuple(args.refs)
    if not refs:
        try:
            refs = recommended_refs(Path(args.endpoint_dir))
        except NoRecommendedCheckpoint as exc:
            sys.stderr.write(f"gen-worker download: {exc}\n")
            return 2
        sys.stderr.write(
            f"gen-worker download: {Path(args.endpoint_dir) / RUNTIME_CONFIG_NAME} "
            f"names {', '.join(refs)}\n"
        )
    try:
        trees = materialize_refs(refs)
    except DownloadError as exc:
        sys.stderr.write(f"gen-worker download: {exc}\n")
        return 1
    for ref_text in refs:
        sys.stdout.write(f"{ref_text}\t{trees[ref_text]}\n")
    return 0


__all__ = [
    "DownloadError",
    "NoRecommendedCheckpoint",
    "RUNTIME_CONFIG_NAME",
    "add_subparser",
    "materialize_refs",
    "recommended_refs",
    "run_download",
    "runtime_config_refs",
]
