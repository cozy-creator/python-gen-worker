"""``gen-worker download`` — put a checkpoint on this machine.

pgw#1491. CHECKPOINT-centric, never endpoint-centric: an endpoint declares no
checkpoints at all (DESIGN-RULINGS 2026-08-19 — "that's runtime config"), so
the download target is a property of the REF, and the endpoint only ever
supplies a default for which ref you probably meant.

    gen-worker download tensorhub/sd15-base@1
    gen-worker download                      # the configured / recommended ref

## One materialization path, and it is the pod's

The bytes land through ``ModelStore.ensure_local`` — the same call
``boot_materialize`` makes when a pod boots (th#2204). That is not reuse for
tidiness: a second downloader would be a second answer to "is this tree
complete and correct", and the integrity gate lives inside this one
(``_verify_snapshot_tree`` → quarantine → refetch). A CLI that fetched bytes
its own way would be a CLI whose trees the serving path distrusts.

## Progress is the download's own

Nothing here reports progress separately. ``ModelStore`` emits ``ModelEvent``,
``weight_fetch`` positions and the ``snapshot_pull`` roll-up through one
funnel; with no bound transport sink — which is exactly the CLI — every one of
them lands on the logger as local progress, and on a pod the same events ride
the worker stream as the subset the hub consumes.

## Where "the recommended checkpoint" comes from

Paul's ruling puts the recommendation on the HUB, ATTACHED to the endpoint
(never declared BY it). That hub surface DOES NOT EXIST yet — there is no
endpoint→recommended-checkpoint route in tensorhub (verified 2026-08-19; every
`recommend*` hit there is resources, civitai settings, or a training best_step).
So the ladder below is explicit about which rung answered, and the missing rung
refuses BY NAME instead of guessing:

1. an explicit ref on the command line — always wins;
2. this machine's runtime config for the endpoint (``cozy.toml``'s
   ``[runtime] checkpoints``) — which is what "checkpoints are runtime config"
   means locally;
3. the hub's recommended-checkpoint metadata — STUBBED: refuses naming the
   missing hub half rather than silently falling back to a guess.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

logger = logging.getLogger(__name__)

#: The per-machine runtime config beside an endpoint. Rung 2 of the ladder.
RUNTIME_CONFIG_NAME = "cozy.toml"


class DownloadError(RuntimeError):
    """A ref could not be resolved or materialized. Always names which."""


class NoRecommendedCheckpoint(DownloadError):
    """Nothing named a checkpoint and no rung of the ladder could answer."""


# --------------------------------------------------------------------------
# Which checkpoint(s)
# --------------------------------------------------------------------------


def runtime_config_refs(endpoint_dir: Path) -> Tuple[str, ...]:
    """Checkpoint refs this machine's runtime config names for the endpoint.

    Runtime config, not endpoint declaration — the file is the operator's, sits
    beside the endpoint, and is not part of what gets published. An absent file
    is not an error; it means this machine has not been told yet.
    """
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
    """The author's recommended checkpoint(s), by the ladder in the docstring.

    Raises :class:`NoRecommendedCheckpoint` naming every rung it tried, because
    "download did nothing" must never be a silent outcome.
    """
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


# --------------------------------------------------------------------------
# Materialization
# --------------------------------------------------------------------------


def _logging_emit() -> Any:
    """The CLI's transport sink: there isn't one, and that is the design.

    ``ModelStore`` needs an async emitter for the wire messages a pod sends its
    hub. A CLI has no hub, so the events go where every other unbound sink
    sends them — the logger — and the SAME funnel serves both.
    """

    async def emit(_message: Any) -> None:
        return None

    return emit


def _resolve_snapshot(ref_text: str) -> Tuple[Any, Any]:
    """(WireRef, pb.Snapshot) for a ref, resolved against the hub.

    A tensorhub ref carries no file list of its own; the hub composes the
    manifest. The worker can never resolve one itself, so a ref with no
    reachable hub is a refusal here rather than a mysterious
    ``MissingSnapshotError`` four frames down.
    """
    from ..models.hub_client import HubResolveError, resolve_repo
    from ..models.refs import format_model_ref, parse_model_ref
    from ..wire_snapshots import snapshot_from_resolved_repo

    try:
        parsed = parse_model_ref(ref_text)
    except ValueError as exc:
        raise DownloadError(f"{ref_text}: {exc}") from exc
    ref = format_model_ref(parsed)
    if parsed.tensorhub is None:
        # A hf/civitai/modelscope ref resolves inside the download layer
        # itself; no hub manifest exists to fetch, and passing None is how
        # that is stated rather than synthesizing an empty one.
        return ref, None
    try:
        resolved = resolve_repo(parsed.tensorhub)
    except HubResolveError as exc:
        raise DownloadError(f"{ref_text}: {exc}") from exc
    return ref, snapshot_from_resolved_repo(resolved)


async def _materialize(refs: Sequence[str]) -> Dict[str, Path]:
    from ..models.store import ModelStore
    from . import workspace

    # STATE the box weight CAS; do not inherit ModelStore's default.
    # `models.cache_paths.tensorhub_cas_dir()` answers `/tmp/tensorhub-cache/cas`
    # — correct for a pod, where /tmp is the container's own disk and the pod
    # is thrown away with it. On a workstation it is a SECOND store beside the
    # box-shared one every other tool reads (`~/.cache/tensorhub/cas`), and
    # writing there is the failure `cli/workspace` names in its own docstring:
    # a download that pre-warms a directory the serving path never looks in.
    # Measured, not reasoned about — the first run of this verb put 2.6 GB in
    # the wrong place.
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
    """Put every ref on disk; return ``{ref: tree}``. The shared entry.

    ``up`` calls this at boot for the refs its runtime config names, and
    ``download`` calls it for the refs on its command line. ONE path, so an
    endpoint can never be brought up against a tree the download verb would
    have rejected.
    """
    if not refs:
        return {}
    return asyncio.run(_materialize(list(refs)))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


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
