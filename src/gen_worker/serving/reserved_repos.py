"""The reserved-repo payload contract on the v2 serve path (pgw#1475).

A producer payload may name up to four reserved REPO fields — ``source``,
``text_encoder``, ``candidate``, ``resume_from`` — plus the write-side
``destination`` and a ``datasets`` list. The platform materializes each named
repo locally BEFORE the body runs and hands the body a path; the body never
fetches. ``conversion/_common.source_from_ctx`` calls itself *"the ONE door all
25 ``source``-taking producers enter through"*, and its first line is
``if not ctx.source_path: raise``.

**This module is the writer.** ``executor.py`` held it until pgw#1373 deleted
that file, and the v2 rewrite carried the READERS across and not the step that
fills them: ``RequestContext._set_source_path`` was a setter with no caller
anywhere in ``src/``, so 25 of 27 conversion producers died on their own first
line at 0 GPU-seconds. The two survivors are the ingest clones, which name an
upstream URL instead of a reserved ``source`` — the population partitions
exactly at this module's boundary.

Semantics are v1's, preserved deliberately:

* the path handed over is the RAW materialized snapshot directory, not a
  tier-3 view. The projected tree's tensor containers are ``TFSSTUB1``
  pointers, and answering them is the CONSUMER's call — ``real_source_tree``
  (jobs#298) applies ``models.materialized_view.third_party_dir`` at the one
  producer-side door, because only the consumer knows it is about to hand the
  path to a ``safetensors`` reader. Projecting here would cost every producer
  a full copy, including the ones that never read a tensor;
* the ref is NORMALIZED where it enters (pgw#1217). It is client-supplied and
  is then used as the ``snapshots`` lookup key, and that map is keyed in
  normal form — taken verbatim, a non-normal spelling misses its pinned
  snapshot and re-downloads under a second identity. ``snapshots`` is the
  dispatch's own ref-keyed, already-typed view
  (``wire_snapshots.resolved_repos(run.snapshots, run.models)``): the hub
  resolves and ships reserved-repo snapshots unconditionally
  (``ExtractReservedRepoBindingsFromPayload`` -> ``attachJobSourceSnapshots``),
  keyed by REF for artifacts with no composition, which is exactly what a
  payload ``source`` is. v1 passed an EMPTY map here and resolved with no
  digest pin (th#2052); this does not;
* an empty ``ref`` on a declared struct is a typed ``ValidationError``, never
  a silent skip. An ABSENT field is a skip: most producers declare one or two
  of the five.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import msgspec

from .. import weight_position
from ..api.errors import ValidationError
from ..models.cache_paths import tensorhub_cas_dir, tensorhub_fill_source_dir
from ..models.download import ensure_local

from ..models.refs import normalize_model_ref

logger = logging.getLogger(__name__)

#: The reserved repo fields the platform MATERIALIZES, each with the context
#: setter that receives its local path. ``destination`` is deliberately absent:
#: it is a WRITE target that need not exist yet, and fetching it would be a
#: download of the thing the job is about to produce.
RESERVED_REPO_FIELDS: Tuple[str, ...] = (
    "source",
    "text_encoder",
    "candidate",
    "resume_from",
)

#: Every reserved struct STAMPED onto the context as a plain dict, materialized
#: or not. ``ctx.source`` is read beside ``ctx.source_path`` — `source_from_ctx`
#: reads `info["ref"]` and `info["attributes"]` — so a stamp missing here is the
#: same defect one field over.
RESERVED_INFO_FIELDS: Tuple[str, ...] = RESERVED_REPO_FIELDS + ("destination",)

_SETTER_FOR: Dict[str, str] = {
    "source": "_set_source_path",
    "text_encoder": "_set_text_encoder_path",
    "candidate": "_set_candidate_path",
    "resume_from": "_set_resume_from_path",
}


def reserved_repo_info(payload: Any, field_name: str) -> Dict[str, Any]:
    """``payload.<field_name>`` as a plain dict ({} when absent).

    Producer payloads carry these reserved-name structs (#376, pgw#594,
    pgw#684, pgw#1242). The set of names is hardcoded above; pgw#690 tracks
    making it declarative.
    """
    obj = getattr(payload, field_name, None)
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    try:
        out = msgspec.to_builtins(obj)
    except Exception:
        return {}
    return out if isinstance(out, dict) else {}


def reserved_context_kwargs(payload: Any) -> Dict[str, Any]:
    """The ``*_info`` constructor kwargs for this payload's reserved structs.

    Passed to ``RequestContext`` at construction, because the mixin holding
    them takes them as constructor arguments and the body may read
    ``ctx.source`` without ever reading ``ctx.source_path``.
    """
    return {
        f"{name}_info": reserved_repo_info(payload, name)
        for name in RESERVED_INFO_FIELDS
    }


async def _materialize_one(
    ctx: Any,
    payload: Any,
    field_name: str,
    snapshots: Mapping[str, Any],
) -> None:
    info = reserved_repo_info(payload, field_name)
    if not info:
        return
    raw = str(info.get("ref") or "").strip()
    if not raw:
        raise ValidationError(
            f"payload.{field_name}.ref must be a non-empty repo ref"
        )
    try:
        ref = normalize_model_ref(raw)
    except ValueError as exc:
        raise ValidationError(
            f"payload.{field_name}.ref {raw!r} is not a valid repo ref: {exc}"
        ) from exc
    setter: Optional[Callable[[str], None]] = getattr(
        ctx, _SETTER_FOR[field_name], None
    )
    if not callable(setter):
        raise ValidationError(
            f"payload.{field_name} needs a producer context; this context has "
            f"no {_SETTER_FOR[field_name]}"
        )
    snapshot = snapshots.get(ref)
    # pgw#1524 — THE JOB PLANE HAS ONE WEIGHT SOURCE TOO, and it is the CAS.
    #
    # This branch used to fall through to `ensure_local`'s provider-direct
    # download when the hub shipped no snapshot for a reserved repo. That was
    # the second direct-serve door in the tree (the first being ModelStore's),
    # and Paul's hardcut closes both: a reserved repo with no resolved snapshot
    # is a RESOLUTION defect on the hub side, not an invitation to pull
    # un-normalized bytes off an upstream registry into a producer's hands.
    #
    # THE VERDICT IS NOT RE-DERIVED HERE. `ensure_local` owns it and keeps the
    # two no-snapshot conditions distinguishable BY TYPE, which matters because
    # they need opposite operator actions: `MissingSnapshotError` says the hub
    # owes this ref a resolve (retryable there), `NonCasWeightSourceRefused`
    # says the binding names a source class that can never be served and points
    # at the ingest route. Which one fires is decided by the ref's PROVIDER,
    # from the endpoint.lock binding index — so both are reachable at this door.
    #
    # pgw#1485 / pgw#1455 kept, and now honestly: `track` is the job plane's own
    # weight_fetch instrumentation and it closes the position on every exit
    # path. The no-snapshot call below is deliberately UNinstrumented, and that
    # is no longer a hole — every path through it raises, so there are no bytes
    # for a position to describe. EVERY reserved-repo byte is on the funnel,
    # which is the claim pgw#1455 made and could not previously hold here.
    if snapshot is None:
        path = await ensure_local(
            str(ref),
            cache_dir=tensorhub_cas_dir(),
            fill_source_dir=tensorhub_fill_source_dir(),
        )
        setter(str(path))
        return
    with weight_position.track(
        str(ref), weight_position.snapshot_bytes(snapshot),
    ) as position:
        path = await ensure_local(
            str(ref),
            snapshot=snapshot,
            cache_dir=tensorhub_cas_dir(),
            fill_source_dir=tensorhub_fill_source_dir(),
            progress=position.progress,
        )
    setter(str(path))
    logger.info(
        "reserved repo materialized field=%s ref=%s path=%s",
        field_name, ref, path,
    )


async def _materialize_datasets(ctx: Any, payload: Any) -> None:
    """Reserved-datasets contract (gw#425): every ``payload.datasets`` entry
    resolved into a local dataset snapshot before the body runs. Paths land in
    ``ctx.dataset_paths``."""
    datasets = getattr(payload, "datasets", None)
    if not datasets:
        return
    resolve = getattr(ctx, "resolve_dataset", None)
    if not callable(resolve):
        raise ValidationError(
            "payload.datasets requires a producer context with resolve_dataset"
        )
    for entry in datasets:
        ref = str(getattr(entry, "ref", "") or "").strip()
        if not ref:
            raise ValidationError("payload.datasets entries need a non-empty ref")
        await asyncio.to_thread(resolve, ref)


async def materialize_reserved_inputs_async(
    ctx: Any,
    payload: Any,
    snapshots: Mapping[str, Any],
) -> None:
    """Fill every reserved path this payload declares, then its datasets."""
    for field_name in RESERVED_REPO_FIELDS:
        await _materialize_one(ctx, payload, field_name, snapshots)
        ctx.raise_if_cancelled("canceled")
    await _materialize_datasets(ctx, payload)
    ctx.raise_if_cancelled("canceled")


def materialize_reserved_inputs(
    ctx: Any,
    payload: Any,
    snapshots: Mapping[str, Any],
) -> None:
    """The SYNCHRONOUS entry the serve loop calls.

    ``ServeLoop.invoke`` runs on a worker thread (``asyncio.to_thread``) and
    has no loop of its own, so ``asyncio.run`` here is correct rather than
    lucky — the same reason ``invoke`` already drives an ``async def``
    entrypoint that way. Downloading on the dispatch's event loop instead
    would block every other job's heartbeat behind a multi-GB fetch.
    """
    asyncio.run(materialize_reserved_inputs_async(ctx, payload, snapshots))


def declared_reserved_fields(payload: Any) -> Tuple[str, ...]:
    """Which reserved repo fields this payload actually names — for logs and
    for the tests that assert the population boundary."""
    return tuple(
        name for name in RESERVED_REPO_FIELDS
        if reserved_repo_info(payload, name)
    )


__all__ = [
    "RESERVED_INFO_FIELDS",
    "RESERVED_REPO_FIELDS",
    "declared_reserved_fields",
    "materialize_reserved_inputs",
    "materialize_reserved_inputs_async",
    "reserved_context_kwargs",
    "reserved_repo_info",
]
