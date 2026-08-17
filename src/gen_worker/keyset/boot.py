"""Serve-boot consumption: a key set WITHOUT owning a tracer.

The whole of pgw#1327's serve half. Address the document by this pod's closure
digest, parse it, fold the shipped graph hashes with this pod's own ``sm`` and
toolchain, and hand ``boot_adopt`` the same :class:`DerivedKeySet` a trace would
have produced. No ``torch.export``, no child process, no endpoint model code
imported, and — the structural half — nothing on this path imports ``boot_key``
or ``boot_trace_child``.

Time cost: the closure digest (source hashes plus ``compile_cache``'s cached
version probes) and one JSON parse. What it replaces was measured by the module
it replaces: *"< 60 s, every time, ~99 % of it the traces."*
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..child_contract import CompileSpec, MintSlot
from . import store
from .closure import closure_digest
from .fold import DerivedKeySet, KeySource, MemoVerdict, fold_entry_keys
from .hub import HubTier, fetch_closure
from .identifiers import ClosureDigest, KeySetError

logger = logging.getLogger(__name__)

__all__ = ["key_set_from_data", "closure_of"]


def closure_of(
    *,
    family: str,
    function: str,
    modules: Sequence[str],
    cfg: CompileSpec,
    slots: Mapping[str, MintSlot],
) -> ClosureDigest:
    """This boot's closure digest — the address, and nothing else."""
    return closure_digest(
        family, cfg, function=function, slots=slots, modules=modules)


def key_set_from_data(
    *,
    family: str,
    function: str,
    modules: Sequence[str],
    cfg: CompileSpec,
    slots: Mapping[str, MintSlot],
    cache_dir: Optional[Path] = None,
    extra_roots: Sequence[Path] = (),
    hub: Optional[HubTier] = None,
) -> DerivedKeySet:
    """Derive this boot's ``cg-key-v1`` set from SHIPPED DATA.

    Raises :class:`KeySetError` with ``reason='keyset_absent'`` when no root
    holds this closure — which is a complete answer, not a failure: the pod
    either hands the miss to a mint-lane deriver (an ordinary Python serving
    pod, §4.28) or, on an adopt-only role, refuses and serves as it booted
    before. It is never a boot that guesses a key.

    ``reason='keyset_invalid'`` means a LOCAL document was found and is
    malformed or a version this worker does not read. That propagates rather
    than degrading to "absent" on purpose: a broken shipped key set is a
    mint-lane defect that must be visible, and reading past it is how every pod
    in a release silently re-traces forever.

    A HUB answer is held to the same admission and refused the same way, but a
    refusal there is a MISS rather than a propagating error (see
    :mod:`gen_worker.keyset.hub`): a local document is an artifact this image or
    this endpoint's storage is responsible for, while the hub is a network peer
    on the boot path, and a pod that refuses to boot because a cache was
    unreachable is strictly worse than the 805 s it was avoiding.
    """
    started = time.monotonic()
    digest = closure_of(
        family=family, function=function, modules=modules, cfg=cfg, slots=slots)
    hit = store.lookup(digest, cache_dir=cache_dir, extra_roots=extra_roots)
    closure = hit.closure if hit is not None else None
    source = hit.source if hit is not None else KeySource.HUB
    where = str(hit.path) if hit is not None else ""
    hub_reason = ""
    if closure is None and hub is not None:
        # ── THE HUB TIER (pgw#1353 option (b) / th#2123) ────────────────────
        # Asked LAST among the readers and only when every local root missed: a
        # local document is already on this machine, so asking the network
        # first would put a round trip in front of an answer the pod is already
        # holding. Asked BEFORE the deriver for the obvious reason — a 10 s
        # timeout against 805 s of `torch.export`.
        #
        # `fetch_closure` never raises: a hub that is down, slow, or answering
        # nonsense degrades to the derive this pod would have run anyway, and
        # the reason rides the `keyset_absent` detail below so the degradation
        # is readable instead of silent.
        closure, hub_reason = fetch_closure(digest, hub)
        where = f"hub {hub.base_url or 'via broker'}"
    if closure is None:
        searched = [str(root) for root in store.shipped_roots(extra=extra_roots)]
        searched.extend(
            str(root) for root, _source in store.writable_roots(cache_dir))
        searched.append(
            f"hub ({hub_reason})" if hub is not None else "hub (no tier configured)")
        raise KeySetError(
            "keyset_absent",
            f"no cg-keyset-v1 document holds closure {digest} for family "
            f"{family!r}; roots searched: {searched}")
    entry_keys = fold_entry_keys(closure.class_hashes, family=family)
    wall_ms = int((time.monotonic() - started) * 1000)
    logger.info(
        "keyset: %d key(s) for %s from %s data at %s in %d ms — no trace "
        "(closure %s, emitted by %s)",
        len(entry_keys), family, source.value, where, wall_ms, digest,
        closure.emitted_by or "?")
    return DerivedKeySet(
        entry_keys=entry_keys,
        source=source,
        closure=digest,
        wall_ms=wall_ms,
        memo=(
            MemoVerdict.HIT if source is KeySource.MEMO
            else MemoVerdict.ABSENT),
        width_reason=f"{source.value} key set at {where} — no trace child",
    )
