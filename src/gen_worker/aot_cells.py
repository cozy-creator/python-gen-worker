"""AOT cell discovery — the pgw#722 pilot's F1 seam (flag-gated, default OFF).

A serving pod cannot COMPUTE an exported cell's key: an ``aot-inductor``
cell is STAMPED at mint with axes only the mint holds (its traced-graph
contract), so the runtime's ``cell_key.compute`` (kind="inductor") can
never name it and the hub's worker-owned pull-by-key delivery (th#883)
never lights it up. Published AOT cells are therefore provably dark.

This module is the minimal pilot delivery: **fetch-and-filter**. At arm
time (``fleet_cells.enable_compiled``, behind ``Settings.
compile_prefer_aot``) the worker lists the family cell repo's checkpoints
through the hub's existing catalog read API, filters for a cell THIS
runtime can serve, downloads the artifact, and feeds it into the existing
``provision.enable_compiled`` HIT path — where the pgw#709 receipt gate
and the aot_serve arm gates (B1/B2, lifted contract) run unchanged.

Deliberately NOT the advertised-key route (extending ``cell_lookups`` /
``requested_cell_key``): two resolving keys (inductor + aot) for one
function make the hub's ``requestedKeyCellLocked`` ambiguous and
fail-close mandatory-lane dispatch. That route is #724's design work and
needs a hub tie-break first; the runbook records the hazard.

Filter (SDXL-AOT-PILOT-RUNBOOK.md §3 F1):

* ``kind == aot-inductor`` with a stamped ck5 ``cell_key``;
* runtime-key + contract check via ``aot_serve.verify`` (sm, torch,
  cuda, family, code-only flag, contract/constants parse);
* canonical lane/bucket match against THIS pipeline's traced lane;
* preference: same SKU first (autotune affinity), then a stamped ``sm``,
  then newest (checkpoint ``updated_at``; key digest tie-break). SKU is a
  preference and never a filter (pgw#765) — an l4-minted cell is adopted on
  an rtx-4090 because both are sm_89.

Receipt verification is NOT duplicated here — the downloaded artifact
flows through ``receipts.gate_delivered_artifact`` at the one arming
choke point exactly like a hub-attached cell.

Every miss is fail-soft: discovery returning ``None`` leaves the boot on
today's path (delivered dynamo cell / self-mint), and the reason rides a
typed ``aot_cell_discovery`` activity event, never only pod logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from . import activity as activity_mod
from . import aot_serve, cell_key
from . import boot_phases as boot_mod
from . import compile_cache as cc
from .procsplit import broker
from . import config
from .config import Settings
from .models.chunk_cas import sha256_file
from .models.chunk_cas import (
    CAS_CHUNK_SIZE_BYTES,
    ChunkSpec,
    DigestMismatch,
    download_chunked_file,
    verify_file_digest,
)
from .models.hub_client import parse_chunk_list, resolved_entry_digest

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_S = 30
_DOWNLOAD_TIMEOUT_S = 120
_LIST_LIMIT = 200

EVENT = "aot_cell_discovery"

# th#1335: statuses that mean "you may not look", as opposed to "there is
# nothing there". 403 is what the hub now answers a worker credential with no
# read grant on a platform cell repo; 401 covers a credential the hub could not
# verify at all.
_NOT_AUTHORIZED = (401, 403)


def prefer_aot(settings: Optional[Settings] = None) -> bool:
    """The pilot flip switch (typed pod-launch knob, default OFF).

    Takes the `Settings` when the caller has them; otherwise reads the ones the
    process entry installed (§1.18 — never the environment).
    """
    return bool((settings or config.current()).compile_prefer_aot)


@dataclass(frozen=True)
class AdoptedAotCell:
    """Identity of one discovered, downloaded, runtime-matched AOT cell.

    Duck-compatible with ``fleet_cells.SelfMint`` where the executor reads
    it (``_selection_for``): ``ref``/``snapshot_digest``/``artifact``. The
    advertised digest is the artifact tar's sha256 — the worker's own view
    of the exact bytes it armed, receipt-verified downstream — so the
    advertisement never names bytes this pipe does not serve.
    """

    family: str
    cell_key: str
    ref: str  # "root/family-<f>#<stamped ck5 key>"
    snapshot_digest: str  # "sha256:<hex>" of the artifact tarball
    artifact: Path


def _emit(phase: str, detail: str) -> None:
    activity_mod.emit_event(EVENT, detail, phase=phase)


def _get(
    base_url: str,
    path: str,
    bearer: str,
    *,
    params: Optional[Dict[str, str]] = None,
    timeout: float = _HTTP_TIMEOUT_S,
) -> "broker.HubResponse":
    """GET with the worker bearer.

    th#1335: the anonymous retry on 401/403 is GONE. It existed because the
    family cell repos used to be public, but th#1310 made them private and
    recorded the retry as the hazard that would turn one visibility flip into
    unauthenticated cross-tenant enumeration. The worker JWT now carries a
    hub-issued ``read_repo`` grant for exactly the families its release
    declares, so a refusal is a real authorization answer and must be reported
    as one — never laundered into an anonymous 404.

    pgw#763 delta 1: parent-mediated under the split (the compute child holds
    no worker JWT); the identical GET otherwise. th#1335 deleted the only
    unauthenticated leg, so EVERY call from here carries worker identity and
    rides the authorization surface — the two changes compose.
    """
    return broker.request(
        "GET", path, base_url=base_url, bearer=bearer, params=params, timeout=timeout)


def _execution_lane_of_meta(meta: Dict[str, Any]) -> str:
    return cell_key._canonical_execution_lane(
        str(meta.get("weight_lane") or ""), int(meta.get("lora_bucket") or 0))


def _candidates(
    items: List[Dict[str, Any]], family: str, want_execution_lane: str,
    rejected: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """(updated_at, checkpoint_id, metadata) rows that pass the filter, best
    candidate first.

    The filter is the REAL identity (``aot_serve.verify``: sm/torch/cuda,
    host ISA, family, contract) — never ``sku`` (pgw#765). SKU is the
    SELECTION PREFERENCE below: a same-SKU cell was autotuned on this exact
    board, so it is preferred when one exists, and a same-sm cell from
    another SKU is adopted rather than refused when one does not.
    """
    # pgw#824: every rejection is COUNTED by class. Each of these filters was
    # a `logger.debug` or a bare `continue`, and the caller's `miss` event then
    # said only "no matching cell among N checkpoint(s)" — so a family with 12
    # published cells that rejects all 12 was indistinguishable from a family
    # with none, and the two have completely different fixes. Coalescing with
    # counts is the pattern; dropping is not.
    def _reject(cls: str) -> None:
        if rejected is not None:
            rejected[cls] = rejected.get(cls, 0) + 1

    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata")
        if not isinstance(meta, dict):
            continue
        if str(meta.get("kind") or "") != aot_serve.ARTIFACT_KIND:
            _reject("not_an_aot_cell")
            continue
        key = str(meta.get("cell_key") or "").strip()
        if not cell_key.is_key(key):
            _reject("unreadable_cell_key")
            continue
        # Pre-clamp retirement (arm-events lane, live-proven 2026-07-29):
        # a cell carrying no host-ISA stamp states no requirement, so its true
        # ISA need is undiscoverable from metadata and an AVX-512-built .pt2
        # SIGILLs a non-AVX-512 host. ``aot_serve.verify`` refuses it too;
        # ruling here first keeps the fact its OWN reject class rather than
        # folding it into a ``verify:`` string.
        if not isinstance(meta.get("host_isa"), dict):
            logger.debug("aot-cells: %s filtered: no host_isa stamp", key)
            _reject(aot_serve.NO_HOST_ISA_STAMP)
            continue
        reason = aot_serve.verify(meta, family=family)
        if reason:
            logger.debug("aot-cells: %s filtered: %s", key, reason)
            # The verify reason is already a classified token — keep it, so a
            # fleet-wide sm/torch/ISA/contract mismatch surfaces as itself
            # rather than as a generic "verify failed".
            _reject(f"verify:{reason}")
            continue
        if _execution_lane_of_meta(meta) != want_execution_lane:
            _reject("lane_mismatch")
            continue
        out.append((
            str(item.get("updated_at") or ""),
            str(item.get("checkpoint_id") or "").strip(),
            meta,
        ))
    # Preference order (pgw#765), all descending:
    #   1. same SKU — the cell's inductor autotuning ran on this board;
    #   2. newest (checkpoint ``updated_at``), key digest as the
    #      deterministic tie-break.
    # (``sm`` is no longer a tie-break: ``verify`` refuses a cell silent on
    # the axis, so every survivor here is stamped and matching.)
    here_sku = aot_serve.runtime_key()["sku"]
    out.sort(
        key=lambda row: (
            str(row[2].get("sku") or "") == here_sku and bool(here_sku),
            row[0],
            str(row[2].get("cell_key")),
        ),
        reverse=True)
    return out


def _download_artifact(
    base_url: str,
    bearer: str,
    family: str,
    checkpoint_id: str,
    key: str,
    cache_dir: Optional[Path],
) -> Optional[Path]:
    """Resolve the checkpoint's manifest and fetch its artifact tarball,
    digest-verified against the manifest entry. Cached by cell key.

    pgw#764: bracketed as the ``cell_fetch`` boot phase. Whether this was a
    cold pull or a cache hit — and how many bytes it cost — is the difference
    between a fast boot and a slow one, and nothing measured it before.
    """
    with boot_mod.span(
        boot_mod.PHASE_CELL_FETCH,
        artifact_kind=aot_serve.ARTIFACT_KIND,
        artifact_key=key,
    ) as fetch:
        dest = _download_artifact_inner(
            base_url, bearer, family, checkpoint_id, key, cache_dir, fetch)
        if dest is None:
            # A miss is a typed decline, not an exception: the pilot lane
            # self-mints instead. Reason tokens already rode _emit above.
            fetch.refused("fetch_miss")
        return dest


def _download_artifact_inner(
    base_url: str,
    bearer: str,
    family: str,
    checkpoint_id: str,
    key: str,
    cache_dir: Optional[Path],
    fetch: "boot_mod.BootSpan",
) -> Optional[Path]:
    repo = cc.system_repo(family)
    dest_dir = (
        Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gen-worker"
    ) / "aot-cells"
    dest = dest_dir / f"{key}.tar.gz"

    resp = _get(
        base_url, f"/api/v1/repos/{repo}/resolve",
        bearer, params={"digest": checkpoint_id})
    if resp.status_code in _NOT_AUTHORIZED:
        _emit("not_authorized",
              f"family={family} key={key}: resolve -> {resp.status_code} "
              f"(worker credential carries no read grant for {repo})")
        return None
    if resp.status_code != 200:
        _emit("resolve_failed",
              f"family={family} key={key}: resolve -> {resp.status_code}")
        return None
    body = resp.json() if resp.text else {}
    entry = next(
        (f for f in (body.get("files") or [])
         if isinstance(f, dict) and str(f.get("path") or "").endswith(".tar.gz")),
        None)
    if entry is None:
        _emit("resolve_failed",
              f"family={family} key={key}: manifest has no artifact tarball")
        return None
    # th#1303: read the ALGORITHM-TAGGED digest, and REFUSE when there is
    # none. The old code read `entry["blake3"]` — absent on every manifest v2
    # entry — and then guarded both integrity checks on its truthiness, so a
    # v2 cell was installed and executed with no verification at all while
    # reporting success. A cell artifact is compiled code: an unreadable
    # digest must fail closed, never degrade into "no check".
    try:
        want_ref = resolved_entry_digest(
            entry, what=f"family={family} key={key}: manifest entry")
    except ValueError as exc:
        _emit("resolve_failed", str(exc))
        return None

    if dest.is_file():
        try:
            verify_file_digest(dest, want_ref)
        except DigestMismatch:
            dest.unlink(missing_ok=True)  # stale cache — refetch below
        except ValueError as exc:
            _emit("resolve_failed", f"family={family} key={key}: {exc}")
            return None
        else:
            fetch.bytes_moved(dest.stat().st_size, boot_mod.SOURCE_LOCAL)
            return dest

    try:
        chunks = parse_chunk_list(
            f"family={family} key={key}", str(entry.get("path") or ""),
            entry.get("chunks"), entry.get("chunk_urls"))
    except Exception as exc:  # noqa: BLE001 — a malformed list is a typed miss
        _emit("resolve_failed",
              f"family={family} key={key}: unreadable chunk list: {exc}")
        return None
    url = str(entry.get("url") or "")
    if not chunks and not url:
        # A chunked entry has no whole-file URL; a whole entry has no chunks.
        _emit("resolve_failed",
              f"family={family} key={key}: manifest entry has no URL")
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        if chunks:
            # Every chunk's sha256 AND the whole-file digest are verified
            # inside the commit stream — one pass, no second read.
            download_chunked_file(
                [ChunkSpec(sha256=c.sha256, url=c.url, length=c.length)
                 for c in chunks],
                tmp,
                whole_digest=want_ref,
                total_size=int(entry.get("size_bytes") or 0),
                chunk_size_bytes=(
                    int(entry.get("chunk_size_bytes") or 0)
                    or CAS_CHUNK_SIZE_BYTES),
            )
        else:
            with requests.get(url, timeout=_DOWNLOAD_TIMEOUT_S, stream=True) as dl:
                dl.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in dl.iter_content(1 << 20):
                        f.write(chunk)
            verify_file_digest(tmp, want_ref)
    except (ValueError, RuntimeError) as exc:
        tmp.unlink(missing_ok=True)
        _emit("digest_mismatch",
              f"family={family} key={key}: downloaded bytes refused "
              f"against {want_ref[:20]}…: {exc}")
        return None
    tmp.replace(dest)
    fetch.bytes_moved(dest.stat().st_size, boot_mod.SOURCE_R2)
    return dest


def discover(
    pipe: Any,
    cfg: Any,
    *,
    base_url: str,
    worker_jwt: Callable[[], str],
    cache_dir: Optional[Path] = None,
) -> Optional[AdoptedAotCell]:
    """Fetch-and-filter one servable AOT cell for this pipeline, or None.

    Never raises: every failure is a MISS with a typed event — a broken
    discovery must cost the pilot lane, never the boot.

    pgw#764: bracketed as the ``cell_discover`` boot phase, so the hub can see
    what asking the catalog cost even when the answer was "no cell".
    """
    with boot_mod.span(boot_mod.PHASE_CELL_DISCOVER) as found:
        cell = _discover_inner(
            pipe, cfg, base_url=base_url, worker_jwt=worker_jwt,
            cache_dir=cache_dir)
        if cell is None:
            found.refused("no_cell")
        else:
            found.note(f"family={cell.family} key={cell.cell_key}")
        return cell


def _discover_inner(
    pipe: Any,
    cfg: Any,
    *,
    base_url: str,
    worker_jwt: Callable[[], str],
    cache_dir: Optional[Path] = None,
) -> Optional[AdoptedAotCell]:
    family = str(getattr(cfg, "family", "") or "")
    if not family:
        return None
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return None
    bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
    try:
        want_execution_lane = cell_key._canonical_execution_lane(cc.cell_base_execution_lane(pipe), bucket)
        repo = cc.system_repo(family)
        bearer = str(worker_jwt() or "").strip()
        resp = _get(
            base, f"/api/v1/repos/{repo}/checkpoints",
            bearer, params={"limit": str(_LIST_LIMIT)})
        if resp.status_code in _NOT_AUTHORIZED:
            # th#1335/th#1230: an authorization refusal is NOT an empty family.
            # It is a config defect — this release's worker credential carries
            # no read grant for its own family's cell repo — and it must be
            # nameable as one, because the fail-soft path (self-mint) looks
            # identical from the outside either way.
            _emit("not_authorized",
                  f"family={family}: checkpoint listing -> {resp.status_code} "
                  f"(worker credential carries no read grant for {repo})")
            return None
        if resp.status_code != 200:
            _emit("list_failed",
                  f"family={family}: checkpoint listing -> {resp.status_code}")
            return None
        items = (resp.json() or {}).get("items") or []
        rejected: Dict[str, int] = {}
        rows = _candidates(items, family, want_execution_lane, rejected)
        if not rows:
            # pgw#824: WHY every candidate lost, as counts. "No matching cell
            # among 12 checkpoints" was true and useless — it could not tell an
            # empty family from a family whose every cell is one sm short, and
            # those are different bugs with different owners.
            histogram = ", ".join(
                f"{cls}={n}" for cls, n in sorted(
                    rejected.items(), key=lambda kv: (-kv[1], kv[0])))
            _emit("miss",
                  f"family={family} lane={want_execution_lane or 'plain'}: no matching "
                  f"aot-inductor cell among {len(items)} checkpoint(s)"
                  + (f" — rejected by class: {histogram}" if histogram
                     else " (the family has no published cells at all)"))
            return None
        _updated, checkpoint_id, meta = rows[0]
        key = str(meta.get("cell_key") or "")
        if not checkpoint_id:
            _emit("resolve_failed",
                  f"family={family} key={key}: listing row has no checkpoint id")
            return None
        artifact = _download_artifact(
            base, bearer, family, checkpoint_id, key, cache_dir)
        if artifact is None:
            return None
        # From this moment the executor's kind dispatch (#734/#735) must
        # recognize the ck5-flavored ref as an exported cell.
        aot_serve.note_aot_key(key)
        adopted = AdoptedAotCell(
            family=family,
            cell_key=key,
            ref=f"{cc.system_repo(family)}#{key}",
            snapshot_digest="sha256:" + sha256_file(artifact),
            artifact=artifact,
        )
        _emit("hit",
              f"family={family} key={key} checkpoint={checkpoint_id[:16]} "
              f"lane={want_execution_lane or 'plain'}")
        logger.info(
            "aot-cells: discovered %s (checkpoint %s, %.1f MB)",
            adopted.ref, checkpoint_id[:16],
            artifact.stat().st_size / 1e6)
        return adopted
    except Exception as exc:  # noqa: BLE001 — discovery is never fatal
        logger.warning("aot-cells: discovery failed (%s); staying on the "
                       "ordinary arming policy", exc)
        _emit("error", f"family={family}: {type(exc).__name__}: {exc}")
        return None


__all__ = [
    "AdoptedAotCell",
    "EVENT",
    "discover",
    "prefer_aot",
]
