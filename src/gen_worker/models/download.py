"""ONE ensure-local path for the serving fleet, and it has ONE weight source.

``ensure_local(ref)`` materializes a model ref on disk and returns its local
path. Since pgw#1524 (Paul's 2026-08-19 hardcut ruling — *"only store + support
loading our new tensorfs laid out files"*) the only thing it can materialize is
a **tensor-layout-contract-cut tensorfs CAS snapshot**: an orchestrator-resolved
manifest projected by ``models/cozy_snapshot.py``. Every other weight source is
an INGEST edge and is refused here by name
(:class:`~gen_worker.models.errors.NonCasWeightSourceRefused`).

What that means per provider tag, which still rides bindings because it records
where a model CAME FROM:

  - tensorhub : the snapshot the orchestrator resolved -> CAS -> projected tree.
                A tensorhub ref with no snapshot is ``MissingSnapshotError``
                (the orchestrator re-mints), not a fallback.
  - hf / civitai / modelscope : servable ONLY through a hub-resolved snapshot
                (mirror-first mirrors the upstream ref into the platform CAS).
                With no snapshot there is nothing to serve and the refusal names
                the ingest route.

The fetchers themselves did not all die with the direct-serve branches: the
INGEST edges live in ``gen_worker.convert`` (``ingest_huggingface`` has its own
bounded ``snapshot_download``; ``ingest_civitai`` calls :func:`download_civitai`
below), and they feed normalization -> contract -> CAS -> serve.

One progress-reporter shape everywhere: ``progress(bytes_done, bytes_total)``
(total may be None). Blocking library calls always run off the event loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlparse

from ..api.errors import ValidationError
from ..bounded_stream import copy_bounded, free_space_bound
from ..config import Settings
from ..manifest_blocks import declaration_rows

_STANDALONE = Settings()
from ..stall import ProgressFloor, SilenceWindow
from .cache_paths import tensorhub_cas_dir
from .errors import PickleWeightRefused, first_pickle_weight_path, non_cas_refusal
from .refs import HuggingFaceRef, TensorhubRef, fold_ref, parse_model_ref
import hashlib

if TYPE_CHECKING:
    from .hub_client import WorkerResolvedRepo

logger = logging.getLogger("gen_worker.download")

ProgressFn = Callable[[int, Optional[int]], None]

# ---------------------------------------------------------------------------
# Provider index: normal-form ref -> provider. Built once at boot from the
# endpoint.lock manifest (the wire carries bare refs without a provider field).
#
# ONE keying function normalizes both index keys and lookups. Keys are
# (repo, release)-granular, with a repo-identity fallback so a hub-minted DIGEST
# pick still routes to its repo's provider.
# ---------------------------------------------------------------------------

_provider_by_ref: Mapping[str, str] = {}


def _provider_index_keys(ref: str) -> tuple[str, str]:
    """THE keying function for the provider index: ``(exact, base)`` where
    ``exact`` is the ref's normal form without digest/revision (they never
    appear on manifest binding entries) and ``base`` is the repo identity
    (``owner/repo``). Tries the tensorhub grammar first, then the HF form
    (which allows a non-digest ``@revision``); refs outside both key as
    their stripped raw string (e.g. civitai numeric ids)."""
    s = str(ref or "").strip()
    if not s:
        return "", ""
    try:
        parsed = parse_model_ref(s)
    except ValueError:
        try:
            parsed = parse_model_ref(s, provider="hf")
        except ValueError:
            return s, s
    if parsed.tensorhub is not None:
        th = parsed.tensorhub
        exact = TensorhubRef(owner=th.owner, repo=th.repo, release=th.release,
                             digest=None, fragment=th.fragment).canonical()
        return exact, th.repo_id()
    assert parsed.hf is not None
    hf = parsed.hf
    exact = HuggingFaceRef(repo_id=hf.repo_id, revision=None).canonical()
    return exact, hf.repo_id


def set_provider_index(mapping: Optional[Mapping[str, str]]) -> None:
    global _provider_by_ref
    index: dict[str, str] = {}
    for k, v in (mapping or {}).items():
        exact, base = _provider_index_keys(k)
        if not exact:
            continue
        index.setdefault(exact, v)
        index.setdefault(base, v)
    _provider_by_ref = index


def lookup_provider_for_ref(ref: str, *, default: str = "tensorhub") -> str:
    """Provider tag for ``ref`` from the index: exact normal-form match,
    then the repo-identity fallback."""
    if not ref:
        return default
    mapping = _provider_by_ref
    if not mapping:
        return default
    exact, base = _provider_index_keys(ref)
    hit = mapping.get(exact)
    if hit is None:
        hit = mapping.get(base)
    return hit if hit is not None else default


def _collect_binding_entries(bindings: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(bindings, dict):
        return out
    for entry in bindings.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("ref"):
            out.append(entry)
    return out


def build_provider_index_from_manifest(manifest: Optional[Mapping[str, Any]]) -> dict[str, str]:
    """{normal_form_ref: provider} from a loaded endpoint.lock manifest.

    th#1987: the release rides the ref, so there is no side-channel field to
    fold — the entry's ref is normalized through the ONE grammar module before
    keying and ``set_provider_index`` adds the repo-identity fallback keys.

    pgw#1395: reads EVERY declaration block, not ``functions[]`` alone. A
    binding on a v2 ``entrypoints[]`` row would otherwise index to nothing and
    every ref in it would silently fall back to the default provider."""
    index: dict[str, str] = {}
    if not isinstance(manifest, Mapping):
        return index
    for fn in declaration_rows(manifest):
        for entry in _collect_binding_entries(fn.get("bindings")):
            ref = str(entry.get("ref") or "").strip()
            if not ref:
                continue
            provider = str(entry.get("provider") or "").strip() or "tensorhub"
            try:
                key = str(fold_ref(
                    ref,
                    provider=provider
                    if provider in ("tensorhub", "hf", "civitai", "modelscope")
                    else "tensorhub",
                ))
            except ValueError:
                key = ref
            index.setdefault(key, provider)
    return index


# ---------------------------------------------------------------------------
# ensure_local: the ONE entry point
# ---------------------------------------------------------------------------


def _snapshot_ref(parsed: Any, raw: str) -> TensorhubRef:
    """Ref identity for a snapshot download. Snapshot trees are addressed by
    digest, so for non-tensorhub providers (mirror-first refs) this only names
    the download in logs."""
    if parsed.tensorhub is not None:
        return parsed.tensorhub
    if parsed.hf is not None:
        owner, _, repo = parsed.hf.repo_id.partition("/")
        return TensorhubRef(owner=owner, repo=repo)
    if parsed.civitai is not None:
        return TensorhubRef(owner="civitai", repo=parsed.civitai.model_id)
    if parsed.modelscope is not None:
        owner, _, repo = parsed.modelscope.repo_id.partition("/")
        return TensorhubRef(owner=owner, repo=repo)
    owner, _, repo = raw.partition("/")
    return TensorhubRef(owner=owner or "unknown", repo=repo or raw)


async def ensure_local(
    ref: str,
    *,
    provider: Optional[str] = None,
    snapshot: Optional["WorkerResolvedRepo"] = None,
    cache_dir: Optional[Path] = None,
    progress: Optional[ProgressFn] = None,
    fill_source_dir: Optional[Path] = None,
) -> Path:
    """Materialize ``ref`` on disk; return its local path.

    ``snapshot`` is the orchestrator-resolved manifest (the typed
    ``WorkerResolvedRepo``) carrying presigned URLs or transfer grants. The
    orchestrator is the only resolver: when it ships a snapshot for a ref —
    including an hf/civitai binding ref resolved through a platform mirror
    under mirror-first — the snapshot is authoritative and the bytes come from
    tensorhub-CAS, never the upstream registry.

    **pgw#1524: a ref with no snapshot has nothing to serve.** A tensorhub ref
    raises ``MissingSnapshotError`` (retryable at the orchestrator, which
    re-mints); every other provider raises
    :class:`~gen_worker.models.errors.NonCasWeightSourceRefused`, which is
    terminal and names the ingest route. The direct-download branches
    (``download_hf``, ``download_modelscope``, and the civitai stream) are
    DELETED, not flagged off: a fallback that serves un-normalized upstream
    bytes is exactly what the hardcut removes.

    **The registry-shaped parameters are GONE with the branches that read
    them** (``hf_home``, ``hf_token``, ``civitai_api_key``, ``allow_patterns``,
    ``components``). Only the deleted direct-download branches ever consumed
    them: file selection on the CAS branch is the ORCHESTRATOR's, because the
    residency layer digest-verifies the materialized tree against the FULL
    ``snapshot.files`` list it was handed (``ModelStore._verify_snapshot_tree``)
    — filtering here without the orchestrator also filtering what it verifies
    turns every boot into a spurious corruption/quarantine loop. Selective
    fetch for tensorhub refs is the hub's desired-snapshot scoping, and on the
    hub-less CLI path ``models/provision.py::_fetch_tensorhub_snapshot``, which
    owns its own resolve+download+materialize loop end to end. Keeping the
    parameters as accepted-and-ignored would be the "silently wrong answer"
    failure this repo keeps paying for.

    ``fill_source_dir``: an endpoint-scoped datacenter-warm CAS mount (RunPod
    volume) consulted before R2 on the tensorhub-snapshot branch only.
    ``None`` (the default, and always true for cozy-local / non-tensorhub
    providers) goes straight to R2.
    """
    base = Path(cache_dir) if cache_dir is not None else tensorhub_cas_dir()
    prov = provider or lookup_provider_for_ref(ref)
    parsed = parse_model_ref(ref, provider=prov)

    if snapshot is not None:
        from .cozy_snapshot import ensure_snapshot_async

        return await ensure_snapshot_async(
            base_dir=base,
            ref=_snapshot_ref(parsed, ref),
            resolved=snapshot,
            progress=progress,
            fill_source_dir=fill_source_dir,
        )

    if parsed.provider == "tensorhub" and parsed.tensorhub is not None:
        # The worker cannot resolve tensorhub-CAS refs itself:
        # typed + terminal so callers fail fast with "missing_snapshot"
        # instead of burning retries on a deterministic local condition.
        from .errors import MissingSnapshotError

        raise MissingSnapshotError(
            f"tensorhub ref {ref!r} needs an orchestrator-resolved snapshot "
            "and none was provided"
        )

    # THE HARDCUT (pgw#1524). Every remaining provider is an INGEST source, so
    # the only honest answer to "serve this without a CAS snapshot" is a typed
    # refusal that names the ingest route. Deliberately NOT a per-provider
    # branch: a source class this function has never heard of is refused by the
    # same rule, because the rule is about the CAS, not about the registry.
    if parsed.provider in ("hf", "civitai", "modelscope"):
        raise non_cas_refusal(ref=str(ref), provider=parsed.provider)

    # Typed so the executor classifies it INVALID (bad input, never retry) —
    # a bare ValueError maps FATAL.
    raise ValidationError(f"unsupported model ref {ref!r} (provider={prov!r})")


def select_component_paths(
    paths: Sequence[str],
    components: Sequence[str],
) -> set[str]:
    """Narrow a repo file listing to declared pipeline COMPONENTS:
    every path under a ``<component>/`` subfolder, plus every root-level
    ``*.json`` (``model_index.json`` and siblings — always kept so
    downstream component-set introspection / pipeline-class detection still
    works off the narrowed tree). Empty ``components`` returns every path
    unchanged (whole-repo, today's default). Shared by the HF downloader and
    the tensorhub CAS snapshot downloader (``cozy_snapshot.py``) — the ONE
    filter both sources apply.

    Positive selection only: th#1941 deleted the negative ``exclude`` arm
    with the override-on-base composition it compensated for. A hub-composed
    manifest already IS the file list to fetch.
    """
    comps = {c.strip() for c in components if c and str(c).strip()}
    if not comps:
        return set(paths)
    keep: set[str] = set()
    for p in paths:
        if not p:
            continue
        if "/" not in p:
            if p.lower().endswith(".json"):
                keep.add(p)
            continue
        if p.split("/", 1)[0] in comps:
            keep.add(p)
    return keep


def components_present(paths: Sequence[str], components: Sequence[str]) -> bool:
    """Whether every declared component names an ACTUAL subfolder in
    ``paths``. Root config files are always kept by
    :func:`select_component_paths` regardless of ``components`` — so a
    typo'd component name can't be detected by an empty result alone (the
    root jsons keep the selection non-empty); this checks the component
    NAMES themselves matched something."""
    comps = {c.strip() for c in components if c and str(c).strip()}
    if not comps:
        return True
    dirs = {p.split("/", 1)[0] for p in paths if p and "/" in p}
    return bool(dirs & comps)


_HF_DOWNLOAD_STALL_TIMEOUT_S = 180.0
_HF_DOWNLOAD_MIN_WINDOW_BYTES = 8 * 1024 * 1024

class DownloadStalledError(RuntimeError):
    """Raised when a blocking snapshot download fails the progress-rate floor
    (less than ``min_window_bytes`` of new bytes within the stall window) — a
    bounded, observable failure instead of a silent hang."""


def _scan_bytes(root: Path) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    try:
        for dirpath, _dirs, names in os.walk(root):
            for name in names:
                try:
                    st = os.stat(os.path.join(dirpath, name))
                except OSError:
                    continue
                key = (int(st.st_dev), int(st.st_ino))
                if key in seen:
                    continue
                seen.add(key)
                total += int(st.st_size)
    except OSError:
        return 0
    return total


def _run_with_stall_watchdog(
    download_fn: Callable[[], str],
    *,
    label: str,
    progress_root: Optional[Path],
    progress_callback: Optional[ProgressFn],
    total_hint: Optional[int],
    stall_timeout: float,
    min_window_bytes: int = _HF_DOWNLOAD_MIN_WINDOW_BYTES,
    scan_bytes: Callable[[Path], int] = _scan_bytes,
    poll_interval: float = 0.5,
) -> str:
    """Run a blocking download on a daemon thread; the watchdog doubles as the
    progress reporter (scans bytes-on-disk under ``progress_root``) and raises
    :class:`DownloadStalledError` when the transfer falls below the progress
    floor: fewer than ``min_window_bytes`` new bytes within ``stall_timeout``
    (a trickle is a stall, and there is no wall-clock cap)."""
    holder: Dict[str, Any] = {}

    def _run() -> None:
        try:
            holder["local"] = download_fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller thread
            holder["exc"] = exc
        finally:
            holder["done"] = True

    dl_thread = threading.Thread(target=_run, name="model-download", daemon=True)
    dl_thread.start()

    last_bytes = 0
    # The progress window as two shared values: the floor decides
    # what counts as an advance (a trickle never does), the window decides how
    # long an unadvanced loop may run. The same pair guards the CAS fetch.
    floor = ProgressFloor(max(int(min_window_bytes), 1))
    # An unset limit is a REFUSAL, never an emergent one: never widen this to
    # `stall_timeout if stall_timeout > 0 else math.inf`, which defeats
    # SilenceWindow's own `window_s must be positive` guard and lets a zero
    # silently delete the watchdog. Let it refuse.
    window = SilenceWindow(stall_timeout)
    while not holder.get("done"):
        dl_thread.join(timeout=poll_interval)
        if holder.get("done"):
            break
        if progress_root is not None:
            try:
                seen = scan_bytes(progress_root)
            except Exception:
                seen = last_bytes
            if seen > last_bytes:
                last_bytes = seen
                if progress_callback is not None:
                    try:
                        progress_callback(seen, total_hint)
                    except Exception:
                        pass
            if floor.cleared(seen):
                window.touch()
            elif window.stalled():
                moved, silent_for = floor.moved(seen), window.silent_for()
                logger.error(
                    "download STALLED %s: %d bytes in the last %.0fs (floor %d bytes; "
                    "downloaded=%d total); abandoning the wedged thread (#379, pgw#655)",
                    label, moved, silent_for, int(min_window_bytes), last_bytes,
                )
                raise DownloadStalledError(
                    f"download({label}) stalled: only {moved} bytes in "
                    f"{stall_timeout:.0f}s (floor {int(min_window_bytes)} bytes) "
                    f"after {last_bytes} bytes"
                )

    if "exc" in holder:
        raise holder["exc"]
    return str(holder["local"])


# ---------------------------------------------------------------------------
# Civitai: bounded provider fetch (the only conversion-free civitai path)
# ---------------------------------------------------------------------------

_CIVITAI_API = "https://civitai.com/api/v1"
_CIVITAI_AUTH_HOSTS = {"civitai.com", "www.civitai.com", "api.civitai.com"}
_CIVITAI_CHUNK = 4 * 1024 * 1024
_CIVITAI_JSON_TIMEOUT = (30.0, 120.0)    # (connect, read) seconds
_CIVITAI_STREAM_TIMEOUT = (60.0, 180.0)  # read timeout doubles as stall bound
#: How far a civitai stream may exceed its declared `sizeBytes` before it is
#: refused. Not slack for its own sake: the declaration is derived from
#: `sizeKB`, a rounded float, so the true byte count is legitimately off by up
#: to a kilobyte. The in-loop cap and the post-transfer size check use this one
#: number, because a cap tighter than the acceptance rule refuses files the
#: acceptance rule would accept.
_CIVITAI_SIZE_SLACK = 1024


def _civitai_attempts() -> int:
    raw = os.environ.get("COZY_CIVITAI_DOWNLOAD_ATTEMPTS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 3


def _civitai_get_json(url: str, api_key: str = "") -> dict[str, Any]:
    import requests  # lazy (all sites): download is on the `import gen_worker` path; stays requests-free

    headers: Dict[str, str] = {}
    if api_key and urlparse(url).hostname in _CIVITAI_AUTH_HOSTS:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(url, headers=headers, timeout=_CIVITAI_JSON_TIMEOUT)
    if resp.status_code in (401, 403):
        raise ValueError("civitai_access_denied")
    if resp.status_code == 404:
        raise ValueError("civitai_not_found")
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("civitai_fetch_failed")
    return data


def fetch_civitai_model_version(version_id: int, *, api_key: str = "") -> dict[str, Any]:
    return _civitai_get_json(f"{_CIVITAI_API}/model-versions/{int(version_id)}", api_key)


def _civitai_file_entry(raw: Mapping[str, Any]) -> dict[str, Any]:
    size = raw.get("sizeBytes")
    if not isinstance(size, int) or size <= 0:
        kb = raw.get("sizeKB")
        size = int(float(kb) * 1024) if isinstance(kb, (int, float)) and kb > 0 else 0
    hashes = raw.get("hashes") if isinstance(raw.get("hashes"), Mapping) else {}
    meta = raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else {}
    return {
        "id": int(raw.get("id") or 0),
        "name": Path(str(raw.get("name") or "").strip()).name,
        "url": str(raw.get("downloadUrl") or raw.get("download_url") or "").strip(),
        "size_bytes": int(size),
        "sha256": str((hashes or {}).get("SHA256") or "").strip().lower(),
        "primary": bool(raw.get("primary")),
        "quant_type": str((meta or {}).get("quantType") or "").strip().lower(),
    }


# Servable-first gguf quant preference (mirrors gen_worker.convert.classifier
# _GGUF_QUANT_PREFERENCE; duplicated to avoid a models→convert import cycle).
_CIVITAI_GGUF_QUANT_PREFERENCE = (
    "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q4_0",
    "q3_k_m", "q3_k_s", "q2_k", "f16", "bf16", "f32",
)

_CIVITAI_GGUF_QTYPE_RE = re.compile(r"(?:ud-)?(?:i?q\d[0-9a-z_]*|bf16|f16|f32)")


def _civitai_gguf_quant_of(f: Mapping[str, Any]) -> str:
    if f.get("quant_type"):
        return str(f["quant_type"]).lower()
    # The model-versions API omits metadata.quantType; the per-file
    # downloadUrl carries it as a query param (the version's PRIMARY file
    # gets a bare default URL instead — quant unknowable pre-download).
    url = str(f.get("url") or "")
    q = parse_qs(urlparse(url).query).get("quantType", [""])[0]
    if q:
        return q.strip().lower()
    m = _CIVITAI_GGUF_QTYPE_RE.search(str(f.get("name") or "").lower())
    return m.group(0) if m else ""


def _civitai_select_files(
    payload: Mapping[str, Any], *, gguf_quant: str | None = None,
) -> list[dict[str, Any]]:
    """Downloadable weight files of a model version, primary first.

    Safetensors files win when present. Civitai may publish alternative
    precisions under the same filename; keep the primary/first alternative
    once so a later variant cannot overwrite it on disk. GGUF-only versions
    select exactly ONE gguf — civitai reuses a single filename across
    quantType variants, so downloading several would collide on disk:
    ``gguf_quant`` picks it explicitly, else the preference order applies.
    """
    st: list[dict[str, Any]] = []
    gg: list[dict[str, Any]] = []
    for raw in payload.get("files") or []:
        if not isinstance(raw, Mapping):
            continue
        entry = _civitai_file_entry(raw)
        if not entry["url"] or not entry["name"]:
            continue
        lower = entry["name"].lower()
        if lower.endswith(".safetensors"):
            st.append(entry)
        elif lower.endswith(".gguf"):
            gg.append(entry)
    if st:
        st.sort(key=lambda f: (0 if f["primary"] else 1, f["id"], f["name"]))
        unique: dict[str, dict[str, Any]] = {}
        for f in st:
            unique.setdefault(f["name"], f)
        return list(unique.values())
    if not gg:
        return []
    gg.sort(key=lambda f: (f["id"], f["name"]))
    if gguf_quant:
        want = str(gguf_quant).strip().lower()
        picked = [f for f in gg if want in (_civitai_gguf_quant_of(f) or "")
                  or want in f["name"].lower()]
        if not picked:
            raise ValueError(f"civitai_gguf_quant_not_found: {want}")
        return picked[:1]
    for q in _CIVITAI_GGUF_QUANT_PREFERENCE:
        picked = [f for f in gg if _civitai_gguf_quant_of(f) == q]
        if picked:
            return picked[:1]
    return gg[:1]


def _civitai_stream_one(
    url: str,
    dst: Path,
    *,
    api_key: str,
    expected_size: int,
    expected_sha256: str,
    on_bytes: Callable[[int], None],
) -> tuple[int, str]:

    import requests  # lazy (all sites): download is on the `import gen_worker` path; stays requests-free

    headers: Dict[str, str] = {}
    if api_key and urlparse(url).hostname in _CIVITAI_AUTH_HOSTS:
        # Bearer only against civitai's own hosts — requests strips the
        # Authorization header on cross-host redirects (signed CDN URLs).
        headers["Authorization"] = f"Bearer {api_key}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    h = hashlib.sha256()
    # This is the THIRD-PARTY origin in the set — civitai chooses both the byte
    # stream and the `sizeBytes` we would check it against — so the stream is
    # bounded in-loop, not merely checked after the body is on disk. Two
    # bounds, because the declaration is optional here in a way it is not on
    # our own surfaces:
    #   * declared: the cap is the declaration plus the SAME 1 KiB tolerance
    #     the post-check applies (sizeKB is a rounded float). One tolerance,
    #     used twice — a cap tighter than the acceptance check would refuse
    #     files the very next line accepts.
    #   * undeclared: civitai omits `sizeBytes` on real files, so refusing is
    #     not available. The bound is the destination filesystem, which is the
    #     resource an unbounded weights download actually exhausts.
    if expected_size:
        cap = int(expected_size) + _CIVITAI_SIZE_SLACK
    else:
        cap = free_space_bound(tmp.parent)
    with requests.get(url, headers=headers, stream=True, timeout=_CIVITAI_STREAM_TIMEOUT) as resp:
        if resp.status_code in (401, 403):
            raise ValueError("civitai_access_denied")
        if resp.status_code == 404:
            raise ValueError("civitai_not_found")
        resp.raise_for_status()
        try:
            with open(tmp, "wb") as f:
                written = copy_bounded(
                    resp.iter_content(chunk_size=_CIVITAI_CHUNK), f.write,
                    limit_bytes=cap, what=f"civitai file {dst.name}",
                    hasher=h, on_bytes=on_bytes,
                )
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    # Integrity is the sha256 check below; the size check only catches
    # truncated streams (an overlong one never reaches here any more).
    if expected_size and abs(written - expected_size) > _CIVITAI_SIZE_SLACK:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"civitai size mismatch for {dst.name}: expected {expected_size}, got {written}")
    observed = h.hexdigest().lower()
    if expected_sha256 and observed != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"civitai sha256 mismatch for {dst.name}")
    tmp.replace(dst)
    # The OBSERVED digest travels back whether or not civitai published one to
    # check it against. Refusing an unhashed file is not available — civitai
    # routinely omits SHA256 for large/GGUF files and this lane exists to
    # ingest them — so instead the manifest distinguishes a verified from an
    # unverified download.
    return written, observed


def _civitai_prior_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """This directory's own record of what a previous run actually landed,
    keyed by file name. ``{}`` when there is none or it does not parse."""
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    rows = raw.get("files") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return {}
    return {
        str(r["name"]): dict(r) for r in rows
        if isinstance(r, Mapping) and str(r.get("name") or "")
    }


def _civitai_adoptable(
    dst: Path, declared: Mapping[str, Any], prior: Optional[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    """The manifest row for an existing file that is provably complete, or
    ``None`` to (re)download it.

    Never adopt on existence alone: a truncated prior attempt is exactly the
    state that produces an existing file and no declared size at once.
    With nothing declared, the only evidence available is this directory's own
    manifest from a run that COMPLETED — so that is what is required, and its
    absence means "download it", never "assume it".
    """
    if not dst.exists():
        return None
    try:
        on_disk = dst.stat().st_size
    except OSError:
        return None
    size = int(declared.get("size_bytes") or 0)
    declared_sha = str(declared.get("sha256") or "")
    if size:
        if on_disk != size:
            return None
        return {
            "name": str(declared["name"]), "size_bytes": size,
            "sha256": declared_sha or str((prior or {}).get("sha256") or ""),
            "sha256_source": "civitai" if declared_sha else (
                str((prior or {}).get("sha256_source") or "") or "unverified"),
        }
    if not prior:
        return None
    prior_size = int(prior.get("size_bytes") or 0)
    if not prior_size or prior_size != on_disk:
        return None
    return {
        "name": str(declared["name"]), "size_bytes": on_disk,
        "sha256": str(prior.get("sha256") or ""),
        "sha256_source": str(prior.get("sha256_source") or "") or "unverified",
    }


def download_civitai(
    version_id: int,
    out_dir: Path,
    *,
    api_key: str = "",
    progress: Optional[ProgressFn] = None,
    gguf_quant: str | None = None,
) -> Path:
    """Blocking civitai model-version fetch (call via ``ensure_local`` /
    ``asyncio.to_thread``). Downloads the version's weight files with
    size + sha256 validation. Returns the single artifact path when the
    version has exactly one file, else the directory."""
    payload = fetch_civitai_model_version(version_id, api_key=api_key)
    files = _civitai_select_files(payload, gguf_quant=gguf_quant)
    if not files:
        # The selector is an ALLOW-list, so a pickle-only version already
        # cannot be downloaded — but it said `no_supported_files`, which reads
        # as "civitai is broken". Name the real reason, with the same typed
        # refusal the other two lanes raise (HARDCUT E5).
        names = [str(f.get("name") or "") for f in (payload.get("files") or [])
                 if isinstance(f, Mapping)]
        if bad := first_pickle_weight_path(names):
            raise PickleWeightRefused(
                f"refusing civitai version {version_id}: {bad!r} is a "
                "pickle-format weight and it is the only weight published. "
                "Unpickling is arbitrary code execution in this process."
            )
        raise ValueError("civitai_no_supported_files")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / ".civitai.json"

    total = sum(f["size_bytes"] for f in files) or None
    done = 0

    def _on_bytes(n: int) -> None:
        nonlocal done
        done += n
        if progress is not None:
            try:
                progress(done, total)
            except Exception:
                pass

    import requests  # lazy (all sites): download is on the `import gen_worker` path; stays requests-free

    prior = _civitai_prior_manifest(manifest_path)
    landed: dict[str, dict[str, Any]] = {}
    local_paths: list[Path] = []
    for f in files:
        dst = out_dir / f["name"]
        local_paths.append(dst)
        adopted = _civitai_adoptable(dst, f, prior.get(f["name"]))
        if adopted is not None:
            done += f["size_bytes"]
            landed[f["name"]] = adopted
            continue
        attempts = _civitai_attempts()
        file_start = done
        for attempt in range(1, attempts + 1):
            try:
                written, observed = _civitai_stream_one(
                    f["url"], dst,
                    api_key=api_key,
                    expected_size=f["size_bytes"],
                    expected_sha256=f["sha256"],
                    on_bytes=_on_bytes,
                )
                landed[f["name"]] = {
                    "name": f["name"],
                    "size_bytes": int(written),
                    "sha256": observed,
                    "sha256_source": "civitai" if f["sha256"] else "observed",
                }
                break
            except (requests.RequestException, OSError) as exc:
                done = file_start  # rewind progress from the failed partial
                if attempt >= attempts:
                    raise RuntimeError(
                        f"civitai download of {f['name']} failed after "
                        f"{attempts} attempt(s): {type(exc).__name__}: {exc}") from exc
                logger.warning(
                    "civitai download %s attempt %d/%d failed (%s: %s); retrying",
                    f["name"], attempt, attempts, type(exc).__name__, exc)
                time.sleep(min(10.0, 2.0 * attempt))
    manifest_path.write_text(json.dumps(
        {"model_version_id": int(version_id),
         "files": [landed[f["name"]] for f in files]},
        indent=2,
    ), encoding="utf-8")
    if progress is not None and total:
        try:
            progress(total, total)
        except Exception:
            pass
    return local_paths[0] if len(local_paths) == 1 else out_dir


__all__ = [
    "ensure_local",
    # INGEST-only fetchers (gen_worker.convert drives them). They are NOT on
    # the serving path any more — pgw#1524 deleted every direct-serve branch.
    "download_civitai",
    "fetch_civitai_model_version",
    "select_component_paths",
    "components_present",
    "DownloadStalledError",
    "set_provider_index",
    "lookup_provider_for_ref",
    "build_provider_index_from_manifest",
]
