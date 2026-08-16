"""ONE ensure-local path for every model provider.

``ensure_local(ref)`` materializes a model ref on disk and returns its local
path, dispatching on provider:

  - tensorhub : CAS blob download + snapshot materialization (orchestrator
                pre-resolves and ships presigned URLs; see cozy_snapshot.py)
  - hf        : ``huggingface_hub.snapshot_download`` in a thread, with
                ``allow_patterns`` from a small variant selector
  - civitai   : bounded provider fetch of a model-version's safetensors files
  - modelscope: ``modelscope.snapshot_download`` in a thread

One progress-reporter shape everywhere: ``progress(bytes_done, bytes_total)``
(total may be None). Blocking library calls always run off the event loop.
"""

from __future__ import annotations

import asyncio
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
from ..config import Settings, current_or

_STANDALONE = Settings()
from ..stall import ProgressFloor, SilenceWindow
from .cache_paths import tensorhub_cas_dir
from .errors import PickleWeightRefused, first_pickle_weight_path
from .refs import HuggingFaceRef, TensorhubRef, fold_ref, parse_model_ref
import hashlib
from fnmatch import fnmatch

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
    keying and ``set_provider_index`` adds the repo-identity fallback keys."""
    index: dict[str, str] = {}
    if not isinstance(manifest, Mapping):
        return index
    functions = manifest.get("functions")
    if not isinstance(functions, list):
        return index
    for fn in functions:
        if not isinstance(fn, dict):
            continue
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
    hf_home: Optional[str] = None,
    hf_token: Optional[str] = None,
    civitai_api_key: str = "",
    allow_patterns: Sequence[str] = (),
    components: Sequence[str] = (),
    progress: Optional[ProgressFn] = None,
    fill_source_dir: Optional[Path] = None,
) -> Path:
    """Materialize ``ref`` on disk; return its local path.

    ``snapshot`` is the orchestrator-resolved manifest (the typed
    ``WorkerResolvedRepo``) carrying presigned URLs or transfer grants. The
    orchestrator is the only resolver: when it ships a snapshot for a ref —
    including an hf/civitai binding ref resolved through a platform mirror
    under mirror-first — the snapshot is authoritative and the bytes come from
    tensorhub-CAS, never the upstream registry. Refs without a snapshot fall
    back to their provider's direct download (hf/civitai/modelscope) or fail
    retryably (tensorhub).

    ``components`` is honored on the HF direct-download branch.
    It is deliberately NOT applied to the ``snapshot is not None`` branch:
    that snapshot is orchestrator-resolved and the executor's residency
    layer digest-verifies the materialized tree against the FULL
    ``snapshot.files`` list it was handed (``ModelStore._verify_snapshot_tree``)
    — filtering what gets written here without the orchestrator also
    filtering what it verifies would turn every boot into a spurious
    corruption/quarantine loop. Selective fetch for tensorhub-sourced
    snapshots is therefore the hub's desired-snapshot scoping (the manifest
    now carries ``components`` for it to read) — worker-side selective fetch
    for tensorhub refs lands on the hub-less CLI path instead
    (``models/provision.py::_fetch_tensorhub_snapshot``), which owns its own
    resolve+download+materialize loop end to end.

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

    if parsed.provider == "hf" and parsed.hf is not None:
        return await asyncio.to_thread(
            download_hf,
            parsed.hf,
            hf_home=hf_home,
            hf_token=hf_token,
            allow_patterns=tuple(allow_patterns),
            components=tuple(components),
            progress=progress,
        )

    if parsed.provider == "civitai" and parsed.civitai is not None:
        version_id = parse_civitai_version_id(parsed.civitai.model_id)
        return await asyncio.to_thread(
            download_civitai,
            version_id,
            base / "civitai" / str(version_id),
            api_key=civitai_api_key or current_or(_STANDALONE).civitai_api_key,
            progress=progress,
        )

    if parsed.provider == "modelscope" and parsed.modelscope is not None:
        ms = parsed.modelscope
        return await asyncio.to_thread(
            download_modelscope,
            ms.repo_id,
            cache_dir=base / "modelscope",
            revision=ms.revision or "",
            allow_patterns=tuple(allow_patterns),
        )

    # Typed so the executor classifies it INVALID (bad input, never retry) —
    # a bare ValueError maps FATAL.
    raise ValidationError(f"unsupported model ref {ref!r} (provider={prov!r})")


# ---------------------------------------------------------------------------
# Hugging Face: snapshot_download + small variant selector
# ---------------------------------------------------------------------------

from ..component_vocab import denoiser_components, pipeline_component_dirs
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".msgpack", ".h5")
_VARIANT_TAGS = ("bf16", "fp16", "fp8", "int8", "int4")
_component_dirs = pipeline_component_dirs


def _variant_of(filename: str) -> str:
    low = filename.lower()
    for tag in _VARIANT_TAGS:
        if f".{tag}." in low or f"-{tag}." in low or f"_{tag}." in low:
            return tag
    return ""


def select_hf_files(repo_files: Sequence[str]) -> Optional[set[str]]:
    """Small variant selector: which repo files to download.

    - Diffusers-style repos (component dirs): every non-weight file (configs,
      tokenizers) plus ONE weight set per component directory — bf16 > fp16 >
      untagged, safetensors preferred. Root monolithic checkpoints are
      excluded (redundant). There is no ``flavor=`` override: HF has no flavor
      axis, and an explicit file set is what ``files=`` is for.
    - Root-weights repos: the root weight files (same variant rule) + sidecars.
    - Anything else: None (download the whole repo).
    """
    files = [f for f in repo_files if f and not f.endswith("/")]
    if not files:
        return None
    dirs = {f.split("/", 1)[0] for f in files if "/" in f}
    diffusers_like = "model_index.json" in files or any(d in dirs for d in denoiser_components())

    weights_by_dir: Dict[str, list[str]] = {}
    non_weights: set[str] = set()
    for f in files:
        name = f.rsplit("/", 1)[-1]
        low = name.lower()
        if low.endswith(_WEIGHT_SUFFIXES):
            d = f.split("/", 1)[0] if "/" in f else ""
            weights_by_dir.setdefault(d, []).append(f)
        else:
            non_weights.add(f)

    if not weights_by_dir:
        return None

    if not diffusers_like and set(weights_by_dir) != {""}:
        return None  # unrecognized layout: full repo

    if diffusers_like and "" in weights_by_dir and len(weights_by_dir) > 1:
        # Root single-file distributions (sd1.5's v1-5-pruned*.safetensors)
        # duplicate the component-dir weights — diffusers loads from the
        # component dirs, so never pull the monolithic root checkpoints.
        del weights_by_dir[""]

    def _pick(group: list[str]) -> list[str]:
        st = [f for f in group if f.lower().endswith(".safetensors")]
        pool = st or group
        by_variant: Dict[str, list[str]] = {}
        for f in pool:
            by_variant.setdefault(_variant_of(f.rsplit("/", 1)[-1]), []).append(f)
        order: list[str] = ["bf16", "fp16", ""]
        for v in order:
            if by_variant.get(v):
                return by_variant[v]
        # Only exotic variants exist (fp8/int8/...) — take the smallest-named
        # deterministic group rather than all of them.
        first = sorted(by_variant)[0]
        return by_variant[first]

    selected: set[str] = set(non_weights)
    for group in weights_by_dir.values():
        selected.update(_pick(group))
    return selected


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


def _match_allow_patterns(repo_files: Sequence[str], patterns: Sequence[str]) -> set[str]:
    """Repo files matched by ``patterns`` — huggingface_hub semantics
    (fnmatch; a trailing ``/`` means the whole directory)."""

    pats = [p + "*" if p.endswith("/") else p for p in patterns]
    return {f for f in repo_files if any(fnmatch(f, p) for p in pats)}


# HF snapshot-download guards: the progress window and the
# accidental-huge-repo cap (0 = off). No deployment has ever overridden these,
# so they're fixed constants rather than Settings fields.
#
# The bound is a PROGRESS-RATE FLOOR, never a wall clock. A wall-clock cap
# cannot distinguish a healthy 200GB pull from a wedge, so it is either
# useless or it kills real downloads; the honest question is "is this transfer
# still moving fast enough to ever finish?". A transfer that cannot put
# _HF_DOWNLOAD_MIN_WINDOW_BYTES on disk within _HF_DOWNLOAD_STALL_TIMEOUT_S is
# stalled — including a trickle, which resets an any-progress watchdog forever
# with a byte a minute. 8 MiB / 180s ~= 46 KiB/s: three orders of magnitude
# under any real pod link, and still 60+ hours for a 10GB checkpoint, so
# nothing healthy is near it.
_HF_DOWNLOAD_STALL_TIMEOUT_S = 180.0
_HF_DOWNLOAD_MIN_WINDOW_BYTES = 8 * 1024 * 1024
_HF_MAX_REPO_BYTES = 60_000_000_000


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


def _refuse_pickles_on_disk(root: Path, label: str) -> None:
    """Refuse if any pickle-format weight is materialized under ``root``.

    Used where there is no listing to refuse against ahead of time. Symlinks
    are not followed: the HF cache materializes snapshots as symlinks into
    ``blobs/``, and the name that matters is the one a loader will open.
    """
    try:
        names = [p.name for p in root.rglob("*") if p.is_file() or p.is_symlink()]
        if root.is_file() or root.is_symlink():
            names.append(root.name)  # a lane may return the artifact, not a tree
    except OSError:
        return
    if bad := first_pickle_weight_path(names):
        raise PickleWeightRefused(
            f"refusing {label}: {bad!r} is a pickle-format weight. "
            "Unpickling is arbitrary code execution in this process."
        )


def download_modelscope(
    repo_id: str,
    *,
    cache_dir: Optional[Path] = None,
    revision: str = "",
    allow_patterns: Sequence[str] = (),
    local_files_only: bool = False,
) -> Path:
    """ModelScope snapshot fetch — the THIRD entry lane, and the one pgw#1273
    left open (HARDCUT E5).

    HF refuses a pickle off the file listing before a byte moves and civitai
    selects `.safetensors`/`.gguf` by allow-list, but modelscope mirrors
    HF-shaped repos verbatim — `.bin` components and all — and both call sites
    handed `snapshot_download`'s return straight to a loader. The refusal is on
    what LANDED rather than on a listing (modelscope's listing API is a network
    call this lane does not otherwise make), so bytes move and then nothing
    reads them: the door is still shut in front of every deserializer.
    """
    from modelscope import snapshot_download as ms_snap

    kwargs: Dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision:
        kwargs["revision"] = revision
    if allow_patterns:
        kwargs["allow_patterns"] = list(allow_patterns)
    if local_files_only:
        kwargs["local_files_only"] = True
    local = Path(str(ms_snap(model_id=repo_id, **kwargs)))
    _refuse_pickles_on_disk(local, f"modelscope:{repo_id}")
    return local


def _hf_progress_dir(hf_home: Optional[str], ref: HuggingFaceRef) -> Path:
    base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    safe = ref.repo_id.replace("/", "--").replace(":", "_")
    rev = (ref.revision or "main").replace("/", "--").replace(":", "_")
    return base / "gen-worker-progress-snapshots" / safe / rev


def download_hf(
    ref: HuggingFaceRef,
    *,
    hf_home: Optional[str] = None,
    hf_token: Optional[str] = None,
    allow_patterns: Sequence[str] = (),
    components: Sequence[str] = (),
    progress: Optional[ProgressFn] = None,
    local_files_only: bool = False,
) -> Path:
    """Blocking HF snapshot download (call via ``ensure_local`` /
    ``asyncio.to_thread``). Transfer, cache, resume and locking are
    huggingface_hub's; this only plans the file selection.

    ``components`` narrows the repo listing to the named pipeline
    component subfolders (+ root config files, via
    :func:`select_component_paths`) BEFORE the existing ``files=``/auto
    variant-selection logic runs — so a component-scoped fetch still gets
    per-component flavor picking (bf16 > fp16 > untagged), just over a
    smaller candidate set.
    """
    from ..net import hf  # lazy: net imports requests; keeps `import gen_worker` light

    try:
        hub = hf()  # no HF socket may wait forever
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for hf model refs; install gen-worker with the HF extra."
        ) from e
    HfApi, snapshot_download = hub.HfApi, hub.snapshot_download

    repo_id = (ref.repo_id or "").strip()
    if not repo_id:
        raise ValueError("empty hf repo_id")
    hf_home = (hf_home or "").strip() or None
    hf_token = (hf_token or "").strip() or None
    comps = tuple(c.strip() for c in components if c and str(c).strip())
    kwargs: dict[str, Any] = {}
    if hf_home:
        kwargs["cache_dir"] = hf_home
    if hf_token:
        kwargs["token"] = hf_token

    if local_files_only:
        patterns = list(allow_patterns)
        if comps and not patterns:
            # No repo listing available offline to narrow precisely — fall
            # back to component-subfolder + root-json glob patterns.
            patterns = [f"{c}/" for c in comps] + ["*.json"]
        offline = Path(snapshot_download(
            repo_id=repo_id, revision=ref.revision, local_files_only=True,
            allow_patterns=patterns or None, **kwargs,
        ))
        # pgw#1273: this path returns before the selection-time refusal below,
        # and offline there is no listing to refuse against — so refuse on what
        # is actually on disk. A cache populated by an earlier build (or by
        # hand) is exactly the "reaches a worker by another path" case.
        _refuse_pickles_on_disk(offline, f"{repo_id}@{ref.revision or 'main'} (cached)")
        return offline

    api = HfApi(token=hf_token)
    repo_files = list(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=ref.revision))

    if comps:
        if not components_present(repo_files, comps):
            raise ValueError(
                f"components= {list(comps)!r} matched nothing in {repo_id}"
            )
        repo_files = sorted(select_component_paths(repo_files, comps))

    selected: Optional[set[str]]
    if allow_patterns:
        # Size-guard the MATCHED subset, not the whole repo — a small files=
        # subset of a huge repo must not trip the repo-size cap. Same fnmatch
        # semantics as huggingface_hub.filter_repo_objects.
        selected = _match_allow_patterns(repo_files, allow_patterns)
        if not selected:
            raise ValueError(
                f"files= patterns matched nothing in {repo_id}: {list(allow_patterns)!r}"
            )
        kwargs["allow_patterns"] = list(allow_patterns)
    else:
        selected = select_hf_files(repo_files)
        if selected is not None:
            kwargs["allow_patterns"] = sorted(selected)
        elif comps:
            # components= narrowed the listing but the repo has no
            # recognized diffusers/root-weights layout for select_hf_files
            # to specialize further — fetch exactly the components= subset
            # rather than silently reverting to the whole repo.
            selected = set(repo_files)
            kwargs["allow_patterns"] = repo_files

    # pgw#1273 — REFUSE PICKLES BEFORE A BYTE MOVES.
    #
    # `selection` is the final resolved set across all three selection paths
    # (files= patterns, select_hf_files, or the narrowed listing), so this is
    # the one chokepoint that sees every HF fetch.
    #
    # The gap was structural, not an oversight: `_pick` prefers safetensors but
    # falls back to the whole group when a component has none
    # (`pool = st or group`), `select_hf_files` returns None -> full repo on an
    # unrecognized layout, and `select_component_paths` — which calls itself
    # "the ONE filter both sources apply" — filters by COMPONENT and never by
    # extension. Nothing downstream re-checked: `use_safetensors=True` appears
    # nowhere in this tree, so a selected `pytorch_model.bin` reaches
    # `from_pretrained` and is unpickled in the process holding hub credentials.
    #
    # The tensorhub lane already refuses on the resolved manifest
    # (cozy_snapshot._validate_resolved); the civitai lane is an allow-list.
    # This is the same refusal on the same list, for the third provider —
    # exactly the case PickleWeightRefused's docstring already named:
    # "defence in depth for blobs that ... reach a worker by another path."
    selection = selected if selected is not None else set(repo_files)
    if bad := first_pickle_weight_path(sorted(selection)):
        raise PickleWeightRefused(
            f"refusing {repo_id}@{ref.revision or 'main'}: {bad!r} is a pickle-format "
            "weight. Unpickling is arbitrary code execution in this process. "
            "Use a safetensors revision, or ingest through the conversion endpoint."
        )

    # Best-effort size guard against accidental huge downloads.
    total_hint: Optional[int] = None
    try:
        sizes = {
            e.path: int(getattr(e, "size", 0))
            for e in api.list_repo_tree(repo_id=repo_id, repo_type="model",
                                        revision=ref.revision, recursive=True)
            if isinstance(getattr(e, "size", None), int)
        }
        wanted = selected if selected is not None else set(repo_files)
        total_hint = sum(sizes.get(p, 0) for p in wanted)
        if _HF_MAX_REPO_BYTES > 0 and total_hint > _HF_MAX_REPO_BYTES:
            raise RuntimeError(
                f"refusing an excessively large Hugging Face selection for {repo_id}: "
                f"{total_hint} bytes (limit {_HF_MAX_REPO_BYTES})"
            )
    except RuntimeError:
        raise
    except Exception:
        total_hint = None

    progress_root: Optional[Path] = None
    if progress is not None:
        progress_root = _hf_progress_dir(hf_home, ref)
        progress_root.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(progress_root)
        try:
            progress(0, total_hint)
        except Exception:
            pass

    local = _run_with_stall_watchdog(
        lambda: snapshot_download(repo_id=repo_id, revision=ref.revision, **kwargs),
        label=f"{repo_id}@{ref.revision or 'main'}",
        progress_root=progress_root,
        progress_callback=progress,
        total_hint=total_hint,
        stall_timeout=_HF_DOWNLOAD_STALL_TIMEOUT_S,
    )
    if progress is not None:
        try:
            final = _scan_bytes(Path(local))
            progress(final, total_hint or final or None)
        except Exception:
            pass
    return Path(local)


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


def parse_civitai_version_id(raw: str) -> int:
    """Extract a model-VERSION id from a ref string / URL fragment."""
    s = str(raw or "").strip()
    if not s:
        raise ValueError("empty civitai model ref")
    for key in ("modelVersionId=", "model_version_id=", "version_id="):
        if key in s:
            s = s.split(key, 1)[1].split("&", 1)[0].strip()
            break
    if s.startswith("versions/"):
        s = s.split("/", 1)[1].strip()
    if not s.isdigit():
        raise ValueError(f"civitai model ref must be a model version id, got {raw!r}")
    return int(s)


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


def fetch_civitai_model(model_id: int, *, api_key: str = "") -> dict[str, Any]:
    """GET /models/{id} — used to map a model id to its latest version id."""
    return _civitai_get_json(f"{_CIVITAI_API}/models/{int(model_id)}", api_key)


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
    "download_hf",
    "download_civitai",
    "download_modelscope",
    "select_hf_files",
    "select_component_paths",
    "components_present",
    "fetch_civitai_model",
    "fetch_civitai_model_version",
    "parse_civitai_version_id",
    "DownloadStalledError",
    "set_provider_index",
    "lookup_provider_for_ref",
    "build_provider_index_from_manifest",
]
