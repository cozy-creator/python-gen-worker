"""ONE ensure-local path for every model provider (#366).

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
from ..config import get_settings
from .cache_paths import tensorhub_cas_dir
from .refs import HuggingFaceRef, TensorhubRef, fold_ref, parse_model_ref

if TYPE_CHECKING:
    from .hub_client import WorkerResolvedRepo

logger = logging.getLogger("gen_worker.download")

ProgressFn = Callable[[int, Optional[int]], None]

# ---------------------------------------------------------------------------
# Provider index: normal-form ref -> provider. Built once at boot from the
# endpoint.lock manifest (the wire carries bare refs without a provider field).
#
# ONE keying function (gw#492) normalizes both index keys and lookups —
# replacing the old raw/stripped/tag-removed fallback chain and its
# `_binding_canonical_ref` twin. Keys are flavor-granular (a dispatch table
# may bind two providers to one repo name via different flavors) with a
# repo-identity fallback so hub-minted picks of NEW flavors (`#svdq-int4`)
# still route to their repo's provider.
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
        exact = TensorhubRef(owner=th.owner, repo=th.repo, tag=th.tag,
                             digest=None, flavor=th.flavor).canonical()
        return exact, th.repo_id()
    assert parsed.hf is not None
    hf = parsed.hf
    exact = HuggingFaceRef(repo_id=hf.repo_id, revision=None,
                           flavor=hf.flavor).canonical()
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
        if str(entry.get("kind") or "").strip() == "dispatch":
            table = entry.get("table")
            if isinstance(table, dict):
                out.extend(sub for sub in table.values() if isinstance(sub, dict))
            continue
        if entry.get("ref"):
            out.append(entry)
    return out


def build_provider_index_from_manifest(manifest: Optional[Mapping[str, Any]]) -> dict[str, str]:
    """{normal_form_ref: provider} from a loaded endpoint.lock manifest.

    Entry ``tag``/``flavor`` side-channel fields fold into the ref via the
    ONE grammar module before keying; ``set_provider_index`` adds the
    repo-identity fallback keys."""
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
                    tag=str(entry.get("tag") or ""),
                    flavor=str(entry.get("flavor") or ""),
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
    progress: Optional[ProgressFn] = None,
) -> Path:
    """Materialize ``ref`` on disk; return its local path.

    ``snapshot`` is the orchestrator-resolved manifest (the typed
    ``WorkerResolvedRepo``, gw#497) carrying presigned URLs or transfer
    grants. The orchestrator is the only resolver: when it ships a
    snapshot for a ref — including an hf/civitai binding ref resolved through
    a platform mirror under mirror-first (tensorhub #557) — the snapshot is
    authoritative and the bytes come from tensorhub-CAS, never the upstream
    registry. Refs without a snapshot fall back to their provider's direct
    download (hf/civitai/modelscope) or fail retryably (tensorhub).
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
        )

    if parsed.provider == "tensorhub" and parsed.tensorhub is not None:
        # The worker cannot resolve tensorhub-CAS refs itself (gw#465):
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
            progress=progress,
        )

    if parsed.provider == "civitai" and parsed.civitai is not None:
        version_id = parse_civitai_version_id(parsed.civitai.model_id)
        return await asyncio.to_thread(
            download_civitai,
            version_id,
            base / "civitai" / str(version_id),
            api_key=civitai_api_key or get_settings().civitai_api_key,
            progress=progress,
        )

    if parsed.provider == "modelscope" and parsed.modelscope is not None:
        ms = parsed.modelscope

        def _ms_download() -> str:
            from modelscope import snapshot_download as ms_snap

            kw: dict[str, Any] = {"cache_dir": str(base / "modelscope")}
            if ms.revision:
                kw["revision"] = ms.revision
            if allow_patterns:
                kw["allow_patterns"] = list(allow_patterns)
            return ms_snap(model_id=ms.repo_id, **kw)

        return Path(await asyncio.to_thread(_ms_download))

    # Typed so the executor classifies it INVALID (bad input, never retry) —
    # a bare ValueError maps FATAL since pgw#514/P9.
    raise ValidationError(f"unsupported model ref {ref!r} (provider={prov!r})")


# ---------------------------------------------------------------------------
# Hugging Face: snapshot_download + small variant selector
# ---------------------------------------------------------------------------

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".msgpack", ".h5")
_VARIANT_TAGS = ("bf16", "fp16", "fp8", "int8", "int4")
_COMPONENT_DIRS = (
    "transformer", "transformer_2", "unet", "vae", "text_encoder",
    "text_encoder_2", "text_encoder_3", "tokenizer", "tokenizer_2",
    "tokenizer_3", "scheduler", "feature_extractor", "image_encoder",
    "safety_checker", "prior", "decoder",
)


def _variant_of(filename: str) -> str:
    low = filename.lower()
    for tag in _VARIANT_TAGS:
        if f".{tag}." in low or f"-{tag}." in low or f"_{tag}." in low:
            return tag
    return ""


def select_hf_files(
    repo_files: Sequence[str],
    *,
    flavor: Optional[str] = None,
) -> Optional[set[str]]:
    """Small variant selector: which repo files to download.

    - Diffusers-style repos (component dirs): every non-weight file (configs,
      tokenizers) plus ONE weight set per component directory — the requested
      ``flavor`` when present, else bf16 > fp16 > untagged, safetensors
      preferred. Root monolithic checkpoints are excluded (redundant).
    - Root-weights repos: the root weight files (same variant rule) + sidecars.
    - Anything else: None (download the whole repo).
    """
    files = [f for f in repo_files if f and not f.endswith("/")]
    if not files:
        return None
    dirs = {f.split("/", 1)[0] for f in files if "/" in f}
    diffusers_like = "model_index.json" in files or any(d in dirs for d in ("transformer", "unet"))

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
        order: list[str] = []
        if flavor:
            order.append(flavor.strip().lower())
        order += ["bf16", "fp16", ""]
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


def _match_allow_patterns(repo_files: Sequence[str], patterns: Sequence[str]) -> set[str]:
    """Repo files matched by ``patterns`` — huggingface_hub semantics
    (fnmatch; a trailing ``/`` means the whole directory)."""
    from fnmatch import fnmatch

    pats = [p + "*" if p.endswith("/") else p for p in patterns]
    return {f for f in repo_files if any(fnmatch(f, p) for p in pats)}


# HF snapshot-download guards (#379): stall window (no byte progress), wall-
# clock cap (0 = off), and the accidental-huge-repo cap (0 = off). No
# deployment has ever overridden these (pgw#514 dead-config sweep found zero
# producers for the env vars that used to back them), so they're fixed
# constants rather than Settings fields.
_HF_DOWNLOAD_STALL_TIMEOUT_S = 180.0
_HF_DOWNLOAD_MAX_SECONDS = 0.0
_HF_MAX_REPO_BYTES = 60_000_000_000


class DownloadStalledError(RuntimeError):
    """Raised when a blocking snapshot download makes no byte progress for the
    stall window (or exceeds the wall-clock cap) — a bounded, observable
    failure instead of a silent hang (#379)."""


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
    wall_clock_max: float,
    scan_bytes: Callable[[Path], int] = _scan_bytes,
    poll_interval: float = 0.5,
) -> str:
    """Run a blocking download on a daemon thread; the watchdog doubles as the
    progress reporter (scans bytes-on-disk under ``progress_root``) and raises
    :class:`DownloadStalledError` on no-progress / wall-clock breach."""
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

    started_at = time.monotonic()
    last_bytes = 0
    last_progress_at = started_at
    while not holder.get("done"):
        dl_thread.join(timeout=poll_interval)
        if holder.get("done"):
            break
        now = time.monotonic()
        if progress_root is not None:
            try:
                seen = scan_bytes(progress_root)
            except Exception:
                seen = last_bytes
            if seen > last_bytes:
                last_bytes = seen
                last_progress_at = now
                if progress_callback is not None:
                    try:
                        progress_callback(seen, total_hint)
                    except Exception:
                        pass
            elif stall_timeout > 0 and (now - last_progress_at) > stall_timeout:
                logger.error(
                    "download STALLED %s: no byte progress for %.0fs (downloaded=%d bytes); "
                    "abandoning the wedged thread (#379)", label, now - last_progress_at, last_bytes,
                )
                raise DownloadStalledError(
                    f"download({label}) stalled: no progress for "
                    f"{stall_timeout:.0f}s after {last_bytes} bytes"
                )
        if wall_clock_max > 0 and (now - started_at) > wall_clock_max:
            raise DownloadStalledError(
                f"download({label}) exceeded {wall_clock_max:.0f}s wall-clock cap"
            )

    if "exc" in holder:
        raise holder["exc"]
    return str(holder["local"])


def _hf_progress_dir(hf_home: Optional[str], ref: HuggingFaceRef) -> Path:
    base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    safe = ref.repo_id.replace("/", "--").replace(":", "_")
    rev = (ref.revision or "main").replace("/", "--").replace(":", "_")
    flv = (ref.flavor or "default").replace("/", "--").replace(":", "_")
    return base / "gen-worker-progress-snapshots" / safe / rev / flv


def download_hf(
    ref: HuggingFaceRef,
    *,
    hf_home: Optional[str] = None,
    hf_token: Optional[str] = None,
    allow_patterns: Sequence[str] = (),
    progress: Optional[ProgressFn] = None,
    local_files_only: bool = False,
) -> Path:
    """Blocking HF snapshot download (call via ``ensure_local`` /
    ``asyncio.to_thread``). Transfer, cache, resume and locking are
    huggingface_hub's; this only plans the file selection."""
    from ..net import hf

    try:
        hub = hf()  # gw#456: no HF socket may wait forever
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
    kwargs: dict[str, Any] = {}
    if hf_home:
        kwargs["cache_dir"] = hf_home
    if hf_token:
        kwargs["token"] = hf_token

    if local_files_only:
        return Path(snapshot_download(
            repo_id=repo_id, revision=ref.revision, local_files_only=True,
            allow_patterns=list(allow_patterns) or None, **kwargs,
        ))

    api = HfApi(token=hf_token)
    repo_files = list(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=ref.revision))

    selected: Optional[set[str]]
    if allow_patterns:
        # Size-guard the MATCHED subset, not the whole repo (ie#355: a 795KB
        # tokenizer files= subset of google/t5-v1_1-xxl tripped the 60GB cap).
        # Same fnmatch semantics as huggingface_hub.filter_repo_objects.
        selected = _match_allow_patterns(repo_files, allow_patterns)
        if not selected:
            raise ValueError(
                f"files= patterns matched nothing in {repo_id}: {list(allow_patterns)!r}"
            )
        kwargs["allow_patterns"] = list(allow_patterns)
    else:
        selected = select_hf_files(repo_files, flavor=ref.flavor)
        if selected is not None:
            kwargs["allow_patterns"] = sorted(selected)

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
        wall_clock_max=_HF_DOWNLOAD_MAX_SECONDS,
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
    import requests
    from urllib.parse import urlparse

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

    Safetensors files win when present (unchanged behavior). GGUF-only
    versions (th#611: klein/qwen fine-tunes published only as quants)
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
        return st
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
) -> int:
    import hashlib

    import requests
    from urllib.parse import urlparse

    headers: Dict[str, str] = {}
    if api_key and urlparse(url).hostname in _CIVITAI_AUTH_HOSTS:
        # Bearer only against civitai's own hosts — requests strips the
        # Authorization header on cross-host redirects (signed CDN URLs).
        headers["Authorization"] = f"Bearer {api_key}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    h = hashlib.sha256()
    written = 0
    with requests.get(url, headers=headers, stream=True, timeout=_CIVITAI_STREAM_TIMEOUT) as resp:
        if resp.status_code in (401, 403):
            raise ValueError("civitai_access_denied")
        if resp.status_code == 404:
            raise ValueError("civitai_not_found")
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=_CIVITAI_CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                h.update(chunk)
                written += len(chunk)
                on_bytes(len(chunk))
    # civitai's file size often comes from `sizeKB`, a ROUNDED float — the
    # derived byte count can be off by ±1KB (live: wan i2v 19224728441 vs
    # actual ...442, e2e #112). Integrity is the sha256 check below; the size
    # check only catches truncated streams.
    if expected_size and abs(written - expected_size) > 1024:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"civitai size mismatch for {dst.name}: expected {expected_size}, got {written}")
    if expected_sha256 and h.hexdigest().lower() != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"civitai sha256 mismatch for {dst.name}")
    tmp.replace(dst)
    return written


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

    import requests

    local_paths: list[Path] = []
    for f in files:
        dst = out_dir / f["name"]
        local_paths.append(dst)
        if dst.exists() and (not f["size_bytes"] or dst.stat().st_size == f["size_bytes"]):
            done += f["size_bytes"]
            continue
        attempts = _civitai_attempts()
        file_start = done
        for attempt in range(1, attempts + 1):
            try:
                _civitai_stream_one(
                    f["url"], dst,
                    api_key=api_key,
                    expected_size=f["size_bytes"],
                    expected_sha256=f["sha256"],
                    on_bytes=_on_bytes,
                )
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
         "files": [{"name": f["name"], "size_bytes": f["size_bytes"], "sha256": f["sha256"]} for f in files]},
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
    "select_hf_files",
    "fetch_civitai_model",
    "fetch_civitai_model_version",
    "parse_civitai_version_id",
    "DownloadStalledError",
    "set_provider_index",
    "lookup_provider_for_ref",
    "build_provider_index_from_manifest",
]
