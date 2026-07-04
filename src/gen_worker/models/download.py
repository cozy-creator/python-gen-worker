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
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .cache_paths import tensorhub_cas_dir
from .refs import HuggingFaceRef, parse_model_ref

logger = logging.getLogger("gen_worker.download")

ProgressFn = Callable[[int, Optional[int]], None]

_DEFAULT_MAX_REPO_BYTES = 60_000_000_000  # COZY_HF_MAX_REPO_BYTES overrides

# ---------------------------------------------------------------------------
# Provider index: bare ref string -> provider. Built once at boot from the
# endpoint.lock manifest (the wire carries bare refs without a provider field).
# ---------------------------------------------------------------------------

_provider_by_ref: Mapping[str, str] = {}


def set_provider_index(mapping: Optional[Mapping[str, str]]) -> None:
    global _provider_by_ref
    _provider_by_ref = dict(mapping or {})


def lookup_provider_for_ref(ref: str, *, default: str = "tensorhub") -> str:
    """Provider tag for ``ref`` from the index, trying (1) the raw ref,
    (2) stripped, (3) with the ``:tag`` segment removed (runtime payloads may
    stamp ``:latest`` on HF refs; index keys are bare)."""
    if not ref:
        return default
    mapping = _provider_by_ref
    if not mapping:
        return default
    if ref in mapping:
        return mapping[ref]
    stripped = ref.strip()
    if stripped and stripped != ref and stripped in mapping:
        return mapping[stripped]
    base = stripped or ref
    if "/" in base:
        slash = base.rfind("/")
        head, tail = base[:slash], base[slash:]
        colon = tail.find(":")
        if colon >= 0:
            hash_idx = tail.find("#")
            no_tag = head + tail[:colon] + (tail[hash_idx:] if hash_idx >= 0 else "")
            if no_tag != base and no_tag in mapping:
                return mapping[no_tag]
    return default


def _binding_canonical_ref(entry: Mapping[str, Any]) -> str:
    ref = str(entry.get("ref") or "").strip()
    if not ref:
        return ""
    provider = str(entry.get("provider") or "").strip() or "tensorhub"
    flavor = str(entry.get("flavor") or "").strip()
    tag = str(entry.get("tag") or "").strip()
    if provider == "tensorhub":
        out = ref if ("@" in ref or ":" in ref) else f"{ref}:{tag or 'latest'}"
    else:
        out = ref
    if flavor and "#" not in out:
        out = f"{out}#{flavor}"
    return out


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
    """{bare_ref_string: provider} from a loaded endpoint.lock manifest."""
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
            ref_key = _binding_canonical_ref(entry)
            if not ref_key:
                continue
            provider = str(entry.get("provider") or "").strip() or "tensorhub"
            index.setdefault(ref_key, provider)
            ref_bare = str(entry.get("ref") or "").strip()
            flavor = str(entry.get("flavor") or "").strip()
            if ref_bare:
                if flavor:
                    index.setdefault(f"{ref_bare}#{flavor}", provider)
                index.setdefault(ref_bare, provider)
    return index


# ---------------------------------------------------------------------------
# ensure_local: the ONE entry point
# ---------------------------------------------------------------------------


async def ensure_local(
    ref: str,
    *,
    provider: Optional[str] = None,
    snapshot: Any = None,
    cache_dir: Optional[Path] = None,
    hf_home: Optional[str] = None,
    hf_token: Optional[str] = None,
    civitai_api_key: str = "",
    allow_patterns: Sequence[str] = (),
    progress: Optional[ProgressFn] = None,
) -> Path:
    """Materialize ``ref`` on disk; return its local path.

    ``snapshot`` (tensorhub only) is the orchestrator-resolved manifest — a
    mapping with ``snapshot_digest`` + ``files``/``entries`` carrying presigned
    URLs or transfer grants. hf/civitai/modelscope refs need none.
    """
    base = Path(cache_dir) if cache_dir is not None else tensorhub_cas_dir()
    prov = provider or lookup_provider_for_ref(ref)
    parsed = parse_model_ref(ref, provider=prov)

    if parsed.provider == "tensorhub" and parsed.tensorhub is not None:
        if snapshot is None:
            # Hub-side residency bug (the orchestrator failed to pre-resolve),
            # not bad client input — RETRYABLE, never a client-visible 400.
            from ..api.errors import RetryableError

            raise RetryableError(
                f"tensorhub ref {ref!r} needs an orchestrator-resolved snapshot "
                "and none was provided"
            )
        from .cozy_snapshot import ensure_snapshot_async

        return await ensure_snapshot_async(
            base_dir=base, ref=parsed.tensorhub, resolved=snapshot, progress=progress,
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
            api_key=civitai_api_key or os.getenv("CIVITAI_API_KEY", "") or os.getenv("CIVITAI_TOKEN", ""),
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

    raise ValueError(f"unsupported model ref {ref!r} (provider={prov!r})")


def ensure_local_sync(ref: str, **kwargs: Any) -> Path:
    """Blocking wrapper for sync callers (CLI, trainer runtime)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(ensure_local(ref, **kwargs))
    # Called from inside a running loop's thread: run in a fresh thread/loop.
    out: dict[str, Any] = {}

    def _runner() -> None:
        try:
            out["v"] = asyncio.run(ensure_local(ref, **kwargs))
        except BaseException as e:  # noqa: BLE001
            out["e"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if "e" in out:
        raise out["e"]
    return out["v"]


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
      tokenizers) plus ONE weight set per directory — the requested ``flavor``
      when present, else bf16 > fp16 > untagged, safetensors preferred.
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


def _hf_stall_timeout_s() -> float:
    raw = os.environ.get("COZY_HF_DOWNLOAD_STALL_TIMEOUT_S", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 180.0


def _hf_wallclock_max_s() -> float:
    raw = os.environ.get("COZY_HF_DOWNLOAD_MAX_SECONDS", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 0.0


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
    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for hf model refs; install gen-worker with the HF extra."
        ) from e

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
        selected = None
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
        try:
            cap = int(os.getenv("COZY_HF_MAX_REPO_BYTES", "").strip() or _DEFAULT_MAX_REPO_BYTES)
        except ValueError:
            cap = _DEFAULT_MAX_REPO_BYTES
        if cap > 0 and total_hint > cap:
            raise RuntimeError(
                f"refusing an excessively large Hugging Face selection for {repo_id}: "
                f"{total_hint} bytes (limit {cap}; raise COZY_HF_MAX_REPO_BYTES to override)"
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
        stall_timeout=_hf_stall_timeout_s(),
        wall_clock_max=_hf_wallclock_max_s(),
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
    resp = requests.get(url, headers=headers, timeout=(30, 120))
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


def _civitai_select_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Downloadable safetensors files of a model version, primary first."""
    out: list[dict[str, Any]] = []
    for raw in payload.get("files") or []:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "").strip()
        url = str(raw.get("downloadUrl") or raw.get("download_url") or "").strip()
        if not url or not name.lower().endswith(".safetensors"):
            continue
        size = raw.get("sizeBytes")
        if not isinstance(size, int) or size <= 0:
            kb = raw.get("sizeKB")
            size = int(float(kb) * 1024) if isinstance(kb, (int, float)) and kb > 0 else 0
        hashes = raw.get("hashes") if isinstance(raw.get("hashes"), Mapping) else {}
        out.append({
            "id": int(raw.get("id") or 0),
            "name": Path(name).name,
            "url": url,
            "size_bytes": int(size),
            "sha256": str((hashes or {}).get("SHA256") or "").strip().lower(),
            "primary": bool(raw.get("primary")),
        })
    out.sort(key=lambda f: (0 if f["primary"] else 1, f["id"], f["name"]))
    return out


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
    with requests.get(url, headers=headers, stream=True, timeout=(60, 180)) as resp:
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
    if expected_size and written != expected_size:
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
) -> Path:
    """Blocking civitai model-version fetch (call via ``ensure_local`` /
    ``asyncio.to_thread``). Downloads the version's safetensors files with
    size + sha256 validation. Returns the single artifact path when the
    version has exactly one file, else the directory."""
    payload = fetch_civitai_model_version(version_id, api_key=api_key)
    files = _civitai_select_files(payload)
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

    local_paths: list[Path] = []
    for f in files:
        dst = out_dir / f["name"]
        local_paths.append(dst)
        if dst.exists() and (not f["size_bytes"] or dst.stat().st_size == f["size_bytes"]):
            done += f["size_bytes"]
            continue
        _civitai_stream_one(
            f["url"], dst,
            api_key=api_key,
            expected_size=f["size_bytes"],
            expected_sha256=f["sha256"],
            on_bytes=_on_bytes,
        )
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
    "ensure_local_sync",
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
