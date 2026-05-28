from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, cast

import requests

from .refs import HuggingFaceRef
from .hf_selection import HFSelectionPolicy, finalize_diffusers_download, plan_diffusers_download

_LOG = logging.getLogger(__name__)


class DownloadStalledError(RuntimeError):
    """#379: raised when a HuggingFace snapshot download makes no byte progress
    for the stall window (or exceeds the wall-clock cap). Converts an indefinite
    silent hang in the prefetch path into a bounded, OBSERVABLE failure so the
    worker reports ``model.download.failed`` and the orchestrator reaps/replaces
    it, instead of the worker sitting forever in ``models_downloading``.
    """


def _hf_stall_timeout_s() -> float:
    """No-byte-progress window before a download is declared stalled (#379).
    `COZY_HF_DOWNLOAD_STALL_TIMEOUT_S` overrides; 0 disables stall detection."""
    raw = os.environ.get("COZY_HF_DOWNLOAD_STALL_TIMEOUT_S", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 180.0


def _hf_wallclock_max_s() -> float:
    """Hard wall-clock cap for a single repo download (#379 backstop, covers the
    no-progress-dir case). `COZY_HF_DOWNLOAD_MAX_SECONDS` overrides; 0 (default)
    disables it so legitimately-slow large downloads aren't false-tripped."""
    raw = os.environ.get("COZY_HF_DOWNLOAD_MAX_SECONDS", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 0.0


def _run_with_stall_watchdog(
    download_fn: Callable[[], str],
    *,
    label: str,
    progress_root: Optional[Path],
    progress_callback: Optional[Callable[[int, Optional[int]], None]],
    total_hint: Optional[int],
    stall_timeout: float,
    wall_clock_max: float,
    scan_bytes: Callable[[Path], int],
    poll_interval: float = 0.5,
) -> str:
    """Run a blocking ``download_fn`` on a daemon thread and watch it (#379).

    The watchdog doubles as the progress emitter: it scans bytes-on-disk under
    ``progress_root``, reports growth via ``progress_callback``, and — crucially —
    trips when there is NO byte progress for ``stall_timeout`` seconds (or the
    download exceeds ``wall_clock_max``), raising :class:`DownloadStalledError`.
    The old emit-only-when-growing poller stayed silent on a stall while the
    synchronous call blocked forever, producing the observed "models_downloading
    then silence" hang with no error and no liveness signal.

    Returns the download's local path, or re-raises the download's own exception,
    or raises :class:`DownloadStalledError` on stall / wall-clock breach. On a
    stall the still-wedged daemon thread is abandoned (it dies with the process /
    on reap); the caller's failure path emits ``model.download.failed``.
    """
    holder: Dict[str, Any] = {}

    def _run() -> None:
        try:
            holder["local"] = download_fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller thread
            holder["exc"] = exc
        finally:
            holder["done"] = True

    dl_thread = threading.Thread(target=_run, name="hf-snapshot-download", daemon=True)
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
                _LOG.error(
                    "hf download STALLED %s: no byte progress for %.0fs (downloaded=%d "
                    "bytes) — abandoning the wedged download thread and raising so the "
                    "worker reports model.download.failed instead of hanging silently in "
                    "models_downloading (#379).",
                    label, now - last_progress_at, last_bytes,
                )
                raise DownloadStalledError(
                    f"snapshot_download({label}) stalled: no progress for "
                    f"{stall_timeout:.0f}s after {last_bytes} bytes"
                )
        if wall_clock_max > 0 and (now - started_at) > wall_clock_max:
            _LOG.error(
                "hf download exceeded wall-clock cap %s: %.0fs > %.0fs — abandoning + "
                "raising (#379).",
                label, now - started_at, wall_clock_max,
            )
            raise DownloadStalledError(
                f"snapshot_download({label}) exceeded {wall_clock_max:.0f}s wall-clock cap"
            )

    if "exc" in holder:
        raise holder["exc"]
    return cast(str, holder["local"])


@dataclass(frozen=True)
class HuggingFaceDownloadResult:
    local_dir: Path


def _default_weight_precisions() -> list[str]:
    # Default: prefer 16-bit (bf16 OR fp16, whichever the repo has). fp32 is
    # NOT listed, so it is only ever selected as a last resort when no smaller
    # precision exists — never preferred. `COZY_HF_WEIGHT_PRECISIONS` overrides
    # this list (e.g. add `fp32` to allow downloading full-precision weights).
    return ["bf16", "fp16"]


def _weight_precisions_from_env() -> Optional[list[str]]:
    """Read `COZY_HF_WEIGHT_PRECISIONS` (comma-separated) if set.

    Honored as documented in the planner's error messages. Adding `fp32`
    flips on full-precision selection; otherwise it just reorders/limits the
    accepted precision family. Returns None when unset/empty so the caller
    falls back to `_default_weight_precisions()`.
    """
    raw = (os.getenv("COZY_HF_WEIGHT_PRECISIONS") or "").strip()
    if not raw:
        return None
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return parts or None


def _full_repo_download_requested() -> bool:
    """Whether `COZY_HF_FULL_REPO_DOWNLOAD` asks for the entire repo.

    When set truthy, selection is bypassed and the whole repo is downloaded
    (no `allow_patterns`). Escape hatch for repos the planner can't reduce.
    """
    raw = (os.getenv("COZY_HF_FULL_REPO_DOWNLOAD") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


class HuggingFaceHubDownloader:
    """
    Download Hugging Face repos using the official huggingface_hub cache/layout.

    This intentionally delegates caching/resume/locking behavior to huggingface_hub.
    """

    def __init__(self, hf_home: Optional[str] = None, hf_token: Optional[str] = None) -> None:
        # Caller passes hf_home/hf_token from gen_worker.config.Settings.
        # No env fallback — Settings is the single source of truth (issue #253).
        self.hf_home = (hf_home or "").strip() or None
        self.hf_token = (hf_token or "").strip() or None

    def download(
        self,
        ref: HuggingFaceRef,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
        allow_patterns: Optional[Sequence[str]] = None,
    ) -> HuggingFaceDownloadResult:
        try:
            from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required for hf: model refs. Install gen-worker with the Hugging Face extra."
            ) from e
        hf_hub_download_fn = cast(Callable[..., str], hf_hub_download)
        snapshot_download_fn = cast(Callable[..., str], snapshot_download)
        hf_hub_url_fn: Optional[Callable[..., str]] = None
        try:
            from huggingface_hub import hf_hub_url as _hf_hub_url
            hf_hub_url_fn = cast(Callable[..., str], _hf_hub_url)
        except Exception:
            pass

        # The wire format is bare ref + typed provider — HF refs reach
        # this loader without any prefix. No defensive strip needed.
        repo_id = (ref.repo_id or "").strip()
        if not repo_id:
            raise ValueError("empty hf repo_id")

        kwargs: dict[str, Any] = {"resume_download": True}
        if self.hf_home:
            kwargs["cache_dir"] = self.hf_home
        if self.hf_token:
            kwargs["token"] = self.hf_token

        # Explicit file selection (HFRepo.files(...)): fetch exactly these globs
        # and SKIP all diffusers-layout inference. This is how ComfyUI /
        # split-checkpoint repos (no model_index.json, weights nested under
        # subdirs, no root .safetensors) load from HF — and it avoids pulling
        # unused shards / sibling models.
        if allow_patterns:
            local = snapshot_download_fn(
                repo_id=repo_id,
                revision=ref.revision,
                allow_patterns=list(allow_patterns),
                **kwargs,
            )
            return HuggingFaceDownloadResult(local_dir=Path(local))

        # Safety guard against accidental huge downloads. Default 60GB so
        # large diffusers pipelines (e.g. FLUX.2-klein-9B at ~35GB: 9B
        # transformer + Qwen text encoder + VAE) download out of the box;
        # override with COZY_HF_MAX_REPO_BYTES for anything larger.
        try:
            max_total_bytes = int(os.getenv("COZY_HF_MAX_REPO_BYTES", "").strip() or 60_000_000_000)
        except (TypeError, ValueError):
            max_total_bytes = 60_000_000_000
        if max_total_bytes <= 0:
            max_total_bytes = 60_000_000_000

        full_repo = _full_repo_download_requested()
        policy = HFSelectionPolicy(
            components_override=None,
            include_optional_components=False,
            weight_precisions=_weight_precisions_from_env() or _default_weight_precisions(),
            allow_root_json=False,
        )

        # Best-effort local completeness check: if we already have a local snapshot folder that
        # contains all required files, skip network calls and downloads.
        local_snapshot = _try_get_local_snapshot_dir(
            snapshot_download=snapshot_download_fn,
            repo_id=repo_id,
            revision=ref.revision,
            cache_dir=self.hf_home,
            token=self.hf_token,
        )
        if local_snapshot is not None:
            # `snapshot_download(local_files_only=True)` only checks for the
            # `refs/<branch>` file — a partial cache (some LFS blobs still
            # unmaterialized) still returns the snapshot path. Walking it
            # then feeds an incomplete file list into the planner, which
            # raises (e.g. "weight shard referenced by ... not found in
            # repo: ...") because the index JSON references shards that
            # aren't on disk yet. The fast-path is only valid for *fully
            # complete* caches; on ANY failure fall through to the API
            # path below — never let the partial-cache RuntimeError bubble.
            try:
                local_files = _walk_relative_files(local_snapshot)
                model_index = _try_load_local_model_index(local_snapshot)
                if model_index is not None:
                    plan = plan_diffusers_download(model_index=model_index, repo_files=sorted(local_files), policy=policy)
                    needed = finalize_diffusers_download(
                        plan=plan,
                        repo_files=sorted(local_files),
                        weight_index_json_by_file=_load_local_weight_indexes(local_snapshot, plan.required_weight_index_files),
                    )
                    if needed.issubset(local_files) and not _has_incomplete_markers(local_snapshot):
                        return HuggingFaceDownloadResult(local_dir=local_snapshot)
            except Exception:
                pass  # partial cache or planner mismatch — fall through to API path

        api = HfApi(token=self.hf_token)
        repo_files: Sequence[str] = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=ref.revision)

        # Best-effort: get file sizes from list_repo_tree when available.
        repo_file_sizes: dict[str, int] = {}
        if hasattr(api, "list_repo_tree"):
            try:
                tree = api.list_repo_tree(repo_id=repo_id, repo_type="model", revision=ref.revision, recursive=True)
                for ent in tree:
                    # huggingface_hub RepoFile has .path and .size
                    p = getattr(ent, "path", None)
                    sz = getattr(ent, "size", None)
                    if isinstance(p, str) and isinstance(sz, int):
                        repo_file_sizes[p] = sz
            except Exception:
                repo_file_sizes = {}

        # Fetch model_index.json if present; otherwise infer components
        # from repo structure. Public quantized repos may be a single
        # root-level safetensors file plus small sidecars; those are valid
        # HFRepo bindings even though they are not full Diffusers folders.
        selected_files: set[str] | None = None
        model_index = _try_fetch_model_index_json(
            hf_hub_download=hf_hub_download_fn,
            repo_id=repo_id,
            revision=ref.revision,
            cache_dir=self.hf_home,
            token=self.hf_token,
        )
        if model_index is None and policy.components_override is None:
            inferred = _infer_diffusers_components_from_repo_files(repo_files)
            if inferred:
                policy = HFSelectionPolicy(
                    components_override=inferred,
                    include_optional_components=policy.include_optional_components,
                    weight_precisions=policy.weight_precisions,
                    allow_root_json=policy.allow_root_json,
                )
                model_index = {"_class_name": "Unknown"}
            else:
                selected_files = _select_root_safetensors_repo_files(repo_files)

        if model_index is None and selected_files is None:
            raise RuntimeError(
                f"huggingface ref {ref.repo_id!r} is missing model_index.json and no diffusers-like components could be inferred."
            )

        # Prefetch all sharded-weight index JSONs (small) so we can choose the best weight set per component.
        idx_json_by_file: dict[str, dict] = {}
        for pth in repo_files:
            if not pth.lower().endswith(".safetensors.index.json"):
                continue
            try:
                p = hf_hub_download_fn(
                    repo_id=repo_id,
                    revision=ref.revision,
                    filename=pth,
                    cache_dir=self.hf_home,
                    token=self.hf_token,
                )
                idx_json_by_file[pth] = json.loads(Path(p).read_text("utf-8"))
            except Exception:
                continue

        session = requests.Session()
        dtype_cache: dict[str, Optional[set[str]]] = {}

        def probe_safetensors_dtypes(rel_path: str) -> Optional[set[str]]:
            if rel_path in dtype_cache:
                return dtype_cache[rel_path]
            if hf_hub_url_fn is None:
                dtype_cache[rel_path] = None
                return None
            if not rel_path.lower().endswith(".safetensors"):
                dtype_cache[rel_path] = None
                return None

            url = hf_hub_url_fn(repo_id=repo_id, filename=rel_path, repo_type="model", revision=ref.revision)
            headers = {"Range": "bytes=0-7"}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"

            try:
                r = session.get(url, headers=headers, allow_redirects=True, timeout=60)
                r.raise_for_status()
                if len(r.content) != 8:
                    dtype_cache[rel_path] = None
                    return None
                (header_len,) = struct.unpack("<Q", r.content)
                if header_len <= 0 or header_len > (32 << 20):
                    dtype_cache[rel_path] = None
                    return None

                headers2 = {"Range": f"bytes=8-{8 + header_len - 1}"}
                if self.hf_token:
                    headers2["Authorization"] = f"Bearer {self.hf_token}"
                r2 = session.get(url, headers=headers2, allow_redirects=True, timeout=60)
                r2.raise_for_status()
                raw = json.loads(r2.content.decode("utf-8"))
                dtypes: set[str] = set()
                for name, meta in raw.items():
                    if name == "__metadata__":
                        continue
                    if not isinstance(meta, dict):
                        continue
                    dt = meta.get("dtype")
                    if isinstance(dt, str) and dt.strip():
                        dtypes.add(dt.strip())
                dtype_cache[rel_path] = dtypes or None
                return dtype_cache[rel_path]
            except Exception:
                dtype_cache[rel_path] = None
                return None

        if full_repo:
            # COZY_HF_FULL_REPO_DOWNLOAD escape hatch: bypass selection and let
            # huggingface_hub pull the entire repo (no allow_patterns).
            selected_files = set(repo_files)
        elif selected_files is None:
            plan = plan_diffusers_download(
                model_index=model_index or {},
                repo_files=repo_files,
                policy=policy,
                weight_index_json_by_file=idx_json_by_file,
                repo_file_sizes=repo_file_sizes,
                probe_safetensors_dtypes=probe_safetensors_dtypes,
            )

            selected_files = finalize_diffusers_download(plan=plan, repo_files=repo_files, weight_index_json_by_file=idx_json_by_file)

        total_hint: Optional[int] = None
        if repo_file_sizes:
            total_hint = sum(int(repo_file_sizes.get(p, 0) or 0) for p in selected_files)
            if total_hint > max_total_bytes:
                raise RuntimeError(
                    f"refusing to download an excessively large Hugging Face repo selection: {total_hint} bytes "
                    f"(limit {max_total_bytes} bytes)."
                )

        if not full_repo:
            # Deterministic order helps debugging and keeps behavior stable.
            kwargs["allow_patterns"] = sorted(selected_files)

        progress_root: Optional[Path] = None
        if progress_callback is not None:
            progress_root = _progress_local_dir(self.hf_home, repo_id, ref.revision, ref.flavor)
            progress_root.mkdir(parents=True, exist_ok=True)
            kwargs["local_dir"] = str(progress_root)
            try:
                progress_callback(0, total_hint)
            except Exception:
                pass

        # #379: bound the blocking download with a stall watchdog so a wedged
        # HTTP/hf_xet read surfaces as DownloadStalledError instead of an
        # indefinite silent hang in models_downloading.
        local = _run_with_stall_watchdog(
            lambda: snapshot_download_fn(repo_id=repo_id, revision=ref.revision, **kwargs),
            label=f"{repo_id}@{ref.revision}",
            progress_root=progress_root,
            progress_callback=progress_callback,
            total_hint=total_hint,
            stall_timeout=_hf_stall_timeout_s(),
            wall_clock_max=_hf_wallclock_max_s(),
            scan_bytes=_scan_bytes,
        )

        if progress_callback is not None:
            try:
                final_root = Path(local)
                final_bytes = _scan_bytes(final_root)
                progress_callback(final_bytes, total_hint or final_bytes or None)
            except Exception:
                pass
        return HuggingFaceDownloadResult(local_dir=Path(local))


def _try_fetch_model_index_json(
    *,
    hf_hub_download: Callable[..., str],
    repo_id: str,
    revision: str | None,
    cache_dir: str | None,
    token: str | None,
) -> Optional[dict]:
    try:
        index_path = hf_hub_download(
            repo_id=repo_id,
            revision=revision,
            filename="model_index.json",
            cache_dir=cache_dir,
            token=token,
        )
    except Exception:
        return None
    try:
        return json.loads(Path(index_path).read_text("utf-8"))
    except Exception:
        return None


def _infer_diffusers_components_from_repo_files(repo_files: Sequence[str]) -> list[str]:
    known = [
        "transformer",
        "unet",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "scheduler",
    ]
    present: list[str] = []
    for c in known:
        if any(p.startswith(f"{c}/") for p in repo_files):
            present.append(c)
    # Require at least one heavyweight component to avoid selecting random repos.
    if not any(c in present for c in ("transformer", "unet")):
        return []
    return present


def _select_root_safetensors_repo_files(repo_files: Sequence[str]) -> Optional[set[str]]:
    """Return a minimal safe selection for single-file/root-weight repos.

    Some public HF repos intentionally publish a pre-quantized component as a
    root-level ``.safetensors`` file rather than a full Diffusers directory.
    The worker still owns materialization for ``HFRepo`` bindings, so support
    that shape here instead of forcing endpoint code to call
    ``hf_hub_download`` directly.
    """
    root_safetensors = sorted(
        p for p in repo_files
        if "/" not in p and p.lower().endswith(".safetensors")
    )
    if not root_safetensors:
        return None

    selected = set(root_safetensors)
    sidecar_names = {
        ".gitattributes",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "model_index.json",
        "preprocessor_config.json",
        "README.md",
        "LICENSE",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
    }
    for p in repo_files:
        if "/" in p:
            continue
        name = p.rsplit("/", 1)[-1]
        low = name.lower()
        if name in sidecar_names or low.endswith(".json"):
            selected.add(p)
    return selected


def _try_get_local_snapshot_dir(
    *,
    snapshot_download: Callable[..., str],
    repo_id: str,
    revision: str | None,
    cache_dir: str | None,
    token: str | None,
) -> Optional[Path]:
    try:
        p = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
            resume_download=True,
            cache_dir=cache_dir,
            token=token,
        )
        return Path(p)
    except Exception:
        return None


def _walk_relative_files(root: Path) -> set[str]:
    out: set[str] = set()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        out.add(rel)
    return out


def _progress_local_dir(
    hf_home: str | None,
    repo_id: str,
    revision: str | None,
    flavor: str | None,
) -> Path:
    base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    safe = repo_id.replace("/", "--").replace(":", "_")
    rev = (revision or "main").replace("/", "--").replace(":", "_")
    flv = (flavor or "default").replace("/", "--").replace(":", "_")
    return base / "gen-worker-progress-snapshots" / safe / rev / flv


def _scan_bytes(root: Path) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    try:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
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


def _try_load_local_model_index(root: Path) -> Optional[dict]:
    p = root / "model_index.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return None


def _load_local_weight_indexes(root: Path, idx_paths: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rel in idx_paths:
        p = root / rel
        if not p.exists():
            continue
        try:
            out[rel] = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
    return out


def _has_incomplete_markers(root: Path) -> bool:
    # huggingface_hub uses *.incomplete markers for partial files.
    return any(p.is_file() and p.name.endswith(".incomplete") for p in root.rglob("*.incomplete"))
