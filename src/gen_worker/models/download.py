"""ONE ensure-local path for the serving fleet, and it has ONE weight source."""

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

_provider_by_ref: Mapping[str, str] = {}


def _provider_index_keys(ref: str) -> tuple[str, str]:
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
    """Provider tag for ``ref`` from the index: exact normal-form match, then the repo-identity fallback."""
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
    """{normal_form_ref: provider} from a loaded endpoint.lock manifest."""
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


def _snapshot_ref(parsed: Any, raw: str) -> TensorhubRef:
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
    """Materialize ``ref`` on disk; return its local path."""
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
        from .errors import MissingSnapshotError

        raise MissingSnapshotError(
            f"tensorhub ref {ref!r} needs an orchestrator-resolved snapshot "
            "and none was provided"
        )

    if parsed.provider in ("hf", "civitai", "modelscope"):
        raise non_cas_refusal(ref=str(ref), provider=parsed.provider)

    raise ValidationError(f"unsupported model ref {ref!r} (provider={prov!r})")


def select_component_paths(
    paths: Sequence[str],
    components: Sequence[str],
) -> set[str]:
    """Narrow a repo file listing to declared pipeline COMPONENTS: every path under a ``<component>/`` subfolder, plus every root-level ``*.json`` (``model_index.json`` and siblings — always kept so downs..."""
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
    """Whether every declared component names an ACTUAL subfolder in ``paths``."""
    comps = {c.strip() for c in components if c and str(c).strip()}
    if not comps:
        return True
    dirs = {p.split("/", 1)[0] for p in paths if p and "/" in p}
    return bool(dirs & comps)


_HF_DOWNLOAD_STALL_TIMEOUT_S = 180.0
_HF_DOWNLOAD_MIN_WINDOW_BYTES = 8 * 1024 * 1024

class DownloadStalledError(RuntimeError):
    """Raised when a blocking snapshot download fails the progress-rate floor (less than ``min_window_bytes`` of new bytes within the stall window) — a bounded, observable failure instead of a silent hang."""


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
    floor = ProgressFloor(max(int(min_window_bytes), 1))
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


_CIVITAI_API = "https://civitai.com/api/v1"
_CIVITAI_AUTH_HOSTS = {"civitai.com", "www.civitai.com", "api.civitai.com"}
_CIVITAI_CHUNK = 4 * 1024 * 1024
_CIVITAI_JSON_TIMEOUT = (30.0, 120.0)
_CIVITAI_STREAM_TIMEOUT = (60.0, 180.0)
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
    import requests

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


_CIVITAI_GGUF_QUANT_PREFERENCE = (
    "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q4_0",
    "q3_k_m", "q3_k_s", "q2_k", "f16", "bf16", "f32",
)

_CIVITAI_GGUF_QTYPE_RE = re.compile(r"(?:ud-)?(?:i?q\d[0-9a-z_]*|bf16|f16|f32)")


def _civitai_gguf_quant_of(f: Mapping[str, Any]) -> str:
    if f.get("quant_type"):
        return str(f["quant_type"]).lower()
    url = str(f.get("url") or "")
    q = parse_qs(urlparse(url).query).get("quantType", [""])[0]
    if q:
        return q.strip().lower()
    m = _CIVITAI_GGUF_QTYPE_RE.search(str(f.get("name") or "").lower())
    return m.group(0) if m else ""


def _civitai_select_files(
    payload: Mapping[str, Any], *, gguf_quant: str | None = None,
) -> list[dict[str, Any]]:
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

    import requests

    headers: Dict[str, str] = {}
    if api_key and urlparse(url).hostname in _CIVITAI_AUTH_HOSTS:
        headers["Authorization"] = f"Bearer {api_key}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    h = hashlib.sha256()
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
    if expected_size and abs(written - expected_size) > _CIVITAI_SIZE_SLACK:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"civitai size mismatch for {dst.name}: expected {expected_size}, got {written}")
    observed = h.hexdigest().lower()
    if expected_sha256 and observed != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"civitai sha256 mismatch for {dst.name}")
    tmp.replace(dst)
    return written, observed


def _civitai_prior_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
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
    """Blocking civitai model-version fetch (call via ``ensure_local`` / ``asyncio.to_thread``)."""
    payload = fetch_civitai_model_version(version_id, api_key=api_key)
    files = _civitai_select_files(payload, gguf_quant=gguf_quant)
    if not files:
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

    import requests

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
                done = file_start
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
    "download_civitai",
    "fetch_civitai_model_version",
    "select_component_paths",
    "components_present",
    "DownloadStalledError",
    "set_provider_index",
    "lookup_provider_for_ref",
    "build_provider_index_from_manifest",
]
