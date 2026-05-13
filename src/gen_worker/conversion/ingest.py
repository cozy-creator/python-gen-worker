from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .hf_classifier import (
    ClassificationInputs,
    RepoClassification,
    RepoRefusal,
    RepoTooLarge,
    SelectionResult,
    _SIZE_REFUSE_BYTES,
    _SIZE_WARN_BYTES,
    classify_huggingface_repo,
    select_for_classification,
)
from .layout import (
    canonical_model_family_from_variant,
    infer_model_family_variant_from_hint,
)
from .safetensors_header import read_safetensors_header_metadata_from_hf

_log = logging.getLogger(__name__)

_MAX_SOURCE_FILE_BYTES = 20 * 1024 * 1024 * 1024
_MAX_REDIRECTS = 6
_CHUNK_BYTES = 1024 * 1024
_MAX_JSON_BYTES = 16 * 1024 * 1024
_RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}

# Stable per-(repo, revision) HF mirror directory used as
# `hf_hub_download(local_dir=...)`. Persistent across requests within
# the worker container's lifetime so a retry of the same ingest
# etag-checks against the existing files and skips re-download.
# Passing the per-request output_dir as `local_dir` (older behavior)
# meant the cache was per-request and every retry pulled bytes again.
_HF_MIRROR_ROOT = Path(
    os.getenv("CONVERSION_HF_MIRROR_DIR", "/var/cache/conversion/hf-mirror")
)


def _hf_mirror_dir_for(repo: str, rev: object) -> Path:
    safe_repo = str(repo or "").replace("/", "--").replace("..", "_")
    safe_rev = str(rev or "main").replace("/", "--").replace("..", "_")
    return _HF_MIRROR_ROOT / safe_repo / safe_rev
_FORBIDDEN_SOURCE_HEADER_NAMES = {
    "host",
    "authorization",
    "cookie",
    "content-length",
    "connection",
    "transfer-encoding",
    "upgrade",
    "proxy-authenticate",
    "proxy-authorization",
    "keep-alive",
    "te",
    "trailer",
}

# Force plain HTTP/LFS download path. huggingface_hub reads HF_HUB_DISABLE_XET
# from the environment at import / runtime — there is no Python API for this,
# so we have to set the env. In some environments the Hugging Face Xet flow
# can stall after xet token negotiation during large model pulls.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_HF_SOURCE_AUTH_HOSTS = {
    "huggingface.co",
    "www.huggingface.co",
    "api.huggingface.co",
    "hf.co",
    "cdn-lfs.huggingface.co",
}
_CIVITAI_SOURCE_AUTH_HOSTS = {
    "civitai.com",
    "www.civitai.com",
    "api.civitai.com",
    "image.civitai.com",
}
_CIVITAI_COMPONENT_ALIAS_PATHS = {
    "transformer": "transformer/diffusion_pytorch_model.safetensors",
    "unet": "unet/diffusion_pytorch_model.safetensors",
    "vae": "vae/diffusion_pytorch_model.safetensors",
    "text_encoder": "text_encoder/model.safetensors",
    "text_encoder_2": "text_encoder_2/model.safetensors",
}
_CIVITAI_REQUIRED_COMPONENTS_BY_FAMILY = {
    "flux": ("transformer", "text_encoder", "text_encoder_2", "vae"),
    "sdxl": ("unet", "text_encoder", "text_encoder_2", "vae"),
    "sd15_sd2": ("unet", "text_encoder", "vae"),
}
_CIVITAI_REQUIRED_COMPONENTS_BY_VARIANT = {
    "flux1": ("transformer", "text_encoder", "text_encoder_2", "vae"),
    "flux2": ("transformer", "text_encoder", "vae"),
    "flex2": ("transformer", "text_encoder", "text_encoder_2", "vae"),
    "z_image": ("transformer", "text_encoder", "vae"),
    "qwen_image": ("transformer", "text_encoder", "vae"),
    "auraflow": ("transformer", "text_encoder", "vae"),
    "wan22": ("transformer", "text_encoder", "vae"),
    "wan21": ("transformer", "text_encoder", "vae"),
    "sdxl": ("unet", "text_encoder", "text_encoder_2", "vae"),
    "sd15": ("unet", "text_encoder", "vae"),
}
_CIVITAI_COMPONENT_EXCLUDED_HINTS = (
    "lora",
    "lycoris",
    "adapter",
    "controlnet",
    "ip_adapter",
    "ip-adapter",
    "textual_inversion",
    "textual-inversion",
    "embedding",
)

class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        _ = req, fp, code, msg, headers, newurl
        return None


@dataclass(frozen=True)
class CivitaiImportCandidate:
    file_id: int
    name: str
    rel_path: str
    download_url: str
    primary: bool
    size_bytes: int | None
    hashes: dict[str, str]
    fingerprint: str
    size_bytes_exact: bool = False


@dataclass(frozen=True)
class CivitaiResolvedIdentity:
    model_version_id: int
    model_id: int
    base_model: str
    base_model_type: str
    air: str
    source_ref: str
    source_revision: str
    source_manifest_sha256: str
    file_fingerprints: dict[str, str]
    selected_files: list[CivitaiImportCandidate]
    selected_file_id: int | None
    pipeline_hint: str
    model_family: str
    model_family_variant: str = "unknown"


@dataclass(frozen=True)
class CivitaiFrontendURLInfo:
    original_url: str
    normalized_url: str
    model_id: int
    model_version_id: int | None


def _http_host_allowed(host: str, allowed_hosts: set[str]) -> bool:
    host_norm = str(host or "").strip().lower()
    if host_norm == "":
        return False
    if host_norm in allowed_hosts:
        return True
    for candidate in allowed_hosts:
        c = str(candidate or "").strip().lower()
        if c == "":
            continue
        if host_norm.endswith("." + c):
            return True
    return False


def _normalize_sha256(raw: object) -> str:
    value = str(raw or "").strip().lower()
    if value.startswith("sha256:"):
        value = value.split(":", 1)[1].strip().lower()
    if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
        return value
    return ""




def _normalize_source_auth_hosts(raw_hosts: object) -> set[str]:
    out: set[str] = set()
    if raw_hosts is None:
        return out
    if isinstance(raw_hosts, (list, tuple, set)):
        items = list(raw_hosts)
    else:
        items = [raw_hosts]
    for item in items:
        token = str(item or "").strip().lower()
        if token == "":
            continue
        parsed = urllib.parse.urlparse(token if "://" in token else f"https://{token}")
        host = str(parsed.hostname or token).strip().lower()
        if host != "":
            out.add(host)
    return out


def _normalize_source_headers(source_headers: Mapping[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if source_headers is None:
        return out
    for key, value in source_headers.items():
        name = str(key or "").strip()
        if name == "":
            continue
        lower_name = name.lower()
        if lower_name in _FORBIDDEN_SOURCE_HEADER_NAMES:
            raise ValueError(f"source_header_forbidden:{name}")
        if lower_name.startswith("proxy-"):
            raise ValueError(f"source_header_forbidden:{name}")
        text = str(value or "").strip()
        if text == "":
            continue
        out[name] = text
    return out


def _civitai_error_from_status(status_code: int, *, stage: str) -> ValueError:
    if status_code in {401, 403}:
        return ValueError("civitai_access_denied")
    if status_code == 404:
        return ValueError("civitai_not_found")
    stage_norm = str(stage or "").strip().lower()
    if stage_norm not in {"fetch", "download"}:
        stage_norm = "fetch"
    return ValueError(f"civitai_{stage_norm}_failed")


def parse_civitai_frontend_model_url(source_url: str) -> CivitaiFrontendURLInfo:
    raw = str(source_url or "").strip()
    if raw == "":
        raise ValueError("civitai_frontend_url_invalid")

    parsed = urllib.parse.urlparse(raw)
    scheme = str(parsed.scheme or "").strip().lower()
    host = str(parsed.hostname or "").strip().lower()
    if scheme not in {"http", "https"}:
        raise ValueError("civitai_frontend_url_invalid")
    if host not in {"civitai.com", "www.civitai.com"}:
        raise ValueError("civitai_frontend_url_not_civitai")

    path_parts = [part for part in str(parsed.path or "").split("/") if part != ""]
    if len(path_parts) < 2 or path_parts[0].lower() != "models":
        raise ValueError("civitai_frontend_url_invalid")
    model_id_raw = str(path_parts[1] or "").strip()
    if model_id_raw == "":
        raise ValueError("civitai_frontend_url_model_id_invalid")
    try:
        model_id = int(model_id_raw)
    except Exception:
        raise ValueError("civitai_frontend_url_model_id_invalid") from None
    if model_id <= 0:
        raise ValueError("civitai_frontend_url_model_id_invalid")

    query = urllib.parse.parse_qs(str(parsed.query or ""), keep_blank_values=True)
    model_version_id_raw = ""
    for key in ("modelVersionId", "modelversionid"):
        values = query.get(key)
        if values:
            model_version_id_raw = str(values[0] or "").strip()
            break
    model_version_id: int | None = None
    if model_version_id_raw != "":
        try:
            parsed_model_version = int(model_version_id_raw)
        except Exception:
            raise ValueError("civitai_frontend_url_model_version_invalid") from None
        if parsed_model_version <= 0:
            raise ValueError("civitai_frontend_url_model_version_invalid")
        model_version_id = parsed_model_version

    normalized_url = urllib.parse.urlunparse(
        (
            "https",
            host,
            str(parsed.path or ""),
            "",
            str(parsed.query or ""),
            "",
        )
    )
    return CivitaiFrontendURLInfo(
        original_url=raw,
        normalized_url=normalized_url,
        model_id=model_id,
        model_version_id=model_version_id,
    )


def _stream_http_download(
    source_url: str,
    *,
    output_path: Path | None,
    provider: str | None,
    civitai_stage: str | None,
    expected_size_bytes: int | None,
    expected_sha256: str,
    source_auth_token: str | None,
    source_auth_hosts: set[str] | None,
    source_headers: Mapping[str, str] | None,
    max_response_bytes: int | None,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> dict[str, str | int | bytes | None]:
    src = str(source_url or "").strip()
    if src == "":
        raise ValueError("missing_source_url")

    parsed0 = urllib.parse.urlparse(src)
    if parsed0.scheme not in {"http", "https"} or str(parsed0.hostname or "").strip() == "":
        raise ValueError("invalid_source_url")

    # Auth: caller must pass `source_auth_token` explicitly with the matching
    # `source_auth_hosts`. This function does not read env / fall back to a
    # bag of provider-default credentials — Settings is the single source of
    # truth and provider-specific tokens flow in via the caller chain.
    _ = provider  # accepted for callsite compatibility but unused
    token = str(source_auth_token or "").strip()
    explicit_hosts = _normalize_source_auth_hosts(source_auth_hosts)
    if token != "" and not explicit_hosts:
        base_host = str(parsed0.hostname or "").strip().lower()
        allowed_hosts: set[str] = ({base_host} if base_host != "" else set())
    else:
        allowed_hosts = set(explicit_hosts)
    header_overrides = _normalize_source_headers(source_headers)

    opener = urllib.request.build_opener(_NoRedirectHandler())
    current_url = src
    retry_attempts = 3
    retry_backoff_ms = 250

    sha = hashlib.sha256()
    downloaded = 0
    content_length_hint: int | None = None
    body_chunks: list[bytes] = []
    file_writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_writer = output_path.open("wb")

    try:
        for _ in range(_MAX_REDIRECTS + 1):
            parsed = urllib.parse.urlparse(current_url)
            scheme = str(parsed.scheme or "").strip().lower()
            host = str(parsed.hostname or "").strip().lower()
            if scheme not in {"http", "https"} or host == "":
                raise ValueError("invalid_source_url")

            headers = {"User-Agent": "conversion-endpoint/1", **header_overrides}
            if token != "" and _http_host_allowed(host, allowed_hosts):
                headers["Authorization"] = f"Bearer {token}"

            req = urllib.request.Request(current_url, method="GET", headers=headers)

            redirected = False
            resp = None
            for attempt in range(retry_attempts):
                try:
                    resp = opener.open(req, timeout=120)
                    break
                except urllib.error.HTTPError as exc:
                    status = int(getattr(exc, "code", 0) or 0)
                    if status in {301, 302, 303, 307, 308}:
                        loc = str(exc.headers.get("Location") or "").strip()
                        exc.close()
                        if loc == "":
                            raise ValueError("source_download_redirect_missing_location")
                        current_url = urllib.parse.urljoin(current_url, loc)
                        redirected = True
                        break
                    exc.close()
                    if status in _RETRYABLE_HTTP_STATUS_CODES and (attempt + 1) < retry_attempts:
                        if retry_backoff_ms > 0:
                            time.sleep((retry_backoff_ms / 1000.0) * float(attempt + 1))
                        continue
                    if str(provider or "").strip().lower() == "civitai":
                        raise _civitai_error_from_status(status, stage=str(civitai_stage or "fetch")) from None
                    raise ValueError(f"source_download_http_status:{status}") from None
                except urllib.error.URLError:
                    if (attempt + 1) < retry_attempts:
                        if retry_backoff_ms > 0:
                            time.sleep((retry_backoff_ms / 1000.0) * float(attempt + 1))
                        continue
                    if str(provider or "").strip().lower() == "civitai":
                        stage = str(civitai_stage or "fetch").strip().lower()
                        if stage not in {"fetch", "download"}:
                            stage = "download"
                        raise ValueError(f"civitai_{stage}_failed") from None
                    raise ValueError("source_download_failed") from None

            if redirected:
                continue
            if resp is None:
                raise ValueError("source_download_failed")

            with resp:
                raw_cl = str(resp.headers.get("Content-Length") or "").strip()
                if raw_cl != "":
                    try:
                        parsed_cl = int(raw_cl)
                        if parsed_cl >= 0:
                            content_length_hint = parsed_cl
                    except Exception:
                        content_length_hint = None
                if (
                    isinstance(expected_size_bytes, int)
                    and expected_size_bytes > 0
                    and isinstance(content_length_hint, int)
                    and content_length_hint > 0
                    and content_length_hint != expected_size_bytes
                ):
                    raise ValueError(
                        f"source_download_size_mismatch: expected={expected_size_bytes} got={content_length_hint}"
                    )

                while True:
                    chunk = resp.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > _MAX_SOURCE_FILE_BYTES:
                        raise ValueError(f"source_download_size_limit_exceeded:{_MAX_SOURCE_FILE_BYTES}")
                    if (
                        isinstance(max_response_bytes, int)
                        and max_response_bytes > 0
                        and downloaded > max_response_bytes
                    ):
                        raise ValueError(f"source_download_size_limit_exceeded:{max_response_bytes}")

                    sha.update(chunk)
                    if file_writer is not None:
                        file_writer.write(chunk)
                    else:
                        body_chunks.append(chunk)
                    if progress_callback is not None:
                        progress_callback(downloaded, content_length_hint)

            break
        else:
            raise ValueError("source_download_too_many_redirects")
    finally:
        if file_writer is not None:
            file_writer.close()

    if isinstance(expected_size_bytes, int) and expected_size_bytes > 0 and downloaded != expected_size_bytes:
        raise ValueError(f"source_download_size_mismatch: expected={expected_size_bytes} got={downloaded}")

    expected_sha = _normalize_sha256(expected_sha256)
    got_sha = sha.hexdigest().lower()
    if expected_sha != "" and expected_sha != got_sha:
        raise ValueError(f"source_download_sha256_mismatch: expected={expected_sha} got={got_sha}")

    return {
        "size_bytes": int(downloaded),
        "sha256": got_sha,
        "body": (b"".join(body_chunks) if output_path is None else None),
        "content_length_hint": content_length_hint,
        "final_url": current_url,
    }


def source_url_to_cas(
    source_url: str,
    output_path: Path,
    *,
    progress_callback: Callable[[int, int | None], None] | None = None,
    source_provider: str | None = None,
    source_auth_token: str = "",
    source_auth_hosts: list[str] | tuple[str, ...] | set[str] | None = None,
    source_headers: dict[str, str] | None = None,
    expected_size_bytes: int | None = None,
    expected_sha256: str = "",
) -> dict[str, str | int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    info = _stream_http_download(
        source_url,
        output_path=output_path,
        provider=source_provider,
        civitai_stage="download",
        expected_size_bytes=expected_size_bytes,
        expected_sha256=expected_sha256,
        source_auth_token=source_auth_token,
        source_auth_hosts=_normalize_source_auth_hosts(source_auth_hosts),
        source_headers=source_headers,
        max_response_bytes=None,
        progress_callback=progress_callback,
    )
    return {
        "output_path": str(output_path),
        "sha256": str(info.get("sha256") or ""),
        "size_bytes": int(info.get("size_bytes") or 0),
    }


def _split_hf_repo_ref(source_repo: str) -> tuple[str, str | None]:
    raw = str(source_repo or "").strip().strip("/")
    if raw == "":
        raise ValueError("huggingface source_repo is required")
    if raw.startswith("http://") or raw.startswith("https://"):
        raise ValueError("huggingface source_repo must be a repo id, not URL")
    # Accept owner/repo[:revision]
    revision: str | None = None
    if ":" in raw:
        base, tail = raw.rsplit(":", 1)
        if "/" in tail:
            # Likely part of repo path; keep as-is.
            base = raw
            tail = ""
        raw = str(base or "").strip()
        if str(tail or "").strip() != "":
            revision = str(tail or "").strip()
    if raw.count("/") < 1:
        raise ValueError("huggingface source_repo must be '<owner>/<repo>'")
    return raw, revision


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def resolve_huggingface_source_identity(
    source_repo: str,
    *,
    source_revision: str | None = None,
    hf_token: str = "",
) -> tuple[str, str]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for clone_huggingface source resolution") from exc

    repo_id, embedded_revision = _split_hf_repo_ref(source_repo)
    revision = str(source_revision or "").strip() or embedded_revision or "main"
    token = (hf_token or "").strip() or None
    api = HfApi(token=token)
    info = api.model_info(repo_id=repo_id, revision=revision)
    resolved_revision = str(getattr(info, "sha", "") or "").strip() or revision
    return repo_id, resolved_revision


def _normalize_hf_tree_entry_size(raw: object) -> int | None:
    try:
        if raw is None:
            return None
        size = int(raw)
        if size < 0:
            return None
        return size
    except Exception:
        return None


def _extract_hf_expected_sha256(node: object) -> str:
    lfs_meta = getattr(node, "lfs", None)
    if isinstance(lfs_meta, dict):
        for key in ("sha256", "oid"):
            sha = _normalize_sha256(lfs_meta.get(key))
            if sha != "":
                return sha
    elif lfs_meta is not None:
        for key in ("sha256", "oid"):
            sha = _normalize_sha256(getattr(lfs_meta, key, None))
            if sha != "":
                return sha
    return ""


def list_huggingface_repo_files(
    source_repo: str,
    *,
    source_revision: str | None = None,
    hf_token: str = "",
) -> dict[str, object]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for clone_huggingface source_repo ingestion") from exc

    repo_id, revision = resolve_huggingface_source_identity(
        source_repo, source_revision=source_revision, hf_token=hf_token,
    )
    token = (hf_token or "").strip() or None
    api = HfApi(token=token)
    tree = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        recursive=True,
        expand=True,
    )

    files: list[dict[str, object]] = []
    for node in tree:
        node_type = str(getattr(node, "type", "") or "").strip().lower()
        if node_type != "" and node_type != "file":
            continue
        rel_path = str(getattr(node, "path", "") or "").strip().replace("\\", "/").lstrip("/")
        if rel_path == "" or rel_path.endswith("/"):
            continue
        if ".." in rel_path.split("/"):
            continue
        size_bytes = _normalize_hf_tree_entry_size(getattr(node, "size", None))
        if size_bytes is None:
            lfs_meta = getattr(node, "lfs", None)
            if isinstance(lfs_meta, dict):
                size_bytes = _normalize_hf_tree_entry_size(lfs_meta.get("size"))
            elif lfs_meta is not None:
                size_bytes = _normalize_hf_tree_entry_size(getattr(lfs_meta, "size", None))
        # Hugging Face tree entries include directories with no size/type metadata.
        # We only ingest concrete file entries.
        if size_bytes is None:
            continue

        entry: dict[str, object] = {
            "path": rel_path,
            "size_bytes": size_bytes,
        }
        expected_sha256 = _extract_hf_expected_sha256(node)
        if expected_sha256 != "":
            entry["expected_sha256"] = expected_sha256
        files.append(entry)

    files.sort(key=lambda item: str(item.get("path") or ""))
    if not files:
        raise ValueError("huggingface repo has no files")

    total_bytes = 0
    total_known = True
    for item in files:
        size_raw = item.get("size_bytes")
        if isinstance(size_raw, int) and size_raw >= 0:
            total_bytes += int(size_raw)
        else:
            total_known = False
    return {
        "source_repo": repo_id,
        "source_revision": revision,
        "files": files,
        "total_bytes": int(total_bytes) if total_known else None,
    }


def _fetch_classification_inputs(
    repo_id: str,
    revision: str,
    listing_files: list[dict[str, object]],
    *,
    token: str | None,
    work_dir: Path,
) -> ClassificationInputs:
    """Pull just the cheap signals the classifier needs.

    - file paths + sizes (from the listing the caller already fetched)
    - tiny root config files (model_index.json / config.json /
      adapter_config.json / modules.json / config_sentence_transformers.json) —
      only when present in the listing
    - README YAML frontmatter (parsed via HfApi.model_info().card_data)
    - safetensors `__metadata__` block from the largest root .safetensors —
      only when the listing has any root .safetensors and no structured
      config (i.e. native-LoRA candidate territory)

    No weight bytes are downloaded.
    """
    from huggingface_hub import HfApi, hf_hub_download

    paths: list[str] = []
    sizes: dict[str, int] = {}
    for item in listing_files:
        rel = str(item.get("path") or "").strip().replace("\\", "/").lstrip("/")
        if not rel:
            continue
        paths.append(rel)
        size = item.get("size_bytes")
        if isinstance(size, int):
            sizes[rel] = int(size)

    root_set = {p for p in paths if "/" not in p}

    def _maybe_load_json(filename: str) -> Mapping[str, object] | None:
        if filename not in root_set:
            return None
        try:
            local = hf_hub_download(
                repo_id=repo_id, filename=filename, revision=revision,
                local_dir=str(work_dir), local_dir_use_symlinks=False, token=token,
            )
        except Exception as exc:
            _log.debug("could not fetch %s for classification: %s", filename, exc)
            return None
        try:
            with open(local, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            _log.debug("could not parse %s as JSON: %s", filename, exc)
            return None

    def _maybe_load_modules_json() -> Mapping[str, object] | list | None:
        # modules.json is typically a top-level list, not an object
        if "modules.json" not in root_set:
            return None
        try:
            local = hf_hub_download(
                repo_id=repo_id, filename="modules.json", revision=revision,
                local_dir=str(work_dir), local_dir_use_symlinks=False, token=token,
            )
            with open(local, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            _log.debug("could not load modules.json: %s", exc)
            return None

    model_index = _maybe_load_json("model_index.json")
    config = _maybe_load_json("config.json")
    adapter_config = _maybe_load_json("adapter_config.json")
    modules = _maybe_load_modules_json()
    cfg_st = _maybe_load_json("config_sentence_transformers.json")

    # README YAML frontmatter via HfApi.model_info (one HTTP call).
    frontmatter: dict[str, object] = {}
    try:
        api = HfApi(token=token)
        info = api.model_info(repo_id=repo_id, revision=revision or None)
        card_data = getattr(info, "card_data", None)
        if card_data is not None:
            # ModelCardData supports to_dict(); fall back to vars() if not
            try:
                frontmatter = dict(card_data.to_dict())  # type: ignore[union-attr]
            except Exception:
                try:
                    frontmatter = {k: v for k, v in vars(card_data).items()
                                   if not k.startswith("_")}
                except Exception:
                    frontmatter = {}
    except Exception as exc:
        _log.debug("could not fetch model_info for frontmatter: %s", exc)

    # Safetensors __metadata__ peek for native-LoRA detection.
    # Only relevant when there's a root .safetensors AND no structured
    # signals (model_index/config/adapter/modules) — that's the only path
    # native_lora classification could fire on.
    root_st_metadata: Mapping[str, str] | None = None
    root_st_path: str | None = None
    has_structured = bool(model_index or config or adapter_config or modules or cfg_st)
    if not has_structured:
        root_st = sorted(
            (p for p in root_set if p.lower().endswith(".safetensors")),
            key=lambda p: -int(sizes.get(p, 0)),
        )
        for candidate in root_st[:1]:  # only the largest
            md = read_safetensors_header_metadata_from_hf(
                repo_id=repo_id,
                filename=candidate,
                revision=revision or "main",
                token=token,
            )
            if md:
                root_st_metadata = md
                root_st_path = candidate
                break

    return ClassificationInputs(
        file_paths=paths,
        file_sizes=sizes,
        model_index_json=model_index,
        config_json=config,
        adapter_config_json=adapter_config,
        modules_json=modules,  # type: ignore[arg-type]
        config_sentence_transformers_json=cfg_st,
        readme_frontmatter=frontmatter,
        root_safetensors_metadata=root_st_metadata,
        root_safetensors_path=root_st_path,
    )


def download_huggingface_repo_files(
    source_repo: str,
    output_dir: Path,
    *,
    source_revision: str | None = None,
    source_dtype_preference: list[str] | None = None,
    gguf_quant: str | None = None,
    allow_large: bool = False,
    progress_callback: Callable[[int, int | None], None] | None = None,
    # Legacy parameter kept for compatibility — ignored under the new
    # classifier-driven flow. The classifier picks the strategy from the
    # repo's structured signals; there is no per-call layout override.
    source_layout_preference: str | None = None,
    # when the user requested multiple concrete
    # dtypes from a single repo (`outputs: [{dtype: f16}, {dtype: q4_k_m}]`),
    # the per-strategy selectors are run once per dtype and the results
    # union'd into the download set. The return dict carries `selections`
    # (list[dict]), one entry per requested dtype, so the caller can publish
    # N checkpoints under the same destination tag with distinct attributes.
    # Empty / None falls back to the single-dtype path (back-compat).
    dtype_outputs: list[str] | None = None,
    hf_token: str = "",
) -> dict[str, object]:
    """Classify the HF repo, select the minimal file set, and download it.

    See gen_worker.conversion.hf_classifier for the strategy table.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for clone_huggingface source_repo ingestion") from exc

    _ = source_layout_preference  # accepted for back-compat, no longer consulted
    dtype_pref = tuple(
        str(p or "").strip().lower() for p in (source_dtype_preference or []) if str(p or "").strip()
    ) or ("bf16", "fp16", "fp32")

    listing = list_huggingface_repo_files(
        source_repo, source_revision=source_revision, hf_token=hf_token,
    )
    repo_id = str(listing.get("source_repo") or "")
    revision = str(listing.get("source_revision") or "")
    all_files_list = [dict(item) for item in list(listing.get("files") or []) if isinstance(item, dict)]

    output_dir.mkdir(parents=True, exist_ok=True)
    token = (hf_token or "").strip() or None

    # Classification (no weight downloads).
    classification_inputs = _fetch_classification_inputs(
        repo_id, revision, all_files_list, token=token, work_dir=output_dir,
    )
    classification = classify_huggingface_repo(classification_inputs)
    if classification.refusal is not None:
        # Refusal is informative — propagate to caller.
        raise classification.refusal

    # Strategy-specific selection.
    # when caller passes dtype_outputs with >1 entry,
    # run the per-strategy selector once per dtype and union the file set.
    # Each entry carries its own attrs (with the concrete dtype stamped),
    # surfaced in `selections` of the return dict for the caller to publish
    # N checkpoints under the same tag.
    requested_dtypes = [str(d or "").strip().lower() for d in (dtype_outputs or []) if str(d or "").strip()]
    multi_selections: list[SelectionResult] = []
    if len(requested_dtypes) > 1:
        if classification.strategy == "gguf":
            from .hf_classifier import select_for_classification_multi
            multi_selections = select_for_classification_multi(
                classification, classification_inputs,
                gguf_quants=requested_dtypes,
                weight_index_json_by_file=None,
            )
        else:
            # transformers/diffusers Phase 2: detect side-by-side dtype
            # variants (`model.bf16.safetensors` + `model.fp16.safetensors`)
            # by re-running the selector with each requested dtype as the
            # primary preference; entries that don't resolve fall through
            # to the existing convert/quantize flow downstream.
            for dt in requested_dtypes:
                single_pref = (dt,) + tuple(p for p in dtype_pref if p != dt)
                try:
                    sel = select_for_classification(
                        classification, classification_inputs,
                        dtype_pref=single_pref,
                        gguf_quant=None,
                    )
                except Exception as exc:  # noqa: BLE001
                    _log.info(
                        "multi-dtype selector skipped dtype=%s: %s — caller falls "
                        "through to convert/quantize flow",
                        dt, exc,
                    )
                    continue
                # Stamp the concrete dtype on the per-checkpoint attrs.
                attrs = dict(sel.attrs)
                attrs.setdefault("dtype", dt)
                multi_selections.append(SelectionResult(
                    selected_paths=list(sel.selected_paths),
                    skipped_paths=list(sel.skipped_paths),
                    attrs=attrs,
                    pickle_files_refused=list(sel.pickle_files_refused),
                ))

    if multi_selections:
        # Union the file sets and dedupe; one download per unique path.
        union_paths: set[str] = set()
        for s in multi_selections:
            for p in s.selected_paths:
                union_paths.add(p)
        # Use the first selection's attrs as the canonical "primary" for
        # back-compat with single-dtype consumers; each per-checkpoint
        # attrs is preserved in `selections` of the return dict.
        selection = SelectionResult(
            selected_paths=sorted(union_paths),
            skipped_paths=[],
            attrs=dict(multi_selections[0].attrs),
            pickle_files_refused=list(multi_selections[0].pickle_files_refused),
        )
    else:
        selection = select_for_classification(
            classification, classification_inputs,
            dtype_pref=dtype_pref,
            gguf_quant=gguf_quant,
        )

    # Map the new selection back to the listing rows so we have sizes + sha256.
    selected_set = set(selection.selected_paths)
    files_to_download = [
        item for item in all_files_list
        if str(item.get("path") or "").strip().replace("\\", "/").lstrip("/") in selected_set
    ]
    if not files_to_download:
        raise ValueError("huggingface classification produced empty selection")

    # Size budget gate.
    total_bytes_hint = sum(int(item.get("size_bytes") or 0) for item in files_to_download)
    if total_bytes_hint > _SIZE_REFUSE_BYTES and not allow_large:
        raise RepoTooLarge(
            files_seen=[str(item.get("path") or "") for item in files_to_download],
            extra=f"selected_bytes={total_bytes_hint} cap={_SIZE_REFUSE_BYTES}",
        )
    if total_bytes_hint > _SIZE_WARN_BYTES:
        _log.warning(
            "clone selected size %.1f GB exceeds soft warn threshold (%d GB)",
            total_bytes_hint / (1024 ** 3),
            _SIZE_WARN_BYTES // (1024 ** 3),
        )

    # Structured selection summary log.
    skipped_bytes = sum(
        int(item.get("size_bytes") or 0)
        for item in all_files_list
        if str(item.get("path") or "").strip().replace("\\", "/").lstrip("/") not in selected_set
    )
    _log.info(
        "clone.ingest.selection_summary repo_type=%s runtime_library=%s subtype=%s "
        "selected_count=%d selected_bytes=%d skipped_count=%d skipped_bytes=%d "
        "pickle_files_refused=%d lineage_source=%s",
        classification.strategy,
        classification.runtime_library,
        classification.subtype or "",
        len(selection.selected_paths),
        total_bytes_hint,
        len(selection.skipped_paths),
        skipped_bytes,
        len(selection.pickle_files_refused),
        selection.attrs.get("lineage_source", ""),
    )

    # Pre-filter pickle files (defense-in-depth — selector should have already).
    cleaned: list[dict[str, object]] = []
    for item in files_to_download:
        rel_path = str(item.get("path") or "").strip().replace("\\", "/").lstrip("/")
        if not rel_path:
            continue
        low = rel_path.lower()
        if low.endswith((".bin", ".ckpt", ".pt", ".pth")):
            _log.error("pickle_blocklist_violation: %s slipped past selector", rel_path)
            continue
        size_bytes = item.get("size_bytes")
        if isinstance(size_bytes, int) and size_bytes > _MAX_SOURCE_FILE_BYTES:
            raise ValueError(
                f"huggingface source file exceeds 20GB policy limit: {rel_path} ({size_bytes} bytes)"
            )
        cleaned.append(item)
    files_to_download = cleaned
    files_total = len(files_to_download)

    # Parallel download with aggregate progress.
    # Default parallelism=4; HF hub rate-limits aggressively above ~8.
    # Per-file hash + size verification is preserved.
    import threading
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = 4
    if files_total < max_workers:
        max_workers = max(1, files_total)

    state_lock = threading.Lock()
    state = {
        "bytes_completed": 0,
        "bytes_in_flight": 0,
        "files_completed": 0,
        "files_in_flight": 0,
        "start_ts": _time.monotonic(),
        "last_emit_ts": 0.0,
    }

    def _emit_progress() -> None:
        if progress_callback is None:
            return
        with state_lock:
            now = _time.monotonic()
            if now - state["last_emit_ts"] < 1.5:
                return
            state["last_emit_ts"] = now
            elapsed = max(now - state["start_ts"], 1e-3)
            # `hf_hub_download` doesn't expose chunk-level progress, so the
            # poller below scans on-disk bytes between completions. The
            # display value is the larger of (a) completed-file bytes and
            # (b) live on-disk sample — monotonic across both.
            display_bytes = max(state["bytes_completed"], state["bytes_in_flight"])
            payload = {
                "bytes_written": display_bytes,
                "bytes_total": total_bytes_hint,
                "files_completed": state["files_completed"],
                "files_total": files_total,
                "files_in_flight": state["files_in_flight"],
                "bytes_per_sec_avg": float(display_bytes) / elapsed,
            }
        # Best-effort — older callers expect (bytes_written, total_bytes) only;
        # newer callers can introspect a richer dict by checking the type.
        try:
            progress_callback(payload["bytes_written"], payload["bytes_total"])  # type: ignore[arg-type]
        except Exception:
            pass

    # In-flight progress poller. `hf_hub_download` only signals completion,
    # so without this the user sees nothing for the multi-minute duration
    # of a single multi-GB file. We scan `output_dir` on a 5s cadence and
    # update `bytes_in_flight` so `_emit_progress` can report live bytes.
    _poll_stop = threading.Event()

    def _scan_on_disk_bytes() -> int:
        # Scan both the per-request output_dir AND the stable mirror_dir
        # the HF downloader writes into. Bytes accumulate in the mirror
        # during transfer; we hardlink into output_dir only on completion,
        # so without the mirror scan the poller would see nothing during
        # a multi-GB single-file download.
        total = 0
        seen: set[int] = set()
        for scan_root in (str(output_dir), str(mirror_dir)):
            try:
                for root, _dirs, files in os.walk(scan_root):
                    for f in files:
                        try:
                            st = os.stat(os.path.join(root, f))
                        except OSError:
                            continue
                        # Hardlinks share an inode — count once.
                        key = (st.st_dev, st.st_ino)
                        # Pack into a single int for the set.
                        ikey = hash(key)
                        if ikey in seen:
                            continue
                        seen.add(ikey)
                        total += int(st.st_size)
            except OSError:
                pass
        return total

    def _in_flight_poll_loop() -> None:
        while not _poll_stop.wait(5.0):
            on_disk = _scan_on_disk_bytes()
            with state_lock:
                # Monotonic — protects against transient filesystem
                # inconsistencies (e.g. rename-into-place atomicity).
                if on_disk > state["bytes_in_flight"]:
                    state["bytes_in_flight"] = on_disk
                # Force-bypass the 1.5s rate-limit on the next emit; this
                # is our only mid-flight signal.
                state["last_emit_ts"] = 0.0
            _emit_progress()

    # Stable mirror dir for this (repo, revision). Reused across requests so
    # huggingface_hub etag-checks against existing files and skips re-download
    # on retries.
    mirror_dir = _hf_mirror_dir_for(repo_id, revision)
    mirror_dir.mkdir(parents=True, exist_ok=True)

    def _download_one(item: dict) -> dict:
        rel_path = str(item.get("path") or "").strip().replace("\\", "/").lstrip("/")
        size_bytes = item.get("size_bytes")
        with state_lock:
            state["files_in_flight"] += 1
        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                revision=revision,
                local_dir=str(mirror_dir),
                token=token,
            )
        finally:
            pass
        # Hardlink the cached file into the per-request output_dir so
        # downstream layout/walk code sees the expected tree layout. Hardlink
        # is zero-cost on the same filesystem and keeps the central mirror
        # populated when this request's tmp dir is later cleaned up. Falls
        # back to copy on cross-device or filesystems without hardlink
        # support.
        target = Path(output_dir) / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            try:
                target.unlink()
            except OSError:
                pass
        try:
            os.link(cached_path, target)
        except OSError:
            shutil.copy2(cached_path, target)
        local_path = str(target)
        local_size = int(os.path.getsize(local_path))
        if local_size > _MAX_SOURCE_FILE_BYTES:
            raise ValueError(
                f"huggingface source file exceeds 20GB policy limit: {rel_path} ({local_size} bytes)"
            )
        if isinstance(size_bytes, int) and size_bytes > 0 and local_size != size_bytes:
            raise ValueError(f"source_download_size_mismatch: expected={size_bytes} got={local_size}")
        expected_sha256 = _normalize_sha256(item.get("expected_sha256"))
        if expected_sha256 != "":
            got_sha = _sha256_file(Path(local_path))
            if got_sha != expected_sha256:
                raise ValueError(f"source_download_sha256_mismatch: expected={expected_sha256} got={got_sha}")
        with state_lock:
            state["bytes_completed"] += local_size
            state["files_completed"] += 1
            state["files_in_flight"] -= 1
        _emit_progress()
        row: dict[str, object] = {
            "path": rel_path,
            "local_path": str(local_path),
            "size_bytes": int(local_size),
        }
        if expected_sha256 != "":
            row["expected_sha256"] = expected_sha256
        return row

    poller = threading.Thread(target=_in_flight_poll_loop, name="hf-ingest-poller", daemon=True)
    poller.start()
    materialized: list[dict[str, object]] = []
    try:
        if max_workers <= 1 or files_total <= 1:
            # Single-thread path (legacy behavior; same code path)
            for item in files_to_download:
                materialized.append(_download_one(item))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_download_one, item) for item in files_to_download]
                try:
                    for fut in as_completed(futures):
                        materialized.append(fut.result())
                except Exception:
                    # Cancel remaining work; let the ThreadPoolExecutor wind down.
                    for fut in futures:
                        fut.cancel()
                    raise
    finally:
        _poll_stop.set()
        poller.join(timeout=2.0)

    total_bytes = sum(int(v.get("size_bytes") or 0) for v in materialized)
    # Final progress flush (overrides the 1.5s rate limit) so callers see the
    # last 100% tick.
    if progress_callback is not None:
        try:
            progress_callback(total_bytes, total_bytes_hint)
        except Exception:
            pass

    # Lineage-resolved log event (LoRA-only meaningful but always emit).
    _log.info(
        "clone.ingest.lineage_resolved repo=%s strategy=%s runtime=%s lineage=%s source_kind=%s",
        repo_id,
        classification.strategy,
        classification.runtime_library,
        selection.attrs.get("base_model_lineage", ""),
        selection.attrs.get("lineage_source", ""),
    )

    # Map strategy → legacy `selected_source_layout` value used by callers
    # that haven't been ported to attrs yet.
    legacy_layout = {
        "diffusers": "diffusers",
        "transformers": "transformers",
        "peft_canonical": "peft",
        "native_lora": "native_lora",
        "sentence_transformers": "sentence_transformers",
        "gguf": "gguf",
        "aio_singlefile": "singlefile",
    }.get(classification.strategy, "unknown")

    return {
        "source_repo": repo_id,
        "source_revision": revision,
        "files": materialized,
        "file_count": len(materialized),
        "total_bytes": int(total_bytes),
        # New classification fields.
        "strategy": classification.strategy,
        "runtime_library": classification.runtime_library,
        "subtype": classification.subtype,
        "attrs": dict(selection.attrs),
        "pickle_files_refused": list(selection.pickle_files_refused),
        "selected_count": len(selection.selected_paths),
        "skipped_count": len(selection.skipped_paths),
        "selected_bytes": int(total_bytes_hint),
        "skipped_bytes": int(skipped_bytes),
        # Legacy fields for back-compat with _shared.py + downstream consumers
        # that haven't been updated to the new attrs/runtime_library shape.
        "selected_source_layout": legacy_layout,
        "selected_aio_path": (
            selection.selected_paths[0]
            if classification.strategy == "aio_singlefile" and selection.selected_paths
            else ""
        ),
        "source_layout_selection_reason": classification.detection_reason,
        "source_layout_detection_reason": classification.detection_reason,
        "source_dtype_by_component": {},
        "source_dtype_dropped_paths": [],
        "source_dtype_preference": list(dtype_pref),
        # per-checkpoint selections when the caller
        # requested multiple concrete dtypes from a single repo. Empty when
        # `dtype_outputs` is empty / single-element. Each entry mirrors the
        # singleton fields above scoped to one resolved dtype.
        "selections": [
            {
                "attrs": dict(s.attrs),
                "selected_paths": list(s.selected_paths),
                "selected_count": len(s.selected_paths),
            }
            for s in multi_selections
        ],
    }


def _parse_civitai_size_bytes(raw: object) -> int | None:
    if isinstance(raw, int) and raw > 0:
        return int(raw)
    if isinstance(raw, float) and raw > 0:
        return int(round(raw))
    try:
        txt = str(raw or "").strip()
        if txt == "":
            return None
        if "." in txt:
            value = float(txt)
            if value <= 0:
                return None
            return int(round(value))
        value_i = int(txt)
        if value_i <= 0:
            return None
        return value_i
    except Exception:
        return None


def _normalize_civitai_hashes(raw: object) -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        items = raw.items()
    else:
        return out
    for k, v in items:
        key = str(k or "").strip().upper()
        value = str(v or "").strip()
        if key == "" or value == "":
            continue
        out[key] = value
    return out


def _civitai_file_rel_path(*, file_id: int, name: str) -> str:
    rel = Path(str(name or "").strip()).name
    if rel == "":
        rel = f"civitai-{int(file_id)}.safetensors"
    if not rel.lower().endswith(".safetensors"):
        rel = f"{rel}.safetensors"
    return rel


def _civitai_file_fingerprint(
    *,
    file_id: int,
    name: str,
    size_bytes: int | None,
    hashes: dict[str, str],
) -> str:
    # Prefer cryptographic hashes for stable identity when supplied by Civitai.
    for key in ("SHA256", "BLAKE3", "MD5"):
        value = str(hashes.get(key) or "").strip().lower()
        if value != "":
            return f"{key.lower()}:{value}"

    parts = [
        f"id:{int(file_id)}",
        f"name:{str(name or '').strip().lower()}",
        f"size_bytes:{int(size_bytes or 0)}",
    ]
    for key in sorted(hashes.keys()):
        value = str(hashes.get(key) or "").strip().lower()
        if value != "":
            parts.append(f"hash:{key.lower()}={value}")
    return "|".join(parts)


def _normalize_civitai_component_name(raw: str) -> str:
    text = str(raw or "").strip().lower()
    chars: list[str] = []
    for ch in text:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            chars.append(ch)
    return "".join(chars)


def _civitai_component_for_name(raw_name: str) -> str:
    normalized = _normalize_civitai_component_name(raw_name)
    if normalized == "":
        return ""
    if any(hint in normalized for hint in _CIVITAI_COMPONENT_EXCLUDED_HINTS):
        return ""
    if any(tok in normalized for tok in ("textencoder2", "clipg", "t5", "t5xxl")):
        return "text_encoder_2"
    if any(tok in normalized for tok in ("textencoder", "clipl")):
        return "text_encoder"
    if any(tok in normalized for tok in ("autoencoder", "vae", "ae")):
        return "vae"
    if "unet" in normalized:
        return "unet"
    if "transformer" in normalized or "flux" in normalized:
        return "transformer"
    return ""


def _civitai_required_components(
    *,
    model_family: str,
    model_family_variant: str,
    all_components: set[str] | None = None,
) -> tuple[str, ...]:
    variant = str(model_family_variant or "").strip().lower()
    by_variant = _CIVITAI_REQUIRED_COMPONENTS_BY_VARIANT.get(variant)
    if by_variant is not None:
        return by_variant

    family = str(model_family or "").strip().lower()
    if family == "flux":
        # FLUX.1 commonly includes text_encoder_2; FLUX.2/QwenImage/Z-Image generally do not.
        if all_components is not None and "text_encoder_2" not in all_components:
            return _CIVITAI_REQUIRED_COMPONENTS_BY_VARIANT["flux2"]
        return _CIVITAI_REQUIRED_COMPONENTS_BY_VARIANT["flux1"]
    if family == "auraflow":
        return _CIVITAI_REQUIRED_COMPONENTS_BY_VARIANT["auraflow"]
    if family == "wan":
        return _CIVITAI_REQUIRED_COMPONENTS_BY_VARIANT["wan22"]
    if family in _CIVITAI_REQUIRED_COMPONENTS_BY_FAMILY:
        return _CIVITAI_REQUIRED_COMPONENTS_BY_FAMILY[family]
    return ()


def _civitai_required_components_for_selected(
    selected_component: str,
    all_components: set[str],
) -> tuple[str, ...]:
    if selected_component == "transformer":
        return _civitai_required_components(
            model_family="flux",
            model_family_variant=("flux1" if "text_encoder_2" in all_components else "flux2"),
            all_components=all_components,
        )
    if selected_component == "unet":
        if "text_encoder_2" in all_components:
            return _CIVITAI_REQUIRED_COMPONENTS_BY_FAMILY["sdxl"]
        return _CIVITAI_REQUIRED_COMPONENTS_BY_FAMILY["sd15_sd2"]
    if selected_component == "text_encoder_2":
        if "transformer" in all_components:
            return _CIVITAI_REQUIRED_COMPONENTS_BY_FAMILY["flux"]
        return _CIVITAI_REQUIRED_COMPONENTS_BY_FAMILY["sdxl"]
    if selected_component == "text_encoder":
        if "transformer" in all_components:
            return _civitai_required_components(
                model_family="flux",
                model_family_variant=("flux1" if "text_encoder_2" in all_components else "flux2"),
                all_components=all_components,
            )
        if "text_encoder_2" in all_components:
            return _CIVITAI_REQUIRED_COMPONENTS_BY_FAMILY["sdxl"]
        return _CIVITAI_REQUIRED_COMPONENTS_BY_FAMILY["sd15_sd2"]
    if selected_component == "vae":
        if "transformer" in all_components:
            return _civitai_required_components(
                model_family="flux",
                model_family_variant=("flux1" if "text_encoder_2" in all_components else "flux2"),
                all_components=all_components,
            )
        if "text_encoder_2" in all_components:
            return _CIVITAI_REQUIRED_COMPONENTS_BY_FAMILY["sdxl"]
        return _CIVITAI_REQUIRED_COMPONENTS_BY_FAMILY["sd15_sd2"]
    return ()


def _build_civitai_candidates(files: list[dict[str, object]]) -> list[CivitaiImportCandidate]:
    candidates: list[dict[str, object]] = []
    for row in files:
        download_url = str(row.get("download_url") or "").strip()
        name = str(row.get("name") or "").strip()
        if download_url == "" or not name.lower().endswith(".safetensors"):
            continue
        candidates.append(dict(row))

    if not candidates:
        return []

    candidates.sort(
        key=lambda row: (
            0 if bool(row.get("primary")) else 1,
            int(row.get("id") or 0),
            str(row.get("name") or "").strip().lower(),
        )
    )

    used_paths: set[str] = set()
    out: list[CivitaiImportCandidate] = []
    for row in candidates:
        file_id = int(row.get("id") or 0)
        name = str(row.get("name") or "").strip()
        hashes = dict(row.get("hashes") or {})
        size_bytes = row.get("size_bytes")
        size_norm = int(size_bytes) if isinstance(size_bytes, int) and size_bytes > 0 else None
        rel_path = _civitai_file_rel_path(file_id=file_id, name=name)
        if rel_path in used_paths:
            stem = rel_path[: -len(".safetensors")] if rel_path.lower().endswith(".safetensors") else rel_path
            rel_path = f"{stem}-{file_id}.safetensors"
        used_paths.add(rel_path)
        out.append(
            CivitaiImportCandidate(
                file_id=file_id,
                name=name,
                rel_path=rel_path,
                download_url=str(row.get("download_url") or "").strip(),
                primary=bool(row.get("primary")),
                size_bytes=size_norm,
                size_bytes_exact=bool(row.get("size_bytes_exact")),
                hashes=hashes,
                fingerprint=_civitai_file_fingerprint(
                    file_id=file_id,
                    name=name,
                    size_bytes=size_norm,
                    hashes=hashes,
                ),
            )
        )
    return out


def _expand_selected_civitai_component_bundle(
    all_candidates: list[CivitaiImportCandidate],
    selected: CivitaiImportCandidate,
) -> list[CivitaiImportCandidate]:
    selected_component = _civitai_component_for_name(selected.name)
    if selected_component == "":
        return [selected]

    component_to_candidate: dict[str, CivitaiImportCandidate] = {}
    all_components: set[str] = set()
    for item in all_candidates:
        component = _civitai_component_for_name(item.name)
        if component == "":
            continue
        all_components.add(component)
        existing = component_to_candidate.get(component)
        if existing is None:
            component_to_candidate[component] = item
            continue
        # Stable tie-break: prefer explicit primary first, then lower file id.
        existing_rank = (0 if existing.primary else 1, int(existing.file_id))
        item_rank = (0 if item.primary else 1, int(item.file_id))
        if item_rank < existing_rank:
            component_to_candidate[component] = item

    required = _civitai_required_components_for_selected(selected_component, all_components)
    if not required:
        return [selected]

    out: list[CivitaiImportCandidate] = [selected]
    seen_ids: set[int] = {int(selected.file_id)}
    for component in required:
        candidate = component_to_candidate.get(component)
        if candidate is None:
            continue
        cid = int(candidate.file_id)
        if cid in seen_ids:
            continue
        out.append(candidate)
        seen_ids.add(cid)
    return out


def _select_civitai_import_files(
    files: list[dict[str, object]],
    *,
    selected_file_id: int | None,
) -> list[CivitaiImportCandidate]:
    requested_file_id = int(selected_file_id or 0)
    if not files:
        return []

    all_candidates = _build_civitai_candidates(files)
    if not all_candidates:
        return []

    if requested_file_id > 0:
        for candidate in all_candidates:
            if int(candidate.file_id) != requested_file_id:
                continue
            return _expand_selected_civitai_component_bundle(all_candidates, candidate)
        return []

    return all_candidates


def _civitai_materialize_component_aliases(
    *,
    output_dir: Path,
    files: list[dict[str, object]],
    model_family: str,
    model_family_variant: str = "unknown",
) -> tuple[list[dict[str, object]], dict[str, str], list[str]]:
    existing_paths = {
        str(item.get("path") or "").strip().replace("\\", "/").lstrip("/")
        for item in files
        if isinstance(item, dict)
    }
    component_rows: dict[str, dict[str, object]] = {}
    for item in files:
        if not isinstance(item, dict):
            continue
        rel_path = str(item.get("path") or "").strip().replace("\\", "/").lstrip("/")
        if rel_path == "":
            continue
        hinted_name = str(item.get("file_name") or Path(rel_path).name).strip()
        component = _civitai_component_for_name(hinted_name)
        if component == "":
            component = _civitai_component_for_name(rel_path)
        if component == "":
            continue
        existing = component_rows.get(component)
        if existing is None:
            component_rows[component] = item
            continue
        existing_rank = (0 if bool(existing.get("primary")) else 1, int(existing.get("file_id") or 0))
        item_rank = (0 if bool(item.get("primary")) else 1, int(item.get("file_id") or 0))
        if item_rank < existing_rank:
            component_rows[component] = item

    required = _civitai_required_components(
        model_family=model_family,
        model_family_variant=model_family_variant,
        all_components=set(component_rows.keys()),
    )
    if not required:
        return files, {}, []

    aliases: dict[str, str] = {}
    out: list[dict[str, object]] = list(files)
    for component in required:
        row = component_rows.get(component)
        if row is None:
            continue
        alias_rel_path = str(_CIVITAI_COMPONENT_ALIAS_PATHS.get(component) or "").strip()
        if alias_rel_path == "" or alias_rel_path in existing_paths:
            continue

        source_local = Path(str(row.get("local_path") or "").strip())
        if not source_local.exists() or not source_local.is_file():
            continue
        alias_local = output_dir / alias_rel_path
        alias_local.parent.mkdir(parents=True, exist_ok=True)
        if alias_local.exists() or alias_local.is_symlink():
            alias_local.unlink()
        try:
            os.symlink(str(source_local.resolve()), str(alias_local))
        except Exception:
            shutil.copyfile(str(source_local), str(alias_local))

        aliases[component] = alias_rel_path
        existing_paths.add(alias_rel_path)
        out.append(
            {
                "path": alias_rel_path,
                "local_path": str(alias_local),
                "size_bytes": int(row.get("size_bytes") or int(source_local.stat().st_size)),
                "sha256": str(row.get("sha256") or ""),
                "file_id": int(row.get("file_id") or 0),
                "file_name": str(row.get("file_name") or ""),
                "file_fingerprint": str(row.get("file_fingerprint") or ""),
                "primary": False,
                "hashes": dict(row.get("hashes") or {}),
                "download_url": str(row.get("download_url") or ""),
                "component_alias": component,
                "aliased_from_path": str(row.get("path") or ""),
            }
        )

    missing_components = [component for component in required if component not in aliases and component not in component_rows]
    return out, aliases, missing_components


def _normalize_letters_digits(raw: str) -> str:
    chars: list[str] = []
    for ch in str(raw or "").strip().lower():
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            chars.append(ch)
    return "".join(chars)


def infer_singlefile_pipeline_hint_from_civitai_metadata(
    *,
    base_model: str,
    base_model_type: str,
    air: str,
) -> str:
    combined = _normalize_letters_digits(f"{base_model} {base_model_type} {air}")
    if combined == "":
        return ""
    if "auraflow" in combined:
        return "AuraFlowPipeline"
    if "flex2" in combined:
        return "FluxPipeline"
    if "wan22" in combined:
        return "WanPipeline"
    if "wan21" in combined or ("wan" in combined and any(tok in combined for tok in ("video", "i2v", "t2v", "vace"))):
        return "WanPipeline"
    if "qwenimageedit" in combined:
        return "QwenImageEditPipeline"
    if "qwenimage" in combined:
        return "QwenImagePipeline"
    if "zimageturbo" in combined or "zimage" in combined:
        return "ZImagePipeline"
    if ("flux2" in combined and "klein" in combined) or ("flux" in combined and "klein" in combined):
        return "Flux2KleinPipeline"
    if "flux2" in combined:
        return "Flux2Pipeline"
    if "flux" in combined:
        return "FluxPipeline"
    if "sdxl" in combined or "illustrious" in combined or "pony" in combined:
        return "StableDiffusionXLPipeline"
    if "stablediffusion" in combined or "sd15" in combined or "sd2" in combined:
        return "StableDiffusionPipeline"
    return ""


def infer_model_family_from_civitai_metadata(
    *,
    base_model: str,
    base_model_type: str,
    air: str,
    selected_files: list[CivitaiImportCandidate],
) -> str:
    variant = infer_model_family_variant_from_civitai_metadata(
        base_model=base_model,
        base_model_type=base_model_type,
        air=air,
        selected_files=selected_files,
    )
    canonical = canonical_model_family_from_variant(variant)
    if canonical != "unknown":
        return canonical

    combined = _normalize_letters_digits(f"{base_model} {base_model_type} {air}")
    if "sdxl" in combined or "illustrious" in combined or "pony" in combined:
        return "sdxl"
    if any(tok in combined for tok in ("sd15", "sd1", "sd2", "stablediffusion1", "stablediffusion2")):
        return "sd15_sd2"
    if any(tok in combined for tok in ("flux", "zimage", "zimageturbo", "qwenimage", "qwenimageedit")):
        return "flux"
    if "flex2" in combined:
        return "flux"
    if "auraflow" in combined:
        return "auraflow"
    if "wan22" in combined or "wan21" in combined or ("wan" in combined and any(tok in combined for tok in ("video", "i2v", "t2v", "vace"))):
        return "wan"

    for item in selected_files:
        name = str(item.name or "").strip().lower()
        if "sdxl" in name:
            return "sdxl"
        if any(tok in name for tok in ("flux", "zimage", "qwen")):
            return "flux"
        if "auraflow" in name:
            return "auraflow"
    return "unknown"


def infer_model_family_variant_from_civitai_metadata(
    *,
    base_model: str,
    base_model_type: str,
    air: str,
    selected_files: list[CivitaiImportCandidate],
) -> str:
    combined = _normalize_letters_digits(f"{base_model} {base_model_type} {air}")
    if "auraflow" in combined:
        return "auraflow"
    if "flex2" in combined:
        return "flex2"
    if "wan22" in combined:
        return "wan22"
    if "wan21" in combined or ("wan" in combined and any(tok in combined for tok in ("video", "i2v", "t2v", "vace"))):
        return "wan21"
    if "qwenimageedit" in combined or "qwenimage" in combined:
        return "qwen_image"
    if "zimageturbo" in combined or "zimage" in combined:
        return "z_image"
    if "flux2" in combined or ("flux" in combined and "klein" in combined):
        return "flux2"
    if "flux" in combined:
        return "flux1"
    if "sdxl" in combined or "stablediffusionxl" in combined or "illustrious" in combined or "pony" in combined:
        return "sdxl"
    if (
        "sd15" in combined
        or "stablediffusion15" in combined
        or "stablediffusionv15" in combined
        or "stablediffusionv1" in combined
    ):
        return "sd15"

    for item in selected_files:
        name = str(item.name or "").strip().lower()
        detected = infer_model_family_variant_from_hint(name)
        if detected != "unknown":
            return detected
    return "unknown"


def _fetch_civitai_json_url(url: str, *, stage: str) -> dict[str, object]:
    info = _stream_http_download(
        str(url or "").strip(),
        output_path=None,
        provider="civitai",
        civitai_stage=stage,
        expected_size_bytes=None,
        expected_sha256="",
        source_auth_token=None,
        source_auth_hosts=None,
        source_headers=None,
        max_response_bytes=_MAX_JSON_BYTES,
        progress_callback=None,
    )
    body = info.get("body")
    if not isinstance(body, (bytes, bytearray)):
        raise ValueError("civitai_fetch_failed")
    try:
        parsed = json.loads(body)
    except Exception:
        raise ValueError("civitai_fetch_failed") from None
    if not isinstance(parsed, dict):
        raise ValueError("civitai_fetch_failed")
    return {str(k): v for k, v in parsed.items()}


def fetch_civitai_model_version(model_version_id: int) -> dict[str, object]:
    if int(model_version_id) <= 0:
        raise ValueError("civitai_model_version_id is required")
    meta_url = f"https://civitai.com/api/v1/model-versions/{int(model_version_id)}"
    return _fetch_civitai_json_url(meta_url, stage="fetch")


def fetch_civitai_model(model_id: int) -> dict[str, object]:
    if int(model_id) <= 0:
        raise ValueError("civitai_frontend_url_model_id_invalid")
    model_url = f"https://civitai.com/api/v1/models/{int(model_id)}"
    return _fetch_civitai_json_url(model_url, stage="fetch")


def resolve_civitai_frontend_model_url(
    source_url: str,
    *,
    explicit_model_version_id: int | None = None,
) -> CivitaiFrontendURLInfo:
    info = parse_civitai_frontend_model_url(source_url)
    explicit_id = int(explicit_model_version_id or 0)
    if explicit_id < 0:
        raise ValueError("civitai_frontend_url_model_version_invalid")

    if info.model_version_id is not None:
        if explicit_id > 0 and explicit_id != int(info.model_version_id):
            raise ValueError("civitai_frontend_url_model_version_mismatch")
        return CivitaiFrontendURLInfo(
            original_url=info.original_url,
            normalized_url=info.normalized_url,
            model_id=info.model_id,
            model_version_id=(explicit_id if explicit_id > 0 else int(info.model_version_id)),
        )

    if explicit_id > 0:
        return CivitaiFrontendURLInfo(
            original_url=info.original_url,
            normalized_url=info.normalized_url,
            model_id=info.model_id,
            model_version_id=explicit_id,
        )

    model_payload = fetch_civitai_model(info.model_id)
    raw_versions = model_payload.get("modelVersions")
    if not isinstance(raw_versions, list):
        raise ValueError("civitai_frontend_url_resolution_failed")

    candidates: list[tuple[int, int]] = []
    for item in raw_versions:
        if not isinstance(item, dict):
            continue
        version_id = int(item.get("id") or 0)
        if version_id <= 0:
            continue
        index_value = item.get("index")
        index_rank = 1_000_000
        try:
            index_parsed = int(index_value)
            if index_parsed >= 0:
                index_rank = index_parsed
        except Exception:
            index_rank = 1_000_000
        candidates.append((index_rank, version_id))
    if not candidates:
        raise ValueError("civitai_frontend_url_resolution_failed")

    # Deterministic default: choose lowest index first (UI default ordering),
    # then highest version id as a stable tie-breaker among equal index values.
    candidates.sort(key=lambda pair: (pair[0], -pair[1]))
    resolved_version_id = int(candidates[0][1])
    return CivitaiFrontendURLInfo(
        original_url=info.original_url,
        normalized_url=info.normalized_url,
        model_id=info.model_id,
        model_version_id=resolved_version_id,
    )


def _parse_civitai_files(model_version_payload: dict[str, object]) -> list[dict[str, object]]:
    raw_files = model_version_payload.get("files")
    if not isinstance(raw_files, list):
        return []

    out: list[dict[str, object]] = []
    for raw in raw_files:
        if not isinstance(raw, dict):
            continue
        file_id = int(raw.get("id") or 0)
        if file_id <= 0:
            continue
        name = str(raw.get("name") or "").strip()
        download_url = str(raw.get("downloadUrl") or raw.get("download_url") or "").strip()

        hashes = _normalize_civitai_hashes(raw.get("hashes"))
        size_bytes = _parse_civitai_size_bytes(raw.get("sizeBytes"))
        size_bytes_exact = size_bytes is not None
        if size_bytes is None:
            size_kb = _parse_civitai_size_bytes(raw.get("sizeKB"))
            if isinstance(size_kb, int) and size_kb > 0:
                size_bytes = int(size_kb * 1024)

        out.append(
            {
                "id": file_id,
                "name": name,
                "download_url": download_url,
                "size_bytes": size_bytes,
                "size_bytes_exact": bool(size_bytes_exact),
                "primary": bool(raw.get("primary")),
                "hashes": hashes,
            }
        )
    return out


def resolve_civitai_source_identity(
    civitai_model_version_id: int,
    *,
    civitai_file_id: int | None = None,
    model_version_payload: dict[str, object] | None = None,
) -> CivitaiResolvedIdentity:
    model_version_id = int(civitai_model_version_id or 0)
    selected_file_id = int(civitai_file_id or 0)
    if model_version_id <= 0:
        raise ValueError("civitai_model_version_id is required")
    if selected_file_id < 0:
        raise ValueError("civitai_file_id is invalid")

    payload = model_version_payload or fetch_civitai_model_version(model_version_id)
    parsed_files = _parse_civitai_files(payload)
    selected = _select_civitai_import_files(parsed_files, selected_file_id=(selected_file_id or None))
    if not selected:
        raise ValueError("civitai_no_supported_files")

    fingerprints: dict[str, str] = {}
    for item in selected:
        if item.rel_path != "" and item.fingerprint != "":
            fingerprints[item.rel_path] = item.fingerprint

    manifest_lines = [f"{p}:{fingerprints[p]}" for p in sorted(fingerprints.keys())]
    source_manifest_sha256 = hashlib.sha256("\n".join(manifest_lines).encode("utf-8")).hexdigest()

    source_ref = f"civitai:model_version_id={model_version_id}"
    if selected_file_id > 0:
        source_ref = f"{source_ref}:file_id={selected_file_id}"

    base_model = str(payload.get("baseModel") or "").strip()
    base_model_type = str(payload.get("baseModelType") or "").strip()
    air = str(payload.get("air") or "").strip()
    pipeline_hint = infer_singlefile_pipeline_hint_from_civitai_metadata(
        base_model=base_model,
        base_model_type=base_model_type,
        air=air,
    )
    model_family_variant = infer_model_family_variant_from_civitai_metadata(
        base_model=base_model,
        base_model_type=base_model_type,
        air=air,
        selected_files=selected,
    )
    model_family = infer_model_family_from_civitai_metadata(
        base_model=base_model,
        base_model_type=base_model_type,
        air=air,
        selected_files=selected,
    )

    return CivitaiResolvedIdentity(
        model_version_id=model_version_id,
        model_id=int(payload.get("modelId") or 0),
        base_model=base_model,
        base_model_type=base_model_type,
        air=air,
        source_ref=source_ref,
        source_revision=f"manifest_sha256:{source_manifest_sha256}",
        source_manifest_sha256=source_manifest_sha256,
        file_fingerprints=fingerprints,
        selected_files=selected,
        selected_file_id=(selected_file_id if selected_file_id > 0 else None),
        pipeline_hint=pipeline_hint,
        model_family=model_family,
        model_family_variant=model_family_variant,
    )


def download_civitai_model_version_files(
    civitai_model_version_id: int,
    output_dir: Path,
    *,
    civitai_file_id: int | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
    resolved_identity: CivitaiResolvedIdentity | None = None,
) -> dict[str, object]:
    resolved = resolved_identity or resolve_civitai_source_identity(
        civitai_model_version_id,
        civitai_file_id=civitai_file_id,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    total_hint: int | None = 0
    total_known = True
    for item in resolved.selected_files:
        if isinstance(item.size_bytes, int) and item.size_bytes > 0:
            total_hint += int(item.size_bytes)
        else:
            total_known = False
    if not total_known:
        total_hint = None

    downloaded_total = 0
    files: list[dict[str, object]] = []

    for item in resolved.selected_files:
        expected_sha256 = _normalize_sha256(item.hashes.get("SHA256"))
        local_path = output_dir / item.rel_path

        def _file_progress(written: int, _total: int | None) -> None:
            if progress_callback is None:
                return
            progress_callback(int(downloaded_total + written), total_hint)

        try:
            info = source_url_to_cas(
                item.download_url,
                local_path,
                progress_callback=_file_progress,
                source_provider="civitai",
                expected_size_bytes=(item.size_bytes if item.size_bytes_exact else None),
                expected_sha256=expected_sha256,
            )
        except ValueError as exc:
            code = str(exc).strip().split(":", 1)[0]
            if code in {"civitai_access_denied", "civitai_not_found"}:
                raise
            raise ValueError("civitai_download_failed") from None

        file_size = int(info.get("size_bytes") or 0)
        downloaded_total += file_size
        if progress_callback is not None:
            progress_callback(downloaded_total, total_hint)

        hashes = dict(item.hashes)
        row: dict[str, object] = {
            "path": item.rel_path,
            "local_path": str(local_path),
            "size_bytes": file_size,
            "sha256": str(info.get("sha256") or ""),
            "file_id": item.file_id,
            "file_name": item.name,
            "file_fingerprint": item.fingerprint,
            "primary": item.primary,
            "hashes": hashes,
            "download_url": item.download_url,
        }
        files.append(row)

    files_with_aliases, component_aliases, missing_components = _civitai_materialize_component_aliases(
        output_dir=output_dir,
        files=files,
        model_family=str(resolved.model_family or ""),
        model_family_variant=str(resolved.model_family_variant or "unknown"),
    )

    return {
        "source_ref": resolved.source_ref,
        "source_revision": resolved.source_revision,
        "source_manifest_sha256": resolved.source_manifest_sha256,
        "model_version_id": resolved.model_version_id,
        "model_id": resolved.model_id,
        "base_model": resolved.base_model,
        "base_model_type": resolved.base_model_type,
        "air": resolved.air,
        "model_family": resolved.model_family,
        "model_family_variant": resolved.model_family_variant,
        "pipeline_hint": resolved.pipeline_hint,
        "selected_file_id": resolved.selected_file_id,
        "file_fingerprints": dict(resolved.file_fingerprints),
        "files": files_with_aliases,
        "file_count": len(files_with_aliases),
        "total_bytes": int(downloaded_total),
        "component_aliases": component_aliases,
        "missing_components": missing_components,
    }


__all__ = [
    "CivitaiFrontendURLInfo",
    "CivitaiImportCandidate",
    "CivitaiResolvedIdentity",
    "download_civitai_model_version_files",
    "download_huggingface_repo_files",
    "fetch_civitai_model",
    "fetch_civitai_model_version",
    "infer_model_family_from_civitai_metadata",
    "infer_model_family_variant_from_civitai_metadata",
    "infer_singlefile_pipeline_hint_from_civitai_metadata",
    "list_huggingface_repo_files",
    "parse_civitai_frontend_model_url",
    "resolve_civitai_source_identity",
    "resolve_civitai_frontend_model_url",
    "resolve_huggingface_source_identity",
    "source_url_to_cas",
]
