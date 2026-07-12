"""C2PA Content Credentials — sign generated media at the finalize seam (th#714).

EU AI Act Art. 50 (applies 2026-08-02) requires machine-readable marking of
AI-generated audio/image/video. We embed a signed C2PA manifest (issuer =
platform signing cert, ``c2pa.created`` action with digitalSourceType
``trainedAlgorithmicMedia``, model refs, request-id hash — no user PII) into
every generated media asset as it passes through ``RequestContext.save_bytes``
/ ``save_file``, i.e. the last point the bytes touch trusted compute before
upload.

Config (Settings / env):
- ``GEN_WORKER_C2PA_CERT_PATH`` — PEM signing-cert chain (leaf first, then
  intermediates/root). Signing is ON iff this is set.
- ``GEN_WORKER_C2PA_KEY_PATH``  — PKCS#8 PEM private key for the leaf.
- ``GEN_WORKER_C2PA_ALG``      — COSE alg (default ``es256``).
- ``GEN_WORKER_C2PA_TA_URL``   — optional RFC3161 timestamp authority URL.

Policy: default-ON when the cert is configured; configured-but-broken fails
worker startup (never silently ship unlabeled media believing signing is on);
unconfigured no-ops with a loud startup warning. A sign failure at request
time raises — the request fails rather than shipping an unlabeled asset.

Uses c2pa-python (official CAI binding over c2pa-rs; ``signing`` extra).
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

_GENERATOR_NAME = "cozy-gen-worker"

# Formats we sign, by content sniff (magic bytes) with an extension fallback
# for BMFF/audio containers whose sniff needs an offset. Everything else
# (JSON payloads, checkpoints, tensors, …) passes through untouched.
_SIGNABLE_MIMES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/avif",
        "video/mp4",
        "video/quicktime",
        "audio/wav",
        "audio/mpeg",
        "audio/flac",
        "audio/mp4",
    }
)

_EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".avif": "image/avif",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}


class C2paSigningError(RuntimeError):
    """Signing was configured but failed — the asset must not ship unlabeled."""


def sniff_media_mime(head: bytes, ref: str) -> Optional[str]:
    """Return the signable media MIME for content ``head`` + ref extension, or None."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image/gif"
    if len(head) >= 12 and head[0:4] == b"RIFF":
        if head[8:12] == b"WEBP":
            return "image/webp"
        if head[8:12] == b"WAVE":
            return "audio/wav"
    if head.startswith(b"fLaC"):
        return "audio/flac"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        # BMFF: mp4 / mov / m4a — disambiguate by extension, default mp4.
        ext_mime = _ext_mime(ref)
        if ext_mime in ("video/quicktime", "audio/mp4"):
            return ext_mime
        return "video/mp4"
    if head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        # MP3 (ID3 tag or bare frame sync) — only trust with a matching extension.
        if _ext_mime(ref) == "audio/mpeg":
            return "audio/mpeg"
    mime = _ext_mime(ref)
    return mime if mime in _SIGNABLE_MIMES else None


def _ext_mime(ref: str) -> Optional[str]:
    dot = ref.rfind(".")
    if dot < 0:
        return None
    return _EXT_TO_MIME.get(ref[dot:].lower())


@dataclass(frozen=True)
class _SignerConfig:
    cert_pem: bytes
    key_pem: bytes
    alg: str
    ta_url: str
    generator_version: str


_lock = threading.Lock()
_configured = False
_config: Optional[_SignerConfig] = None


def configure(settings: Any) -> None:
    """Install (or clear) the process-wide signer from Settings.

    Called once at worker startup. Raises when a cert path is set but the
    material is unusable — a worker that *thinks* it signs but doesn't is a
    compliance hole, so it must not come up. Logs a loud warning when no
    cert is configured (signing disabled).
    """
    global _configured, _config
    cert_path = str(getattr(settings, "c2pa_cert_path", "") or "").strip()
    key_path = str(getattr(settings, "c2pa_key_path", "") or "").strip()
    with _lock:
        if not cert_path:
            _config = None
            _configured = True
            logger.warning(
                "C2PA content-credential signing DISABLED — GEN_WORKER_C2PA_CERT_PATH is not set. "
                "Generated media will NOT carry Content Credentials "
                "(EU AI Act Art. 50 machine-readable AI-marking, th#714)."
            )
            return
        if not key_path:
            raise C2paSigningError(
                "GEN_WORKER_C2PA_CERT_PATH is set but GEN_WORKER_C2PA_KEY_PATH is not"
            )
        try:
            cert_pem = open(cert_path, "rb").read()
            key_pem = open(key_path, "rb").read()
        except OSError as e:
            raise C2paSigningError(f"cannot read C2PA signing material: {e}") from e
        cfg = _SignerConfig(
            cert_pem=cert_pem,
            key_pem=key_pem,
            alg=str(getattr(settings, "c2pa_alg", "") or "es256").strip().lower(),
            ta_url=str(getattr(settings, "c2pa_ta_url", "") or "").strip(),
            generator_version=_generator_version(),
        )
        # Probe: build a signer now so bad PEM / a missing c2pa wheel fails
        # startup, not the first request.
        _build_signer(cfg)
        _config = cfg
        _configured = True
        logger.info(
            "C2PA content-credential signing ENABLED (alg=%s, cert=%s)", cfg.alg, cert_path
        )


def enabled() -> bool:
    return _active_config() is not None


def sign_media_bytes(
    data: bytes,
    *,
    ref: str,
    request_id: str = "",
    models: Iterable[str] = (),
) -> bytes:
    """Sign ``data`` in memory if it is signable media and signing is enabled.

    Returns the signed bytes, or ``data`` unchanged when signing is disabled
    or the payload is not a signable media format. Raises
    :class:`C2paSigningError` when signing is enabled but fails.
    """
    cfg = _active_config()
    if cfg is None:
        return data
    mime = sniff_media_mime(data[:16], ref)
    if mime is None:
        return data
    dst = io.BytesIO()
    _sign_stream(cfg, mime, io.BytesIO(data), dst, request_id=request_id, models=models)
    return dst.getvalue()


def sign_media_file(
    src_path: str,
    *,
    ref: str,
    request_id: str = "",
    models: Iterable[str] = (),
) -> Optional[str]:
    """Sign the media file at ``src_path`` into a NamedTemporaryFile.

    Returns the signed temp-file path (caller owns cleanup), or None when
    signing is disabled or the file is not a signable media format. The
    source file is never mutated. Raises :class:`C2paSigningError` when
    signing is enabled but fails.
    """
    cfg = _active_config()
    if cfg is None:
        return None
    with open(src_path, "rb") as f:
        head = f.read(16)
    mime = sniff_media_mime(head, ref)
    if mime is None:
        return None
    import os
    import tempfile

    suffix = os.path.splitext(str(src_path))[1] or ".bin"
    fd, out_path = tempfile.mkstemp(suffix=suffix, prefix="c2pa-")
    try:
        with open(src_path, "rb") as src, os.fdopen(fd, "wb") as dst:
            _sign_stream(cfg, mime, src, dst, request_id=request_id, models=models)
    except BaseException:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise
    return out_path


# ---------------------------------------------------------------------------
# internals


def _active_config() -> Optional[_SignerConfig]:
    global _configured, _config
    if _configured:
        return _config
    # Library-standalone path (no worker bring-up called configure()):
    # resolve lazily from the cached Settings loader. configure() is
    # idempotent, so a benign race between threads is fine.
    try:
        from .config import get_settings

        settings = get_settings()
    except Exception:
        with _lock:
            _config = None
            _configured = True
        return None
    configure(settings)
    return _config


def _generator_version() -> str:
    try:
        from importlib.metadata import version

        return version("gen-worker")
    except Exception:
        return "unknown"


def _build_signer(cfg: _SignerConfig) -> Any:
    try:
        import c2pa
    except ImportError as e:
        raise C2paSigningError(
            "C2PA signing is configured but c2pa-python is not installed. "
            "Install with `pip install gen-worker[signing]`."
        ) from e
    # The wrapper's C2paSignerInfo.__init__ rejects ta_url=None and passes
    # b"" through as an (invalid) empty TSA URL, so build the ctypes struct
    # directly to get a NULL ta_url when no TSA is configured.
    info = c2pa.C2paSignerInfo.__new__(c2pa.C2paSignerInfo)
    ctypes.Structure.__init__(
        info,
        cfg.alg.encode(),
        cfg.cert_pem,
        cfg.key_pem,
        cfg.ta_url.encode() if cfg.ta_url else None,
    )
    try:
        return c2pa.Signer.from_info(info)
    except Exception as e:
        raise C2paSigningError(f"cannot build C2PA signer: {e}") from e


def _manifest_json(cfg: _SignerConfig, request_id: str, models: Iterable[str]) -> str:
    generator = {"name": _GENERATOR_NAME, "version": cfg.generator_version}
    cozy: dict[str, Any] = {}
    if request_id:
        # Hash, not the raw id: links back to platform records without
        # exposing request identifiers (and never user PII) in public files.
        cozy["request_sha256"] = hashlib.sha256(request_id.encode()).hexdigest()
    model_refs = sorted({str(m) for m in models if str(m).strip()})
    if model_refs:
        cozy["models"] = model_refs
    manifest: dict[str, Any] = {
        "claim_generator_info": [generator],
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": (
                                "http://cv.iptc.org/newscodes/digitalsourcetype/"
                                "trainedAlgorithmicMedia"
                            ),
                            "softwareAgent": generator,
                        }
                    ]
                },
            }
        ],
    }
    if cozy:
        manifest["assertions"].append({"label": "com.cozy.generation", "data": cozy})
    return json.dumps(manifest, separators=(",", ":"))


def _sign_stream(
    cfg: _SignerConfig,
    mime: str,
    src: Any,
    dst: Any,
    *,
    request_id: str,
    models: Iterable[str],
) -> None:
    try:
        import c2pa

        signer = _build_signer(cfg)
        settings = c2pa.Settings()
        # Claim thumbnails add tens of KB per asset for no compliance value.
        settings.update(json.dumps({"builder": {"thumbnail": {"enabled": False}}}))
        context = c2pa.ContextBuilder().with_settings(settings).with_signer(signer).build()
        builder = c2pa.Builder(_manifest_json(cfg, request_id, models), context=context)
        builder.sign(mime, src, dst)
    except C2paSigningError:
        raise
    except Exception as e:
        raise C2paSigningError(f"C2PA signing failed for {mime}: {e}") from e


def _reset_for_tests() -> None:
    global _configured, _config
    with _lock:
        _configured = False
        _config = None


__all__ = [
    "C2paSigningError",
    "configure",
    "enabled",
    "sign_media_bytes",
    "sign_media_file",
    "sniff_media_mime",
]
