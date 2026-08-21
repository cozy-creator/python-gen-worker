"""C2PA Content Credentials — sign generated media at the finalize seam (EU AI Act Art. 50). The private key is NEVER on the pod: the worker builds the claim and the hub signs its COSE octets over POST /v1/worker/c2pa/sign; GEN_WORKER_C2PA_KEY_PEM/_KEY_PATH present in ANY config source is a boot refusal. Signing is ON iff cert material is set, and a sign failure at request time RAISES — never ship an unlabeled asset."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Optional
import os
import tempfile
from .procsplit import broker

if TYPE_CHECKING:
    from .config import Settings

logger = logging.getLogger(__name__)

_GENERATOR_NAME = "cozy-gen-worker"

_ALG_NAMES = {
    "es256": "ES256",
    "es384": "ES384",
    "es512": "ES512",
    "ed25519": "ED25519",
    "ps256": "PS256",
    "ps384": "PS384",
    "ps512": "PS512",
}

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
        ext_mime = _ext_mime(ref)
        if ext_mime in ("video/quicktime", "audio/mp4"):
            return ext_mime
        return "video/mp4"
    if head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
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
    alg: str
    ta_url: str
    generator_version: str


@dataclass(frozen=True)
class _RemoteSigner:

    base_url: str
    worker_jwt: Any


_REFUSED_KEY_ENVS = ("GEN_WORKER_C2PA_KEY_PEM", "GEN_WORKER_C2PA_KEY_PATH")

SIGN_PATH = "/v1/worker/c2pa/sign"
_SIGN_TIMEOUT_S = 30

_lock = threading.Lock()
_configured = False
_config: Optional[_SignerConfig] = None
_remote: Optional[_RemoteSigner] = None


def configure(settings: Settings) -> None:
    """Install (or clear) the process-wide signer config from Settings."""
    global _configured, _config
    _refuse_pod_private_key_material(settings)
    inline_cert = settings.c2pa_cert_pem.strip()
    cert_path = settings.c2pa_cert_path.strip()
    with _lock:
        if not inline_cert and not cert_path:
            _config = None
            _configured = True
            logger.warning(
                "C2PA content-credential signing DISABLED — neither GEN_WORKER_C2PA_CERT_PEM "
                "nor GEN_WORKER_C2PA_CERT_PATH is set. Generated media will NOT carry Content "
                "Credentials (EU AI Act Art. 50 machine-readable AI-marking, th#714)."
            )
            return
        if inline_cert:
            cert_pem = inline_cert.encode()
        else:
            try:
                cert_pem = open(cert_path, "rb").read()
            except OSError as e:
                raise C2paSigningError(f"cannot read C2PA signing cert: {e}") from e
        alg = (settings.c2pa_alg or "es256").strip().lower()
        cfg = _SignerConfig(
            cert_pem=cert_pem,
            alg=alg,
            ta_url=settings.c2pa_ta_url.strip(),
            generator_version=_generator_version(),
        )
        _validate_signing_cert(cfg)
        _config = cfg
        _configured = True
        logger.info(
            "C2PA content-credential signing ENABLED (alg=%s, cert=%s, signer=hub-side th#1307)",
            cfg.alg,
            "<inline env PEM>" if inline_cert else cert_path,
        )


def configure_remote_signer(base_url: str, worker_jwt: Any) -> None:
    """Arm the hub signing oracle."""
    global _remote
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return
    with _lock:
        _remote = _RemoteSigner(base_url=base, worker_jwt=worker_jwt)
    logger.info("C2PA hub-side signer armed (%s%s)", base, SIGN_PATH)


def _validate_signing_cert(cfg: _SignerConfig) -> None:
    if cfg.alg not in _ALG_NAMES:
        raise C2paSigningError(
            f"unsupported C2PA alg {cfg.alg!r} (want one of {sorted(_ALG_NAMES)})"
        )
    try:
        from cryptography import x509

        x509.load_pem_x509_certificate(cfg.cert_pem)
    except ImportError:  # pragma: no cover - cryptography is a hard dep
        return
    except Exception as e:
        raise C2paSigningError(f"C2PA signing cert is not a usable PEM certificate: {e}") from e


def enabled() -> bool:
    return _active_config() is not None


def sign_media_bytes(
    data: bytes,
    *,
    ref: str,
    request_id: str = "",
    models: Iterable[str] = (),
) -> bytes:
    """Sign ``data`` in memory if it is signable media and signing is enabled."""
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
    """Sign the media file at ``src_path`` into a NamedTemporaryFile."""
    cfg = _active_config()
    if cfg is None:
        return None
    with open(src_path, "rb") as f:
        head = f.read(16)
    mime = sniff_media_mime(head, ref)
    if mime is None:
        return None

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


def _refuse_pod_private_key_material(settings: Any = None) -> None:
    for env_name in _REFUSED_KEY_ENVS:
        if str(os.environ.get(env_name, "") or "").strip() or str(
            getattr(settings, env_name.lower().removeprefix("gen_worker_"), "") or ""
        ).strip():
            raise C2paSigningError(
                f"{env_name} is set: a C2PA PRIVATE KEY must never be delivered to a pod "
                "(th#1307 — tenant code runs in this process and can read it). Signing is "
                "hub-side: the hub holds the key and signs claims over POST " + SIGN_PATH
            )


def _active_config() -> Optional[_SignerConfig]:
    global _configured, _config
    _refuse_pod_private_key_material()
    if _configured:
        return _config
    with _lock:
        _config = None
        _configured = True
    return None


def _generator_version() -> str:
    try:
        from importlib.metadata import version

        return version("gen-worker")
    except Exception:
        return "unknown"


def _hub_sign_claim(remote: _RemoteSigner, alg: str, claim: bytes) -> bytes:

    try:
        resp = broker.request(
            "POST",
            SIGN_PATH,
            base_url=remote.base_url,
            bearer=remote.worker_jwt(),
            json={"alg": alg, "claim_b64": base64.b64encode(claim).decode()},
            timeout=_SIGN_TIMEOUT_S,
        )
    except Exception as e:
        raise C2paSigningError(f"hub C2PA signer unreachable: {e}") from e
    if resp.status_code != 200:
        detail = resp.text[:300]
        raise C2paSigningError(
            f"hub C2PA signer refused ({resp.status_code}): {detail}"
        )
    try:
        signature = base64.b64decode(resp.json()["signature_b64"])
    except Exception as e:
        raise C2paSigningError(f"hub C2PA signer returned a malformed signature: {e}") from e
    if not signature:
        raise C2paSigningError("hub C2PA signer returned an empty signature")
    return signature


def _build_signer(cfg: _SignerConfig) -> Any:
    try:
        import c2pa
    except ImportError as e:
        raise C2paSigningError(
            "C2PA signing is configured but c2pa-python is not installed. "
            "Install with `pip install gen-worker[signing]`."
        ) from e
    remote = _remote
    if remote is None:
        raise C2paSigningError(
            "C2PA signing is configured but the hub signer is not armed "
            "(no HelloAck file_base_url yet) — refusing to ship unsigned media"
        )
    alg_enum = getattr(c2pa.C2paSigningAlg, _ALG_NAMES[cfg.alg])
    callback = lambda claim: _hub_sign_claim(remote, cfg.alg, claim)  # noqa: E731
    try:
        return c2pa.Signer.from_callback(
            callback,
            alg_enum,
            cfg.cert_pem.decode(),
            cfg.ta_url or None,
        )
    except C2paSigningError:
        raise
    except Exception as e:
        raise C2paSigningError(f"cannot build C2PA signer: {e}") from e


def _manifest_json(cfg: _SignerConfig, request_id: str, models: Iterable[str]) -> str:
    generator = {"name": _GENERATOR_NAME, "version": cfg.generator_version}
    cozy: dict[str, Any] = {}
    if request_id:
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
        settings.update(json.dumps({"builder": {"thumbnail": {"enabled": False}}}))
        context = c2pa.ContextBuilder().with_settings(settings).with_signer(signer).build()
        builder = c2pa.Builder(_manifest_json(cfg, request_id, models), context=context)
        builder.sign(mime, src, dst)
    except C2paSigningError:
        raise
    except Exception as e:
        raise C2paSigningError(f"C2PA signing failed for {mime}: {e}") from e


__all__ = [
    "C2paSigningError",
    "configure",
    "configure_remote_signer",
    "enabled",
    "sign_media_bytes",
    "sign_media_file",
    "sniff_media_mime",
]
