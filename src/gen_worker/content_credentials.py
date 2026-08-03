"""C2PA Content Credentials — sign generated media at the finalize seam (th#714).

EU AI Act Art. 50 (applies 2026-08-02) requires machine-readable marking of
AI-generated audio/image/video. We embed a signed C2PA manifest (issuer =
platform signing cert, ``c2pa.created`` action with digitalSourceType
``trainedAlgorithmicMedia``, model refs, request-id hash — no user PII) into
every generated media asset as it passes through ``RequestContext.save_bytes``
/ ``save_file``, i.e. the last point the bytes touch trusted compute before
upload.

**The private key is NOT here** (th#1307). A pod imports untrusted tenant code
into this process, so a signing key in pod env or on the pod filesystem is one
``print(os.environ[...])`` from being exfiltrated — and a leaked platform leaf
key forges or strips provenance on every asset Cozy ever signed. So the split
is: the worker builds the claim (it has the bytes) and the HUB signs it. The
c2pa-rs callback signer sends the claim's COSE to-be-signed octets to
``POST /v1/worker/c2pa/sign``, authenticated with this pod's worker JWT, and
gets back a signature. No media leaves the pod; no key enters it.

Config (Settings / env) — the PUBLIC half only:
- ``GEN_WORKER_C2PA_CERT_PEM`` — inline PEM signing-cert chain (leaf first,
  then intermediates/root). The hub injects this into pod env at launch
  (RunPod pods have no file mounts). Takes precedence over ``_CERT_PATH``.
- ``GEN_WORKER_C2PA_CERT_PATH`` — file-path variant for mounted deploys.
- ``GEN_WORKER_C2PA_ALG``      — COSE alg (default ``es256``).
- ``GEN_WORKER_C2PA_TA_URL``   — optional RFC3161 timestamp authority URL.

``GEN_WORKER_C2PA_KEY_PEM`` / ``_KEY_PATH`` are REFUSED: if either is present
this module raises instead of using it. That is the ratchet — a hub regression
that re-injects the key kills the pod loudly rather than quietly re-creating
the leak.

The refusal is a property of the POD ENVIRONMENT, not of one call. It is
therefore evaluated at every read of the signing state (:func:`_active_config`,
and so :func:`enabled` and both ``sign_media_*``), not only inside
:func:`configure`. A process that never called ``configure()`` — a library
embed, a compute child, a test harness — must not be the process where a
delivered private key goes unnoticed and media then ships unsigned.

Signing is ON iff cert material is set. The hub transport is armed at HelloAck
(``configure_remote_signer``, same wiring moment as the cell-receipt gate).

Policy: default-ON when the cert is configured; a cert that does not parse
fails worker startup (never silently ship unlabeled media believing signing is
on); unconfigured no-ops with a loud startup warning. A sign failure at request
time — including "the hub signer is unreachable / not armed" — RAISES: the
request fails rather than shipping an unlabeled asset.

Uses c2pa-python (official CAI binding over c2pa-rs; ``signing`` extra).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Optional
import os
import tempfile

logger = logging.getLogger(__name__)

_GENERATOR_NAME = "cozy-gen-worker"

# COSE alg name -> c2pa.C2paSigningAlg member. The hub's signer (Go,
# internal/orchestrator/c2pasign) speaks the same set; ECDSA signatures cross
# the wire as COSE fixed-width r||s.
_ALG_NAMES = {
    "es256": "ES256",
    "es384": "ES384",
    "es512": "ES512",
    "ed25519": "ED25519",
    "ps256": "PS256",
    "ps384": "PS384",
    "ps512": "PS512",
}

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
    alg: str
    ta_url: str
    generator_version: str


@dataclass(frozen=True)
class _RemoteSigner:
    """The hub signing oracle (th#1307). Armed at HelloAck."""

    base_url: str
    worker_jwt: Any  # Callable[[], str]


# th#1307: env names that would carry a private key INTO the pod. Their
# presence is a platform regression, not a configuration option.
_REFUSED_KEY_ENVS = ("GEN_WORKER_C2PA_KEY_PEM", "GEN_WORKER_C2PA_KEY_PATH")

SIGN_PATH = "/v1/worker/c2pa/sign"
_SIGN_TIMEOUT_S = 30

_lock = threading.Lock()
_configured = False
_config: Optional[_SignerConfig] = None
_remote: Optional[_RemoteSigner] = None


def configure(settings: Any) -> None:
    """Install (or clear) the process-wide signer config from Settings.

    Called once at worker startup. Raises when signing is configured but
    unusable — a worker that *thinks* it signs but doesn't is a compliance
    hole, so it must not come up — and raises when private-key material was
    delivered to this pod at all (th#1307). Logs a loud warning when no cert
    is configured (signing disabled).
    """
    global _configured, _config
    _refuse_pod_private_key_material(settings)
    inline_cert = str(getattr(settings, "c2pa_cert_pem", "") or "").strip()
    cert_path = str(getattr(settings, "c2pa_cert_path", "") or "").strip()
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
            # Inline PEM (hub-injected pod env) wins over a mounted path.
            cert_pem = inline_cert.encode()
        else:
            try:
                cert_pem = open(cert_path, "rb").read()
            except OSError as e:
                raise C2paSigningError(f"cannot read C2PA signing cert: {e}") from e
        alg = str(getattr(settings, "c2pa_alg", "") or "es256").strip().lower()
        cfg = _SignerConfig(
            cert_pem=cert_pem,
            alg=alg,
            ta_url=str(getattr(settings, "c2pa_ta_url", "") or "").strip(),
            generator_version=_generator_version(),
        )
        # Startup probe of everything checkable without the hub: the cert
        # chain parses, the alg is one c2pa knows. The signer itself needs the
        # HelloAck-armed transport, so it is built at first sign (and any
        # failure there raises, failing the request — never an unsigned ship).
        _validate_signing_cert(cfg)
        _config = cfg
        _configured = True
        logger.info(
            "C2PA content-credential signing ENABLED (alg=%s, cert=%s, signer=hub-side th#1307)",
            cfg.alg,
            "<inline env PEM>" if inline_cert else cert_path,
        )


def configure_remote_signer(base_url: str, worker_jwt: Any) -> None:
    """Arm the hub signing oracle (th#1307).

    Called at HelloAck, the moment the hub wiring exists — the same seam that
    arms the cell-receipt gate. Until this lands, a configured signer FAILS
    requests rather than shipping unsigned media.
    """
    global _remote
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return
    with _lock:
        _remote = _RemoteSigner(base_url=base, worker_jwt=worker_jwt)
    logger.info("C2PA hub-side signer armed (%s%s)", base, SIGN_PATH)


def _validate_signing_cert(cfg: _SignerConfig) -> None:
    """Fail startup on a cert we could never sign with."""
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


def _refuse_pod_private_key_material(settings: Any = None) -> None:
    """Refuse, loudly, any C2PA PRIVATE KEY delivered to this pod (th#1307).

    This is a ratchet on the pod's environment, so it is checked at every read
    of the signing state rather than once at ``configure()``. pgw#931 removed
    ``_active_config``'s lazy ``configure()`` fallback — correctly, since a
    signing module must not go and find its own config — but the refusal rode
    along inside that call, so it became reachable only from the one entry that
    calls ``configure()``. Anything else (a library embed, a compute child, an
    endpoint importing the module directly) got a quiet ``enabled() == False``
    and shipped unsigned media beside a key that was sitting right there in the
    environment. The presence of the key is the fact; no entry point owns it.

    ``settings`` is checked too when supplied, so a field smuggled back into
    Settings is refused on the same breath as the env.
    """
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
    # Before the memoised answer, not after: a pod carrying key material is
    # refused whether or not this process ever configured signing.
    _refuse_pod_private_key_material()
    if _configured:
        return _config
    # pgw#931 (§1.18): this used to resolve lazily from the cached Settings
    # loader, i.e. a signing module that could go and find its own signing
    # config from the environment at first use. It cannot: `configure()` is
    # called by the process entry with the entry's `Settings`, and a process
    # that never configured signing is a process where signing is OFF.
    #
    # That is also the honest answer. th#1307 makes cert material's PRESENCE
    # the gate, so "unconfigured" and "configured with no cert" must not be the
    # same silent state as "found some env" — see the C2PA disposition in
    # pgw#929: signing being dark is a reported fact, never an inference.
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
    """Ask the hub to sign one claim's COSE to-be-signed octets (th#1307).

    Only the claim travels — a few hundred bytes of hashes and assertions,
    never the media. Any refusal raises, so the request fails instead of
    shipping an asset with a missing or bogus manifest.
    """
    # pgw#763 delta 5: under the process split this callback runs in the compute
    # child, which holds no worker JWT — so the ASK is a parent-side IPC action.
    # The child sends a hash (the COSE to-be-signed octets) and gets a signature
    # back; the credential that authorizes the oracle, like the key behind it,
    # is somewhere this process cannot reach. `broker.request` is the same POST
    # off the split, so there is one code path either way.
    from .procsplit import broker

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
        # Fail CLOSED. A configured worker with no hub signer must fail the
        # request, not hand back unsigned bytes that look signed (th#1307).
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


__all__ = [
    "C2paSigningError",
    "configure",
    "configure_remote_signer",
    "enabled",
    "sign_media_bytes",
    "sign_media_file",
    "sniff_media_mime",
]
