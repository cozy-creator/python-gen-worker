"""th#1307: no C2PA private key ever lives in a pod; the hub signs.

The old shape was: the hub injected GEN_WORKER_C2PA_KEY_PEM into every tenant
pod and this process signed in-process. Tenant endpoint code is imported into
this same process, so `print(os.environ["GEN_WORKER_C2PA_KEY_PEM"])` handed any
endpoint author the platform-wide leaf signing key — one leak forges or strips
provenance on every asset Cozy ever signed.

This suite pins the replacement:
  1. private-key material delivered to a pod is REFUSED, loudly (the ratchet);
  2. the worker cannot even be configured to read one (no Settings field, no
     env mapping);
  3. signing goes to the hub, and only the CLAIM travels (never the media);
  4. every failure mode fails the request CLOSED — an unsigned asset never
     ships pretending to be signed.
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from gen_worker import content_credentials as cc


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> dict:
    d = tmp_path_factory.mktemp("c2pa-1307")
    run = lambda *a: subprocess.run(a, check=True, capture_output=True)  # noqa: E731
    run("openssl", "ecparam", "-genkey", "-name", "prime256v1", "-noout", "-out", str(d / "raw.pem"))
    run("openssl", "pkcs8", "-topk8", "-nocrypt", "-in", str(d / "raw.pem"), "-out", str(d / "key.pem"))
    run("openssl", "req", "-x509", "-new", "-key", str(d / "key.pem"),
        "-subj", "/O=Cozy Test/CN=cozy signer", "-days", "365", "-sha256",
        "-addext", "basicConstraints=CA:FALSE",
        "-addext", "keyUsage=critical,digitalSignature",
        "-addext", "extendedKeyUsage=emailProtection", "-out", str(d / "cert.pem"))
    return {"cert": d / "cert.pem", "key": d / "key.pem"}


def _reset(monkeypatch):
    monkeypatch.setattr(cc, "_configured", False)
    monkeypatch.setattr(cc, "_config", None)
    monkeypatch.setattr(cc, "_remote", None)


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (7, 90, 210)).save(buf, format="PNG")
    return buf.getvalue()


def _big_png() -> bytes:
    """~1 MiB of incompressible noise: a stand-in for a real generated asset."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(1307)
    arr = rng.integers(0, 256, size=(700, 700, 3), dtype="uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _configure_cert_only(chain, monkeypatch):
    _reset(monkeypatch)
    cc.configure(SimpleNamespace(
        c2pa_cert_pem=chain["cert"].read_text(), c2pa_cert_path="",
        c2pa_alg="es256", c2pa_ta_url="",
    ))


class _HubSigner:
    """Minimal stand-in for POST /v1/worker/c2pa/sign (see the Go route)."""

    def __init__(self, key_pem: bytes, status: int = 200):
        from cryptography.hazmat.primitives import serialization

        self.key = serialization.load_pem_private_key(key_pem, password=None)
        self.status = status
        self.claim_sizes: list[int] = []
        self.tokens: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import ec, utils as au

                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.tokens.append((self.headers.get("Authorization") or "").removeprefix("Bearer "))
                if outer.status != 200:
                    self.send_response(outer.status)
                    self.end_headers()
                    return
                claim = base64.b64decode(body["claim_b64"])
                outer.claim_sizes.append(len(claim))
                der = outer.key.sign(claim, ec.ECDSA(hashes.SHA256()))
                r, s = au.decode_dss_signature(der)
                sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
                out = json.dumps({"alg": "es256",
                                  "signature_b64": base64.b64encode(sig).decode()}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *_a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()


# ---------------------------------------------------------------------------
# 1 + 2: the pod may not hold, or even be configured with, a private key


@pytest.mark.parametrize("env_name", ["GEN_WORKER_C2PA_KEY_PEM", "GEN_WORKER_C2PA_KEY_PATH"])
def test_private_key_in_pod_env_is_refused(chain, monkeypatch, env_name):
    """A hub regression that re-injects the key must kill the pod, loudly.

    Fails on pre-th#1307 code, where this env var was the happy path.
    """
    _reset(monkeypatch)
    monkeypatch.setenv(env_name, chain["key"].read_text())
    with pytest.raises(cc.C2paSigningError) as e:
        cc.configure(SimpleNamespace(
            c2pa_cert_pem=chain["cert"].read_text(), c2pa_cert_path="",
            c2pa_alg="es256", c2pa_ta_url="",
        ))
    assert env_name in str(e.value)
    # And the refusal is not swallowed by the lazy path either: a caller that
    # merely asks "is signing on?" gets the same hard error, never a quiet
    # False that would ship media unsigned.
    with pytest.raises(cc.C2paSigningError):
        cc.enabled()
    _reset(monkeypatch)


@pytest.mark.parametrize("env_name", ["GEN_WORKER_C2PA_KEY_PEM", "GEN_WORKER_C2PA_KEY_PATH"])
def test_key_bearing_pod_is_refused_even_if_configure_was_never_called(chain, monkeypatch, env_name):
    """The refusal belongs to the POD, not to one compiled graph point.

    pgw#931 removed `_active_config`'s lazy `configure()` fallback. That is the
    right call on its own terms — a signing module must not resolve its own
    config from the environment — but the th#1307 refusal was riding inside
    that `configure()` call, so it left with it. Every process that does not
    run the worker bring-up (a library embed, a compute child, an endpoint that
    imports the module itself) then got a quiet `enabled() == False` while a
    platform-wide private key sat in `os.environ`, and shipped media unsigned.

    So: no `configure()` here, deliberately. This is the shape that regressed.
    """
    _reset(monkeypatch)
    monkeypatch.setenv(env_name, chain["key"].read_text())

    for call in (
        lambda: cc.enabled(),
        lambda: cc.sign_media_bytes(_png(), ref="outputs/a.png"),
        lambda: cc.sign_media_file("/nonexistent.png", ref="outputs/a.png"),
    ):
        with pytest.raises(cc.C2paSigningError) as e:
            call()
        assert env_name in str(e.value)
    _reset(monkeypatch)


def test_key_delivered_after_a_clean_configure_is_still_refused(chain, monkeypatch):
    """Memoisation must not outrank the ratchet.

    `_active_config` short-circuits on `_configured`. If the refusal sat behind
    that fast path, a pod that configured cleanly and was handed key material
    afterwards would sign happily for the rest of its life.
    """
    _configure_cert_only(chain, monkeypatch)
    try:
        assert cc.enabled()
        monkeypatch.setenv("GEN_WORKER_C2PA_KEY_PEM", chain["key"].read_text())
        with pytest.raises(cc.C2paSigningError) as e:
            cc.enabled()
        assert "GEN_WORKER_C2PA_KEY_PEM" in str(e.value)
    finally:
        _reset(monkeypatch)


def test_settings_have_no_private_key_field_at_all():
    """The ratchet in the config layer: nothing can carry a key into Settings."""
    from gen_worker.config.loader import _ENV_TO_FIELD
    from gen_worker.config.settings import Settings

    for env_name in ("GEN_WORKER_C2PA_KEY_PEM", "GEN_WORKER_C2PA_KEY_PATH"):
        assert env_name not in _ENV_TO_FIELD, f"{env_name} must not map into Settings (th#1307)"
    fields = set(getattr(Settings, "__dataclass_fields__", {}) or Settings.__annotations__)
    assert "c2pa_key_pem" not in fields
    assert "c2pa_key_path" not in fields
    # The PUBLIC half is still configurable.
    assert "c2pa_cert_pem" in fields


# ---------------------------------------------------------------------------
# 3: signing works hub-side, and only the claim travels


def test_hub_signs_and_media_verifies(chain, monkeypatch):
    import c2pa

    hub = _HubSigner(chain["key"].read_bytes())
    try:
        _configure_cert_only(chain, monkeypatch)
        cc.configure_remote_signer(hub.base_url, lambda: "worker-jwt-abc")
        raw = _big_png()
        signed = cc.sign_media_bytes(raw, ref="outputs/a.png", request_id="req-1307")
        assert signed != raw
        with c2pa.Reader("image/png", io.BytesIO(signed)) as reader:
            assert str(reader.get_validation_state()) in ("Valid", "Trusted")

        assert hub.claim_sizes, "the hub was never asked to sign"
        assert hub.tokens == ["worker-jwt-abc"], "the worker JWT must authenticate the oracle"
        # The media never transits the hub: a claim is hashes + assertions, so
        # its size is bounded and INDEPENDENT of the asset (here ~1 KiB of
        # claim for a ~1 MiB asset). This is why hub-side signing does not
        # drag multi-GB video through the control plane.
        assert max(hub.claim_sizes) < 8 * 1024, f"claim is {max(hub.claim_sizes)} bytes"
        assert max(hub.claim_sizes) * 20 < len(raw), (
            f"claim bytes {max(hub.claim_sizes)} vs media {len(raw)} — only the claim may travel"
        )
    finally:
        hub.close()
        _reset(monkeypatch)


# ---------------------------------------------------------------------------
# 4: every failure fails the REQUEST, never ships unsigned


def test_unarmed_signer_fails_closed(chain, monkeypatch):
    """Cert configured, hub transport not armed yet: raise, never return the
    unsigned bytes as if they had been signed."""
    _configure_cert_only(chain, monkeypatch)
    try:
        assert cc.enabled()
        with pytest.raises(cc.C2paSigningError):
            cc.sign_media_bytes(_png(), ref="outputs/a.png")
    finally:
        _reset(monkeypatch)


def test_hub_refusal_fails_closed(chain, monkeypatch):
    hub = _HubSigner(chain["key"].read_bytes(), status=503)
    try:
        _configure_cert_only(chain, monkeypatch)
        cc.configure_remote_signer(hub.base_url, lambda: "worker-jwt-abc")
        with pytest.raises(cc.C2paSigningError):
            cc.sign_media_bytes(_png(), ref="outputs/a.png")
    finally:
        hub.close()
        _reset(monkeypatch)


def test_unreachable_hub_fails_closed(chain, monkeypatch):
    _configure_cert_only(chain, monkeypatch)
    try:
        # Port 1 is never listening.
        cc.configure_remote_signer("http://127.0.0.1:1", lambda: "worker-jwt-abc")
        with pytest.raises(cc.C2paSigningError):
            cc.sign_media_bytes(_png(), ref="outputs/a.png")
    finally:
        _reset(monkeypatch)


def test_file_variant_also_fails_closed(chain, monkeypatch, tmp_path: Path):
    src = tmp_path / "a.png"
    src.write_bytes(_png())
    _configure_cert_only(chain, monkeypatch)
    try:
        with pytest.raises(cc.C2paSigningError):
            cc.sign_media_file(str(src), ref="outputs/a.png")
        assert src.read_bytes() == _png(), "source must never be mutated"
    finally:
        _reset(monkeypatch)
