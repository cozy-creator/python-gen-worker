from __future__ import annotations

import json
import threading
import time
import urllib.request
from types import ModuleType
from typing import Any, Dict, Optional

import jwt

_jwt_algorithms: Optional[ModuleType]
try:
    import jwt.algorithms as _jwt_algorithms
except Exception:  # pragma: no cover - optional crypto backend
    _jwt_algorithms = None
RSAAlgorithm: Optional[Any] = getattr(_jwt_algorithms, "RSAAlgorithm", None) if _jwt_algorithms else None


class _JWKSCache:
    def __init__(self, url: str, ttl_seconds: int = 300) -> None:
        self._url = url
        self._ttl_seconds = max(ttl_seconds, 0)
        self._lock = threading.Lock()
        self._fetched_at = 0.0
        self._keys: Dict[str, Any] = {}

    def _fetch(self) -> None:
        if RSAAlgorithm is None:
            raise RuntimeError(
                "PyJWT RSA support is unavailable (missing cryptography). "
                "Install gen-worker with a JWT/RSA-capable build of PyJWT."
            )
        with urllib.request.urlopen(self._url, timeout=5) as resp:
            body = resp.read()
        payload = json.loads(body.decode("utf-8"))
        keys: Dict[str, Any] = {}
        for jwk in payload.get("keys", []):
            kid = jwk.get("kid")
            if not kid:
                continue
            try:
                keys[kid] = RSAAlgorithm.from_jwk(json.dumps(jwk))
            except Exception:
                continue
        self._keys = keys
        self._fetched_at = time.time()

    def _needs_refresh(self) -> bool:
        if not self._keys:
            return True
        if self._ttl_seconds <= 0:
            return False
        return (time.time() - self._fetched_at) > self._ttl_seconds

    def get_key(self, kid: Optional[str]) -> Optional[Any]:
        with self._lock:
            if self._needs_refresh():
                self._fetch()
            if kid and kid in self._keys:
                return self._keys[kid]
            # refresh on miss (rotation)
            self._fetch()
            if kid and kid in self._keys:
                return self._keys[kid]
            return None
