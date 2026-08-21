"""The immutable Bind Contract: fetch, address verification, and refusal report.

The hub chooses the contract for (release, derive image digest, config digest)
and sends its CAS address plus a bounded URL. This module verifies the address
before decoding. Tensor semantics stay in ``serving.streaming.census``; the
wire layer only carries the decoded expected census and its bind identity.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from . import worker_credential
from .serving.streaming.census import Census, CensusMismatch

BIND_CONTRACT_KIND = "tensorhub.bind-contract@1"
BIND_CONTRACT_VERSION = 1
MAX_BIND_CONTRACT_BYTES = 4 << 20
REPORT_PATH = "/v1/worker/release-bind-refusals"


class BindContractError(RuntimeError):
    """The selected bind document is absent, corrupt, or unreadable."""


@dataclass(frozen=True, slots=True)
class BindIdentity:
    release_id: str
    derive_image_digest: str
    config_digest: str


@dataclass(frozen=True, slots=True)
class BindContract:
    digest: str
    identity: BindIdentity
    census: Census


def _sha256(ref: str) -> str:
    algorithm, separator, digest = str(ref or "").strip().partition(":")
    if separator != ":" or algorithm.lower() != "sha256":
        raise BindContractError(
            f"bind contract address {ref!r} is not sha256:<64 lowercase hex>"
        )
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise BindContractError(
            f"bind contract address {ref!r} is not sha256:<64 lowercase hex>"
        )
    return digest


def decode(raw: bytes, *, digest: str) -> BindContract:
    expected = _sha256(digest)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise BindContractError(
            f"bind contract {digest} fetched bytes hashing to sha256:{actual}"
        )
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BindContractError(
            f"bind contract {digest} is not JSON: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise BindContractError(f"bind contract {digest} is not a JSON object")
    if document.get("v") != BIND_CONTRACT_VERSION:
        raise BindContractError(
            f"bind contract {digest} states v={document.get('v')!r}; "
            f"this worker reads v={BIND_CONTRACT_VERSION}"
        )
    if document.get("kind") != BIND_CONTRACT_KIND:
        raise BindContractError(
            f"bind contract {digest} states kind={document.get('kind')!r}; "
            f"this worker reads {BIND_CONTRACT_KIND!r}"
        )
    identity = document.get("identity")
    if not isinstance(identity, dict):
        raise BindContractError(f"bind contract {digest} has no identity object")
    release_id = str(identity.get("release_id") or "").strip()
    image = str(identity.get("derive_image_digest") or "").strip()
    config = str(identity.get("config_digest") or "").strip()
    if not release_id or not image or not config:
        raise BindContractError(
            f"bind contract {digest} identity lacks release_id, "
            "derive_image_digest, or config_digest"
        )
    census_row = document.get("construction_census")
    if not isinstance(census_row, dict):
        raise BindContractError(
            f"bind contract {digest} has no construction_census object"
        )
    try:
        census = Census.from_document(census_row)
    except Exception as exc:
        raise BindContractError(
            f"bind contract {digest} carries an unreadable construction "
            f"census: {type(exc).__name__}: {exc}"
        ) from exc
    return BindContract(
        digest=digest,
        identity=BindIdentity(release_id, image, config),
        census=census,
    )


def fetch(
    digest: str,
    url: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> BindContract:
    _sha256(digest)
    if not str(url or "").strip():
        raise BindContractError(f"bind contract {digest} has no fetch URL")
    request = urllib.request.Request(
        str(url), headers={"Accept-Encoding": "identity"}, method="GET"
    )
    try:
        with opener(request, timeout=60.0) as response:
            raw = response.read(MAX_BIND_CONTRACT_BYTES + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise BindContractError(f"bind contract {digest} fetch failed: {exc}") from exc
    if len(raw) > MAX_BIND_CONTRACT_BYTES:
        raise BindContractError(
            f"bind contract {digest} exceeds {MAX_BIND_CONTRACT_BYTES} bytes"
        )
    return decode(bytes(raw), digest=digest)


def refusal_payload(contract: BindContract, mismatch: CensusMismatch) -> bytes:
    """Stable report body; no caller parses the mismatch's prose."""

    body: Mapping[str, Any] = {
        "release_id": contract.identity.release_id,
        "derive_image_digest": contract.identity.derive_image_digest,
        "config_digest": contract.identity.config_digest,
        "bind_contract_digest": contract.digest,
        "code": "bind_contract_census_mismatch",
        "invariant": mismatch.invariant,
        "component": mismatch.component,
        "tensor": mismatch.tensor,
        "detail": str(mismatch),
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def report_refusal(
    hub_base_url: str,
    contract: BindContract,
    mismatch: CensusMismatch,
    *,
    token: Optional[str] = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> None:
    base = str(hub_base_url or "").strip().rstrip("/")
    if not base:
        raise BindContractError("cannot report bind refusal: tensorhub URL is absent")
    bearer = str(token if token is not None else worker_credential.current() or "").strip()
    if not bearer:
        raise BindContractError("cannot report bind refusal: worker credential is absent")
    request = urllib.request.Request(
        base + REPORT_PATH,
        data=refusal_payload(contract, mismatch),
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=15.0) as response:
            status = int(getattr(response, "status", 200))
            if status < 200 or status >= 300:
                raise BindContractError(
                    f"bind refusal report answered HTTP {status}"
                )
    except (OSError, urllib.error.URLError) as exc:
        raise BindContractError(f"bind refusal report failed: {exc}") from exc


__all__ = [
    "BIND_CONTRACT_KIND",
    "BIND_CONTRACT_VERSION",
    "BindContract",
    "BindContractError",
    "BindIdentity",
    "decode",
    "fetch",
    "refusal_payload",
    "report_refusal",
]
