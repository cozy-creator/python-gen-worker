"""The ONE guarded HTTP fetch for endpoint code — never hand-roll urlopen(url).read(): that is an SSRF hole (the worker sits inside a datacenter network with a cloud metadata endpoint at 169.254.169.254) plus an unbounded read. The policy: http/https only; every hop's host resolved and refused on private/loopback/link-local/multicast/reserved addresses; redirects followed MANUALLY with the policy re-applied per hop (urlopen follows silently — a public URL 302-ing to the metadata service is the standard bypass); size streamed against a byte cap checked against Content-Length first and again per chunk; an explicit timeout on every hop; MIME sniffed from the bytes, never trusted from the header. Known limit, stated rather than papered over: validation resolves the hostname and the connection resolves it again, so a DNS-rebinding attacker can win that race — per-hop validation is the practical guard; do not describe it as proof against a rebinding adversary."""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

from .api.errors import RetryableError, ValidationError
from .request_context._helpers import _infer_mime_type, _url_is_blocked

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_TIMEOUT_S",
    "FetchedUrl",
    "MAX_REDIRECTS",
    "fetch_bytes",
    "open_guarded_stream",
]

DEFAULT_MAX_BYTES = 50 << 20
DEFAULT_TIMEOUT_S = 30.0
MAX_REDIRECTS = 5
_CHUNK = 1 << 20

ALLOWED_HOSTS_ENV = "GEN_WORKER_URL_FETCH_ALLOWED_HOSTS"


@dataclass(frozen=True)
class FetchedUrl:
    """Bytes plus the provenance a caller needs to reason about them."""

    url: str
    final_url: str
    data: bytes
    mime: str

    @property
    def size_bytes(self) -> int:
        return len(self.data)


def _env_allowed_hosts() -> frozenset[str]:
    raw = os.environ.get(ALLOWED_HOSTS_ENV, "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def _check_hop(
    url: str,
    *,
    allowed_hosts: frozenset[str],
    what: str,
    is_blocked: Callable[[str], bool] | None = None,
) -> None:
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        raise ValidationError(f"url_fetch_refused: {what} is not a valid URL") from None
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise ValidationError(
            f"url_fetch_refused: {what} must be an http(s) URL with a host"
        )
    host = parsed.hostname.lower()
    env_hosts = _env_allowed_hosts()
    if env_hosts and host not in env_hosts:
        raise ValidationError(
            f"url_fetch_refused: {what} host {host!r} is outside this deployment's "
            f"{ALLOWED_HOSTS_ENV} allowlist"
        )
    if allowed_hosts and host not in allowed_hosts:
        raise ValidationError(
            f"url_fetch_refused: {what} host {host!r} is not in the caller's allowlist"
        )
    check = is_blocked or _url_is_blocked
    try:
        blocked = check(url)
    except Exception:
        raise RetryableError("url_fetch_failed: destination policy check failed") from None
    if blocked:
        raise ValidationError(
            f"url_fetch_refused: {what} resolves to a non-public address"
        )


class _NoRedirect(urllib.request.HTTPRedirectHandler):

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_opener = urllib.request.build_opener(_NoRedirect)


@contextmanager
def open_guarded_stream(
    url: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    allowed_hosts: Sequence[str] = (),
    extra_allowed_hosts: Sequence[str] = (),
    max_redirects: int = MAX_REDIRECTS,
    headers: dict[str, str] | None = None,
    is_blocked: Callable[[str], bool] | None = None,
) -> Iterator[Any]:
    """Open ``url`` with the full policy applied to every hop."""
    allow = frozenset(h.strip().lower() for h in allowed_hosts if h and h.strip())
    exempt = frozenset(h.strip().lower() for h in extra_allowed_hosts if h and h.strip())
    current = str(url or "")
    for hop in range(max_redirects + 1):
        host = (urllib.parse.urlsplit(current).hostname or "").lower()
        what = "url" if hop == 0 else f"redirect hop {hop}"
        if host and host in exempt:
            if urllib.parse.urlsplit(current).scheme.lower() not in ("http", "https"):
                raise ValidationError(f"url_fetch_refused: {what} must be an http(s) URL")
        else:
            _check_hop(
                current, allowed_hosts=allow, what=what, is_blocked=is_blocked,
            )
        request = urllib.request.Request(
            current, headers={"User-Agent": "gen-worker", **(headers or {})}
        )
        try:
            response = _opener.open(request, timeout=timeout_s)
        except urllib.error.HTTPError as exc:
            location = str(exc.headers.get("Location") or "").strip() if exc.headers else ""
            if 300 <= int(exc.code) < 400 and location:
                exc.close()
                current = urllib.parse.urljoin(current, location)
                continue
            exc.close()
            raise
        except (ValidationError, RetryableError):
            raise
        except Exception as exc:
            raise RetryableError(f"url_fetch_failed: {type(exc).__name__}") from None
        try:
            yield response
        finally:
            response.close()
        return
    raise ValidationError(f"url_fetch_refused: {url} exceeded {max_redirects} redirects")


def fetch_bytes(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    allowed_hosts: Sequence[str] = (),
    allowed_mime_types: Sequence[str] = (),
    max_redirects: int = MAX_REDIRECTS,
) -> FetchedUrl:
    """Fetch ``url`` under the full policy and return its bytes."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    try:
        return _read_capped(
            url, max_bytes, timeout_s, allowed_hosts, allowed_mime_types, max_redirects,
        )
    except urllib.error.HTTPError as exc:
        raise ValidationError(
            f"url_fetch_refused: {url} returned HTTP {exc.code}"
        ) from None


def _read_capped(
    url: str,
    max_bytes: int,
    timeout_s: float,
    allowed_hosts: Sequence[str],
    allowed_mime_types: Sequence[str],
    max_redirects: int,
) -> FetchedUrl:
    chunks: list[bytes] = []
    total = 0
    final_url = str(url)
    with open_guarded_stream(
        url,
        timeout_s=timeout_s,
        allowed_hosts=allowed_hosts,
        max_redirects=max_redirects,
    ) as response:
        final_url = str(getattr(response, "url", url) or url)
        raw_length = str(response.headers.get("Content-Length") or "").strip()
        if raw_length.isdigit() and int(raw_length) > max_bytes:
            raise ValidationError(
                f"url_fetch_too_large: {url} declares {raw_length} bytes "
                f"(cap {max_bytes})"
            )
        while True:
            try:
                chunk = response.read(_CHUNK)
            except Exception as exc:
                raise RetryableError(f"url_fetch_failed: {type(exc).__name__}") from None
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValidationError(
                    f"url_fetch_too_large: {url} exceeds its {max_bytes}-byte cap"
                )
            chunks.append(chunk)
    data = b"".join(chunks)
    mime = _infer_mime_type(final_url, data[:64])
    if allowed_mime_types:
        allowed = {m.strip().lower() for m in allowed_mime_types if m and m.strip()}
        if mime.lower() not in allowed:
            raise ValidationError(
                f"url_fetch_refused: {url} content type {mime!r} is not allowed"
            )
    return FetchedUrl(url=str(url), final_url=final_url, data=data, mime=mime)


