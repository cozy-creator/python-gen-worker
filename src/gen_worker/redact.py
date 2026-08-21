"""Redaction for anything that may reach a tenant or the hub."""

import re
from typing import Pattern, Sequence, Tuple

CREDENTIAL_REDACTIONS: Tuple[Pattern[str], ...] = (
    re.compile(r"Bearer\s+[^\s\"'&]+"),
    re.compile(r"(?:X-Amz-[A-Za-z0-9-]+|Signature)=[^&\s\"']*"),
    re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+"),
)

PATH_REDACTIONS: Tuple[Pattern[str], ...] = (
    re.compile(r"(?<![\w:/])/(?:[\w.@+-]+/)+[\w.@+-]*"),
)

REDACTIONS = CREDENTIAL_REDACTIONS + PATH_REDACTIONS


def _apply(message: str, patterns: Sequence[Pattern[str]]) -> str:
    out = str(message or "")
    for pat in patterns:
        out = pat.sub("[redacted]", out)
    return out


def sanitize(message: str) -> str:
    """The short client-facing spelling: credentials AND paths, capped at 1 KB."""
    return _apply(str(message or "").strip(), REDACTIONS)[:1024]


def sanitize_credentials(message: str) -> str:
    """Secrets only, and NOT length-capped — the caller owns the bound."""
    return _apply(message, CREDENTIAL_REDACTIONS)
