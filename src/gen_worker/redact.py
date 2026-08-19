"""Redaction for anything that may reach a tenant or the hub.

`executor.py` and `models/store.py` both sanitize outbound
messages, and the store may not import the executor — so the helper lands in
its own module rather than being duplicated or re-exported.

pgw#1474 split the patterns into two GROUPS, because they answer two different
questions and one consumer needs only the first. Credentials are secrets and
never belong on a wire. Absolute filesystem paths are not secrets — they are
redacted from short client-facing messages because a pod's layout is not the
caller's business, but in a TRACEBACK they are the entire diagnosis
(``File "/opt/endpoint/jobs/quantize.py", line 118``). Running the path pattern
over a traceback would delete exactly the thing th#2201 exists to deliver.
"""

import re
from typing import Pattern, Sequence, Tuple

CREDENTIAL_REDACTIONS: Tuple[Pattern[str], ...] = (
    re.compile(r"Bearer\s+[^\s\"'&]+"),
    re.compile(r"(?:X-Amz-[A-Za-z0-9-]+|Signature)=[^&\s\"']*"),
    # A JWT anywhere in free text (a capability token dragged into an
    # exception message by whatever refused it). Three base64url segments.
    re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+"),
)

PATH_REDACTIONS: Tuple[Pattern[str], ...] = (
    # Absolute unix filesystem paths (/tmp/..., /app/..., /home/...): require
    # two segments so bare "/" and owner/repo-style refs survive, and no
    # scheme/word directly before the slash so URL paths inside https://...
    # stay intact. Pods are linux-only; no Windows drive-path variant.
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
    """Secrets only, and NOT length-capped — the caller owns the bound.

    For a traceback, where the paths are the answer and the bound is a tail
    truncation rather than a head one.
    """
    return _apply(message, CREDENTIAL_REDACTIONS)
