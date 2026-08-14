"""Redaction for anything that may reach a tenant or the hub.

`executor.py` and `models/store.py` both sanitize outbound
messages, and the store may not import the executor — so the helper lands in
its own module rather than being duplicated or re-exported.
"""

import re

REDACTIONS = (
    re.compile(r"Bearer\s+[^\s\"'&]+"),
    re.compile(r"(?:X-Amz-[A-Za-z0-9-]+|Signature)=[^&\s\"']*"),
    # Absolute unix filesystem paths (/tmp/..., /app/..., /home/...): require
    # two segments so bare "/" and owner/repo-style refs survive, and no
    # scheme/word directly before the slash so URL paths inside https://...
    # stay intact. Pods are linux-only; no Windows drive-path variant.
    re.compile(r"(?<![\w:/])/(?:[\w.@+-]+/)+[\w.@+-]*"),
)


def sanitize(message: str) -> str:
    out = str(message or "").strip()
    for pat in REDACTIONS:
        out = pat.sub("[redacted]", out)
    return out[:1024]
