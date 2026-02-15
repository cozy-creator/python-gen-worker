from __future__ import annotations

import re

_NON_SLUG_CHARS = re.compile(r"[^a-z0-9.]+")
_DUP_SLUG_SEPARATORS = re.compile(r"-{2,}")


def slugify_name(raw: str) -> str:
    """
    Normalize an identifier to a URL/path-safe slug.

    Rules:
    - lowercase
    - `_` -> `-`
    - replace non [a-z0-9.] with `-`
    - collapse repeated `-`
    - trim leading/trailing `-` and `.`
    - cap to 128 chars
    """
    raw = (raw or "").strip().lower().replace("_", "-")
    if not raw:
        return ""
    raw = _NON_SLUG_CHARS.sub("-", raw)
    raw = _DUP_SLUG_SEPARATORS.sub("-", raw)
    raw = raw.strip("-.")
    if len(raw) > 128:
        raw = raw[:128].strip("-.")
    return raw


def slugify_endpoint_name(raw: str) -> str:
    return slugify_name(raw)


def slugify_function_name(raw: str) -> str:
    return slugify_name(raw)

