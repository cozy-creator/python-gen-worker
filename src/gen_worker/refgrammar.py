from __future__ import annotations

# Bounds the ref grammar's #fragment token, and therefore a compiled-graph key — the key IS the fragment of root/family-<f>#<key> on both sides of the wire (Go's compilecache.KeyRef is the twin). 96 = 56 hex + one separator + a 39-byte scheme budget; a bound, not a target. Shared with tensorhub through the vendored corpora (boundary vectors at 96 and 97 on both sides), pinned by KEY_GRAMMAR_DIGEST.
MAX_FRAGMENT_LEN = 96
