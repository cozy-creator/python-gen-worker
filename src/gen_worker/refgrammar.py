"""The constants of THE tensorhub model-ref grammar (th#597 C5) that every
layer must agree on — the Python twin of tensorhub's ``internal/refgrammar``.

A leaf module on purpose: the ref parser (:mod:`gen_worker.models.refs`) and
the compiled-graph key grammar (:mod:`gen_worker.compiled_graph_key`) both need the
bound, and neither may import the other. The Go half states the same reason
for the same shape.
"""

from __future__ import annotations

#: Bounds the ref grammar's ``#fragment`` token (``[a-z0-9][a-z0-9._-]*``), and
#: is therefore also the bound on a compiled-graph key — the key IS the
#: fragment of ``root/family-<f>#<key>``, on both sides of the wire
#: (:func:`gen_worker.compile_cache.parse_compiled_graph_ref` routes through
#: :func:`gen_worker.models.refs.parse_model_ref`, so it enforces this same
#: token grammar; Go's ``compilecache.KeyRef`` is the twin).
#:
#: th#1897 — WHY IT MOVED FROM 64. It was 64 because every key ever minted was
#: ``<3-4 char scheme>-<56 hex>`` = 60, and the 64 was recorded as headroom.
#: DESIGN-RULINGS §1.38 makes the scheme ``cg-key-v1``, so a key is 10 + 56 =
#: 66 characters and does not fit. The consequence is not cosmetic and does not
#: need the hub to appear: ``parse_compiled_graph_ref`` raises on the fragment regex and
#: returns ``("", "")``, so the pod cannot name the family of the artifact it
#: just armed.
#:
#: 96 is 56 hex + one separator + a 39-byte scheme budget. It is a bound, not a
#: target. The number is shared with tensorhub through the vendored corpora
#: (boundary vectors at 96 and 97 in both) and fenced by
#: ``scripts/grammar-vector-drift.sh``.
MAX_FRAGMENT_LEN = 96
