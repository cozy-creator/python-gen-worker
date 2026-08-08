"""Worker-owned compile-cell identity (gw#581, th#883).

ONE compatibility brain: the worker computes the exact key of the compile
cell its runtime can execute, from its OWN diagnostics, with this module —
and every consumer of cell identity uses the same code:

* the fleet executor advertises the key it wants (pull-by-key: the hub
  serves bytes by key or records demand for the forge; it never matches);
* the production mint (``compile_cache.build``) stamps the key it actually
  produced, derived from the artifact's own recorded axes;
* cozy-local's self-mint (gw#555) looks up / saves its store by the same
  key.

The key is the RECIPE DIGEST (Paul's exact-identity ruling): "look at
our code and say 'this is the graph we need' — that is our unique
identifier. If the recipe changes it gets a new identifier, stranding the
old ones, which is fine." No version comparison, no relaxable axes, no
cross-key candidates — a key either matches exactly or the cell does not
exist for this runtime.

    format        artifact format version (compile_cache.ARTIFACT_FORMAT)
    kind          "inductor" (TRT engines keep their own legacy identity)
    family        graph identity: fine-tunes of one family share cells
    lane          canonical traced weight lane token ("", w8a16, w8a8,
                  [-loraN]) — lane graphs differ (gw#534/gw#561)
    sm            compute capability (sm_100, ...)
    contract      digest of the DECLARED shape contract (SDK v2, pgw#647):
                  shapes, targets, text_len, dynamic dims, regional mode,
                  lora bucket, warm guidance classes
    env_seal      digest of the execution-environment seal (pgw#696):
                  process posture + frozen config flags + portable inductor
                  config (+ operator epoch). Internally versioned (seal_v)
    toolchain     CONTENT digest of the compile stack (pgw#710): dist-info
                  RECORDs of torch/triton/nvidia-*/diffusers/transformers
                  + the bundled ptxas/nvdisasm binaries. Replaces the old
                  torch/triton/cuda/diffusers/transformers VERSION axes —
                  content, never version strings

pgw#990 DROPS ``code_closure`` from the key. It is still RECORDED on
every artifact and still drives ``compile_cache``'s local re-trace memo, but
it is not identity: Paul's final ruling is that identity is the COMPUTATION
(traced graph x sm x toolchain x env_seal) and "code hashes are a memo, never
identity". A 147-file content hash made every wheel release re-key every cell
in the fleet for edits that could not change a traced graph — 0.93.0 -> 0.93.1
moved the sdxl key on three plumbing files. Cells are stranded by SCHEME here,
by name, once, instead of silently on every release.

Axes deliberately NOT in the key, recorded in metadata for observability
and runtime compat checks (``compile_cache.verify``) only: ``sku`` (pgw#691
— no guard or artifact fact observes it), ``cuda_driver`` (gw#577),
``torch``/``triton``/``cuda``/``gen_worker``/``diffusers``/``transformers``
version strings and ``image_digest`` (exact identity — their CONTENT rides the
toolchain and code_closure axes; version strings and image identity are
observability, not identity).

A wrong key can only produce a MISS (eager + demand + forge), never a
refusal: verify-on-receipt of a self-requested cell degenerates to a digest
check, and any failure to arm one is by construction a selection-logic bug
that must surface loudly (``cell_selection_bug``), never a silent eager
fallback.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping
from . import env_seal

# pgw#958 (DESIGN-RULINGS §1.27(g); Paul 2026-08-04, reaffirmed 2026-08-07
# over pgw#990's ck6 — "the ck6 is wrong and can be removed, the v1 is the
# correct one"): the counter restarts at 1, and the pre-existing ck1..ck6
# corpus is PURGED in the same cut (th#1636), so no two cells ever share a
# scheme token with different meanings. ck1 is the only live scheme; new
# identity facts ride the content digests (seal_v / closure / toolchain
# values), never new axes and never a new scheme number.
KEY_SCHEME = "ck1"
_PREFIX = KEY_SCHEME + "-"
# The key digest doubles as the store flavor token, whose shared grammar
# (th#597 C5: [a-z0-9][a-z0-9._-]{0,63}, Go+Py identical) caps tokens at 64
# chars: 56 hex chars of SHA-256 (224 bits) keeps the whole key at 60.
_DIGEST_HEX = 56

# Axes that must be non-empty for a computable key: a runtime that cannot
# state them has no cell identity (CPU-only build, failed CUDA probe).
_REQUIRED = ("format", "kind", "family", "sm", "contract", "env_seal",
             "toolchain")
# Axes that may be legitimately absent ("" => omitted from canonical form):
# lane "" is the plain-resident graph family; mode "" is whole-graph
# compilation ("regional" per-block cells are different artifacts, ie#381).
_OPTIONAL = ("lane", "mode")


class CellKeyError(ValueError):
    """The runtime cannot state a required key axis."""


@dataclass(frozen=True)
class CellKey:
    """A computed cell identity: canonical axes + their digest."""

    axes: tuple  # sorted ((name, value), ...) — empty values omitted

    def axes_dict(self) -> Dict[str, str]:
        return dict(self.axes)

    def canonical(self) -> str:
        return json.dumps(
            self.axes_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        )

    @property
    def digest(self) -> str:
        h = hashlib.sha256(self.canonical().encode()).hexdigest()
        return _PREFIX + h[:_DIGEST_HEX]


def is_key(value: str) -> bool:
    """True when ``value`` has cell-key SHAPE: ``ck`` + 1-2 scheme digits +
    ``-`` + 56 lowercase hex.

    Scheme-AGNOSTIC, byte-for-byte the grammar tensorhub's
    ``compilecache.IsCellKey`` enforces, and for the same reason it gives
    (th#1183): pinning the current scheme here turns every other-scheme cell
    into ``unreadable_cell_key``, which is both a lie and a filter no axis
    justifies. A cell of an older scheme is admitted to the candidate list and
    then ruled on by the axes that actually decide whether this runtime can
    execute it — the artifact contract, the identity axes and the numerics
    gate — not by the label on it.
    """
    v = str(value or "")
    rest = v[2:] if v.startswith("ck") else ""
    if not rest:
        return False
    digits = 0
    while digits < len(rest) and rest[digits].isdigit():
        digits += 1
    if not 1 <= digits <= 2 or digits >= len(rest) or rest[digits] != "-":
        return False
    hexpart = rest[digits + 1:]
    return (
        len(hexpart) == _DIGEST_HEX
        and all(c in "0123456789abcdef" for c in hexpart)
    )


def from_axes(axes: Mapping[str, str]) -> CellKey:
    """Canonicalize an axes mapping into a :class:`CellKey`.

    Unknown axes are rejected (a new axis is a KEY_SCHEME bump, never a
    silent widening); empty optional axes are omitted so "" and absent can
    never diverge.
    """
    clean: Dict[str, str] = {}
    for name, value in axes.items():
        text = str(value or "").strip()
        if name not in _REQUIRED and name not in _OPTIONAL:
            raise CellKeyError(f"unknown cell-key axis {name!r}")
        if text:
            clean[name] = text
    missing = [name for name in _REQUIRED if not clean.get(name)]
    if missing:
        raise CellKeyError(
            f"cell key requires axes {missing!r} (got {sorted(clean)!r})")
    return CellKey(axes=tuple(sorted(clean.items())))


def _canonical_execution_lane(weight_lane: str, lora_bucket: int = 0) -> str:
    from . import compile_cache as cc  # cycle: compile_cache imports cell_key

    base, observed = cc.execution_lane_bucket(str(weight_lane or ""))
    bucket = observed or int(lora_bucket or 0)
    token = cc.execution_lane_token(base)
    if bucket:
        return f"{token}-lora{bucket}" if token else f"lora{bucket}"
    return token


def facts_digest(facts: Mapping[str, Any]) -> str:
    """16-hex canonical digest of one recorded fact block (toolchain /
    code_closure axes) — computed identically from live probes (compute)
    and recorded metadata (from_artifact_metadata), so a stamp can never
    disagree with the facts it summarizes."""
    encoded = json.dumps(
        dict(facts), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def contract_digest(facts: Mapping[str, Any]) -> str:
    """Digest of the DECLARED shape-contract facts (SDK v2 axis). Both
    sides state the same canonical dict — the worker from its EndpointSpec
    (``compile_cache.declared_contract_facts``), the mint from its build
    inputs, the artifact from its recorded ``shape_contract`` block — so the
    digest can never disagree with the facts it summarizes."""
    encoded = json.dumps(
        dict(facts), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def compute(
    family: str,
    weight_lane: str = "",
    lora_bucket: int = 0,
    *,
    contract: str,
    regional: bool = False,
) -> CellKey:
    """The key THIS runtime wants for ``family`` on ``weight_lane`` —
    computed purely statically (no trace, no execution): live probes for
    sm/posture/config, dist-info + binary content for the toolchain.
    (pgw#1030: the ``closure_roots`` parameter is deleted — pgw#990 removed
    code identity from the key and the body never read it.) Raises
    :class:`CellKeyError` when a required axis is unavailable — callers on
    non-CUDA runtimes simply have no key."""
    from . import compile_cache as cc  # cycle: compile_cache imports cell_key

    rt = cc.runtime_key()
    return from_axes({
        "format": str(cc.ARTIFACT_FORMAT),
        "kind": "inductor",
        "family": str(family or ""),
        "lane": _canonical_execution_lane(weight_lane, lora_bucket),
        "mode": "regional" if regional else "",
        "sm": rt["sm"],
        "contract": str(contract or ""),
        "env_seal": env_seal.seal_digest(env_seal.effective_seal()),
        "toolchain": facts_digest(dict(cc.toolchain_digest())),
    })


def from_artifact_metadata(meta: Mapping[str, Any]) -> CellKey:
    """The key an artifact's OWN recorded axes describe.

    Derived from the metadata, never from the stamped ``cell_key`` field, so
    a stamp can never disagree with the axes it summarizes. Raises
    :class:`CellKeyError` for artifacts that don't record every required axis.
    That is the ONLY verdict — pgw#950 deleted the second, axis-by-axis verify
    path keyless cells used to fall back to, on both the fleet and the local
    store. A cell with no computable key is refused and re-minted.

    EXPORTED (``aot-inductor``) cells are refused here BY NAME (pgw#735): they
    ride the same key space — the axis names are what :func:`from_axes`
    validates, and the kind is an envelope value, so no scheme bump was needed —
    but their axes are not an inductor cache's, and their key is STAMPED at mint.
    Read ``meta["cell_key"]``; do not recompute an exported cell's identity from
    these fields.
    """
    kind = str(meta.get("kind") or "")
    if kind == "aot-inductor":
        raise CellKeyError(
            "artifact kind 'aot-inductor' (exported .pt2) has a STAMPED key — "
            "read meta['cell_key'] instead of recomputing from inductor-cache "
            f"axes (stamped={str(meta.get('cell_key') or '') or 'MISSING'})")
    if kind != "torch-inductor-cache":
        raise CellKeyError(f"artifact kind {kind!r} has no cell-key identity")

    mode = str(meta.get("compile_mode") or "whole")
    contract_facts = meta.get("shape_contract")
    if not isinstance(contract_facts, dict) or not contract_facts:
        raise CellKeyError(
            "artifact records no shape_contract block (pre-cell-key cell); "
            "no key identity — a newer-contract worker must not consume it"
        )
    seal = meta.get(env_seal.SEAL_KEY)
    if not isinstance(seal, dict) or not seal:
        raise CellKeyError(
            "artifact records no env_seal block; no key identity — its "
            "execution environment is unproven"
        )
    toolchain = meta.get("toolchain")
    if not isinstance(toolchain, dict) or not toolchain:
        raise CellKeyError(
            "artifact records no toolchain block; no recipe identity")
    return from_axes({
        "format": str(meta.get("format") or ""),
        "kind": "inductor",
        "family": str(meta.get("family") or ""),
        "lane": _canonical_execution_lane(
            str(meta.get("weight_lane") or ""),
            int(meta.get("lora_bucket") or 0),
        ),
        "mode": "" if mode == "whole" else mode,
        "sm": str(meta.get("sm") or ""),
        "contract": contract_digest(contract_facts),
        "env_seal": env_seal.seal_digest(seal),
        "toolchain": facts_digest(toolchain),
    })


def stamp(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp ``meta`` with the key its axes describe (mint-time, both the
    production build and the local self-mint). No-op when the axes are not
    key-complete (e.g. focused unit fixtures)."""
    try:
        meta["cell_key"] = from_artifact_metadata(meta).digest
    except CellKeyError:
        meta.pop("cell_key", None)
    return meta


def mismatch(meta: Mapping[str, Any], requested: "str | CellKey") -> str:
    """'' when the artifact's axes describe exactly the requested key, else
    a named reason. This is the entire receipt check for a self-requested
    cell: transport integrity is the CAS digest; identity is this. Passing
    the full :class:`CellKey` (callers that computed it themselves) names
    the first differing axis with both values in the reason."""
    requested_key = requested.digest if isinstance(requested, CellKey) \
        else str(requested or "")
    if not is_key(requested_key):
        return f"requested key {requested_key!r} is not a cell key"
    try:
        have = from_artifact_metadata(meta)
    except CellKeyError as exc:
        return f"artifact records no computable key ({exc})"
    if have.digest == requested_key:
        return ""
    if isinstance(requested, CellKey):
        want_axes, have_axes = requested.axes_dict(), have.axes_dict()
        for name in sorted(set(want_axes) | set(have_axes)):
            if want_axes.get(name, "") != have_axes.get(name, ""):
                return (
                    f"{name}: cell {have_axes.get(name, '')!r} != runtime "
                    f"{want_axes.get(name, '')!r}"
                )
    return f"artifact key {have.digest} != requested {requested_key}"


__all__ = [
    "KEY_SCHEME",
    "CellKey",
    "CellKeyError",
    "compute",
    "contract_digest",
    "facts_digest",
    "from_artifact_metadata",
    "from_axes",
    "is_key",
    "mismatch",
    "stamp",
]
