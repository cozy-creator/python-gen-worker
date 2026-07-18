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

The key is a deterministic digest over the post-gw#577 HONEST axes — the
facts that actually change compiled-kernel identity:

    format        artifact format version (compile_cache.ARTIFACT_FORMAT)
    kind          "inductor" (TRT engines keep their own legacy identity)
    family        graph identity: fine-tunes of one family share cells
    lane          canonical traced weight lane token ("", w8a16, w8a8,
                  [-loraN]) — lane graphs differ (gw#534/gw#561)
    sku           GPU SKU slug
    sm            compute capability (sm_100, ...)
    cuda          CUDA runtime the torch wheel was built for
    torch/triton  exact wheel versions (triton's disk cache keys on the
                  wheel's ptxas + SM arch — the host driver deliberately
                  never enters the key, gw#577)
    gen_worker    exact gen-worker version (graph-shaping code)
    diffusers/transformers  exact lib versions when installed
    image_digest  the SERVING image OCI digest (absent for local runtimes)

A wrong key can only produce a MISS (eager + demand + forge), never a
refusal: verify-on-receipt of a self-requested cell degenerates to a digest
check, and any failure to arm one is by construction a selection-logic bug
that must surface loudly (``cell_selection_bug``), never a silent eager
fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

KEY_SCHEME = "ck1"
_PREFIX = KEY_SCHEME + "-"
# The key digest doubles as the store flavor token, whose shared grammar
# (th#597 C5: [a-z0-9][a-z0-9._-]{0,63}, Go+Py identical) caps tokens at 64
# chars: 56 hex chars of SHA-256 (224 bits) keeps the whole key at 60.
_DIGEST_HEX = 56

# Axes that must be non-empty for a computable key: a runtime that cannot
# state them has no cell identity (CPU-only build, failed CUDA probe).
_REQUIRED = ("format", "kind", "family", "sku", "sm", "cuda", "torch",
             "triton", "gen_worker")
# Axes that may be legitimately absent ("" => omitted from canonical form):
# image_digest is absent on local runtimes; libs may not be installed; lane
# "" is the plain-resident graph family; mode "" is whole-graph compilation
# ("regional" per-block cells are different artifacts, ie#381).
_OPTIONAL = ("lane", "mode", "image_digest", "diffusers", "transformers")


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
    """True when ``value`` is a cell-key digest string."""
    v = str(value or "")
    return (
        v.startswith(_PREFIX)
        and len(v) == len(_PREFIX) + _DIGEST_HEX
        and all(c in "0123456789abcdef" for c in v[len(_PREFIX):])
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


def _canonical_lane(weight_lane: str, lora_bucket: int = 0) -> str:
    from . import compile_cache as cc

    base, observed = cc.lane_bucket(str(weight_lane or ""))
    bucket = observed or int(lora_bucket or 0)
    token = cc.lane_token(base)
    if bucket:
        return f"{token}-lora{bucket}" if token else f"lora{bucket}"
    return token


def compute(
    family: str,
    weight_lane: str = "",
    lora_bucket: int = 0,
    *,
    regional: bool = False,
    image_digest: Optional[str] = None,
) -> CellKey:
    """The key THIS runtime wants for ``family`` on ``weight_lane``.

    Probes the live process (the same probes ``compile_cache.verify`` trusts).
    ``image_digest=None`` uses this process's ``WORKER_IMAGE_DIGEST``; pass an
    explicit value (the SERVING image digest at mint time, or ``""`` for a
    local runtime) to override. Raises :class:`CellKeyError` when a required
    axis is unavailable — callers on non-CUDA runtimes simply have no key.
    """
    from . import compile_cache as cc

    rt = cc.runtime_key()
    if image_digest is None:
        image_digest = os.environ.get("WORKER_IMAGE_DIGEST", "")
    libs = cc._lib_versions()
    return from_axes({
        "format": str(cc.ARTIFACT_FORMAT),
        "kind": "inductor",
        "family": str(family or ""),
        "lane": _canonical_lane(weight_lane, lora_bucket),
        "mode": "regional" if regional else "",
        "sku": rt["sku"],
        "sm": rt["sm"],
        "cuda": rt["cuda"],
        "torch": rt["torch"],
        "triton": rt["triton"],
        "gen_worker": cc.gen_worker_version(),
        "diffusers": libs.get("diffusers", ""),
        "transformers": libs.get("transformers", ""),
        "image_digest": str(image_digest or ""),
    })


def from_artifact_metadata(meta: Mapping[str, Any]) -> CellKey:
    """The key an artifact's OWN recorded axes describe.

    Derived from the metadata, never from the stamped ``cell_key`` field, so
    a stamp can never disagree with the axes it summarizes. Raises
    :class:`CellKeyError` for artifacts that don't record every required axis
    (pre-gw#581 cells have no key and stay on the legacy verify path).
    """
    kind = str(meta.get("kind") or "")
    if kind != "torch-inductor-cache":
        raise CellKeyError(f"artifact kind {kind!r} has no cell-key identity")
    libs = meta.get("libs") or {}
    mode = str(meta.get("compile_mode") or "whole")
    return from_axes({
        "format": str(meta.get("format") or ""),
        "kind": "inductor",
        "family": str(meta.get("family") or ""),
        "lane": _canonical_lane(
            str(meta.get("weight_lane") or ""),
            int(meta.get("lora_bucket") or 0),
        ),
        "mode": "" if mode == "whole" else mode,
        "sku": str(meta.get("sku") or ""),
        "sm": str(meta.get("sm") or ""),
        "cuda": str(meta.get("cuda") or ""),
        "torch": str(meta.get("torch") or ""),
        "triton": str(meta.get("triton") or ""),
        "gen_worker": str(meta.get("gen_worker") or ""),
        "diffusers": str(libs.get("diffusers") or ""),
        "transformers": str(libs.get("transformers") or ""),
        "image_digest": str(meta.get("image_digest") or ""),
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
    "from_artifact_metadata",
    "from_axes",
    "is_key",
    "mismatch",
    "stamp",
]
