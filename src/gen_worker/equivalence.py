"""Equivalence adoption (pgw#700): change-detection instead of conservatism.

Paul's ruling (th#1229, 2026-07-26): the version-axis barriers drop —
``gen_worker``/``image_digest`` stay RECORDED in the key for exact-match
fast lookup and observability, but they are non-blocking for adoption.
What replaces them is DETECTION of actual change, two tiers behind
``GEN_WORKER_EQUIVALENCE_ADOPTION`` (default flips ON once pgw#710/711/712
land and the SDXL proof verdict is green):

* **FAST tier — code closure.** The mint records the source files actually
  loaded from the trace-relevant packages, content-digested
  (``compile_cache.code_closure``), plus the pgw#710 toolchain digests and
  torch/triton ``content_keys``. The consumer digests ITS copies of the
  recorded file set (:func:`closure_delta`); all identical means the code
  that shapes graphs is byte-identical — adopt, no probe mint, no manifest
  comparison. Guard-closure philosophy applied to code identity: recorded
  dependencies, never declared versions.
* **SLOW tier — manifest + fingerprint.** When the closure differs, the
  cell is adoptable only on proof: closed manifest, sealed posture live,
  composition fingerprint identical module-for-module, clean
  :func:`compile_cache.artifact_drift`. Fleet-scale, the slow tier is the
  hub's probe-mint compare-and-bless flow (one probe per (family, sm),
  manifests compared, identical -> old cells blessed; the th-side issue
  carries that design) — SDK-side this module IS the comparison.

Safety floor (unchanged by the ruling): pgw#711 confirmed-only candidates,
pgw#712 fencing — a successful verdict stamps :data:`ADOPTION_MARK` and
:func:`fleet_cells` refuses to publish a marked cell (the relaxation must
never launder into an exact-key mint); :func:`select` enforces the Nix
unicity rule (differing manifests for one relaxed identity =
``cell_equivalence_divergence``, serve neither).

The hub-candidate wiring (executor/preload) belongs to the train lane;
this module is the complete SDK half.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from . import cell_key, compile_cache, guard_closure

logger = logging.getLogger(__name__)

FLAG_ENV = "GEN_WORKER_EQUIVALENCE_ADOPTION"

# The metadata mark an equivalence adoption stamps: the sorted list of axes
# the cell's key differed on. Its PRESENCE fences republication (pgw#712).
ADOPTION_MARK = "equivalence_adopted"

# The conservative axes a proof may bridge (pgw#700): SDK version and image
# identity — env_seal joins only after pgw#710's toolchain closure ships in
# the SEAL (see th#1229 preconditions). Everything else is identity.
DESIGNATED_AXES = frozenset({"gen_worker", "image_digest"})


def enabled() -> bool:
    return os.environ.get(FLAG_ENV, "").strip().lower() in ("1", "true", "on")


def closure_delta(recorded: Mapping[str, str]) -> List[str]:
    """Named differences between a cell's recorded code closure and THIS
    process's copies of the same files (fast tier). Empty list = the code
    that shaped the cell's graphs is byte-identical here."""
    out: List[str] = []
    roots: Dict[str, Optional[Path]] = {}
    for rel in sorted(recorded):
        root = rel.split("/", 1)[0]
        if root not in roots:
            base: Optional[Path] = None
            try:
                spec = importlib.util.find_spec(root)
                if spec and spec.origin:
                    base = Path(spec.origin).parent.parent
            except (ImportError, ValueError):
                base = None
            roots[root] = base
        base = roots[root]
        if base is None:
            out.append(f"{rel}: package {root!r} not importable here")
            continue
        path = base / rel
        if not path.is_file():
            out.append(f"{rel}: absent in this runtime")
            continue
        have = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        want = str(recorded.get(rel) or "")
        if have != want:
            out.append(f"{rel}: cell {want} != runtime {have}")
    return out


def _axes_delta(
    cell: Mapping[str, str], want: Mapping[str, str],
) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for axis in sorted(set(cell) | set(want)):
        have, need = cell.get(axis, ""), want.get(axis, "")
        if have != need:
            out.append((axis, have, need))
    return out


def verdict(
    meta: MutableMapping[str, Any],
    want: "cell_key.CellKey",
    pipeline: Any,
    cfg: Any,
) -> str:
    """'' when ``meta``'s cell may be adopted by equivalence against the key
    THIS runtime wants; else the named refusal. Two tiers (module doc):
    FAST = the recorded code closure is byte-identical here; SLOW = the
    composition-fingerprint proof. The safety floor (closed manifest, live
    posture, byte-identical compile stack, clean drift) holds for both.
    Stamps :data:`ADOPTION_MARK` (the bridged axes) on success."""
    if not enabled():
        return f"equivalence adoption disabled ({FLAG_ENV} unset)"
    try:
        have = cell_key.from_artifact_metadata(meta)
    except cell_key.CellKeyError as exc:
        return f"cell has no computable key ({exc})"
    if have.digest == want.digest:
        return ""  # exact key — nothing to bridge, no mark
    delta = _axes_delta(have.axes_dict(), want.axes_dict())
    undesignated = [d for d in delta if d[0] not in DESIGNATED_AXES]
    if undesignated:
        axis, cell_v, want_v = undesignated[0]
        return (
            f"axis {axis!r} is not equivalence-designated: cell {cell_v!r} "
            f"!= runtime {want_v!r}")
    manifest = meta.get(guard_closure.MANIFEST_KEY)
    if not isinstance(manifest, dict) or not manifest.get("graphs"):
        return "cell carries no guard manifest — equivalence needs the proof"
    if manifest.get("leaks"):
        return "cell manifest records leaks — never equivalence-adoptable"
    sealed = manifest.get(guard_closure.POSTURE_KEY)
    if not isinstance(sealed, dict) or not sealed:
        return "cell manifest carries no posture seal (pre-pgw#695 mint)"
    try:
        guard_closure.assert_posture(sealed, label="equivalence")
    except guard_closure.PostureError as exc:
        return str(exc)
    for block, live in (
        ("content_keys", dict(compile_cache.content_keys())),
        ("toolchain", dict(compile_cache.toolchain_digest())),
    ):
        recorded = meta.get(block)
        if not isinstance(recorded, dict) or not recorded:
            return f"cell records no {block} — equivalence precondition missing"
        for name in sorted(set(recorded) | set(live)):
            cell_v = str(recorded.get(name) or "")
            live_v = str(live.get(name) or "")
            if cell_v != live_v:
                return (
                    f"{block}/{name}: cell {cell_v or '<absent>'} != runtime "
                    f"{live_v or '<absent>'} — the compile stack is not "
                    "byte-identical")
    drift = compile_cache.artifact_drift(dict(meta), pipeline, cfg)
    if drift:
        return drift
    bridged = sorted(d[0] for d in delta)
    # --- FAST tier: byte-identical code closure (Paul's ruling) --------
    tier = ""
    recorded_closure = meta.get("code_closure")
    if isinstance(recorded_closure, dict) and recorded_closure:
        closure_diffs = closure_delta(recorded_closure)
        if not closure_diffs:
            tier = "fast/code-closure"
        else:
            logger.info(
                "equivalence: code closure differs (%s; +%d more) — "
                "slow tier", closure_diffs[0], len(closure_diffs) - 1)
    # --- SLOW tier: composition-fingerprint proof (fleet-scale this is
    # the hub's probe-mint compare-and-bless flow) ----------------------
    if not tier:
        cell_rows = meta.get("composition") or []
        if not cell_rows:
            return "cell records no composition fingerprint (pre-pgw#697 mint)"
        named = compile_cache._first_composition_difference(
            cell_rows, compile_cache.composition_fingerprint(pipeline, cfg))
        if named:
            return f"module composition: {named}"
        tier = "slow/manifest-fingerprint"
    meta[ADOPTION_MARK] = bridged
    logger.info(
        "equivalence adoption (%s): cell %s bridged axes %s to runtime "
        "key %s", tier, have.digest, bridged, want.digest)
    return ""


def select(candidates: Sequence[Mapping[str, Any]]) -> Tuple[int, str]:
    """(index, '') of the adoptable candidate, or (-1, reason).

    Nix unicity rule (pgw#712): two candidates for one relaxed identity
    whose guard manifests differ is a ``cell_equivalence_divergence`` —
    NEITHER is served; the divergence is the finding. Identical manifests
    pick the first candidate (the exchange orders newest-first, th#1229)."""
    digests: Dict[str, int] = {}
    for index, meta in enumerate(candidates):
        manifest = meta.get(guard_closure.MANIFEST_KEY)
        if not isinstance(manifest, dict) or not manifest:
            continue
        digests.setdefault(guard_closure.manifest_digest(manifest), index)
    if not digests:
        return -1, "no candidate carries a guard manifest"
    if len(digests) > 1:
        return -1, (
            "cell_equivalence_divergence: "
            f"{len(digests)} distinct manifests for one relaxed identity "
            f"({sorted(digests)!r}) — serving none until resolved")
    return next(iter(digests.values())), ""


__all__ = [
    "ADOPTION_MARK",
    "DESIGNATED_AXES",
    "FLAG_ENV",
    "closure_delta",
    "enabled",
    "select",
    "verdict",
]
