"""Execution lanes: a lane IS a tensor-layout STAMP (tcg#41; re-keyed tcg#79).

Paul's ruling from the main_v2.py line review (2026-08-18): the named-Lane
class was under-determined -- with arbitrary names and a free-form contract
field, two lanes were functionally indistinguishable. A lane is therefore a
reference to the LAYOUT its checkpoints are in, and nothing else.

Under the tensor-layout contract **v2** (tensorfs#151) a layout is COMPUTED,
never stored: ``LAYOUT = quant(topology)``. So the reference is a PAIR of
document handles rather than one, and its wire rendering is::

    <topology>@<version>+<quant>@<version>
    sdxl.diffusers@1+plain.bf16@1

The author names contract OBJECTS (``lanes=(contracts.SDXL_DIFFUSERS_BF16,)``);
that rendering is what resolution spells them as, and it is the row key of
every document torchcg writes. The v1 spelling ``sdxl.diffusers-bf16@1`` --
ONE handle naming an already-crossed (topology, quant) product -- is RETIRED,
not aliased. It survives only as a display name, is never parsed, and is
refused here BY NAME so a stale producer cannot quietly key a document to a
stamp nothing publishes.

Neither half of the grammar is torchcg's to choose:

* the JOIN is ``+`` because tensorfs renders it there (``LayoutID.String``,
  ``v2doc.go``), the tensorfs SDK's ``render()`` renders it there, and the
  derived-artifact CAS address is built from it. A spelling drift here is a
  CAS-address fork.
* one HALF is ``<producer>.<format>@<version>``, mirroring tensorfs'
  ``ParseHandle`` + ``isTFM1ContractName`` character for character -- notably
  a producer that ADMITS a hyphen but never leads with one, because half the
  v2 topology corpus needs it (``flux2-klein.diffusers@1``,
  ``hidream-o1.diffusers@1``, ``minimax-h3.diffusers@1``, ``z-image.diffusers@1``,
  ``ltx2-upsampler.diffusers@1``, ``sdxl-inpainting.diffusers@1``,
  ``qwen3-6-35b-a3b.transformers@1``). A grammar written by eye refuses every
  one of those.

``compile`` marking is imperative and target-shaped (``ctx.compile(module)``);
everything else about a lane -- its load dtype, its layout document -- comes
from the contract objects, never typed by the author. ``LaneRef`` is that
RESOLVED form: the stamp plus the resolution's dtype, which serve code reads
back as ``ctx.lane.dtype``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_PATH_SEPARATOR = "."

#: The join between a stamp's two halves. tensorfs ``LayoutID.String`` and the
#: derived-artifact CAS address both spell it; it is not negotiable.
LANE_JOIN = "+"

#: One document handle, ``<producer>.<format>@<version>``. The name grammar is
#: tensorfs' ``isTFM1ContractName``: a producer of lowercase alphanumerics plus
#: ``-`` (never leading), then ONE ``.``, then a format of lowercase
#: alphanumerics plus ``.``, ``-`` and ``_``. The producer segment cannot hold a
#: ``.``, so this regex cuts at the FIRST dot exactly as ``strings.Cut`` does.
_PRODUCER = r"[a-z0-9][a-z0-9-]*"
_FORMAT = r"[a-z0-9][a-z0-9._\-]*"
_HANDLE = rf"{_PRODUCER}\.{_FORMAT}@[1-9][0-9]*"
_HANDLE_RE = re.compile(rf"{_HANDLE}\Z")

#: ``<topology>+<quant>`` -- the whole lane.
_LANE_RE = re.compile(rf"(?P<topology>{_HANDLE})\{LANE_JOIN}(?P<quant>{_HANDLE})\Z")

#: tensorfs ``TFM1MaxContractNameBytes``: a manifest has to be able to carry
#: the name, so the bound is the manifest's, not a taste.
_MAX_NAME_BYTES = 64

#: tensorfs ``ParseHandle`` reads the version with ``ParseUint(_, 10, 32)``.
#: Version ``0`` is refused there and here. LEADING ZEROS are refused here and
#: not there: Go would read ``@01`` as 1 and render it back as ``@1``, which is
#: a second spelling of one stamp -- and a second spelling is the whole hazard.
_MAX_VERSION = 0xFFFFFFFF

#: ``<name>@<major>`` -- one transform pass. Versioned because a pass NAME is
#: a ``cg-graph-v1`` derivation input (tcg#52): two lanes differing only by a
#: pass are different graphs, exactly as two lanes differing by a pin are.
_PASS_RE = re.compile(r"[a-z0-9][a-z0-9-]*[a-z0-9]@[1-9][0-9]*\Z")

_LANE_GRAMMAR = (
    "a lane is a tensor-layout STAMP -- the (topology, quant) PAIR, spelled "
    "'<topology>@<v>+<quant>@<v>' (e.g. 'sdxl.diffusers@1+plain.bf16@1')"
)


class LaneError(ValueError):
    """An execution-lane declaration is incomplete or noncanonical."""


def _refuse_bare_handle(value: str) -> None:
    """Refuse a lone handle BY NAME, never by coercion.

    ``sdxl.diffusers-bf16@1`` (the RETIRED v1 spelling, one handle naming an
    already-crossed product), ``sdxl.diffusers@1`` (a topology alone) and
    ``plain.bf16@1`` (a rule alone) are all well-formed handles and none of
    them is a lane. Left to the generic refusal they read as typos; named, they
    read as the migration they are. Coercion is not on the table either way --
    nothing recovers ``(sdxl.diffusers@1, plain.bf16@1)`` from those bytes, and
    inventing a default for the missing axis would silently key a document to a
    stamp the producer never meant.
    """

    if _HANDLE_RE.fullmatch(value) is None:
        return
    raise LaneError(
        f"lane {value!r} is ONE handle, and a lane is a PAIR. Under the v2 layout "
        f"contract a layout is COMPUTED -- quant(topology) -- so {_LANE_GRAMMAR}. "
        f"A lone handle is either half a stamp or the RETIRED v1 spelling "
        f"('sdxl.diffusers-bf16@1'), which survives as a DISPLAY NAME only and is "
        f"never parsed. Neither is coerced: there is no rule that recovers the "
        f"missing axis"
    )


def parse_lane_id(value: object) -> tuple[str, str]:
    """Split one lane spelling into its ``(topology, quant)`` handles."""

    if not isinstance(value, str):
        raise LaneError(f"{_LANE_GRAMMAR}; got {value!r}")
    _refuse_bare_handle(value)
    match = _LANE_RE.fullmatch(value)
    if match is None:
        raise LaneError(f"{_LANE_GRAMMAR}; got {value!r}")
    topology, quant = match.group("topology"), match.group("quant")
    for axis, handle in (("topology", topology), ("quant", quant)):
        name, _, digits = handle.rpartition("@")
        if len(name.encode("ascii")) > _MAX_NAME_BYTES:
            raise LaneError(
                f"lane {value!r}: the {axis} name {name!r} is "
                f"{len(name.encode('ascii'))} bytes; a contract name is at most "
                f"{_MAX_NAME_BYTES} (tensorfs TFM1MaxContractNameBytes)"
            )
        if int(digits) > _MAX_VERSION:
            raise LaneError(
                f"lane {value!r}: the {axis} version {digits} does not fit the "
                f"uint32 tensorfs parses it as"
            )
    return topology, quant


def require_lane_id(value: object) -> str:
    """Validate one lane spelling: the (topology, quant) stamp, nothing else.

    Returns the stamp RE-RENDERED from its parsed halves, which is the same
    bytes: the grammar admits exactly one spelling per stamp, so this can only
    ever return its input. Rendering rather than echoing is what keeps that
    true if the grammar is ever loosened.
    """

    topology, quant = parse_lane_id(value)
    return f"{topology}{LANE_JOIN}{quant}"


def require_pass_ref(value: object) -> str:
    """Validate one transform-pass spelling: ``<name>@<major>``, nothing else."""

    if not isinstance(value, str) or _PASS_RE.fullmatch(value) is None:
        raise LaneError(
            f"a pass name is '<name>@<major>' (e.g. 'precompute-and-free@1'); "
            f"got {value!r}"
        )
    return value


def require_passes(names: object) -> tuple[str, ...]:
    """Validate a lane's pass tuple: canonical, ordered as given, no repeats."""

    if names is None:
        return ()
    if not isinstance(names, (tuple, list)):
        raise LaneError("a lane's passes must be a tuple of pass names")
    passes = tuple(require_pass_ref(name) for name in names)
    if len(set(passes)) != len(passes):
        raise LaneError(f"a lane names a pass twice: {passes!r}")
    return passes


def require_targets(paths: object) -> tuple[str, ...]:
    """Validate the endpoint-level compile-target paths."""

    if not isinstance(paths, tuple) or not paths:
        raise LaneError(
            "compile= must be a non-empty tuple of attribute paths on the "
            "author's own objects, e.g. ('unet',) or ('unet', 'vae.decoder')"
        )
    for path in paths:
        if not isinstance(path, str) or not path or path != path.strip():
            raise LaneError("compile-target paths must be canonical strings")
        segments = path.split(_PATH_SEPARATOR)
        if any(not segment.isidentifier() for segment in segments):
            raise LaneError(
                f"compile target {path!r} must be a dotted attribute path of "
                f"identifiers, e.g. 'unet' or 'vae.decoder'"
            )
    if len(set(paths)) != len(paths):
        raise LaneError(f"compile targets repeat a path: {paths!r}")
    return paths


@dataclass(frozen=True, slots=True)
class LaneRef:
    """One RESOLVED lane: the layout stamp + what resolution derived.

    ``contract`` is the stamp -- ``<topology>@<v>+<quant>@<v>``. The FIELD
    keeps its name because a lane still references contracts; there are simply
    two of them now, and the name is a wire key the hub decodes by
    (``json:"contract"``, ``releasegraphs/derive_document.go``).

    The author names contract objects; ``dtype`` is filled by the resolver
    from the topology/quant pair those objects resolve to. Serve code reads it
    back as ``ctx.lane.dtype``.

    ``passes`` is the lane's TRANSFORM binding (tcg#52): the ordered pass
    names that MAKE this lane's format. It is resolved from the registry
    entry exactly like ``dtype`` -- an fp8 lane's weights are fp8 because
    something converted them, and a precompute lane's blocks hold a side
    table because something folded them. Empty means no pass, stated.
    """

    contract: str
    dtype: Any = None
    passes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_lane_id(self.contract)
        object.__setattr__(self, "passes", require_passes(self.passes))


def resolve_target(roots: Mapping[str, object], path: str) -> object:
    """Resolve one compile-target path against the author's root namespace.

    The first segment names a root object; the rest are attribute accesses on
    the author's own graph of objects. A miss is refused with the exact
    segment that failed, because a silently skipped target would turn a typo
    into an eager-forever lane.
    """

    if not isinstance(path, str) or not path:
        raise LaneError("compile-target path must be a non-empty string")
    segments = path.split(_PATH_SEPARATOR)
    root_name = segments[0]
    if root_name not in roots:
        raise LaneError(
            f"compile target {path!r} names root {root_name!r}, which is not in "
            f"the author namespace {sorted(roots)!r}"
        )
    value: object = roots[root_name]
    for depth, segment in enumerate(segments[1:], start=1):
        try:
            value = getattr(value, segment)
        except AttributeError as exc:
            raise LaneError(
                f"compile target {path!r}: {segment!r} does not exist on the "
                f"object at {'.'.join(segments[:depth])!r}"
            ) from exc
    return value


__all__ = [
    "LANE_JOIN",
    "LaneError",
    "LaneRef",
    "parse_lane_id",
    "require_lane_id",
    "require_pass_ref",
    "require_passes",
    "require_targets",
    "resolve_target",
]
