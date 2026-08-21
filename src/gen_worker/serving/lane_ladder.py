from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

import msgspec

from ..models import execution_lanes as el
from .lane_spec import DeclaredLane as LaneDeclaredLane


#: pgw#1599's value object, READ at class-definition time and carried through
#: this module verbatim. Imported rather than restated: while this issue was
#: being built ahead of that surface it held a structural Protocol here, and a
#: Protocol that outlives the type it stood in for becomes a second opinion
#: about what a lane is. The ladder reads `contract_id`, `dtype` and `min_sm`
#: and re-derives none of them — in particular `min_sm` keeps its single
#: producer (`capability_floor_for_rule`, applied at declaration), because one
#: hand-written floor could never be right for a bf16/fp8/nvfp4 class at once.
DeclaredLane = LaneDeclaredLane


VERDICT_SATISFIES = "satisfies"
VERDICT_DERIVABLE = "derivable"
VERDICT_INCOMPATIBLE = "incompatible"
VERDICT_ABSENT = "absent"

VERDICTS = (
    VERDICT_SATISFIES, VERDICT_DERIVABLE, VERDICT_INCOMPATIBLE, VERDICT_ABSENT,
)


class LaneVerdict(Protocol):
    """The derivability verdict, per lane contract."""

    def verdict(self, contract_id: str) -> str:
        """One of :data:`VERDICTS`."""
        ...

    def transfer_bytes(self, contract_id: str) -> int:
        """Bytes this contract's tree would cost to fetch; 0 = not known."""
        ...


class KernelGates(Protocol):
    """The host's own veto on a quantized lane."""

    def w8a8_mode(self) -> str: ...
    def w4a4_mode(self) -> str: ...


class CardFacts(msgspec.Struct, frozen=True, kw_only=True):
    """The card, as a value."""

    sm: int
    vram_gb: float = 0.0
    name: str = ""

    @property
    def label(self) -> str:
        return f"sm{self.sm}" + (f"/{self.name}" if self.name else "")


REJECT_SM_FLOOR = "sm_floor"
REJECT_KERNEL_UNQUALIFIED = "kernel_unqualified"
REJECT_NO_ARTIFACT = "no_artifact"
REJECT_INCOMPATIBLE = "incompatible"
REJECT_CONVERTIBLE = "convertible_not_staged"
REJECT_UNKNOWN_RULE = "unknown_quant_rule"
REJECT_OUTRANKED = "outranked"

REJECTIONS = (
    REJECT_SM_FLOOR, REJECT_KERNEL_UNQUALIFIED, REJECT_NO_ARTIFACT,
    REJECT_INCOMPATIBLE, REJECT_CONVERTIBLE, REJECT_UNKNOWN_RULE,
    REJECT_OUTRANKED,
)

CHOSE_GATE_PASSED = "gate_passed"
CHOSE_BASELINE = "baseline"
CHOSE_UPCAST = "upcast_from_quantized"
CHOSE_PENDING_CONVERSION = "pending_conversion"

CHOICES = (
    CHOSE_GATE_PASSED, CHOSE_BASELINE, CHOSE_UPCAST, CHOSE_PENDING_CONVERSION,
)


class LaneLadderError(RuntimeError):
    """The ladder cannot be walked — a declaration defect, never a card fact."""


class RejectedRung(msgspec.Struct, frozen=True, kw_only=True):
    """One candidate the ladder passed over, and why."""

    body: str
    contract_id: str
    reason: str
    detail: str = ""

    def line(self) -> str:
        tail = f": {self.detail}" if self.detail else ""
        return f"{self.body}({self.reason}{tail})"


class ConversionAsk(msgspec.Struct, frozen=True, kw_only=True):
    """A priced, producible conversion — the answer when no lane has bytes."""

    from_contract: str
    to_contract: str
    recipe: str = ""
    detail: str = ""


class ResolvedLane(msgspec.Struct, frozen=True, kw_only=True):
    """What the platform picked, why, and everything it passed over IN ORDER."""

    declared: Any
    body: str
    reason: str
    rejected: tuple[RejectedRung, ...] = ()
    fetch_contract: str = ""
    transfer_saved_bytes: int = 0
    conversion: Optional[ConversionAsk] = None
    card: Optional[CardFacts] = None

    @property
    def contract_id(self) -> str:
        return str(getattr(self.declared, "contract_id", ""))

    @property
    def upcast(self) -> bool:
        """Serving a wider dtype than the bytes on the wire were stored at."""
        return bool(self.fetch_contract) and self.fetch_contract != self.contract_id

    def confession(self) -> str:
        """THE line."""
        card = self.card.label if self.card is not None else "sm?"
        parts = [
            f"LANE={self.body}",
            f"contract={self.contract_id or '?'}",
            f"card={card}",
            f"reason={self.reason}",
        ]
        if self.upcast:
            saved = self.transfer_saved_bytes
            parts.append(
                f"fetch={self.fetch_contract}"
                + (f" saved={saved / 1e9:.2f}GB" if saved > 0 else " saved=unmeasured")
            )
        if self.conversion is not None:
            parts.append(
                f"conversion={self.conversion.from_contract}"
                f"->{self.conversion.to_contract}"
                + (f"[{self.conversion.recipe}]" if self.conversion.recipe else "")
            )
        rejected = ",".join(rung.line() for rung in self.rejected) or "none"
        parts.append(f"rejected={rejected}")
        return " ".join(parts)


# --------------------------------------------------------------------------
# quant RULE -> lane body. ONE producer.
# --------------------------------------------------------------------------

#: The serving body a lane's QUANT RULE implies. Keyed on the ratified rule
#: handle, one row per rule in the vendored v2 corpus, and there are eight.
#:
#: ⚠️ THIS WAS KEYED ON THE DTYPE SPELLING AND THAT WAS A REAL DEFECT, found by
#: the pgw#1621 re-key rather than by a failure in production. `cozy.fp8-
#: storage@1` and `cozy.fp8-rowwise@1` BOTH declare `float8_e4m3fn`, and they
#: execute in DIFFERENT LANES:
#:
#:   * `cozy.fp8-rowwise@1` stores an F32 `[out]` `weight_scale` beside each
#:     weight and is consumed by the w8a8 GEMM — `fp8-w8a8-dynamic`.
#:   * `cozy.fp8-storage@1` is SCALE-FREE (`"scale": "none"`) and its own
#:     conventions say `"consumption": "diffusers layerwise cast to bf16"` —
#:     fp8 bytes resident, **bf16 compute**. Its body is `bf16-w16a16`.
#:
#: A dtype-keyed table cannot tell those apart, so it answered `fp8-w8a8-
#: dynamic` for both — which would have offered a scale-free tree to a GEMM
#: that multiplies by scales that do not exist, and floored the lane at sm89
#: for arithmetic it never performs. The dtype names the ELEMENT; the rule
#: names the EXECUTOR, and only one of those is what a lane body is.
#:
#: fp8-rowwise maps to the w8a8 GEMM body and NOT to `fp8-w8a16`: fp8 is the
#: canonical quantization (Paul), and w8a16 is fp8-storage-with-bf16-compute —
#: which is exactly what `cozy.fp8-storage@1` IS, and it is expressed as the
#: bf16 body because that is the arithmetic that runs. The upcast rung below is
#: how fp8 GEMM bytes reach a bf16 GEMM, and it is a property of the
#: resolution, not a fourth body.
_RULE_BODY: dict[str, str] = {
    "plain.f32@1": el.WEIGHTS_BF16 + "-" + el.ACT_W16A16,
    "plain.f16@1": el.WEIGHTS_BF16 + "-" + el.ACT_W16A16,
    "plain.bf16@1": el.WEIGHTS_BF16 + "-" + el.ACT_W16A16,
    "cozy.fp8-storage@1": el.WEIGHTS_BF16 + "-" + el.ACT_W16A16,
    "cozy.fp8-rowwise@1": el.WEIGHTS_FP8 + "-" + el.ACT_W8A8 + "-" + el.SCALE_DYNAMIC,
    "hf.fp8-blockwise@1": el.WEIGHTS_FP8 + "-" + el.ACT_W8A8 + "-" + el.SCALE_DYNAMIC,
    "cozy.nvfp4-flat@1": el.WEIGHTS_NVFP4 + "-" + el.ACT_W4A4 + "-" + el.SCALE_STATIC,
    "bfl.nvfp4-preswizzled@1": el.WEIGHTS_NVFP4 + "-" + el.ACT_W4A4 + "-" + el.SCALE_STATIC,
}


def rule_body(quant: Any) -> str:
    """The ranked lane body a quant rule serves as, or `""` if this table has
    no row for it.

    `""` is not an error here — it becomes a NAMED rejection
    (`unknown_quant_rule`) so a rule the fleet has no executor for is passed
    over loudly rather than crashing a boot. Unlike the dtype table this
    replaced, a missing row is a REAL gap rather than a spelling accident:
    there are eight rules, they are enumerated above, and
    `test_every_ratified_rule_has_a_serving_body` fails the moment tensorfs
    ratifies a ninth.
    """
    return _RULE_BODY.get(str(quant or "").strip(), "")


def _rank(body: str) -> int:
    bodies = el.known_execution_lane_bodies()
    try:
        return bodies.index(body)
    except ValueError:
        return len(bodies)


def is_baseline(body: str) -> bool:
    return body.startswith(el.WEIGHTS_BF16 + "-")


def _gate_for(body: str, gates: KernelGates) -> tuple[bool, str]:
    if is_baseline(body):
        return True, ""
    if body.startswith(el.WEIGHTS_FP8 + "-" + el.ACT_W8A8):
        mode = gates.w8a8_mode()
        return bool(mode), (f"gemm_mode={mode}" if mode else
                            "w8a8_gemm_mode()='' — no fp8 GEMM qualified on "
                            "this host (call ok + >=1.10x over bf16)")
    if body.startswith(el.WEIGHTS_NVFP4):
        mode = gates.w4a4_mode()
        return bool(mode), (f"gemm_mode={mode}" if mode else
                            "w4a4_gemm_mode()='' — no fp4 GEMM qualified on "
                            "this host (numerics + >=1.10x over bf16)")
    return False, f"no kernel gate is wired for body {body!r}"


class _Candidate(msgspec.Struct, frozen=True, kw_only=True):
    declared: Any
    body: str
    rank: int


def _candidates(declared: Sequence[Any]) -> tuple[list[_Candidate], list[RejectedRung]]:
    ranked: list[_Candidate] = []
    unknown: list[RejectedRung] = []
    for lane in declared:
        body = rule_body(getattr(lane, "quant", None))
        contract_id = str(getattr(lane, "contract_id", "") or "")
        if not body:
            unknown.append(RejectedRung(
                body="?", contract_id=contract_id, reason=REJECT_UNKNOWN_RULE,
                detail=f"quant rule {getattr(lane, 'quant', None)!r} maps "
                       f"to no serving body in the fleet's lane table",
            ))
            continue
        ranked.append(_Candidate(declared=lane, body=body, rank=_rank(body)))
    ranked.sort(key=lambda c: c.rank)
    return ranked, unknown


def resolve_lane(
    *,
    declared: Sequence[Any],
    card: CardFacts,
    verdicts: LaneVerdict,
    gates: KernelGates,
) -> ResolvedLane:
    """Pick ONE lane out of the declared candidate set, and say why."""
    if not declared:
        raise LaneLadderError(
            "lane resolution: the candidate set is EMPTY. A Model subclass "
            "declares its lanes (pgw#1599) and the platform picks among them; "
            "with none declared there is nothing to pick and no default to "
            "invent"
        )

    ranked, rejected = _candidates(declared)
    if not ranked:
        raise LaneLadderError(
            "lane resolution: every declared lane names a dtype the fleet has "
            "no executor for — "
            + ", ".join(r.line() for r in rejected)
        )

    staged_quantized = [
        cand for cand in ranked
        if not is_baseline(cand.body)
        and verdicts.verdict(str(getattr(cand.declared, "contract_id", "") or ""))
        == VERDICT_SATISFIES
    ]

    derivable: list[_Candidate] = []
    runnable: list[_Candidate] = []
    winner: Optional[_Candidate] = None
    for cand in ranked:
        contract_id = str(getattr(cand.declared, "contract_id", "") or "")
        min_sm = int(getattr(cand.declared, "min_sm", 0) or 0)
        # THE BASELINE RUNG IS NEVER FLOORED OUT. `capability_floor_for_rule`
        # answers 80 for bf16, which is a statement about tensor cores and not
        # about whether the model runs — bf16 runs on Ampere, on Pascal and on
        # a CPU, at some speed. Flooring it out would leave a host with no CUDA
        # holding an empty ladder, and the ladder would then ask for a
        # CONVERSION, which is a confident wrong answer: no conversion has ever
        # fixed a missing GPU. This issue's ladder says "else bf16", and "else"
        # has to mean it.
        if not is_baseline(cand.body) and card.sm < min_sm:
            rejected.append(RejectedRung(
                body=cand.body, contract_id=contract_id, reason=REJECT_SM_FLOOR,
                detail=f"needs sm{min_sm}, card is sm{card.sm}",
            ))
            continue
        ok, detail = _gate_for(cand.body, gates)
        if not ok:
            rejected.append(RejectedRung(
                body=cand.body, contract_id=contract_id,
                reason=REJECT_KERNEL_UNQUALIFIED, detail=detail,
            ))
            continue
        runnable.append(cand)
        verdict = verdicts.verdict(contract_id)
        if verdict == VERDICT_SATISFIES:
            if winner is None:
                winner = cand
                continue
            rejected.append(RejectedRung(
                body=cand.body, contract_id=contract_id, reason=REJECT_OUTRANKED,
                detail=f"{winner.body} ranks higher and also has bytes",
            ))
            continue
        if verdict == VERDICT_DERIVABLE:
            derivable.append(cand)
            rejected.append(RejectedRung(
                body=cand.body, contract_id=contract_id, reason=REJECT_CONVERTIBLE,
                detail="the bytes are derivable but no tree is staged for this "
                       "contract yet",
            ))
            continue
        if verdict == VERDICT_INCOMPATIBLE:
            rejected.append(RejectedRung(
                body=cand.body, contract_id=contract_id, reason=REJECT_INCOMPATIBLE,
                detail="this checkpoint cannot satisfy the lane at any price",
            ))
            continue
        rejected.append(RejectedRung(
            body=cand.body, contract_id=contract_id, reason=REJECT_NO_ARTIFACT,
            detail="nothing is staged for this contract",
        ))

    if winner is not None:
        return _with_upcast(
            winner=winner, rejected=tuple(rejected), card=card,
            staged_quantized=staged_quantized, verdicts=verdicts,
        )

    target = (derivable[0] if derivable
              else runnable[0] if runnable
              else ranked[0])
    source = _conversion_source(ranked, verdicts)
    ask = ConversionAsk(
        from_contract=source,
        to_contract=str(getattr(target.declared, "contract_id", "") or ""),
        recipe="",
        detail="no declared lane has a staged tree on this card; the lane "
               "resolves and serving waits on a priced conversion rather than "
               "the boot refusing",
    )
    return ResolvedLane(
        declared=target.declared, body=target.body,
        reason=CHOSE_PENDING_CONVERSION, rejected=tuple(rejected),
        conversion=ask, card=card,
    )


def _conversion_source(ranked: Sequence[_Candidate], verdicts: LaneVerdict) -> str:
    for cand in ranked:
        contract_id = str(getattr(cand.declared, "contract_id", "") or "")
        if verdicts.verdict(contract_id) == VERDICT_DERIVABLE:
            return contract_id
    return ""


def _with_upcast(
    *,
    winner: _Candidate,
    rejected: tuple[RejectedRung, ...],
    card: CardFacts,
    staged_quantized: Sequence[_Candidate],
    verdicts: LaneVerdict,
) -> ResolvedLane:
    contract_id = str(getattr(winner.declared, "contract_id", "") or "")
    if not is_baseline(winner.body) or not staged_quantized:
        return ResolvedLane(
            declared=winner.declared, body=winner.body,
            reason=CHOSE_GATE_PASSED if not is_baseline(winner.body)
            else CHOSE_BASELINE,
            rejected=rejected, card=card,
        )
    best = min(
        staged_quantized,
        key=lambda c: (
            verdicts.transfer_bytes(str(getattr(c.declared, "contract_id", "")))
            or 1 << 62,
            c.rank,
        ),
    )
    fetch_id = str(getattr(best.declared, "contract_id", "") or "")
    own = verdicts.transfer_bytes(contract_id)
    theirs = verdicts.transfer_bytes(fetch_id)
    saved = own - theirs if own > 0 and theirs > 0 and own > theirs else 0
    if theirs <= 0 or (own > 0 and theirs >= own):
        return ResolvedLane(
            declared=winner.declared, body=winner.body, reason=CHOSE_BASELINE,
            rejected=rejected, card=card,
        )
    return ResolvedLane(
        declared=winner.declared, body=winner.body, reason=CHOSE_UPCAST,
        rejected=rejected, card=card,
        fetch_contract=fetch_id, transfer_saved_bytes=saved,
    )


__all__ = [
    "CHOICES",
    "CHOSE_BASELINE",
    "CHOSE_GATE_PASSED",
    "CHOSE_PENDING_CONVERSION",
    "CHOSE_UPCAST",
    "CardFacts",
    "ConversionAsk",
    "DeclaredLane",
    "KernelGates",
    "LaneLadderError",
    "LaneVerdict",
    "REJECTIONS",
    "REJECT_CONVERTIBLE",
    "REJECT_INCOMPATIBLE",
    "REJECT_KERNEL_UNQUALIFIED",
    "REJECT_NO_ARTIFACT",
    "REJECT_OUTRANKED",
    "REJECT_SM_FLOOR",
    "REJECT_UNKNOWN_RULE",
    "RejectedRung",
    "ResolvedLane",
    "VERDICTS",
    "VERDICT_ABSENT",
    "VERDICT_DERIVABLE",
    "VERDICT_INCOMPATIBLE",
    "VERDICT_SATISFIES",
    "rule_body",
    "is_baseline",
    "resolve_lane",
]
