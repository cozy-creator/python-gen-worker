"""Boot-time LANE SELECTION — platform machinery, never endpoint code (pgw#1606).

The declared lanes are the CANDIDATE SET. This module ranks them, picks one,
and says why it picked that one **and why it rejected each of the others, in
order**. An endpoint contains no dtype branch, no quant if-tree and no
`get_device_capability()` read, because the answer is computed here.

Three rules the audit (pgw#1606, banked in the tracker) forced on the design:

1. **The ranking is not invented here.** `models/execution_lanes` already holds
   the fleet's ranked lane table — the byte-identical twin of tensorhub's
   `internal/orchestrator/precision/lane.go`. A second ranking would be a
   second opinion, and the two would drift.
2. **The kernel gate is not re-implemented here.** `w8a8_gemm_mode()` and
   `w4a4_gemm_mode()` are live micro-benchmarks (a 4096-cubed GEMM, median of
   ten, requiring 1.10x over bf16) cached once per process. They are a REAL
   veto and they cost real time. The ladder CONSUMES that verdict and pays it
   once. It never grows a rival opinion about whether fp8 is worth having.
3. **No artifact is never a refusal.** A lane whose bytes do not exist yet is a
   priced, producible CONVERSION ASK (tensorfs#128's producer), which is a
   different thing from "this card cannot serve this model".

Every collaborator this module needs is a PORT, injected. That is not
ceremony: it is what makes the whole ladder provable on a CPU box against
fabricated cards, which is where it gets tested.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

import msgspec

from ..models import execution_lanes as el
from .lane_spec import DeclaredLane as LaneDeclaredLane


# --------------------------------------------------------------------------
# The seam with pgw#1599
# --------------------------------------------------------------------------


#: pgw#1599's value object, READ at class-definition time and carried through
#: this module verbatim. Imported rather than restated: while this issue was
#: being built ahead of that surface it held a structural Protocol here, and a
#: Protocol that outlives the type it stood in for becomes a second opinion
#: about what a lane is. The ladder reads `contract_id`, `dtype` and `min_sm`
#: and re-derives none of them — in particular `min_sm` keeps its single
#: producer (`capability_floor_for_dtype`, applied at declaration), because one
#: hand-written floor could never be right for a bf16/fp8/nvfp4 class at once.
DeclaredLane = LaneDeclaredLane


# --------------------------------------------------------------------------
# Ports — everything the ladder asks the world about
# --------------------------------------------------------------------------


#: What a checkpoint's bytes are to a lane's contract. The tri-state of
#: tensorfs#123 (`Satisfies | DerivableVia | Incompatible`) plus the fourth
#: state a worker actually meets: nothing is staged for this contract at all.
VERDICT_SATISFIES = "satisfies"
VERDICT_DERIVABLE = "derivable"
VERDICT_INCOMPATIBLE = "incompatible"
VERDICT_ABSENT = "absent"

VERDICTS = (
    VERDICT_SATISFIES, VERDICT_DERIVABLE, VERDICT_INCOMPATIBLE, VERDICT_ABSENT,
)


class LaneVerdict(Protocol):
    """The derivability verdict, per lane contract.

    **THE HUB IS AUTHORITATIVE, AND THIS PORT IS NOT A SECOND OPINION.**

    pgw#1606's audit asked for a pyo3 binding so a pod could compute the
    tri-state (`Satisfies | DerivableVia | Incompatible`, tensorfs#123)
    locally. That was the wrong ask and the pgw#1599 lane was right to decline
    it: the verdict is an ADMIT DECISION, it already runs at BIND time in the
    hub's bind gate (`th internal/bindgate`) against `tensorfs/verdict.go`, and
    a worker-side implementation of an admit decision is a second
    implementation that can disagree with the gate that let the deployment
    exist. The tree already counts three copies of the pattern matcher
    (tensorfs#129); this is not the fourth.

    So the authoritative answer TRAVELS rather than being recomputed:
    `lane_host.BindingVerdicts` reads it off the `DeployBinding` the hub sent.
    No round-trip at boot, no rival matcher, one producer.

    The port stays because the ladder must be provable without a hub and
    without a card — every test in `test_lane_ladder_pgw1606` fabricates one.
    What it must never become is a place where pgw decides admissibility for
    itself: an implementation here answers `absent` for what it was not told,
    which degrades to a conversion ask, never to a silent admit.
    """

    def verdict(self, contract_id: str) -> str:
        """One of :data:`VERDICTS`."""
        ...

    def transfer_bytes(self, contract_id: str) -> int:
        """Bytes this contract's tree would cost to fetch; 0 = not known.

        Read for the upcast rung's saving, which pgw#1606 acceptance (c)
        requires be MEASURED rather than asserted.
        """
        ...


class KernelGates(Protocol):
    """The host's own veto on a quantized lane.

    Both arms answer `""` for "this host cannot profitably run this kernel",
    which is exactly what `models/w8a8.w8a8_gemm_mode` and
    `models/w4a4.w4a4_gemm_mode` already answer. The ladder treats `""` as a
    rejection with a reason, never as a crash and never as a silent fallback.
    """

    def w8a8_mode(self) -> str: ...
    def w4a4_mode(self) -> str: ...


class CardFacts(msgspec.Struct, frozen=True, kw_only=True):
    """The card, as a value. Fabricable, which is the point.

    `sm` is the integer form the fleet already uses everywhere
    (`major * 10 + minor`), so an RTX 4070 is 89 and a B200 is 100.
    """

    sm: int
    vram_gb: float = 0.0
    name: str = ""

    @property
    def label(self) -> str:
        return f"sm{self.sm}" + (f"/{self.name}" if self.name else "")


# --------------------------------------------------------------------------
# The closed reason vocabularies
# --------------------------------------------------------------------------

#: Why a declared lane was NOT chosen. Closed, for the same reason torchcg's
#: `KEPT_*` set is closed: "rejected, reason unknown" is the row that hides a
#: model serving the wrong numerics.
REJECT_SM_FLOOR = "sm_floor"
REJECT_KERNEL_UNQUALIFIED = "kernel_unqualified"
REJECT_NO_ARTIFACT = "no_artifact"
REJECT_INCOMPATIBLE = "incompatible"
REJECT_CONVERTIBLE = "convertible_not_staged"
REJECT_UNKNOWN_DTYPE = "unknown_dtype"
REJECT_OUTRANKED = "outranked"

REJECTIONS = (
    REJECT_SM_FLOOR, REJECT_KERNEL_UNQUALIFIED, REJECT_NO_ARTIFACT,
    REJECT_INCOMPATIBLE, REJECT_CONVERTIBLE, REJECT_UNKNOWN_DTYPE,
    REJECT_OUTRANKED,
)

#: Why the chosen lane won.
CHOSE_GATE_PASSED = "gate_passed"
CHOSE_BASELINE = "baseline"
CHOSE_UPCAST = "upcast_from_quantized"
CHOSE_PENDING_CONVERSION = "pending_conversion"

CHOICES = (
    CHOSE_GATE_PASSED, CHOSE_BASELINE, CHOSE_UPCAST, CHOSE_PENDING_CONVERSION,
)


class LaneLadderError(RuntimeError):
    """The ladder cannot be walked — a declaration defect, never a card fact."""


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


class RejectedRung(msgspec.Struct, frozen=True, kw_only=True):
    """One candidate the ladder passed over, and why.

    `detail` carries the NUMBERS. A rejection that says only "sm_floor" makes
    an operator go read code; one that says "needs sm100, card is sm89" ends
    the question.
    """

    body: str
    contract_id: str
    reason: str
    detail: str = ""

    def line(self) -> str:
        tail = f": {self.detail}" if self.detail else ""
        return f"{self.body}({self.reason}{tail})"


class ConversionAsk(msgspec.Struct, frozen=True, kw_only=True):
    """A priced, producible conversion — the answer when no lane has bytes.

    Never a refusal. `from_contract` is what IS staged; `to_contract` is the
    lane that would then serve. `recipe` is tensorfs' own token
    (`dtype-cast` / `fp8-rowwise`), carried rather than re-derived.
    """

    from_contract: str
    to_contract: str
    recipe: str = ""
    detail: str = ""


class ResolvedLane(msgspec.Struct, frozen=True, kw_only=True):
    """What the platform picked, why, and everything it passed over IN ORDER.

    `declared` is pgw#1599's object, carried VERBATIM — so `request=` and
    `resident=` travel intact to varena and `min_sm` keeps its single
    producer. Nothing here re-derives a field off the Contract.
    """

    declared: Any
    body: str
    reason: str
    rejected: tuple[RejectedRung, ...] = ()
    #: The contract whose BYTES are fetched, when that is not the serving
    #: lane's own — the upcast rung: fp8 on the wire, bf16 in the GEMM.
    fetch_contract: str = ""
    #: Bytes saved on the wire by fetching `fetch_contract` instead of the
    #: serving lane's own tree. 0 when nothing was measurable.
    transfer_saved_bytes: int = 0
    #: Set only when NO lane had bytes. The lane still resolves; serving waits
    #: on the job rather than the boot refusing.
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
        """THE line. Chosen lane, the reason, and the rejected rungs in order.

        Shaped after `models/rung.transition_line` — the placement ladder's
        format, which is the one an operator already knows how to read. One
        line, greppable, and it names what it did NOT do, because a ladder that
        only reports its winner cannot be audited.
        """
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
# dtype -> lane body. ONE producer.
# --------------------------------------------------------------------------

#: The serving body a contract dtype implies. Keyed on the safetensors/torch
#: spellings a tensorfs document actually carries.
#:
#: fp8 maps to the w8a8 GEMM body and NOT to `fp8-w8a16`: fp8 is the canonical
#: quantization (Paul), and w8a16 is fp8-STORAGE-with-bf16-compute, which is a
#: fit mechanism on the placement ladder rather than a declared serving lane.
#: The upcast rung below is how fp8 BYTES reach a bf16 GEMM, and it is a
#: property of the resolution, not a fourth body.
_DTYPE_BODY: dict[str, str] = {
    "bfloat16": el.WEIGHTS_BF16 + "-" + el.ACT_W16A16,
    "bf16": el.WEIGHTS_BF16 + "-" + el.ACT_W16A16,
    "float16": el.WEIGHTS_BF16 + "-" + el.ACT_W16A16,
    "fp16": el.WEIGHTS_BF16 + "-" + el.ACT_W16A16,
    "half": el.WEIGHTS_BF16 + "-" + el.ACT_W16A16,
    "float32": el.WEIGHTS_BF16 + "-" + el.ACT_W16A16,
    "fp32": el.WEIGHTS_BF16 + "-" + el.ACT_W16A16,
    "float8_e4m3fn": el.WEIGHTS_FP8 + "-" + el.ACT_W8A8 + "-" + el.SCALE_DYNAMIC,
    "float8_e4m3fnuz": el.WEIGHTS_FP8 + "-" + el.ACT_W8A8 + "-" + el.SCALE_DYNAMIC,
    "fp8_e4m3": el.WEIGHTS_FP8 + "-" + el.ACT_W8A8 + "-" + el.SCALE_DYNAMIC,
    "fp8": el.WEIGHTS_FP8 + "-" + el.ACT_W8A8 + "-" + el.SCALE_DYNAMIC,
    "float4_e2m1fn": el.WEIGHTS_NVFP4 + "-" + el.ACT_W4A4 + "-" + el.SCALE_STATIC,
    "nvfp4": el.WEIGHTS_NVFP4 + "-" + el.ACT_W4A4 + "-" + el.SCALE_STATIC,
    "fp4": el.WEIGHTS_NVFP4 + "-" + el.ACT_W4A4 + "-" + el.SCALE_STATIC,
}


def dtype_body(dtype: Any) -> str:
    """The ranked lane body a contract dtype serves as, or `""` if unknown.

    `""` is not an error here — it becomes a NAMED rejection
    (`unknown_dtype`) so a document declaring something the fleet has no
    executor for is passed over loudly rather than crashing a boot.
    """
    name = getattr(dtype, "name", None) or dtype
    key = str(name or "").strip().lower().removeprefix("torch.")
    return _DTYPE_BODY.get(key, "")


def _rank(body: str) -> int:
    """Position in the fleet's ranked lane table; unranked sorts last."""
    bodies = el.known_execution_lane_bodies()
    try:
        return bodies.index(body)
    except ValueError:
        return len(bodies)


def is_baseline(body: str) -> bool:
    return body.startswith(el.WEIGHTS_BF16 + "-")


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def _gate_for(body: str, gates: KernelGates) -> tuple[bool, str]:
    """Does the HOST admit this body's kernel? `(ok, detail)`.

    A baseline body has no kernel gate — bf16 matmul needs no benchmark.
    """
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
    # An executor the ladder does not know how to gate is not silently
    # admitted. svdq bodies land here today and that is correct: nothing
    # selects them at boot (native_kernels._decide is hard-wired off).
    return False, f"no kernel gate is wired for body {body!r}"


class _Candidate(msgspec.Struct, frozen=True, kw_only=True):
    declared: Any
    body: str
    rank: int


def _candidates(declared: Sequence[Any]) -> tuple[list[_Candidate], list[RejectedRung]]:
    """Rank the declared lanes; a dtype with no executor is rejected by name."""
    ranked: list[_Candidate] = []
    unknown: list[RejectedRung] = []
    for lane in declared:
        body = dtype_body(getattr(lane, "dtype", None))
        contract_id = str(getattr(lane, "contract_id", "") or "")
        if not body:
            unknown.append(RejectedRung(
                body="?", contract_id=contract_id, reason=REJECT_UNKNOWN_DTYPE,
                detail=f"dtype {getattr(lane, 'dtype', None)!r} maps to no "
                       f"serving body in the fleet's lane table",
            ))
            continue
        ranked.append(_Candidate(declared=lane, body=body, rank=_rank(body)))
    # Author ORDER carries no priority (pgw#1599) — the table does. Ties keep
    # declaration order, which is stable and therefore reproducible.
    ranked.sort(key=lambda c: c.rank)
    return ranked, unknown


def resolve_lane(
    *,
    declared: Sequence[Any],
    card: CardFacts,
    verdicts: LaneVerdict,
    gates: KernelGates,
) -> ResolvedLane:
    """Pick ONE lane out of the declared candidate set, and say why.

    The walk, stated so it is testable:

    1. Rank the candidates by the fleet's own lane table (fp8 before nvfp4
       before bf16). Author order is not priority.
    2. For each in rank order: the card's capability floor (pgw#1599's DERIVED
       `min_sm`), then the host's kernel gate, then the bytes' verdict.
       The first candidate that clears all three WINS.
    3. If a baseline lane wins and some quantized lane's bytes ARE staged,
       fetch THOSE and upcast at load — half the transfer, full-precision
       serve. That is the upcast rung, and it is a CHOICE here rather than
       the failure fallback it is everywhere else in the tree today.
    4. If nothing has bytes, resolve anyway and carry a priced conversion ask.
       Never a refusal.

    Raises only for a declaration defect: an empty candidate set. A card that
    cannot serve anything is not a defect, it is a conversion.
    """
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

    # Which quantized trees are ON DISK, asked INDEPENDENTLY of whether their
    # rung is runnable. This is the upcast rung's whole premise and the
    # ordering bug the first version of this function had: on an Ampere card
    # the fp8 rung is floored out before its bytes are ever looked at, and
    # those bytes are exactly what the rung wants to fetch. "Can this card run
    # fp8" and "are the fp8 bytes staged" are two questions, and conflating
    # them silently deletes half the transfer saving.
    staged_quantized = [
        cand for cand in ranked
        if not is_baseline(cand.body)
        and verdicts.verdict(str(getattr(cand.declared, "contract_id", "") or ""))
        == VERDICT_SATISFIES
    ]

    # Pass one: the first candidate that clears floor, gate and bytes.
    derivable: list[_Candidate] = []
    #: Cleared the card's floor AND the host's kernel gate — i.e. lanes this
    #: machine could actually RUN if the bytes existed. Tracked separately
    #: from `derivable` because a conversion must target a runnable lane.
    runnable: list[_Candidate] = []
    winner: Optional[_Candidate] = None
    for cand in ranked:
        contract_id = str(getattr(cand.declared, "contract_id", "") or "")
        min_sm = int(getattr(cand.declared, "min_sm", 0) or 0)
        # THE BASELINE RUNG IS NEVER FLOORED OUT. `capability_floor_for_dtype`
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

    # Nothing had bytes. Resolve to the best rung that CLEARS THE CARD and ask
    # for the conversion that would fill it. Degrade, never refuse.
    # Convert TOWARDS a lane this card can run. `derivable` is already
    # floor-and-gate-cleared, so it is the first choice; `runnable` catches the
    # case where nothing was even derivable but some rung would work given
    # bytes. Falling straight to `ranked[0]` — as the first version did —
    # could ask for a conversion to a lane the card is floored out of, which
    # spends money to arrive back at the same refusal.
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
    """The contract whose bytes a conversion would start from, or `""`.

    A DERIVABLE verdict means tensorfs can name a conversion from what IS on
    disk; the first such contract is the honest source. `""` means even that
    is unknown, and the ask says so rather than inventing a provenance.
    """
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
    """Apply the upcast rung when a baseline lane won over staged fp8 bytes.

    pgw#1606: *"fetch the fp8 bytes, upcast at load: half the transfer, full
    precision serve."* The mechanism already exists four times over in this
    tree (`w8a8.py:680-696`, the `mode="dequant"` host lane,
    `sanitize_w8a8_state_dict`, `hf_fp8_blockwise`) — every one of them entered
    because a gate FAILED. This is the first place it is entered because a
    ladder CHOSE it, and the first place the saving is a number.
    """
    contract_id = str(getattr(winner.declared, "contract_id", "") or "")
    if not is_baseline(winner.body) or not staged_quantized:
        return ResolvedLane(
            declared=winner.declared, body=winner.body,
            reason=CHOSE_GATE_PASSED if not is_baseline(winner.body)
            else CHOSE_BASELINE,
            rejected=rejected, card=card,
        )
    # Cheapest staged quantized tree on the wire wins; ties keep table rank.
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
        # Not actually cheaper (or unmeasurable): serve the baseline tree.
        # A rung that cannot show its saving does not get to claim one.
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
    "REJECT_UNKNOWN_DTYPE",
    "RejectedRung",
    "ResolvedLane",
    "VERDICTS",
    "VERDICT_ABSENT",
    "VERDICT_DERIVABLE",
    "VERDICT_INCOMPATIBLE",
    "VERDICT_SATISFIES",
    "dtype_body",
    "is_baseline",
    "resolve_lane",
]
