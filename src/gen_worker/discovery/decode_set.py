"""The DECLARED DECODE-SET — which QUANT RULES this image's code can decode,
derived at IMAGE BUILD from the decoders that survived the import.

th#1938's `resolve()` intersects three things: the endpoint's declared demand,
the worker's decode-set, and card executability. This module is the second
one's ground truth. It exists because different quant recipes are not one
"quantized" concept — `cozy.fp8-rowwise@1`, `hf.fp8-blockwise@1` and
`cozy.nvfp4-flat@1` are distinct byte formats needing distinct decode paths,
and no config may select bytes the code cannot read.

Three properties, each of them a thing that could otherwise be faked:

1. **Derived, never written.** The census imports every decoder module and
   harvests the `@implements_quant_rule` markers that survived. A module that
   raises contributes no declaration and one excluded-with-reason record.
2. **A property of the BUILT IMAGE.** `python -m gen_worker.discovery` runs in
   the image build and stamps the block into `endpoint.lock`; the digest
   below is what makes "the set the hub read" and "the set this process would
   derive" comparable rather than merely plausible.
3. **Complete.** Decode paths whose bytes no ratified rule covers are recorded
   as such instead of living in a source comment nothing can read.

**THE INTERSECTION IS ON THE RULE HANDLE ALONE (pgw#1621).** An entry used to
carry five DECODE DIMENSIONS beside its handle — elements, scales, key
topologies, file layouts, bakes — because a v1 handle named a byte FORMAT and
said nothing about which of that format's legal shapes a decoder read, so a
declaration that named a handle and stopped was incomplete. The axes did not
become unnecessary; they became part of the RULE'S IDENTITY. `cozy.nvfp4-flat@1`
and `bfl.nvfp4-preswizzled@1` are two rules and two digests precisely because
one is LOW-nibble with flat scales and the other HIGH-nibble pre-swizzled, and
under v1 that difference could only be spelled on a side axis beside a shared
handle. Naming the rule is now the whole declaration, and intersecting on the
handle intersects on every convention it carries.

Two things that cost, stated rather than papered over:

* a decoder can no longer say it reads a VARIANT of a rule — the w8a8 loader
  decodes the optional `input_scale` leaf, which `cozy.fp8-rowwise@1` excludes
  by convention ("dynamic — no stored input_scale"). That variant is readable
  here and nameable nowhere until its own rule document is ratified.
* the per-decoder FILE-LAYOUT and KEY-TOPOLOGY intersections are gone with the
  axes. The key question survives as a fail-closed CLASSIFICATION below (a
  denoiser addressed in a way nothing here recognizes still refuses); the
  file-layout question has no successor in this image at all.
"""

from __future__ import annotations

import hashlib
import importlib
import pkgutil
from typing import Any, Dict, Optional

import msgspec

from gen_worker.models.key_topology import SnapshotKeys
from gen_worker.models.tensor_layout_contract import (
    QuantRuleDecoder,
    UnregisteredDecodePath,
    quant_rule_decoders_of,
    unregistered_decode_path_of,
)

DERIVATION = "gen_worker.discovery.decode_set@1"

DEFAULT_DECODER_PACKAGES: tuple[str, ...] = ("gen_worker.models",)

REFUSAL_RULE_UNDECLARED = "decode_set_rule_undeclared"
REFUSAL_KEY_TOPOLOGY_UNCLASSIFIED = "decode_set_key_topology_unclassified"
REFUSAL_DECODE_SET_DRIFT = "decode_set_drift"


class ExcludedDecoderModule(msgspec.Struct, frozen=True, kw_only=True):
    module: str
    reason: str


class DecodeEntry(msgspec.Struct, frozen=True, kw_only=True):
    """One (quant rule, decoder) the image ships."""

    rule: str
    decoder: str
    serves: tuple[str, ...]
    composes_lora: bool


class DecodeSet(msgspec.Struct, frozen=True, kw_only=True):
    derivation: str
    entries: tuple[DecodeEntry, ...]
    unregistered: tuple[UnregisteredDecodePath, ...]
    excluded_modules: tuple[ExcludedDecoderModule, ...]
    digest: str = ""

    def rules(self) -> tuple[str, ...]:
        seen: list[str] = []
        for entry in self.entries:
            if entry.rule not in seen:
                seen.append(entry.rule)
        return tuple(seen)


def _harvest(
    module: object,
    declared: dict[tuple[str, str], QuantRuleDecoder],
    unregistered: dict[str, UnregisteredDecodePath],
) -> None:
    for attr in sorted(dir(module)):
        obj = getattr(module, attr, None)
        for dec in quant_rule_decoders_of(obj):
            declared[(dec.rule, dec.decoder)] = dec
        for path in unregistered_decode_path_of(obj):
            unregistered[path.decoder] = path


def _census(
    packages: tuple[str, ...],
) -> tuple[
    tuple[QuantRuleDecoder, ...],
    tuple[UnregisteredDecodePath, ...],
    tuple[ExcludedDecoderModule, ...],
]:
    declared: dict[tuple[str, str], QuantRuleDecoder] = {}
    unregistered: dict[str, UnregisteredDecodePath] = {}
    excluded: list[ExcludedDecoderModule] = []

    for package in packages:
        try:
            pkg = importlib.import_module(package)
        except Exception as exc:
            excluded.append(ExcludedDecoderModule(
                module=package, reason=f"{type(exc).__name__}: {exc}"))
            continue
        _harvest(pkg, declared, unregistered)
        paths = getattr(pkg, "__path__", None)
        if not paths:
            continue
        for info in sorted(pkgutil.iter_modules(paths), key=lambda m: m.name):
            name = f"{package}.{info.name}"
            try:
                _harvest(importlib.import_module(name), declared, unregistered)
            except Exception as exc:
                excluded.append(ExcludedDecoderModule(
                    module=name, reason=f"{type(exc).__name__}: {exc}"))
    return (
        tuple(declared[k] for k in sorted(declared)),
        tuple(unregistered[k] for k in sorted(unregistered)),
        tuple(excluded),
    )


def _digest(ds: DecodeSet) -> str:
    payload = msgspec.json.encode(msgspec.structs.replace(ds, digest=""))
    return hashlib.sha256(payload).hexdigest()


def derive_decode_set(
    packages: tuple[str, ...] = DEFAULT_DECODER_PACKAGES,
) -> DecodeSet:
    """The quant rules this image's decoders implement."""
    declared, unregistered, excluded = _census(packages)
    ds = DecodeSet(
        derivation=DERIVATION,
        entries=tuple(
            DecodeEntry(
                rule=d.rule,
                decoder=d.decoder,
                serves=tuple(sorted(d.serves)),
                composes_lora=d.composes_lora,
            )
            for d in declared
        ),
        unregistered=unregistered,
        excluded_modules=excluded,
    )
    return msgspec.structs.replace(ds, digest=_digest(ds))


def manifest_block(ds: DecodeSet) -> Dict[str, Any]:
    """The ``[decode_set]`` block as it lands in endpoint.lock.

    TWO RENDERS OF ONE CENSUS, AND THEY CANNOT DISAGREE — asserted below, not
    hoped for.

    `rules` is the v2 name and the forward one. `contracts` is the name the
    hub's reader uses TODAY: tensorhub's `internal/builder/manifest_contract.go`
    (`manifestDecodeSetBlock`) unmarshals `contracts[].{contract, decoder,
    key_topologies}`, and `internal/builder/execution_lanes.go`
    (`declaredKeyTopologies`) is its only consumer — it unions `key_topologies`
    per contract into the th#2037 row and refuses a publish whose
    `[execution_lanes]` names a contract `[decode_set]` has no entry for.

    **Emitting only `rules` would not refuse — IT WOULD GO QUIET.** The hub's
    `DecodeSet` pointer is nil-able and absence is legal (every endpoint
    published before the block existed is that manifest), so a lock carrying
    only the new name reads to the hub exactly like an old image: the
    cross-check is skipped and the key-topology rows arrive empty. That is the
    silent-permissive direction, and it is the failure mode this whole census
    exists to remove. A field name is a wire name; renaming one is a
    coordinated act, and going dark while waiting for the other side is not the
    same thing as cutting cleanly.

    So both names carry the SAME rows, and `key_topologies` is `[]` — which is
    the hub's own UNDECLARED rung, and now the only honest value: a decoder no
    longer constrains the key convention, because the key convention became the
    TOPOLOGY half of a lane's stamp. The hub-side rename (`contracts` ->
    `rules`, dropping `key_topologies`) is the follow-up sequenced with
    th#2250's glob-registry retirement; when it lands, the `contracts` key
    below is deleted and nothing else moves.
    """
    rules = [
        {
            "rule": e.rule,
            "decoder": e.decoder,
            "serves": list(e.serves),
            "composes_lora": e.composes_lora,
        }
        for e in ds.entries
    ]
    contracts = [
        {
            "contract": e.rule,
            "decoder": e.decoder,
            # The tri-state's UNDECLARED rung, in the hub's own spelling of it.
            "key_topologies": [],
        }
        for e in ds.entries
    ]
    if [r["rule"] for r in rules] != [c["contract"] for c in contracts]:
        raise AssertionError(  # pragma: no cover - structural, cannot diverge
            "decode_set manifest_block: the `rules` and `contracts` renders "
            "disagree. They are one census written twice for one wire rename; "
            "if they can disagree the hub can be told two different things "
            "about which decoders shipped."
        )
    return {
        "derivation": ds.derivation,
        "digest": ds.digest,
        "rules": rules,
        "contracts": contracts,
        "unregistered": [
            {"decoder": u.decoder, "reason": u.reason} for u in ds.unregistered
        ],
        "excluded_modules": [
            {"module": m.module, "reason": m.reason} for m in ds.excluded_modules
        ],
    }


def _parts(handle: str) -> tuple[str, str, frozenset[str]]:
    namespace, _, rest = handle.partition(".")
    name, _, _major = rest.partition("@")
    return namespace, name, frozenset(t for t in name.replace(".", "-").split("-") if t)


def nearest_declared(rule: str, ds: DecodeSet) -> str:
    """The declared quant rule closest to ``rule``, or ``""``.

    Nearest is a REMEDY, not a substitution: it is what the refusal names so a
    reader can tell "you pinned the wrong major" from "nothing in this image
    is remotely this format". Ranked by same NAME (a major difference), then
    shared FORMAT TOKENS, then same namespace — the namespace is the PRODUCER
    and ranks last on purpose: `cozy.fp8-rowwise@1` is nearer to
    `hf.fp8-blockwise@1` than to `cozy.nvfp4-flat@1`, and whole-string
    similarity gets both of those backwards. Ties break on the handle so two
    builds answer alike.
    """
    from difflib import SequenceMatcher

    want_ns, want_name, want_tokens = _parts(rule)
    best, best_score = "", -1.0
    for candidate in sorted(ds.rules()):
        ns, name, tokens = _parts(candidate)
        union = want_tokens | tokens
        score = 0.5 * SequenceMatcher(None, rule, candidate).ratio()
        score += 2.0 * len(want_tokens & tokens) / len(union) if union else 0.0
        if name == want_name:
            score += 4.0
        if ns == want_ns:
            score += 0.5
        if score > best_score:
            best, best_score = candidate, score
    return best


class RuleNotDecodableError(Exception):
    """A bound variant's quant rule is outside this image's decode-set.

    Typed and named on both halves — the rule asked for and the nearest one
    declared — because the remedy is on one side or the other and the operator
    cannot tell which from "load failed".
    """

    code = REFUSAL_RULE_UNDECLARED

    def __init__(
        self,
        rule: str,
        *,
        declared: tuple[str, ...],
        nearest: str = "",
        unregistered: tuple[UnregisteredDecodePath, ...] = (),
        where: str = "",
    ) -> None:
        self.rule = rule
        self.declared = declared
        self.nearest = nearest
        self.unregistered = unregistered
        self.where = where
        subject = f" for {where}" if where else ""
        remedy = (
            f"nearest declared rule is {nearest!r}"
            if nearest else
            "this image declares NO quant rule at all — its decoder modules "
            "failed to import, or it ships none"
        )
        note = ""
        if unregistered:
            note = (
                "; this image also carries decode paths no ratified quant rule "
                "covers (" + ", ".join(u.decoder for u in unregistered) + ")"
            )
        super().__init__(
            f"quant rule {rule!r}{subject} is not in this image's declared "
            f"decode-set (declared: {', '.join(declared) or 'none'}); "
            f"{remedy}. Bind a variant in a declared rule, or ship an "
            f"image whose decoder declares this one{note}."
        )


_RUNTIME_SET: Optional[DecodeSet] = None


def runtime_decode_set() -> DecodeSet:
    """This process's decode-set, derived once."""
    global _RUNTIME_SET
    if _RUNTIME_SET is None:
        _RUNTIME_SET = derive_decode_set()
    return _RUNTIME_SET


class DecodeSetDriftError(Exception):
    """The set this process derives is not the one stamped at image build."""

    code = REFUSAL_DECODE_SET_DRIFT

    def __init__(
        self, *, stamped: str, live: str,
        gained: tuple[str, ...] = (), lost: tuple[str, ...] = (),
        excluded: tuple[ExcludedDecoderModule, ...] = (),
    ) -> None:
        self.stamped, self.live = stamped, live
        self.gained, self.lost, self.excluded = gained, lost, excluded
        detail = []
        if lost:
            detail.append(
                f"the lock claims {', '.join(lost)} and this process derives "
                "no decoder for them")
        if gained:
            detail.append(
                f"this process decodes {', '.join(gained)} and the lock does "
                "not declare them")
        if not detail:
            detail.append(
                "the RULE set agrees, so a lane body, a decoder name or an "
                "excluded module differs")
        if excluded:
            detail.append(
                "excluded modules here: " + "; ".join(
                    f"{m.module} ({m.reason})" for m in excluded))
        super().__init__(
            f"decode-set drift: endpoint.lock stamped {stamped[:16]}…, this "
            f"process derives {live[:16]}… — " + ". ".join(detail) +
            ". Rebuild the image so the lock and the code are one artifact."
        )


def assert_matches_baked(baked: Dict[str, Any], ds: Optional[DecodeSet] = None) -> None:
    """Refuse when the process's decode-set is not the image's baked one.

    A block with no digest is an OLDER image and is not evidence of anything;
    a digest that disagrees is a real divergence and fails closed, naming the
    rules that appeared or vanished so the remedy is readable.
    """
    stamped = str((baked or {}).get("digest") or "")
    if not stamped:
        return
    live = ds or runtime_decode_set()
    if stamped == live.digest:
        return
    was = {str(r.get("rule") or "") for r in (baked.get("rules") or [])}
    now = set(live.rules())
    raise DecodeSetDriftError(
        stamped=stamped, live=live.digest,
        gained=tuple(sorted(now - was)), lost=tuple(sorted(was - now)),
        excluded=live.excluded_modules,
    )

class KeyTopologyUnclassifiedError(Exception):
    """A DENOISER tree whose key convention matches nothing this image knows.

    Fails closed, deliberately. The publish side refuses an underivable
    topology rather than guessing a nearest match (th#1937), and the load side
    holds the same line where it costs the most: the model class is selected
    by the architecture the keys describe, so a convention nothing here
    recognizes is exactly the state that produced `Cannot detect the model
    type` after a 71 GB fetch.

    Scoped to the denoiser ON PURPOSE. A VAE, text encoder, scheduler or
    tokenizer is not addressed by an architecture-specific model class, so the
    axis does not apply to them and "unclassified" there is a fact rather than
    a hedge — refusing it would refuse the entire fleet to catch nothing.

    THE SISTER REFUSAL IS GONE (pgw#1621). `decode_set_key_topology_unsupported`
    said "classified, and no decoder of this handle ingests THAT convention" —
    an intersection on the deleted `key_topologies` decode axis, which is now
    the TOPOLOGY half of a v2 lane stamp and not a decoder's to declare. What
    survives here is the CLASSIFICATION, which was never an intersection: a
    fourth attention spelling nothing in this image can address still refuses
    before the fetch. What no longer refuses is a tree in a convention this
    image recognizes but this decoder cannot ingest.
    """

    code = REFUSAL_KEY_TOPOLOGY_UNCLASSIFIED

    def __init__(
        self, rule: str, *, sample: tuple[str, ...],
        registered: tuple[str, ...], where: str = "",
    ) -> None:
        self.rule = rule
        self.sample = sample
        self.registered = registered
        self.where = where
        subject = f" at {where}" if where else ""
        super().__init__(
            f"the denoiser tree{subject} is in a tensor-KEY convention nothing "
            f"in this image recognizes (it knows: {', '.join(registered)}). "
            f"Keys seen: {', '.join(sample)}. The quant rule {rule} says what "
            f"the bytes ARE; this says nothing here knows how they are "
            f"ADDRESSED, and a model class chosen hopefully fails at load with "
            f"an unrelated message. Bind a tree in a known convention, or ship "
            f"an image whose model class reads this one."
        )


def require_decodable(
    rule: str,
    *,
    decode_set: Optional[DecodeSet] = None,
    where: str = "",
    keys: Optional["SnapshotKeys"] = None,
) -> None:
    """Refuse, typed, unless this image can decode the artifact. Two checks,
    two codes, both answered from headers and directory shape before any
    tensor is read:

    1. ``rule`` is in the image's decode-set, else ``decode_set_rule_undeclared``;
    2. the artifact's key convention was CLASSIFIED — a DENOISER tree matching
       nothing this image recognizes is
       ``decode_set_key_topology_unclassified``, never a hopeful pass.

    Check 1 now carries what used to take three: a v2 quant rule's conventions
    (nibble order, scale rank, scale layout, scale leaf) ARE its identity, so
    a decoder that names the rule has named every axis the old
    ``DecodeDimensions`` spelled beside it, and the old per-decoder FILE-LAYOUT
    intersection has no successor here at all.

    **What "unknown" does, stated so it cannot be misread:** for the DENOISER
    it REFUSES. For every other component — vae, text encoder, scheduler — the
    key axis does not apply and is not evaluated, which is the tri-state's
    UNDECLARED rung, not a fail-open: a refusal there would refuse the whole
    fleet to catch nothing, since no architecture-specific model class is
    chosen from those trees.

    ``keys`` absent is likewise UNDECLARED — a caller that did not classify is
    not a caller that classified as fine.
    """
    ds = decode_set if decode_set is not None else runtime_decode_set()
    declared = ds.rules()
    if rule not in declared:
        raise RuleNotDecodableError(
            rule,
            declared=tuple(sorted(declared)),
            nearest=nearest_declared(rule, ds),
            unregistered=ds.unregistered,
            where=where,
        )
    if keys is None:
        return
    if keys.unclassified_denoiser:
        from gen_worker.models.key_topology import known_key_conventions

        raise KeyTopologyUnclassifiedError(
            rule, sample=keys.sample, registered=known_key_conventions(),
            where=where)
