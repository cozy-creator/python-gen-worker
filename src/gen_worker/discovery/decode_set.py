"""The DECLARED DECODE-SET — what contracts this image's code can decode,
derived at IMAGE BUILD from the decoders that survived the import.

th#1938's `resolve()` intersects three things: the endpoint's contract
patterns, the worker's decode-set, and card executability. This module is the
second one's ground truth. It exists because different quant recipes are not
one "quantized" concept — `cozy.fp8-rowwise@1`, `hf.fp8-blockwise@1`,
`cozy.svdq-nvfp4-lr8@1` and `nunchaku.v1@1` are four byte formats needing four
decode paths, and no config may select bytes the code cannot read.

Three properties, each of them a thing that could otherwise be faked:

1. **Derived, never written.** The census imports every decoder module and
   harvests the `@implements_contract` markers that survived. A module that
   raises contributes no declaration and one excluded-with-reason record.
2. **A property of the BUILT IMAGE.** `python -m gen_worker.discovery` runs in
   the image build and stamps the block into `endpoint.lock`; the digest
   below is what makes "the set the hub read" and "the set this process would
   derive" comparable rather than merely plausible.
3. **Complete.** A declaration carries the contract's DIMENSIONS the decoder
   reads (`DecodeDimensions`), not just its handle, and decode paths whose
   bytes no registered contract covers are recorded as such instead of living
   in a source comment nothing can read.
"""

from __future__ import annotations

import hashlib
import importlib
import pkgutil
from typing import Any, Dict, Optional

import msgspec

from gen_worker.models.key_topology import SnapshotKeys
from gen_worker.models.tensor_layout_contract import (
    ContractDecoder,
    DecodeDimensions,
    UnregisteredDecodePath,
    contract_decoders_of,
    unregistered_decode_path_of,
)

# Names the MECHANISM, so an image that predates it emits no block at all and
# is UNPROVEN to a reader — distinct from one that derived an empty set.
DERIVATION = "gen_worker.discovery.decode_set@1"

DEFAULT_DECODER_PACKAGES: tuple[str, ...] = ("gen_worker.models",)

#: The typed refusal wire codes. One spelling each, here, so the worker and the
#: hub cannot drift into two names for one refusal.
REFUSAL_CONTRACT_UNDECLARED = "decode_set_contract_undeclared"
REFUSAL_KEY_TOPOLOGY_UNSUPPORTED = "decode_set_key_topology_unsupported"
REFUSAL_KEY_TOPOLOGY_UNCLASSIFIED = "decode_set_key_topology_unclassified"
REFUSAL_FILE_LAYOUT_UNSUPPORTED = "decode_set_file_layout_unsupported"
REFUSAL_DECODE_SET_DRIFT = "decode_set_drift"


class ExcludedDecoderModule(msgspec.Struct, frozen=True, kw_only=True):
    module: str
    reason: str


class DecodeEntry(msgspec.Struct, frozen=True, kw_only=True):
    """One (contract, decoder) the image ships, with what it decodes."""

    contract: str
    decoder: str
    serves: tuple[str, ...]
    composes_lora: bool
    decodes: DecodeDimensions


class DecodeSet(msgspec.Struct, frozen=True, kw_only=True):
    derivation: str
    entries: tuple[DecodeEntry, ...]
    unregistered: tuple[UnregisteredDecodePath, ...]
    excluded_modules: tuple[ExcludedDecoderModule, ...]
    digest: str = ""

    def contracts(self) -> tuple[str, ...]:
        seen: list[str] = []
        for entry in self.entries:
            if entry.contract not in seen:
                seen.append(entry.contract)
        return tuple(seen)


def _harvest(
    module: object,
    declared: dict[tuple[str, str], ContractDecoder],
    unregistered: dict[str, UnregisteredDecodePath],
) -> None:
    for attr in sorted(dir(module)):
        obj = getattr(module, attr, None)
        for dec in contract_decoders_of(obj):
            declared[(dec.contract, dec.decoder)] = dec
        for path in unregistered_decode_path_of(obj):
            unregistered[path.decoder] = path


def _census(
    packages: tuple[str, ...],
) -> tuple[
    tuple[ContractDecoder, ...],
    tuple[UnregisteredDecodePath, ...],
    tuple[ExcludedDecoderModule, ...],
]:
    """Import every decoder module and harvest the markers that survived.

    Import is the whole test: a module that raises contributes NO declaration
    and one excluded-with-reason record. There is no third state where a
    decoder is claimed but absent.
    """
    declared: dict[tuple[str, str], ContractDecoder] = {}
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
    """sha256 over the canonical encoding of everything but the digest.

    Sorted throughout, so two builds of one image agree byte-for-byte and a
    changed declaration moves the digest. This is what makes "derived at image
    build" checkable rather than asserted.
    """
    payload = msgspec.json.encode(msgspec.structs.replace(ds, digest=""))
    return hashlib.sha256(payload).hexdigest()


def derive_decode_set(
    packages: tuple[str, ...] = DEFAULT_DECODER_PACKAGES,
) -> DecodeSet:
    """The contracts this image's decoders implement, with their dimensions."""
    declared, unregistered, excluded = _census(packages)
    ds = DecodeSet(
        derivation=DERIVATION,
        entries=tuple(
            DecodeEntry(
                contract=d.contract,
                decoder=d.decoder,
                serves=tuple(sorted(d.serves)),
                composes_lora=d.composes_lora,
                decodes=d.decodes,
            )
            for d in declared
        ),
        unregistered=unregistered,
        excluded_modules=excluded,
    )
    return msgspec.structs.replace(ds, digest=_digest(ds))


def manifest_block(ds: DecodeSet) -> Dict[str, Any]:
    """The ``[decode_set]`` block as it lands in endpoint.lock."""
    return {
        "derivation": ds.derivation,
        "digest": ds.digest,
        "contracts": [
            {
                "contract": e.contract,
                "decoder": e.decoder,
                "serves": list(e.serves),
                "composes_lora": e.composes_lora,
                "elements": list(e.decodes.elements),
                "scales": list(e.decodes.scales),
                "key_topologies": list(e.decodes.key_topologies),
                "file_layouts": list(e.decodes.file_layouts),
                "bakes": list(e.decodes.bakes),
            }
            for e in ds.entries
        ],
        "unregistered": [
            {"decoder": u.decoder, "reason": u.reason} for u in ds.unregistered
        ],
        "excluded_modules": [
            {"module": m.module, "reason": m.reason} for m in ds.excluded_modules
        ],
    }


# ── The refusal ──────────────────────────────────────────────────────────────


def _parts(handle: str) -> tuple[str, str, frozenset[str]]:
    namespace, _, rest = handle.partition(".")
    name, _, _major = rest.partition("@")
    return namespace, name, frozenset(t for t in name.replace(".", "-").split("-") if t)


def nearest_declared(contract: str, ds: DecodeSet) -> str:
    """The declared contract closest to ``contract``, or ``""``.

    Nearest is a REMEDY, not a substitution: it is what the refusal names so a
    reader can tell "you pinned the wrong major" from "nothing in this image
    is remotely this format". Ranked by same NAME (a major difference), then
    shared FORMAT TOKENS, then same namespace — the namespace is the PRODUCER
    and ranks last on purpose: `cozy.fp8-rowwise@1` is nearer to
    `hf.fp8-blockwise@1` than to `cozy.svdq-nvfp4-lr8@1`, and whole-string
    similarity gets both of those backwards. Ties break on the handle so two
    builds answer alike.
    """
    from difflib import SequenceMatcher

    want_ns, want_name, want_tokens = _parts(contract)
    best, best_score = "", -1.0
    for candidate in sorted(ds.contracts()):
        ns, name, tokens = _parts(candidate)
        union = want_tokens | tokens
        score = 0.5 * SequenceMatcher(None, contract, candidate).ratio()
        score += 2.0 * len(want_tokens & tokens) / len(union) if union else 0.0
        if name == want_name:
            score += 4.0
        if ns == want_ns:
            score += 0.5
        if score > best_score:
            best, best_score = candidate, score
    return best


class ContractNotDecodableError(Exception):
    """A bound variant's contract is outside this image's decode-set.

    Typed and named on both halves — the contract asked for and the nearest
    one declared — because the remedy is on one side or the other and the
    operator cannot tell which from "load failed".
    """

    code = REFUSAL_CONTRACT_UNDECLARED

    def __init__(
        self,
        contract: str,
        *,
        declared: tuple[str, ...],
        nearest: str = "",
        unregistered: tuple[UnregisteredDecodePath, ...] = (),
        where: str = "",
    ) -> None:
        self.contract = contract
        self.declared = declared
        self.nearest = nearest
        self.unregistered = unregistered
        self.where = where
        subject = f" for {where}" if where else ""
        remedy = (
            f"nearest declared contract is {nearest!r}"
            if nearest else
            "this image declares NO contract at all — its decoder modules "
            "failed to import, or it ships none"
        )
        note = ""
        if unregistered:
            note = (
                "; this image also carries decode paths no registered contract "
                "covers (" + ", ".join(u.decoder for u in unregistered) + ")"
            )
        super().__init__(
            f"contract {contract!r}{subject} is not in this image's declared "
            f"decode-set (declared: {', '.join(declared) or 'none'}); "
            f"{remedy}. Bind a variant in a declared contract, or ship an "
            f"image whose decoder declares this one{note}."
        )


_RUNTIME_SET: Optional[DecodeSet] = None


def runtime_decode_set() -> DecodeSet:
    """This process's decode-set, derived once.

    The image's build-time block in endpoint.lock is the artifact the HUB
    reads; this is the same derivation run against the same code, and
    :func:`assert_matches_baked` is what proves they are the same answer.
    """
    global _RUNTIME_SET
    if _RUNTIME_SET is None:
        _RUNTIME_SET = derive_decode_set()
    return _RUNTIME_SET


class DecodeSetDriftError(Exception):
    """The set this process derives is not the one stamped at image build.

    Fails the boot closed. The hub selected variants against the LOCK's set,
    so a process that would derive a different one accepts work it may not be
    able to decode — a stale `endpoint.lock` layer, or a decoder whose
    dependency is present at build and absent in the shipped image.
    """

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
                "the contract SET agrees, so a dimension, a decoder name or "
                "an excluded module differs")
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
    contracts that appeared or vanished so the remedy is readable.
    """
    stamped = str((baked or {}).get("digest") or "")
    if not stamped:
        return
    live = ds or runtime_decode_set()
    if stamped == live.digest:
        return
    was = {str(c.get("contract") or "") for c in (baked.get("contracts") or [])}
    now = set(live.contracts())
    raise DecodeSetDriftError(
        stamped=stamped, live=live.digest,
        gained=tuple(sorted(now - was)), lost=tuple(sorted(was - now)),
        excluded=live.excluded_modules,
    )


class KeyTopologyUnsupportedError(Exception):
    """The artifact's tensor-KEY convention is one no decoder here ingests.

    Distinct from :class:`ContractNotDecodableError` because everything about
    the bytes is right — right contract, right element, right scales — and the
    model class still cannot read them: the keys are addressed differently.
    Measured cost of not having this: `Cannot detect the model type` from an
    md5-over-key:shape lookup, after a 71 GB fetch onto a rented 4xH100.
    """

    code = REFUSAL_KEY_TOPOLOGY_UNSUPPORTED

    def __init__(
        self,
        contract: str,
        *,
        observed: str,
        accepted: tuple[str, ...],
        where: str = "",
    ) -> None:
        self.contract = contract
        self.observed = observed
        self.accepted = accepted
        self.where = where
        subject = f" at {where}" if where else ""
        super().__init__(
            f"the artifact{subject} is written in key topology {observed!r}, "
            f"which this image's {contract} decoder does not ingest (it "
            f"accepts: {', '.join(accepted) or 'none'}). The bytes are the "
            f"right format and the KEYS are addressed differently, so no "
            f"quantization or dtype change fixes it — bind the variant in an "
            f"accepted topology, or ship an image whose model class ingests "
            f"this one."
        )


class KeyTopologyUnclassifiedError(Exception):
    """A DENOISER tree whose key convention matches no registered topology.

    Fails closed, deliberately. The publish side refuses an underivable
    topology rather than guessing a nearest match (th#1937), and the load side
    holds the same line where it costs the most: the model class is selected
    by the architecture the keys describe, so a convention nothing registers is
    exactly the state that produced `Cannot detect the model type` after a
    71 GB fetch.

    Scoped to the denoiser ON PURPOSE. A VAE, text encoder, scheduler or
    tokenizer is not addressed by an architecture-specific model class, so the
    axis does not apply to them and "unclassified" there is a fact rather than
    a hedge — refusing it would refuse the entire fleet to catch nothing.
    """

    code = REFUSAL_KEY_TOPOLOGY_UNCLASSIFIED

    def __init__(
        self, contract: str, *, sample: tuple[str, ...],
        registered: tuple[str, ...], where: str = "",
    ) -> None:
        self.contract = contract
        self.sample = sample
        self.registered = registered
        self.where = where
        subject = f" at {where}" if where else ""
        super().__init__(
            f"the denoiser tree{subject} is in a tensor-KEY convention no "
            f"registered topology matches (registered: "
            f"{', '.join(registered)}). Keys seen: {', '.join(sample)}. The "
            f"contract {contract} says what the bytes ARE; this says nothing "
            f"here knows how they are ADDRESSED, and a model class chosen "
            f"hopefully fails at load with an unrelated message. Register the "
            f"topology and declare a decoder for it, or bind a tree in a "
            f"known one."
        )


class FileLayoutUnsupportedError(Exception):
    """The artifact's on-disk SHAPE is one no decoder of this contract reads.

    A third fact, distinct from the handle and from the key convention: the
    bytes may be exactly the right format, addressed in exactly the right
    convention, and still arrive as a shape the decoder's entry point cannot
    open. Measured case: both svdq engines refuse a bare single-file nunchaku
    transformer — a servable flavor must be a full diffusers tree with the
    checkpoint under its denoiser directory — and until this axis existed that
    fact lived only in two `raise` statements reached AFTER the CUDA gate, so
    a host without a GPU reported "svdq artifacts require a CUDA GPU" for an
    artifact no GPU would have helped.
    """

    code = REFUSAL_FILE_LAYOUT_UNSUPPORTED

    def __init__(
        self,
        contract: str,
        *,
        observed: str,
        accepted: tuple[str, ...],
        where: str = "",
    ) -> None:
        self.contract = contract
        self.observed = observed
        self.accepted = accepted
        self.where = where
        subject = f" at {where}" if where else ""
        super().__init__(
            f"the artifact{subject} is in file layout {observed!r}, which "
            f"this image's {contract} decoder does not read (it accepts: "
            f"{', '.join(accepted) or 'none'}). The bytes may be the right "
            f"format in the right key convention and still be the wrong "
            f"SHAPE — bind a variant published in an accepted layout, or ship "
            f"an image whose decoder opens this one."
        )


def accepted_file_layouts(contract: str, ds: DecodeSet) -> tuple[str, ...]:
    """Every on-disk shape this image's decoders of ``contract`` read."""
    out: list[str] = []
    for entry in ds.entries:
        if entry.contract != contract:
            continue
        for token in entry.decodes.file_layouts:
            if token not in out:
                out.append(token)
    return tuple(sorted(out))


def accepted_key_topologies(contract: str, ds: DecodeSet) -> tuple[str, ...]:
    """Every key convention this image's decoders of ``contract`` ingest."""
    out: list[str] = []
    for entry in ds.entries:
        if entry.contract != contract:
            continue
        for token in entry.decodes.key_topologies:
            if token not in out:
                out.append(token)
    return tuple(sorted(out))


def require_decodable(
    contract: str,
    *,
    decode_set: Optional[DecodeSet] = None,
    where: str = "",
    keys: Optional["SnapshotKeys"] = None,
    layout: str = "",
) -> None:
    """Refuse, typed, unless this image can decode the artifact. Four checks,
    four codes, all answered from headers and directory shape before any
    tensor is read:

    1. ``contract`` is in the image's decode-set, else
       ``decode_set_contract_undeclared``;
    2. a decoder of ``contract`` reads the artifact's on-disk SHAPE, else
       ``decode_set_file_layout_unsupported``;
    3. the artifact's key convention was CLASSIFIED — a DENOISER tree matching
       no registered topology is ``decode_set_key_topology_unclassified``,
       never a hopeful pass;
    4. a decoder of ``contract`` ingests that convention, else
       ``decode_set_key_topology_unsupported``.

    ``layout`` is an OBSERVATION (``models.file_layout.observed_file_layout``),
    and an empty one is an unclassifiable shape: not evaluated, the same
    UNDECLARED rung an unconstrained axis takes.

    **What "unknown" does, stated so it cannot be misread:** for the DENOISER
    it REFUSES (check 2). For every other component — vae, text encoder,
    scheduler — the key-topology axis does not apply and is not evaluated,
    which is the tri-state's UNDECLARED rung, not a fail-open: a refusal there
    would refuse the whole fleet to catch nothing, since no
    architecture-specific model class is chosen from those trees.
    """
    ds = decode_set if decode_set is not None else runtime_decode_set()
    declared = ds.contracts()
    if contract not in declared:
        raise ContractNotDecodableError(
            contract,
            declared=tuple(sorted(declared)),
            nearest=nearest_declared(contract, ds),
            unregistered=ds.unregistered,
            where=where,
        )
    if layout:
        readable = accepted_file_layouts(contract, ds)
        if readable and layout not in readable:
            raise FileLayoutUnsupportedError(
                contract, observed=layout, accepted=readable, where=where)
    if keys is None:
        return
    accepted = accepted_key_topologies(contract, ds)
    if not accepted:
        # This contract's decoders do not CONSTRAIN the key axis — the quant
        # handle already fixes the keys (th#1937 declined a synonym token for
        # exactly that reason). An unconstrained axis is not evaluated, which
        # is the tri-state's UNDECLARED rung and not a fail-open: the handle
        # check above already refused anything this image cannot decode.
        return
    if keys.unclassified_denoiser:
        from gen_worker.models.tensor_layout_contract import (
            KNOWN_KEY_TOPOLOGIES,
        )

        raise KeyTopologyUnclassifiedError(
            contract, sample=keys.sample, registered=KNOWN_KEY_TOPOLOGIES,
            where=where)
    if not keys.topology:
        return
    if keys.topology not in accepted:
        raise KeyTopologyUnsupportedError(
            contract, observed=keys.topology, accepted=accepted, where=where)
