"""pgw#903 — does this compiled artifact belong to the execution the hub
issued? Answered BEFORE ``dlopen``, by comparing DECLARED identities.

An AOT cell is not data. It is a ``dlopen``-ed ELF this pod executes, built by
some other pod against one exact toolchain, one environment seal and one traced
graph contract. Until now the serve path verified the package against ITSELF —
``aot_serve.verify`` rules on sm/torch/cuda/host-ISA/family, all probed from
this runtime, and ``verify_contract`` checks the envelope's internal
consistency. Both are necessary and neither asks the question this module asks:
*is this the artifact the current ExecutionSpec named?*

Nothing asked it because nothing could: before th#1457 there was no immutable
spec to compare against, so the mint's baked facts (``aot_mint``'s
``shared_identity_blocks``) had no counterpart. They do now —
``ExecutionSpec.arm.artifact`` carries the cell key, toolchain digest and env
seal digest, and ``ExecutionSpec.arm.graph_contract_digest`` carries the graph
contract — so the comparison is a lookup on both sides rather than a
reconstruction on either.

**The comparison is DECLARED-IDENTITY, never bytes.** §4.25/§4.26 forbid
confirming an artifact by comparing it to another mint of the same key: pgw#1006
measured that two mints of one key legitimately differ, so a byte comparison
would refuse healthy cells and prove nothing about unhealthy ones. What is
compared here are the digests each side already committed to. (Verifying the
DELIVERED bytes against the content digest the hub named is a different and
legitimate check — it belongs to delivery, pgw#904, not to identity.)

Every refusal names the axis, the expectation and what arrived, because these
are read as fleet events: "toolchain_digest: expected a1b2…, have c3d4…" tells
the hub team an image rebuilt without re-minting; "compatibility failed" tells
them nothing. And compatibility is never a preference — a mismatch on any axis
is a refusal, never a reason to prefer a different cell. That is the whole
distinction between this and the fetch-and-filter resolver pgw#904 deletes.

Scope note: the publisher claim is verified where the hub-signed receipt is
read, and "this arm may materialize no artifact at all" is enforced where
arming happens. Both are pgw#904's, at their call sites; putting them here
would add surface this PR cannot reach.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import msgspec

from . import cell_key, env_seal


class NamesAnArtifact(Protocol):
    """A source that NAMES one compiled cell, by the axes it already carries.

    ``cell_resolve.ResolvedCell`` (the hub answered a pull by derived key) is
    the source today; the Protocol stays because the map must not learn a
    second shape when a second source appears.
    """

    @property
    def cell_key(self) -> str: ...

    @property
    def toolchain_digest(self) -> str: ...

    @property
    def env_seal_digest(self) -> str: ...

    @property
    def publisher_org(self) -> str: ...


class ExpectedIdentity(msgspec.Struct, frozen=True):
    """What the current ExecutionSpec says the armed artifact must be.

    Projected from what the hub already digested, so it cannot drift from the
    execution identity. Every field is a digest that some producer
    already committed to; none is probed from this runtime (the runtime axes
    are ``aot_serve.verify``'s job and stay there).

    ONE map builds it — :meth:`named_by`, fenced by
    ``scripts/lint_arm_state_feeders.py``. This is the object every arm gate
    compares a cell against, so a field reaching one source's map and not the
    other's is pgw#1150's defect one level in.
    """

    cell_key: str
    toolchain_digest: str
    env_seal_digest: str
    graph_contract_digest: str
    publisher_org: str

    # DELIBERATELY ABSENT: a code-closure axis.
    #
    # pgw#903's brief asks for closure identity, and the landed schema does not
    # carry a comparable one. `EndpointRelease.code_closure_id` is the HUB's
    # identifier for a release's code; the artifact records
    # `code_closure` = {source path: content digest} from `cc.static_code_closure`.
    # They are different quantities, and asserting they are equal would be this
    # lane inventing a cross-repo contract on the strength of two similar names
    # — which would then refuse every healthy cell in the fleet.
    #
    # Note also that the closure is not a `cell_key` axis, so it is not covered
    # transitively either. Closing this needs ONE ruling from the th#1457 lane:
    # either `ArtifactIdentity` gains a `code_closure_digest` the mint stamps
    # and the hub echoes, or the closure block goes (pgw#1034 already carries a
    # row to delete it as readerless — this is its other possible resolution,
    # and the two lanes must not settle it independently).

    @classmethod
    def named_by(
        cls, named: NamesAnArtifact, graph_contract_digest: str,
    ) -> ExpectedIdentity:
        """THE map from a source that named an artifact to the expectation.

        The graph contract is passed rather than read off ``named`` because it
        is a property of the ARM, not of the artifact — the resolve answer
        carries it as ``graph_contract``. Every
        other axis is the artifact's own and is copied by name, so a new axis
        cannot reach one source and not the other.
        """
        return ExpectedIdentity(
            cell_key=named.cell_key,
            toolchain_digest=named.toolchain_digest,
            env_seal_digest=named.env_seal_digest,
            graph_contract_digest=graph_contract_digest,
            publisher_org=named.publisher_org,
        )


def artifact_identity(meta: Mapping[str, Any]) -> ExpectedIdentity:
    """The identity an artifact's own recorded facts describe.

    Recomputed from the blocks the mint stored (``aot_mint``'s
    ``shared_identity_blocks``), never read from a separate stamp — the same
    discipline ``cell_key`` already enforces, for the same reason: a stamp that
    is not derived from the facts it summarizes can disagree with them, and
    then the disagreement is invisible.

    ``publisher_org`` is deliberately empty here. An artifact cannot attest to
    its own publisher — that claim lives inside the hub-signed receipt, and
    :func:`verify_receipt_publisher` is where it is checked.

    Not built through :meth:`ExpectedIdentity.named_by`, and classified
    PROJECTION in the fence's allowlist for that reason: this is the OBSERVED
    side, DERIVED from the mint's stored blocks rather than copied from a
    source that named an artifact. A projection from a different source shape
    is not a second copy of the map.
    """
    seal = meta.get(env_seal.SEAL_KEY)
    toolchain = meta.get("toolchain")
    return ExpectedIdentity(
        cell_key=str(meta.get("cell_key") or ""),
        toolchain_digest=(
            cell_key.toolchain_axis_digest(dict(toolchain))
            if isinstance(toolchain, dict) else ""),
        env_seal_digest=(
            env_seal.seal_digest(dict(seal)) if isinstance(seal, dict) else ""),
        # pgw#1176: the DECLARATION-wide coverage label, not identity.
        # Identity is `cell_key` above, per entry.
        graph_contract_digest=str(meta.get("manifest_digest") or ""),
        publisher_org="",
    )


#: The axes compared, in the order they are reported. Order is deliberate:
#: identity first (which cell is this), then what it was built from, then what
#: it traced. A cell that is the wrong cell should say so before it says its
#: toolchain differs, because the second fact is a consequence of the first.
_COMPARED_AXES: tuple[str, ...] = (
    "cell_key", "toolchain_digest", "env_seal_digest", "graph_contract_digest")

#: The axes NOT compared here, each with the gate that does compare them. An
#: axis is verified here or it is verified somewhere named — never neither.
#: The claim is CHECKED, not trusted: `test_artifact_identity_pgw903.py`
#: imports each named gate and asserts it reads the axis.
_VERIFIED_ELSEWHERE: Mapping[str, str] = {
    "publisher_org": (
        "fleet_cells.arm_ordered — an artifact cannot attest to its own "
        "producer, so §4.26's match is against the hub-signed RECEIPT's "
        "publisher_org_id, fail-closed on silence. Distinct from "
        "receipts.refuse_untrusted_publisher, which asks whether the producer "
        "is trusted AT ALL, not whether it is the one the spec named"),
}

if set(_COMPARED_AXES) | set(_VERIFIED_ELSEWHERE) != set(
    ExpectedIdentity.__struct_fields__
):
    # Unspellable rather than merely tested: an axis added to the identity and
    # to neither set is an axis NOTHING verifies, and a silently unverified
    # axis on this object is how a cell gets armed on an unchecked claim.
    raise AssertionError(
        "every ExpectedIdentity axis must be compared by "
        "verify_declared_identity or named in _VERIFIED_ELSEWHERE; "
        f"unaccounted: {sorted(set(ExpectedIdentity.__struct_fields__) - set(_COMPARED_AXES) - set(_VERIFIED_ELSEWHERE))}")


def verify_declared_identity(
    meta: Mapping[str, Any], expected: ExpectedIdentity,
) -> str:
    """``''`` when the artifact IS the one the spec named, else the reason.

    Fail-closed on every compared axis: an axis the artifact is SILENT on is a
    refusal, not a skip. A cell that cannot state its own toolchain cannot be
    shown to match the toolchain the spec named, and "cannot be shown to match"
    is exactly what a refusal means here.

    There is no partial pass and no conditional axis: every axis in
    :data:`_COMPARED_AXES` is compared, or the comparison is not one. The
    closure axis is absent rather than optional — see :class:`ExpectedIdentity`
    for why inventing it would be worse than owing it.
    """
    have = artifact_identity(meta)
    for axis in _COMPARED_AXES:
        want, got = getattr(expected, axis), getattr(have, axis)
        if not want:
            # An expectation that names no value cannot be verified, and an
            # unverifiable expectation must not read as a pass. BOTH sources
            # refuse an artifact missing any of these at the point they read
            # it — `plan._artifact` and `cell_resolve.resolve` — so reaching
            # this branch means a caller built an ExpectedIdentity by hand.
            return f"{axis}: the spec named no expected value"
        if want != got:
            return f"{axis}: expected {want}, have {got or '<absent>'}"
    return ""


#: The entry-block field carrying the node-level digest of the traced program
#: (``graph_hash.graph_hash``), stamped by ``aot_mint.keying_block``. Since
#: pgw#1031 (option a) it is FOLDED into ``class_hash`` (the key), and it also
#: stays recorded here as a top-level sibling for the adopt backstop — see
#: :func:`verify_graph_witness`.
GRAPH_WITNESS_FIELD = "graph_witness"


def verify_graph_witness(
    meta: Mapping[str, Any], witnesses: Mapping[str, str],
) -> str:
    """``''`` when the cell's recorded graph witnesses equal the ones this pod
    traced, else the reason. pgw#1031's fail-closed backstop.

    Since pgw#1031 (option a) the cell key IS the traced computation:
    ``class_hash`` folds ``graph_witness`` (the node-level body digest) beside
    target, fork, class dims, declared ranges, the graph-interface block and
    the declared envelope, so two endpoints whose bodies differ while their
    declarations agree now key APART — a collision is a MISS, not a wrong hit.
    Before the fold they minted under one ``ck1`` key (measured 2026-08-10 on
    ``micro-pad32`` vs ``micro-pad32-branchy``: 112 vs 102 nodes, byte-identical
    keying block, identical key), and pull-by-key adoption (§4.27/pgw#1090)
    would hand one endpoint the other's kernels with only the arm-time numerics
    tolerance between that and served output.

    This function is the belt-and-braces beneath the now-sound key: the mint
    records ``graph_witness`` per entry, the adopter derives its own from the
    same traced programs its key came from (``boot_key``), and a disagreement —
    a witness-blind cell, a hash-broken entry, a cross forced by any non-key
    path — is a typed refusal naming both digests. The pod then boots exactly
    as it booted before pull-by-key existed — eager, then mint — which is the
    correct outcome for "this cell is not my computation".

    Fail-closed in both directions, the doctrine
    :func:`verify_declared_identity` already keeps: a cell that is SILENT on an
    entry's witness cannot be shown to compute this pod's graph, and "cannot be
    shown to match" is what a refusal means. A witness set that does not cover
    the same entries is the same failure — a partial agreement is not a
    narrower match, it is an unproven one (pgw#716).

    ``witnesses`` is ``{entry: digest}`` as
    ``boot_key.DerivedKey.graph_witnesses`` reports it. An empty mapping is a
    caller error, not a pass.
    """
    if not witnesses:
        return (
            "this pod derived no graph witnesses, so the cell's compiled "
            "graphs cannot be shown to be the graphs it traced (pgw#1031)")
    entry = meta.get("entry")
    if not isinstance(entry, Mapping) or not entry:
        return "artifact records no entry block; it has no graph witness"
    recorded = {
        str(entry.get("name") or ""):
            str(entry.get(GRAPH_WITNESS_FIELD) or ""),
    }
    silent = sorted(name for name, value in recorded.items() if not value)
    if silent:
        return (
            f"entries {silent[:4]!r} record no {GRAPH_WITNESS_FIELD!r}: this "
            f"cell predates pgw#1031 and cannot state which graph it compiled")
    extra = sorted(set(recorded) - set(witnesses))
    missing = sorted(set(witnesses) - set(recorded))
    if extra or missing:
        return (
            f"graph witness class set differs (cell-only {extra[:3]!r}, "
            f"traced-only {missing[:3]!r}): the cell was compiled from a "
            f"different class set than this pod traced")
    for name in sorted(recorded):
        want, have = recorded[name], str(witnesses[name] or "")
        if want != have:
            return (
                f"entry {name!r}: graph witness {want} on the cell, {have} "
                f"traced here — the cell key matched but the COMPUTATION did "
                f"not (pgw#1031: the key folds the ingress contract, not the "
                f"graph nodes). Refusing to arm another graph's kernels")
    return ""


__all__ = [
    "GRAPH_WITNESS_FIELD",
    "ExpectedIdentity",
    "NamesAnArtifact",
    "artifact_identity",
    "verify_declared_identity",
    "verify_graph_witness",
]
