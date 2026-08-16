"""pgw#1059 — the ck1 KEY-SCHEMA REDEFINITION.

Four things are enforced here, each of which is a ruling rather than a
convenience:

1. THE MEMBERSHIP AXIOM (amendment 6, Paul: "don't key on parameters that
   don't require us to recompile"): the key is EXACTLY
   {graph, envelope, sm, toolchain}. Adding an axis fails this file first,
   with a pointer to the axiom.
2. ONE DERIVATION PER AXIS: the graph digest, the envelope digest and the
   exported-key recomputation each have exactly one implementation, and the
   set of call sites is pinned. The fence is RED-provable — it fires on a
   synthetic tree carrying a second derivation.
3. NON-COLLISION WITH THE PRE-REDEFINITION CORPUS: an old-schema key (the
   fused ``contract`` axis plus kind/format/family/lane/env_seal) can never
   equal a post-redefinition key, and a pre-redefinition artifact can never
   restate a post-redefinition identity — which is what makes the dev-stack
   purge (pgw#868 runbook) hygiene, not a correctness precondition.
4. TCG PARITY: while the worker still projects publish metadata, that
   projection derives exactly TCG's public ``CompiledGraphKey``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from gen_worker._vendor.torch_compiled_graphs import (
    GRAPH_CLASS_BLOCK,
    REQUIRED_AXES,
    CallIngress,
    CallInput,
    GraphClassDeclaration,
    RuntimeCompatibility,
)

from gen_worker._vendor.torch_compiled_graphs import identity as tcg_identity
from gen_worker._vendor.torch_compiled_graphs import is_compiled_graph_key

from gen_worker import fleet_cells, graph_facts

from harness.cell_meta import exported_cell_meta
import sys

# pgw#1310: one home for "which subtrees a guard may not judge" —
# scripts/_lint_scope.py, shared with the CI lint scanners.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _lint_scope import is_unowned  # noqa: E402


SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "gen_worker"


# ---------------------------------------------------------------------------
# 1. The membership axiom
# ---------------------------------------------------------------------------


def test_membership_axiom_the_key_is_exactly_three_axes():
    """The cg-key-v1 key contains exactly {graph, sm, toolchain}.

    If this test is failing because you added an axis: STOP. The membership
    axiom (pgw#1059 amendment 6, Paul 2026-08-09) admits an axis only when
    the fact provably alters the compiled artifact AND cannot ride an
    existing axis's fact block (trace-shaping facts ride ``graph`` via THIS
    entry's class hash; compiler configuration rides ``toolchain``).
    kind/format (single-valued), family/lane (store metadata + discovery
    scoping) and env_seal (folded into toolchain; gates remain) were each
    REMOVED by that test — and pgw#1176 removed ``envelope`` by it too: it
    digests the UNION of the ladder across the whole declaration, which is a
    property of the collection and not of any computation.
    """
    assert REQUIRED_AXES == ("graph", "sm", "toolchain")
    # Asserted against the AUTHORITY, not against a worker-side tuple:
    # pgw#1288 deleted the last worker copy, so there is nothing left to fence
    # — the axiom is TCG's to enforce and the wire reads its export directly.
    sample = {name: f"{name}-fact" for name in REQUIRED_AXES}
    assert tcg_identity.from_axes(sample).as_dict().keys() == set(REQUIRED_AXES)


def test_adding_an_axis_refuses_with_the_axiom_named():
    axes = {
        "graph": "g" * 16, "sm": "sm_89", "toolchain": "t" * 16, "sku": "l4",
    }
    with pytest.raises(tcg_identity.IdentityError) as exc:
        tcg_identity.from_axes(axes)
    assert "unknown identity axes" in str(exc.value)
    assert "sku" in str(exc.value)


@pytest.mark.parametrize(
    "stale", ["contract", "env_seal", "kind", "format", "family", "lane",
              "mode", "code_closure", "envelope"])
def test_every_dropped_axis_is_a_typed_refusal(stale):
    """``envelope`` joins the list under pgw#1176 — a stale caller shipping it
    must fail here, not silently widen the key."""
    axes = {
        "graph": "g" * 16, "sm": "sm_89",
        "toolchain": "t" * 16, stale: "anything",
    }
    with pytest.raises(tcg_identity.IdentityError, match="unknown identity axes"):
        tcg_identity.from_axes(axes)


def test_missing_axis_is_a_typed_refusal():
    with pytest.raises(tcg_identity.IdentityError, match="requires canonical string"):
        tcg_identity.from_axes({"graph": "g" * 16, "sm": "sm_89"})


# ---------------------------------------------------------------------------
# 2. One derivation per axis — the fence, and its RED proof
# ---------------------------------------------------------------------------

#: Call sites allowed to COMPUTE each single-derivation surface, as
#: (module-relative path, count-limit-free) pairs. Everything else in
#: src/gen_worker must READ the stamped value instead of re-deriving it.
#: Extending an allowlist is a conscious act reviewed against the module
#: docstrings of the named authorities — never a drive-by.
_DERIVATION_ALLOWLIST = {
    # the declaration-wide coverage LABEL (pgw#1176: demoted from identity,
    # so the fence is about ONE arithmetic, not one identity).
    "manifest_digest(": {
        "graph_facts.py",    # def
    },
    # the exported-ENTRY key: definition plus retained publish projection.
    # pgw#1277: the DEFINITION moved to torch_compiled_graphs.identity, which
    # is outside src/gen_worker — so the worker now has exactly ONE call site
    # and no definition to fence.
    "from_artifact_metadata(": {
        "fleet_cells.py",  # _recomputed_key (the publish recompute)
    },
    # the declared-envelope digest.
    "envelope_digest(": {
        "graph_facts.py",    # def (+ envelope_facts)
    },
    # the toolchain-axis digest, i.e. its MEMBERSHIP (pgw#1050). The producer
    # (`compile_cache.toolchain_digest`) collects components; this is the one
    # place that says which of them ARE the axis, and every reader that
    # restates the axis from a recorded block must come through it or the
    # two ends can disagree about membership.
    "toolchain_axis_digest(": {
        "fleet_cells.py",  # arm_identity + arm_axis_divergence
        "boot_key.py",     # boot compatibility closure
        # pgw#1205: the device-peak census's provenance. Added CONSCIOUSLY,
        # which is what this fence's own message offers as the alternative to
        # reading a stamped value — and here there is no stamped value to
        # read: the census is taken in the MINT CHILD, which holds an
        # `arm_token` (a hash of the facts) and never the facts themselves.
        #
        # It is the right call rather than merely the available one. A banked
        # reading says "this graph class cost this much under THIS toolchain",
        # and the only useful meaning of "this toolchain" is the one the CELL
        # uses — same membership, same digest. A second notion computed
        # locally would drift from the cell's exactly where it matters (the
        # header above: "the two ends can disagree about membership"), and the
        # bank would then be keyed on a toolchain nothing else recognises.
        #
        # It derives NO KEY: nothing compares this value against a cell key,
        # an arm key or a boot key. It is a bank axis, and the bank sizes
        # nothing (`test_device_peak_bank_pgw1205` fences that structurally).
        "aot_mint.py",
    },
}


def _derivation_sites(root: Path, needle: str) -> dict[str, int]:
    """{relative_path: occurrence_count} of ``needle`` in ``root``'s .py files,
    comments and docstrings excluded crudely by requiring the needle outside a
    line starting with ``#``."""
    sites: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        # pgw#1310: `_vendor/torch_compiled_graphs` is the AUTHORITY these
        # needles name. Counting it as a second derivation site inverts the
        # rule — the rule is that nothing in gen_worker re-derives what TCG
        # already computes, and TCG computing it is the premise.
        if is_unowned(path, root):
            continue
        count = 0
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            count += len(re.findall(re.escape(needle), line))
        if count:
            sites[rel] = count
    return sites


@pytest.mark.parametrize("needle", sorted(_DERIVATION_ALLOWLIST))
def test_single_derivation_fence(needle):
    """RED when a second derivation site appears anywhere in src/gen_worker.

    The attempt-28 phantom ("key divergence") was exactly a second
    derivation: an arm key derived from declared facts beside a stamped key
    derived from traced facts, under one axis name. The redefinition makes
    the derivations single-sited; this fence keeps them that way.
    """
    allowed = _DERIVATION_ALLOWLIST[needle]
    sites = _derivation_sites(SRC_ROOT, needle)
    offenders = {rel: n for rel, n in sites.items() if rel not in allowed}
    assert not offenders, (
        f"second derivation of {needle!r} found at {offenders!r} — the "
        f"membership axiom's one-derivation rule (pgw#1059) allows only "
        f"{sorted(allowed)!r}; read the stamped value instead, or extend the "
        f"allowlist consciously in tests/test_compiled_graph_key_pgw1059.py")


def test_single_derivation_fence_is_red_provable(tmp_path):
    """The fence FIRES on a synthetic tree carrying a second derivation —
    proving the scanner sees what it claims to see (the pgw#1049 fence
    discipline: a fence that has never gone red is not a fence)."""
    rogue = tmp_path / "rogue.py"
    rogue.write_text(
        "def my_own_key(meta):\n"
        "    return from_artifact_metadata(meta)\n")
    sites = _derivation_sites(tmp_path, "from_artifact_metadata(")
    assert sites == {"rogue.py": 1}
    # ...and comments do not trip it (the scanner reads code, not prose).
    commented = tmp_path / "commented.py"
    commented.write_text("# from_artifact_metadata( in prose\n")
    sites = _derivation_sites(tmp_path, "from_artifact_metadata(")
    assert "commented.py" not in sites


# ---------------------------------------------------------------------------
# 3. Non-collision with the pre-redefinition corpus
# ---------------------------------------------------------------------------


def _old_schema_digest(meta: dict) -> str:
    """A pre-pgw#1059 ck1 digest, reconstructed byte-for-byte from the old
    formula (8 axes incl. the fused contract digest) for the SAME artifact
    facts — the strongest possible collision candidate."""
    contract_facts = {
        "v": 3,
        # fence-symbol-exempt: this helper reconstructs the PRE-pgw#1059
        # payload byte-for-byte; renaming the dead key would make it
        # rebuild the CURRENT format and assert nothing.
        "combined_graph_hash": "0" * 16,
        "shell_digest": "",
        # The pre-pgw#1176 shape, spelled out verbatim because the whole point
        # of this helper is to reconstruct a key the tree can no longer
        # produce. It read an `entries` MAP and a `declared_envelope`; an
        # entry artifact records neither, which is itself the structural
        # reason an orphaned cell can never be re-derived.
        "targets": [str(
            (meta.get(GRAPH_CLASS_BLOCK) or {}).get("target") or "")],
        "shapes": [[1024, 1024]],
        "text_lens": [77],
        "guidance": [7.5],
        "lora_bucket": int(meta.get("lora_bucket") or 0),
        "strict": bool(meta.get("strict_export")),
    }

    def _digest16(facts: dict) -> str:
        encoded = json.dumps(
            facts, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    axes = {
        "format": str(meta["format"]),
        "kind": "aot-inductor",
        "family": meta["family"],
        "sm": meta["sm"],
        "contract": _digest16(contract_facts),
        "env_seal": _digest16(dict(meta["env_seal"])),
        "toolchain": _digest16(dict(meta["toolchain"])),
    }
    canonical = json.dumps(
        axes, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    # STAYS "ck1-": this helper deliberately reconstructs the OLD scheme, and
    # a blanket ck1->cg-key-v1 rename would have made it reconstruct the new one and
    # assert nothing.
    return "ck1-" + hashlib.sha256(canonical.encode()).hexdigest()[:56]


def test_old_and_new_keys_cannot_collide():
    """A pre-redefinition row's key can never equal a post-redefinition
    key for the same artifact — the canonical forms differ in their axis
    NAME SETS, so equal digests would require a SHA-256 collision. This is
    the verification the purge note rests on: old dev-stack rows are
    unreachable by new derivations, so the purge is hygiene."""
    meta = exported_cell_meta()
    new_key = tcg_identity.from_artifact_metadata(meta).value
    old_key = _old_schema_digest(meta)
    assert old_key != new_key
    # th#1897 puts it back on the COLLISION argument, which is the only one
    # either repo can make alone: the shared grammar refuses shape, never
    # scheme (th#1183), so both tokens are key-SHAPED and the old one simply
    # names nothing — its digest was computed over axes no current derivation
    # can restate, so an orphaned ref misses at the comparison.
    assert is_compiled_graph_key(old_key)
    assert is_compiled_graph_key(new_key)
    assert old_key != new_key


def _current_worker_meta() -> dict:
    """A current worker publish projection, keyed from its recorded facts."""
    return exported_cell_meta()


def test_pre_redefinition_artifact_is_structurally_refused():
    """An artifact recording the OLD blocks (an ``entries`` MAP and a
    ``combined_graph_hash``, the 36-entry-cell era) cannot restate a
    per-entry identity: the retained publish projection refuses typed before
    anything can move. That is what makes the ck1 corpus purge hygiene rather
    than a correctness precondition."""
    meta = _current_worker_meta()
    old = dict(meta)
    entry = old.pop(GRAPH_CLASS_BLOCK)
    old["entries"] = {entry["name"]: entry}
    # fence-symbol-exempt: the pre-atom artifact shape, on purpose — this
    # row proves a format-2 cell cannot restate a per-entry identity.
    old["combined_graph_hash"] = "0" * 16
    old["format"] = 2
    with pytest.raises(tcg_identity.IdentityError, match="graph_class"):
        tcg_identity.from_artifact_metadata(old)


def test_worker_publish_projection_matches_public_tcg_key() -> None:
    # RuntimeCompatibility("cpu") imports torch to name the target, and torch
    # is deliberately absent from this box's env (no-compile rule). The
    # ASSERTION below is untouched and CI, which has torch, is its authority;
    # an honest skip beats a hard local red that everyone learns to ignore.
    pytest.importorskip("torch")
    ingress = CallIngress(
        parameters=("sample",),
        flat_arity=1,
        inputs=(CallInput(
            "sample", 0, "sample", 0, (), "sample", "float32", (1, 4),
        ),),
    )
    declaration = GraphClassDeclaration(
        graph_class="unet/main",
        target="unet",
        graph={
            "v": 3,
            "constant_fqns": [],
            "lifted_inputs": [],
            "pytree": {"ingress": ingress.as_dict()},
            "specialization": {},
        },
        graph_witness="a" * 16,
        range_digest=ingress.digest(),
        class_dims=(("b", 1),),
    )
    toolchain = {
        "torch": "x" * 16,
        "settings_declaration": "d" * 16,
        "loaded_libs": "l" * 16,
    }
    runtime = RuntimeCompatibility("cpu", toolchain=toolchain)
    meta = {
        "kind": "aot-inductor",
        "sm": runtime.sm,
        GRAPH_CLASS_BLOCK: {
            "name": declaration.graph_class,
            "class_hash": declaration.class_hash,
        },
        "toolchain": runtime.toolchain,
    }

    assert tcg_identity.from_artifact_metadata(meta).value == str(
        runtime.key(declaration)
    )


# ---------------------------------------------------------------------------
# 4. Envelope canonicalization + the (empty) overlay slot
# ---------------------------------------------------------------------------


def test_envelope_facts_canonicalize():
    a = graph_facts.envelope_facts({
        "shapes": [[1024, 768], [768, 1024]],
        "text_lens": [77, 77, 248],
        "guidance": [7.5, 1.0],
    })
    b = graph_facts.envelope_facts({
        "shapes": [[768, 1024], [1024, 768]],
        "text_lens": [248, 77],
        "guidance": [1.0, 7.5],
    })
    assert a == b
    assert graph_facts.envelope_digest(a) == graph_facts.envelope_digest(b)


def test_overlay_slot_empty_is_absent_and_nonempty_keys():
    """Amendment 5: the behavior-posture overlay digests into the envelope
    WHEN DECLARED; the menu is empty today, so an absent/empty overlay must
    not enter the canonical form (a field that says "unchanged" must never
    re-key the fleet)."""
    base = {"shapes": [[64, 64]], "text_lens": [7], "guidance": [1.0]}
    assert "overlay" not in graph_facts.envelope_facts(base)
    assert "overlay" not in graph_facts.envelope_facts({**base, "overlay": {}})
    with_overlay = graph_facts.envelope_facts(
        {**base, "overlay": {"tf32": "off"}})
    assert with_overlay["overlay"] == {"tf32": "off"}
    assert (graph_facts.envelope_digest({**base, "overlay": {"tf32": "off"}})
            != graph_facts.envelope_digest(base))


def test_only_the_graph_rekeys_and_the_envelope_no_longer_can():
    """HALF OF THIS ROW'S SUBJECT WAS DELETED, so half of it is INVERTED —
    read this before "fixing" it.

    It used to assert that widening the declared envelope re-keys. Under
    pgw#1176 that is the disease, not the contract: `envelope_facts` digests
    the union of the ladder across the whole declaration, so a widening
    re-keyed 35 sdxl classes that traced byte-identically. Measured on
    unmodified master @ 4dfdcd60 for two byte-identical classes:
    ck1-c4c134db... -> ck1-48512ea3...

    The graph half is untouched and is the axis that MUST still move. The one
    real edge — widening a DYNAMIC dim's range genuinely changes the traced
    graph — re-keys through that class's own hash, which is the honest
    channel, rather than through a union digest that punishes its siblings.
    """
    meta = exported_cell_meta()
    key = tcg_identity.from_artifact_metadata(meta).value

    # The envelope is not an input to identity at all any more: there is no
    # `declared_envelope` on an entry artifact, and adding one changes nothing.
    wider = dict(meta)
    wider[graph_facts.EXPORT_ENVELOPE_KEY] = {
        "shapes": [[1024, 1024], [768, 768]],
        "text_lens": [77], "guidance": [7.5]}
    assert tcg_identity.from_artifact_metadata(wider).value == key

    other_graph = dict(meta)
    other_graph[GRAPH_CLASS_BLOCK] = {
        **meta[GRAPH_CLASS_BLOCK], "class_hash": "b" * 16}
    assert tcg_identity.from_artifact_metadata(other_graph).value != key


# ---------------------------------------------------------------------------
# 5. The arm token is NOT a cell key
# ---------------------------------------------------------------------------


class _Cfg:
    shapes = ((64, 64),)
    text_lens = (7,)
    guidance_scales = (1.0,)
    lora_bucket = 0


def test_arm_token_never_passes_is_key(monkeypatch):
    monkeypatch.setattr(
        fleet_cells.cc, "runtime_key",
        lambda: {"sm": "sm_89", "sku": "l4", "torch": "t", "triton": "",
                 "cuda": "", "image_digest": ""})
    monkeypatch.setattr(
        fleet_cells.cc, "toolchain_digest", lambda: (("torch", "x" * 16),))
    identity = fleet_cells.arm_identity("fam", "", 0, _Cfg())
    token = identity.token
    # pgw#1113: the scheme digit is the token's FACT SET, and the fact set
    # gained the compile SUBJECT — so the prefix moved with it, which is what
    # makes a predecessor memo row unaddressable rather than misreadable.
    assert token.startswith(fleet_cells.ARM_SCHEME + "-")
    assert not token.startswith("ck")
    # th#1897/pgw#1213: disjointness is carried by the DIGEST WIDTH now, not by
    # the prefix. The shared grammar is scheme-agnostic, so `arm2-` buys no
    # separation at all — any scheme followed by 56 hex is a key to BOTH
    # validators. A 64-hex tail is not, on either side.
    assert not is_compiled_graph_key(token)
    assert len(token.split("-", 1)[1]) == fleet_cells.ARM_DIGEST_HEX != 56
    # the compared facts are exactly the pre-trace set — graph is absent, and
    # so is `envelope`: pgw#1176 dropped it from ARM_ENVIRONMENT_FACTS because
    # a per-entry artifact records no declared envelope, so comparing it would
    # refuse every child handback by construction.
    assert set(identity.facts_dict()) == set(fleet_cells.ARM_FACTS)
    assert "envelope" not in fleet_cells.ARM_FACTS
    assert "graph" not in identity.facts_dict()
