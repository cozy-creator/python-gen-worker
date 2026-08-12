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
4. The stamp discipline: the key an artifact carries is the key its own
   recorded facts describe, proven at admission (``verify_contract``).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from gen_worker import aot_serve, cell_key, fleet_cells

from harness.cell_meta import exported_cell_meta

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "gen_worker"


# ---------------------------------------------------------------------------
# 1. The membership axiom
# ---------------------------------------------------------------------------


def test_membership_axiom_the_key_is_exactly_three_axes():
    """The ek1 key contains exactly {graph, sm, toolchain}.

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
    assert cell_key._REQUIRED == ("graph", "sm", "toolchain")
    assert cell_key._OPTIONAL == ()


def test_adding_an_axis_refuses_with_the_axiom_named():
    axes = {
        "graph": "g" * 16, "sm": "sm_89", "toolchain": "t" * 16, "sku": "l4",
    }
    with pytest.raises(cell_key.CellKeyError) as exc:
        cell_key.from_axes(axes)
    assert "membership axiom" in str(exc.value)
    assert "pgw#1059" in str(exc.value)


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
    with pytest.raises(cell_key.CellKeyError, match="unknown cell-key axis"):
        cell_key.from_axes(axes)


def test_missing_axis_is_a_typed_refusal():
    with pytest.raises(cell_key.CellKeyError, match="requires axes"):
        cell_key.from_axes({"graph": "g" * 16, "sm": "sm_89"})


# ---------------------------------------------------------------------------
# 2. One derivation per axis — the fence, and its RED proof
# ---------------------------------------------------------------------------

#: Call sites allowed to COMPUTE each single-derivation surface, as
#: (module-relative path, count-limit-free) pairs. Everything else in
#: src/gen_worker must READ the stamped value instead of re-deriving it.
#: Extending an allowlist is a conscious act reviewed against the module
#: docstrings of cell_key/aot_serve — never a drive-by.
_DERIVATION_ALLOWLIST = {
    # the traced-graph digest: stamped once, proven once at admission.
    "combined_graph_hash(": {
        "aot_serve.py",   # def + artifact_metadata stamp + verify_contract
    },
    # the exported-cell key: mint stamp, publish recompute, admission proof.
    "from_exported_artifact_metadata(": {
        "cell_key.py",    # def
        "aot_mint.py",    # cell_identity (the stamp)
        "fleet_cells.py",  # _recomputed_key (the publish recompute)
        "aot_serve.py",   # verify_contract (the admission proof)
        # pgw#1089 (§4.27 step 1): the BOOT-side derivation. Added
        # consciously, and it is the reason the fence is an allowlist rather
        # than a ban: the boot key must be THE cell key, so the fourth site
        # had to be this function and not a fourth arithmetic. `boot_key.fold`
        # assembles the mint's own entry blocks, stamps them through
        # `aot_serve.artifact_metadata` (which is where `combined_graph_hash`
        # and every `class_hash` are computed — note `boot_key.py` is NOT in
        # the `combined_graph_hash(` allowlist below, and must not be), and
        # asks THIS function for the key. A boot derivation that computed the
        # digest itself would be attempt 28 exactly: a declared-facts key
        # beside a traced-facts key under one axis name.
        "boot_key.py",
    },
    # the declared-envelope digest.
    "envelope_digest(": {
        "cell_key.py",    # def (+ envelope_facts)
        "fleet_cells.py",  # arm_identity + arm_axis_divergence (same derivation
                           # both sides of the handback seam — the pgw#1042 check)
    },
    # the toolchain-axis digest, i.e. its MEMBERSHIP (pgw#1050). The producer
    # (`compile_cache.toolchain_digest`) collects components; this is the one
    # place that says which of them ARE the axis, and every reader that
    # restates the axis from a recorded block must come through it or the
    # two ends can disagree about membership.
    "toolchain_axis_digest(": {
        "cell_key.py",     # def (+ toolchain_facts)
        "fleet_cells.py",  # arm_identity + arm_axis_divergence
        "aot_identity.py",  # artifact_identity (the wire fact)
    },
}


def _derivation_sites(root: Path, needle: str) -> dict[str, int]:
    """{relative_path: occurrence_count} of ``needle`` in ``root``'s .py files,
    comments and docstrings excluded crudely by requiring the needle outside a
    line starting with ``#``."""
    sites: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
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
        f"allowlist consciously in tests/test_cell_key_pgw1059.py")


def test_single_derivation_fence_is_red_provable(tmp_path):
    """The fence FIRES on a synthetic tree carrying a second derivation —
    proving the scanner sees what it claims to see (the pgw#1049 fence
    discipline: a fence that has never gone red is not a fence)."""
    rogue = tmp_path / "rogue.py"
    rogue.write_text(
        "def my_own_key(meta):\n"
        "    return from_exported_artifact_metadata(meta)\n")
    sites = _derivation_sites(tmp_path, "from_exported_artifact_metadata(")
    assert sites == {"rogue.py": 1}
    # ...and comments do not trip it (the scanner reads code, not prose).
    commented = tmp_path / "commented.py"
    commented.write_text("# from_exported_artifact_metadata( in prose\n")
    sites = _derivation_sites(tmp_path, "from_exported_artifact_metadata(")
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
        "combined_graph_hash": "0" * 16,
        "shell_digest": "",
        # The pre-pgw#1176 shape, spelled out verbatim because the whole point
        # of this helper is to reconstruct a key the tree can no longer
        # produce. It read an `entries` MAP and a `declared_envelope`; an
        # entry artifact records neither, which is itself the structural
        # reason an orphaned cell can never be re-derived.
        "targets": [str(
            (meta.get(cell_key.ENTRY_BLOCK_KEY) or {}).get("target") or "")],
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
    return "ek1-" + hashlib.sha256(canonical.encode()).hexdigest()[:56]


def test_old_and_new_keys_cannot_collide():
    """A pre-redefinition row's key can never equal a post-redefinition
    key for the same artifact — the canonical forms differ in their axis
    NAME SETS, so equal digests would require a SHA-256 collision. This is
    the verification the purge note rests on: old dev-stack rows are
    unreachable by new derivations, so the purge is hygiene."""
    meta = exported_cell_meta()
    new_key = cell_key.from_entry_metadata(meta).digest
    old_key = _old_schema_digest(meta)
    assert old_key != new_key
    # pgw#1176 makes this STRUCTURAL rather than a collision argument: a ck1
    # key does not even parse as an entry key, so an orphaned cell ref cannot
    # reach a per-entry code path and fail late — it fails at the comparison.
    assert not cell_key.is_key(old_key)
    assert cell_key.is_key(new_key)


def _admissible_meta() -> dict:
    """A fully self-consistent ENTRY metadata: the block stamped by the ONE
    producer (``aot_serve.entry_metadata``), identity blocks recorded, key
    stamped from the recorded facts."""
    meta = aot_serve.entry_metadata(
        family="fam", precision="bf16", cell_key="", name="unet/main",
        entry={
            "target": "unet",
            "fork": [],
            "class_dims": [["b", 1]],
            "inputs": [{"name": "sample", "position": 0, "dtype": "float16",
                        "shape": [1, 4, 64, 64]}],
            "symbols": {},
            "constants": [],
            "graph": {"v": 2},
        })
    meta.update({
        "sm": "sm_89",
        "weight_lane": "",
        "env_seal": {"seal_v": 4, "env": {"PYTHONHASHSEED": "0"}},
        "toolchain": {"torch": "x" * 16, "settings_declaration": "d" * 16,
                      "loaded_libs": "l" * 16},
    })
    meta["cell_key"] = cell_key.from_entry_metadata(meta).digest
    return meta


def test_pre_redefinition_artifact_is_structurally_refused():
    """An artifact recording the OLD blocks (an ``entries`` MAP and a
    ``combined_graph_hash``, the 36-entry-cell era) cannot restate a
    per-entry identity: the key derivation refuses typed, and admission
    (``verify_contract``) turns the unrestatable stamp into a named refusal
    before anything can arm. That is what makes the ck1 corpus purge hygiene
    rather than a correctness precondition."""
    meta = _admissible_meta()
    old = dict(meta)
    entry = old.pop(cell_key.ENTRY_BLOCK_KEY)
    old["entries"] = {entry["name"]: entry}
    old["combined_graph_hash"] = "0" * 16
    old["format"] = 2
    with pytest.raises(cell_key.CellKeyError, match="entry"):
        cell_key.from_entry_metadata(old)
    # its (old-formula) stamp is present but unrestatable => admission refuses.
    # An ek-shaped stamp, because a ck1 stamp is not an identity CLAIM to this
    # runtime at all — `is_key` refuses it, so it never reaches the restate.
    old["cell_key"] = "ek1-" + "5" * 56
    reason = aot_serve.verify_contract(old)
    assert "malformed declared contract" in reason or "not restatable" in reason


def test_forged_stamp_is_refused_at_admission():
    meta = _admissible_meta()
    meta["cell_key"] = "ek1-" + "0" * 56
    reason = aot_serve.verify_contract(meta)
    assert "recorded facts describe" in reason


def test_true_stamp_passes_admission():
    assert aot_serve.verify_contract(_admissible_meta()) == ""


# ---------------------------------------------------------------------------
# 4. Envelope canonicalization + the (empty) overlay slot
# ---------------------------------------------------------------------------


def test_envelope_facts_canonicalize():
    a = cell_key.envelope_facts({
        "shapes": [[1024, 768], [768, 1024]],
        "text_lens": [77, 77, 248],
        "guidance": [7.5, 1.0],
    })
    b = cell_key.envelope_facts({
        "shapes": [[768, 1024], [1024, 768]],
        "text_lens": [248, 77],
        "guidance": [1.0, 7.5],
    })
    assert a == b
    assert cell_key.envelope_digest(a) == cell_key.envelope_digest(b)


def test_overlay_slot_empty_is_absent_and_nonempty_keys():
    """Amendment 5: the behavior-posture overlay digests into the envelope
    WHEN DECLARED; the menu is empty today, so an absent/empty overlay must
    not enter the canonical form (a field that says "unchanged" must never
    re-key the fleet)."""
    base = {"shapes": [[64, 64]], "text_lens": [7], "guidance": [1.0]}
    assert "overlay" not in cell_key.envelope_facts(base)
    assert "overlay" not in cell_key.envelope_facts({**base, "overlay": {}})
    with_overlay = cell_key.envelope_facts(
        {**base, "overlay": {"tf32": "off"}})
    assert with_overlay["overlay"] == {"tf32": "off"}
    assert (cell_key.envelope_digest({**base, "overlay": {"tf32": "off"}})
            != cell_key.envelope_digest(base))


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
    key = cell_key.from_entry_metadata(meta).digest

    # The envelope is not an input to identity at all any more: there is no
    # `declared_envelope` on an entry artifact, and adding one changes nothing.
    wider = dict(meta)
    wider[cell_key.EXPORT_ENVELOPE_KEY] = {
        "shapes": [[1024, 1024], [768, 768]],
        "text_lens": [77], "guidance": [7.5]}
    assert cell_key.from_entry_metadata(wider).digest == key

    other_graph = dict(meta)
    other_graph[cell_key.ENTRY_BLOCK_KEY] = {
        **meta[cell_key.ENTRY_BLOCK_KEY], "class_hash": "b" * 16}
    assert cell_key.from_entry_metadata(other_graph).digest != key


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
    assert not cell_key.is_key(token)
    # the compared facts are exactly the pre-trace set — graph is absent, and
    # so is `envelope`: pgw#1176 dropped it from ARM_ENVIRONMENT_FACTS because
    # a per-entry artifact records no declared envelope, so comparing it would
    # refuse every child handback by construction.
    assert set(identity.facts_dict()) == set(fleet_cells.ARM_FACTS)
    assert "envelope" not in fleet_cells.ARM_FACTS
    assert "graph" not in identity.facts_dict()
