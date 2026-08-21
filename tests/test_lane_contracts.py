"""pgw#1391: the vendored pure-Python contract reader agrees with tensorfs.

`_vendor/tensorfs/contract.py` is a PORT of the Rust validator and canonical
rendering (pgw#1310 rules the compiled extension out of a source-vendored
wheel — the same reason `planner.py` is a port, pgw#1365). A port is worth
nothing asserted, so it is pinned to the language-neutral corpus upstream
released with tensorfs#114, `spec/v1/contract-vectors`, which Rust and Go both
satisfy: every golden digest, every typed refusal.

The golden LIBRARY vectors are resolved against the DOCUMENTS THIS REPO
VENDORS, not against a copy shipped beside the corpus — so the test proves the
bytes gen-worker actually publishes, and a bad re-vendor fails here.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from gen_worker._vendor.tensorfs.contract import (
    CONTRACT_FORMAT,
    Contract,
    ContractError,
    MissingDtype,
)

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests" / "testdata" / "contract-vectors"
DOCUMENTS = ROOT / "src" / "gen_worker" / "_vendor" / "tensorfs" / "_contracts"
def _sibling_tensorfs() -> Path:
    """The neighbouring tensorfs checkout, from a worktree OR the main clone.

    The old spelling was `ROOT.parent.parent.parent / "tensorfs"`, which is
    correct for exactly one layout. From a worktree
    (`~/cozy/.worktrees/python-gen-worker/<branch>`) it lands on
    `~/cozy/tensorfs` and the drift checks run; from the MAIN checkout
    (`~/cozy/python-gen-worker`) the same three hops land on `/home/tensorfs`,
    which does not exist, so both checks SKIP. An instrument that is invisible
    exactly where reviewers and CI run it reports green by not looking.
    """

    for ancestor in ROOT.parents:
        candidate = ancestor / "tensorfs"
        if (candidate / ".git").exists():
            return candidate
    return ROOT.parent / "tensorfs"  # a stable non-existent path -> honest skip


UPSTREAM = _sibling_tensorfs()

CORPUS = json.loads((VECTORS / "contract-vectors.json").read_text())


def _document_of(case: dict) -> str:
    """A golden vector's document: inline for the anonymous customs, and for a
    library entry the VENDORED file of that name."""

    if "document" in case:
        return case["document"]
    return (DOCUMENTS / Path(case["file"]).name).read_text(encoding="utf-8")


def test_the_corpus_is_the_shape_this_test_assumes() -> None:
    assert CORPUS["format"] == "tensorfs-contract-vectors-v1"
    assert len(CORPUS["golden"]) >= 13
    assert len(CORPUS["refusals"]) >= 19


@pytest.mark.parametrize("case", CORPUS["golden"], ids=lambda case: case["name"])
def test_every_golden_vector_digests_identically(case: dict) -> None:
    """THE proof the canonical rendering is byte-identical to Rust's: a
    SHA-256 over it cannot agree by luck."""

    contract = Contract.from_document(_document_of(case))
    assert contract.digest == case["digest"]
    assert contract.stamp == case["stamp"]


@pytest.mark.parametrize("case", CORPUS["refusals"], ids=lambda case: case["name"])
def test_every_refusal_vector_refuses_with_the_shared_reason(case: dict) -> None:
    with pytest.raises(ContractError) as caught:
        Contract.from_document(case["document"])
    assert caught.value.reason == case["reason"]


def test_the_vendored_documents_are_exactly_the_corpus_library() -> None:
    """No document may be added to the vendored library without a golden
    vector pinning its digest — otherwise a hand-edited contract publishes."""

    vendored = {path.name for path in DOCUMENTS.glob("*.json")}
    pinned = {Path(case["file"]).name for case in CORPUS["golden"] if "file" in case}
    assert vendored == pinned


def test_the_library_loads_and_names_what_the_corpus_names() -> None:
    from gen_worker._vendor.tensorfs import contracts

    assert {contract.stamp for contract in contracts.all()} == {
        case["stamp"] for case in CORPUS["golden"] if "file" in case
    }
    assert contracts.SDXL_DIFFUSERS_BF16.dtype == "bfloat16"
    # tensorfs#121 authored the four se#757 blocker-A documents, so these now
    # resolve where they used to be MissingContract sentinels.
    for constant in (
        "SD15_DIFFUSERS_BF16",
        "SD2_DIFFUSERS_BF16",
        "HIDREAM_O1_DIFFUSERS_BF16",
        "WAN22_DIFFUSERS_BF16",
        "FLUX2_KLEIN_DIFFUSERS_BF16",
        # tensorfs#136 — the last of the five sentinels to clear.
        "FLUX1_DIFFUSERS_BF16",
    ):
        assert getattr(contracts, constant).dtype == "bfloat16"
    assert contracts.SD15_DIFFUSERS_BF16.digest != contracts.SD2_DIFFUSERS_BF16.digest
    with pytest.raises(AttributeError, match="no contract constant"):
        contracts.NOPE_NOT_A_DOCUMENT  # noqa: B018 -- the refusal IS the assertion


def test_a_document_declaring_no_dtype_raises_rather_than_answering_none() -> None:
    """`release/derive.py` and `Model.__init_subclass__` both rely on this
    being a RAISE: a `None` would be read as "no opinion" and travel.

    `dit.blocks-fused-qkv@1` is a FRAGMENT — a family-shared block spelling no
    `lanes=` header names on its own — so it is deliberately dtype-less
    upstream and is the standing example.
    """

    from gen_worker._vendor.tensorfs import contracts

    with pytest.raises(MissingDtype):
        contracts.DIT_BLOCKS_FUSED_QKV.dtype


def test_the_h3_dtype_waiver_deletes_itself_when_upstream_lands_the_dtype() -> None:
    """pgw#1391 ordering waiver, SELF-DELETING BY CONSTRUCTION — AND IT DID.

    The waiver briefly held `minimax.h3-dit-diffusers@1`, which was live on
    deploy lane `ab56185761f1597f1` while its document declared no top-level
    dtype. tensorfs#121 gave it `bfloat16`; re-vendoring made this test fail,
    and the entry was deleted. That is the fence working, not a hypothetical.

    It stays because the next such ordering hazard gets the same treatment: a
    waiver is a fact about the vendored document, never a preference, so it
    cannot outlive its reason.
    """

    from gen_worker.serving.model import DTYPELESS_UPSTREAM_LANES

    still_dtypeless = set()
    for path in DOCUMENTS.glob("*.json"):
        contract = Contract.from_document(path.read_text(encoding="utf-8"))
        try:
            contract.dtype
        except MissingDtype:
            still_dtypeless.add(contract.stamp)

    assert not DTYPELESS_UPSTREAM_LANES or DTYPELESS_UPSTREAM_LANES <= still_dtypeless, (
        "a vendored contract GAINED its top-level dtype: delete it from "
        "DTYPELESS_UPSTREAM_LANES — the waiver's whole reason is gone "
        f"(now declared: {sorted(DTYPELESS_UPSTREAM_LANES - still_dtypeless)})"
    )


def test_the_document_property_is_the_canonical_json_string() -> None:
    """`release/derive.py::_contract_document` parses this; a dict would have
    shipped `"document": null` on every lane."""

    from gen_worker._vendor.tensorfs import contracts

    document = contracts.SDXL_DIFFUSERS_BF16.document
    assert isinstance(document, str)
    parsed = json.loads(document)
    assert parsed["format"] == CONTRACT_FORMAT
    assert parsed["name"] == "sdxl.diffusers-bf16"
    assert document == json.dumps(parsed, sort_keys=True, separators=(",", ":"))


def _upstream_blob(path: str) -> bytes | None:
    """One upstream file AT the pinned `rev`, or None if unavailable.

    THE REV, NOT THE SIBLING'S WORKING TREE, and the difference is the whole
    point of pinning one. Both drift checks below used to read
    `UPSTREAM / <path>` — whatever the neighbouring checkout happened to have
    checked out — which asserts that this repo is always at tensorfs TIP. That
    contradicts the pin existing at all, and it goes red for
    reasons that have nothing to do with the change under test: while tensorfs
    is under active multi-lane development its master gains a lane document
    every few hours, so every vendor PR inherits an unrelated failure and the
    honest fix looks like "vendor three other lanes' documents".

    A snapshot is faithful to the REV IT RECORDS. That is what is asserted.
    """

    import subprocess

    if not (UPSTREAM / ".git").exists():
        return None
    manifest = tomllib.loads(
        (ROOT / "src" / "gen_worker" / "_vendor" / "VENDORED.toml").read_text()
    )
    rev = manifest["packages"]["tensorfs"]["rev"]
    done = subprocess.run(
        ["git", "-C", str(UPSTREAM), "show", f"{rev}:{path}"],
        capture_output=True,
    )
    # An unfetched rev is "cannot check here", never "drifted": the sibling is
    # a convenience, and CI has no sibling at all.
    return done.stdout if done.returncode == 0 else None


def test_the_corpus_matches_the_pinned_upstream_rev() -> None:
    """The vendored corpus is a SNAPSHOT; when the real repo is beside us,
    prove the snapshot has not drifted from the rev it claims."""

    theirs = _upstream_blob("spec/v1/contract-vectors/contract-vectors.json")
    if theirs is None:
        pytest.skip("no sibling tensorfs checkout carrying the pinned rev")
    ours = (VECTORS / "contract-vectors.json").read_bytes()
    assert hashlib.sha256(theirs).hexdigest() == hashlib.sha256(ours).hexdigest()


def test_the_vendored_documents_match_the_pinned_upstream_rev() -> None:
    checked = 0
    for path in DOCUMENTS.glob("*.json"):
        theirs = _upstream_blob(f"spec/v1/contracts/{path.name}")
        if theirs is None:
            pytest.skip("no sibling tensorfs checkout carrying the pinned rev")
        assert path.read_bytes() == theirs, f"{path.name} drifted from upstream"
        checked += 1
    # A loop that checked nothing is not a passing drift check.
    assert checked == len(list(DOCUMENTS.glob("*.json"))) >= 14


def test_the_contract_plane_is_digest_fenced_at_the_one_rev() -> None:
    """pgw#1575: the contract plane has no rev of its own any more.

    It used to, because the snapshot's `rev` was a lineage that predated
    contracts entirely. One master-ancestor rev now supplies every plane, so
    the only thing left to assert is that these files are in the SINGLE digest
    inventory rather than sitting on disk unfenced — which is the failure that
    third-document drift actually took.
    """

    manifest = tomllib.loads(
        (ROOT / "src" / "gen_worker" / "_vendor" / "VENDORED.toml").read_text()
    )
    spec = manifest["packages"]["tensorfs"]
    assert len(spec["rev"]) == 40
    recorded = set(spec["files"])
    assert {"contract.py", "contracts.py"} <= recorded
    assert {f"_contracts/{path.name}" for path in DOCUMENTS.glob("*.json")} <= recorded


def _fixture_lane():
    """A minimal but REAL lane declaration (pgw#1599). These tests are about
    the CONTRACT half of the header, so the demand formula is deliberately
    the simplest legal one."""

    from gen_worker import lane
    from gen_worker.demand import GiB, const

    return lane(request=const(GiB(1)))


def test_a_dtypeless_FRAGMENT_is_importable_but_refuses_as_a_serve_lane() -> None:
    """pgw#1393's `dit.blocks-fused-qkv@1` is the standing case, and both
    halves matter.

    It is a real library document, so `gen_worker.models` must import. But it
    declares no top-level dtype BECAUSE IT IS A FRAGMENT (a family-plural
    block spelling), and it matches zero of the 169 tensors in the real Klein
    transformer header. So claiming it as a serve lane must refuse — the
    refusal is the point, not a gap to be waived. pgw#1599 removes the OTHER
    half of the old hazard entirely: there is no longer an implicit borrow to
    claim a fragment by accident, because `lanes=` is required and explicit.
    """

    from gen_worker.models import SDXL
    from gen_worker.models.model_types import DIT_BLOCKS_FUSED_QKV
    from gen_worker.serving import Model, ModelDeclarationError

    assert DIT_BLOCKS_FUSED_QKV.stamp == "dit.blocks-fused-qkv@1"
    with pytest.raises(MissingDtype):
        DIT_BLOCKS_FUSED_QKV.dtype

    with pytest.raises(ModelDeclarationError, match="declares no load dtype"):

        class ExplicitFragment(
            Model[SDXL], lanes={DIT_BLOCKS_FUSED_QKV: _fixture_lane()}
        ):
            def load(self, ctx: object) -> None: ...


def test_the_two_flux_roots_resolve_to_their_own_documents() -> None:
    """tensorfs#124, both halves, and the reason the sentinel machinery stayed.

    `flux1.diffusers-bf16@1` was a `MissingContract` from pgw#1393 until
    tensorfs#136 authored it, and clearing it took NO code change in
    `model_types.py` — vendoring the document was the whole edit. That is the
    property the sentinel shape exists to have, and this is the second time it
    has been exercised (tensorfs#121 cleared four).

    Before either document existed, FLUX.1's declared lane was
    `dit.blocks-fused-qkv@1`, which declares `blocks.{i}.attn.qkv.weight`
    REQUIRED and matches 0 tensors in a real Flux transformer header (the tree
    is `transformer_blocks.*` / `single_transformer_blocks.*`) — a GUARANTEED
    refusal wearing a resolved lane's clothes, the pgw#1391 bug exactly. The
    quieter hazard, measured upstream when the document was authored, was the
    OTHER flux document: `flux2-klein.diffusers-bf16@1` explains 308 of a
    FLUX.1 transformer's 1160 tensors with no dtype or rank refusal, so it won
    every FLUX.1 file outright. Two roots, two documents, and neither may
    borrow the other's.
    """

    from gen_worker.models import Flux1, Flux2Klein
    from gen_worker.models.model_types import (
        FLUX1_DIFFUSERS_BF16,
        FLUX2_KLEIN_DIFFUSERS_BF16,
    )
    from gen_worker.serving import Model, model_declared_lanes

    assert FLUX1_DIFFUSERS_BF16.stamp == "flux1.diffusers-bf16@1"
    assert FLUX1_DIFFUSERS_BF16.dtype == "bfloat16"
    assert FLUX2_KLEIN_DIFFUSERS_BF16.stamp == "flux2-klein.diffusers-bf16@1"
    assert FLUX2_KLEIN_DIFFUSERS_BF16.dtype == "bfloat16"
    # Separate documents, separate digests — never one shared "flux" lane.
    assert FLUX1_DIFFUSERS_BF16.digest != FLUX2_KLEIN_DIFFUSERS_BF16.digest

    # pgw#1599: each root NAMES its own document. There is no canonical
    # borrow left to get wrong — the header is the only source.
    class Flux1Model(Model[Flux1], lanes={FLUX1_DIFFUSERS_BF16: _fixture_lane()}):
        def load(self, ctx: object) -> None: ...

    class KleinModel(
        Model[Flux2Klein], lanes={FLUX2_KLEIN_DIFFUSERS_BF16: _fixture_lane()}
    ):
        def load(self, ctx: object) -> None: ...

    assert Flux1Model.__cozy_model_type__ is Flux1
    assert KleinModel.__cozy_model_type__ is Flux2Klein
    assert [row.contract_id for row in model_declared_lanes(Flux1Model)] == [
        "flux1.diffusers-bf16@1"
    ]
    assert [row.contract_id for row in model_declared_lanes(KleinModel)] == [
        "flux2-klein.diffusers-bf16@1"
    ]


def test_a_lane_naming_no_document_refuses_at_declaration() -> None:
    """The seam the DISCOVERY guarantee rides on now that pgw#1394 has removed
    lane enumeration from `discovery/entrypoints_v2.py`: discovery imports the
    author module, so `__init_subclass__` is what refuses."""

    from gen_worker.models import SDXL
    from gen_worker.models.model_types import MissingContract
    from gen_worker.serving import Model, ModelDeclarationError

    nope = MissingContract("nope.not-a-document", 1)
    with pytest.raises(ModelDeclarationError, match="DOES NOT EXIST"):

        class Claimed(Model[SDXL], lanes={nope: _fixture_lane()}):
            def load(self, ctx: object) -> None: ...

    # `requires=` is deleted and the refusal names the spelling that replaces
    # it — a silently-ignored floor is exactly what this class of bug was.
    with pytest.raises(ModelDeclarationError, match="requires.*is DELETED"):

        class Floored(Model[SDXL], requires={"nope.not-a-document@1": "vram12g"}):
            def load(self, ctx: object) -> None: ...

    # ...and the same missing document refuses through the merged spelling,
    # so folding the kwargs did not cost this fence anything.
    with pytest.raises(ModelDeclarationError, match="DOES NOT EXIST"):

        class ClaimedWithSpec(Model[SDXL], lanes={nope: _fixture_lane()}):
            def load(self, ctx: object) -> None: ...


def test_no_vendored_stamp_ties_between_two_model_types() -> None:
    """tensorfs#124 warned that eleven documents sharing component spellings
    could make family detection TIE. It does not, and this pins it.

    `model_type_for_contract` returns the FIRST matching type, so a tie would
    be resolved by declaration order — a silent, order-dependent
    classification. `dit.blocks-fused-qkv@1` matching nothing is the correct
    answer for a family-plural fragment: unclassified is legal and visible.
    """

    from fnmatch import fnmatchcase

    from gen_worker._vendor.tensorfs import contracts
    from gen_worker.models.model_types import MODEL_TYPES

    ties = {}
    for contract in contracts.all():
        hits = [
            mt.name
            for mt in MODEL_TYPES
            if any(fnmatchcase(contract.stamp, pattern) for pattern in mt.contracts)
        ]
        if len(hits) > 1:
            ties[contract.stamp] = hits
    assert not ties, f"stamps claimed by more than one model type: {ties}"


def test_a_dtypeless_lane_cannot_buy_a_silent_runs_anywhere_floor() -> None:
    """pgw#1404: the derivation FAILS CLOSED on a contract with no dtype.

    The h3 lane audited all eleven vendored documents against the new
    dtype->min_sm derivation and found three with no top-level dtype —
    `dit.blocks-fused-qkv@1` and both `sdxl.clip-g-*-qkv@1`. They are
    component/sub-tree layouts, so they arguably never belong in a `lanes=` at
    all — but "arguably never" is an assumption, not a fence.

    Note how the hazard MUTATES under the derivation, which is why this test
    exists rather than a comment: a dtypeless contract used to be a LOUD load
    crash (`MissingDtype`). Deriving 0 from it would turn the same gap into a
    SILENT floor of none, which the resolver reads as "runs anywhere" —
    th#1754's shape with a new cause, and strictly worse because nothing
    raises. A floor is the one place failing open is invisible.
    """

    from gen_worker._vendor.tensorfs import contracts
    from gen_worker.models.model_types import SDXL
    from gen_worker.serving import Model, ModelDeclarationError
    from gen_worker.serving import model as model_module

    dtypeless = [c for c in contracts.all() if getattr(c, "_dtype", None) is None]
    assert {c.stamp for c in dtypeless} == {
        "dit.blocks-fused-qkv@1",
        "sdxl.clip-g-fused-qkv@1",
        "sdxl.clip-g-split-qkv@1",
    }, "the dtypeless set moved; re-audit the derivation against it"

    for contract in dtypeless:
        with pytest.raises(ModelDeclarationError, match="no load dtype"):

            class Floored(Model[SDXL], lanes={contract: _fixture_lane()}):  # noqa: B903
                def load(self, ctx: object) -> None: ...

    # ...and the one path that could still buy silence is closed too.
    # `lane_dtype` answers None instead of raising for a handle listed in
    # `DTYPELESS_UPSTREAM_LANES` (an empty escape hatch today). If anything is
    # ever added to it, the lane would clear declaration AND derive no floor.
    # A floor declaration refuses on the dtype itself, so the hatch cannot
    # reopen this hole behind someone's back.
    original = model_module.DTYPELESS_UPSTREAM_LANES
    try:
        model_module.DTYPELESS_UPSTREAM_LANES = frozenset({"sdxl.clip-g-fused-qkv@1"})
        with pytest.raises(ModelDeclarationError, match="no load dtype"):

            class Hatched(
                Model[SDXL],
                lanes={contracts.SDXL_CLIP_G_FUSED_QKV: _fixture_lane()},
            ):
                def load(self, ctx: object) -> None: ...
    finally:
        model_module.DTYPELESS_UPSTREAM_LANES = original
