#!/usr/bin/env python3
"""pgw#1391 probe: does a lane whose contract document does not exist REFUSE?

Three parts, run against whatever tree it is invoked from:

  A. THE FENCE STILL BITES — a ``Model`` subclass whose lane names a document
     that genuinely does not exist must refuse, at DECLARATION (which is what
     carries the discovery guarantee: discovery imports the author module).
  B. IT BITES ONLY WHAT DESERVES IT — ``sd15`` was the se#757 example and its
     document now EXISTS (tensorfs#121), so it must RESOLVE. Same for the
     sdxl and minimax-h3 controls, both live deployed lanes.
  C. THE CORPUS — every golden vector in tensorfs's shared conformance corpus
     must digest identically here, and every refusal vector must refuse.

Discovery no longer enumerates lanes at all (pgw#1394 deleted ``_lane_stamps``
because lane stamps were being written into the hub's ARTIFACT-layout fields).
The refusal therefore rides on ``Model.__init_subclass__``, which is strictly
better: it cannot be bypassed by a consumer that does not happen to enumerate
lanes, and it fires on the author's machine at import.

Run: ``python scripts/probe_lane_contracts.py``
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show(label: str, thunk, *, brief: bool = False) -> None:
    try:
        print(f"  {label} -> {thunk()!r}")
    except Exception as exc:  # noqa: BLE001 -- the probe reports, never raises
        message = str(exc)
        if brief and len(message) > 96:
            message = message[:96] + " ...[refusal text elided]"
        print(f"  {label} -> {type(exc).__name__}: {message}")


# ── A. the silent lie ────────────────────────────────────────────────────────

rule("A. THE FENCE BITES — a lane naming a document that does not exist")

from gen_worker.models import SD15, SDXL  # noqa: E402
from gen_worker.models.model_types import MissingContract  # noqa: E402
from gen_worker.release import derive as _derive  # noqa: E402
from gen_worker.serving import Model, lane_handle, model_lanes  # noqa: E402

#: An invented stamp: tensorfs ships no such document and never will.
NOPE = MissingContract("nope.not-a-document", 1)

print("  -- declaration time (the seam discovery rides on) --")


def _declare_explicit():
    class NopeExplicit(Model[SDXL], lanes=(NOPE,)):
        def load(self, ctx) -> None:  # noqa: ANN001
            pass

    return NopeExplicit


show("class NopeExplicit(Model[SDXL], lanes=(NOPE,))", _declare_explicit)


def _declare_requires():
    class NopeRequires(Model[SDXL], requires={"nope.not-a-document@1": "vram12g"}):
        def load(self, ctx) -> None:  # noqa: ANN001
            pass

    return NopeRequires


show("class NopeRequires(..., requires={bogus stamp})", _declare_requires, brief=True)

print("\n  -- the publish path (release/derive.py) --")
warnings: list[str] = []
show("_contract_document('NopeModel', NOPE, warnings)",
     lambda: _derive._contract_document("NopeModel", NOPE, warnings), brief=True)
print(f"  warnings collected -> {warnings!r}")
show("_contract_digest(NOPE)", lambda: _derive._contract_digest(NOPE), brief=True)
show("_resolve_lane(torchcg, NopeModel, NOPE)", lambda: _derive._resolve_lane(
    __import__("gen_worker._vendor.torchcg", fromlist=["x"]), SDXL, NOPE
), brief=True)

print("\n  -- discovery imports the author module, so the class never builds --")
show("model_lanes over a class that could not be declared",
     lambda: "unreachable: __init_subclass__ already refused")

print("\n  -- the SAME fence on a REAL type: Flux1 (tensorfs#124 owes its doc) --")
from gen_worker.models import Flux1, Flux2Klein  # noqa: E402


def _declare_flux1():
    class Flux1Model(Model[Flux1]):
        def load(self, ctx) -> None:  # noqa: ANN001
            pass

    return Flux1Model


show("Flux1.canonical_contract", lambda: Flux1.canonical_contract)
show("class Flux1Model(Model[Flux1])  # lanes= omitted", _declare_flux1, brief=True)
print("  (it used to point at dit.blocks-fused-qkv@1, which matches 0 of 169")
print("   real Flux tensors — a guaranteed refusal dressed as a resolved lane)")


def _declare_klein():
    class KleinModel(Model[Flux2Klein]):
        def load(self, ctx) -> None:  # noqa: ANN001
            pass

    return KleinModel


show("Flux2Klein.canonical_contract.stamp", lambda: Flux2Klein.canonical_contract.stamp)
show("Flux2Klein.canonical_contract.dtype", lambda: Flux2Klein.canonical_contract.dtype)
show("class KleinModel(Model[Flux2Klein])  # lanes= omitted",
     lambda: _declare_klein().__name__)


# ── A2. sd15 now RESOLVES — the fence bites only what deserves it ────────────

rule("A2. sd15 — se#757's example, whose document tensorfs#121 authored")


class Sd15Model(Model[SD15]):
    """Exactly what a wave-2 migration lane writes: no ``lanes=`` at all."""

    def load(self, ctx) -> None:  # noqa: ANN001
        pass


print("  class Sd15Model(Model[SD15]) declared with NO lanes= ... imported OK")
show("model_lanes(Sd15Model)", lambda: [lane_handle(x) for x in model_lanes(Sd15Model)])
for attribute in ("stamp", "dtype", "digest"):
    show(f"canonical_contract.{attribute}", lambda a=attribute: getattr(SD15.canonical_contract, a))
warnings = []
show("_contract_document('Sd15Model', lane, warnings) keys",
     lambda: sorted(_derive._contract_document("Sd15Model", SD15.canonical_contract, warnings)))
print(f"  warnings collected -> {warnings!r}")
show("_resolve_lane(torchcg, Sd15Model, lane)", lambda: _derive._resolve_lane(
    __import__("gen_worker._vendor.torchcg", fromlist=["x"]), Sd15Model, SD15.canonical_contract
))


# ── B. the sdxl control ──────────────────────────────────────────────────────

rule("B. sdxl control — the live deployed lane must keep working")


class SdxlModel(Model[SDXL]):
    def load(self, ctx) -> None:  # noqa: ANN001
        pass


show("model_lanes(SdxlModel)", lambda: [lane_handle(x) for x in model_lanes(SdxlModel)])
for attribute in ("stamp", "dtype", "digest"):
    show(f"canonical_contract.{attribute}", lambda a=attribute: getattr(SDXL.canonical_contract, a))
show(
    "canonical_contract.document is a real dict/str",
    lambda: type(getattr(SDXL.canonical_contract, "document")).__name__,
)
warnings = []
document = None
try:
    document = _derive._contract_document("SdxlModel", SDXL.canonical_contract, warnings)
except Exception as exc:  # noqa: BLE001
    print(f"  _contract_document -> {type(exc).__name__}: {exc}")
print(f"  _contract_document keys -> {sorted(document) if document else document!r}")
print(f"  warnings -> {warnings!r}")
show("_contract_digest(lane)", lambda: _derive._contract_digest(SDXL.canonical_contract))

#: tensorfs#121 re-declared this document COMPLETE over the shipped
#: checkpoint header, so it re-digested from f1455f56... at tensorfs c3a831d.
PINNED_SDXL_DIGEST = "ef01dd65f57bd95ae05d70f5a9893e9abab6b4f0831b05c4edf68ae9ebb148e8"
show(
    "digest == tensorfs pinned ef01dd65...",
    lambda: _derive._contract_digest(SDXL.canonical_contract) == f"sha256:{PINNED_SDXL_DIGEST}",
)


# ── B2. the minimax-h3 control (the dtype-ordering waiver) ───────────────────

rule("B2. minimax-h3 control — LIVE, dtype gained in tensorfs#121")

from gen_worker.models import MiniMaxH3  # noqa: E402


class H3Model(Model[MiniMaxH3]):
    def load(self, ctx) -> None:  # noqa: ANN001
        pass


show("model_lanes(H3Model)", lambda: [lane_handle(x) for x in model_lanes(H3Model)])
show("canonical_contract.digest", lambda: MiniMaxH3.canonical_contract.digest)
show("canonical_contract.dtype", lambda: MiniMaxH3.canonical_contract.dtype)
warnings = []
show(
    "_contract_document('H3Model', lane, warnings) keys",
    lambda: sorted(_derive._contract_document("H3Model", MiniMaxH3.canonical_contract, warnings)),
)
show("_contract_digest(lane)", lambda: _derive._contract_digest(MiniMaxH3.canonical_contract))
show("_resolve_lane", lambda: _derive._resolve_lane(
    __import__("gen_worker._vendor.torchcg", fromlist=["x"]), H3Model,
    MiniMaxH3.canonical_contract
), brief=True)


# ── C. the shared conformance corpus ─────────────────────────────────────────

rule("C. tensorfs shared conformance corpus (spec/v1/contract-vectors)")

try:
    from gen_worker._vendor.tensorfs.contract import Contract, ContractError
except Exception as exc:  # noqa: BLE001
    print(f"  no vendored contract reader on this tree -> {type(exc).__name__}: {exc}")
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests/testdata/contract-vectors"
DOCUMENTS = ROOT / "src/gen_worker/_vendor/tensorfs/_contracts"
vectors = json.loads((CORPUS / "contract-vectors.json").read_text())

ok = bad = 0
for case in vectors["golden"]:
    # A library vector resolves against the documents THIS REPO VENDORS, so
    # the corpus proves the bytes gen-worker actually publishes.
    document = (
        (DOCUMENTS / Path(case["file"]).name).read_text()
        if "file" in case
        else case["document"]
    )
    try:
        contract = Contract.from_document(document)
        agrees = contract.digest == case["digest"] and contract.stamp == case["stamp"]
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        agrees = False
    print(f"  golden {case['name']:<32} {'OK' if agrees else 'MISMATCH'}")
    ok, bad = (ok + 1, bad) if agrees else (ok, bad + 1)

for case in vectors["refusals"]:
    try:
        Contract.from_document(case["document"])
        agrees, detail = False, "ACCEPTED (should have refused)"
    except ContractError as exc:
        agrees = exc.reason == case["reason"]
        detail = f"reason={exc.reason!r}"
    except Exception as exc:  # noqa: BLE001
        agrees, detail = False, f"{type(exc).__name__}: {exc}"
    print(f"  refusal {case['name']:<31} {'OK' if agrees else 'WRONG'} {detail}")
    ok, bad = (ok + 1, bad) if agrees else (ok, bad + 1)

print(f"\n  corpus: {ok} agree, {bad} disagree")
sys.exit(1 if bad else 0)
