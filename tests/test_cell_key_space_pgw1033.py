"""pgw#1033 — the STAMPED key is the cell's identity; the COMPUTED key names
an ARM. Three live protections were comparing one against the other.

A worker holds two key digests that look identical (``ck1-<56 hex>``) and are
different spaces by construction:

* the **computed** key (``cell_key.compute``, ``kind="inductor"``) — what THIS
  runtime's static axes ask for. It exists before anything is compiled, which
  is exactly why it can name a pending mint;
* the **stamped** key (``aot_mint.cell_identity``, ``kind="aot-inductor"``,
  contract axis folding the combined graph hash) — what the exported cell
  actually IS. It does not exist until the export finishes.

Every protection below was written on one side and fed from the other, so each
one silently stopped firing. The three tests that matter are RED on
``origin/master`` ``872f16e5``:

1. ``finalized_in_process`` (the pgw#672 no-second-mint memo) read the computed
   key while ``adopt_delegated_mint`` wrote the stamped one — the memo could
   never hit, and every same-key re-arm in a process paid a second full export;
2. the pgw#672 quarantine gate read the computed ref while the two writers that
   fire on a real proof failure (``executor``'s runtime guard and boot proof)
   record the STAMPED ref of the armed cell — so the churn-loop gate was blind
   to the refs that actually fail;
3. ``aot_serve.note_aot_key`` had ONE caller, discovery — so a SELF-MINTED cell
   was never registered and the pgw#734/#735 kind dispatch scored this pod's own
   ``.pt2`` by FX cache hits it cannot produce.

Plus the dead-attribute guard: ``_selection_for`` tested ``mint.recipe ==
"aot"``, an attribute pgw#1010 deleted from every mint object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from gen_worker import aot_serve, compile_cache, fleet_cells
from gen_worker import config as gw_config
from gen_worker import mint_delegate
from gen_worker.api.decorators import Compile, Dim, GraphClass, Input
from gen_worker.api.export_contract import (
    register_export_declaration, reset_export_declarations,
)
from gen_worker.cell_adopt import AdoptOutcome, EagerPhase

FAMILY = "sdxl"
#: What this runtime's axes COMPUTE — the arm identity, known before any
#: compile has run.
ARM_KEY = "ck1-" + "a" * 56
#: What the child's envelope is STAMPED with — the cell identity, unknowable
#: until the export finishes. A different digest, always.
STAMPED_KEY = "ck1-" + "b" * 56


class _Pipe:
    pass


@dataclass
class _Cfg:
    family: str = FAMILY
    lora_bucket: int = 0
    shapes: Tuple[Tuple[int, int], ...] = ((1024, 1024),)
    targets: Tuple[str, ...] = ("unet",)
    text_lens: Tuple[int, ...] = (77,)
    guidance_scales: Tuple[float, ...] = (1.0, 5.0)
    regional: bool = False


class _Publisher:
    base_url = "http://hub.invalid"

    def enabled(self) -> bool:
        return True

    def worker_jwt(self) -> str:
        return "jwt"


def _declaration() -> Compile:
    return Compile(
        family=FAMILY,
        targets=("unet",),
        text_len=77,
        shapes=((1024, 1024),),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", 4, 128, 128), dtype="bfloat16"),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    )


@pytest.fixture(autouse=True)
def _clean_declarations() -> Any:
    reset_export_declarations()
    yield
    reset_export_declarations()


@pytest.fixture()
def _events(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str, str]]:
    seen: List[Tuple[str, str, str]] = []

    def _sink(kind: str, detail: str, phase: str = "", duration_ms: int = 0) -> None:
        seen.append((kind, phase, detail))

    monkeypatch.setattr(fleet_cells.activity_mod, "emit_event", _sink)
    monkeypatch.setattr(mint_delegate.activity_mod, "emit_event", _sink)
    return seen


@pytest.fixture()
def _miss(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A mint-capable pod with no cell: the pgw#805 miss shape, one lane."""
    gw_config.reload_for_test()
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: AdoptOutcome.miss("no_cell"))
    monkeypatch.setattr(fleet_cells.cc, "has_compile_target", lambda p, c: True)
    monkeypatch.setattr(fleet_cells.cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(fleet_cells.cc, "mandatory_serving", lambda p: False)
    monkeypatch.setattr(
        fleet_cells.cc, "apply_lora_execution_lane", lambda p, b: None)
    monkeypatch.setattr(
        fleet_cells.cc, "drop_lora_execution_lane", lambda p: None)
    monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
    monkeypatch.setattr(fleet_cells, "_PENDING", {})
    # No GPU on a dev box: `sm` is a real-runtime fact and this suite is about
    # key SPACES, not key computation.
    monkeypatch.setattr(
        fleet_cells.cell_key, "compute",
        lambda *a, **k: type("_K", (), {"digest": ARM_KEY})())
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "w8a8")
    register_export_declaration(_declaration())
    yield
    gw_config.reload_for_test()


def _arm() -> Any:
    return fleet_cells.enable_compiled(
        _Pipe(), _Cfg(), publisher=_Publisher(), delegate=True)  # type: ignore[arg-type]


def _adopt(
    monkeypatch: pytest.MonkeyPatch, pending: Any, *, stamped: str = STAMPED_KEY,
) -> Any:
    """Run the real ``adopt_delegated_mint`` over a child cell stamped with a
    key of the cell's OWN space — the production shape, in one line."""
    pending.target.parent.mkdir(parents=True, exist_ok=True)
    pending.target.write_bytes(b"packed-cell")
    monkeypatch.setattr(
        fleet_cells.provision, "arm_aot",
        lambda *a, **k: AdoptOutcome.hit(f"key={stamped}"))
    monkeypatch.setattr(
        fleet_cells, "_packed_metadata",
        lambda artifact: {
            "kind": "aot-inductor", "cell_key": stamped, "family": FAMILY,
        })
    return fleet_cells.adopt_delegated_mint(_Pipe(), pending, pending.target)


# ---------------------------------------------------------------------------
# 1. The pgw#672 no-second-mint memo
# ---------------------------------------------------------------------------


def test_the_memo_is_keyed_on_the_arm_key_the_next_arm_can_look_up(
    _miss: None, _events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED at HEAD: written under the STAMPED key, read under the COMPUTED
    one — two disjoint spaces, so the memo could never hit."""
    pending = _arm().self_mint
    assert pending is not None and pending.cell_key == ARM_KEY

    minted = _adopt(monkeypatch, pending)
    assert minted is not None and minted.cell_key == STAMPED_KEY

    prior = fleet_cells.finalized_in_process(ARM_KEY)
    assert prior is minted, (
        "the arm identity this process just minted for cannot find its own "
        "cell — every same-key re-arm pays a second full export")
    # The VALUE still carries the cell's own identity; only the INDEX is the
    # arm key. Nothing looks the ledger up by the stamped key.
    assert prior.cell_key == STAMPED_KEY
    assert prior.ref.endswith("#" + STAMPED_KEY)


def test_a_second_arm_of_the_same_identity_re_arms_instead_of_minting_again(
    _miss: None, _events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The protection ITSELF, at its own call site: the second arm must serve
    from the finalized cell and open no second mint."""
    pending = _arm().self_mint
    assert pending is not None
    minted = _adopt(monkeypatch, pending)
    assert minted is not None

    again = _arm()

    assert again.armed, "the re-arm did not serve from the finalized cell"
    assert again.self_mint is minted, (
        "the second arm opened a FRESH mint for an identity this process has "
        "already minted and adopted — the pgw#672 churn loop")
    assert not isinstance(again.self_mint, fleet_cells.PendingSelfMint)
    starts = [phase for kind, phase, _ in _events if kind == "self_mint_started"]
    assert starts == ["aot"], (
        f"expected exactly ONE mint for this identity, saw {len(starts)}")


# ---------------------------------------------------------------------------
# 2. The pgw#672 quarantine gate
# ---------------------------------------------------------------------------


def test_a_quarantined_STAMPED_ref_declines_the_next_arm_of_that_identity(
    _miss: None, _events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED at HEAD: the two writers that fire on a real proof failure
    (``executor`` runtime guard `:4160`, boot proof `:7333`) record the ARMED
    cell's STAMPED ref, and the gate only ever asked about the computed one —
    so the churn-loop gate was blind to every ref that actually fails."""
    pending = _arm().self_mint
    assert pending is not None
    minted = _adopt(monkeypatch, pending)
    assert minted is not None

    # Exactly what `_revoke_compiled_proof` does with the ref it was serving.
    compile_cache.record_cell_quarantined(minted.ref)

    again = _arm()

    assert not again.armed
    assert again.eager_reason == EagerPhase.CELL_QUARANTINED, (
        "the arm re-minted an identity whose own cell was disproven seconds "
        "ago — a deterministic recipe rebuilds the same disproven cell")
    skipped = [
        detail for kind, phase, detail in _events
        if kind == "self_mint_skipped" and phase == EagerPhase.CELL_QUARANTINED
    ]
    assert skipped and minted.ref in skipped[0], (
        "the decline must NAME the quarantined ref; two identities can decline "
        "one arm and the hub cannot tell them apart from the arm key alone")


def test_a_quarantined_ARM_ref_still_declines_the_next_arm(
    _miss: None, _events: List[Tuple[str, str, str]],
) -> None:
    """The other half, unchanged: a mint that died before it produced anything
    has no stamped identity, so `_abandon_pending_mint` quarantines the
    PENDING's ref — the computed one. Both are consulted."""
    pending = _arm().self_mint
    assert pending is not None

    compile_cache.record_cell_quarantined(pending.ref)

    again = _arm()

    assert not again.armed
    assert again.eager_reason == EagerPhase.CELL_QUARANTINED


# ---------------------------------------------------------------------------
# 3. The pgw#734/#735 kind dispatch
# ---------------------------------------------------------------------------


def test_a_self_minted_cell_is_registered_as_an_AOT_ref(
    _miss: None, _events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED at HEAD: ``note_aot_key`` had one caller — discovery. The one
    artifact this process is CERTAIN is exported, because it just built it,
    was the one ref ``is_aot_ref`` did not recognize; the executor then scored
    it by FX cache hits an AOTI package cannot produce and disproved it."""
    pending = _arm().self_mint
    assert pending is not None
    assert not aot_serve.is_aot_ref(pending.ref), (
        "an OWED mint has no cell yet — nothing may claim its computed ref "
        "names an exported artifact")

    minted = _adopt(monkeypatch, pending)
    assert minted is not None

    assert aot_serve.is_aot_ref(minted.ref), (
        "this pod's own exported cell reads as a dynamo ref, so the boot "
        "proof scores it by FX cache hits and fails it closed")
    assert aot_serve.is_aot_ref(minted.ref, FAMILY)
    # The registry is the STAMPED key's, not the arm's: an arm key names no
    # artifact and must never be classified as one.
    assert not aot_serve.is_aot_ref(pending.ref)


# ---------------------------------------------------------------------------
# 4. The dead-attribute guard (pgw#805's, on an attribute pgw#1010 deleted)
# ---------------------------------------------------------------------------


def test_an_owed_mint_advertises_no_artifact_identity(
    _miss: None, _events: List[Tuple[str, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED at HEAD: the guard tested ``mint.recipe == "aot"`` — no mint object
    has carried a ``recipe`` attribute since pgw#1010 — so `_selection_for`
    handed back a selection carrying the pending's COMPUTED ref, an identity
    no artifact will ever be stamped with. Only the caller's `delegated`
    branch, which drops the selection it just asked for, kept it off the wire.
    """
    from gen_worker import executor

    pending = _arm().self_mint
    assert pending is not None
    assert not hasattr(pending, "recipe"), (
        "pgw#1010 deleted the recipe axis; a guard that reads one is dead")

    delivered = executor._CompileArtifactSelection(
        path=Path("/dev/null"), ref=f"root/family-{FAMILY}#ck1-" + "c" * 56,
        snapshot_digest="sha256:c", self_mint=False)

    assert executor._selection_for(None, pending) is None
    assert executor._selection_for(delivered, pending) is delivered

    minted = _adopt(monkeypatch, pending)
    assert minted is not None
    sel = executor._selection_for(delivered, minted)
    assert sel is not None and sel.self_mint
    assert sel.ref == minted.ref, (
        "an ADOPTED cell advertises the key stamped on the bytes it serves")


# ---------------------------------------------------------------------------
# 5. The pgw#686 divergence warning — DELETED by pgw#1032
#
# `_warn_cell_key_divergence` and its two tests
# (`test_a_healthy_self_minted_AOT_boot_logs_no_key_divergence`,
# `test_a_divergence_inside_one_key_space_is_still_loud`) are gone with
# `requested_cell_key` itself. This issue's fix was the INTERIM it announced:
# silencing a warning whose whole premise — that a self-minted cell's key
# should equal the key this runtime computes — is false for every mint there
# is. With the computed key no longer produced there is no divergence left to
# judge, loudly or quietly. The disjointness the warning kept tripping over is
# now asserted directly in `test_computed_key_demand_retired_pgw1032.py`.
# ---------------------------------------------------------------------------
