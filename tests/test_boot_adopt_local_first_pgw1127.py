"""pgw#1127 S2 — the boot asks THIS MACHINE before it asks the hub, and can ask nobody.

DESIGN-RULINGS §4.28: *"Untrusted hardware mints for ITSELF: local cell, local
repo-CAS, reused across its own boots — **never uploaded, never requested**."*

THE DEFECT. ``executor._boot_adopt`` returned ``no_hub`` **before deriving the
key**, on the premise that *"deriving a key nobody will answer is pure boot
latency"*. That premise is false on exactly the machines §4.28 is about: the
derived ``ck1`` key IS ``local_cell_store``'s own address, so an offline box
holding the exact cell it needs was told there was nobody to ask.
``boot_adopt.py`` imported ``cell_resolve`` and not ``local_cell_store``; step 2
of §4.27 was hub-or-nothing.

Reuse still happened — through ``arm_from_local_store``'s arm-token memo — so
"fully offline-capable" was TRUE BY ACCIDENT of a shortcut rather than by
design. The sequence is now **derive -> local ``lookup(key)`` -> hub
``resolve(key)``**, and ``no_hub`` is demoted from a pre-derive gate to two
honest termini: ``no_compiled_graph_source`` (nobody at all, so the derive is skipped) and
``local_miss_no_hub`` (derived, this machine does not hold it, no hub either).

THE STRONGEST ARGUMENT, and the row that measures it:
``test_an_arm_scheme_bump_costs_a_TRACE_and_not_a_MINT``. ``sweep_superseded_
memos`` deletes the shortcut and leaves the CELLS under their own ``ck1`` keys.
Before S2 that cost one full MINT per family per machine — on a cozy-local box,
the user's product promise briefly breaking. After S2 the boot addresses the CAS
directly, finds the cell the memo used to name, and rewrites the shortcut from
the proven arm. **The next key/scheme bump costs a TRACE instead of a MINT.**

RED before this issue: every row here. `_boot_adopt` refused pre-derive with no
hub; `local_cell_store` had no reader at boot at all; `boot_local_key` did not
exist as a route into the store; and `local_hit`/`local_miss_no_hub`/
`no_compiled_graph_source` were not in the pgw#1116 vocabulary, so a fence that reads
refusal sites out of the tree would have failed on them.
"""

from __future__ import annotations

import atexit
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

import tcg_artifacts
from gen_worker import (
    activity, boot_adopt, boot_key, cell_resolve, fleet_cells, local_cell_store,
)
from gen_worker import executor as executor_mod
from gen_worker.api import export_contract as export_contract_mod
from gen_worker.cell_adopt import AdoptOutcome

#: The graphs this pod "traced". Real values in the pgw#1031 sense: the boot's
#: witnesses and the cell's recorded ones are compared entry by entry, and the
#: floor is fail-closed in both directions (a silent cell is a refusal too).
WITNESSES = {"transformer": "9f" * 8}
(_ENTRY, _WITNESS), = WITNESSES.items()

# pgw#1283: the keys are DERIVED from real TCG artifacts rather than typed.
# `local_cell_store.store` hands its bytes to `Engine.import_artifact`, which
# refuses an artifact whose own metadata does not restate the key it is filed
# under — so a hand-typed `cg-key-v1-aaa…` can no longer name a storable cell,
# and a fixture that pretended otherwise would only ever test the refusal.
#
# `KEY_DERIVED` is therefore both "what the boot derives" and "what this
# machine's own artifact states" — the §4.27 identity this file is about, now
# ENFORCED by the store instead of asserted by the fixture.
_FIXTURE_DIR = Path(tempfile.mkdtemp(prefix="pgw1127-boot-adopt-"))
atexit.register(shutil.rmtree, _FIXTURE_DIR, True)
ARTIFACT_LOCAL = tcg_artifacts.build(
    _FIXTURE_DIR / "local.tar.gz", graph_class=_ENTRY, witness=_WITNESS)
ARTIFACT_OTHER = tcg_artifacts.build(
    _FIXTURE_DIR / "other.tar.gz", graph_class=_ENTRY, witness=_WITNESS,
    sm="sm_90")
KEY_A = KEY_DERIVED = tcg_artifacts.key_of(ARTIFACT_LOCAL)
KEY_B = tcg_artifacts.key_of(ARTIFACT_OTHER)
assert KEY_A != KEY_B
ARM_A = fleet_cells.ARM_SCHEME + "-" + "1" * fleet_cells.ARM_DIGEST_HEX


# ---------------------------------------------------------------------------
# Fixtures — the machine's own store, and a wire that CANNOT be used
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cozy-cells"
    monkeypatch.setenv(local_cell_store.ENV_STORE_DIR, str(root))
    return root


@pytest.fixture()
def cas(tmp_path: Path) -> Path:
    """The CAS the BOOT will address — ``_executor``'s own model-store root.

    pgw#1283: the bytes live in the worker's CAS now, and the boot reaches it
    through the same ``cache_dir`` the executor threads into every other TCG
    call (``executor._boot_adopt`` -> ``boot_adopt.attempt(cache_dir=…)``).
    Storing a cell somewhere the arm cannot resolve it is the bug this fixture
    keeps the file honest about.
    """
    return tmp_path / "cas"


@pytest.fixture()
def events(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    seen: List[Any] = []
    monkeypatch.setattr(activity, "_sink", seen.append, raising=False)
    return seen


def _adopt_events(seen: List[Any]) -> List[Any]:
    return [u for u in seen if u.kind == activity.KIND_BOOT_ADOPT]


@pytest.fixture()
def no_wire(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """The never-request fence, at the SOCKET.

    Not "``cell_resolve.resolve`` was not called" — that is a claim about one
    function, and the property §4.28 states is about the machine. Any attempt to
    open any connection, by any layer, records itself and fails.
    """
    attempts: List[Any] = []
    real_connect = socket.socket.connect

    def _connect(self: Any, address: Any) -> None:
        attempts.append(address)
        raise OSError(
            "pgw#1127: this boot must reach the network ZERO times — §4.28's "
            f"'never requested' — and it tried to reach {address!r}")

    monkeypatch.setattr(socket.socket, "connect", _connect)
    monkeypatch.setattr(
        socket, "create_connection",
        lambda address, *a, **k: _connect(socket.socket(), address))
    assert real_connect is not None
    return attempts


def _cell(
    tmp_path: Path, *, source: Path = ARTIFACT_LOCAL, name: str = "cell",
) -> Path:
    """One real TCG artifact, staged where an earlier boot's mint left it.

    pgw#1283: this used to hand-roll a tarball carrying an ``entry`` block —
    a shape the identity cut (pgw#1277) had already made unwritable by
    anything in ``src/``. The envelope is now built by TCG itself, so the
    witness the boot compares against is the witness TCG records: one shape,
    one writer.
    """
    p = tmp_path / name / "cell.tar.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, p)
    return p


class _Cfg:
    family = "micro-diffusion"
    targets = ("transformer",)
    shapes = ((64, 64),)
    text_lens = (8,)
    guidance_scales = ()
    lora_bucket = 0


class _Spec:
    name = "generate"
    module = "micro_diffusion.main"
    compile = _Cfg()

    def compile_cell(self) -> _Cfg:
        return _Cfg()


class _Pipe:
    pass


class _Arm:
    def __init__(self, token: str = ARM_A) -> None:
        self.token = token

    def facts_dict(self) -> Dict[str, str]:
        return {}


def _executor(tmp_path: Path) -> Any:
    from gen_worker.executor import Executor
    from gen_worker.models.store import ModelStore

    async def _send(msg: Any) -> None:
        pass

    ex = Executor([], _send, store=ModelStore(_send, cache_dir=tmp_path / "cas"))
    ex.file_base_url = ""
    ex.worker_jwt_provider = lambda: ""
    return ex


def _derived() -> Any:
    """A REAL ``DerivedKey`` — the key is built by ``tcg.identity.from_axes`` over
    real axes, so the address the boot hands the store is the address the store
    is addressed by everywhere else. Only the TRACE is stood in for (there is no
    card on a CI runner, and `sm` is a key axis)."""
    from gen_worker._vendor.torchcg import identity as ck

    del ck  # pgw#1283: the address comes from the artifact this machine HOLDS
    return boot_key.DerivedKey(
        entry_keys={"a": KEY_DERIVED},
        workers=2, width_reason="test", traced=1, memo="miss", wall_ms=7)


# pgw#1176: a boot derives a key SET. This declaration traces to one class, so
# the set has one member; a caller that wants "the address" takes it from
# `keys`, never from a `digest` property that no longer exists because there is
# no single key to have one.
assert _derived().keys == (KEY_DERIVED,)


@pytest.fixture()
def declared(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two pre-attempt gates that are NOT under test here."""
    monkeypatch.setattr(
        export_contract_mod, "export_declaration", lambda f: object())
    monkeypatch.setattr(
        executor_mod.aot_declaration, "cell_plans", lambda d: [object()])
    from gen_worker.procsplit import broker

    monkeypatch.setattr(broker, "_broker", None, raising=False)


# ---------------------------------------------------------------------------
# 1. The gate is demoted: nobody-at-all still short-circuits, a machine that
#    could answer does not
# ---------------------------------------------------------------------------


def test_an_empty_store_and_no_hub_still_skips_the_derivation(
    store: Path, tmp_path: Path, declared: None, events: List[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The half of the old gate that was RIGHT, kept.

    With no hub AND nothing stored, no route could answer, so paying a
    structure-only trace would be pure boot latency — the old comment's claim,
    true in exactly this case and only this case.
    """
    monkeypatch.setattr(
        boot_key, "derive",
        lambda **kw: pytest.fail("nobody could answer; the derive is latency"))

    # pgw#1176: a boot returns ONE outcome per declared graph class. This
    # declaration traces to one, so the unpack ASSERTS that arity rather than
    # indexing past a set nobody checked.
    (out,) = _executor(tmp_path)._boot_adopt(_Spec(), {})

    assert out.reason == "no_compiled_graph_source"
    assert not out.derived_key
    assert [u.phase for u in _adopt_events(events)] == ["no_compiled_graph_source"]


def test_a_machine_holding_cells_DERIVES_even_with_no_hub_at_all(
    store: Path, cas: Path, tmp_path: Path, declared: None, events: List[Any],
    no_wire: List[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The half that was WRONG. §4.28's machine has an answerer: itself.

    RED before pgw#1127: `_boot_adopt` returned `no_hub` here without deriving,
    so the one address that could have been answered was never computed.
    """
    local_cell_store.store(
        _cell(tmp_path, source=ARTIFACT_OTHER), key=KEY_B, family="other",
        arm_token=ARM_A, cas_root=cas)
    monkeypatch.setattr(boot_key, "derive", lambda **kw: _derived())

    # pgw#1176: a boot returns ONE outcome per declared graph class. This
    # declaration traces to one, so the unpack ASSERTS that arity rather than
    # indexing past a set nobody checked.
    (out,) = _executor(tmp_path)._boot_adopt(_Spec(), {})

    assert out.derived_key == KEY_DERIVED, "the key was not derived at all"
    assert out.reason == "local_miss_no_hub"
    assert not out.local_key
    assert not no_wire, f"an offline boot touched the network: {no_wire}"
    row = _adopt_events(events)[-1]
    assert row.phase == "local_miss_no_hub"
    assert "base_url=<unset>" in row.detail


# ---------------------------------------------------------------------------
# 2. THE FENCE (pgw#1127 §4): a populated store + an unreachable hub = ZERO HTTP
# ---------------------------------------------------------------------------


def test_a_populated_store_and_an_unreachable_hub_makes_zero_http_attempts(
    store: Path, cas: Path, tmp_path: Path, declared: None, events: List[Any],
    no_wire: List[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§4.28's *"never requested"*, enforced rather than described.

    A hub URL is CONFIGURED and points nowhere — the shape that matters, since
    "no base_url" would make the old gate accidentally right. The boot derives,
    its own store answers, and no connection is attempted by any layer.
    """
    local_cell_store.store(
        _cell(tmp_path), key=KEY_DERIVED, family="micro-diffusion",
        arm_token=ARM_A, cas_root=cas)
    monkeypatch.setattr(boot_key, "derive", lambda **kw: _derived())

    ex = _executor(tmp_path)
    ex.file_base_url = "http://127.0.0.1:9"   # discard port: nothing listens
    ex.worker_jwt_provider = lambda: "worker-jwt"

    (out,) = ex._boot_adopt(_Spec(), {})

    assert not no_wire, (
        "the boot reached for the wire while holding the answer on disk: "
        f"{no_wire}")
    assert out.reason == boot_adopt.LOCAL_HIT
    assert out.local_key == KEY_DERIVED == out.derived_key
    row = _adopt_events(events)[-1]
    assert row.phase == "local_hit"
    assert KEY_DERIVED in row.detail


def test_the_local_store_is_asked_BEFORE_a_perfectly_reachable_hub(
    store: Path, cas: Path, tmp_path: Path, declared: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordering is the promise, not availability. A local-SECOND design would
    request what the machine already holds — and on a community-cloud pod that
    is the hub shipping back bytes the pod itself minted."""
    local_cell_store.store(
        _cell(tmp_path), key=KEY_DERIVED, family="micro-diffusion",
        arm_token=ARM_A, cas_root=cas)
    monkeypatch.setattr(boot_key, "derive", lambda **kw: _derived())
    monkeypatch.setattr(
        cell_resolve, "resolve_batch",
        lambda *a, **k: pytest.fail("the hub was asked for a resident cell"))

    ex = _executor(tmp_path)
    ex.file_base_url = "http://hub.local"
    ex.worker_jwt_provider = lambda: "worker-jwt"

    assert ex._boot_adopt(_Spec(), {})[0].reason == boot_adopt.LOCAL_HIT


def test_a_local_hit_carries_an_ADDRESS_and_never_an_adoption(
    store: Path, cas: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-minted cell has no hub receipt and no publisher org, so riding the
    ``arm_ordered`` path a hub-resolved cell rides would refuse it
    ``receipt_gate_unconfigured``. The boot hands over the ADDRESS instead, and
    ``fleet_cells._arm_exported_cell`` — the gate a child's own mint passes —
    decides."""
    local_cell_store.store(
        _cell(tmp_path), key=KEY_DERIVED, family="micro-diffusion",
        arm_token=ARM_A, cas_root=cas)
    monkeypatch.setattr(boot_key, "derive", lambda **kw: _derived())

    (out,) = boot_adopt.attempt(
        function="generate", modules=("micro_diffusion.main",), cfg=_Cfg(),
        slots={}, declared_hint=1, work_root=tmp_path, cache_dir=cas,
        hub_absent="nobody to ask")

    assert out.reason == boot_adopt.LOCAL_HIT
    assert out.adoption is None and not out.adopted, (
        "a local cell must not become an `_ArmOrder`: the receipt gate would "
        "refuse it, and pgw#1122's degrade would spend a whole arm learning it")
    assert out.local_key == KEY_DERIVED


# ---------------------------------------------------------------------------
# 4. THE COST ARGUMENT: an arm-scheme bump costs a TRACE, not a MINT
# ---------------------------------------------------------------------------


@pytest.fixture()
def armable(monkeypatch: pytest.MonkeyPatch) -> List[Path]:
    seen: List[Path] = []

    def _arm(pipe: Any, cfg: Any, cache_dir: Any, artifact: Path,
             bucket: int, expected: Any = None, *,
             verify_numerics: bool = False, **_kw: Any) -> AdoptOutcome:
        # pgw#1141 / §4.32: the local store's route is an ADOPTION — these
        # bytes were proven at their own mint — so it must not ask for the
        # mint-time gate. Asserted rather than absorbed: a bare `**kwargs`
        # shim would keep this file green if the per-adopter tax came back.
        assert verify_numerics is False, (
            "the local store's ADOPT path asked for the mint-time gate")
        seen.append(Path(artifact))
        return AdoptOutcome.hit(KEY_A)

    monkeypatch.setattr(fleet_cells.provision, "arm_aot", _arm)
    monkeypatch.setattr(
        fleet_cells.artifact_meta, "read_metadata",
        lambda p: {"compiled_graph_key": KEY_A, "family": "micro-diffusion"})
    monkeypatch.setattr(fleet_cells, "arm_axis_divergence", lambda key, meta: "")
    return seen


def test_an_arm_scheme_bump_costs_a_TRACE_and_not_a_MINT(
    store: Path, cas: Path, tmp_path: Path, armable: List[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1127 §5 item 2 — the strongest argument for S2, measured.

    `sweep_superseded_memos` deletes every shortcut written under a superseded
    arm-token scheme and leaves the CELLS under their own `ck1` keys ("an
    arm-token change re-derives the shortcut, never the identity"). Before S2
    the swept memo WAS the only address, so the machine paid a full mint. After
    S2 the boot-derived key still names the cell.

    On a cozy-local box that is the difference between a user upgrading
    gen-worker and their next run being instant, and their next run paying a
    40-minute compile for a cell already on their disk.
    """
    local_cell_store.store(
        _cell(tmp_path), key=KEY_A, family="micro-diffusion", arm_token=ARM_A,
        cas_root=cas)
    # The upgrade: every memo written under the previous scheme is swept.
    removed = local_cell_store.sweep_superseded_memos("arm99")
    assert removed == 1
    assert local_cell_store.lookup_for_arm(ARM_A, cas_root=cas) is None, (
        "memo not swept")
    assert local_cell_store.lookup(KEY_A, cas_root=cas) is not None, (
        "the CELL must survive")

    # WITHOUT the derived key this is a miss, and a miss is a mint.
    assert fleet_cells.arm_from_local_store(
        _Pipe(), _Cfg(), cas, 0, _Arm(), "micro-diffusion") is None

    # WITH it, the same cell arms — and the shortcut is rewritten from the
    # proven arm, so the boot after this one is a memo hit again.
    minted = fleet_cells.arm_from_local_store(
        _Pipe(), _Cfg(), cas, 0, _Arm(), "micro-diffusion",
        boot_local_key=KEY_A)

    assert minted is not None and minted.compiled_graph_key == KEY_A
    assert armable and armable[-1] == local_cell_store.cell_dir(KEY_A) / "cell.tar.gz"
    repaired = local_cell_store.lookup_for_arm(ARM_A, cas_root=cas)
    assert repaired is not None and repaired.key == KEY_A, (
        "the memo must be repaired from the proven arm — otherwise every boot "
        "after a scheme bump pays the trace again")


def test_the_boot_key_route_refuses_WITHOUT_dropping_and_the_memo_route_drops(
    store: Path, cas: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two routes, two different things a refusal is allowed to do.

    The memo was written by this machine's own mint for this exact arm token,
    so a cell that will not arm under it is stale — drop it, one honest
    re-mint. A boot-key hit is an inference; destroying another pipe's cell to
    punish a wrong guess costs two.
    """
    monkeypatch.setattr(
        fleet_cells, "_arm_exported_cell",
        lambda *a, **k: (False, None, ("key_axis_divergence", "sm")))

    local_cell_store.store(
        _cell(tmp_path), key=KEY_A, family="micro-diffusion", arm_token="",
        cas_root=cas)
    assert fleet_cells.arm_from_local_store(
        _Pipe(), _Cfg(), cas, 0, _Arm(), "micro-diffusion",
        boot_local_key=KEY_A) is None
    assert local_cell_store.lookup(KEY_A, cas_root=cas) is not None, (
        "route B must not drop")

    local_cell_store.note_memo(ARM_A, KEY_A)
    assert fleet_cells.arm_from_local_store(
        _Pipe(), _Cfg(), cas, 0, _Arm(), "micro-diffusion") is None
    assert local_cell_store.lookup(KEY_A, cas_root=cas) is None, (
        "route A must drop a stale cell")


# ---------------------------------------------------------------------------
# 5. The address actually reaches the arming brain
# ---------------------------------------------------------------------------


def test_the_boot_derived_key_is_threaded_to_the_arming_brain(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A boot that finds its own cell and then cannot say so is the same boot
    that mints. The address is an explicit parameter the whole way down —
    `enable_compiled` -> `_arming_policy` -> `arm_from_local_store` — so it is
    greppable rather than ambient."""
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(
        fleet_cells, "arm_from_local_store",
        lambda *a, **k: seen.update(k) or None)
    monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
    monkeypatch.setattr(fleet_cells.cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(
        fleet_cells.cc, "has_compile_target", lambda pipe, cfg: True)
    monkeypatch.setattr(
        fleet_cells, "mint_recipe",
        lambda pipe, cfg, **kw: fleet_cells.RECIPE_AOT)
    monkeypatch.setattr(fleet_cells, "arm_identity", lambda *a, **k: _Arm())
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda *a, **k: AdoptOutcome.miss("no_compiled_graph", "nothing delivered"))

    fleet_cells.enable_compiled(
        _Pipe(), _Cfg(), None, publisher=None, boot_local_key=KEY_A)

    assert seen.get("boot_local_key") == KEY_A


# ---------------------------------------------------------------------------
# 6. The vocabulary — pgw#1116's fence, extended rather than worked around
# ---------------------------------------------------------------------------


def test_every_new_terminus_is_in_the_typed_vocabulary() -> None:
    """pgw#1116's rule: a path that can produce a token ``REASONS`` does not
    carry is a path that can be silent again. That telemetry has paid for itself
    twice in a day; a new gate that skipped it would be the next unattributable
    pod."""
    for token in ("local_hit", "local_miss_no_hub"):
        assert token in boot_adopt.LOCAL_REASONS
        assert token in boot_adopt.REASONS
    assert "no_compiled_graph_source" in boot_adopt.GATE_REASONS
    assert "no_hub" not in boot_adopt.REASONS, (
        "the token that refused on behalf of two answerers is DELETED, not "
        "aliased — pre-launch hardcut")


def test_each_new_terminus_emits_exactly_one_typed_event(
    store: Path, cas: Path, tmp_path: Path, declared: None, events: List[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One decision, one event, ``phase`` = the gate's own token. "adopt
    failed" makes eight different bugs look like one."""
    monkeypatch.setattr(boot_key, "derive", lambda **kw: _derived())
    ex = _executor(tmp_path)

    ex._boot_adopt(_Spec(), {})                       # empty store, no hub
    local_cell_store.store(
        _cell(tmp_path, source=ARTIFACT_OTHER, name="b"), key=KEY_B,
        family="other", arm_token="", cas_root=cas)
    ex._boot_adopt(_Spec(), {})                       # stored, but not this key
    local_cell_store.store(
        _cell(tmp_path), key=KEY_DERIVED, family="micro-diffusion",
        arm_token=ARM_A, cas_root=cas)
    ex._boot_adopt(_Spec(), {})                       # this machine holds it

    phases = [u.phase for u in _adopt_events(events)]
    assert phases == ["no_compiled_graph_source", "local_miss_no_hub", "local_hit"], (
        f"three different facts must read as three different phases: {phases}")


def test_a_key_that_missed_locally_is_distinguishable_from_one_never_derived(
    store: Path, cas: Path, tmp_path: Path, declared: None, events: List[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1116's regression box, on the new pair. `derived_key` is what tells
    "nothing here holds it" apart from "I never computed an address"."""
    monkeypatch.setattr(boot_key, "derive", lambda **kw: _derived())
    ex = _executor(tmp_path)

    (never,) = ex._boot_adopt(_Spec(), {})
    local_cell_store.store(
        _cell(tmp_path, source=ARTIFACT_OTHER, name="b"), key=KEY_B,
        family="other", arm_token="", cas_root=cas)
    (missed,) = ex._boot_adopt(_Spec(), {})

    assert never.reason == "no_compiled_graph_source" and not never.derived_key
    assert missed.reason == "local_miss_no_hub"
    assert missed.derived_key == KEY_DERIVED
    details = {u.phase: u.detail for u in _adopt_events(events)}
    assert "key=-" in details["no_compiled_graph_source"]
    assert f"key={KEY_DERIVED}" in details["local_miss_no_hub"]


def test_boot_adopt_reads_the_local_store_and_says_so_in_its_imports() -> None:
    """The one-line summary of the defect: ``boot_adopt`` imported
    ``cell_resolve`` and not ``local_cell_store``, so step 2 of §4.27 was
    hub-or-nothing. RED before: this import did not exist."""
    import ast

    tree = ast.parse(Path(boot_adopt.__file__).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert {"local_cell_store", "cell_resolve"} <= imported, (
        "one key, two lookup routes — the boot must be able to address BOTH")
