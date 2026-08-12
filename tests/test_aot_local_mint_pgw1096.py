"""pgw#1096 / §4.28 — an untrusted machine mints for ITSELF, and reuses it forever.

Paul, 2026-08-10: *"Untrusted hardware (community cloud, cozy-local) mints for
ITSELF: local cell, local repo-CAS, reused across its own boots — never
uploaded, never requested… download model + code ONCE, compile ONCE, and every
subsequent run of that code reuses the same compiled cell."*

What was RED before this issue, on every one of these:

* a community-cloud pod minted a cell, ate the hub's
  ``cell_publish_untrusted_tier`` 403, and ``_publish_async``'s ``finally``
  rmtree'd the bytes — th#1643's SUNK case, once per boot, forever;
* nothing in the tree could address a locally-stored AOT cell (``local_cells``
  is JIT-only and knows no ``ck1`` key);
* ``aot_resume``'s PRODUCTION resume bank was rooted through the module
  pgw#1086 wave 1 deletes.

The store's own guarantees are proven against REAL bytes on disk (digest,
atomicity, drop-on-corruption). The arm is proven to be the SAME gate the
child's own cell passes — structurally, so it cannot drift into a weaker one.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from gen_worker import aot_resume, cell_key, fleet_cells, local_cell_store
from gen_worker.cell_adopt import AdoptOutcome

KEY_A = "ek1-" + "a" * 56
KEY_B = "ek1-" + "b" * 56
# pgw#1113: spelled in the CURRENT token scheme — a predecessor-scheme memo
# is swept by `fleet_cells._sweep_superseded_memos_once`, which is the point.
ARM_A = fleet_cells.ARM_SCHEME + "-" + "1" * 56
ARM_B = fleet_cells.ARM_SCHEME + "-" + "2" * 56


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The store root, relocated by its env — a PATH knob, never a behavior one."""
    root = tmp_path / "cozy-cells"
    monkeypatch.setenv(local_cell_store.ENV_STORE_DIR, str(root))
    return root


def _artifact(tmp_path: Path, body: bytes = b"packed-cell-bytes") -> Path:
    """Opaque bytes. The STORE does not care what a cell is — these tests are
    about addressing, digests and rot, and they assert the bytes round-trip
    unchanged, so this deliberately stays raw."""
    p = tmp_path / "mint" / "cell.tar.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return p


def _armable_artifact(tmp_path: Path, *, key: str = KEY_A) -> Path:
    """A cell with a READABLE envelope — for the tests that reach the ARM.

    pgw#1098: `_arm_exported_cell` refuses `cell_envelope_unreadable` before
    any other gate, for the local store exactly as for a child's fresh mint —
    which is precisely pgw#1096's "one gate, two sources" intent. Opaque bytes
    used to reach the arm only because the metadata read swallowed its own
    failure into `None`, so a test that ARMS has to supply a real envelope
    while a test that only STORES does not.
    """
    import io as _io
    import tarfile as _tarfile

    p = tmp_path / "mint" / "cell.tar.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"kind": "aot-inductor", "cell_key": key, "family": "micro-diffusion"}
    ).encode()
    with _tarfile.open(p, mode="w:gz") as tar:
        info = _tarfile.TarInfo("metadata.json")
        info.size = len(payload)
        tar.addfile(info, _io.BytesIO(payload))
    return p


# ---------------------------------------------------------------------------
# 1. The store: one identity, two stores
# ---------------------------------------------------------------------------


def test_a_cell_is_addressed_by_the_same_ck1_key_the_hub_store_uses(
    store: Path, tmp_path: Path,
) -> None:
    """§4.28's "one identity, two stores": the local address IS the stamped key.

    RED before: no local address existed for an AOT cell at all — `local_cells`
    keys on a flavor LABEL (sku/torch/lane), a space with no `ck1` in it.
    """
    cell = local_cell_store.store(
        _artifact(tmp_path), key=KEY_A, family="micro-diffusion", arm_token=ARM_A)

    assert cell is not None
    assert cell.key == KEY_A and cell_key.is_key(cell.key)
    assert cell.artifact == store / "aot-cells" / KEY_A / "cell.tar.gz"
    assert cell.artifact.read_bytes() == b"packed-cell-bytes"

    found = local_cell_store.lookup(KEY_A)
    assert found is not None and found.content_digest == cell.content_digest
    assert found.family == "micro-diffusion"


def test_the_store_refuses_to_be_addressed_by_anything_that_is_not_a_key(
    store: Path,
) -> None:
    """A store addressed by a key-shaped string it never verified is a store
    whose layout depends on what a caller happened to pass."""
    for bad in ("arm1-" + "a" * 56, "../escape", "", "sha256:deadbeef"):
        with pytest.raises(ValueError):
            local_cell_store.cell_dir(bad)
    assert local_cell_store.lookup("../escape") is None


def test_a_record_written_but_never_filled_is_absent_not_short(
    store: Path, tmp_path: Path,
) -> None:
    """The record is written LAST (aot_resume's rule): a crash mid-store leaves
    a directory the next lookup treats as absent, never as a short cell."""
    local_cell_store.store(_artifact(tmp_path), key=KEY_A, family="f")
    (store / "aot-cells" / KEY_A / local_cell_store.RECORD_NAME).unlink()
    assert local_cell_store.lookup(KEY_A) is None


# ---------------------------------------------------------------------------
# 2. RED — a corrupted cell is refused, and costs exactly one re-mint
# ---------------------------------------------------------------------------


def test_corrupting_a_stored_cell_refuses_admission_and_drops_it(
    store: Path, tmp_path: Path,
) -> None:
    """`aot_resume`'s safety property, applied to the store: **a bank cannot
    vouch for itself.** The digest is RECOMPUTED over the bytes on disk, so a
    truncated, half-written or edited artifact refuses HERE rather than being
    handed to an arm that would then serve it.

    RED before: there was no local store, and the JIT one it succeeds compared
    only recorded FACTS — nothing ever hashed the bytes.
    """
    cell = local_cell_store.store(
        _artifact(tmp_path), key=KEY_A, family="f", arm_token=ARM_A)
    assert cell is not None and local_cell_store.lookup(KEY_A) is not None

    cell.artifact.write_bytes(b"packed-cell-byteZ")  # same length, one bit

    assert local_cell_store.lookup(KEY_A) is None, "a corrupted cell armed"
    assert not local_cell_store.cell_dir(KEY_A).exists(), (
        "a refused cell must be DROPPED — leaving it costs a re-verify (and a "
        "re-hash of every boot) forever, for bytes nothing can ever use")
    assert local_cell_store.lookup_for_arm(ARM_A) is None, (
        "the memo must not resurrect a cell the digest refused")


# ---------------------------------------------------------------------------
# 3. The memo — a shortcut, never an authority; and the honest re-mint
# ---------------------------------------------------------------------------


def test_the_memo_turns_a_pre_trace_identity_into_the_ck1_key(
    store: Path, tmp_path: Path,
) -> None:
    """The `ck1` graph axis needs a trace; the pre-trace `ArmIdentity` does not.
    The memo is what lets boot 2 address the CAS without paying an export."""
    local_cell_store.store(
        _artifact(tmp_path), key=KEY_A, family="f", arm_token=ARM_A)

    hit = local_cell_store.lookup_for_arm(ARM_A)
    assert hit is not None and hit.key == KEY_A
    assert local_cell_store.lookup_for_arm(ARM_B) is None


def test_a_moved_toolchain_is_an_HONEST_re_mint_and_leaves_the_old_cell_alone(
    store: Path, tmp_path: Path,
) -> None:
    """Paul: *"the key's toolchain/sm axes make local re-mints honest when the
    user upgrades torch or changes GPU."* The upgraded runtime computes a
    different ArmIdentity, so it misses — and the PREVIOUS cell stays resident
    under its own key, because a user who rolls torch back must not have to
    recompile what they already compiled."""
    local_cell_store.store(
        _artifact(tmp_path, b"cell-under-torch-A"), key=KEY_A, family="f",
        arm_token=ARM_A)

    # The upgrade: a different toolchain digest => a different arm token.
    assert local_cell_store.lookup_for_arm(ARM_B) is None, (
        "an upgraded toolchain must not adopt the old toolchain's cell")

    local_cell_store.store(
        _artifact(tmp_path, b"cell-under-torch-B"), key=KEY_B, family="f",
        arm_token=ARM_B)

    assert {c.key for c in local_cell_store.stored_cells()} == {KEY_A, KEY_B}
    old = local_cell_store.lookup(KEY_A)
    assert old is not None and old.artifact.read_bytes() == b"cell-under-torch-A"


def test_a_stale_memo_costs_one_re_mint_and_never_a_wrong_cell(
    store: Path, tmp_path: Path,
) -> None:
    local_cell_store.store(
        _artifact(tmp_path), key=KEY_A, family="f", arm_token=ARM_A)
    local_cell_store.drop(KEY_A)
    assert local_cell_store.lookup_for_arm(ARM_A) is None

    (store / "aot-cells" / local_cell_store.MEMO_DIRNAME / f"{ARM_A}.json"
     ).write_text(json.dumps({"cell_key": "not-a-key"}))
    assert local_cell_store.lookup_for_arm(ARM_A) is None


# ---------------------------------------------------------------------------
# 4. The posture is HUB-ASSERTED — the worker never declares its own trust
# ---------------------------------------------------------------------------


def test_the_worker_learns_it_is_untrusted_only_from_the_hubs_typed_refusal(
    store: Path,
) -> None:
    """§4.28 + the brief: *"the trust class is hub-asserted, learned via typed
    publish refusal — do NOT invent a worker-side self-declaration."*

    Unknown is not trusted and not untrusted: a machine that never tried to
    publish has learned nothing, and the honest consequence is that its first
    mint attempts the publish and reads the answer.
    """
    assert local_cell_store.trust_class() == ""
    assert local_cell_store.keeps_cells_locally() is False

    for unrelated in ("cell_publish_forged_axis", "cell_publish_quota_exceeded",
                      "cell_publish_family_undeclared", "http_429", ""):
        assert local_cell_store.note_refusal(unrelated, "detail") is False
        assert local_cell_store.trust_class() == "", (
            f"{unrelated!r} says something about the CELL or the moment, never "
            f"about the hardware; concluding 'untrusted' from it would mis-file "
            f"this machine permanently off one bad minute")

    assert local_cell_store.note_refusal(
        local_cell_store.UNTRUSTED_REFUSAL_CODE, "community_tier") is True
    assert local_cell_store.trust_class() == local_cell_store.TRUST_UNTRUSTED
    assert local_cell_store.keeps_cells_locally() is True


def test_no_env_declares_the_trust_class(store: Path) -> None:
    """§1.18: an env may carry a VALUE, never a decision. The only env this
    module reads is a path relocation."""
    tree = ast.parse(Path(local_cell_store.__file__).read_text())
    read = [
        node.args[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"get", "getenv"}
        and isinstance(node.func.value, (ast.Attribute, ast.Name))
        and "environ" in ast.dump(node.func.value)
        and node.args
    ]
    assert [ast.unparse(a) for a in read] == ["ENV_STORE_DIR"], (
        "the only env this module may read is the store LOCATION")
    assert local_cell_store.ENV_STORE_DIR == "GEN_WORKER_LOCAL_CELLS_DIR", (
        "the name is a cross-repo contract with cozy-local's paths.go")


# ---------------------------------------------------------------------------
# 5. The trust boundary is STRUCTURAL — there is no publish path to find
# ---------------------------------------------------------------------------


_FORBIDDEN_IMPORTS = {
    "gen_worker.s3_transfer", "gen_worker.presigned_upload",
    "gen_worker._upload_transport", "gen_worker.media_transfer",
    "gen_worker.net", "gen_worker.callout", "gen_worker.transport",
    "gen_worker.fleet_cells", "gen_worker.hubio.client",
    "urllib", "urllib.request", "requests", "httpx", "boto3", "aiohttp",
}


def test_the_local_store_has_no_publish_path(store: Path) -> None:
    """A compile cell is user-generated EXECUTABLE code. Accepting one from a
    user machine into shared storage would let any user ship arbitrary code
    into other people's GPU workers, so the boundary is enforced by the
    ABSENCE of machinery, not by a flag anyone can flip."""
    tree = ast.parse(Path(local_cell_store.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            base = f"gen_worker.{mod.lstrip('.')}" if node.level else mod
            imported.add(base)
            imported.update(f"{base}.{a.name}" for a in node.names)
    hits = {
        i for i in imported
        if i in _FORBIDDEN_IMPORTS
        or any(i.startswith(f + ".") for f in _FORBIDDEN_IMPORTS)
    }
    assert not hits, f"the local cell store must never grow a publish path: {hits}"

    idents = {
        n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name)
    } | {
        n.attr.lower() for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    } | {
        n.name.lower() for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for word in ("upload", "publish", "presign", "put_object", "post", "http"):
        bad = {i for i in idents if word in i}
        assert not bad, f"the local cell store references {bad}"


def test_the_production_resume_bank_still_roots_in_the_local_store(
    store: Path,
) -> None:
    """pgw#1092 §4 landmine 1, closed. `aot_resume` is a SERVE-PATH module and
    its bank was rooted through `local_cells.store_root()` — the module
    pgw#1086 wave 1 deletes. The accessor moved here with its value unchanged,
    so no bank is orphaned and wave 1 is unblocked.
    """
    bank = aot_resume.bank_root("some-scope")
    assert bank.is_relative_to(local_cell_store.store_root())
    assert aot_resume.RESUME_DIRNAME in bank.parts

    resume_tree = ast.parse(Path(aot_resume.__file__).read_text())
    names = {
        n.value.id for n in ast.walk(resume_tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
    }
    assert "local_cells" not in names, (
        "aot_resume must not reach into the JIT store pgw#1086 deletes")


# ---------------------------------------------------------------------------
# 6. Reuse: ONE gate, and the local cell gets no weaker verification
# ---------------------------------------------------------------------------


class _Pipe:
    pass


@dataclass
class _Cfg:
    family: str = "micro-diffusion"
    lora_bucket: int = 0
    shapes: Tuple[Tuple[int, int], ...] = ((64, 64),)
    targets: Tuple[str, ...] = ("transformer",)
    text_lens: Tuple[int, ...] = (16,)
    guidance_scales: Tuple[float, ...] = (1.0,)
    regional: bool = False


class _Arm:
    """A real-shaped `ArmIdentity` stand-in: this box states no `sm`."""

    def __init__(self, token: str = ARM_A) -> None:
        self.token = token

    def facts_dict(self) -> Dict[str, str]:
        return {}


@pytest.fixture()
def armable(monkeypatch: pytest.MonkeyPatch) -> List[Path]:
    """`provision.arm_aot` succeeds; record WHICH artifact it was handed."""
    seen: List[Path] = []

    def _arm(pipe: Any, cfg: Any, cache_dir: Any, artifact: Path,
             bucket: int, meta: Any = None, *,
             verify_numerics: bool = False, **_kw: Any) -> AdoptOutcome:
        # pgw#1141 / §4.32: this is an ADOPTION — the bytes were proven at
        # their own mint — so it must NOT ask for the quality gate. Asserted
        # rather than absorbed: a `**kwargs` stub would have swallowed the
        # difference and this file would still read green if the tax came back.
        assert verify_numerics is False, (
            "the local store's ADOPT path asked for the mint-time gate")
        seen.append(Path(artifact))
        return AdoptOutcome.hit(str((meta or {}).get("cell_key") or ""))

    monkeypatch.setattr(fleet_cells.provision, "arm_aot", _arm)
    monkeypatch.setattr(
        fleet_cells.artifact_meta, "try_read_metadata",
        lambda p: {"cell_key": KEY_A, "family": "micro-diffusion"})
    # The pgw#1042 axis gate has its own tests (and, since pgw#1096, ONE
    # implementation shared with the delegated adopt). Here the runtime and
    # the cell agree, so the gate passes and what is under test is the
    # LOOKUP -> ARM wiring.
    monkeypatch.setattr(fleet_cells, "arm_axis_divergence", lambda arm, meta, **_kw: "")
    monkeypatch.setattr(fleet_cells.aot_serve, "note_aot_key", lambda k: None)
    monkeypatch.setattr(fleet_cells.activity_mod, "emit_event",
                        lambda *a, **k: None)
    monkeypatch.setattr(fleet_cells, "_FINALIZED", {})
    return seen


def test_a_second_boot_arms_from_this_machines_own_store_with_no_mint(
    store: Path, tmp_path: Path, armable: List[Path],
) -> None:
    """The product promise, at the seam that decides a miss: compile once, run
    forever. RED before: no lookup existed, so boot 2 opened a pending and paid
    the whole export again."""
    local_cell_store.store(
        _armable_artifact(tmp_path), key=KEY_A, family="micro-diffusion",
        arm_token=ARM_A)

    minted = fleet_cells.arm_from_local_store(
        _Pipe(), _Cfg(), None, 0, _Arm(), "micro-diffusion")  # type: ignore[arg-type]

    assert minted is not None
    assert minted.cell_key == KEY_A
    assert minted.ref.endswith("#" + KEY_A)
    assert armable == [store / "aot-cells" / KEY_A / "cell.tar.gz"], (
        "the arm must be handed the STORE's bytes, not a copy nobody hashed")
    assert fleet_cells._FINALIZED[ARM_A] is minted, (
        "a sibling pipe of the same record must re-arm these bytes in-process")


def test_a_local_cell_that_cannot_arm_is_dropped_not_retried_forever(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fleet_cells.provision, "arm_aot",
        lambda *a, **k: AdoptOutcome.miss("constants_unbound", "norm.weight"))
    monkeypatch.setattr(
        fleet_cells.artifact_meta, "try_read_metadata", lambda p: {"cell_key": KEY_A})
    monkeypatch.setattr(fleet_cells, "arm_axis_divergence", lambda arm, meta, **_kw: "")
    events: List[Tuple[str, str]] = []
    monkeypatch.setattr(
        fleet_cells.activity_mod, "emit_event",
        lambda kind, detail, phase="", duration_ms=0, **_kw: events.append((kind, phase)))
    local_cell_store.store(
        _armable_artifact(tmp_path), key=KEY_A, family="f", arm_token=ARM_A)

    assert fleet_cells.arm_from_local_store(
        _Pipe(), _Cfg(), None, 0, _Arm(), "f") is None  # type: ignore[arg-type]
    assert not local_cell_store.cell_dir(KEY_A).exists()
    assert ("local_cell_refused", "constants_unbound") in events


def test_a_local_cell_that_does_not_describe_this_runtime_is_refused_by_FACT(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pgw#1042 axis gate, on the local path, with NOTHING stubbed out.

    This is the RED for "change the toolchain (or the sm, or the envelope) and
    the stored cell must not arm": the cell states facts, this runtime states
    facts, and the FIRST one that differs is named. A store that skipped this
    would serve last month's toolchain's kernels after a torch upgrade.
    """
    monkeypatch.setattr(
        fleet_cells.provision, "arm_aot",
        lambda *a, **k: pytest.fail("a diverging cell must never reach the arm"))
    monkeypatch.setattr(
        fleet_cells.artifact_meta, "read_metadata",
        lambda p: {"cell_key": KEY_A, "family": "micro-diffusion",
                   "sm": "sm_120", "format": 2})
    events: List[Tuple[str, str]] = []
    monkeypatch.setattr(
        fleet_cells.activity_mod, "emit_event",
        lambda kind, detail, phase="", duration_ms=0, **_kw: events.append((kind, phase)))
    local_cell_store.store(
        _artifact(tmp_path), key=KEY_A, family="micro-diffusion", arm_token=ARM_A)

    assert fleet_cells.arm_from_local_store(
        _Pipe(), _Cfg(), None, 0, _Arm(), "micro-diffusion") is None  # type: ignore[arg-type]
    assert ("local_cell_refused", "key_axis_divergence") in events
    assert not local_cell_store.cell_dir(KEY_A).exists()


def test_the_local_cell_and_the_childs_cell_pass_the_SAME_gate(
    store: Path,
) -> None:
    """A local cell gets NO weaker verification. Structural, because "they agree
    today" is not the property — one gate is."""
    tree = ast.parse(Path(fleet_cells.__file__).read_text())
    calls: Dict[str, set] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls[node.name] = {
            c.func.id for c in ast.walk(node)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }
    assert "_arm_exported_cell" in calls["adopt_delegated_mint"]
    assert "_arm_exported_cell" in calls["arm_from_local_store"]
    # ...and neither reaches around it to the raw arm.
    bodies = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    for fn in ("adopt_delegated_mint", "arm_from_local_store"):
        reached = [
            c for c in ast.walk(bodies[fn])
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr == "arm_aot"
        ]
        assert not reached, f"{fn} must arm through the one gate, not around it"


def test_the_miss_path_consults_the_local_store_before_opening_a_mint(
    store: Path,
) -> None:
    """Ordering is the whole promise: a pending opened before the lookup is a
    machine that recompiles what it already has."""
    tree = ast.parse(Path(fleet_cells.__file__).read_text())
    policy = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_arming_policy")
    lookup_at = min(
        c.lineno for c in ast.walk(policy)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        and c.func.id == "arm_from_local_store")
    pending_at = min(
        c.lineno for c in ast.walk(policy)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        and c.func.id == "PendingSelfMint")
    assert lookup_at < pending_at


# ---------------------------------------------------------------------------
# 7. The salvage: the hub's refusal keeps the cell instead of burning it
# ---------------------------------------------------------------------------


def test_an_untrusted_refusal_keeps_the_cell_instead_of_discarding_it(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """th#1643 books the refusal as SUNK — *"a sealed cell was produced and
    thrown away"*. It is no longer thrown away: the machine learns its trust
    class from the hub's own code and keeps the bytes for its next boot.

    pgw#1183 moved WHERE that is true. The salvage this row used to drive — a
    `store` inside `_publish_async`'s refusal branch, racing the `finally`
    that was about to rmtree the same bytes — is DELETED, because the cell is
    written to the local CAS before it arms and is therefore already durable
    whatever the hub answers. What is left to prove here is the pair: the
    refusal still teaches the machine its trust class, and it does not disturb
    the durable copy.
    """
    artifact = _artifact(tmp_path, b"packed-cell-bytes")
    monkeypatch.setattr(fleet_cells.activity_mod, "emit_event",
                        lambda *a, **k: None)
    monkeypatch.setattr(fleet_cells, "_note_durable", lambda *a, **k: None)
    local_cell_store.store(
        artifact, key=KEY_A, family="micro-diffusion", arm_token=ARM_A,
        sink=local_cell_store.SINK_OWED)

    class _Refusing:
        def publish(self, family: str, art: Path, meta: dict,
                    mint_duration_ms: int = 0) -> str:
            raise fleet_cells.CellPublishRefused(
                "publish-intent refused (403): community_tier",
                status=403, code=local_cell_store.UNTRUSTED_REFUSAL_CODE)

    fleet_cells._publish_async(
        _Refusing(), "micro-diffusion",  # type: ignore[arg-type]
        local_cell_store.lookup(KEY_A).artifact,  # type: ignore[union-attr]
        {"cell_key": KEY_A}, cell_key_digest=KEY_A, arm_token=ARM_A,
    ).join(timeout=30)

    assert local_cell_store.trust_class() == local_cell_store.TRUST_UNTRUSTED
    kept = local_cell_store.lookup_for_arm(ARM_A)
    assert kept is not None and kept.key == KEY_A
    assert kept.artifact.read_bytes() == b"packed-cell-bytes"
    # And the obligation is CLOSED, not left owed: a hub that says "never"
    # must not be re-asked on every boot forever.
    assert kept.sink == local_cell_store.SINK_REFUSED


def test_a_transport_failure_is_not_a_trust_verdict(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fleet_cells.activity_mod, "emit_event",
                        lambda *a, **k: None)
    monkeypatch.setattr(fleet_cells, "_note_durable", lambda *a, **k: None)

    class _Broken:
        def publish(self, family: str, art: Path, meta: dict,
                    mint_duration_ms: int = 0) -> str:
            raise RuntimeError("connection reset")

    fleet_cells._publish_async(
        _Broken(), "f", _artifact(tmp_path),  # type: ignore[arg-type]
        {"cell_key": KEY_A}, cell_key_digest=KEY_A, arm_token=ARM_A,
    ).join(timeout=30)

    assert local_cell_store.trust_class() == ""
    assert local_cell_store.lookup_for_arm(ARM_A) is None


# ---------------------------------------------------------------------------
# 7b. WHY a machine keeps its cell — three sink facts, and none is a trust claim
# ---------------------------------------------------------------------------


class _Sink:
    def __init__(self, on: bool = True) -> None:
        self._on = on

    def enabled(self) -> bool:
        return self._on


def test_cozy_local_keeps_its_cell_without_ever_asking_a_hub(
    store: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The §4.28 cozy-local promise, and the bug this test exists to stop.

    cozy-local NEVER constructs a publisher and NEVER reaches a hub — so a
    design that waited for a hub's typed refusal before keeping anything would
    leave *"download model + code ONCE, compile ONCE, reuse forever"*
    permanently unimplemented on the one product it was written about. The
    reason must therefore be the ABSENCE OF A SINK, not a verdict that will
    never arrive.
    """
    assert local_cell_store.trust_class() == "", "no hub has said anything"
    assert fleet_cells.no_publish_sink_reason(None) == fleet_cells.KEEP_NO_PUBLISHER
    assert (fleet_cells.no_publish_sink_reason(_Sink(on=False))  # type: ignore[arg-type]
            == fleet_cells.KEEP_NO_PUBLISHER)


def test_a_probe_pod_keeps_its_cell_because_its_publish_is_DISARMED(
    store: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#980 removes `cells.publish_intent` from the parent's allowlist, so a
    probe's publish never reaches a hub and never produces a typed refusal.
    Read the predicate that OWNS that decision rather than sniffing the
    exception it raises."""
    from gen_worker.procsplit import actions

    monkeypatch.setattr(actions, "publish_disarmed", lambda: True)
    assert (fleet_cells.no_publish_sink_reason(_Sink())  # type: ignore[arg-type]
            == fleet_cells.KEEP_PUBLISH_DISARMED)

    monkeypatch.setattr(actions, "publish_disarmed", lambda: False)
    assert fleet_cells.no_publish_sink_reason(_Sink()) == ""  # type: ignore[arg-type]


def test_a_trusted_fleet_pod_with_a_live_sink_OWES_an_upload(
    store: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1183 INVERTED what this row asserts, and the old claim is the bug.

    It used to read *"a pod that CAN publish keeps no local copy, so the
    fleet's ephemeral disks are unchanged"* — which made the pods that mint
    for the fleet exactly the pods with no durable copy of what they minted,
    and is how a 1 h 37 m mint was destroyed by a `finally`. Every pod keeps
    its cell now (§1.5); what a live sink decides is only whether an UPLOAD is
    owed on top."""
    from gen_worker.procsplit import actions

    monkeypatch.setattr(actions, "publish_disarmed", lambda: False)
    assert fleet_cells.no_publish_sink_reason(_Sink()) == ""  # type: ignore[arg-type]

    # ...until the hub says otherwise, which is the ONLY trust input.
    local_cell_store.note_refusal(
        local_cell_store.UNTRUSTED_REFUSAL_CODE, "community_tier")
    assert (fleet_cells.no_publish_sink_reason(_Sink())  # type: ignore[arg-type]
            == fleet_cells.KEEP_HUB_ASSERTED_UNTRUSTED)


# ---------------------------------------------------------------------------
# 8. Accumulation — recorded, not swept (the follow-up this lane owes)
# ---------------------------------------------------------------------------


def test_nothing_evicts_and_the_accumulation_is_enumerable(
    store: Path, tmp_path: Path,
) -> None:
    """No GC in this lane, deliberately: a wrong sweep deletes an hour of
    compile. What accumulates is one cell per (graph x envelope x sm x
    toolchain) — every torch/gen-worker/settings move leaves its predecessor
    resident — and `stored_cells` is the fact base a future
    toolchain-aware sweep needs. An AGE-based sweep would be wrong: a cell
    nobody booted for six months is still exactly the cell the next boot wants.
    """
    for key in (KEY_A, KEY_B):
        local_cell_store.store(_artifact(tmp_path), key=key, family="f")

    resident = local_cell_store.stored_cells()
    assert {c.key for c in resident} == {KEY_A, KEY_B}
    assert all(c.bytes > 0 and c.stored_at > 0 for c in resident)
    assert not any(
        name in dir(local_cell_store)
        for name in ("gc", "sweep", "evict", "prune", "expire")), (
        "eviction is a follow-up with an owner, not a surprise in this module")
