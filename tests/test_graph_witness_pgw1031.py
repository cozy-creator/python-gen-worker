"""pgw#1031 — the cell key is the traced COMPUTATION, and two different
computations behind one declaration now key APART (option a, Paul-ruled).

The LIVE SIGHTING this file pins, measured 2026-08-10 during pgw#1079: the
gauntlet's ``micro-pad32`` and ``micro-pad32-branchy`` members are the same
model with two spellings of the same pad, so every DECLARED fact agrees —
signature, symbol ranges, pytree spec, constant FQNs, declared envelope —
while the traced bodies do not (112 nodes vs 102). Before pgw#1031's depth fix
``class_hash`` folded only the declared/interface facts, so the key could not
see the body and both minted under ONE ``ck1`` key. The fix folds the
node-level ``graph_witness`` into ``class_hash`` (facts v3), so the ``graph``
axis is now the computation: the two members derive DIFFERENT keys and a
collision is a MISS (eager + mint), the cheap outcome.

The rows:

* :func:`test_the_bodies_now_key_apart` traces both families through the mint's
  OWN ``trace_for_key`` and asserts the keying blocks are byte-identical on the
  INTERFACE half while the graph witnesses differ — AND the derived keys now
  differ. It is the RED→GREEN proof: pre-fix one key, post-fix two.
* :func:`test_the_witness_backstops_a_residual_collision` runs the
  defense-in-depth backstop over the same two traced blocks: even given a cell
  built from one member's blocks, the matching pod admits and the colliding pod
  is REFUSED by a reason naming both digests. The witness stays as belt-and-
  braces beneath the now-sound key.

**No compile anywhere.** ``trace_for_key`` is ``torch.export`` and stops there
(Paul 2026-08-10: tracing for key derivation is explicitly permitted locally;
mints are not). The weights are the rig's 1.1 MB generated checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")

from gen_worker import aot_identity, aot_serve, boot_key  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"
if str(MICRO_SRC) not in sys.path:
    sys.path.insert(0, str(MICRO_SRC))

#: The two gauntlet members. Same weights, same declaration, two pad spellings.
PAIR = ("micro-pad32", "micro-pad32-branchy")


@pytest.fixture(scope="module", autouse=True)
def _gpu_runtime() -> Any:
    """A key-complete runtime on a card-less box — probes only.

    Module-scoped because the traces are: ``sm`` is a KEY AXIS, so the fold in
    :func:`_trace` needs it, and that runs inside the module-scoped fixture.
    """
    from gen_worker import compile_cache

    full = {
        "sku": "l4", "sm": "sm_89", "torch": str(torch.__version__),
        "triton": "3.6.0", "cuda": "13.0",
        "image_digest": "sha256:" + "ab" * 32,
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(compile_cache, "runtime_key", lambda: dict(full))
        mp.setattr(aot_serve, "runtime_key", lambda: {
            "sku": full["sku"], "sm": full["sm"], "torch": full["torch"],
            "cuda": full["cuda"]})
        yield


def _trace(
    vehicle_name: str, tree: Path,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """``({entry: keying block}, {entry: cg-key-v1 key}, declared envelope)`` — trace
    only. pgw#1176: a declaration folds to a KEY SET, not one key."""
    from harness import rig_vehicles

    from gen_worker import aot_mint, fleet_cells
    from gen_worker.cli.run import run_setup
    from gen_worker.child_preflight import pick_compile_target
    from gen_worker.registry import collect_endpoints

    veh = rig_vehicles.vehicle(vehicle_name)
    cfg = veh.compile_cell()
    specs = collect_endpoints(list(veh.modules))
    chosen = next(s for s in specs if s.name == veh.function)
    loaded = run_setup(
        chosen.cls(), {"pipeline": str(tree)}, arm_compile=False,
        return_loaded=True) or {}
    _slot, pipeline = pick_compile_target(loaded, cfg)
    export_spec = fleet_cells.aot_export_spec(pipeline, cfg)
    decl = aot_mint.export_declaration(export_spec.family)
    blocks: Dict[str, Any] = {}
    for traced in aot_mint.trace_for_key(pipeline, export_spec, decl):
        blocks[traced.name] = traced.block
        traced.program = None  # the largest object here; nothing below reads it
    envelope = fleet_cells.declared_envelope_block(cfg)
    entry_keys, _hashes, _manifest = boot_key.fold(
        blocks, family=export_spec.family, precision="", strict=True,
        lora_bucket=int(cfg.lora_bucket or 0), envelope=envelope)
    return blocks, entry_keys, envelope


@pytest.fixture(scope="module")
def traced_pair(
    _gpu_runtime: Any, tmp_path_factory: pytest.TempPathFactory,
) -> Dict[str, Any]:
    """Both families traced once — the export is the expensive part."""
    from harness import rig_vehicles

    root = tmp_path_factory.mktemp("pgw1031")
    tree = rig_vehicles.vehicle(PAIR[0]).build_checkpoint(root / "checkpoint")
    out: Dict[str, Any] = {}
    for name in PAIR:
        blocks, key, envelope = _trace(name, tree)
        out[name] = {"blocks": blocks, "key": key, "envelope": envelope}
    return out


def _canon(block: Any) -> str:
    return json.dumps(block, sort_keys=True, separators=(",", ":"))


def test_the_bodies_now_key_apart(traced_pair: Dict[str, Any]) -> None:
    """GREEN (was RED): the INTERFACE half is identical, the bodies differ,
    and the depth fix makes the two members derive DIFFERENT keys.

    Pre-pgw#1031 this asserted ONE shared key (the collision). The fix folds
    ``graph_witness`` into ``class_hash``, so the same identical-declaration /
    different-body pair keys apart — a collision is now a MISS, not a wrong hit.
    """
    fixed, branchy = traced_pair[PAIR[0]], traced_pair[PAIR[1]]
    assert set(fixed["blocks"]) == set(branchy["blocks"]) == {"transformer"}

    a, b = fixed["blocks"]["transformer"], branchy["blocks"]["transformer"]
    interface_a = {k: v for k, v in a.items() if k != "graph_witness"}
    interface_b = {k: v for k, v in b.items() if k != "graph_witness"}
    assert _canon(interface_a) == _canon(interface_b), (
        "the INTERFACE half of the block is expected to be IDENTICAL — the two "
        "members declare the same ingress; if this fails the pair no longer "
        "isolates the body axis and pgw#1031's sighting needs re-stating")

    # The two graphs really are different computations…
    assert a["graph_witness"] and b["graph_witness"]
    assert a["graph_witness"] != b["graph_witness"], (
        "the witness must separate the bodies; equal digests here would mean "
        "the pair no longer differs in its computation")

    # …and THE FIX: the key now sees the body, so the members key apart.
    # the claim is now per GRAPH CLASS, which is what it always
    # meant — the two declarations trace the same class names with different
    # bodies, so every shared class must key apart. Asserting it per class is
    # strictly stronger than asserting it once over a combined digest, which
    # could have hidden one colliding class behind another that differed.
    shared = sorted(set(fixed["key"]) & set(branchy["key"]))
    assert shared, "the pair no longer shares a class name; the axis is untested"
    assert all(fixed["key"][n] != branchy["key"][n] for n in shared), (
        "THE FIX (pgw#1031 option a): identical ingress, different bodies must "
        "derive DIFFERENT keys. A red here means the graph axis went body-blind "
        "again — class_hash must fold graph_witness")


def test_the_witness_backstops_a_residual_collision(
    traced_pair: Dict[str, Any],
) -> None:
    """Defense-in-depth: given a cell built from one member's blocks, the
    matching pod adopts and the colliding pod is refused by name.

    The key now separates the two members (``test_the_bodies_now_key_apart``),
    so pull-by-key never hands the wrong cell over in the first place. This
    proves the backstop still holds beneath the sound key: were a witness-blind
    cell ever handed over, the adopt path still refuses on the witness."""
    fixed, branchy = traced_pair[PAIR[0]], traced_pair[PAIR[1]]
    # ONE artifact, ONE class — so the witness backstop is asked
    # about the class this artifact carries, which is the only thing it could
    # ever honestly answer about.
    name = sorted(fixed["blocks"])[0]
    cell = aot_serve.entry_metadata(
        family=PAIR[0], precision="", cell_key=fixed["key"],
        name=name, entry=fixed["blocks"][name],
        strict_export=True, lora_bucket=0)

    mine = {name: boot_key.graph_witnesses_of(fixed["blocks"])[name]}
    theirs = {name: boot_key.graph_witnesses_of(branchy["blocks"])[name]}

    assert aot_identity.verify_graph_witness(cell, mine) == "", (
        "the pod whose graph this cell WAS compiled from must admit it")

    refusal = aot_identity.verify_graph_witness(cell, theirs)
    assert refusal, "the colliding pod must be REFUSED, not admitted"
    assert "transformer" in refusal
    assert mine["transformer"] in refusal and theirs["transformer"] in refusal, (
        "a refusal that does not name BOTH digests cannot be acted on")


def test_a_witnessless_cell_is_refused_not_skipped() -> None:
    """Fail-closed: silence is a refusal (``verify_declared_identity``'s rule).

    A pre-pgw#1031 cell records no witness, so it cannot be shown to compute
    this pod's graph — and 'cannot be shown to match' is what a refusal means.
    """
    meta = {"entry": {"name": "unet", "class_hash": "aa" * 8}}
    reason = aot_identity.verify_graph_witness(meta, {"unet": "d" * 16})
    assert "graph_witness" in reason and "unet" in reason

    assert aot_identity.verify_graph_witness(
        {"entry": {"name": "unet", "graph_witness": "d" * 16}}, {})
    assert aot_identity.verify_graph_witness(
        {}, {"unet": "d" * 16})


# pgw#1176 DELETED `test_a_differing_class_set_is_refused`. Its subject was a
# CLASS SET on one artifact ("a partial agreement is not a narrower match, it
# is an unproven one") — and one artifact carries one class now, so the set it
# guarded cannot exist. Porting it would have preserved the collection in the
# suite after removing it from the code. What survives is the row above: an
# entry that records no witness is refused, never skipped.


def _meta(row: Dict[str, Any], family: str) -> Dict[str, Any]:
    """One family's cell metadata, as the mint would stamp it."""
    from gen_worker import cell_key as ck, compile_cache as cc, env_seal

    name = sorted(row["blocks"])[0]
    meta = aot_serve.entry_metadata(
        family=family, precision="", cell_key=row["key"][name],
        name=name, entry=row["blocks"][name],
        strict_export=True, lora_bucket=0)
    meta["kind"] = ck.EXPORTED_KIND
    meta["toolchain"] = dict(cc.toolchain_digest())
    meta[env_seal.SEAL_KEY] = env_seal.effective_seal()
    return meta


def test_the_key_now_separates_what_the_gates_could_not(
    traced_pair: Dict[str, Any],
) -> None:
    """GREEN (was RED): the depth fix separates the two cells AT THE KEY.

    Before pgw#1031 nothing that shipped separated these cells: pod
    ``micro-pad32-branchy`` derived the SAME key as ``micro-pad32``, the hub
    answered its pull with the wrong cell, and every admission gate agreed
    (``verify_declared_identity`` clean on all four axes, ``verify_contract``
    self-consistent) — only the arm-time numerics tolerance stood between that
    and wrong output. The fix folds the body into ``graph``, so:

    * the two members derive DIFFERENT keys — pull-by-key never even offers
      cell A to pod B; the collision is a MISS, not a wrong hit;
    * were the wrong cell forced across anyway, ``verify_declared_identity``
      now REFUSES on the ``graph`` axis (this class's ``class_hash`` differs); and
    * the witness backstop still refuses beneath both.
    """
    fixed, branchy = traced_pair[PAIR[0]], traced_pair[PAIR[1]]
    cell_a = _meta(fixed, PAIR[0])

    # The expectation pod B states from its OWN traced facts and its OWN
    # runtime — nothing borrowed from the cell it is about to be handed.
    expected_b = aot_identity.artifact_identity(_meta(branchy, PAIR[1]))
    assert expected_b.cell_key != cell_a["cell_key"], (
        "THE FIX: pod B must derive a different key from cell A's, so the hub "
        "never answers B's pull with A. A red here means the key went body-"
        "blind — class_hash must fold graph_witness")

    # Cell A is still self-consistent (its own stamp verifies)…
    assert aot_serve.verify_contract(cell_a) == ""
    # …but the identity gate now REFUSES the cross, naming the graph axis.
    refusal = aot_identity.verify_declared_identity(cell_a, expected_b)
    assert refusal, "the identity gate must refuse the wrong cell post-fix"

    # …and the witness backstop refuses it too (defense-in-depth).
    assert aot_identity.verify_graph_witness(
        cell_a, boot_key.graph_witnesses_of(branchy["blocks"]))


# ---------------------------------------------------------------------------
# The ADOPT PATH — where a collision would actually arm another graph's kernels
# ---------------------------------------------------------------------------


def _artifact(tmp_path: Path, meta: Dict[str, Any]) -> Path:
    """A cell whose only interesting member is its ``metadata.json``."""
    import io
    import tarfile

    path = tmp_path / "cell.tar.gz"
    blob = json.dumps(meta).encode()
    with tarfile.open(path, mode="w:gz") as tar:
        info = tarfile.TarInfo(aot_serve.METADATA_NAME)
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    return path


def _attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
             derived: Any, artifact: Path) -> Any:
    from gen_worker import boot_adopt, boot_key, cell_resolve

    class _Cell:
        publisher_org = "org-a"
        cell_ref = "root/family-micro-pad32"
        publisher_tier = "platform"

    monkeypatch.setattr(boot_key, "derive", lambda **_kw: derived)
    monkeypatch.setattr(cell_resolve, "resolve", lambda *_a, **_k: _Cell())
    monkeypatch.setattr(
        cell_resolve, "materialize", lambda *_a, **_k: artifact)

    class _Cfg:
        family = "micro-pad32"
        targets = ("transformer",)
        shapes = ((12, 12),)
        text_lens = ()
        guidance_scales = ()
        lora_bucket = 0

    return boot_adopt.attempt(
        function="generate-pad32", modules=("m",), cfg=_Cfg(), slots={},
        declared_hint=1, envelope={"shapes": [[12, 12]]},
        work_root=tmp_path)


def _derived_key(traced: Dict[str, Any], family: str) -> Any:
    """A ``DerivedKey`` carrying one family's real traced witnesses."""
    blocks = traced[family]["blocks"]
    return boot_key.DerivedKey(
        entry_keys=dict(traced[family]["key"]), class_hashes={}, manifest="",
        workers=1, width_reason="pgw#1031 fixture", traced=len(blocks),
        memo="miss", wall_ms=1,
        graph_witnesses=boot_key.graph_witnesses_of(blocks))


def test_the_adopt_path_admits_the_pod_whose_graph_it_is(
    traced_pair: Dict[str, Any], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixed = traced_pair[PAIR[0]]
    name = sorted(fixed["blocks"])[0]
    meta = aot_serve.entry_metadata(
        family=PAIR[0], precision="", cell_key=fixed["key"][name],
        name=name, entry=fixed["blocks"][name],
        strict_export=True, lora_bucket=0)
    # a boot returns ONE outcome per declared class; this fixture
    # traces one, so the unpack ASSERTS that arity.
    (out,) = _attempt(
        monkeypatch, tmp_path, _derived_key(traced_pair, PAIR[0]),
        _artifact(tmp_path, meta))
    assert out.adopted and out.reason == "hit"


def test_the_adopt_path_refuses_the_colliding_pod(
    traced_pair: Dict[str, Any], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """THE backstop: a FORCED cross — pod B handed cell A — is refused, and it
    says which. Post-fix the keys differ, so pull-by-key never offers this cross
    on its own; the monkeypatched ``derive`` forces it to prove the backstop
    still refuses beneath the sound key. Without it this returns ``hit`` and the
    pod arms ``micro-pad32``'s kernels for ``micro-pad32-branchy``'s
    computation, with only the arm-time numerics tolerance between that and
    served output.
    """
    fixed, branchy = traced_pair[PAIR[0]], traced_pair[PAIR[1]]
    name = sorted(fixed["blocks"])[0]
    # the fix: the shared class keys apart
    assert fixed["key"][name] != branchy["key"][name]
    meta = aot_serve.entry_metadata(
        family=PAIR[0], precision="", cell_key=fixed["key"][name],
        name=name, entry=fixed["blocks"][name],
        strict_export=True, lora_bucket=0)
    (out,) = _attempt(
        monkeypatch, tmp_path, _derived_key(traced_pair, PAIR[1]),
        _artifact(tmp_path, meta))
    assert not out.adopted
    assert out.reason == "graph_witness_mismatch"
    assert boot_key.graph_witnesses_of(
        branchy["blocks"])["transformer"] in out.detail
