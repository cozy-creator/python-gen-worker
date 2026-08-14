"""pgw#1031 — the cell key is the traced COMPUTATION, and two different
computations behind one declaration now key APART (option a, Paul-ruled).

The case this file pins: the gauntlet's ``micro-pad32`` and
``micro-pad32-branchy`` members are the same model with two spellings of the
same pad, so every DECLARED fact agrees — signature, symbol ranges, pytree spec,
constant FQNs, declared envelope — while the traced bodies do not (112 nodes vs
102). A ``class_hash`` folding only the declared/interface facts cannot see the
body, and both mint under ONE ``ck1`` key. Folding the node-level
``graph_witness`` into ``class_hash`` (facts v3) makes the ``graph`` axis the
computation: the two members derive DIFFERENT keys and a collision is a MISS
(eager + mint), the cheap outcome.

The rows:

* :func:`test_the_bodies_now_key_apart` traces both families through the mint's
  OWN ``trace_for_key`` and asserts the keying blocks are byte-identical on the
  INTERFACE half while the graph witnesses differ — AND the derived keys now
  differ. It is the RED→GREEN proof: pre-fix one key, post-fix two.
TCG derives the compiled-graph key from this graph-class declaration. There is
no worker-side witness comparison beneath it: a different witness is a
different key and therefore a resolve miss.

**No compile anywhere.** ``trace_for_key`` is ``torch.export`` and stops there
(tracing for key derivation is explicitly permitted locally; mints are not).
The weights are the rig's 1.1 MB generated checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")

from gen_worker import aot_serve, boot_key  # noqa: E402

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
    from gen_worker.api.export_contract import export_declaration
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
    decl = export_declaration(export_spec.family)
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
    # The claim is now per GRAPH CLASS, which is what it always
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
    * TCG imports and resolves by that exact key, so there is no residual
      worker projection that can reinterpret the class as a match.
    """
    fixed, branchy = traced_pair[PAIR[0]], traced_pair[PAIR[1]]
    cell_a = _meta(fixed, PAIR[0])

    cell_b = _meta(branchy, PAIR[1])
    assert cell_b["cell_key"] != cell_a["cell_key"], (
        "THE FIX: pod B must derive a different key from cell A's, so the hub "
        "never answers B's pull with A. A red here means the key went body-"
        "blind — class_hash must fold graph_witness")

    # Each declaration remains internally self-consistent; separation belongs
    # to the key, not to a second post-hoc comparison.
    assert aot_serve.verify_contract(cell_a) == ""
    assert aot_serve.verify_contract(cell_b) == ""
