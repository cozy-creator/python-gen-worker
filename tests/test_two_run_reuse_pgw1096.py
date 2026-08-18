"""pgw#1096 — compile once, run forever, across PROCESS DEATH.

§4.28's product claim is not about one process: *"download model + code ONCE,
compile ONCE, and every subsequent RUN of that code reuses the same compiled
compiled graph."* The unit tests in ``test_aot_local_mint_pgw1096`` prove the store and
the gate inside one interpreter, where `fleet_compiled_graphs._FINALIZED` — the
in-process arm-token -> compiled graph index — is warm and could mask a store that does
not actually work.

This one proves the claim it is: TWO fresh OS processes, sharing nothing but a
directory on disk. Run 1 mints and keeps; run 2 starts cold, with an empty
`_FINALIZED`, and must arm WITHOUT opening a mint. If the memo, the CAS layout
or the sidecar record were wrong in any way that a warm process papers over,
run 2 opens a `PendingSelfMint` and this test says so.

WHAT IS REAL HERE: the store (a real TCG envelope in a real tensorfs CAS, real
atomic replace), the memo, `fleet_compiled_graphs._arming_policy` — the actual production
arming brain, entered the way the executor enters it — the ordering (local
check before the pending), and process death between the runs.

pgw#1283 — byte custody moved to TCG, and the third run below is what that
buys. Corruption in the CAS is a TCG STORAGE quarantine, which is repairable;
it is deliberately NOT this worker's verdict, so a repaired CAS arms again with
no mint. On master the same rot dropped the compiled graph and cost a full GPU pod run.

WHAT IS FAKED, and why: the COMPILE. Paul's standing rule (2026-08-10) is that
no mint, compile or AOTI link runs on the shared dev box — those go to a pod.
So the mint child is not spawned and `provision.arm_aot` is stubbed: this test
is about the STORE and the REUSE DECISION, and it is honest about the fact that
"a real AOTI compiled graph arms on a real card" is a claim only a pod can make. That is
the pod leg's job, and it is owed, not assumed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Run 1 and run 2 are the SAME program under a flag, deliberately: the thing
#: under test is that the second run of one program behaves differently only
#: because the first run left something on disk.
_PROGRAM = r'''
import json, os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import tcg_artifacts
from gen_worker import fleet_compiled_graphs, local_compiled_graph_store
from gen_worker.compiled_graph_adopt import AdoptOutcome

MODE = sys.argv[1]
ARTIFACT = Path(os.environ["PGW1096_ARTIFACT"])
SOURCE = Path(os.environ["PGW1096_SOURCE"])
CAS = Path(os.environ["PGW1096_CAS"])
KEY = tcg_artifacts.key_of(SOURCE)
ARM = fleet_compiled_graphs.ARM_SCHEME + "-" + "9" * fleet_compiled_graphs.ARM_DIGEST_HEX
FAMILY = "micro-diffusion"


class Pipe:
    pass


@dataclass
class Cfg:
    family: str = FAMILY
    lora_bucket: int = 0
    shapes: Tuple[Tuple[int, int], ...] = ((64, 64),)
    targets: Tuple[str, ...] = ("transformer",)
    text_lens: Tuple[int, ...] = (16,)
    guidance_scales: Tuple[float, ...] = (1.0,)
    regional: bool = False


class Arm:
    token = ARM

    def facts_dict(self) -> Dict[str, str]:
        return {}


# --- the compile is FAKED; everything below it is production code ------------
opened_mints = []
fleet_compiled_graphs.provision.arm_aot = (            # type: ignore[assignment]
    lambda *a, **k: AdoptOutcome.hit(KEY))
fleet_compiled_graphs.artifact_meta.try_read_metadata = (  # type: ignore[assignment]
    lambda p: {"compiled_graph_key": KEY, "family": FAMILY})
fleet_compiled_graphs.arm_axis_divergence = lambda arm, meta, **_kw: ""   # type: ignore[assignment]
fleet_compiled_graphs.activity_mod.emit_event = lambda *a, **k: None  # type: ignore[assignment]

_real_pending = fleet_compiled_graphs.PendingSelfMint


def _spy(*a: Any, **k: Any) -> Any:
    opened_mints.append(1)
    return _real_pending(*a, **k)


fleet_compiled_graphs.PendingSelfMint = _spy           # type: ignore[assignment]

if MODE == "mint":
    # Stand in for the child's packed compiled graph. On a pod this comes out of a real
    # AOTI link; here it is a real TCG envelope built without torch, because
    # since pgw#1283 the store hands its bytes to `Engine.import_artifact` and
    # an artifact that does not unpack and restate its own key is refused.
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_bytes(SOURCE.read_bytes())
    reason = fleet_compiled_graphs.no_publish_sink_reason(None)
    stored = local_compiled_graph_store.store(
        ARTIFACT, key=KEY, family=FAMILY, arm_token=ARM, cas_root=CAS)
    materialized = local_compiled_graph_store.materialize(KEY, cas_root=CAS)
    print("RESULT " + json.dumps({
        "run": 1, "keep_reason": reason,
        "stored": stored is not None,
        "key": KEY,
        "bytes": stored.bytes if stored else 0,
        "stored_at_path": str(materialized or ""),
    }))
else:
    # A COLD process: no `_FINALIZED`, no pending, nothing warm. The only
    # thing that exists is what run 1 wrote to disk.
    assert not fleet_compiled_graphs._FINALIZED, "a fresh process must start with no index"
    minted = fleet_compiled_graphs.arm_from_local_store(
        Pipe(), Cfg(), CAS, 0, Arm(), FAMILY)
    print("RESULT " + json.dumps({
        "run": 2,
        "armed": minted is not None,
        "compiled_graph_key": getattr(minted, "compiled_graph_key", ""),
        "artifact": str(getattr(minted, "artifact", "")),
        "mints_opened": len(opened_mints),
        "resident": [c.key for c in local_compiled_graph_store.stored_compiled_graphs()],
        "verdict": local_compiled_graph_store.verdict_of(KEY),
    }))
'''


def _source(tmp_path: Path) -> Path:
    """One real TCG artifact, built once per test."""
    import tcg_artifacts

    return tcg_artifacts.build(tmp_path / "source" / "cell.tar.gz")


def _run(mode: str, store: Path, artifact: Path, source: Path,
         cas: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO / "tests"), env.get("PYTHONPATH", "")])
    env["GEN_WORKER_LOCAL_CELLS_DIR"] = str(store)
    env["PGW1096_ARTIFACT"] = str(artifact)
    env["PGW1096_SOURCE"] = str(source)
    env["PGW1096_CAS"] = str(cas)
    proc = subprocess.run(
        [sys.executable, "-c", _PROGRAM, mode],
        capture_output=True, text=True, env=env, timeout=300)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise AssertionError(
        f"run {mode!r} produced no result\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")


def _cas_object(cas: Path) -> Path:
    """The one CAS object holding the artifact — the bytes TCG vouches for."""
    objects = sorted(
        (p for p in cas.rglob("*") if p.is_file()),
        key=lambda p: p.stat().st_size)
    assert objects, f"no CAS objects under {cas}"
    return objects[-1]


def test_run_one_mints_and_keeps_run_two_reuses_with_no_mint(
    tmp_path: Path,
) -> None:
    """Paul's compile-once-run-forever, across process death, measured."""
    store = tmp_path / "cozy-compiled graphs"
    cas = tmp_path / "cas"
    artifact = tmp_path / "mint" / "cell.tar.gz"
    source = _source(tmp_path)

    one = _run("mint", store, artifact, source, cas)
    assert one["stored"] is True
    # cozy-local's reason: no publisher was ever constructed, and no hub will
    # ever refuse this machine, because it never talks to one.
    assert one["keep_reason"] == "no_publish_sink"
    assert one["bytes"] == source.stat().st_size

    # The mint's own workdir is gone, exactly as `_publish_async`'s `finally`
    # and `abandon_self_mint` leave it. Only the store survives — which is the
    # point: run 2 must not be reading the mint's leftovers.
    artifact.unlink()

    two = _run("reuse", store, artifact, source, cas)
    assert two["armed"] is True, "the second run did not reuse the stored compiled graph"
    assert two["mints_opened"] == 0, (
        "the second run opened a mint — compile-once-run-forever is the whole "
        "product promise and this is what breaking it looks like")
    assert two["compiled_graph_key"] == one["key"]
    assert two["artifact"].startswith(str(store)), (
        "run 2 must serve the STORE's bytes, not a leftover from the mint")
    assert two["resident"] == [one["key"]]


def test_a_cold_run_with_an_EMPTY_store_mints_rather_than_inventing_a_hit(
    tmp_path: Path,
) -> None:
    """The negative that makes the positive mean something: with nothing on
    disk the same cold process reports no arm, so `armed=True` above is the
    store's doing and not the stub's."""
    two = _run("reuse", tmp_path / "empty-store", tmp_path / "unused.tar.gz",
               _source(tmp_path), tmp_path / "empty-cas")
    assert two["armed"] is False
    assert two["resident"] == []


def test_cas_rot_refuses_the_arm_WITHOUT_becoming_this_workers_verdict(
    tmp_path: Path,
) -> None:
    """pgw#1283 criterion 4, across process death — the repairable case.

    Run 1 keeps a compiled graph. The bytes rot IN THE CAS, which is a fact about a
    storage record: TCG quarantines it and no cold run may arm it. What must
    NOT happen is the worker recording its own :data:`VERDICT_QUARANTINED` —
    that verdict means "a parity/arm gate refused these bytes", it is terminal
    by design (§1.3.4 keeps such a compiled graph for forensics and never serves it), and
    writing it here would strand a compiled graph forever on a defect a re-store fixes.

    Run 3 proves the repair: the same artifact is stored again, TCG repairs its
    own record, and the compiled graph arms — with no mint, because the worker's
    admission was never destroyed.
    """
    store = tmp_path / "cozy-compiled graphs"
    cas = tmp_path / "cas"
    artifact = tmp_path / "mint" / "cell.tar.gz"
    source = _source(tmp_path)

    one = _run("mint", store, artifact, source, cas)
    assert one["stored"] is True

    rotted = bytearray(_cas_object(cas).read_bytes())
    rotted[0] ^= 0xFF          # same length; only the content moved
    _cas_object(cas).write_bytes(bytes(rotted))
    # The materialized copy goes too, or TCG would hand back the export it
    # already made instead of re-reading the record that rotted.
    (store / "aot-cells" / one["key"] / "cell.tar.gz").unlink()

    two = _run("reuse", store, artifact, source, cas)
    assert two["armed"] is False, "a compiled graph TCG cannot verify armed on a cold boot"
    assert two["verdict"] == "admitted", (
        "a CAS-storage quarantine was written into this worker's verdict; a "
        "repair can then never bring the compiled graph back")
    assert two["resident"] == [one["key"]], (
        "the worker's own record must survive rot it did not cause")

    three = _run("mint", store, artifact, source, cas)
    assert three["stored"] is True, "re-storing the same artifact must repair"
    four = _run("reuse", store, artifact, source, cas)
    assert four["armed"] is True, (
        "a repaired CAS record must arm again — the whole reason the two "
        "quarantines are kept apart")
    assert four["mints_opened"] == 0
