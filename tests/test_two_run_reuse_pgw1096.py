"""pgw#1096 — compile once, run forever, across PROCESS DEATH.

§4.28's product claim is not about one process: *"download model + code ONCE,
compile ONCE, and every subsequent RUN of that code reuses the same compiled
cell."* The unit tests in ``test_aot_local_mint_pgw1096`` prove the store and
the gate inside one interpreter, where `fleet_cells._FINALIZED` — the
in-process arm-token -> cell index — is warm and could mask a store that does
not actually work.

This one proves the claim it is: TWO fresh OS processes, sharing nothing but a
directory on disk. Run 1 mints and keeps; run 2 starts cold, with an empty
`_FINALIZED`, and must arm WITHOUT opening a mint. If the memo, the CAS layout
or the digest record were wrong in any way that a warm process papers over,
run 2 opens a `PendingSelfMint` and this test says so.

WHAT IS REAL HERE: the store (real files, real digests, real atomic replace),
the memo, `fleet_cells._arming_policy` — the actual production arming brain,
entered the way the executor enters it — the ordering (local check before the
pending), and process death between the runs.

WHAT IS FAKED, and why: the COMPILE. Paul's standing rule (2026-08-10) is that
no mint, compile or AOTI link runs on the shared dev box — those go to a pod.
So the mint child is not spawned and `provision.arm_aot` is stubbed: this test
is about the STORE and the REUSE DECISION, and it is honest about the fact that
"a real AOTI cell arms on a real card" is a claim only a pod can make. That is
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
import io, json, os, sys, tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from gen_worker import fleet_cells, local_cell_store
from gen_worker.cell_adopt import AdoptOutcome

MODE = sys.argv[1]
KEY = "cg-key-v1-" + "c" * 56
ARM = fleet_cells.ARM_SCHEME + "-" + "9" * fleet_cells.ARM_DIGEST_HEX
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
fleet_cells.provision.arm_aot = (            # type: ignore[assignment]
    lambda *a, **k: AdoptOutcome.hit(KEY))
fleet_cells.artifact_meta.try_read_metadata = (  # type: ignore[assignment]
    lambda p: {"cell_key": KEY, "family": FAMILY})
fleet_cells.arm_axis_divergence = lambda arm, meta, **_kw: ""   # type: ignore[assignment]
fleet_cells.aot_serve.note_aot_key = lambda k: None      # type: ignore[assignment]
fleet_cells.activity_mod.emit_event = lambda *a, **k: None  # type: ignore[assignment]

_real_pending = fleet_cells.PendingSelfMint


def _spy(*a: Any, **k: Any) -> Any:
    opened_mints.append(1)
    return _real_pending(*a, **k)


fleet_cells.PendingSelfMint = _spy           # type: ignore[assignment]

if MODE == "mint":
    # Stand in for the child's packed cell. On a pod this is a real .pt2
    # tarball out of a real AOTI link. The store itself does not care what the
    # cell IS — but run 2 ARMS this cell, and since pgw#1098 the arm refuses
    # `cell_envelope_unreadable` before any other gate, so the stand-in has to
    # carry a readable `metadata.json` exactly as the real thing does.
    art = Path(os.environ["PGW1096_ARTIFACT"])
    art.parent.mkdir(parents=True, exist_ok=True)
    _meta = json.dumps(
        {"kind": "aot-inductor", "cell_key": KEY, "family": FAMILY}).encode()
    _body = b"a-real-cell-would-be-here" * 40
    with tarfile.open(art, mode="w:gz") as _tar:
        _mi = tarfile.TarInfo("metadata.json")
        _mi.size = len(_meta)
        _tar.addfile(_mi, io.BytesIO(_meta))
        _bi = tarfile.TarInfo("payload.bin")
        _bi.size = len(_body)
        _tar.addfile(_bi, io.BytesIO(_body))
    reason = fleet_cells.no_publish_sink_reason(None)
    stored = local_cell_store.store(
        art, key=KEY, family=FAMILY, arm_token=ARM)
    print("RESULT " + json.dumps({
        "run": 1, "keep_reason": reason,
        "stored": stored is not None,
        "stored_at_path": str(stored.artifact) if stored else "",
        "digest": stored.content_digest if stored else "",
    }))
else:
    # A COLD process: no `_FINALIZED`, no pending, nothing warm. The only
    # thing that exists is the directory run 1 wrote.
    assert not fleet_cells._FINALIZED, "a fresh process must start with no index"
    minted = fleet_cells.arm_from_local_store(
        Pipe(), Cfg(), None, 0, Arm(), FAMILY)
    print("RESULT " + json.dumps({
        "run": 2,
        "armed": minted is not None,
        "cell_key": getattr(minted, "cell_key", ""),
        "artifact": str(getattr(minted, "artifact", "")),
        "mints_opened": len(opened_mints),
        "resident": [c.key for c in local_cell_store.stored_cells()],
    }))
'''


def _run(mode: str, store: Path, artifact: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO / "tests"), env.get("PYTHONPATH", "")])
    env["GEN_WORKER_LOCAL_CELLS_DIR"] = str(store)
    env["PGW1096_ARTIFACT"] = str(artifact)
    proc = subprocess.run(
        [sys.executable, "-c", _PROGRAM, mode],
        capture_output=True, text=True, env=env, timeout=300)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise AssertionError(
        f"run {mode!r} produced no result\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")


def test_run_one_mints_and_keeps_run_two_reuses_with_no_mint(
    tmp_path: Path,
) -> None:
    """Paul's compile-once-run-forever, across process death, measured."""
    store = tmp_path / "cozy-cells"
    artifact = tmp_path / "mint" / "cell.tar.gz"

    one = _run("mint", store, artifact)
    assert one["stored"] is True
    # cozy-local's reason: no publisher was ever constructed, and no hub will
    # ever refuse this machine, because it never talks to one.
    assert one["keep_reason"] == "no_publish_sink"
    assert one["digest"].startswith("sha256:")

    # The mint's own workdir is gone, exactly as `_publish_async`'s `finally`
    # and `abandon_self_mint` leave it. Only the store survives — which is the
    # point: run 2 must not be reading the mint's leftovers.
    artifact.unlink()

    two = _run("reuse", store, artifact)
    assert two["armed"] is True, "the second run did not reuse the stored cell"
    assert two["mints_opened"] == 0, (
        "the second run opened a mint — compile-once-run-forever is the whole "
        "product promise and this is what breaking it looks like")
    assert two["cell_key"] == "cg-key-v1-" + "c" * 56
    assert two["artifact"].startswith(str(store)), (
        "run 2 must serve the STORE's bytes, not a leftover from the mint")
    assert two["resident"] == ["cg-key-v1-" + "c" * 56]


def test_a_cold_run_with_an_EMPTY_store_mints_rather_than_inventing_a_hit(
    tmp_path: Path,
) -> None:
    """The negative that makes the positive mean something: with nothing on
    disk the same cold process reports no arm, so `armed=True` above is the
    store's doing and not the stub's."""
    two = _run("reuse", tmp_path / "empty-store", tmp_path / "unused.tar.gz")
    assert two["armed"] is False
    assert two["resident"] == []


def test_a_corrupted_store_makes_the_cold_run_refuse_and_drop(
    tmp_path: Path,
) -> None:
    """RED, across processes: run 1 keeps a cell, the bytes rot on disk, and
    the cold run refuses it, drops it and is left with an empty store — one
    honest re-mint, never a wrong arm."""
    store = tmp_path / "cozy-cells"
    artifact = tmp_path / "mint" / "cell.tar.gz"
    one = _run("mint", store, artifact)
    stored = Path(one["stored_at_path"])

    rotted = bytearray(stored.read_bytes())
    rotted[0] ^= 0xFF          # same length; only the content moved
    stored.write_bytes(bytes(rotted))

    two = _run("reuse", store, artifact)
    assert two["armed"] is False, "a corrupted cell armed on a cold boot"
    assert two["resident"] == [], "the refused cell was not dropped"
