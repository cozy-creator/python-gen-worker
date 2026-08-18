"""pgw#1341 — the PUBLISH path reads five facts a TCG artifact cannot carry.

THE DEFECT, structural and provable for $0. Since pgw#1270 every artifact is
minted by TCG, and ``torchcg.artifact.validate_metadata`` refuses metadata whose
field set is not exactly ``artifact_meta.compiled_graph_metadata_fields()``::

    compiled_graph_format  compiled_graph_key  constant_folding_fenced
    graph_class  host_isa  kind  package_constants_in_so  sm  toolchain

``fleet_compiled_graphs._identity_axes`` read ``env_seal``, ``manifest_digest``, ``family``
and ``weight_lane``/``lora_bucket`` out of that dict, and ``intent_entry`` read
``sku`` and ``gen_worker``. None of the six is in the vocabulary. The first one
FAILS CLOSED, so the outcome was not a degraded row — it was
``CompiledGraphPublishRefused("compiled graph records no env_seal block")`` for **every real
artifact, unconditionally**. pgw#1340 made a pod ARM its own compiled graph;
this is the seam that decides whether the FLEET may adopt it, and it was shut.
Consequence, in money: every pod re-mints a graph another pod already compiled.

WHY IT IS A DESIGN CHANGE AND NOT A LINE. ``resume_owed_publishes`` discharges
an owed upload on a LATER BOOT, in a different process, potentially on a
different pod. Recomputing these from the live runtime would therefore attest
one machine's card, wheel and env seal against another machine's bytes — a
publish that is wrong in exactly the cases the retry exists for. The facts have
to be DURABLE beside the compiled graph, so they are: ``local_compiled_graph_store.MintProvenance``,
written into the same sidecar as the upload obligation, by the mint, at the
moment the bytes become durable.

WHAT THIS FILE PROVES, in the order the design has to be believed:

1. the RED, on a compiled graph TCG really built: no provenance -> refused BY NAME;
2. with the provenance the mint recorded, all five fields resolve;
3. **BOOT 1 mints in one PROCESS, BOOT 2 publishes in another** — a real
   subprocess writes the store, this process ships it over the real publish
   wire, and the hub's recorded intent carries the mint's facts, not the
   shipper's;
4. the controls: a genuinely provenance-less compiled graph still refuses, and the
   ARM/serve path is untouched by any of it.

THE FIXTURE FENCE (pgw#1340, applied here). Every compiled graph below is built by
``torchcg.artifact.build_metadata`` through ``tests/tcg_artifacts.py``. A
hand-written envelope is exactly how this survived two wheels: pgw#1277 recorded
the rule after finding the same class one block over — *"CI stayed green because
every fixture built the obsolete shape."*
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict

import pytest

import tcg_artifacts
from gen_worker import aot_serve, env_seal, fleet_compiled_graphs, graph_facts
from gen_worker import local_compiled_graph_store
from gen_worker._vendor.torchcg import identity as tcg_identity
from harness.compiled_graph_hub import local_compiled_graph_hub

FAMILY = "sd15"
LANE = "bf16-w16a16"
SKU = "l4"
GEN_WORKER = "0.123.0"
ARM_TOKEN = "arm2-" + "1" * 64

_FIXTURE_DIR = Path(tempfile.mkdtemp(prefix="pgw1341-"))
atexit.register(shutil.rmtree, _FIXTURE_DIR, True)
ARTIFACT = tcg_artifacts.build(_FIXTURE_DIR / "cell.tar.gz")
KEY = tcg_artifacts.key_of(ARTIFACT)
META: Dict[str, Any] = tcg_artifacts.metadata()
MANIFEST = graph_facts.manifest_digest([str(META["graph_class"]["class_hash"])])
SEAL_DIGEST = env_seal.seal_digest(env_seal.effective_seal())


def _provenance(**kw: Any) -> local_compiled_graph_store.MintProvenance:
    facts: Dict[str, Any] = dict(
        env_seal=SEAL_DIGEST, lane=LANE, graph_contract=MANIFEST,
        sku=SKU, gen_worker=GEN_WORKER)
    facts.update(kw)
    return local_compiled_graph_store.MintProvenance(**facts)


# ---------------------------------------------------------------------------
# 1. the RED — every real artifact, unconditionally
# ---------------------------------------------------------------------------


def test_a_real_TCG_cell_states_none_of_the_five_publish_fields() -> None:
    """THE $0 PROOF, on the vocabulary itself.

    This is what makes the defect structural rather than a bad compiled graph: the five
    reads name fields the closed vocabulary does not contain, so no artifact
    that exists can answer any of them.
    """
    from gen_worker import artifact_meta

    vocabulary = artifact_meta.compiled_graph_metadata_fields()
    for field in (env_seal.SEAL_KEY, "manifest_digest", "family",
                  "weight_lane", "lora_bucket", "sku", "gen_worker"):
        assert field not in vocabulary, field
        assert field not in META, field


def test_the_publish_refuses_a_cell_this_machine_can_say_nothing_about() -> None:
    """THE RED, preserved and NARROWED (pgw#939: absence is a verdict).

    Before this issue the same refusal fired on a metadata field no compiled graph can
    carry — i.e. always. It now fires on the case it was written for, and only
    that one: bytes with no recorded mint provenance beside them.
    """
    with pytest.raises(fleet_compiled_graphs.CompiledGraphPublishRefused) as exc:
        fleet_compiled_graphs._identity_axes(
            FAMILY, dict(META), local_compiled_graph_store.MintProvenance())
    assert "no mint provenance" in str(exc.value)
    assert "pgw#1341" in str(exc.value)


def test_the_five_fields_all_resolve_from_the_recorded_provenance() -> None:
    """The green half, field by field — each one named, so a silent blank in
    any of them is a failure here rather than an unarmable row on the fleet."""
    axes = fleet_compiled_graphs._identity_axes(FAMILY, dict(META), _provenance())
    key = tcg_identity.from_artifact_metadata(META)

    # The three KEY axes still come from the artifact: a compiled graph must corroborate
    # its own identity, and provenance must never be able to restate it.
    assert axes["graph"] == str(META["graph_class"]["class_hash"])
    assert axes["sm"] == str(META["sm"])
    assert axes["toolchain"] == tcg_identity.toolchain_axis_digest(
        dict(META["toolchain"]))
    assert key.value == META["compiled_graph_key"]

    # The five that could not be stated before.
    assert axes[fleet_compiled_graphs.ENV_SEAL_AXIS] == SEAL_DIGEST
    assert axes[fleet_compiled_graphs.GRAPH_CONTRACT_AXIS] == MANIFEST
    assert axes["family"] == FAMILY
    assert axes["lane"] == LANE

    entry, sku, gen_worker = fleet_compiled_graphs.intent_entry(
        FAMILY, dict(META), _provenance(), 347_940)
    assert (sku, gen_worker) == (SKU, GEN_WORKER)
    assert entry.compiled_graph_key == key.value
    assert entry.mint_duration_ms == 347_940


def test_no_publish_fact_is_read_from_the_compiled_graph_metadata() -> None:
    """The rule, enforced rather than described.

    A metadata dict that LIES — carrying the old field names with hostile
    values — must not move a single published axis. Anything that still reads
    them would show up here as an axis taking the forged value.
    """
    lying = dict(
        META,
        family="not-this-family",
        weight_lane="w4a4",
        lora_bucket=128,
        manifest_digest="forged-manifest",
        sku="h100",
        gen_worker="0.0.1",
        **{env_seal.SEAL_KEY: {"v": 99}},
    )
    entry, sku, gen_worker = fleet_compiled_graphs.intent_entry(
        FAMILY, lying, _provenance())
    axes = dict(entry.identity_axes)
    assert axes["family"] == FAMILY
    assert axes["lane"] == LANE
    assert axes[fleet_compiled_graphs.GRAPH_CONTRACT_AXIS] == MANIFEST
    assert axes[fleet_compiled_graphs.ENV_SEAL_AXIS] == SEAL_DIGEST
    assert (sku, gen_worker) == (SKU, GEN_WORKER)


# ---------------------------------------------------------------------------
# 2. the provenance is DURABLE — and survives the process that wrote it
# ---------------------------------------------------------------------------


#: BOOT 1. A real, separate interpreter: it mints (fixture bytes, no compile),
#: stages the compiled graph durable exactly as ``adopt_delegated_mint`` does, and dies.
#: Everything the publish will need has to be on disk when it does.
_BOOT_1 = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    from gen_worker import compile_cache as cc
    from gen_worker import fleet_compiled_graphs, local_compiled_graph_store

    artifact, cas, family, lane, seal, manifest, token, sku, gw = sys.argv[1:10]
    # THE CARD THIS BOX DOES NOT HAVE. `mint_provenance`'s derivation runs for
    # real — it calls these, and a cardless probe honestly answers "" — so the
    # two POD axes are given a pod's answer here rather than a mint's worth of
    # L4. Nothing else is faked: the store write, the sidecar, the artifact and
    # the publish wire are all production's.
    cc.runtime_key = lambda: {"sku": sku, "sm": "sm_89"}
    cc.gen_worker_version = lambda: gw
    pending = fleet_compiled_graphs.PendingSelfMint(
        family=family,
        arm_token=token,
        ref="root/family-%s#pending" % family,
        cfg=None,
        target=Path(artifact),
        mint_root=Path(artifact).parent,
        # A publisher this boot never gets to use: it is what makes the upload
        # OWED, which is the obligation boot 2 discharges.
        publisher=fleet_compiled_graphs.CompiledGraphPublisher(
            base_url="http://boot-1.invalid", worker_jwt=lambda: "jwt",
            image_digest="sha256:image"),
        cache_dir=Path(cas),
        arm_key=fleet_compiled_graphs.ArmIdentity(facts=tuple(sorted({
            "family": family, "lane": lane, "env_seal": seal,
            "subject": "", "targets": "unet", "dynamic": "{}",
            "regional": "0", "sm": "sm_89", "toolchain": "t",
            "compiled_graph_format": "1",
        }.items()))),
    )
    provenance = fleet_compiled_graphs.mint_provenance(pending, manifest=manifest)
    key = fleet_compiled_graphs._stage_durable(pending, Path(artifact), provenance)
    # The arm gate passing, which is what promotes the staged bytes: only an
    # ADMITTED compiled graph is ever owed to the sink.
    fleet_compiled_graphs._admit_durable(pending, key, key)
    print(json.dumps({"key": key, "provenance": provenance.as_dict()}))
    """
)


@pytest.fixture()
def store_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cozy-compiled graphs"
    monkeypatch.setenv(local_compiled_graph_store.ENV_STORE_DIR, str(root))
    return root


def _boot_1(tmp_path: Path, store_root: Path) -> Dict[str, Any]:
    """Run boot 1 in a REAL child process and return what it recorded."""
    staged = tmp_path / "mint" / "cell.tar.gz"
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ARTIFACT, staged)
    env = dict(os.environ)
    env[local_compiled_graph_store.ENV_STORE_DIR] = str(store_root)
    proc = subprocess.run(
        [sys.executable, "-c", _BOOT_1, str(staged), str(tmp_path / "cas"),
         FAMILY, LANE, SEAL_DIGEST, MANIFEST, ARM_TOKEN, SKU, GEN_WORKER],
        capture_output=True, text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_boot_1_records_the_mint_facts_beside_the_bytes(
    tmp_path: Path, store_root: Path,
) -> None:
    """The durable half, read back from disk by a process that did not mint.

    The sidecar — not the artifact — is what carries them, which is the whole
    design: TCG's vocabulary is closed and stays closed (pgw#1270).
    """
    recorded = _boot_1(tmp_path, store_root)
    assert recorded["key"] == KEY

    record = local_compiled_graph_store.read_record(KEY, store_root)
    assert record is not None
    assert record.sink == local_compiled_graph_store.SINK_OWED
    assert record.family == FAMILY
    assert record.provenance.env_seal == SEAL_DIGEST
    assert record.provenance.lane == LANE
    assert record.provenance.graph_contract == MANIFEST
    # The two POD axes are read on the pod that minted, by the mint itself.
    assert record.provenance.sku == SKU
    assert record.provenance.gen_worker == GEN_WORKER
    assert recorded["provenance"] == record.provenance.as_dict()

    # And the artifact still says nothing about any of it.
    from gen_worker import artifact_meta

    bytes_on_disk = local_compiled_graph_store.materialize(
        KEY, store_root, tmp_path / "cas")
    assert bytes_on_disk is not None
    packed = artifact_meta.read_metadata(bytes_on_disk)
    assert env_seal.SEAL_KEY not in packed
    assert "family" not in packed


def test_boot_2_PUBLISHES_what_boot_1_minted(
    tmp_path: Path, store_root: Path,
) -> None:
    """THE END-STATE PROOF. Two processes, one compiled graph, one publish.

    Boot 1 (a subprocess, now dead) minted and recorded. Boot 2 — this process,
    holding nothing but the store — discharges the owed upload over the REAL
    publish wire, and the hub receives the MINT's axes.

    Before pgw#1341 this could not happen at all: ``resume_owed_publishes``
    read the artifact's metadata, ``_identity_axes`` demanded an ``env_seal``
    block no TCG artifact has, and the thread died on
    ``CompiledGraphPublishRefused`` without a byte moving.
    """
    _boot_1(tmp_path, store_root)

    with local_compiled_graph_hub() as hub:
        publisher = fleet_compiled_graphs.CompiledGraphPublisher(
            base_url=hub.base, worker_jwt=lambda: "worker-jwt",
            image_digest="sha256:" + "e" * 64)
        threads = fleet_compiled_graphs.resume_owed_publishes(
            publisher, cas_root=tmp_path / "cas")
        assert len(threads) == 1, "the owed compiled graph must be shipped"
        threads[0].join(timeout=120)
        assert not threads[0].is_alive()

        assert len(hub.intents) == 1, hub.routes()
        intent = hub.intents[0]

    # The two POD axes the hub ATTESTS — empty before this issue, on every row.
    assert intent["axes"]["sku"] == SKU
    assert intent["axes"]["gen_worker"] == GEN_WORKER
    assert intent["axes"]["image_digest"] == "sha256:" + "e" * 64
    assert intent["family"] == FAMILY

    entry = intent["entries"][0]
    assert entry["compiled_graph_key"] == KEY
    axes = entry["identity_axes"]
    # pgw#903's pre-dlopen fence compares this; it was published empty.
    assert axes[fleet_compiled_graphs.GRAPH_CONTRACT_AXIS] == MANIFEST
    # The hub's ArtifactIdentity requires this; the publish never got here.
    assert axes[fleet_compiled_graphs.ENV_SEAL_AXIS] == SEAL_DIGEST
    # The store-row self-description, blank before.
    assert axes["family"] == FAMILY
    assert axes["lane"] == LANE

    # The obligation is discharged, durably, so a third boot ships nothing.
    record = local_compiled_graph_store.read_record(KEY, store_root)
    assert record is not None and record.sink == local_compiled_graph_store.SINK_DELIVERED


# ---------------------------------------------------------------------------
# 3. the controls
# ---------------------------------------------------------------------------


def test_a_cell_with_NO_recorded_provenance_still_refuses_by_name(
    tmp_path: Path, store_root: Path,
) -> None:
    """The refusal survives, narrowed — the positive control.

    A record written before this field existed (or by a route that recorded
    nothing) states no provenance. Publishing it would invent the hub's
    ``ArtifactIdentity`` out of the shipping pod's runtime, which is precisely
    the unsound thing. It is refused, by name, before a byte moves.
    """
    staged = tmp_path / "mint" / "cell.tar.gz"
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ARTIFACT, staged)
    stored = local_compiled_graph_store.store(
        staged, key=KEY, family=FAMILY, arm_token=ARM_TOKEN,
        sink=local_compiled_graph_store.SINK_OWED, root=store_root,
        cas_root=tmp_path / "cas")
    assert stored is not None and not stored.provenance.stated

    with local_compiled_graph_hub() as hub:
        publisher = fleet_compiled_graphs.CompiledGraphPublisher(
            base_url=hub.base, worker_jwt=lambda: "worker-jwt",
            image_digest="sha256:" + "e" * 64)
        threads = fleet_compiled_graphs.resume_owed_publishes(
            publisher, cas_root=tmp_path / "cas")
        assert len(threads) == 1
        threads[0].join(timeout=120)
        assert hub.intents == [], "nothing may reach the hub"

    record = local_compiled_graph_store.read_record(KEY, store_root)
    assert record is not None
    assert record.sink == local_compiled_graph_store.SINK_REFUSED


def test_a_pre_pgw1341_sidecar_reads_back_without_inventing_facts(
    tmp_path: Path, store_root: Path,
) -> None:
    """A record file with no ``provenance`` block at all.

    It must read as an EMPTY provenance — never as defaults, and never as an
    unreadable record: the bytes are fine and the compiled graph still ARMS. Only the
    publish is refused.
    """
    staged = tmp_path / "mint" / "cell.tar.gz"
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ARTIFACT, staged)
    assert local_compiled_graph_store.store(
        staged, key=KEY, family=FAMILY, root=store_root,
        cas_root=tmp_path / "cas") is not None
    path = local_compiled_graph_store.compiled_graph_dir(KEY, store_root) / local_compiled_graph_store.RECORD_NAME
    raw = json.loads(path.read_text())
    raw.pop("provenance")
    path.write_text(json.dumps(raw))

    record = local_compiled_graph_store.read_record(KEY, store_root)
    assert record is not None
    assert record.provenance == local_compiled_graph_store.MintProvenance()
    assert not record.provenance.stated


def test_the_ARM_side_is_untouched(tmp_path: Path, store_root: Path) -> None:
    """The negative control for the whole change.

    pgw#1340 made the compiled graph ARM. Nothing here may cost that: the serve-side
    lookup still returns admitted bytes, the arm-token memo still resolves, and
    the handback comparison still agrees with the runtime that minted it.
    """
    staged = tmp_path / "mint" / "cell.tar.gz"
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ARTIFACT, staged)
    assert local_compiled_graph_store.store(
        staged, key=KEY, family=FAMILY, arm_token=ARM_TOKEN,
        provenance=_provenance(), root=store_root,
        cas_root=tmp_path / "cas") is not None

    graph = local_compiled_graph_store.lookup(KEY, store_root, tmp_path / "cas")
    assert graph is not None
    assert graph.verdict == local_compiled_graph_store.VERDICT_ADMITTED
    assert graph.provenance.env_seal == SEAL_DIGEST
    assert local_compiled_graph_store.lookup_for_arm(
        ARM_TOKEN, store_root, tmp_path / "cas") is not None

    # pgw#1340's seam, unchanged: the compared axes are the three a compiled graph states.
    assert fleet_compiled_graphs.unstateable_arm_axes() == ()
    arm = fleet_compiled_graphs.ArmIdentity(facts=tuple(sorted({
        "family": FAMILY, "lane": LANE, "env_seal": SEAL_DIGEST,
        "subject": graph_facts.subject_digest(()), "targets": "unet",
        "dynamic": "{}", "regional": "0",
        "sm": str(META["sm"]),
        aot_serve.COMPILED_GRAPH_FORMAT_KEY: str(
            aot_serve.COMPILED_GRAPH_FORMAT),
        "toolchain": tcg_identity.toolchain_axis_digest(
            dict(META["toolchain"])),
    }.items())))
    assert fleet_compiled_graphs.arm_axis_divergence(arm, dict(META)) == ""
