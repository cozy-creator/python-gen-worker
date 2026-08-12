"""pgw#903 — serving verifies AOT code identity against the current
ExecutionSpec, before ``dlopen``.

Two questions were being conflated. ``aot_serve.verify`` asks *can this runtime
execute these bytes* (sm/torch/cuda/host ISA, all probed here).
``verify_contract`` asks *is this envelope internally consistent*. Neither asks
*is this the artifact the hub named for this attempt* — nothing could, before
there was an immutable spec to ask against.

The fixture is deliberately model-free: identity is a metadata question, so a
GPU, a pipeline and a real ``.pt2`` are all irrelevant to it. Mutating exactly
one expected fact must refuse, and the refusal must name that fact.
"""

from __future__ import annotations

import importlib
import inspect
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from gen_worker import aot_identity, aot_serve, cell_key, env_seal
from gen_worker.aot_identity import ExpectedIdentity
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.plan import AttemptRef, PlanFactory

_TOOLCHAIN = {"cc": "sha256:aaa", "ld": "sha256:bbb"}
_SEAL = {"config": "cfg16", "inductor": "ind16", "loaded_libs": "libs16"}
_CLOSURE = {"gen_worker/executor.py": "sha256:ccc"}

#: One well-formed entry, so `unpack` can parse the envelope. Identity is a
#: metadata question — none of these values participate in it.
_ENTRIES = {
    "unet": {
        "target": "unet",
        "inputs": [{"name": "x", "dtype": "bfloat16", "shape": [1, 4]}],
        "constants": [],
    },
}


def _meta(**over: Any) -> dict[str, Any]:
    """An artifact's ``metadata.json`` as ``aot_mint`` writes it: the stamped
    envelope PLUS the identity blocks ``shared_identity_blocks`` merges in."""
    meta: dict[str, Any] = {
        "format": aot_serve.ARTIFACT_FORMAT,
        "kind": aot_serve.ARTIFACT_KIND,
        "family": "micro",
        "cell_key": "aot-inductor:k1",
        "manifest_digest": "gc_01",
        "entry": {"name": "unet", **dict(_ENTRIES["unet"])},
        env_seal.SEAL_KEY: dict(_SEAL),
        "toolchain": dict(_TOOLCHAIN),
        "code_closure": dict(_CLOSURE),
    }
    meta.update(over)
    return meta


def _expected(**over: Any) -> ExpectedIdentity:
    base: dict[str, Any] = {
        "cell_key": "aot-inductor:k1",
        "toolchain_digest": cell_key.facts_digest(_TOOLCHAIN),
        "env_seal_digest": env_seal.seal_digest(_SEAL),
        "graph_contract_digest": "gc_01",
        "publisher_org": "cozy",
    }
    base.update(over)
    return ExpectedIdentity(**base)


# --- the identity an artifact's own facts describe --------------------------


def test_identity_is_recomputed_from_the_recorded_facts_not_a_stamp() -> None:
    """``cell_key``'s standing discipline: a digest is derived from the facts it
    summarizes, so a stamp can never silently disagree with them."""
    have = aot_identity.artifact_identity(_meta())
    assert have.toolchain_digest == cell_key.facts_digest(_TOOLCHAIN)
    assert have.env_seal_digest == env_seal.seal_digest(_SEAL)
    assert have.graph_contract_digest == "gc_01"
    # An artifact cannot attest to its own publisher; that claim is the
    # hub-signed receipt's.
    assert have.publisher_org == ""


def test_a_matching_identity_passes() -> None:
    assert aot_identity.verify_declared_identity(_meta(), _expected()) == ""


# --- mutate exactly one expected fact ---------------------------------------


@pytest.mark.parametrize(
    "axis,mutated",
    [
        ("cell_key", {"cell_key": "aot-inductor:OTHER"}),
        ("toolchain_digest", {"toolchain_digest": "0" * 16}),
        ("env_seal_digest", {"env_seal_digest": "0" * 16}),
        ("graph_contract_digest", {"graph_contract_digest": "gc_02"}),
    ],
)
def test_one_mutated_fact_refuses_and_names_itself(axis: str, mutated: Any) -> None:
    reason = aot_identity.verify_declared_identity(_meta(), _expected(**mutated))
    assert reason.startswith(f"{axis}: expected ")
    # Both values, not just the verdict: the hub team reads these as events.
    assert "have " in reason
    assert reason.split("expected ")[1].split(",")[0] != reason.split("have ")[1]


@pytest.mark.parametrize(
    "axis,dropped",
    [
        ("cell_key", "cell_key"),
        ("toolchain_digest", "toolchain"),
        ("env_seal_digest", env_seal.SEAL_KEY),
        ("graph_contract_digest", "manifest_digest"),
    ],
)
def test_an_artifact_silent_on_an_axis_refuses_rather_than_skipping(
    axis: str, dropped: str,
) -> None:
    """Fail-closed: a cell that cannot state its own toolchain cannot be shown
    to match the toolchain the spec named, and 'cannot be shown to match' is
    what a refusal means. A skipped axis is a pass nobody proved."""
    meta = _meta()
    meta.pop(dropped)
    reason = aot_identity.verify_declared_identity(meta, _expected())
    assert reason.startswith(f"{axis}: ")
    assert "<absent>" in reason


def test_an_expectation_naming_no_value_is_never_a_pass() -> None:
    """An unverifiable expectation must not read as verified. The Plan refuses
    an artifact missing any of these, so reaching this means a hand-built
    expectation — and it still refuses."""
    reason = aot_identity.verify_declared_identity(_meta(), _expected(cell_key=""))
    assert reason == "cell_key: the spec named no expected value"


def test_the_closure_axis_is_absent_rather_than_invented() -> None:
    """pgw#903 asks for closure identity and the landed schema carries no
    comparable one: `EndpointRelease.code_closure_id` is the hub's release
    identifier, while the artifact records a {path: digest} map. Equating them
    on the strength of two similar names would refuse every healthy cell in the
    fleet, so the axis is OWED, not faked.

    This test exists to stop a later reader from 'finishing' it: closing the
    gap is one ruling by the th#1457 lane (stamp a comparable
    `code_closure_digest`, or delete the block per pgw#1034), never a local
    equality.
    """
    assert "code_closure_id" not in ExpectedIdentity.__struct_fields__
    assert "code_closure" not in aot_identity._COMPARED_AXES
    # The release id still rides the Plan, so the day it becomes comparable
    # nothing has to be re-plumbed to reach it.
    assert _plan().release.code_closure_id == "clo_01"


def test_identity_is_never_confirmed_by_comparing_bytes() -> None:
    """§4.25/§4.26 + pgw#1006: two mints of one key legitimately differ, so a
    byte comparison would refuse healthy cells and prove nothing about
    unhealthy ones. The module must not reach for artifact bytes at all."""
    src = Path(aot_identity.__file__).read_text()
    for banned in ("read_bytes", "sha256_file", "hashlib", "open(", "Path("):
        assert banned not in src, f"aot_identity reaches for bytes via {banned!r}"


# --- projection from the immutable Plan -------------------------------------


def _plan(backend: int = pb.STEADY_BACKEND_AOT_CELL) -> Any:
    arm = pb.Arm(
        graph_contract_digest="gc_01",
        shape=pb.ARM_SHAPE_BRANCHLESS,
        backend=backend,
    )
    if backend == pb.STEADY_BACKEND_AOT_CELL:
        arm.artifact.CopyFrom(pb.ArtifactIdentity(
            cell_ref="cozy/cells-micro#k1",
            content_digest="sha256:" + "c" * 64,
            cell_key="aot-inductor:k1",
            publisher_org="cozy",
            toolchain_digest=cell_key.facts_digest(_TOOLCHAIN),
            env_seal_digest=env_seal.seal_digest(_SEAL),
        ))
    spec = pb.ExecutionSpec(
        digest="sha256:" + "a" * 64,
        spec_version=1,
        release=pb.EndpointRelease(
            org="cozy", endpoint="micro", release_id="r1",
            image_digest="sha256:img", code_closure_id="clo_01"),
        function_name="generate",
        numerical_lane=pb.ExecutionLane(weights=pb.WEIGHT_LANE_BF16),
        arm=arm,
        topology=pb.Topology(accelerator="cuda", gpu_count=1, execution_groups=1),
        components=pb.ComponentManifest(slots=[pb.SlotBinding(
            slot="unet", ref="cozy/micro@v1", snapshot_digest="sha256:d")]),
    )
    return PlanFactory.from_execution_spec(AttemptRef("req", 1), spec)


def test_the_expectation_comes_from_the_plan_and_matches_a_real_artifact() -> None:
    expected = aot_identity.expected_from_plan(_plan())
    assert expected is not None
    assert aot_identity.verify_declared_identity(_meta(), expected) == ""
    assert expected.publisher_org == "cozy"


def test_an_arm_that_names_no_artifact_yields_no_expectation() -> None:
    """``None`` is a complete answer, not a gap: an eager_only arm has nothing
    to verify because it must arm nothing."""
    assert aot_identity.expected_from_plan(
        _plan(backend=pb.STEADY_BACKEND_EAGER_ONLY)) is None
    assert aot_identity.expected_from_plan(_plan()) is not None


# --- the gate runs before dlopen, on a real staged tarball ------------------


def _artifact(tmp_path: Path, meta: dict[str, Any]) -> Path:
    """A staging-shaped tarball. It carries no real ``.pt2``: the point is that
    identity refuses before anything would be loaded, so there is nothing to
    load."""
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "metadata.json").write_text(json.dumps(meta))
    (payload / aot_serve.PACKAGE_NAME).write_bytes(b"not-a-real-package")
    out = tmp_path / "cell.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        for item in sorted(payload.iterdir()):
            tar.add(item, arcname=item.name)
    return out


def test_stage_artifact_refuses_a_wrong_identity_before_any_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal must land while the artifact is still inert bytes. The
    runtime-key and sm gates are neutralized so the ONLY thing under test is
    identity — otherwise a green run could mean 'refused for the wrong
    reason'."""
    monkeypatch.setattr(aot_serve, "verify", lambda meta, **kw: "")
    monkeypatch.setattr(aot_serve, "host_isa_reason", lambda meta: "")
    loaded: list[Any] = []
    monkeypatch.setattr(
        aot_serve, "verify_package_compute_capability",
        lambda path: loaded.append(path) or "")

    meta = _meta()
    path = _artifact(tmp_path, meta)

    with pytest.raises(aot_serve.AdoptError) as exc:
        aot_serve.stage_artifact(
            path, "micro", cache_dir=tmp_path / "cache",
            expected=_expected(cell_key="aot-inductor:SOMETHING-ELSE"))
    assert exc.value.reason == "expected_identity_mismatch"
    assert "cell_key: expected" in str(exc.value)
    # Nothing downstream of the identity gate ran.
    assert loaded == []


def test_stage_artifact_with_a_matching_identity_reaches_the_load_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aot_serve, "verify", lambda meta, **kw: "")
    monkeypatch.setattr(aot_serve, "host_isa_reason", lambda meta: "")
    reached: list[Any] = []
    monkeypatch.setattr(
        aot_serve, "verify_package_compute_capability",
        lambda path: reached.append(path) or "")

    path = _artifact(tmp_path, _meta())
    staged = aot_serve.stage_artifact(
        path, "micro", cache_dir=tmp_path / "cache", expected=_expected())
    try:
        assert staged.metadata["cell_key"] == "aot-inductor:k1"
        assert len(reached) == 1
    finally:
        staged.close()


def test_no_expectation_leaves_the_legacy_path_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Until the cutover, RunJob dispatches carry no immutable spec. Absent an
    expectation the gate must not invent one — a default expectation would
    refuse every cell the fleet serves today."""
    monkeypatch.setattr(aot_serve, "verify", lambda meta, **kw: "")
    monkeypatch.setattr(aot_serve, "host_isa_reason", lambda meta: "")
    monkeypatch.setattr(
        aot_serve, "verify_package_compute_capability", lambda path: "")
    meta = _meta()
    meta.pop("toolchain")
    meta.pop(env_seal.SEAL_KEY)
    staged = aot_serve.stage_artifact(
        _artifact(tmp_path, meta), "micro", cache_dir=tmp_path / "cache")
    try:
        assert staged.metadata["cell_key"] == "aot-inductor:k1"
    finally:
        staged.close()


def test_an_internally_corrupt_artifact_keeps_its_own_distinct_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hash/signature problem inside the package is NOT an identity problem.
    Folding them into one reason would put a supply-chain event and a stale
    image rebuild in the same bucket."""
    monkeypatch.setattr(aot_serve, "host_isa_reason", lambda meta: "")
    monkeypatch.setattr(
        aot_serve, "verify",
        lambda meta, **kw: "entry 'unet': class_hash does not match its recorded facts")
    with pytest.raises(aot_serve.AdoptError) as exc:
        aot_serve.stage_artifact(
            _artifact(tmp_path, _meta()), "micro",
            cache_dir=tmp_path / "cache", expected=_expected())
    assert exc.value.reason == "key_mismatch"
    assert exc.value.reason != "expected_identity_mismatch"


# ---------------------------------------------------------------------------
# pgw#1152: every axis is verified SOMEWHERE, and the claim is checked
# ---------------------------------------------------------------------------


def test_every_identity_axis_is_either_compared_here_or_named_elsewhere() -> None:
    """The accounting is enforced at import (`aot_identity` refuses to load
    otherwise), so this row states the invariant rather than re-deriving it: an
    axis on the identity that nothing verifies is how a cell gets armed on an
    unchecked claim."""
    accounted = (set(aot_identity._COMPARED_AXES)
                 | set(aot_identity._VERIFIED_ELSEWHERE))
    assert accounted == set(ExpectedIdentity.__struct_fields__)


def test_each_axis_verified_elsewhere_names_a_gate_that_really_reads_it() -> None:
    """`_VERIFIED_ELSEWHERE` is a CLAIM, and pgw#1152's rule for this fence
    family is that a claim is CHECKED, not trusted — the arm-state lint accepts
    a `RECOGNIZER` row only after verifying it structurally, for exactly the
    reason that a comment naming the wrong gate reads identical to one naming
    the right gate.

    The checked property is that the named gate **is handed the axis** — it
    takes it as a parameter — because "this gate verifies the expectation"
    means the expectation reaches it. Merely *mentioning* the axis is too weak
    to discriminate: this entry first named
    `receipts.refuse_untrusted_publisher`, whose body does mention
    `publisher_org` while asking a different question (is this producer trusted
    AT ALL, rather than is it the one the spec NAMED). It takes no
    `publisher_org` parameter, so it fails this row; `fleet_cells.arm_ordered`,
    which compares the named org to the signed receipt's `publisher_org_id`
    fail-closed on silence (§4.26), takes one and passes.
    """
    for axis, why in aot_identity._VERIFIED_ELSEWHERE.items():
        dotted = why.split(" ", 1)[0]
        mod_name, _, func_name = dotted.rpartition(".")
        assert mod_name and func_name, f"{axis}: {why!r} must start with mod.func"
        module = importlib.import_module(f"gen_worker.{mod_name}")
        gate = getattr(module, func_name, None)
        assert gate is not None, f"{axis} names {dotted}, which does not exist"
        assert axis in inspect.signature(gate).parameters, (
            f"{axis} claims {dotted} verifies it, but that function is never "
            "handed the axis — a gate that does not receive the expectation "
            "cannot be the gate that checks it")
