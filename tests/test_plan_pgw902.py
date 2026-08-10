"""pgw#902 — the immutable Plan, and the three properties that make it one.

The defect this replaces is not hypothetical: on the RunJob path, ``HelloAck``
calls ``apply_model_resolutions``, which mutates ``EndpointSpec.models`` and
rehomes a live ``_ClassRecord`` AFTER connect. A dispatch therefore means
different bytes depending on when a resolution landed. Every test here asserts
the property that makes that unrepresentable rather than merely unusual.
"""

from __future__ import annotations

import ast
import pathlib

import msgspec
import pytest

from gen_worker import plan as plan_mod
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.plan import (
    Plan,
    PlanConflict,
    PlanFactory,
    PlanLedger,
    PlanRefusal,
)

SRC = pathlib.Path(plan_mod.__file__).resolve().parent


def _spec(**over: object) -> pb.ExecutionSpec:
    """A minimal spec that MUST convert. Every refusal test mutates exactly
    one field of this, so a test that goes red names the axis it broke."""
    spec = pb.ExecutionSpec(
        digest="sha256:" + "a" * 64,
        spec_version=1,
        release=pb.EndpointRelease(
            org="cozy",
            endpoint="micro-diffusion",
            release_id="rel_01",
            image_digest="sha256:" + "b" * 64,
            code_closure_id="clo_01",
        ),
        function_name="generate",
        numerical_lane=pb.ExecutionLane(weights=pb.WEIGHT_LANE_W8A8, mandatory=True),
        arm=pb.Arm(
            graph_contract_digest="gc_01",
            shape=pb.ARM_SHAPE_BRANCHLESS,
            adapter_rank_bucket=0,
            backend=pb.STEADY_BACKEND_AOT_CELL,
            artifact=pb.ArtifactIdentity(
                cell_ref="cozy/cells-sdxl#k1",
                content_digest="sha256:" + "c" * 64,
                cell_key="k1",
                publisher_org="cozy",
                toolchain_digest="tc_01",
                env_seal_digest="es_01",
            ),
        ),
        topology=pb.Topology(
            accelerator="cuda", gpu_count=1, execution_groups=1, device_ordinals=[0]),
        components=pb.ComponentManifest(slots=[
            pb.SlotBinding(
                slot="unet",
                ref="cozy/sdxl@v3",
                snapshot_digest="sha256:" + "d" * 64,
                objective="text-to-image",
            ),
        ]),
        config=pb.ConfigSnapshot(values=b"\x80", digest="cfg_01"),
        contracts=pb.ContractIdentities(
            quality_contract_digest="q_01", runtime_formula_digest="rf_01"),
        attribution=pb.Attribution(org="acme", invoker_id="user_7"),
        output_mode=pb.OUTPUT_MODE_URL,
    )
    for name, value in over.items():
        if value is None:
            spec.ClearField(name)
        elif isinstance(value, (int, str, bytes)):
            setattr(spec, name, value)
        else:
            getattr(spec, name).CopyFrom(value)  # type: ignore[arg-type]
    return spec


def _attempt() -> plan_mod.AttemptRef:
    return plan_mod.AttemptRef(request_id="req_1", attempt=3)


def _plan(**over: object) -> Plan:
    return PlanFactory.from_execution_spec(_attempt(), _spec(**over))


# --- the happy conversion carries EVERYTHING, or the digest lies -------------


def test_every_spec_field_reaches_the_plan() -> None:
    p = _plan()
    assert p.digest == "sha256:" + "a" * 64
    assert p.spec_version == 1
    assert p.release.image_digest == "sha256:" + "b" * 64
    assert p.release.code_closure_id == "clo_01"
    assert p.function_name == "generate"
    # The lane arrives RESOLVED. There is no family token to expand.
    assert p.lane.weights == "w8a8"
    assert p.lane.mandatory is True
    assert p.arm.backend == "aot_cell"
    assert p.arm.artifact is not None
    assert p.arm.artifact.cell_key == "k1"
    assert p.arm.artifact.publisher_org == "cozy"
    assert p.topology.device_ordinals == (0,)
    assert p.slot_refs == {"unet": "cozy/sdxl@v3"}
    assert p.slots[0].objective == "text-to-image"
    assert p.config.values == b"\x80"
    assert p.attribution.org == "acme"
    assert p.output_mode == "url"
    assert p.exploration is None
    assert p.snapshot_digests() == ("sha256:" + "d" * 64,)


def test_two_conversions_of_one_spec_are_equal() -> None:
    """Determinism is the precondition for the digest meaning anything."""
    assert _plan() == _plan()


# --- property 1: PURE. The factory cannot reach anything that resolves ------


def test_plan_module_imports_nothing_that_could_resolve() -> None:
    """A PlanFactory that can import a store can fetch; one that can import a
    device can probe. Either makes it a second resolver again. The rule is
    enforced on the import graph so a future edit trips it, not a reviewer.

    ``pb`` is allowed and is the point: the generated schema is the ONLY
    input. ``msgspec`` is allowed: it is the value vocabulary.
    """
    forbidden = {
        "store", "models", "executor", "lifecycle", "transport", "registry",
        "compile_cache", "aot_cells", "aot_serve", "fleet_cells", "preload",
        "activity", "net", "convert", "cli", "procsplit", "capability",
        "runtime_config", "intent_registry", "cell_key", "local_cells",
    }
    tree = ast.parse((SRC / "plan.py").read_text())
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            seen.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("gen_worker"):
                    seen.add(alias.name.split(".")[-1])
    assert seen & forbidden == set(), f"plan.py reaches resolution machinery: {seen & forbidden}"
    assert "pb" in seen, "plan.py must read the generated schema"


def test_no_torch_or_network_import_survives_module_load() -> None:
    import sys

    assert "gen_worker.plan" in sys.modules
    src = (SRC / "plan.py").read_text()
    for banned in ("import torch", "import requests", "import httpx", "subprocess"):
        assert banned not in src


# --- property 2: IMMUTABLE. Nothing arriving later has anywhere to write ----


def test_the_plan_and_every_nested_value_refuse_mutation() -> None:
    p = _plan()
    for target in (p, p.release, p.lane, p.arm, p.topology, p.slots[0], p.config):
        with pytest.raises((AttributeError, TypeError)):
            setattr(target, "function_name" if target is p else "org", "mutated")
    assert isinstance(p.slots, tuple)
    assert isinstance(p.adapters, tuple)
    assert isinstance(p.input_assets, tuple)
    assert isinstance(p.topology.device_ordinals, tuple)


def test_a_grant_rotation_leaves_the_plan_byte_identical() -> None:
    """pgw#891's acceptance, at the value layer: the grant carries authority,
    the spec carries identity, and the Plan reads only the spec. Rotating a
    capability token mid-attempt must not move a single byte of the Plan."""
    spec = _spec()
    first = pb.RunAttempt(
        attempt=pb.AttemptId(request_id="req_1", attempt=3),
        spec=spec,
        grant=pb.DeliveryGrant(
            capability_token="tok-A", expires_unix_ms=1, epoch=1),
    )
    rotated = pb.RunAttempt(
        attempt=pb.AttemptId(request_id="req_1", attempt=3),
        spec=spec,
        grant=pb.DeliveryGrant(
            capability_token="tok-B-completely-different",
            expires_unix_ms=99999, epoch=2,
            input_assets=[pb.PresignedAsset(asset_id="a1", url="https://x/y?sig=Z")],
        ),
    )
    a = PlanFactory.from_run_attempt(first)
    b = PlanFactory.from_run_attempt(rotated)
    assert a == b
    assert msgspec.json.encode(a) == msgspec.json.encode(b)


def test_no_grant_secret_can_appear_anywhere_in_the_plan() -> None:
    attempt = pb.RunAttempt(
        attempt=pb.AttemptId(request_id="req_1", attempt=3),
        spec=_spec(),
        grant=pb.DeliveryGrant(
            capability_token="SECRET-TOKEN",
            snapshots={"sha256:" + "d" * 64: pb.PresignedSnapshot(files=[
                pb.PresignedFile(path="m.safetensors", url="https://s3/PRESIGNED")])},
        ),
    )
    encoded = msgspec.json.encode(PlanFactory.from_run_attempt(attempt))
    assert b"SECRET-TOKEN" not in encoded
    assert b"PRESIGNED" not in encoded


# --- property 3: REFUSE, never default -------------------------------------


def test_an_unknown_spec_version_refuses_before_anything_else() -> None:
    with pytest.raises(PlanRefusal) as exc:
        _plan(spec_version=2)
    assert exc.value.field == "spec.spec_version"
    assert exc.value.have == "2"


def test_an_unresolved_lane_refuses_instead_of_being_expanded() -> None:
    """The enum has no coarse family member, so the only way to express one is
    UNSPECIFIED — and the worker's answer to that is a refusal, never a ladder
    expansion. This is `executor._execution_lane_effective_spec` deleted by
    construction rather than by a lint."""
    with pytest.raises(PlanRefusal) as exc:
        _plan(numerical_lane=pb.ExecutionLane(weights=pb.WEIGHT_LANE_UNSPECIFIED))
    assert exc.value.field == "spec.numerical_lane.weights"
    assert "WEIGHT_LANE_UNSPECIFIED" in exc.value.have


def test_an_aot_arm_with_no_named_artifact_refuses() -> None:
    """pgw#904's contract at the value layer: one exact identity or explicitly
    none. 'AOT, figure out which' is the fetch-and-filter resolver returning
    through a side door."""
    arm = pb.Arm(
        graph_contract_digest="gc_01",
        shape=pb.ARM_SHAPE_BRANCHLESS,
        backend=pb.STEADY_BACKEND_AOT_CELL,
    )
    with pytest.raises(PlanRefusal) as exc:
        _plan(arm=arm)
    assert exc.value.field == "spec.arm.artifact"


def test_a_non_aot_arm_carrying_an_artifact_refuses() -> None:
    """Since pgw#1010 a dynamo cell is neither sealed nor published, so an
    artifact on a DYNAMO arm describes something that cannot exist."""
    arm = pb.Arm(
        graph_contract_digest="gc_01",
        shape=pb.ARM_SHAPE_BRANCHLESS,
        backend=pb.STEADY_BACKEND_DYNAMO,
        artifact=pb.ArtifactIdentity(
            cell_ref="r", content_digest="d", cell_key="k",
            publisher_org="o", toolchain_digest="t", env_seal_digest="e"),
    )
    with pytest.raises(PlanRefusal) as exc:
        _plan(arm=arm)
    assert exc.value.field == "spec.arm.artifact"
    assert "dynamo" in exc.value.have


@pytest.mark.parametrize(
    "backend,token",
    [
        (pb.STEADY_BACKEND_EAGER_ONLY, "eager_only"),
        (pb.STEADY_BACKEND_DYNAMO, "dynamo"),
    ],
)
def test_a_non_cell_arm_is_a_valid_plan_with_no_artifact_and_no_graph_contract(
    backend: int, token: str,
) -> None:
    """The shape most of the fleet's declared lanes need. `svdq-fp4-w4a4+eager`
    is execution-eager-only in tensorhub's lane table — a lane that cannot be
    compiled at all — so an EAGER_ONLY arm can never name a graph contract; a
    DYNAMO arm compiles at serve time, so the hub cannot name one ahead of
    dispatch either."""
    p = _plan(arm=pb.Arm(shape=pb.ARM_SHAPE_BRANCHLESS, backend=backend))
    assert p.arm.backend == token
    assert p.arm.artifact is None
    assert p.arm.graph_contract_digest == ""


@pytest.mark.parametrize(
    "backend", [pb.STEADY_BACKEND_EAGER_ONLY, pb.STEADY_BACKEND_DYNAMO])
def test_a_graph_contract_on_a_non_cell_arm_refuses(backend: int) -> None:
    """Symmetric with the artifact rule: a graph contract on an arm that names
    no cell is a value the hub cannot produce and this worker never verifies
    (`aot_identity.expected_from_plan` returns None there), riding inside the
    execution digest."""
    with pytest.raises(PlanRefusal) as exc:
        _plan(arm=pb.Arm(
            graph_contract_digest="gc_01",
            shape=pb.ARM_SHAPE_BRANCHLESS,
            backend=backend))
    assert exc.value.field == "spec.arm.graph_contract_digest"


def test_an_aot_cell_arm_without_a_graph_contract_refuses() -> None:
    """pgw#903's pre-dlopen fence compares exactly this value, so the arm that
    HAS an artifact must still name it."""
    with pytest.raises(PlanRefusal) as exc:
        _plan(arm=pb.Arm(
            shape=pb.ARM_SHAPE_BRANCHLESS,
            backend=pb.STEADY_BACKEND_AOT_CELL,
            artifact=pb.ArtifactIdentity(
                cell_ref="r", content_digest="d", cell_key="k",
                publisher_org="o", toolchain_digest="t", env_seal_digest="e")))
    assert exc.value.field == "spec.arm.graph_contract_digest"


def test_an_artifact_with_no_publisher_refuses() -> None:
    """§4.26: a compiled cell is code this pod dlopen()s, so WHO produced it is
    the trust answer. 'Unknown publisher' is not a tier."""
    arm = pb.Arm(
        graph_contract_digest="gc_01",
        shape=pb.ARM_SHAPE_BRANCHLESS,
        backend=pb.STEADY_BACKEND_AOT_CELL,
        artifact=pb.ArtifactIdentity(
            cell_ref="r", content_digest="d", cell_key="k",
            toolchain_digest="t", env_seal_digest="e"),
    )
    with pytest.raises(PlanRefusal) as exc:
        _plan(arm=arm)
    assert exc.value.field == "spec.arm.artifact.publisher_org"


@pytest.mark.parametrize(
    "shape,bucket,field",
    [
        (pb.ARM_SHAPE_ADAPTER, 0, "spec.arm.adapter_rank_bucket"),
        (pb.ARM_SHAPE_BRANCHLESS, 8, "spec.arm.adapter_rank_bucket"),
        (pb.ARM_SHAPE_UNSPECIFIED, 0, "spec.arm.shape"),
    ],
)
def test_the_adapter_partition_is_hard(shape: int, bucket: int, field: str) -> None:
    """§4.4: adapter traffic is a partition, not a modifier. A rank bucket on a
    branchless arm and a missing one on an adapter arm are two different
    statements about which graph runs, and neither is repairable here."""
    arm = pb.Arm(
        graph_contract_digest="gc_01", shape=shape, adapter_rank_bucket=bucket,
        backend=pb.STEADY_BACKEND_EAGER_ONLY)
    with pytest.raises(PlanRefusal) as exc:
        _plan(arm=arm)
    assert exc.value.field == field


def test_a_slot_without_a_snapshot_digest_refuses() -> None:
    """Without the digest the grant's digest-keyed snapshot map cannot be
    joined, so the worker would be back to trusting a name."""
    with pytest.raises(PlanRefusal) as exc:
        _plan(components=pb.ComponentManifest(slots=[
            pb.SlotBinding(slot="unet", ref="cozy/sdxl@v3")]))
    assert exc.value.field == "spec.components.slots[unet].snapshot_digest"


def test_a_duplicate_slot_refuses_rather_than_last_one_winning() -> None:
    with pytest.raises(PlanRefusal) as exc:
        _plan(components=pb.ComponentManifest(slots=[
            pb.SlotBinding(slot="unet", ref="a", snapshot_digest="d1"),
            pb.SlotBinding(slot="unet", ref="b", snapshot_digest="d2"),
        ]))
    assert exc.value.field == "spec.components.slots[unet]"


def test_a_release_that_cannot_name_its_own_bytes_refuses() -> None:
    """pgw#903 verifies a compiled cell against the release's identity axes. A
    release with no image digest cannot support that check, so it is refused
    here rather than making the check optional there."""
    for field in ("image_digest", "code_closure_id"):
        rel = pb.EndpointRelease(
            org="cozy", endpoint="e", release_id="r",
            image_digest="sha256:x", code_closure_id="c")
        rel.ClearField(field)
        with pytest.raises(PlanRefusal) as exc:
            _plan(release=rel)
        assert exc.value.field == f"spec.release.{field}"


def test_an_out_of_range_adapter_weight_refuses() -> None:
    spec = _spec()
    spec.adapters.append(pb.AdapterBinding(
        slot="unet", ref="cozy/lora@v1", snapshot_digest="d", weight=9.0))
    with pytest.raises(PlanRefusal) as exc:
        PlanFactory.from_execution_spec(_attempt(), spec)
    assert exc.value.field == "spec.adapters[unet].weight"


def test_a_topology_with_no_execution_group_refuses() -> None:
    with pytest.raises(PlanRefusal) as exc:
        _plan(topology=pb.Topology(accelerator="cuda", gpu_count=1))
    assert exc.value.field == "spec.topology.execution_groups"


def test_an_input_asset_without_a_digest_refuses() -> None:
    """The identity half of the input-asset split. A changed input that leaves
    the spec digest unmoved is two executions sharing one identity."""
    spec = _spec()
    spec.input_assets.append(pb.InputAssetRef(asset_id="a1", source_ref="s"))
    with pytest.raises(PlanRefusal) as exc:
        PlanFactory.from_execution_spec(_attempt(), spec)
    assert exc.value.field == "spec.input_assets[a1].digest"


def test_a_refusal_names_expected_and_have() -> None:
    """Every refusal is read as an event by whoever owns the hub side. 'invalid
    spec' is not actionable; the axis and both values are."""
    with pytest.raises(PlanRefusal) as exc:
        _plan(spec_version=7)
    assert "expected" in str(exc.value) and "have 7" in str(exc.value)


def test_a_run_attempt_missing_its_spec_refuses() -> None:
    with pytest.raises(PlanRefusal) as exc:
        PlanFactory.from_run_attempt(
            pb.RunAttempt(attempt=pb.AttemptId(request_id="r", attempt=1)))
    assert exc.value.field == "spec"


# --- the ledger: replay is cheap, a changed digest is never adopted ---------


def test_same_attempt_same_digest_replays_and_never_re_executes() -> None:
    ledger = PlanLedger()
    p = _plan()
    assert ledger.admit(p) is True
    assert ledger.admit(_plan()) is False
    assert len(ledger) == 1


def test_same_attempt_different_digest_conflicts_and_keeps_the_first() -> None:
    ledger = PlanLedger()
    first = _plan()
    ledger.admit(first)
    with pytest.raises(PlanConflict) as exc:
        ledger.admit(_plan(digest="sha256:" + "e" * 64))
    assert exc.value.expected == first.digest
    assert exc.value.have == "sha256:" + "e" * 64
    # The earlier state stands: a conflict never adopts the newcomer.
    seen = ledger.get(first.attempt)
    assert seen is not None and seen.digest == first.digest


def test_a_different_attempt_of_one_request_is_a_new_admission() -> None:
    """The attempt integer is the fencing token: attempt 4 of the same request
    is different work, not a replay of attempt 3."""
    ledger = PlanLedger()
    ledger.admit(_plan())
    later = PlanFactory.from_execution_spec(
        plan_mod.AttemptRef(request_id="req_1", attempt=4),
        _spec(digest="sha256:" + "f" * 64))
    assert ledger.admit(later) is True
    assert len(ledger) == 2
