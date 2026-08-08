"""The immutable execution Plan (pgw#902), derived from one Hub-issued
``ExecutionSpec`` and from nothing else.

A Plan is the worker's whole answer to "what am I running". It is built once,
by a pure function, from the generated ``ExecutionSpec`` the hub digested; no
later message may change it. That is the property the legacy path never had:
``HelloAck`` calls ``apply_model_resolutions``, which mutates ``EndpointSpec``
bindings and rehomes a live ``_ClassRecord`` after connect, so the same
dispatch can mean different bytes depending on when a resolution arrived.

Three rules make this module's shape non-negotiable, and each has a test:

1. **Pure.** ``PlanFactory`` imports no store, transport, lifecycle, cache or
   device module. It cannot fetch, resolve, substitute or probe — it can only
   read the spec and refuse. ``test_plan_factory_is_pure`` walks the import
   graph, so the rule survives a well-meaning future import.
2. **Immutable.** Every struct here is ``frozen=True`` and every sequence is a
   tuple. A grant rotation, a HelloAck, a config generation bump or a
   residency reconcile has nowhere to write.
3. **Refuse, never default.** A missing or unresolved required fact raises
   :class:`PlanRefusal` BEFORE model acquisition. There is no coarse lane to
   expand, no root-slot fallback, no "first acceptable" artifact: the hub
   already decided, and a worker that can reconstruct a decision is a second
   resolver (§1.31, §1.10).

The Plan deliberately does NOT carry: presigned URLs, capability tokens, grant
epochs or expiry. Those live in ``DeliveryGrant``, are keyed by content digest,
and are rotatable precisely because they are not execution identity. Reading
one through the Plan would put a credential back inside the thing the digest
covers — see ``ExecutionSpec``'s own header in the proto.
"""

from __future__ import annotations

from typing import Any, Mapping

import msgspec

from .pb import worker_scheduler_pb2 as pb

#: Spec versions this worker build understands. ``ExecutionSpec.spec_version``
#: is the generated-schema version; an unknown one refuses the attempt rather
#: than degrading silently, because an unknown version can carry a REQUIRED
#: field whose absence changes what gets produced. Widening this tuple is a
#: deliberate code change in this repo, reviewed against the schema diff.
SUPPORTED_SPEC_VERSIONS: tuple[int, ...] = (1,)

#: ``WeightLane`` enum value -> the worker's canonical lane token. The mapping
#: exists because the enum IS the resolved lane: there is no coarse family to
#: expand, so this is a rename and never a decision. ``WEIGHT_LANE_UNSPECIFIED``
#: is absent on purpose — it refuses.
_WEIGHT_LANE_TOKENS: dict[int, str] = {
    pb.WEIGHT_LANE_BF16: "bf16",
    pb.WEIGHT_LANE_W8A16: "w8a16",
    pb.WEIGHT_LANE_W8A8: "w8a8",
    pb.WEIGHT_LANE_W4A4: "w4a4",
}

_ARM_SHAPE_TOKENS: dict[int, str] = {
    pb.ARM_SHAPE_BRANCHLESS: "branchless",
    pb.ARM_SHAPE_ADAPTER: "adapter",
}

_BACKEND_TOKENS: dict[int, str] = {
    pb.STEADY_BACKEND_AOT_CELL: "aot_cell",
    pb.STEADY_BACKEND_DYNAMO: "dynamo",
    pb.STEADY_BACKEND_EAGER_ONLY: "eager_only",
}

_OUTPUT_MODE_TOKENS: dict[int, str] = {
    pb.OUTPUT_MODE_UNSPECIFIED: "",
    pb.OUTPUT_MODE_URL: "url",
    pb.OUTPUT_MODE_INLINE: "inline",
}


class PlanRefusal(Exception):
    """A spec that cannot become a Plan, named exactly.

    Carries ``field``/``expected``/``have`` because the fleet reads these as
    events, not as prose: "components.unet.snapshot_digest expected non-empty,
    have ''" is actionable by the hub team; "invalid spec" is not. Every
    refusal in this module is raised before any model acquisition.
    """

    def __init__(self, field: str, expected: str, have: str = "") -> None:
        self.field = field
        self.expected = expected
        self.have = have
        detail = f"{field}: expected {expected}"
        if have:
            detail += f", have {have}"
        super().__init__(detail)


class AttemptRef(msgspec.Struct, frozen=True):
    """The hub-owned fencing token. ``attempt`` is bumped by the orchestrator
    on every (re)queue and by nothing worker-side."""

    request_id: str
    attempt: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.request_id, self.attempt)


class ReleaseRef(msgspec.Struct, frozen=True):
    """The code that runs. ``image_digest``/``code_closure_id`` are what make
    "the same release" mean the same bytes rather than the same name — and
    they are what pgw#903 compares a compiled artifact's baked facts against."""

    org: str
    endpoint: str
    release_id: str
    image_digest: str
    code_closure_id: str


class AttributionRef(msgspec.Struct, frozen=True):
    """Who the outputs belong to and who asked. ``org`` is the CALLER's org,
    a different fact from :attr:`ReleaseRef.org`, the publishing org."""

    org: str
    invoker_id: str


class LaneRef(msgspec.Struct, frozen=True):
    """The RESOLVED numerical lane.

    ``mandatory`` is the FACT that this lane has no eager form. It is not a
    fallback policy: what a worker does when a mandatory lane's cell is
    missing is pgw#888's open conflict with §1.12/§4.5 and Paul's to rule, so
    nothing here reads ``mandatory`` to decide anything.
    """

    weights: str
    mandatory: bool


class ArtifactRef(msgspec.Struct, frozen=True):
    """Exactly one compiled cell (th#1429, §4.26).

    ``publisher_org`` is load-bearing rather than decorative: a compiled cell
    is CODE the adopting pod ``dlopen``s, so WHO produced it is the trust
    answer. §4.25/§4.26 forbid confirming an artifact by byte comparison — two
    mints of one key legitimately differ (pgw#1006 measured it) — so pgw#903's
    verification compares these declared fields and never bytes.
    """

    cell_ref: str
    content_digest: str
    cell_key: str
    publisher_org: str
    toolchain_digest: str
    env_seal_digest: str


class ArmRef(msgspec.Struct, frozen=True):
    """The structured execution arm (§4.4/§4.5).

    Adapter traffic is a HARD partition, so ``adapter_rank_bucket`` rides the
    arm instead of being recovered from the adapter list. ``artifact`` is set
    iff ``backend == "aot_cell"``: since pgw#1010 dynamo cells are neither
    sealed nor published, so DYNAMO has no artifact to name.
    """

    graph_contract_digest: str
    shape: str
    adapter_rank_bucket: int
    backend: str
    artifact: ArtifactRef | None


class TopologyRef(msgspec.Struct, frozen=True):
    """Spec identity, not placement advice: the same work on a different
    device count is different work."""

    accelerator: str
    gpu_count: int
    execution_groups: int
    device_ordinals: tuple[int, ...]


class ComponentRef(msgspec.Struct, frozen=True):
    """One already-resolved per-component substitution on a slot's base
    composition (th#980)."""

    component: str
    ref: str
    snapshot_digest: str


class SlotRef(msgspec.Struct, frozen=True):
    """One slot's final binding. ``ref`` is canonical and resolved: there is
    no slot->ref indirection left for a worker to re-resolve, which is what
    deletes the second resolver (pgw#904)."""

    slot: str
    ref: str
    snapshot_digest: str
    components: tuple[ComponentRef, ...]
    objective: str
    distilled: bool
    distilled_status: str
    inference_defaults: str


class AdapterRef(msgspec.Struct, frozen=True):
    """One resolved LoRA overlay. Adapters are spec identity because they
    change the product."""

    slot: str
    ref: str
    snapshot_digest: str
    weight: float
    inference_defaults: str


class ConfigRef(msgspec.Struct, frozen=True):
    """THIS attempt's own immutable evaluated configuration (§4.16).

    ``values`` stays as the canonical msgpack bytes the hub digested. Decoding
    is the consumer's business; keeping the bytes means the Plan holds exactly
    what ``digest`` covers, with no re-encode in between.
    """

    values: bytes
    digest: str


class ContractsRef(msgspec.Struct, frozen=True):
    """The contracts this attempt is comparable under. The graph contract is
    NOT here — it identifies the arm, so it lives on :class:`ArmRef`."""

    quality_contract_digest: str
    runtime_formula_digest: str


class ExplorationRef(msgspec.Struct, frozen=True):
    """A durable operator exploration intent (§4.7). Absent means ordinary
    traffic; it is never inferred."""

    exploration_id: str
    operator_intent_id: str


class InputAssetRef(msgspec.Struct, frozen=True):
    """A stored input's IDENTITY. Its presigned transport lives in
    ``DeliveryGrant.input_assets``, joined by ``asset_id`` — the split exists
    because a changed input that leaves the digest unmoved breaks what the
    digest MEANS."""

    asset_id: str
    source_ref: str
    digest: str
    size_bytes: int
    kind: str
    mime_type: str


class Plan(msgspec.Struct, frozen=True):
    """One attempt's immutable execution plan.

    ``digest`` is the execution identity: the same attempt with the same
    digest is the same work and replays idempotently; the same attempt with a
    DIFFERENT digest is a typed refusal, never an adoption of earlier state.
    """

    attempt: AttemptRef
    digest: str
    spec_version: int
    release: ReleaseRef
    function_name: str
    lane: LaneRef
    arm: ArmRef
    topology: TopologyRef
    slots: tuple[SlotRef, ...]
    adapters: tuple[AdapterRef, ...]
    config: ConfigRef
    contracts: ContractsRef
    attribution: AttributionRef
    output_mode: str
    input_assets: tuple[InputAssetRef, ...]
    exploration: ExplorationRef | None

    @property
    def key(self) -> tuple[str, int]:
        return self.attempt.key

    @property
    def slot_refs(self) -> Mapping[str, str]:
        """``{slot: resolved ref}`` — the exact set, in declaration order."""
        return {s.slot: s.ref for s in self.slots}

    def snapshot_digests(self) -> tuple[str, ...]:
        """Every content digest this plan pins, deduplicated and ordered.

        This is the join key into ``DeliveryGrant.snapshots``: the grant is
        keyed by CONTENT DIGEST rather than by ref precisely so the transport
        cannot drift from the identity it delivers.
        """
        seen: dict[str, None] = {}
        for slot in self.slots:
            seen.setdefault(slot.snapshot_digest, None)
            for comp in slot.components:
                seen.setdefault(comp.snapshot_digest, None)
        for adapter in self.adapters:
            seen.setdefault(adapter.snapshot_digest, None)
        seen.pop("", None)
        return tuple(seen)


def _require(value: str, field: str, expected: str = "a non-empty value") -> str:
    text = str(value or "").strip()
    if not text:
        raise PlanRefusal(field, expected, "''")
    return text


class PlanFactory:
    """The one ``ExecutionSpec`` -> :class:`Plan` transform.

    A class rather than a function only so the boundary has a name a guard can
    assert on; it holds no state and takes no collaborators. Anything that
    would need a store, a device or the network to answer does not belong in a
    Plan — it belongs downstream, reading the Plan.
    """

    __slots__ = ()

    @staticmethod
    def from_run_attempt(attempt: Any) -> Plan:
        """Build the Plan for one ``RunAttempt``.

        The grant is deliberately not read here. It carries authority, not
        identity, and rotating it must leave this value byte-identical.
        """
        if not attempt.HasField("attempt"):
            raise PlanRefusal("attempt", "an AttemptId")
        if not attempt.HasField("spec"):
            raise PlanRefusal("spec", "an ExecutionSpec")
        ref = AttemptRef(
            request_id=_require(attempt.attempt.request_id, "attempt.request_id"),
            attempt=int(attempt.attempt.attempt),
        )
        return PlanFactory.from_execution_spec(ref, attempt.spec)

    @staticmethod
    def from_execution_spec(attempt: AttemptRef, spec: Any) -> Plan:
        """Pure transform. Raises :class:`PlanRefusal` and nothing else."""
        version = int(spec.spec_version)
        if version not in SUPPORTED_SPEC_VERSIONS:
            raise PlanRefusal(
                "spec.spec_version",
                "one of " + ", ".join(str(v) for v in SUPPORTED_SPEC_VERSIONS),
                str(version),
            )
        digest = _require(spec.digest, "spec.digest", "the canonical spec digest")
        return Plan(
            attempt=attempt,
            digest=digest,
            spec_version=version,
            release=_release(spec),
            function_name=_require(spec.function_name, "spec.function_name"),
            lane=_lane(spec),
            arm=_arm(spec),
            topology=_topology(spec),
            slots=_slots(spec),
            adapters=_adapters(spec),
            config=ConfigRef(values=bytes(spec.config.values), digest=str(spec.config.digest)),
            contracts=ContractsRef(
                quality_contract_digest=str(spec.contracts.quality_contract_digest),
                runtime_formula_digest=str(spec.contracts.runtime_formula_digest),
            ),
            attribution=AttributionRef(
                org=str(spec.attribution.org),
                invoker_id=str(spec.attribution.invoker_id),
            ),
            output_mode=_output_mode(spec),
            input_assets=_input_assets(spec),
            exploration=_exploration(spec),
        )


class PlanConflict(Exception):
    """The same attempt arrived twice under two different execution
    identities. This is never repairable and never adopts the earlier state:
    the hub changed what it asked for without changing the fencing token, and
    silently running either one would make the digest meaningless."""

    def __init__(self, attempt: AttemptRef, expected: str, have: str) -> None:
        self.attempt = attempt
        self.expected = expected
        self.have = have
        super().__init__(
            f"attempt {attempt.request_id}/{attempt.attempt} was admitted with "
            f"spec digest {expected}, and this dispatch carries {have}")


class PlanLedger:
    """Per-attempt Plan identity, and the only place replay is decided.

    Three answers, and there is no fourth: an attempt is NEW (admit it), a
    REPLAY of the identical digest (re-acknowledge, do not re-execute), or a
    :class:`PlanConflict`. Retransmission is normal — the orchestrator resends
    until it sees ``JobAccepted`` — so replay must be cheap and must not
    depend on how far the first copy got.

    Holds Plans only. No store, transport, device or lifecycle state lives
    here, which is what lets the executor consult it before materializing
    anything.
    """

    __slots__ = ("_plans",)

    def __init__(self) -> None:
        self._plans: dict[tuple[str, int], Plan] = {}

    def admit(self, plan: Plan) -> bool:
        """``True`` if this is the attempt's first admission, ``False`` on an
        identical replay. Raises :class:`PlanConflict` on a changed digest."""
        seen = self._plans.get(plan.key)
        if seen is None:
            self._plans[plan.key] = plan
            return True
        if seen.digest != plan.digest:
            raise PlanConflict(plan.attempt, seen.digest, plan.digest)
        return False

    def get(self, attempt: AttemptRef) -> Plan | None:
        return self._plans.get(attempt.key)

    def forget(self, attempt: AttemptRef) -> None:
        self._plans.pop(attempt.key, None)

    def __len__(self) -> int:
        return len(self._plans)


def _release(spec: Any) -> ReleaseRef:
    if not spec.HasField("release"):
        raise PlanRefusal("spec.release", "an EndpointRelease")
    rel = spec.release
    return ReleaseRef(
        org=_require(rel.org, "spec.release.org"),
        endpoint=_require(rel.endpoint, "spec.release.endpoint"),
        release_id=_require(rel.release_id, "spec.release.release_id"),
        # The two identity axes pgw#903 verifies a compiled cell against. A
        # release that cannot name its own bytes cannot support that check, so
        # it is refused here rather than making the check optional there.
        image_digest=_require(rel.image_digest, "spec.release.image_digest"),
        code_closure_id=_require(rel.code_closure_id, "spec.release.code_closure_id"),
    )


def _lane(spec: Any) -> LaneRef:
    if not spec.HasField("numerical_lane"):
        raise PlanRefusal("spec.numerical_lane", "a resolved ExecutionLane")
    weights = int(spec.numerical_lane.weights)
    token = _WEIGHT_LANE_TOKENS.get(weights)
    if token is None:
        raise PlanRefusal(
            "spec.numerical_lane.weights",
            "a resolved WeightLane (" + "|".join(_WEIGHT_LANE_TOKENS.values()) + ")",
            pb.WeightLane.Name(weights) if weights in pb.WeightLane.values() else str(weights),
        )
    return LaneRef(weights=token, mandatory=bool(spec.numerical_lane.mandatory))


def _arm(spec: Any) -> ArmRef:
    if not spec.HasField("arm"):
        raise PlanRefusal("spec.arm", "a structured Arm")
    arm = spec.arm
    shape = _ARM_SHAPE_TOKENS.get(int(arm.shape))
    if shape is None:
        raise PlanRefusal(
            "spec.arm.shape",
            "|".join(_ARM_SHAPE_TOKENS.values()),
            pb.ArmShape.Name(int(arm.shape)),
        )
    backend = _BACKEND_TOKENS.get(int(arm.backend))
    if backend is None:
        raise PlanRefusal(
            "spec.arm.backend",
            "|".join(_BACKEND_TOKENS.values()),
            pb.SteadyBackend.Name(int(arm.backend)),
        )
    bucket = int(arm.adapter_rank_bucket)
    # §4.4 makes adapter traffic a hard partition: a rank bucket on a
    # branchless arm, or none on an adapter arm, is two different statements
    # about what graph runs. Neither is repairable worker-side.
    if shape == "adapter" and bucket <= 0:
        raise PlanRefusal("spec.arm.adapter_rank_bucket", "a positive rank bucket", str(bucket))
    if shape == "branchless" and bucket != 0:
        raise PlanRefusal(
            "spec.arm.adapter_rank_bucket", "0 on a branchless arm", str(bucket))
    artifact: ArtifactRef | None = None
    if arm.HasField("artifact"):
        if backend != "aot_cell":
            raise PlanRefusal(
                "spec.arm.artifact",
                "absent unless backend is aot_cell",
                f"present with backend {backend}",
            )
        artifact = _artifact(arm.artifact)
    elif backend == "aot_cell":
        raise PlanRefusal(
            "spec.arm.artifact", "exactly one ArtifactIdentity on an aot_cell arm", "absent")
    return ArmRef(
        graph_contract_digest=_require(
            arm.graph_contract_digest, "spec.arm.graph_contract_digest"),
        shape=shape,
        adapter_rank_bucket=bucket,
        backend=backend,
        artifact=artifact,
    )


def _artifact(ident: Any) -> ArtifactRef:
    return ArtifactRef(
        cell_ref=_require(ident.cell_ref, "spec.arm.artifact.cell_ref"),
        content_digest=_require(ident.content_digest, "spec.arm.artifact.content_digest"),
        cell_key=_require(ident.cell_key, "spec.arm.artifact.cell_key"),
        # Not optional: an artifact whose producer is unnamed cannot be
        # trusted by any rule, and "unknown publisher" is not a tier.
        publisher_org=_require(ident.publisher_org, "spec.arm.artifact.publisher_org"),
        toolchain_digest=_require(
            ident.toolchain_digest, "spec.arm.artifact.toolchain_digest"),
        env_seal_digest=_require(ident.env_seal_digest, "spec.arm.artifact.env_seal_digest"),
    )


def _topology(spec: Any) -> TopologyRef:
    if not spec.HasField("topology"):
        raise PlanRefusal("spec.topology", "a Topology")
    top = spec.topology
    accelerator = _require(top.accelerator, "spec.topology.accelerator", '"cuda" or "none"')
    if accelerator not in ("cuda", "none"):
        raise PlanRefusal("spec.topology.accelerator", '"cuda" or "none"', accelerator)
    groups = int(top.execution_groups)
    if groups < 1:
        raise PlanRefusal("spec.topology.execution_groups", "at least 1", str(groups))
    return TopologyRef(
        accelerator=accelerator,
        gpu_count=int(top.gpu_count),
        execution_groups=groups,
        device_ordinals=tuple(int(o) for o in top.device_ordinals),
    )


def _slots(spec: Any) -> tuple[SlotRef, ...]:
    if not spec.HasField("components"):
        raise PlanRefusal("spec.components", "a ComponentManifest")
    out: list[SlotRef] = []
    seen: set[str] = set()
    for binding in spec.components.slots:
        slot = _require(binding.slot, "spec.components.slots[].slot")
        if slot in seen:
            raise PlanRefusal(
                f"spec.components.slots[{slot}]", "one binding per slot", "a duplicate")
        seen.add(slot)
        out.append(SlotRef(
            slot=slot,
            ref=_require(binding.ref, f"spec.components.slots[{slot}].ref"),
            # The digest is what pins the bytes. Without it the grant's
            # digest-keyed snapshot map cannot be joined and the worker would
            # be back to trusting a name.
            snapshot_digest=_require(
                binding.snapshot_digest, f"spec.components.slots[{slot}].snapshot_digest"),
            components=tuple(
                ComponentRef(
                    component=_require(
                        comp.component,
                        f"spec.components.slots[{slot}].components[].component"),
                    ref=_require(
                        comp.ref, f"spec.components.slots[{slot}].components[].ref"),
                    snapshot_digest=_require(
                        comp.snapshot_digest,
                        f"spec.components.slots[{slot}].components[].snapshot_digest"),
                )
                for comp in binding.components
            ),
            objective=str(binding.objective),
            distilled=bool(binding.distilled),
            distilled_status=str(binding.distilled_status),
            inference_defaults=str(binding.inference_defaults),
        ))
    return tuple(out)


def _adapters(spec: Any) -> tuple[AdapterRef, ...]:
    out: list[AdapterRef] = []
    for adapter in spec.adapters:
        slot = _require(adapter.slot, "spec.adapters[].slot")
        weight = float(adapter.weight)
        # The hub validates the range; the worker mirror-checks it because an
        # out-of-range overlay silently produces a different image, and a
        # silent product change is the defect class this whole contract exists
        # to close.
        if not -4.0 <= weight <= 4.0:
            raise PlanRefusal(
                f"spec.adapters[{slot}].weight", "a weight in [-4, 4]", str(weight))
        out.append(AdapterRef(
            slot=slot,
            ref=_require(adapter.ref, f"spec.adapters[{slot}].ref"),
            snapshot_digest=_require(
                adapter.snapshot_digest, f"spec.adapters[{slot}].snapshot_digest"),
            weight=weight,
            inference_defaults=str(adapter.inference_defaults),
        ))
    return tuple(out)


def _output_mode(spec: Any) -> str:
    token = _OUTPUT_MODE_TOKENS.get(int(spec.output_mode))
    if token is None:
        raise PlanRefusal(
            "spec.output_mode", "a known OutputMode", str(int(spec.output_mode)))
    return token


def _input_assets(spec: Any) -> tuple[InputAssetRef, ...]:
    out: list[InputAssetRef] = []
    seen: set[str] = set()
    for asset in spec.input_assets:
        asset_id = _require(asset.asset_id, "spec.input_assets[].asset_id")
        if asset_id in seen:
            raise PlanRefusal(
                f"spec.input_assets[{asset_id}]", "one entry per asset_id", "a duplicate")
        seen.add(asset_id)
        out.append(InputAssetRef(
            asset_id=asset_id,
            source_ref=str(asset.source_ref),
            # Identity, not transport: this is the half of the input-asset
            # split that rides the spec digest.
            digest=_require(asset.digest, f"spec.input_assets[{asset_id}].digest"),
            size_bytes=int(asset.size_bytes),
            kind=str(asset.kind),
            mime_type=str(asset.mime_type),
        ))
    return tuple(out)


def _exploration(spec: Any) -> ExplorationRef | None:
    if not spec.HasField("exploration"):
        return None
    return ExplorationRef(
        exploration_id=_require(
            spec.exploration.exploration_id, "spec.exploration.exploration_id"),
        operator_intent_id=str(spec.exploration.operator_intent_id),
    )


__all__ = [
    "SUPPORTED_SPEC_VERSIONS",
    "AdapterRef",
    "ArmRef",
    "ArtifactRef",
    "AttemptRef",
    "AttributionRef",
    "ComponentRef",
    "ConfigRef",
    "ContractsRef",
    "ExplorationRef",
    "InputAssetRef",
    "LaneRef",
    "Plan",
    "PlanConflict",
    "PlanFactory",
    "PlanLedger",
    "PlanRefusal",
    "ReleaseRef",
    "SlotRef",
    "TopologyRef",
]
