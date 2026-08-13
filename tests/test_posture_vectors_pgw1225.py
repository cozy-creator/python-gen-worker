"""Layer 1 of the posture wire fence, THIS side (th#1871 P1 / pgw#1225).

``tests/testdata/posture_wire_vectors.json`` is vendored byte-identically in
tensorhub. Each repo owes the shared ledger one half of the property:

* tensorhub proves every ``wire`` object DECODES to a ``measurement.Posture``
  that digests to the recorded ``posture_digest``, and that the identity RULES
  hold (which pairs of postures are one measurement and which are two).
* this file proves every ``wire`` object is exactly what THIS SDK produces —
  the shape, the field names, and the omission rules.

**The digests are opaque here, deliberately.** The worker does not compute one:
a second canonicalization of the same object is two implementations free to
disagree while both suites stay green, which is pgw#1188's topology divergence
one contract over. We pin them; we never derive them.

``scripts/posture-vector-drift.sh`` is layer 2 and proves the two copies are the
same bytes. Byte-identical fixtures alone only prove both sides are being asked
the same questions — this file is what makes the answers binding here.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, cast

import pytest

from gen_worker import measured_posture as mp
from gen_worker.pb import worker_scheduler_pb2 as pb

VECTORS = pathlib.Path(__file__).parent / "testdata" / "posture_wire_vectors.json"


def _doc() -> Mapping[str, object]:
    return cast(Mapping[str, object], json.loads(VECTORS.read_text()))


def _vectors() -> List[Mapping[str, object]]:
    return cast(List[Mapping[str, object]], _doc()["vectors"])


def _rules() -> List[Mapping[str, object]]:
    return cast(List[Mapping[str, object]], _doc()["identity_rules"])


def _str(src: Mapping[str, object], key: str) -> str:
    return str(src.get(key, "") or "")


def _int(src: Mapping[str, object], key: str) -> int:
    return int(cast(int, src.get(key, 0) or 0))


def _float(src: Mapping[str, object], key: str) -> float:
    return float(cast(float, src.get(key, 0.0) or 0.0))


def _wire_json(posture: mp.MeasuredPosture) -> Dict[str, object]:
    """The posture as the HUB will hold it — the JSON shape of Go's
    ``measurement.Posture``, key for key, omission for omission.

    THIS LIVES IN THE TEST, not in the SDK, and that placement is the point.
    The worker sends PROTO; the hub builds this JSON. A renderer shipped in
    `gen_worker` would be a second implementation of the hub's encoder — the
    same mistake the digest is deliberately not making one field over. Here it
    is a transcription whose only job is to be checked against the ledger.

    Omission follows the Go tags exactly: `omitempty` everywhere except
    `execution_lane` and `attention_backend`, which the hub always renders.
    """
    out: Dict[str, object] = {
        "execution_lane": posture.execution_lane,
        "attention_backend": posture.attention_backend,
    }
    if posture.attention_backend_wanted:
        out["attention_backend_wanted"] = posture.attention_backend_wanted
    if posture.compile_state:
        out["compile_state"] = posture.compile_state
    if posture.compile_state_wanted:
        out["compile_state_wanted"] = posture.compile_state_wanted
    if posture.residency_mode:
        out["residency_mode"] = posture.residency_mode
    if posture.applied:
        applied: List[Dict[str, object]] = []
        for technique in posture.applied:
            entry: Dict[str, object] = {"name": technique.name}
            if technique.component:
                entry["component"] = technique.component
            if technique.wanted:
                entry["wanted"] = technique.wanted
            if technique.reason:
                entry["reason"] = technique.reason
            if technique.est_slowdown:
                entry["est_slowdown"] = technique.est_slowdown
            applied.append(entry)
        out["applied"] = applied
    if posture.components:
        components: List[Dict[str, object]] = []
        for component in posture.components:
            item: Dict[str, object] = {"component": component.component}
            if component.applied_quant:
                item["applied_quant"] = component.applied_quant
            if component.bound_quant:
                item["bound_quant"] = component.bound_quant
            if component.placement:
                item["placement"] = component.placement
            if component.size_bytes:
                item["bytes"] = component.size_bytes
            components.append(item)
        out["components"] = components
    if posture.shortfall is not None:
        shortfall: Dict[str, object] = {"resource": posture.shortfall.resource}
        if posture.shortfall.component:
            shortfall["component"] = posture.shortfall.component
        shortfall["needed_bytes"] = posture.shortfall.needed_bytes
        shortfall["available_bytes"] = posture.shortfall.available_bytes
        out["shortfall"] = shortfall
    return out


def _posture_from_proto(msg: object) -> mp.MeasuredPosture:
    """Wire -> record, for the round-trip assertion below. Also test-owned: the
    worker never reads a posture back in production, and a decoder in the SDK
    would be surface with no caller."""
    proto = cast(pb.MeasuredPosture, msg)
    shortfall: Optional[mp.ResourceShortfall] = None
    if proto.HasField("shortfall"):
        shortfall = mp.ResourceShortfall(
            resource=proto.shortfall.resource,
            component=proto.shortfall.component,
            needed_bytes=proto.shortfall.needed_bytes,
            available_bytes=proto.shortfall.available_bytes)
    return mp.MeasuredPosture(
        execution_lane=proto.execution_lane,
        attention_backend=proto.attention_backend,
        attention_backend_wanted=proto.attention_backend_wanted,
        compile_state=proto.compile_state,
        compile_state_wanted=proto.compile_state_wanted,
        residency_mode=proto.residency_mode,
        applied=tuple(
            mp.AppliedTechnique(
                name=t.name, component=t.component, wanted=t.wanted,
                reason=t.reason, est_slowdown=t.est_slowdown)
            for t in proto.applied),
        components=tuple(
            mp.ComponentPosture(
                component=c.component, applied_quant=c.applied_quant,
                bound_quant=c.bound_quant, placement=c.placement,
                size_bytes=c.bytes)
            for c in proto.components),
        shortfall=shortfall,
    )


def _posture_from_wire(wire: Mapping[str, object]) -> mp.MeasuredPosture:
    """Rebuild the SDK type from the ledger's wire object.

    Transcribed by hand rather than shipped as a library function on purpose:
    the point of the fence is that the field NAMES agree, and a decoder derived
    from the encoder would agree with itself under any rename.
    """
    applied_raw = cast(Sequence[Mapping[str, object]], wire.get("applied", []) or [])
    components_raw = cast(
        Sequence[Mapping[str, object]], wire.get("components", []) or [])
    shortfall_raw = cast(
        Optional[Mapping[str, object]], wire.get("shortfall") or None)
    shortfall: Optional[mp.ResourceShortfall] = None
    if shortfall_raw is not None:
        shortfall = mp.ResourceShortfall(
            resource=_str(shortfall_raw, "resource"),
            component=_str(shortfall_raw, "component"),
            needed_bytes=_int(shortfall_raw, "needed_bytes"),
            available_bytes=_int(shortfall_raw, "available_bytes"),
        )
    return mp.MeasuredPosture(
        execution_lane=_str(wire, "execution_lane"),
        attention_backend=_str(wire, "attention_backend"),
        attention_backend_wanted=_str(wire, "attention_backend_wanted"),
        compile_state=_str(wire, "compile_state"),
        compile_state_wanted=_str(wire, "compile_state_wanted"),
        residency_mode=_str(wire, "residency_mode"),
        applied=tuple(
            mp.AppliedTechnique(
                name=_str(t, "name"), component=_str(t, "component"),
                wanted=_str(t, "wanted"), reason=_str(t, "reason"),
                est_slowdown=_float(t, "est_slowdown"))
            for t in applied_raw),
        components=tuple(
            mp.ComponentPosture(
                component=_str(c, "component"),
                applied_quant=_str(c, "applied_quant"),
                bound_quant=_str(c, "bound_quant"),
                placement=_str(c, "placement"),
                size_bytes=_int(c, "bytes"))
            for c in components_raw),
        shortfall=shortfall,
    )


@pytest.mark.parametrize("vector", _vectors(), ids=lambda v: str(v["name"]))
def test_every_vector_is_what_this_sdk_serializes(
    vector: Mapping[str, object],
) -> None:
    """The producer half: the SDK's wire shape IS the ledger's."""
    wire = cast(Mapping[str, object], vector["wire"])
    posture = _posture_from_wire(wire)
    assert _wire_json(posture) == wire, (
        f"{vector['name']}: this SDK serializes a posture differently than the "
        f"ledger tensorhub digests. That re-keys the measurement relation "
        f"silently — every affected cell forks. ({vector['why']})")


@pytest.mark.parametrize("vector", _vectors(), ids=lambda v: str(v["name"]))
def test_every_vector_survives_the_proto_round_trip(
    vector: Mapping[str, object],
) -> None:
    """And it is the same posture after the WIRE, not only in Python.

    A field that exists on the dataclass but not on the proto message would pass
    the serialization test above (both sides are this repo's own code) and be
    dropped on the way to the hub. This is the only assertion that can see that.
    """
    posture = _posture_from_wire(cast(Mapping[str, object], vector["wire"]))
    back = _posture_from_proto(posture.to_proto())
    assert _wire_json(back) == _wire_json(posture), (
        f"{vector['name']}: the posture did not survive the proto round trip — "
        f"a field the message cannot carry is a field the hub never sees")


def test_degraded_verdict_matches_the_ledger() -> None:
    """The worker's own reading of "was this degraded" must match the hub's.

    Not a duplicate of the hub's test: this one proves the SDK's `degraded`
    (which drives the worker's own log line) cannot say the opposite of the
    column the operator filters on.
    """
    for vector in _vectors():
        posture = _posture_from_wire(cast(Mapping[str, object], vector["wire"]))
        expected = bool(vector["degraded"])
        assert posture.degraded == expected, (
            f"{vector['name']}: SDK degraded={posture.degraded}, hub says "
            f"{expected} — the pod's own logs would contradict the measurement")


def test_identity_rules_name_real_vectors_and_state_a_structural_claim() -> None:
    """What the WORKER can check about the rules, which is not the digests.

    Each rule claims two postures are (or are not) one measurement. This side
    cannot evaluate that — it has no digest. What it CAN do is refuse a ledger
    whose rules point at vectors that do not exist, which is how a rename turns
    a fence into decoration on one side while the other stays green.
    """
    names = {str(v["name"]) for v in _vectors()}
    assert names, "the vector ledger is empty"
    rules = _rules()
    assert rules, "a fence with no rules passes for the same reason an empty file does"
    for rule in rules:
        for side in ("a", "b"):
            assert str(rule[side]) in names, (
                f"identity rule {rule['rule']!r} names unknown vector "
                f"{rule[side]!r}")


def test_the_ie707_pair_differs_only_in_what_was_asked_for() -> None:
    """The structural claim under the wanted-is-identity rule, checkable here.

    tensorhub proves the two DIGEST differently. This proves they are the pair
    the rule says they are — two runs that produced the same numbers and mean
    opposite things. If a future edit makes them differ in some second field,
    the hub's rule keeps passing for the wrong reason, which is exactly how the
    first draft of that rule could not fail (see the hub-side test's header).
    """
    by_name: Dict[str, Mapping[str, object]] = {
        str(v["name"]): cast(Mapping[str, object], v["wire"]) for v in _vectors()}
    flash = dict(by_name["isolated_wanted_flash"])
    sdpa = dict(by_name["isolated_wanted_sdpa"])
    differing: Tuple[str, ...] = tuple(
        sorted(k for k in set(flash) | set(sdpa)
               if flash.get(k) != sdpa.get(k)))
    assert differing == ("attention_backend_wanted",), (
        f"the isolated ie#707 pair differs in {differing}, not in the wanted "
        f"side alone — the hub's wanted-is-identity rule would then pass for a "
        f"reason that is not the rule")
