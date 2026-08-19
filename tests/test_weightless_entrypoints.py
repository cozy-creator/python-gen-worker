"""A WEIGHTLESS entrypoint declares, discovers, derives and SERVES.

# pgw#1392: model-less entrypoints are legal -- zero model slots is a valid
# declaration and the envelope has no model field (se#757 blocker C).

Ten shipped production functions -- `dj-utils`, `music-analysis`,
`quality-benchmark` -- have the signature `def f(ctx, payload) -> Out` with
no model anywhere. This drives one of them through the whole real path on
CPU (no weights, no GPU, no model download) and pins the guarantees KEPT.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import pytest

from gen_worker.serving.context import DeployBinding
from gen_worker.serving.entrypoints import ENTRYPOINT_ATTR, EntrypointDeclarationError
from gen_worker.serving.envelope import EnvelopeError, decode_envelope
from gen_worker.serving.host import EndpointHost
from gen_worker.serving.loader import load_endpoint_module

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"
MODULE = "weightless_endpoint"


@pytest.fixture(scope="module")
def weightless() -> Iterator[ModuleType]:
    sys.path.insert(0, str(FIXTURES))
    try:
        yield importlib.import_module(MODULE)
    finally:
        sys.path.remove(str(FIXTURES))


# -- declaration ------------------------------------------------------------


def test_zero_model_slots_is_a_legal_declaration(weightless: ModuleType) -> None:
    spec = getattr(weightless.transform, ENTRYPOINT_ATTR)
    assert spec.name == "transform"
    assert spec.slots == ()
    assert spec.model_params == ()
    assert spec.model_classes == ()
    assert spec.payload_type is weightless.TransformInput
    assert spec.return_type is weightless.TransformOutput


def test_zero_slots_is_legal_but_junk_slots_are_not() -> None:
    """Permitting ZERO slots is not permitting JUNK slots."""

    source = (
        "import msgspec\n"
        "from gen_worker import Model, RequestContext, entrypoint\n"
        "class In(msgspec.Struct): text: str\n"
        "class Out(msgspec.Struct): text: str\n"
    )

    def declare(signature: str) -> None:
        # A REAL module-level declaration: @entrypoint refuses nested
        # functions first, so an in-function probe measures the wrong wall.
        module = type(sys)("pgw1392_probe")
        module.__dict__["__name__"] = "pgw1392_probe"
        sys.modules["pgw1392_probe"] = module
        try:
            exec(source + signature, module.__dict__)  # noqa: S102
        finally:
            del sys.modules["pgw1392_probe"]

    # The one that must now PASS.
    declare("@entrypoint\ndef f(ctx: RequestContext, payload: In) -> Out: ...\n")

    kept = {
        "junk third parameter": (
            "@entrypoint\n"
            "def f(ctx: RequestContext, payload: In, junk: int) -> Out: ...\n"
        ),
        "bad ctx annotation": (
            "@entrypoint\ndef f(ctx: str, payload: In) -> Out: ...\n"
        ),
        "non-Struct payload": (
            "@entrypoint\ndef f(ctx: RequestContext, payload: dict) -> Out: ...\n"
        ),
        "non-Struct return": (
            "@entrypoint\ndef f(ctx: RequestContext, payload: In) -> dict: ...\n"
        ),
        "keyword-only parameter": (
            "@entrypoint\n"
            "def f(ctx: RequestContext, payload: In, *, flag: bool = False)"
            " -> Out: ...\n"
        ),
        "payload before ctx": (
            "@entrypoint\ndef f(payload: In, ctx: RequestContext) -> Out: ...\n"
        ),
        "fewer than two parameters": (
            "@entrypoint\ndef f(ctx: RequestContext) -> Out: ...\n"
        ),
        "bare Model base as a slot": (
            "@entrypoint\n"
            "def f(ctx: RequestContext, payload: In, model: Model) -> Out: ...\n"
        ),
    }
    for label, signature in kept.items():
        with pytest.raises(EntrypointDeclarationError, match=r".") as caught:
            declare(signature)
        assert caught.value, label


# -- discovery --------------------------------------------------------------


def test_discovery_publishes_a_row_with_no_slots(weightless: ModuleType) -> None:
    from gen_worker.discovery.entrypoints_v2 import (
        assert_manifest_advertises_something,
        discover_entrypoints,
        entrypoints_block,
    )

    rows = discover_entrypoints(MODULE)
    by_name = {row["name"]: row for row in rows}
    assert set(by_name) == {"transform", "closure_gate"}
    for row in rows:
        assert row["slots"] == []
        assert row["input_schema"] and row["output_schema"]

    # A weightless endpoint advertises something: the build does not stop.
    assert_manifest_advertises_something({"entrypoints": entrypoints_block(rows)})


# -- release derive ---------------------------------------------------------


def test_derive_renders_an_envelope_with_no_model_field(
    weightless: ModuleType, tmp_path: Path
) -> None:
    # se#786/pgw#1462: vendored torchcg ships in the wheel — never a skip.
    from gen_worker.release.derive import derive_release

    # pgw#1489: a derive states the compile stack it traced under, read from
    # the endpoint's own uv.lock. There is no installed-set fallback.
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text('version = 1\n\n[[package]]\nname = "torch"\nversion = "2.13.0"\n')
    result = derive_release(weightless, checkpoint_dir=tmp_path, lockfile=lockfile)
    document = json.loads(result.document)

    # No model class -> no lane, no trace subject, no contract.
    assert document["graphs"]["lanes"] == []
    assert document["lane_contracts"] == {}
    assert document["model_type"] is None
    assert document["endpoint"] == MODULE  # no ':Model' half

    # ...but the entrypoints ARE published: the hub's API docs are the point.
    assert set(document["entrypoints"]) == {"transform", "closure_gate"}
    entry = document["entrypoints"]["transform"]
    assert entry["model_slots"] == {}, "an honest empty mapping, never a fake one"
    assert entry["traced_passes"] == 0

    # THE ENVELOPE HAS NO MODEL FIELD -- absent renders as ABSENT, not null,
    # not empty.
    schema = entry["envelope_schema"]
    assert sorted(schema["properties"]) == ["input"]
    assert schema["required"] == ["input"]
    assert "model" not in json.dumps(schema)

    # "No lanes" has two causes and the log may not conflate them: a model
    # held eagerly (`eager_only=`) vs NO MODEL AT ALL.
    assert result.eager_permanent and result.weightless

    sys.path.insert(0, str(FIXTURES))
    try:
        eager = derive_release(
            importlib.import_module("eager_endpoint"), checkpoint_dir=tmp_path,
            lockfile=lockfile,
        )
    finally:
        sys.path.remove(str(FIXTURES))
    assert eager.eager_permanent and not eager.weightless
    # pgw#1488: eager-by-DECLARATION carries the author's reason, and the
    # reason is the difference between this and "traced, nothing marked".
    assert eager.eager_only.startswith("the fixture's subject")
    assert not result.eager_only
    assert json.loads(eager.document)["entrypoints"] == {}


# -- serve ------------------------------------------------------------------


def test_a_weightless_entrypoint_actually_serves(weightless: ModuleType) -> None:
    loaded = load_endpoint_module(MODULE)
    assert loaded.models == ()
    assert sorted(loaded.entrypoints) == ["closure_gate", "transform"]

    host = EndpointHost(
        loaded, DeployBinding(checkpoint_ref="", checkpoint_dir=Path("."))
    )
    with pytest.raises(RuntimeError, match="boot the endpoint first"):
        host.dispatch("transform", {"text": "x"}, request_id="early")

    host.setup()  # nothing to load: the loop over loaded.models is empty
    assert host.instances == {}

    out = host.dispatch(
        "transform",
        {"text": "cozy", "upper": True, "repeat": 2},
        request_id="pgw1392",
    )
    assert isinstance(out, weightless.TransformOutput)
    assert out.text == "COZYCOZY" and out.length == 8

    gate = host.dispatch(
        "closure_gate",
        {"values": [0.1, 0.9, 0.7], "threshold": 0.5},
        request_id="pgw1392b",
    )
    assert gate.passed and gate.above == 2


def test_the_wire_path_serves_and_takes_no_lease(weightless: ModuleType) -> None:
    """The full envelope -> invoke -> outcome path. Nothing is made resident."""

    from gen_worker.serving.residency import ResidencyManager
    from gen_worker.serving.serve_loop import ServeLoop

    class NeverResolver:
        def resolve(self, model_cls: type, checkpoint_ref: str) -> Any:
            raise AssertionError("a weightless request resolved a binding")

        def default_pick(self, model_cls: type, slot_name: str) -> str:
            raise AssertionError("a weightless request asked for a default pick")

    class NeverSizer:
        def resident_bytes(self, checkpoint_ref: str, lane: str) -> int:
            raise AssertionError("a weightless request sized a residency slot")

        def activation_headroom_bytes(self, checkpoint_ref: str, lane: str) -> int:
            raise AssertionError("a weightless request reserved activation bytes")

    loop = ServeLoop(
        load_endpoint_module(MODULE),
        residency=ResidencyManager(1 << 30, NeverSizer()),
        resolver=NeverResolver(),
    )
    outcome = loop.invoke(
        "transform",
        {"input": {"text": "weightless"}},
        request_id="pgw1392-wire",
    )
    assert outcome.result.text == "weightless"
    assert outcome.warnings == ("transformed 10 chars",)


def test_the_envelope_refuses_a_model_pick_by_name(weightless: ModuleType) -> None:
    spec = load_endpoint_module(MODULE).entrypoints["transform"]
    for envelope in (
        {"input": {"text": "x"}, "model": "org/repo@rel"},
        {"input": {"text": "x"}, "models": {"model": "org/repo@rel"}},
    ):
        with pytest.raises(EnvelopeError, match="declares no model slot"):
            decode_envelope(spec, envelope)

    # ...and the happy path decodes with no picks at all.
    decoded = decode_envelope(spec, {"input": {"text": "x"}})
    assert decoded.model_picks == ()
    assert decoded.adapter_values == ()
    assert decoded.payload.text == "x"
