"""The orphaned serve-path wires, exercised through the seam that lost them.

pgw#1418 measured one of these on a rented pod; pgw#1425 found nine siblings
sitting in `scripts/unreached_surface_baseline.txt`, put there wholesale by a
`--write-baseline` run. Every case here drives the WIRE, never the function:
the bug in all ten was that nothing CALLED a function that worked fine.
"""

from __future__ import annotations

import http.server
import importlib
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import msgspec
import pytest

from gen_worker import ImageAsset, receipts
from gen_worker.discovery.moderation import payload_moderation
from gen_worker.serving.loader import load_endpoint_module
from gen_worker.serving.residency import ResidencyManager
from gen_worker.serving.serve_loop import ServeLoop

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"
MODULE = "media_endpoint"

_CLIP_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"video" * 64
_TRACK_BYTES = b"RIFF....WAVEfmt " + b"audio" * 64


class UnorderedAssets(msgspec.Struct):
    """Module scope on purpose: `get_type_hints` resolves against module
    globals, and a locally-declared struct is the OTHER refusal below."""

    images: set[ImageAsset]


@pytest.fixture(scope="module")
def media() -> Iterator[ModuleType]:
    sys.path.insert(0, str(FIXTURES))
    try:
        yield importlib.import_module(MODULE)
    finally:
        sys.path.remove(str(FIXTURES))


class _NeverResolver:
    def resolve(self, model_cls: type, checkpoint_ref: str) -> Any:
        raise AssertionError("a weightless request resolved a binding")

    def default_pick(self, model_cls: type, slot_name: str) -> str:
        raise AssertionError("a weightless request asked for a default pick")


class _NeverSizer:
    def resident_bytes(self, checkpoint_ref: str, lane: str) -> int:
        raise AssertionError("a weightless request sized a residency slot")

    def activation_headroom_bytes(self, checkpoint_ref: str, lane: str) -> int:
        raise AssertionError("a weightless request reserved activation bytes")


def _loop() -> ServeLoop:
    return ServeLoop(
        load_endpoint_module(MODULE),
        residency=ResidencyManager(1 << 30, _NeverSizer()),
        resolver=_NeverResolver(),
    )


@pytest.fixture()
def origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A real HTTP origin on loopback, serving real bytes.

    The ONE seam faked: the SSRF policy blocks loopback, so a local origin
    cannot be a caller transport without lifting that check. Everything else —
    URL validation shape, the streamed download, the byte cap, the mime sniff,
    the `local_path` assignment and the attempt-directory cleanup — is the
    production code path.
    """
    root = tmp_path / "origin"
    root.mkdir()
    (root / "clip.mp4").write_bytes(_CLIP_BYTES)
    (root / "track.wav").write_bytes(_TRACK_BYTES)

    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), lambda *a, **kw: handler(*a, directory=str(root), **kw)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    from gen_worker import input_assets as ia

    monkeypatch.setattr(ia, "_url_is_blocked", lambda url: False)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


# -- pgw#1418, half one: the moderation block ------------------------------


def test_the_moderation_block_names_every_media_and_prompt_path(
    media: ModuleType,
) -> None:
    """The hub will not guess which payload fields are media. This is the only
    thing that tells it, and the v2 row emitted no such key at all."""
    block = payload_moderation(media.ExtractFrameInput)
    assert block["media"] == [
        {"field": "video", "kind": "video"},
        {"field": "cues[].clip", "kind": "video"},
    ]
    assert block["prompts"] == [
        {"field": "caption", "role": "positive"},
        {"field": "cues[].note", "role": "negative"},
    ]

    audio = payload_moderation(media.AnalyzeInput)
    assert audio["media"] == [{"field": "audio", "kind": "audio"}]
    assert "prompts" not in audio


def test_a_text_only_payload_emits_no_block_at_all() -> None:
    """Absence is the answer for a payload with neither, so the row of every
    existing text endpoint stays byte-identical."""
    import msgspec

    class TextOnly(msgspec.Struct):
        text: str
        count: int = 1

    assert payload_moderation(TextOnly) == {}


def test_discovery_emits_the_block_on_the_manifest_row(media: ModuleType) -> None:
    """The WIRE assertion: the block reaches `entrypoints[]`, which is what the
    hub decodes into `PayloadModerationMetadata`."""
    from gen_worker.discovery.entrypoints_v2 import discover_entrypoints

    sys.path.insert(0, str(FIXTURES))
    try:
        rows = {row["name"]: row for row in discover_entrypoints(MODULE)}
    finally:
        sys.path.remove(str(FIXTURES))

    assert set(rows) == {"extract_frame", "analyze"}
    assert rows["analyze"]["moderation"] == {
        "media": [{"field": "audio", "kind": "audio"}]
    }
    assert rows["extract_frame"]["moderation"]["media"][0] == {
        "field": "video", "kind": "video"
    }


def test_an_asset_on_an_unordered_container_is_a_build_error() -> None:
    """The input manifest is ORDERED; a set has no stable occurrence order."""
    with pytest.raises(ValueError, match="unordered set/frozenset"):
        payload_moderation(UnorderedAssets)


def test_unresolvable_hints_REFUSE_instead_of_emitting_an_empty_block() -> None:
    """Found by RUNNING this file, not by reading it.

    Under `from __future__ import annotations` every annotation is a STRING
    until `get_type_hints` resolves it. The v1 collector swallowed a resolution
    failure and fell back to `__annotations__` — strings — which the walk skips,
    so the block came out EMPTY: no media, no prompts, no error, and an
    endpoint that cannot be served an asset. Same silence class as pgw#1418
    itself, one layer down. It now refuses, naming the struct.
    """
    module = ModuleType("pgw1425_unresolvable")
    exec(  # noqa: S102 — a REAL module whose annotations cannot be resolved
        "from __future__ import annotations\n"
        "import msgspec\n"
        "class Payload(msgspec.Struct):\n"
        "    image: SomeAssetTypeTheModuleNeverImported\n",
        module.__dict__,
    )
    with pytest.raises(ValueError, match="cannot resolve the type hints"):
        payload_moderation(module.Payload)


# -- pgw#1418, half two: materialization at the serve seam ------------------


def test_the_serve_path_materializes_a_typed_media_input(
    media: ModuleType, origin: str
) -> None:
    """THE regression. The endpoint's own `_local_path` raises
    `video asset not materialized` when `local_path` is falsy — so this passes
    only if the PLATFORM filled it before the body ran."""
    outcome = _loop().invoke(
        "extract_frame",
        {
            "input": {
                "video": {"ref": f"{origin}/clip.mp4"},
                "caption": "a cat",
                "cues": [{"clip": {"ref": f"{origin}/clip.mp4"}, "note": "blur"}],
            }
        },
        request_id="pgw1418-drive",
        attempt=1,
    )
    assert outcome.result.size_bytes == len(_CLIP_BYTES)
    assert Path(outcome.result.local_path).exists()
    # A duplicate occurrence shares ONE worker-owned download.
    assert outcome.result.nested_paths == [outcome.result.local_path]


def test_the_regression_itself_without_the_wire(
    media: ModuleType, origin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED-PROOF: remove the call the fix added and the measured failure comes
    back verbatim. Without this, a green suite proves only that the fixture is
    easy to satisfy."""
    from gen_worker.api.errors import ValidationError
    from gen_worker.serving import serve_loop as loop_mod

    monkeypatch.setattr(
        loop_mod, "materialize_input_assets", lambda *a, **kw: 0
    )
    with pytest.raises(ValidationError, match="video asset not materialized"):
        _loop().invoke(
            "extract_frame",
            {"input": {"video": {"ref": f"{origin}/clip.mp4"}}},
            request_id="pgw1418-red",
            attempt=1,
        )


def test_the_attempt_directory_is_gone_when_materialization_fails(
    media: ModuleType, tmp_path: Path
) -> None:
    """A blocked transport fails CLOSED and leaves no scratch behind."""
    from gen_worker.api.errors import ValidationError
    from gen_worker.input_assets import inputs_dir_for_request

    with pytest.raises(ValidationError, match="unsupported_input_asset_scheme"):
        _loop().invoke(
            "analyze",
            {"input": {"audio": {"ref": "gopher://nope/track.wav"}}},
            request_id="pgw1418-refuse",
            attempt=1,
        )
    assert not inputs_dir_for_request("pgw1418-refuse", 1).exists()


# -- the stage timer, read out at last -------------------------------------


def test_the_outcome_carries_the_stage_timer(media: ModuleType, origin: str) -> None:
    """`RequestContext` always FILLED a StageTimer; nothing read it out, so
    every v2 `JobResult.metrics` carried two numbers and no breakdown."""
    from gen_worker.stage_timing import stage_ms_for_metrics

    outcome = _loop().invoke(
        "analyze",
        {"input": {"audio": {"ref": f"{origin}/track.wav"}}},
        request_id="pgw1425-stages",
        attempt=1,
    )
    assert outcome.stages is not None
    rendered = stage_ms_for_metrics(outcome.stages, 5)
    # `input_fetch` is a PRE stage: reported, never folded into runtime_ms.
    assert "input_fetch" in rendered


def test_on_context_hands_out_the_live_request_context(
    media: ModuleType, origin: str
) -> None:
    """The seam the capability-renewal loop needs: the token lives on the
    context and the context is built inside `invoke`."""
    seen: list[Any] = []
    _loop().invoke(
        "analyze",
        {"input": {"audio": {"ref": f"{origin}/track.wav"}}},
        request_id="pgw1425-ctx",
        attempt=1,
        context={"worker_capability_token": "cap-abc"},
        on_context=seen.append,
    )
    assert len(seen) == 1
    assert seen[0]._worker_capability_token == "cap-abc"


# -- receipts: the fail-OPEN, closed ---------------------------------------


def test_the_unconfigured_receipt_gate_now_REFUSES(tmp_path: Path) -> None:
    """pgw#1425's security item. `gate_delivered_artifact` returned True when
    nobody had configured it, and the v2 serve path configured nobody — so
    every fleet worker armed hub-delivered native code with no receipt checked
    at all. The default posture must refuse."""
    artifact = tmp_path / "graph.tar"
    artifact.write_bytes(b"not a real artifact")

    receipts.reset()  # the DEFAULT posture, deliberately
    assert receipts.posture() == receipts.POSTURE_UNSET
    assert receipts.gate_delivered_artifact(artifact, "flux") is False

    receipts.trust_local_store("a test's own tmpdir")
    assert receipts.posture() == receipts.POSTURE_LOCAL
    assert receipts.gate_delivered_artifact(artifact, "flux") is True


def test_the_refusal_is_typed_and_named_on_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent False is a second silence. The refusal rides the same typed
    event every other receipt refusal does, with its own reason."""
    from gen_worker import activity as activity_mod

    events: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        activity_mod,
        "emit_event",
        lambda kind, detail, phase="", **kw: events.append((kind, detail, phase)),
    )
    artifact = tmp_path / "graph.tar"
    artifact.write_bytes(b"not a real artifact")

    receipts.reset()
    assert receipts.gate_delivered_artifact(artifact, "flux") is False
    assert [e for e in events if e[0] == "compiled_graph_receipt_refused"]
    assert any(e[2] == "gate_unconfigured" for e in events)


def test_an_empty_base_url_can_no_longer_arm_nothing_quietly() -> None:
    """`configure("")` used to return silently, which made "the hub named no
    file API" indistinguishable from "HelloAck never arrived"."""
    receipts.reset()
    with pytest.raises(receipts.ReceiptError, match="gate_unconfigurable"):
        receipts.configure("", lambda: "jwt")
    assert receipts.posture() == receipts.POSTURE_UNSET


def test_local_trust_must_be_attributed() -> None:
    receipts.reset()
    with pytest.raises(receipts.ReceiptError, match="local_trust_unattributed"):
        receipts.trust_local_store("   ")
