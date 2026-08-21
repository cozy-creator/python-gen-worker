from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pytest

from gen_worker import activity as activity_mod
from gen_worker import boot_phases
from gen_worker import output_integrity
from gen_worker import request_context as request_context_mod
from gen_worker.boot_materialize import (
    STATE_READY,
    CheckpointConfig,
    CheckpointMaterialization,
)
from gen_worker.api.errors import OutputIntegrityError
from gen_worker.models.refs import WireRef
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.serving import DeployBinding, load_endpoint
from gen_worker.serving.loader import load_endpoint_module
from gen_worker.serving.context import RequestContext
from gen_worker.serving.residency import ResidencyManager
from gen_worker.serving.serve_loop import (
    WARM_FAILED,
    WARM_OK,
    WARM_SKIPPED,
    ServeLoop,
    manifest_sizer,
)
from gen_worker.warm_payload import WARMUP_TEXT, neutral_payload

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_v2_endpoint"
RELEASE_FIXTURES = Path(__file__).resolve().parent / "release_fixtures"

GB = 1024**3
DREAM = "org/dreamshaper@2"
LANE = "sdxl.diffusers@1+plain.bf16@1"


class LocalResolver:
    """Deploy state over local config-only trees — the BindingResolver seam."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.resolved: List[str] = []

    def _tree(self, ref: str) -> Path:
        tree = self.root / ref.replace("/", "_").replace("@", "_")
        if not tree.exists():
            tree.mkdir(parents=True)
            (tree / "config.json").write_text(json.dumps({"seed": len(ref)}))
        return tree

    def resolve(self, model_cls: type, checkpoint_ref: str) -> DeployBinding:
        self.resolved.append(checkpoint_ref)
        return DeployBinding(
            checkpoint_ref=checkpoint_ref,
            checkpoint_dir=self._tree(checkpoint_ref),
            model="sdxl",
            defaults={},
        )

    def default_pick(self, model_cls: type, slot_name: str) -> str:
        return DREAM


def make_loop(tmp_path: Path) -> Tuple[ServeLoop, LocalResolver]:
    loaded = load_endpoint(FIXTURE_DIR)
    resolver = LocalResolver(tmp_path / "trees")
    loop = ServeLoop(
        loaded,
        residency=ResidencyManager(
            64 * GB, manifest_sizer({DREAM: 3 * GB}, headroom_bytes=1 * GB)
        ),
        resolver=resolver,
        lane_contract=LANE,
        output_dir=tmp_path / "outputs",
    )
    return loop, resolver


@pytest.fixture()
def judged_calls(monkeypatch: pytest.MonkeyPatch) -> List[bool]:
    """Record `ctx.boot_warmup` AT THE READER, for every judged output."""
    seen: List[bool] = []
    real = output_integrity.judged

    def recording(ctx: Any) -> bool:
        seen.append(bool(getattr(ctx, "boot_warmup", False)))
        return real(ctx)

    monkeypatch.setattr(request_context_mod, "judged", recording)
    return seen


@pytest.fixture(autouse=True)
def _clean_activity() -> Iterator[None]:
    activity_mod.reset_for_tests()
    boot_phases.reset_for_tests()
    yield
    activity_mod.reset_for_tests()
    boot_phases.reset_for_tests()


def test_the_warm_pass_runs_a_REAL_forward_through_the_REAL_serve_path(
    tmp_path: Path,
) -> None:
    """One synthetic invocation per entrypoint, through `invoke`."""
    loop, resolver = make_loop(tmp_path)

    passes = loop.boot_warmup()

    assert [row.function for row in passes] == ["generate"]
    (row,) = passes
    assert row.outcome == WARM_OK, row
    assert row.warmed and row.reason == ""
    assert set(resolver.resolved) == {DREAM} and len(resolver.resolved) == 2
    assert loop.residency.tier_of(DREAM, f"SdxlModel/{LANE}") is not None


def test_the_flag_is_TRUE_on_a_real_boot_path_and_FALSE_on_a_real_request(
    tmp_path: Path, judged_calls: List[bool]
) -> None:
    """HALF ONE of the issue's must-know, asserted AT THE READER."""
    loop, _ = make_loop(tmp_path)

    loop.boot_warmup()
    assert judged_calls == [True], judged_calls

    loop.invoke(
        "generate",
        {"model": DREAM, "input": {"prompt": "a lighthouse", "seed": 3}},
        request_id="req-1",
    )
    assert judged_calls == [True, False], judged_calls


def test_a_BLANK_warm_output_passes_integrity_and_is_refused_without_the_flag(
    tmp_path: Path,
) -> None:
    """HALF TWO, and the reason the restoration needed it."""
    from PIL import Image

    blank = Image.new("RGB", (64, 64), (128, 128, 128))

    warm: RequestContext[Any] = RequestContext(
        "boot-warmup-generate", boot_warmup=True,
        local_output_dir=str(tmp_path / "warm"),
    )
    asset = warm.save_image(blank, format="png")
    assert asset.ref.endswith(".png")
    assert (tmp_path / "warm" / asset.ref).is_file()

    paying: RequestContext[Any] = RequestContext(
        "req-1", local_output_dir=str(tmp_path / "paying")
    )
    with pytest.raises(OutputIntegrityError):
        paying.save_image(blank, format="png")


def test_the_exemption_is_the_predicate_both_ways() -> None:
    """The reader's own verdict, stated as an assertion rather than inferred from a save that happened to succeed."""
    warm: RequestContext[Any] = RequestContext("r", boot_warmup=True)
    paying: RequestContext[Any] = RequestContext("r")
    assert output_integrity.judged(warm) is False
    assert output_integrity.judged(paying) is True


def test_the_warm_payload_is_the_schema_at_its_NEUTRAL_DEFAULTS(
    tmp_path: Path,
) -> None:
    """The int32 incident's own record of the v1 warm plan: *"a single run at the schema's neutral defaults"*."""
    sys.path.insert(0, str(FIXTURE_DIR / "src"))
    try:
        from serving_v2_fixture.main import (  # type: ignore[import-not-found]
            AspectRatio,
            TextToImageInput,
        )
    finally:
        sys.path.remove(str(FIXTURE_DIR / "src"))

    payload, reason = neutral_payload(TextToImageInput, str(tmp_path))

    assert reason == ""
    assert payload.prompt == WARMUP_TEXT
    assert payload.aspect_ratio is AspectRatio.RATIO_1_1
    assert payload.num_inference_steps is None
    assert payload.guidance_scale is None
    assert payload.scheduler is None
    assert payload.enhance_prompt is True
    assert payload.output_format == "webp"


def test_a_payload_that_cannot_synthesize_is_SKIPPED_with_a_reason(
    tmp_path: Path,
) -> None:
    """Stated, never faked."""
    import msgspec

    from gen_worker.api.types import AudioAsset, ImageAsset, VideoAsset

    globals().update(
        AudioAsset=AudioAsset, ImageAsset=ImageAsset, VideoAsset=VideoAsset
    )

    class NeedsVideo(msgspec.Struct):
        clip: VideoAsset

    payload, reason = neutral_payload(NeedsVideo, str(tmp_path))
    assert payload is None
    assert "clip" in reason and "video" in reason

    class NeedsMedia(msgspec.Struct):
        image: ImageAsset
        audio: AudioAsset

    media, reason = neutral_payload(NeedsMedia, str(tmp_path))
    assert reason == ""
    assert Path(media.image.local_path).is_file()
    assert Path(media.audio.local_path).is_file()
    assert media.image.mime_type == "image/png"


def test_a_weightless_entrypoint_is_SKIPPED_not_warmed(tmp_path: Path) -> None:
    sys.path.insert(0, str(RELEASE_FIXTURES))
    try:
        loaded = load_endpoint_module("weightless_endpoint")
    finally:
        sys.path.remove(str(RELEASE_FIXTURES))
    loop = ServeLoop(
        loaded,
        residency=ResidencyManager(GB, manifest_sizer({}, headroom_bytes=1)),
        resolver=LocalResolver(tmp_path / "trees"),
        output_dir=tmp_path / "outputs",
    )
    passes = loop.boot_warmup()
    assert passes and all(row.outcome == WARM_SKIPPED for row in passes), passes
    assert all("weightless" in row.reason for row in passes), passes


def test_a_FAILING_warm_pass_degrades_loudly_and_does_NOT_brick_the_pod(
    tmp_path: Path,
) -> None:
    """The ruling's posture, and the one place v1's differs."""
    loop, _ = make_loop(tmp_path)
    spec = loop.loaded.entrypoints["generate"]

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("the warm forward exploded")

    loop.loaded.entrypoints["generate"] = dataclasses.replace(spec, fn=boom)
    events: List[pb.ActivityUpdate] = []
    activity_mod._sink = events.append

    passes = loop.boot_warmup()

    (row,) = passes
    assert row.outcome == WARM_FAILED, row
    assert "the warm forward exploded" in row.reason
    degrades = [
        e for e in events
        if e.kind == activity_mod.KIND_SERVE_DEGRADE
        and e.phase == "boot_warmup_failed"
    ]
    assert len(degrades) == 1, [(e.kind, e.phase) for e in events]
    assert "generate" in degrades[0].detail
    assert "the warm forward exploded" in degrades[0].detail

    loop.loaded.entrypoints["generate"] = spec
    outcome = loop.invoke(
        "generate",
        {"model": DREAM, "input": {"prompt": "a lighthouse", "seed": 3}},
        request_id="req-after-a-failed-warm",
    )
    assert outcome.result.model == DREAM


def test_a_function_with_no_boot_binding_is_a_SKIP_not_a_failure(
    tmp_path: Path,
) -> None:
    """`prepare` is the worker's seam for the deploy's picks, and declining is a first-class answer."""
    loop, resolver = make_loop(tmp_path)
    events: List[pb.ActivityUpdate] = []
    activity_mod._sink = events.append

    passes = loop.boot_warmup(prepare=lambda _fn: "no boot-time binding")

    assert [(r.outcome, r.reason) for r in passes] == [
        (WARM_SKIPPED, "no boot-time binding")
    ]
    assert resolver.resolved == []
    assert not [e for e in events if e.kind == activity_mod.KIND_SERVE_DEGRADE]


def test_the_warm_pass_runs_BEFORE_the_worker_is_ready_or_servable() -> None:
    """The placement is the deliverable, not an implementation detail."""
    observed: Dict[str, Any] = {}

    class _Store:
        def replace_desired_snapshots(self, *_a: Any, **_k: Any) -> None:
            return None

        async def announce_resident(self, *_a: Any, **_k: Any) -> bool:
            return True

    announced: List[str] = []

    async def announce() -> None:
        announced.append("announce")

    async def warm() -> None:
        observed["ready"] = mat.ready
        observed["state"] = mat.state
        observed["announced_before_warm"] = list(announced)
        observed["servable_ms"] = boot_phases.servable_ms()
        boot_phases.mark_once(boot_phases.PHASE_WARMUP, function="generate")

    mat = CheckpointMaterialization(
        _Store(), announce=announce, warm=warm  # type: ignore[arg-type]
    )

    async def drive() -> None:
        ref = WireRef("acme/model-a")
        desired = pb.DesiredResidency(generation=1, disk_refs=[str(ref)])
        desired.snapshots[str(ref)].CopyFrom(pb.Snapshot(digest="sha256:beef"))
        mat.configure(CheckpointConfig.from_wire(desired))
        assert mat._task is not None
        await mat._task
        for _ in range(8):
            await asyncio.sleep(0)

    asyncio.run(drive())

    assert observed["ready"] is False, observed
    assert observed["state"] != STATE_READY, observed
    assert observed["announced_before_warm"] == ["announce"], observed
    assert observed["servable_ms"] is None, observed
    assert mat.state == STATE_READY
    assert announced == ["announce", "announce"]


def test_a_raising_warm_callback_never_costs_the_pod_its_boot() -> None:
    """The second belt."""

    class _Store:
        def replace_desired_snapshots(self, *_a: Any, **_k: Any) -> None:
            return None

        async def announce_resident(self, *_a: Any, **_k: Any) -> bool:
            return True

    async def warm() -> None:
        raise RuntimeError("the whole warm pass exploded")

    mat = CheckpointMaterialization(_Store(), warm=warm)  # type: ignore[arg-type]

    async def drive() -> None:
        ref = WireRef("acme/model-a")
        desired = pb.DesiredResidency(generation=1, disk_refs=[str(ref)])
        desired.snapshots[str(ref)].CopyFrom(pb.Snapshot(digest="sha256:beef"))
        mat.configure(CheckpointConfig.from_wire(desired))
        assert mat._task is not None
        await mat._task

    asyncio.run(drive())
    assert mat.state == STATE_READY
    assert mat.failure == ""


def test_PHASE_WARMUP_finally_has_a_producer(tmp_path: Path) -> None:
    """`boot_phases`' own rule, printed above the vocabulary: *"a declared phase with no producer is not coverage we have not gotten to: every reader of the ladder sees a name that can only ever report no..."""
    loop, _ = make_loop(tmp_path)
    loop.boot_warmup()

    table = boot_phases.phase_table()
    warm_rows = [row for row in table if row.phase == boot_phases.PHASE_WARMUP]
    assert warm_rows, [row.phase for row in table]
    assert warm_rows[0].function == "generate", warm_rows[0]


def test_for_request_no_longer_takes_a_slot_it_silently_ignores() -> None:
    """`slot=` was accepted and discarded on BOTH context implementations — the serving one passed `objective=""` regardless, the trace one wrote `del slot`."""
    import inspect

    from gen_worker.release.trace_context import TraceRequestContext

    serving = inspect.signature(RequestContext.for_request).parameters
    trace = inspect.signature(TraceRequestContext.for_request).parameters
    assert "slot" not in serving
    assert "slot" not in trace
    assert set(serving) == set(trace)


def test_the_for_request_docstring_states_the_MEASURED_behaviour() -> None:
    """The docstring promised the objective was applied here and that ambiguity raised; the body passed `objective=""` unconditionally, so the promise was contradicted by the code beneath it and the raise..."""
    doc = RequestContext.for_request.__doc__ or ""
    assert "CARRIES NO OBJECTIVE" in doc
    assert "gen_worker.view" in doc and "objective=" in doc
    assert "Ambiguity raises" not in doc

    source = Path(request_context_mod.__file__).read_text(encoding="utf-8")
    assert 'objective="", generator=gen' in source


def _config_for(ref: str, digest: str = "sha256:beef") -> CheckpointConfig:
    desired = pb.DesiredResidency(generation=1, disk_refs=[ref])
    desired.snapshots[ref].CopyFrom(pb.Snapshot(digest=digest))
    return CheckpointConfig.from_wire(desired)


def test_the_boot_picks_come_off_the_hubs_OWN_DesiredResidency_Hot() -> None:
    """Nothing is invented: `DesiredResidency.Hot` is `repeated DesiredInstance {function_name, models}` and `models` is the very `ModelBinding` a dispatch carries — slot, ref, the recognized `model` name..."""
    from gen_worker.worker import boot_picks

    loaded = load_endpoint(FIXTURE_DIR)
    desired = pb.DesiredResidency(generation=1, disk_refs=[DREAM, "org/other@1"])
    desired.snapshots[DREAM].CopyFrom(pb.Snapshot(digest="sha256:beef"))
    instance = desired.hot.add()
    instance.function_name = "generate"
    binding = instance.models.add()
    binding.slot, binding.ref = "model", DREAM
    binding.model, binding.inference_defaults = "sdxl", '{"steps": 4}'

    picks = boot_picks(desired, loaded, CheckpointConfig.from_wire(desired))

    assert set(picks) == {"generate"}
    table = picks["generate"]
    assert table.by_slot == {"model": DREAM}
    assert table.by_ref[DREAM].model == "sdxl"
    assert table.by_ref[DREAM].inference_defaults == '{"steps": 4}'


def test_one_slot_and_one_ref_is_ARITHMETIC_but_anything_else_is_a_GUESS() -> None:
    """The hub seeds `Hot` for dynamic-slot defaults and compile-cache prewarm; a static-binding release can arrive with it empty."""
    from gen_worker.worker import boot_picks

    loaded = load_endpoint(FIXTURE_DIR)

    single = _config_for(DREAM)
    picks = boot_picks(pb.DesiredResidency(), loaded, single)
    assert picks["generate"].by_slot == {"model": DREAM}
    assert picks["generate"].by_ref[DREAM].manifest_digest == "sha256:beef"
    assert picks["generate"].by_ref[DREAM].model == ""

    two = pb.DesiredResidency(generation=1, disk_refs=[DREAM, "org/other@1"])
    assert boot_picks(two, loaded, CheckpointConfig.from_wire(two)) == {}


def test_a_synthesized_ASSET_survives_the_envelope_the_warm_pass_builds(
    tmp_path: Path,
) -> None:
    """The warm payload becomes an envelope (`msgspec.to_builtins`) and is decoded back by the REAL `decode_envelope` — so `local_path` has to survive the round trip, or an asset-taking entrypoint warms w..."""
    import msgspec

    from gen_worker.serving.entrypoints import ENTRYPOINT_ATTR
    from gen_worker.serving.envelope import decode_envelope

    sys.path.insert(0, str(RELEASE_FIXTURES))
    try:
        import media_endpoint  # type: ignore[import-not-found]
    finally:
        sys.path.remove(str(RELEASE_FIXTURES))
    spec = getattr(media_endpoint.analyze, ENTRYPOINT_ATTR)

    payload, reason = neutral_payload(spec.payload_type, str(tmp_path))
    assert reason == "", reason
    written = Path(payload.audio.local_path)
    assert written.is_file() and written.stat().st_size > 0

    decoded = decode_envelope(spec, {"input": msgspec.to_builtins(payload)})

    assert decoded.payload.audio.local_path == str(written)
    assert decoded.payload.audio.mime_type == "audio/wav"
    assert media_endpoint.analyze(
        RequestContext("boot-warmup-analyze", boot_warmup=True), decoded.payload
    ).size_bytes == written.stat().st_size

    video_spec = getattr(media_endpoint.extract_frame, ENTRYPOINT_ATTR)
    nothing, why = neutral_payload(video_spec.payload_type, str(tmp_path))
    assert nothing is None and "video" in why
