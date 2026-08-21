"""THE BOOT WARM PASS HAS A WRITER AGAIN (pgw#1584).

`ctx.boot_warmup` shipped a constructor kwarg, a public `@property`, a
docstring prescribing `steps = 1 if ctx.boot_warmup else steps`, and a READER
in `output_integrity.judged` — and, between the v2 hardcut and this issue, no
producer anywhere in `src/`. `grep -rn "boot_warmup=True" src/` was empty, so
every `if ctx.boot_warmup:` arm in the fleet was unreachable code and the
first-call tax was paid by a PAYING request. Ruled an accidental drop (v1 had a
functioning warm pass and no `v1_deleted.py` tombstone row was ever written for
it), so the wire is restored rather than the feature retired.

Integration, on the real path: the `serving_v2_endpoint` fixture — the
main_v2-contract-exact endpoint that serves real requests on CPU with fake
weights from config-only checkpoints — is booted through `ServeLoop` and warmed
through `ServeLoop.boot_warmup`, which calls the SAME `invoke` a dispatch calls.
No mock of the serve path; the only doubles are the ones the design names
(`BindingResolver`, the static sizer).

The two halves the issue insisted on are BOTH here, because they fail
independently:

* the flag is TRUE on a real boot path — asserted at
  `output_integrity.judged`, i.e. at the reader itself, so it proves the warm
  pass's context is the same object the reader sees rather than a second one;
* a blank warm output PASSES the integrity floor under it, and is REFUSED
  without it. With no writer, that exemption had never engaged, so the restored
  warm pass would otherwise have failed on its own discarded output.
"""

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
    """Record `ctx.boot_warmup` AT THE READER, for every judged output.

    `request_context` imports `judged` into its own namespace, so this patches
    the name the save path actually calls. The recorder DELEGATES — the floor
    keeps its real verdict — it only writes down which context object arrived.
    """
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


# ---------------------------------------------------------------------------
# the writer exists, and it drives the real serve path
# ---------------------------------------------------------------------------


def test_the_warm_pass_runs_a_REAL_forward_through_the_REAL_serve_path(
    tmp_path: Path,
) -> None:
    """One synthetic invocation per entrypoint, through `invoke`.

    Not a second serving path — the same one, called earlier. The proof it is
    the same one is that the resolver was asked to resolve a checkpoint (so an
    admission and a real `Model.load` happened) and the handler's output landed
    on disk.
    """
    loop, resolver = make_loop(tmp_path)

    passes = loop.boot_warmup()

    assert [row.function for row in passes] == ["generate"]
    (row,) = passes
    assert row.outcome == WARM_OK, row
    assert row.warmed and row.reason == ""
    # A real lease, a real load: the resolver was asked for the deploy's bytes
    # exactly as a dispatch asks (the primary-binding lookup, then the lease).
    assert set(resolver.resolved) == {DREAM} and len(resolver.resolved) == 2
    # And a real forward: the residency ledger holds the instance the warm pass
    # loaded, so the FIRST REAL REQUEST reuses it instead of paying for it.
    assert loop.residency.tier_of(DREAM, f"SdxlModel/{LANE}") is not None


def test_the_flag_is_TRUE_on_a_real_boot_path_and_FALSE_on_a_real_request(
    tmp_path: Path, judged_calls: List[bool]
) -> None:
    """HALF ONE of the issue's must-know, asserted AT THE READER.

    "The absence of exactly this test is why the drop survived", so it is
    asserted where it matters: `output_integrity.judged` is handed the context
    the author's `ctx.save_image` was called on, so recording the flag there
    proves the warm pass's `boot_warmup=True` object is the SAME object the
    reader sees — not a second context that merely also exists.
    """
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
    """HALF TWO, and the reason the restoration needed it.

    `output_integrity.py:475` makes `boot_warmup=True` the blank-render
    exemption, and with no writer it had NEVER engaged. The warm pass's whole
    input is degenerate by construction (`WARMUP_TEXT` and, where a schema
    needs one, a flat mid-gray PNG), so its output is exactly the degenerate
    render the floor refuses — which would have made the restored warm pass
    fail on its own discarded output.

    Same image, same call, one flag apart.
    """
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
    """The reader's own verdict, stated as an assertion rather than inferred
    from a save that happened to succeed."""
    warm: RequestContext[Any] = RequestContext("r", boot_warmup=True)
    paying: RequestContext[Any] = RequestContext("r")
    assert output_integrity.judged(warm) is False
    assert output_integrity.judged(paying) is True


# ---------------------------------------------------------------------------
# the payload: the schema's NEUTRAL DEFAULTS, v1's warm-plan shape
# ---------------------------------------------------------------------------


def test_the_warm_payload_is_the_schema_at_its_NEUTRAL_DEFAULTS(
    tmp_path: Path,
) -> None:
    """The int32 incident's own record of the v1 warm plan: *"a single run at
    the schema's neutral defaults"*. Not the largest preset, not an author
    declaration (`NoWarmup` is tombstoned: "warmup is not an author
    declaration"), and not a value invented for a field the schema already
    answers for.
    """
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
    # The ONE required field is filled minimally...
    assert payload.prompt == WARMUP_TEXT
    # ...and every defaulted field keeps the value the schema declares. This is
    # what "at the defaults" means, and it is one branch in the synthesizer:
    # a non-required field is simply left out of the constructor call.
    assert payload.aspect_ratio is AspectRatio.RATIO_1_1
    assert payload.num_inference_steps is None
    assert payload.guidance_scale is None
    assert payload.scheduler is None
    assert payload.enhance_prompt is True
    assert payload.output_format == "webp"


def test_a_payload_that_cannot_synthesize_is_SKIPPED_with_a_reason(
    tmp_path: Path,
) -> None:
    """Stated, never faked. A required `VideoAsset` has no honest 2 KB
    stand-in, and inventing one warms a code path with bytes no request will
    ever carry — so the warm pass declines and says which field stopped it."""
    import msgspec

    from gen_worker.api.types import AudioAsset, ImageAsset, VideoAsset

    # Module-scope names: msgspec resolves a struct's annotations by NAME, and
    # a class defined in a function body cannot see this function's locals.
    globals().update(
        AudioAsset=AudioAsset, ImageAsset=ImageAsset, VideoAsset=VideoAsset
    )

    class NeedsVideo(msgspec.Struct):
        clip: VideoAsset

    payload, reason = neutral_payload(NeedsVideo, str(tmp_path))
    assert payload is None
    assert "clip" in reason and "video" in reason

    # The media kinds that CAN be synthesized are, and the file really exists —
    # the concrete subclasses are matched before the ambiguous `Asset` base, or
    # both would answer "not synthesizable".
    class NeedsMedia(msgspec.Struct):
        image: ImageAsset
        audio: AudioAsset

    media, reason = neutral_payload(NeedsMedia, str(tmp_path))
    assert reason == ""
    assert Path(media.image.local_path).is_file()
    assert Path(media.audio.local_path).is_file()
    assert media.image.mime_type == "image/png"


def test_a_weightless_entrypoint_is_SKIPPED_not_warmed(tmp_path: Path) -> None:
    """pgw#1392: zero model slots means no lease, no admission, no load and no
    activation peak — there is no first-call tax to move, so warming it would
    spend a boot second to save nothing. "Nobody warmed" and "warming was free"
    stay different answers."""
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


# ---------------------------------------------------------------------------
# failure posture: LOUD, and never fatal
# ---------------------------------------------------------------------------


def test_a_FAILING_warm_pass_degrades_loudly_and_does_NOT_brick_the_pod(
    tmp_path: Path,
) -> None:
    """The ruling's posture, and the one place v1's differs.

    v1 propagated a warm failure as a LOAD failure. That turns a defect in an
    OPTIMIZATION into an unservable pod, which is strictly worse than the tax
    the optimization exists to remove. So: a `serve_degrade` event (pgw#760 —
    a hub-spawned worker exposes no stdout, so a log line is invisible), and
    the boot continues.
    """
    loop, _ = make_loop(tmp_path)
    spec = loop.loaded.entrypoints["generate"]

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("the warm forward exploded")

    loop.loaded.entrypoints["generate"] = dataclasses.replace(spec, fn=boom)
    events: List[pb.ActivityUpdate] = []
    activity_mod._sink = events.append  # the bound-sink contract

    passes = loop.boot_warmup()  # must NOT raise

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

    # AND THE POD STILL SERVES. Restore the real body and take a real request:
    # the first one pays the cold cost, which is exactly the state the fleet
    # was in before this issue — never worse.
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
    """`prepare` is the worker's seam for the deploy's picks, and declining is
    a first-class answer.

    Boot carries no dispatch, so a release the hub seeded no per-function
    `ModelBinding` for has no pick — and an empty pick table decodes into
    *"model slot has no envelope pick and no deployment default"*, a CORRECT
    refusal wearing the costume of a warm-pass defect. It is reported as the
    skip it is, and emits no `serve_degrade`.
    """
    loop, resolver = make_loop(tmp_path)
    events: List[pb.ActivityUpdate] = []
    activity_mod._sink = events.append

    passes = loop.boot_warmup(prepare=lambda _fn: "no boot-time binding")

    assert [(r.outcome, r.reason) for r in passes] == [
        (WARM_SKIPPED, "no boot-time binding")
    ]
    # A declined warm never loaded anything and never confessed a degrade.
    assert resolver.resolved == []
    assert not [e for e in events if e.kind == activity_mod.KIND_SERVE_DEGRADE]


# ---------------------------------------------------------------------------
# placement: after the weights, BEFORE this pod calls itself servable
# ---------------------------------------------------------------------------


def test_the_warm_pass_runs_BEFORE_the_worker_is_ready_or_servable() -> None:
    """The placement is the deliverable, not an implementation detail.

    Running the warm pass while the state is still `materializing` is what
    turns `first_request_servable` from "the process is up" into "a real
    forward has completed on this pod" — the readiness probe th#2233's
    false-servable fix needs something real to hang on.

    Asserted on the ORDER as `CheckpointMaterialization` executes it: the warm
    callback observes `ready is False`, and the readiness announce (which is
    what stamps the milestone) has not fired yet when it runs.
    """
    observed: Dict[str, Any] = {}

    class _Store:
        def replace_desired_snapshots(self, *_a: Any, **_k: Any) -> None:
            return None

        async def announce_resident(self, *_a: Any, **_k: Any) -> bool:
            return True  # a warm volume: no byte moves

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
    # `_announce_soon` fires once at CONFIGURE, to say "materializing" — the
    # readiness announce that stamps `first_request_servable` is the one at the
    # END of `_materialize`, and it has not run when the warm pass does.
    assert observed["announced_before_warm"] == ["announce"], observed
    assert observed["servable_ms"] is None, observed
    # ...and only afterwards does this worker become routable.
    assert mat.state == STATE_READY
    assert announced == ["announce", "announce"]


def test_a_raising_warm_callback_never_costs_the_pod_its_boot() -> None:
    """The second belt. `ServeLoop.boot_warmup` already returns rather than
    raising; this proves the materialization would survive one that did."""

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
    """`boot_phases`' own rule, printed above the vocabulary: *"a declared
    phase with no producer is not coverage we have not gotten to: every reader
    of the ladder sees a name that can only ever report nothing"*.

    `PHASE_WARMUP` has been declared, and mapped into `boot_stages`' roll-up,
    with NOTHING in `src/` emitting one since the hardcut. The warm pass is its
    producer.
    """
    loop, _ = make_loop(tmp_path)
    loop.boot_warmup()

    table = boot_phases.phase_table()
    warm_rows = [row for row in table if row.phase == boot_phases.PHASE_WARMUP]
    assert warm_rows, [row.phase for row in table]
    assert warm_rows[0].function == "generate", warm_rows[0]


# ---------------------------------------------------------------------------
# pgw#1583: the contract lie in `for_request`, settled toward the CODE
# ---------------------------------------------------------------------------


def test_for_request_no_longer_takes_a_slot_it_silently_ignores() -> None:
    """`slot=` was accepted and discarded on BOTH context implementations —
    the serving one passed `objective=""` regardless, the trace one wrote
    `del slot`. A kwarg that changes nothing is a third way to be silently
    wrong, so it is gone from both and the two signatures still match."""
    import inspect

    from gen_worker.release.trace_context import TraceRequestContext

    serving = inspect.signature(RequestContext.for_request).parameters
    trace = inspect.signature(TraceRequestContext.for_request).parameters
    assert "slot" not in serving
    assert "slot" not in trace
    assert set(serving) == set(trace)


def test_the_for_request_docstring_states_the_MEASURED_behaviour() -> None:
    """The docstring promised the objective was applied here and that ambiguity
    raised; the body passed `objective=""` unconditionally, so the promise was
    contradicted by the code beneath it and the raise was unreachable.

    Settled toward the CODE, because the fact does not exist to apply:
    `ModelBinding.objective` has no reader anywhere in the SDK. So the
    docstring now says so and names the surface that DOES honour an objective.
    """
    doc = RequestContext.for_request.__doc__ or ""
    assert "CARRIES NO OBJECTIVE" in doc
    assert "gen_worker.view" in doc and "objective=" in doc
    # And the old promises are gone, not merely qualified.
    assert "Ambiguity raises" not in doc

    source = Path(request_context_mod.__file__).read_text(encoding="utf-8")
    # The one remaining `objective=` in `for_request` is the literal empty
    # string the docstring now advertises.
    assert 'objective="", generator=gen' in source


# ---------------------------------------------------------------------------
# the boot picks: read off the ack, inferred only where there is one answer
# ---------------------------------------------------------------------------


def _config_for(ref: str, digest: str = "sha256:beef") -> CheckpointConfig:
    desired = pb.DesiredResidency(generation=1, disk_refs=[ref])
    desired.snapshots[ref].CopyFrom(pb.Snapshot(digest=digest))
    return CheckpointConfig.from_wire(desired)


def test_the_boot_picks_come_off_the_hubs_OWN_DesiredResidency_Hot() -> None:
    """Nothing is invented: `DesiredResidency.Hot` is
    `repeated DesiredInstance {function_name, models}` and `models` is the very
    `ModelBinding` a dispatch carries — slot, ref, the recognized `model` name
    and its `inference_defaults` row. The warm pass resolves through the same
    table a real request resolves through."""
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
    """The hub seeds `Hot` for dynamic-slot defaults and compile-cache prewarm;
    a static-binding release can arrive with it empty. With ONE model slot and
    ONE materialized ref there is exactly one possible answer, so the two are
    bound. Two refs is a guess, and `decode_envelope`'s rule stands: the worker
    never guesses which bytes to serve — no entry, and the warm pass skips."""
    from gen_worker.worker import boot_picks

    loaded = load_endpoint(FIXTURE_DIR)

    single = _config_for(DREAM)
    picks = boot_picks(pb.DesiredResidency(), loaded, single)
    assert picks["generate"].by_slot == {"model": DREAM}
    # The digest travels so `tree_for` resolves exactly as a dispatch does.
    assert picks["generate"].by_ref[DREAM].manifest_digest == "sha256:beef"
    # Unknown classification stays unknown — `ctx.defaults()`'s unclassified arm
    # is pgw#1377's warn-and-serve fallback, and inventing a name would warm
    # under a recipe the hub never resolved.
    assert picks["generate"].by_ref[DREAM].model == ""

    two = pb.DesiredResidency(generation=1, disk_refs=[DREAM, "org/other@1"])
    assert boot_picks(two, loaded, CheckpointConfig.from_wire(two)) == {}


def test_a_synthesized_ASSET_survives_the_envelope_the_warm_pass_builds(
    tmp_path: Path,
) -> None:
    """The warm payload becomes an envelope (`msgspec.to_builtins`) and is
    decoded back by the REAL `decode_envelope` — so `local_path` has to survive
    the round trip, or an asset-taking entrypoint warms with an asset it cannot
    open.

    This is the half `materialize_input_assets` would otherwise have destroyed:
    it NULLs every `local_path` on the correct reasoning that a path in RunJob
    input is caller-controlled wire data. A boot warm payload is the one input
    on that path which is not — the platform wrote the file — so `invoke` skips
    that step, and only that step, under `ctx.boot_warmup`.
    """
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
    # ...and the handler really can open it — the raise site a production
    # failure came from (`_local_path`) is satisfied.
    assert media_endpoint.analyze(
        RequestContext("boot-warmup-analyze", boot_warmup=True), decoded.payload
    ).size_bytes == written.stat().st_size

    # The sibling entrypoint's REQUIRED VideoAsset is the honest refusal.
    video_spec = getattr(media_endpoint.extract_frame, ENTRYPOINT_ATTR)
    nothing, why = neutral_payload(video_spec.payload_type, str(tmp_path))
    assert nothing is None and "video" in why
