"""pgw#669: a SPARSE legal shape set, and the second-scheduler bug class.

``CompileAxis`` partitions payload fields INDEPENDENTLY, so the derived warm
plan is the cross-product of those partitions. For an endpoint with real
inter-field constraints (ltx-video-2.3: 4K is 16:9 and B200-only, ultrawide is
1080p-class and frame-capped, 48 fps caps at 10 s -> **29 legal of 120**) most
of that product is not a servable request, and every illegal row used to fail
the WORKER LOAD while the plan built its payloads. Declaring axes at all was
therefore a boot hazard, which is why ltx kept a hand-written ``warmup()``
carve-out and forfeited the computable "fully warmed?" property.

Tapes here use a payload whose ``__post_init__`` really does reject the
illegal combinations (the ltx shape, scaled down to the same 120-vs-sparse
arithmetic), through the REAL ``warmup.plan`` derivation:

  * an ``IllegalCombination`` combination is DROPPED from the plan, never run;
  * the drop is a counted, explicit coverage claim — not a silent degradation;
  * every OTHER exception during payload construction still propagates, so a
    genuine synthesis bug is not laundered into "sparse legal set";
  * axes admitting NO legal combination are a declaration bug and raise;
  * a legal-set endpoint keeps the derived plan (no ``warmup()`` carve-out
    needed) and its graph keys still dedupe/attribute normally;
  * the typed error is a wire ``ValidationError`` when a REQUEST carries such
    a combination — the sparse-set support changes nothing about the wire.

Plus the sibling finding, in ``view.for_request``: a pipeline with a SECOND
stateful sampler (ltx's ``audio_scheduler``, which diffusers' own ``__call__``
drives with ``retrieve_timesteps`` + a ``.step()`` per denoise step) kept
SHARING it across concurrent requests. Real diffusers schedulers are used for
those rows.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

import msgspec
import pytest
from diffusers import (
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
)

from gen_worker import (
    AxisClass,
    CompileAxis,
    IllegalCombination,
    RequestContext,
    Resources,
    endpoint,
)
from gen_worker import warmup as warmup_mod
from gen_worker.families import GenerationDefaults
from gen_worker.registry import extract_specs
from gen_worker.view import discover_schedulers, for_request


class _Defaults(GenerationDefaults, frozen=True):
    steps: int = 30

# ---------------------------------------------------------------------------
# the ltx shape: independent axes, sparse legal set
# ---------------------------------------------------------------------------

_RES_AXIS = CompileAxis(classes="enum")
_AR_AXIS = CompileAxis(classes="enum")
_FPS_AXIS = CompileAxis(classes=(
    AxisClass("fps24", match=lambda v: v == 24, warm=24),
    AxisClass("fps48", match=lambda v: v == 48, warm=48),
))


class SparseIn(msgspec.Struct):
    """Three axes -> 2 x 3 x 2 = 12 combinations, of which 5 are legal.

    The constraints are ltx's in miniature: 4K is 16:9 only, and 48 fps is
    720p only. Each field is individually legal in every listed value; only
    the COMBINATION is refused, which is exactly what an independent-partition
    axis model cannot express.
    """

    prompt: str = "a cat"
    resolution_p: Annotated[Literal[720, 2160], _RES_AXIS] = 720
    aspect_ratio: Annotated[Literal["16:9", "1:1", "9:16"], _AR_AXIS] = "16:9"
    fps: Annotated[Literal[24, 48], _FPS_AXIS] = 24
    num_inference_steps: int = 30

    def __post_init__(self) -> None:
        if self.resolution_p == 2160 and self.aspect_ratio != "16:9":
            raise IllegalCombination(
                f"4K is 16:9 only (got {self.aspect_ratio})")
        if self.fps == 48 and self.resolution_p != 720:
            raise IllegalCombination(
                f"48 fps caps at 720p (got {self.resolution_p}p)")


class Out(msgspec.Struct):
    ok: bool = True


@endpoint(resources=Resources(gpu=True))
class SparseEndpoint:
    def setup(self) -> None:  # pragma: no cover - never run in these tapes
        self.ready = True

    def generate(self, ctx: RequestContext[_Defaults], p: SparseIn) -> Out:
        return Out()


def _plan(cls: type) -> tuple[list, list]:
    from gen_worker.api.decorators import ATTR as DECL_ATTR

    specs = extract_specs(cls)
    decl = getattr(cls, DECL_ATTR)
    return warmup_mod.plan(specs, decl_warmup=decl.warmup, has_warmup_method=False)


def _legal_payloads(jobs: list, tmp: str) -> list:
    return [job.build(tmp) for job in jobs]


def test_illegal_combinations_are_dropped_not_run(tmp_path: Any) -> None:
    jobs, skips = _plan(SparseEndpoint)
    # 12 combinations in the product; 5 legal (720p x 3 ARs x 24fps = 3,
    # 720p x 3 ARs x 48fps = 3 -> 6, minus none; 2160p x 16:9 x 24fps = 1;
    # 2160p rows at 48fps and non-16:9 ARs are illegal).
    payloads = _legal_payloads(jobs, str(tmp_path))
    assert len(payloads) == 7, [
        (p.resolution_p, p.aspect_ratio, p.fps) for p in payloads]
    for p in payloads:
        # Every planned payload really is constructible/servable.
        msgspec.structs.replace(p)
        assert not (p.resolution_p == 2160 and p.aspect_ratio != "16:9")
        assert not (p.fps == 48 and p.resolution_p != 720)


def test_the_drop_is_a_counted_coverage_claim() -> None:
    _jobs, skips = _plan(SparseEndpoint)
    rows = [s for s in skips if s.illegal]
    assert len(rows) == 1, "one coverage row per handler with a sparse set"
    row = rows[0]
    assert row.combos == 12 and row.illegal == 5
    assert "IllegalCombination" in row.reason
    assert "7 legal graph classes planned" in row.reason


class BrokenIn(msgspec.Struct):
    """Not a sparse legal set — a BUG in the endpoint's own validation."""

    prompt: str = "x"
    fps: Annotated[Literal[24, 48], _FPS_AXIS] = 24

    def __post_init__(self) -> None:
        if self.fps == 48:
            raise RuntimeError("bug in my own validation")


@endpoint(resources=Resources(gpu=True))
class BrokenEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: BrokenIn) -> Out:
        return Out()


class ImpossibleIn(msgspec.Struct):
    """Axes that admit no servable request at all."""

    prompt: str = "x"
    fps: Annotated[Literal[24, 48], _FPS_AXIS] = 24

    def __post_init__(self) -> None:
        raise IllegalCombination("nothing is legal here")


class NoLegalRowIn(msgspec.Struct):
    """Neutral defaults are servable, but every AXIS row is refused."""

    prompt: str = "x"
    fps: Annotated[Literal[24, 48], _FPS_AXIS] = 0  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.fps in (24, 48):
            raise IllegalCombination(f"fps={self.fps} is never servable")


@endpoint(resources=Resources(gpu=True))
class ImpossibleEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: ImpossibleIn) -> Out:
        return Out()


@endpoint(resources=Resources(gpu=True))
class NoLegalRowEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: NoLegalRowIn) -> Out:
        return Out()


def test_a_real_synthesis_bug_still_fails_the_load() -> None:
    """The whole point of the typed error: a sparse legal set and a broken
    payload are no longer the same event. The walk (discovery / CI, and the
    worker's own spec construction) is where it fails — loudly, before any
    request, exactly as an unwarmable class does."""
    with pytest.raises(RuntimeError, match="bug in my own validation"):
        extract_specs(BrokenEndpoint)


def test_axes_with_no_legal_combination_are_a_declaration_bug() -> None:
    with pytest.raises(ValueError, match="admit no servable request"):
        extract_specs(NoLegalRowEndpoint)


def test_illegal_DEFAULTS_name_themselves_rather_than_leaking_a_field_message(
) -> None:
    # The neutral schema defaults ARE the base payload; if they are refused
    # there is nothing to derive from, and the diagnostic must say that.
    with pytest.raises(ValueError, match="own DEFAULT values are declared"):
        extract_specs(ImpossibleEndpoint)


def test_a_sparse_endpoint_needs_no_warmup_carveout() -> None:
    # The property the carve-out forfeited: the plan is DERIVED, so "fully
    # warmed?" stays computable — every planned job carries its graph key and
    # its coverage attribution.
    jobs, _skips = _plan(SparseEndpoint)
    assert jobs, "declaring axes must not make the class unwarmable"
    assert not callable(getattr(SparseEndpoint, "warmup", None))
    assert all(job.graph_key for job in jobs)
    assert {n for job in jobs for n in job.covers} == {
        s.name for s in extract_specs(SparseEndpoint)}
    assert len({job.graph_key for job in jobs}) == len(jobs), "keys are distinct"


def test_the_wire_contract_is_unchanged() -> None:
    # A request carrying an illegal combination is still bad INPUT: msgspec
    # converts a ValueError from __post_init__ into its own ValidationError,
    # and IllegalCombination subclasses both so nothing about the 4xx changes.
    ok = msgspec.msgpack.decode(
        msgspec.msgpack.encode({"resolution_p": 2160, "aspect_ratio": "16:9"}),
        type=SparseIn)
    assert ok.resolution_p == 2160
    with pytest.raises(msgspec.ValidationError, match="4K is 16:9 only"):
        msgspec.msgpack.decode(
            msgspec.msgpack.encode({"resolution_p": 2160, "aspect_ratio": "1:1"}),
            type=SparseIn)
    # ...and worker-side it classifies as a validation refusal, never infra.
    from gen_worker.api.errors import ValidationError, WorkerError

    assert issubclass(IllegalCombination, ValidationError)
    assert issubclass(IllegalCombination, WorkerError)
    assert issubclass(IllegalCombination, ValueError)


# ---------------------------------------------------------------------------
# sibling finding: for_request owns the per-request-state enumeration
# ---------------------------------------------------------------------------


class _TwoSamplerPipeline:
    """The ltx shape: a primary video scheduler plus a SECOND stateful sampler
    for the audio half of a joint latent, both real diffusers schedulers."""

    def __init__(self, *, with_audio: bool = True) -> None:
        self.transformer = object()
        self.vae = object()
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.audio_scheduler = (
            FlowMatchEulerDiscreteScheduler() if with_audio else None)
        self._internal_dict = {"scheduler": ("diffusers", "X")}


def test_every_sampler_is_discovered_and_cloned() -> None:
    pipe = _TwoSamplerPipeline()
    assert discover_schedulers(pipe) == ("scheduler", "audio_scheduler")
    view = for_request(pipe)
    assert view.scheduler is not pipe.scheduler
    assert view.audio_scheduler is not pipe.audio_scheduler, (
        "the SECOND sampler was the half-fixed pipeline that read as fully "
        "fixed — two concurrent requests shared its step_index")
    # Weight modules stay shared by reference (never cloned).
    assert view.transformer is pipe.transformer
    assert view.vae is pipe.vae


def test_a_checkpoint_without_the_second_sampler_is_unaffected() -> None:
    pipe = _TwoSamplerPipeline(with_audio=False)
    assert discover_schedulers(pipe) == ("scheduler",)
    view = for_request(pipe)
    assert view.scheduler is not pipe.scheduler
    assert view.audio_scheduler is None


def test_secondary_samplers_are_cloned_faithfully_not_re_decided() -> None:
    # sampler/objective/scheduler_config are the DENOISER's decision: forcing
    # a video sampler class or a flow `shift` onto a second modality's sampler
    # would re-decide its math instead of making it private.
    pipe = _TwoSamplerPipeline()
    view = for_request(
        pipe, sampler="euler", scheduler_config={"shift": 3.0})
    assert isinstance(view.scheduler, EulerDiscreteScheduler)
    assert type(view.audio_scheduler) is type(pipe.audio_scheduler)
    assert view.audio_scheduler.config.get("shift") == \
        pipe.audio_scheduler.config.get("shift")


def test_an_explicit_scheduler_set_pins_the_enumeration() -> None:
    pipe = _TwoSamplerPipeline()
    view = for_request(pipe, schedulers=("scheduler",))
    assert view.scheduler is not pipe.scheduler
    assert view.audio_scheduler is pipe.audio_scheduler, \
        "an explicit set is honored verbatim"
    # A named attribute the checkpoint does not carry is simply absent.
    view2 = for_request(pipe, schedulers=("scheduler", "nope_scheduler"))
    assert not hasattr(view2, "nope_scheduler")


def test_weight_modules_are_never_mistaken_for_samplers() -> None:
    class _Weighty:
        def __init__(self) -> None:
            self.scheduler = FlowMatchEulerDiscreteScheduler()
            # A module with from_config but no scheduler-ish name/shape.
            self.transformer = EulerDiscreteScheduler
            self.text_encoder = object()

    assert discover_schedulers(_Weighty()) == ("scheduler",)
