"""gw#470 default-on boot warmup: schema synthesis, the `warmup=` declarative
fallback, NoWarmup opt-out, walk-time enforcement, and the executor running
the synthesized job post-setup / pre-READY on the load-failure + drain rails."""

from __future__ import annotations

import asyncio
import enum
import tempfile
import threading
from typing import Iterator, Optional

import msgspec
import pytest

from gen_worker import Hub, NoWarmup, Resources, Slot, endpoint
from gen_worker import warmup as warmup_mod
from gen_worker.api.types import AudioAsset, ImageAsset, VideoAsset
from gen_worker.executor import Executor, _MaterializedLocal
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs


class _Out(msgspec.Struct):
    y: str = ""


# ---------------------------------------------------------------------------
# schema synthesis
# ---------------------------------------------------------------------------


class _Size(enum.Enum):
    SMALL = "small"
    LARGE = "large"


def _build(payload_type: type):
    factory, reason = warmup_mod.synthesize_factory(payload_type)
    assert factory is not None, reason
    with tempfile.TemporaryDirectory() as tmp:
        return factory(tmp)


def test_synthesis_field_fills():
    class In(msgspec.Struct):
        prompt: str                   # required str -> warmup text
        size: _Size                   # enum -> first member
        image: Optional[ImageAsset]   # required-but-optional asset -> None
        steps: int = 4                # default preserved

    p = _build(In)
    assert p.prompt == warmup_mod.WARMUP_TEXT and p.steps == 4
    assert p.size is _Size.SMALL
    assert p.image is None


def test_synthesis_media_assets_have_readable_files():
    class In(msgspec.Struct):
        image: ImageAsset
        audio: AudioAsset

    factory, reason = warmup_mod.synthesize_factory(In)
    assert factory is not None, reason
    with tempfile.TemporaryDirectory() as tmp:
        p = factory(tmp)
        with open(p.image, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"
        with open(p.audio, "rb") as f:
            assert f.read(4) == b"RIFF"


class _NestedItem(msgspec.Struct):
    image: ImageAsset
    id: str = ""


class _NestedIn(msgspec.Struct):
    items: list[_NestedItem]


def test_synthesis_nested_list_of_structs():
    p = _build(_NestedIn)
    assert len(p.items) == 1 and p.items[0].image.local_path


def test_synthesis_video_input_blocks():
    class In(msgspec.Struct):
        video: VideoAsset
        prompt: str = ""

    factory, reason = warmup_mod.synthesize_factory(In)
    assert factory is None and "video" in reason


# ---------------------------------------------------------------------------
# decorator surface + walk-time enforcement
# ---------------------------------------------------------------------------


class _VideoIn(msgspec.Struct):
    video: VideoAsset


class _PromptIn(msgspec.Struct):
    prompt: str
    steps: int = 2


class _SlotPromptIn(msgspec.Struct):
    prompt: str
    steps: int = 2
    model: str = ""


def _nowarmup_blank_reason():
    NoWarmup("  ")


def _warmup_on_function_endpoint():
    @endpoint(warmup={"x": None})
    def fn(ctx, payload: _PromptIn) -> _Out:  # pragma: no cover
        return _Out()


def _gpu_class_unwarmable_payload():
    @endpoint(resources=Resources(vram_gb=8))
    class Ep:
        def generate(self, ctx, payload: _VideoIn) -> _Out:  # pragma: no cover
            return _Out()


def _declared_unknown_method():
    @endpoint(resources=Resources(vram_gb=8), warmup={"nope": {"prompt": "x"}})
    class Ep:
        def generate(self, ctx, payload: _PromptIn) -> _Out:  # pragma: no cover
            return _Out()


def _declared_invalid_payload():
    @endpoint(resources=Resources(vram_gb=8), warmup={"generate": {"steps": "NaN-ish"}})
    class Ep:
        def generate(self, ctx, payload: _PromptIn) -> _Out:  # pragma: no cover
            return _Out()


@pytest.mark.parametrize(
    ("define", "exc", "match"),
    [
        pytest.param(_nowarmup_blank_reason, ValueError, "", id="nowarmup-blank-reason"),
        pytest.param(_warmup_on_function_endpoint, ValueError, "requires a class",
                     id="warmup-on-function"),
        pytest.param(_gpu_class_unwarmable_payload, TypeError, "default-on",
                     id="gpu-unwarmable-payload"),
        pytest.param(_declared_unknown_method, TypeError, "unknown",
                     id="declared-unknown-method"),
        pytest.param(_declared_invalid_payload, TypeError, "not a valid",
                     id="declared-invalid-payload"),
    ],
)
def test_invalid_warmup_declarations_fail_at_decoration(define, exc, match):
    with pytest.raises(exc, match=match):
        define()


def test_gpu_class_opts_out_with_reason():
    @endpoint(resources=Resources(vram_gb=8), warmup=NoWarmup("needs user video"))
    class Ep:
        def generate(self, ctx, payload: _VideoIn) -> _Out:  # pragma: no cover
            return _Out()

    jobs, skips = warmup_mod.plan_for_class(Ep)
    assert not jobs and skips[0].reason == "NoWarmup: needs user video"


def test_gpu_class_with_custom_warmup_method_passes():
    @endpoint(resources=Resources(vram_gb=8))
    class Ep:
        def warmup(self) -> None:  # pragma: no cover
            pass

        def generate(self, ctx, payload: _VideoIn) -> _Out:  # pragma: no cover
            return _Out()

    jobs, skips = warmup_mod.plan_for_class(Ep)
    assert not jobs and not skips  # method path: executor calls warmup()


def test_declared_payload_overrides_and_validates():
    @endpoint(
        resources=Resources(vram_gb=8),
        warmup={"generate": {"prompt": "big", "steps": 1}},
    )
    class Ep:
        def generate(self, ctx, payload: _PromptIn) -> _Out:  # pragma: no cover
            return _Out()

        def other(self, ctx, payload: _PromptIn) -> _Out:  # pragma: no cover
            return _Out()

    jobs, skips = warmup_mod.plan_for_class(Ep)
    assert not skips and len(jobs) == 2
    by_attr = {j.spec.attr_name: j for j in jobs}
    assert by_attr["generate"].declared and not by_attr["other"].declared
    with tempfile.TemporaryDirectory() as tmp:
        assert by_attr["generate"].build(tmp).prompt == "big"
        assert by_attr["other"].build(tmp).prompt == warmup_mod.WARMUP_TEXT


def test_declared_none_skips_method():
    @endpoint(resources=Resources(vram_gb=8), warmup={"generate": None})
    class Ep:
        def generate(self, ctx, payload: _PromptIn) -> _Out:  # pragma: no cover
            return _Out()

    jobs, skips = warmup_mod.plan_for_class(Ep)
    assert not jobs and "declared skip" in skips[0].reason


def test_cpu_class_needs_no_warmup():
    @endpoint(resources=Resources(vcpus=2))
    class Ep:
        def crunch(self, ctx, payload: _VideoIn) -> _Out:  # pragma: no cover
            return _Out()

    jobs, skips = warmup_mod.plan_for_class(Ep)
    assert not jobs and not skips


# ---------------------------------------------------------------------------
# executor: synthesized warmup runs post-setup, pre-READY
# ---------------------------------------------------------------------------


async def _noop_send(msg) -> None:
    pass


def _ensure_setup(cls):
    specs = extract_specs(cls)
    ex = Executor(specs, _noop_send)
    asyncio.run(ex.ensure_setup(specs[0]))
    return ex, specs


def test_executor_runs_synthesized_warmup_pre_ready():
    calls: list[tuple[bool, str, bool]] = []

    @endpoint(resources=Resources(vram_gb=8))
    class Ep:
        ready_at_call: list[bool] = []

        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _PromptIn) -> _Out:
            calls.append((ctx.boot_warmup, payload.prompt, ctx.cancelled))
            return _Out(y="ok")

        def generate_turbo(self, ctx, payload: _PromptIn) -> _Out:
            calls.append((ctx.boot_warmup, payload.prompt, ctx.cancelled))
            return _Out(y="ok")

    ex, specs = _ensure_setup(Ep)
    # Both handlers of the instance group warmed exactly once, as warmup.
    assert calls == [(True, "warmup", False), (True, "warmup", False)]
    assert ex._classes[specs[0].instance_key].ready


def test_executor_warmup_handles_dynamic_slot_split(tmp_path) -> None:
    """A request-time Slot pick can split one method into a new instance key.

    ``warmup=`` is still a class-wide declaration: a skip for the authored
    Turbo sibling remains valid even when this picked instance contains only
    ordinary generate. This is the SDXL release-06b6 production failure.
    """
    calls: list[str] = []

    class _Pipeline:
        pass

    @endpoint(
        models={
            "pipeline": Slot(
                _Pipeline,
                selected_by="model",
                default_checkpoint=Hub("acme/default", tag="prod"),
            ),
        },
        resources=Resources(vram_gb=8),
        warmup={
            "generate": {"prompt": "warmup", "steps": 1},
            "generate_turbo": None,
        },
    )
    class Ep:
        def setup(self, pipeline: str) -> None:
            self.pipeline = pipeline

        def generate(self, ctx, payload: _SlotPromptIn) -> _Out:
            calls.append(payload.prompt)
            return _Out()

        def generate_turbo(self, ctx, payload: _SlotPromptIn) -> _Out:
            calls.append("turbo")
            return _Out()

    specs = extract_specs(Ep)
    ex = Executor(specs, _noop_send)
    generate = next(spec for spec in specs if spec.attr_name == "generate")
    picked = ex._effective_spec(
        generate,
        pb.RunJob(
            models=[
                pb.ModelBinding(slot="pipeline", ref="acme/babes-pony:prod"),
            ],
        ),
    )
    assert picked.instance_key != generate.instance_key

    async def _materialize_local(ref, snapshot=None, *, binding=None):
        return _MaterializedLocal(path=tmp_path, identity=("", 0))

    ex.store._materialize_local = _materialize_local  # type: ignore[method-assign]
    asyncio.run(ex.ensure_setup(picked))

    assert calls == ["warmup"]
    assert ex._classes[picked.instance_key].ready

    # The opposite split is equally important: the Turbo-only group keeps
    # its explicit skip and must not synthesize a Turbo request merely because
    # ordinary generate lives in another instance group.
    turbo = next(spec for spec in specs if spec.attr_name == "generate_turbo")
    turbo_picked = ex._effective_spec(
        turbo,
        pb.RunJob(
            models=[
                pb.ModelBinding(slot="pipeline", ref="acme/other-pony:prod"),
            ],
        ),
    )
    asyncio.run(ex.ensure_setup(turbo_picked))

    assert calls == ["warmup"]
    assert ex._classes[turbo_picked.instance_key].ready


def test_executor_custom_warmup_method_suppresses_synthesis():
    state = {"warmups": 0, "handler": 0}

    @endpoint(resources=Resources(vram_gb=8))
    class Ep:
        def setup(self) -> None:
            pass

        def warmup(self) -> None:
            state["warmups"] += 1

        def generate(self, ctx, payload: _PromptIn) -> _Out:
            state["handler"] += 1
            return _Out()

    _ensure_setup(Ep)
    assert state == {"warmups": 1, "handler": 0}


def test_executor_nowarmup_skips():
    state = {"handler": 0}

    @endpoint(resources=Resources(vram_gb=8), warmup=NoWarmup("engine self-warms"))
    class Ep:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _PromptIn) -> _Out:
            state["handler"] += 1
            return _Out()

    _ensure_setup(Ep)
    assert state["handler"] == 0


def test_executor_warmup_failure_is_load_failure():
    @endpoint(resources=Resources(vram_gb=8))
    class Ep:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _PromptIn) -> _Out:
            raise RuntimeError("cuda exploded")

    specs = extract_specs(Ep)
    ex = Executor(specs, _noop_send)
    with pytest.raises(RuntimeError, match="cuda exploded"):
        asyncio.run(ex.ensure_setup(specs[0]))
    rec = ex._classes[specs[0].instance_key]
    assert not rec.ready and rec.failed is not None


def test_executor_warmup_oom_defers_to_fit_ladder():
    """A warmup CUDA OOM must NOT take the function down — the gw#521
    runtime fit ladder still serves it degraded on the first real request."""

    @endpoint(resources=Resources(vram_gb=8))
    class Ep:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _PromptIn) -> _Out:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")

    ex, specs = _ensure_setup(Ep)
    rec = ex._classes[specs[0].instance_key]
    assert rec.ready and rec.failed is None


def test_executor_undecorated_spec_skips_synthesized_warmup():
    """Internally-constructed EndpointSpecs (no @endpoint decl) have no
    declaration surface — the synthesized warmup must not fire."""
    from gen_worker.registry import EndpointSpec

    calls = {"n": 0}

    class Ep:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _PromptIn) -> _Out:
            calls["n"] += 1
            return _Out()

    spec = EndpointSpec(
        name="ep", method=Ep.generate, kind="inference", payload_type=_PromptIn,
        output_mode="single", cls=Ep, attr_name="generate",
        resources=Resources(vram_gb=8),
    )
    ex = Executor([spec], _noop_send)
    asyncio.run(ex.ensure_setup(spec))
    assert calls["n"] == 0 and ex._classes[spec.instance_key].ready


def test_executor_stream_handler_warmup_consumed():
    state = {"yields": 0}

    @endpoint(resources=Resources(vram_gb=8))
    class Ep:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _PromptIn) -> Iterator[_Out]:
            for _ in range(3):
                state["yields"] += 1
                yield _Out()

    _ensure_setup(Ep)
    assert state["yields"] == 3


def test_executor_warmup_output_stays_local():
    seen: dict[str, str] = {}

    @endpoint(resources=Resources(vram_gb=8))
    class Ep:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _PromptIn) -> _Out:
            import os
            src = os.path.join(tempfile.gettempdir(), "gw-warmup-src.bin")
            with open(src, "wb") as f:
                f.write(b"x")
            asset = ctx.save_file("out.bin", src)
            seen["path"] = asset.local_path or ""
            return _Out()

    _ensure_setup(Ep)
    assert "gw-warmup-" in seen["path"]  # discarded tempdir, never an upload


def test_executor_drain_during_warmup_cancels_cleanly():
    entered = threading.Event()
    release = threading.Event()

    @endpoint(resources=Resources(vram_gb=8))
    class Ep:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _PromptIn) -> _Out:
            entered.set()
            release.wait(timeout=10)
            return _Out()

    specs = extract_specs(Ep)
    ex = Executor(specs, _noop_send)

    async def scenario() -> None:
        task = asyncio.create_task(ex.ensure_setup(specs[0]))
        await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert not ex._classes[specs[0].instance_key].ready
