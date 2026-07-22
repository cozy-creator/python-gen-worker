"""pgw#517: `compile=` is silently inert on self-loading (str-slot)
endpoints. Two layers under test:

- discovery: `compile=Compile(...)` on an endpoint whose setup() model
  slots are ALL str/Path-annotated (self-loading) is a build-time hard
  error, unless the endpoint opts in via `gen_worker.arm_compile(...)`.
- the arming seam: `gen_worker.arm_compile(pipe)`, called from inside a
  self-loading setup(), reaches the SAME cache-artifact-gated policy as the
  automatic worker-loaded path — proven through the real executor
  ensure_setup() codepath, not a mock of it.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import msgspec
import pytest

import gen_worker
import gen_worker.executor as executor_mod
from gen_worker import Compile, RequestContext, Resources, endpoint
from gen_worker import compile_cache as cc
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.api.errors import RetryableError
from gen_worker.executor import Executor, ModelStore
from gen_worker.models import provision
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec, extract_specs

FAMILY = "wan-2.2-t2v-a14b"


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    ok: bool = True


# ---------------------------------------------------------------------------
# discovery: str-slot + compile= is a hard error unless opted in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slot_type", ["str", "Path"])
def test_compile_on_self_loading_slot_is_a_discovery_error(slot_type) -> None:
    if slot_type == "str":

        @endpoint(
            model=Hub("acme/wan"),
            resources=Resources(vram_gb=40),
            compile=Compile(family=FAMILY, shapes=((768, 768),)),
        )
        class SelfLoader:
            def setup(self, model: str) -> None:
                self.model = model  # self-load: never reaches _enable_compiled

            def generate(self, ctx: RequestContext, data: _In) -> _Out:
                return _Out()

    else:

        @endpoint(
            model=Hub("acme/wan"),
            resources=Resources(vram_gb=40),
            compile=Compile(family=FAMILY, shapes=((768, 768),)),
        )
        class SelfLoader:
            def setup(self, model: Path) -> None:
                self.model = model

            def generate(self, ctx: RequestContext, data: _In) -> _Out:
                return _Out()

    with pytest.raises(ValueError, match="self-loading") as exc_info:
        extract_specs(SelfLoader)
    # The error names both fix paths.
    msg = str(exc_info.value)
    assert "annotate the slot with the pipeline class" in msg
    assert "gen_worker.arm_compile(pipe)" in msg


def test_compile_opts_out_via_arm_compile_call_in_setup() -> None:
    """The build-time check is a best-effort source scan for the opt-in
    seam: a `gen_worker.arm_compile(...)` call anywhere in setup() silences
    the error, matching the wan-2.2 fix path (option 2 in the message)."""

    @endpoint(
        model=Hub("acme/wan"),
        resources=Resources(vram_gb=40),
        compile=Compile(family=FAMILY, shapes=((768, 768),)),
    )
    class SelfLoaderArmed:
        def setup(self, model: str) -> None:
            self.model = model
            gen_worker.arm_compile(self.model)

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out()

    (spec,) = extract_specs(SelfLoaderArmed)
    assert spec.compile is not None


def test_compile_on_class_annotated_slot_never_errors() -> None:
    """Worker-loaded (pipeline-class-annotated) slots already arm compile
    automatically — no opt-in needed, no error."""

    class _Pipe:
        @classmethod
        def from_pretrained(cls, path, **kw):
            return cls()

    @endpoint(
        model=Hub("acme/wan"),
        resources=Resources(vram_gb=40),
        compile=Compile(family=FAMILY, shapes=((768, 768),)),
    )
    class WorkerLoaded:
        def setup(self, model: _Pipe) -> None:
            self.model = model

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out()

    (spec,) = extract_specs(WorkerLoaded)
    assert spec.compile is not None


def test_compile_with_no_models_is_not_this_issue() -> None:
    """compile= with no setup() model slots at all is out of scope for
    pgw#517 (nothing here is "self-loading" vs "worker-loaded")."""

    @endpoint(resources=Resources(vram_gb=4), compile=Compile(shapes=((768, 768),)))
    class NoModels:
        def setup(self) -> None:
            pass

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out()

    (spec,) = extract_specs(NoModels)
    assert spec.compile is not None


# ---------------------------------------------------------------------------
# the arming seam: ArmingScope + arm_compile()
# ---------------------------------------------------------------------------


class _StubPipe:
    pass


def test_arm_compile_outside_any_scope_is_a_noop() -> None:
    """ie#522 (Paul's ruling, 2026-07-22): an eager registration (no
    compile=Compile(...) declared) opens ArmingScope(None, ...) — already a
    documented no-op. arm_compile() must agree: every self-loading
    setup() (sdxl, wan-2.2, ...) calls it unconditionally, and each must
    keep working eager, not crash. REVERT-TURNS-RED: this used to raise."""
    assert provision.arm_compile(_StubPipe()) is False


def test_arm_compile_inside_scope_reaches_enable_compiled(monkeypatch, tmp_path) -> None:
    cfg = Compile(family=FAMILY, shapes=((768, 768),))
    seen: list = []
    monkeypatch.setattr(
        provision, "enable_compiled",
        lambda pipe, c, cache_dir, artifact: seen.append((pipe, c, cache_dir, artifact)) or True,
    )
    pipe = _StubPipe()
    scope = provision.ArmingScope(cfg, tmp_path, None)
    with scope:
        assert provision.arm_compile(pipe) is True
    assert seen == [(pipe, cfg, tmp_path, None)]
    assert scope.objects == ((pipe, True),)
    # the scope closed: arming outside it is a no-op again (never armed),
    # not a raise — a compiled release's setup() calling arm_compile() a
    # second time after its own scope exits must not crash either.
    assert provision.arm_compile(pipe) is False


def test_arming_scope_is_a_noop_when_compile_is_none() -> None:
    with provision.ArmingScope(None, Path("."), None):
        assert provision.arm_compile(_StubPipe()) is False


# ---------------------------------------------------------------------------
# executor integration: the real ensure_setup() codepath arms a self-loaded
# pipeline via gen_worker.arm_compile() (test_cast_drop_th737.py pattern —
# no mock of the executor itself, only the compile_cache.apply leaf).
# ---------------------------------------------------------------------------


def _executor(spec: EndpointSpec, tmp_path: Path, sent: list, monkeypatch) -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    store = ModelStore(_send, cache_dir=tmp_path / "cas", vram_budget_bytes=4 << 30)

    async def _fake_ensure_local(ref, **kwargs) -> Path:
        return tmp_path / "snap"

    monkeypatch.setattr(executor_mod, "ensure_local", _fake_ensure_local)
    return Executor([spec], _send, store=store)


def test_executor_arms_self_loaded_pipeline_via_arm_compile(
    tmp_path, monkeypatch
) -> None:
    applied: list[tuple] = []

    def _fake_apply(pipeline, cfg, *, cache_ready, guard=True):
        applied.append((pipeline, cfg, cache_ready))
        return True

    monkeypatch.setattr(cc, "apply", _fake_apply)

    class Endpoint:
        def setup(self, model: str) -> None:
            self.pipe = _StubPipe()
            self.armed = gen_worker.arm_compile(self.pipe)

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out()

    compile_cfg = Compile(family=FAMILY, shapes=((768, 768),))
    spec = EndpointSpec(
        name="wan-generate", method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint, attr_name="run",
        models={"model": Hub("acme/wan")}, resources=Resources(vram_gb=1.0),
        compile=compile_cfg,
    )
    sent: list = []

    async def _go() -> None:
        ex = _executor(spec, tmp_path, sent, monkeypatch)
        inst = await ex.ensure_setup(spec, {
            wire_ref(spec.models["model"]): pb.Snapshot(
                digest="blake3:" + "a" * 64),
        })
        assert inst.armed is True
        assert len(applied) == 1
        pipeline, cfg, cache_ready = applied[0]
        assert pipeline is inst.pipe
        assert cfg is compile_cfg
        # no hub-attached artifact / no local cache seeded -> stays eager.
        assert cache_ready is False

    asyncio.run(_go())


def test_executor_eager_registration_arm_compile_no_raise_setup_completes(
    tmp_path, monkeypatch
) -> None:
    """ie#522 (Paul's ruling, 2026-07-22) — REVERT-TURNS-RED: an EAGER
    registration (no compile=Compile(...) on the EndpointSpec, exactly the
    gw#556 recipe's "no compile payload" release) whose setup() calls
    arm_compile() unconditionally (the real wan-2.2/sdxl style — never
    special-cased with try/except) must complete ensure_setup() cleanly:
    no raise, warmup proceeds, the pipeline is simply never armed. Live
    repro: wan-2.2's eager smoke crashed with `RuntimeError: ... no active
    compile-arming scope` at warmup/phase=load before this fix."""
    applied: list[tuple] = []
    monkeypatch.setattr(cc, "apply", lambda *a, **k: applied.append(1) or True)

    class Endpoint:
        def setup(self, model: str) -> None:
            self.pipe = _StubPipe()
            self.armed = gen_worker.arm_compile(self.pipe)  # unconditional, no guard

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out()

    spec = EndpointSpec(
        name="wan-generate", method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint, attr_name="run",
        models={"model": Hub("acme/wan")}, resources=Resources(vram_gb=1.0),
        compile=None,  # the eager registration: no compile grid declared
    )
    sent: list = []

    async def _go() -> None:
        ex = _executor(spec, tmp_path, sent, monkeypatch)
        inst = await ex.ensure_setup(spec, {
            wire_ref(spec.models["model"]): pb.Snapshot(
                digest="blake3:" + "a" * 64),
        })
        assert inst.armed is False
        assert applied == []

    asyncio.run(_go())


def test_executor_never_arms_without_an_explicit_call(tmp_path, monkeypatch) -> None:
    """A self-loading endpoint that DECLARES compile= but never calls
    arm_compile() stays silently eager at runtime too (this is exactly the
    pre-pgw#517 bug — discovery now blocks it, but EndpointSpec can still be
    hand-built like this in a test; the executor must not paper over it by
    auto-arming something it never saw)."""
    applied: list = []
    monkeypatch.setattr(cc, "apply", lambda *a, **k: applied.append(1) or True)

    class Endpoint:
        def setup(self, model: str) -> None:
            self.pipe = _StubPipe()  # never arms

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out()

    spec = EndpointSpec(
        name="wan-generate", method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint, attr_name="run",
        models={"model": Hub("acme/wan")}, resources=Resources(vram_gb=1.0),
        compile=Compile(family=FAMILY, shapes=((768, 768),)),
    )
    sent: list = []

    async def _go() -> None:
        ex = _executor(spec, tmp_path, sent, monkeypatch)
        await ex.ensure_setup(spec, {
            wire_ref(spec.models["model"]): pb.Snapshot(
                digest="blake3:" + "a" * 64),
        })
        assert applied == []

    asyncio.run(_go())


def test_self_loaded_w8a8_pipeline_emits_exact_target_and_requires_cell_fence(
    tmp_path, monkeypatch,
) -> None:
    """arm_compile ownership reaches the same target/fail-closed production path."""

    class _Denoiser:
        def forward(self, value):  # pragma: no cover - compile contract only
            return value

    class _CompilePipe:
        def __init__(self):
            self.transformer = _Denoiser()
            self._cozy_weight_lane = "w8a8"

    def _fake_enable(pipe, _cfg, _cache_dir, _artifact):
        setattr(pipe, cc._MARKER_ATTR, {
            "failure_signal": {
                "callback": None,
                "lock": threading.Lock(),
                "successful_calls": 0,
                "cache_hits": 0,
                "cache_misses": 0,
            },
            "originals": [],
            "regional_mods": [],
        })
        return True

    class Endpoint:
        def setup(self, model: str) -> None:
            self.pipe = _CompilePipe()
            self.armed = gen_worker.arm_compile(self.pipe)

        def warmup(self) -> None:
            signal = getattr(self.pipe, cc._MARKER_ATTR)["failure_signal"]
            with signal["lock"]:
                signal["successful_calls"] += 1
                signal["cache_hits"] += 1

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out()

    monkeypatch.setattr(provision, "enable_compiled", _fake_enable)
    compile_cfg = Compile(family=FAMILY, shapes=((768, 768),))
    spec = EndpointSpec(
        name="wan-w8a8", method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint, attr_name="run",
        models={"model": Hub("acme/wan", flavor="fp8-w8a8")},
        resources=Resources(vram_gb=1.0), compile=compile_cfg,
    )
    model_ref = wire_ref(spec.models["model"])
    cell_ref = (
        f"root/family-{FAMILY}#inductor-rtx-4090-torch2.9-w8a8"
    )
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "cell.tar.gz").write_bytes(b"delivered-cell")
    sent: list = []

    async def _go() -> Executor:
        ex = _executor(spec, tmp_path, sent, monkeypatch)
        await ex.ensure_setup(spec, {
            model_ref: pb.Snapshot(digest="blake3:" + "a" * 64),
            cell_ref: pb.Snapshot(digest="blake3:" + "b" * 64),
        })
        return ex

    ex = asyncio.run(_go())
    (target,) = ex.compile_targets()
    assert target.pipeline_weight_lane == "w8a8"
    assert target.active_compile_ref == cell_ref
    assert target.active_compile_snapshot_digest == "blake3:" + "b" * 64
    assert list(target.function_names) == ["wan-w8a8"]
    assert [
        (binding.slot, binding.ref, binding.snapshot_digest)
        for binding in target.model_bindings
    ] == [("model", model_ref, "blake3:" + "a" * 64)]
    with pytest.raises(RetryableError, match="required_compile_missing"):
        ex._validate_required_compile(
            spec,
            pb.RunJob(
                function_name=spec.name,
                models=[pb.ModelBinding(slot="model", ref=model_ref)],
                snapshots={model_ref: pb.Snapshot(digest="blake3:" + "a" * 64)},
            ),
        )
