"""The trace-time contexts (pgw#1370).

The publish derive runs the author's ``Model.load`` and entrypoints AS-IS,
so it must answer exactly the ctx surface that code touches -- with trace
semantics: config-only checkpoint tree, hollow instantiation, platform-
fallback defaults, no adapter, no-op egress. ``ctx.is_trace`` is True and
author code may branch on it (the contract file does, to skip the adapter
refusal).

These are NOT the serving contexts: chunk-store streaming, adopt/boot and
the deploy-state defaults read are pgw#1372's surface. The two sides share
the SPELLING; that spelling is frozen by the Paul-reviewed ``main_v2.py``.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional

from ..serving.context import _lane_torch_dtype


class TraceLoadContext:
    """What ``Model.load`` sees under ``gen-worker release derive``.

    Deliberately duck-typed (the author annotates ``LoadContext[MT]``; the
    harness-private object answers the same spelling): the serving
    ``LoadContext`` carries a deploy binding and property-backed facts this
    trace half has no business constructing.

    There is NO ``is_trace`` -- Paul deleted it from the author surface
    (author code branching on it corrupts compilation coverage; author code
    is trace-oblivious by construction). Arm coverage is the DERIVE'S job,
    via input/binding enumeration (payload enums x adapter states x
    checkpoint-defaults variants).
    """

    def __init__(
        self,
        *,
        lane: Any,
        checkpoint_dir: Path,
        model_type: Optional[type] = None,
        defaults_instance: Any = None,
    ) -> None:
        self.lane = lane
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_type = model_type
        self.defaults_instance = defaults_instance
        # pgw#1510 follow-up: PRIVATE, for the same reason the request half's
        # is. The serving `LoadContext` exposes no `log` at all, so a public
        # one here is a name the author can reach at trace and not at serve --
        # and it answered a Logger, so an author who logged inside `load()`
        # got "'Logger' object is not callable" where serve gives a plain
        # AttributeError. Now both say the same thing. Kept (rather than
        # deleted) under the private name so the next internal log line does
        # not re-add `self.log` by muscle memory.
        self._log = logging.getLogger("gen_worker.release.trace")
        #: Modules the author marked via ctx.compile() -- discovery hooks
        #: exactly these during the payload drives.
        self.marked_modules: list[Any] = []

    def load(self, loader: Any) -> Any:
        """Hollow-materialize the CONFIG-ONLY tree through the author's loader.

        At serve, ``ctx.load`` streams tensors from the chunk store straight
        to VRAM in the lane contract's layout (pgw#1372). At trace there are
        no tensors at all: the loader's own ``from_pretrained`` runs against
        the config-only subset snapshot inside the ambient
        ``torchcg.hollow_session`` (fake parameters, real buffers), and the
        lane's registry-derived dtype stands in for the layout's.
        """
        from_pretrained = getattr(loader, "from_pretrained", None)
        if from_pretrained is None:
            raise TypeError(
                f"ctx.load() needs a loader with from_pretrained "
                f"(a diffusers/transformers class); got {loader!r}"
            )
        # pgw#1512, Paul's per-component-passthrough ruling. There is NO
        # `torch_dtype=` here any more, and its absence is the fix.
        #
        # This used to read one dtype off the lane and hand it to
        # `from_pretrained`, casting EVERY component of the tree to it. The
        # serving loader does the opposite and says so: "No `torch_dtype=`
        # (the lane contract IS the dtype)" (`serving/context.py`), and
        # "bytes land verbatim in the container's own dtype ... any conversion
        # is the STORE's contract-negotiation job, NEVER load time. A
        # container that disagrees with the active lane is reported, not
        # silently repaired" (`serving/streaming/engine.py`). So the trace
        # performed exactly the conversion serve refuses, and a DiT lane's
        # bf16 landed on the VAE beside it — a bf16 bias meeting an fp32
        # activation in a decode block that is fine on a pod.
        #
        # Precision is now decided PER COMPONENT, by `component_dtype` below,
        # through the session policy torchcg asks (tcg#68). The lane still
        # governs the component its contract describes, so the marked
        # denoiser's precision — which IS graph identity (pgw#1458) — still
        # comes from the contract, and pgw#1448's "real dtype, never the
        # spelling" rule still applies to it.
        loaded = from_pretrained(self.checkpoint_dir)
        # Adapter application mutates WEIGHTS (or injects adapter layers);
        # at trace every parameter is fake and no adapter bytes exist, so the
        # enumeration's fake-adapter arms must not hit real LoRA I/O. The
        # graphs observed on those arms are the base modules' -- a served
        # adapter that changes the module graph re-keys and first-encounter
        # mints (pgw#1371/#1372 own the branch-bearing lora story).
        for lora_call in ("load_lora_weights", "set_adapters", "unload_lora_weights"):
            if hasattr(loaded, lora_call):
                setattr(loaded, lora_call, _noop)
        return loaded

    def component_dtype(self, tree: Any, subfolder: Any) -> Any:
        """The precision ONE component loads at (pgw#1512, Paul's ruling).

        Installed as torchcg's session policy, so it is asked once per
        component instead of once per tree.

        1. **The lane governs the component its CONTRACT DESCRIBES.** A layout
           contract names the tensors it covers, and a pattern's first segment
           is the component that owns them (``unet.conv_out.weight`` ->
           ``unet``). For h3 that set is the DiT and nothing else, which is the
           whole ruling: a denoiser contract says nothing about the VAE beside
           it. This branch is where precision-is-identity lives (pgw#1458), so
           it keeps pgw#1448's real-dtype-never-the-spelling read.
        2. **Otherwise the component's own container answers**, through the
           SAME stub-aware reader the serve path uses
           (``serving.checkpoint_dtype``): its config's declared dtype first,
           then the safetensors headers. On a projected tree those headers are
           128-byte stubs, and that reader is the one that knows it -- a naive
           open here would read a stub as truth.
        3. **Nothing may default.** If neither the contract nor the container
           can say, this REFUSES by name. A quiet fp32 is the defect pgw#1448
           deleted, and at trace it would silently re-key a graph.
        """

        from ..serving.checkpoint_dtype import (
            _config_dtype,
            _tensor_dtype as _header_dtype,
        )

        directory = Path(str(tree))
        name = str(subfolder).strip("/") if subfolder else ""
        if name:
            directory = directory / name
        else:
            # diffusers loads a component by handing its DIRECTORY over with no
            # subfolder (`AutoencoderKL.from_pretrained(<tree>/vae)`), so an
            # absent subfolder does not mean "the root". The component is
            # whatever this directory is called relative to the tree the author
            # was given; only the tree ITSELF is the root, and that is the
            # single-module checkpoint the lane speaks for.
            try:
                relative = directory.resolve().relative_to(
                    self.checkpoint_dir.resolve()
                )
            except (ValueError, OSError):
                relative = None
            if relative is not None and str(relative) not in (".", ""):
                name = str(relative).strip("/")

        # 1. THE COMPONENT'S OWN BYTES, when it has any. A serving tree has
        #    been CONVERTED through the lane contract by the store, so its
        #    safetensors headers are the precision the pod actually runs —
        #    read through the stub-aware reader, which knows a 128-byte
        #    projection stub from a real file.
        own = _header_dtype(directory)
        if own is not None:
            return own

        # 2. OTHERWISE THE LANE. pgw#1530: this is the half pgw#1512 got
        #    wrong. A contract's `tensors` list enumerates the DENOISER's own
        #    parameter names — every shipped contract does, `conv_in.weight`
        #    for sd15, `transformer_blocks.…` for h3, never a `unet.`/
        #    `transformer.` prefix — but its `dtype` states the precision of
        #    the TREE the store converts to that contract. Treating the
        #    tensor list as a component roster made the governed set match
        #    nothing on every endpoint in the fleet, so the lane's dtype was
        #    never applied anywhere and a config-only derive silently traced
        #    fp32 graphs under a bf16 lane.
        declared = _lane_torch_dtype(self.lane, checkpoint_dir=self.checkpoint_dir)
        if declared is not None:
            return declared

        # 3. THE COMPONENT'S CONFIG, LAST and only when nothing above spoke.
        #    On a converted tree a component config is the PUBLISHER's, not
        #    the store's: sd15's `text_encoder/config.json` still says
        #    `float32` in a tree whose bytes are bf16. Preferring it is what
        #    put an fp32 encoder beside a bf16 denoiser and produced
        #    `Input type (float) and bias type (c10::BFloat16)` on payload 0.
        config = _config_dtype(directory)
        if config is not None:
            return config

        # Nothing anywhere can say. No cast — the absence of a conversion,
        # which is what the streaming loader does with bytes it was given no
        # contract for.
        return None

    def compile(self, target: Any) -> Any:
        """torch.compile-style marking, trace half (pgw#1370/#1372 contract).

        At DERIVE this records the marked module (discovery hooks it during
        the payload drives) and returns it unchanged. At SERVE (pgw#1372) it
        returns the adopted compiled graph for this (graph, lane, sm) when
        the store has it, else the module unchanged while the hole mints in
        the background -- the author's marked line IS the swap point.

        A non-module with ``.components`` (a diffusers pipeline) is sugar:
        every nn.Module component is marked ("compile everything
        compilable"). Typos are real AttributeErrors at the author's line --
        no strings, no self-structure assumptions.
        """
        import torch

        if isinstance(target, torch.nn.Module):
            if all(existing is not target for existing in self.marked_modules):
                self.marked_modules.append(target)
            return target
        components = getattr(target, "components", None)
        if isinstance(components, Mapping):
            for component in components.values():
                if isinstance(component, torch.nn.Module):
                    self.compile(component)
            return target
        raise TypeError(
            f"ctx.compile() marks nn.Modules (or a pipeline-like object "
            f"whose .components carries them); got {type(target).__name__}"
        )

    def defaults(self) -> Any:
        """The enumerated defaults VARIANT for the class-header model type.

        ``Model[SDXL]`` is the single source of the type; at serve the
        checkpoint's hub row decodes as ``SDXL.Defaults`` with missing
        fields filled from platform values. At trace the derive enumerates
        recipe-relevant variants (platform row; cfg flipped when the schema
        carries it) because they change the executed arm and thus the
        observed graphs.
        """
        if self.defaults_instance is not None:
            return self.defaults_instance
        if self.model_type is None:
            raise TypeError(
                "ctx.defaults() reads the model type off the class header "
                "(class X(Model[SDXL], ...)); this model's base is "
                "unparameterized"
            )
        defaults_type = getattr(self.model_type, "Defaults", self.model_type)
        return defaults_type()


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


class StepBudgetReached(Exception):
    """The trace has seen every shape this denoise loop produces.

    A diffusers denoise loop runs the SAME shapes on every step -- step 3 of
    28 teaches the trace nothing step 1 did not. The derive therefore runs
    each enumerated pass under a STEP BUDGET and lets the author's own
    ``callback_on_step_end`` raise this once the budget is spent; the drive
    treats it as a completed pass. Modules that run AFTER the loop (a marked
    VAE decoder) never execute under a budget, so the derive re-drives
    unbudgeted whenever a marked module is still unobserved -- honesty is
    preserved, the redundant 27 steps are not paid.
    """


class TraceSurfaceUnavailable(RuntimeError):
    """An author touched a ctx member the derive genuinely cannot answer.

    Reserved for members whose whole content is BYTES OR A PEER that do not
    exist at trace time -- a dataset to resolve, a blob to materialize, a
    checkpoint stream to open. Everything else on the surface is answered:
    really where trace can be real, as a recorder where the derive wants the
    observation, as a stated empty where the fact simply has no trace value.

    A no-op is wrong for these three because a fabricated path or an empty
    file makes author code read something that is not there and fail two
    frames later, naming the author's line instead of the derive's gap.
    """


class TraceRequestContext:
    """What entrypoints see under ``gen-worker release derive``.

    pgw#1461: this used to answer FIVE members while the serving
    ``RequestContext`` answers 48. The authoring guide's own canonical
    example (``ctx.for_request``) raised ``AttributeError`` straight into a
    hard ``DeriveError``, so most real endpoints could not derive at all --
    and every release fixture passed because none of them called a missing
    member. A surface with holes is not a surface; the derive must answer
    everything an author may write, and the fence
    (``test_trace_context_surface``) is what keeps it that way when
    a new serving member lands.

    Three answer kinds, chosen per member and never by accident:

    * **Real** -- ``for_request``, ``generator``, ``clamp``, ``mktemp``,
      ``stage``, ``workflow_checkpoint``. These are pure functions of things
      a trace HAS, so answering them with a stub would be a lie for no gain.
      ``for_request`` in particular is the guide's canonical line and it
      clones schedulers, which is exactly what changes the observed graph.
    * **Recorder** -- ``progress``, ``adjusted``/``adjustments``, ``warn``/
      ``warnings``, ``log``. The derive wants the observation, and a test
      can read it back.
    * **Stated empty** -- ``models``, ``loras``, ``config``, the job-side
      paths. Empty is a statement here, not a placeholder: a trace resolves
      no checkpoint, applies no adapter and has no dataset.

    No ``is_trace`` (deleted from the author surface): author code branching
    on it corrupts compilation coverage.
    """

    def __init__(
        self,
        *,
        lane: Any,
        checkpoint_ref: str = "",
        step_budget: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
        device: Any = None,
    ) -> None:
        self.lane = lane
        #: None = run the author's full step count.
        self.step_budget = step_budget
        self.checkpoint_ref = checkpoint_ref or "trace:config-only"
        # pgw#1510: PRIVATE. `log` is a METHOD on this surface because it is a
        # method on the serving RequestContext; binding the Logger to the
        # public name is what made `ctx.log("...")` die "'Logger' object is
        # not callable" mid-drive.
        self._log = logging.getLogger("gen_worker.release.trace")
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        # pgw#1458: the trace DEVICE, which is not `cuda_ready()`. A derive on
        # a GPU-less publish box traces on fake cuda, and author code that
        # reads ctx.device must see the device the graph is being stamped
        # with -- reading the host's real availability here would place a
        # tensor on cpu inside a cuda trace and produce the mixed placement
        # AOTI rejects.
        self._device = device
        self._warnings: list[str] = []
        self._adjustments: list[dict[str, str]] = []
        #: (progress, stage, extra) recorded in call order.
        self.progress_events: list[tuple[Optional[float], Optional[str], dict[str, Any]]] = []
        #: Endpoints this trace answered EMPTY because it cannot reach a peer.
        #: The derive reports these: graphs past such a call are unobserved,
        #: which is a stated outcome, not a silent one.
        self.unanswered_calls: list[tuple[str, str]] = []
        self._temporaries: list[Any] = []

    # -- knobs / control ----------------------------------------------------
    def clamp(
        self,
        field: str,
        requested: float,
        *,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        reason: str = "",
    ) -> float:
        """Same arithmetic as the serving ctx, and the same RECORD.

        pgw#1461: "trace records nothing" was the old comment and it was the
        whole bug in miniature -- a clamp that changes the executed arm, and
        therefore the observed graph, left no trace of having happened.
        """
        applied = float(requested)
        if lo is not None and applied < lo:
            applied = float(lo)
        if hi is not None and applied > hi:
            applied = float(hi)
        if applied != float(requested):
            self.adjusted(field, requested, applied, reason)
        return applied

    def raise_if_cancelled(self, message: str = "request cancelled") -> None:
        del message

    def warn(self, message: str) -> None:
        """Caller-visible advisory at serve; recorded AND logged at trace."""
        self._warnings.append(str(message))
        self._log.warning("trace: %s", message)

    def log(self, message: str, level: str = "info", **fields: Any) -> None:
        """The operator diagnostic stream, answered as a METHOD (pgw#1510).

        At serve this rides ``request.log`` to the hub. At trace there is no
        request and no event lane, so it goes to the derive's own logger --
        really where trace can be real, which is this module's rule for every
        member it answers.

        It exists as a METHOD and not as a bound ``Logger`` because that is
        what the serving ``RequestContext`` is: an author writing the
        documented ``ctx.log("...")`` got ``'Logger' object is not callable``
        mid-drive, and the derive died on a line that is correct at serve.
        The KIND of a member is part of the surface, not an implementation
        detail -- ``test_trace_context_surface`` now compares every member's
        kind against the serving class so this cannot recur silently.
        """

        self._log.info(
            "trace: %s%s",
            message,
            f" {fields!r}" if fields else "",
        )

    # -- egress -------------------------------------------------------------
    def step_callback(
        self, num_inference_steps: int = 0, **kwargs: Any
    ) -> Callable[..., dict[str, Any]]:
        """A diffusers ``callback_on_step_end`` that enforces the step budget.

        The parameter is spelled ``num_inference_steps`` to match the serving
        context exactly: it was ``total_steps`` here, so an author passing it
        BY KEYWORD -- which is how diffusers callers usually write it -- got a
        TypeError at trace and a working call at serve. A member that exists
        with the wrong signature is still a hole in the surface (pgw#1461).
        """
        del num_inference_steps, kwargs
        seen = 0
        budget = self.step_budget

        def callback(
            _pipe: Any, _index: Any, _timestep: Any,
            callback_kwargs: Any = None, **_: Any,
        ) -> dict[str, Any]:
            del callback_kwargs
            nonlocal seen
            seen += 1
            if budget is not None and seen >= budget:
                raise StepBudgetReached
            return {}

        return callback

    # -- egress: stub assets; nothing is encoded or uploaded at trace ------
    def save_image(self, image: Any, *, format: str = "webp", **_: Any) -> Any:
        """A stub asset: nothing is encoded or uploaded at trace time."""
        del image
        from ..api.types import ImageAsset

        return ImageAsset(ref=f"trace://image.{format}")

    #: pgw#1522. This context's outputs are DISCARDED BY CONSTRUCTION — every
    #: `save_*` below is a stub returning a `trace://` ref, so nothing is
    #: uploaded and nothing banks. The output-integrity floor reads this to
    #: know it is not judging a render (`output_integrity.judged`): under a
    #: hollow session the parameters carry no bytes, so a blank frame is the
    #: only possible output and the floor's verdict, while TRUE, is about the
    #: substrate rather than the endpoint.
    #:
    #: Private because it is a platform fact, not an author input: there is
    #: deliberately no `ctx.is_trace` for author code to branch on.
    _outputs_discarded = True

    def save_video(self, video: Any, ref: Optional[str] = None, *, format: str = "mp4",
                   **_: Any) -> Any:
        del video
        from ..api.types import VideoAsset

        return VideoAsset(ref=ref or f"trace://video.{format}")

    def save_audio(self, audio: Any, ref: Optional[str] = None, *, format: str = "wav",
                   **_: Any) -> Any:
        del audio
        from ..api.types import AudioAsset

        return AudioAsset(ref=ref or f"trace://audio.{format}")

    def save_bytes(self, ref: str, data: bytes, **_: Any) -> Any:
        del data
        from ..api.types import Asset

        return Asset(ref=f"trace://{ref}")

    def save_file(self, ref: str, local_path: Any, *, create: bool = False, **_: Any) -> Any:
        del local_path, create
        from ..api.types import Asset

        return Asset(ref=f"trace://{ref}")

    def save_checkpoint(self, ref: str, local_path: Any, format: Optional[str] = None,
                        **_: Any) -> Any:
        del local_path, format
        from ..api.types import Asset

        return Asset(ref=f"trace://{ref}")

    # -- real: pure functions of things a trace actually has ---------------
    def for_request(
        self,
        pipeline: Any,
        *,
        slot: str = "",
        sampler: str = "",
        seed: Optional[int] = None,
        generator: Optional[Any] = None,
        scheduler_config: Optional[dict[str, Any]] = None,
        schedulers: Optional[Sequence[str]] = None,
    ) -> Any:
        """The per-request view, for REAL -- the guide's canonical line.

        Answered by the serving implementation rather than stubbed, because
        the clone it performs is exactly what can change the observed graph:
        a different sampler is a different scheduler is a different denoise
        call. A stub returning ``pipeline`` would derive graphs for a
        pipeline no request ever runs, and nothing downstream could tell.
        """

        from ..view import for_request as _view_for_request

        del slot
        gen = generator
        if gen is None and seed is not None:
            gen = self.generator(seed)
        return _view_for_request(
            pipeline, sampler=sampler, objective="", generator=gen,
            scheduler_config=scheduler_config, schedulers=schedulers,
        )

    @property
    def device(self) -> Any:
        """The device THIS TRACE is stamping, not the host's availability."""

        import torch

        if self._device is not None:
            return torch.device(self._device) if isinstance(self._device, str) else self._device
        return torch.device("cpu")

    def generator(self, seed: Optional[int] = None) -> Any:
        import torch

        # A fake-cuda trace has no real cuda device to seed a generator on;
        # the generator's DEVICE is not part of graph identity (its outputs
        # are, and at trace they are zeros), so cpu is the honest home.
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        return gen

    def mktemp(self) -> Path:
        holder = tempfile.TemporaryDirectory(prefix="trace-ctx-")
        self._temporaries.append(holder)
        return Path(holder.name)

    def checkpoint_dir(self, *, key: str = "") -> Path:
        """The CONFIG-ONLY subset the derive resolved; there are no weights."""

        del key
        if self._checkpoint_dir is None:
            raise TraceSurfaceUnavailable(
                "ctx.checkpoint_dir() has no config-only tree in this trace: "
                "the derive was constructed without one. This is the derive's "
                "gap, not the endpoint's."
            )
        return self._checkpoint_dir

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Bracketing is real; only the timing sink is absent at trace."""

        del name
        yield

    def workflow_checkpoint(self, key: str, fn: Callable[[], Any]) -> Any:
        """No resume at trace, so the work always RUNS -- which is the point.

        Answering from a cache would skip exactly the code whose graphs the
        derive exists to observe.
        """

        del key
        return fn()

    def position(self, phase: str = "") -> Optional[float]:
        del phase
        return None

    # -- recorders: the derive wants the observation, and tests read it back
    def progress(
        self,
        progress: Optional[float] = None,
        stage: Optional[str] = None,
        **extra: Any,
    ) -> None:
        self.progress_events.append((progress, stage, dict(extra)))

    def adjusted(self, field: str, requested: Any, applied: Any, reason: str = "") -> None:
        self._adjustments.append({
            "field": str(field), "requested": str(requested),
            "applied": str(applied), "reason": str(reason),
        })

    @property
    def adjustments(self) -> tuple[dict[str, str], ...]:
        return tuple(self._adjustments)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    # -- stated empties: empty is the TRUE value at trace, not a placeholder
    @property
    def models(self) -> dict[str, str]:
        return {}

    @property
    def loras(self) -> dict[str, Any]:
        """No adapter is applied at trace -- TraceLoadContext no-ops the calls."""
        return {}

    @property
    def config(self) -> dict[str, Any]:
        return {}

    @property
    def cancelled(self) -> bool:
        return False

    @property
    def boot_warmup(self) -> bool:
        return False

    @property
    def publishes(self) -> bool:
        """Nothing is published from a trace; egress returns stub refs."""
        return False

    @property
    def emits_media(self) -> bool:
        """TRUE: the media egress path is code whose graphs must be observed."""
        return True

    @property
    def request_id(self) -> str:
        return "trace"

    @property
    def execution_lane(self) -> str:
        return str(getattr(self.lane, "contract", "") or "")

    @property
    def hf_token(self) -> str:
        return ""

    # -- job-side surface: the same three kinds, for the training half ------
    @property
    def candidate(self) -> dict[str, Any]:
        return {}

    @property
    def candidate_path(self) -> Optional[str]:
        return None

    @property
    def source(self) -> dict[str, Any]:
        return {}

    @property
    def source_path(self) -> Optional[str]:
        return None

    @property
    def destination(self) -> dict[str, Any]:
        return {}

    @property
    def resume_from(self) -> dict[str, Any]:
        return {}

    @property
    def resume_from_path(self) -> Optional[str]:
        return None

    @property
    def text_encoder(self) -> dict[str, Any]:
        return {}

    @property
    def text_encoder_path(self) -> Optional[str]:
        return None

    @property
    def dataset_paths(self) -> dict[str, str]:
        return {}

    # -- the three that are BYTES OR A PEER, and so refuse by name ---------
    def resolve_dataset(self, ref: str, **_: Any) -> str:
        raise TraceSurfaceUnavailable(
            f"ctx.resolve_dataset({ref!r}) needs real dataset bytes and a trace "
            f"has none. A fabricated path would fail two frames later inside "
            f"the endpoint's own reader, naming the author's line instead of "
            f"this gap. Guard the call, or derive this endpoint with a "
            f"dataset-free enumeration arm."
        )

    def materialize_blob(self, digest: str, dest: Any, *, origin: str = "payload") -> Any:
        del dest, origin
        raise TraceSurfaceUnavailable(
            f"ctx.materialize_blob({digest!r}) needs real payload bytes and a "
            f"trace has none. An empty file at the destination would be worse "
            f"than this refusal: the endpoint would read it and fail obscurely."
        )

    def open_checkpoint_stream(self, ref: str, **_: Any) -> Any:
        raise TraceSurfaceUnavailable(
            f"ctx.open_checkpoint_stream({ref!r}) streams real checkpoint bytes "
            f"and the derive runs against a CONFIG-ONLY subset -- there is "
            f"nothing to stream. ctx.load() is the weights-free path."
        )

    def call_endpoint(
        self, endpoint: str, function: str, payload: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        """A peer endpoint the derive cannot reach: answered EMPTY, and STATED.

        Refusing would make every composing endpoint underivable, which is
        the pgw#1461 defect in a new place. Answering silently would claim
        coverage the trace does not have. So it answers empty and records the
        call: graphs past this point are unobserved, exactly like an
        unobserved target, and the derive reports it rather than implying it.
        """

        del payload
        self.unanswered_calls.append((str(endpoint), str(function)))
        self._log.warning(
            "trace: ctx.call_endpoint(%r, %r) answered EMPTY -- this trace "
            "cannot reach a peer endpoint, so graphs reached only through its "
            "result are unobserved and will mint on first live encounter.",
            endpoint, function,
        )
        return {}


__all__ = [
    "StepBudgetReached",
    "TraceLoadContext",
    "TraceRequestContext",
    "TraceSurfaceUnavailable",
]
