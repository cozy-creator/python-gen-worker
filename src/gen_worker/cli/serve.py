"""``gen-worker serve`` — persistent local dev server.

``gen-worker run`` reloads the model on every invocation. ``serve`` boots the
endpoint ONCE — importing the ``main`` module, instantiating every
``@inference`` class, and running ``setup()`` so the models stay VRAM-resident
— then loops, serving many requests warm. Ctrl-C tears everything down.

Two ways to talk to a running serve, sharing ONE dispatch handler (the same
``run.dispatch_request`` the one-shot ``run`` subcommand uses):

* **stdin/stdout (default):** read newline-delimited JSON (NDJSON) request
  lines from this process's own stdin, write NDJSON result lines to stdout,
  logs/events to stderr. Zero IPC — great for piping a batch or interactive
  poking. The process exits when stdin closes.
* **Unix domain socket (always on):** a stdlib ``AF_UNIX`` listener at
  ``./.gen-worker.sock`` (override ``--socket PATH``) accepts sequential
  connections; each connection sends one NDJSON request and reads one NDJSON
  result. ``gen-worker invoke`` is the client. No HTTP, no port — it is all
  one machine.

Wire format (symmetric with ``invoke``):

* request:  ``{"function": "<fn_name>", "payload": <decoded-json>}``
* response: ``{"ok": true, "events": [{"event": "result", "value": ...}, ...]}``
            or ``{"ok": false, "error": {"kind": "...", "message": "..."}}``

Transport-fidelity caveat: production dispatch is gRPC-from-the-orchestrator.
``serve`` mirrors setup, context wiring, memory management, and GPU
serialization faithfully (shared code) but the transport differs. The right
trade for warm-model local iteration.

The full design lives in ``progress.json`` issue #340.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgspec

from ..api.errors import CanceledError
from ..models.cache import ModelCache
from . import run as run_mod
from . import transport
from .local_context import _stderr_emitter, build_local_context
from .protocol import PROTOCOL_VERSION, gen_worker_version


DEFAULT_SOCKET_PATH = "./.gen-worker.sock"


# --------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------

def add_subparser(sub: argparse._SubParsersAction[Any]) -> None:
    """Register the ``serve`` subcommand on the top-level parser."""
    p = sub.add_parser(
        "serve",
        help="Run a persistent local dev server (models loaded once, warm).",
        description=(
            "Boot one endpoint once (setup() runs, models stay resident) and "
            "serve many requests warm over NDJSON on stdin/stdout AND a Unix "
            "domain socket. Use 'gen-worker invoke' from another shell to fire "
            "requests at the socket. Ctrl-C tears down (shutdown() if present) "
            "and removes the socket file."
        ),
    )
    p.add_argument(
        "--config", dest="config_path", default=None,
        help="Path to endpoint.toml (defaults to ./endpoint.toml).",
    )
    p.add_argument(
        "--module", dest="module", default=None,
        help="Python module path to import (overrides endpoint.toml `main`).",
    )
    p.add_argument(
        "--socket", dest="socket_path", default=DEFAULT_SOCKET_PATH,
        help=(
            "Unix domain socket path the server listens on "
            f"(default: {DEFAULT_SOCKET_PATH}). Use distinct paths to run "
            "several serves concurrently."
        ),
    )
    p.add_argument(
        "--listen", dest="listen", default=None, metavar="ADDR",
        help=(
            "Listen address, overriding --socket: 'tcp://0.0.0.0:PORT' for TCP "
            "(host:port; works across containers with `docker run -p`), "
            "'unix:///path' or a bare path for a Unix socket. Default: the "
            "--socket Unix path (#347)."
        ),
    )
    p.add_argument(
        "--offline", action="store_true",
        help=(
            "Use only the local CAS — fail with exit 3 instead of fetching "
            "missing model weights from tensorhub / huggingface."
        ),
    )
    p.add_argument(
        "--device", dest="device", default=None,
        help="Override the torch device (e.g. 'cuda:0', 'cpu').",
    )
    p.add_argument(
        "--allow-publish", action="store_true",
        help="Allow ConversionContext producer RPCs to hit real tensorhub.",
    )
    p.add_argument(
        "--no-stdin", action="store_true",
        help=(
            "Do not read NDJSON requests from stdin; serve only over the Unix "
            "socket. Useful when launching serve detached in the background."
        ),
    )
    p.add_argument(
        "--function", dest="functions", action="append", default=None,
        metavar="NAME",
        help=(
            "Boot ONLY the @inference class that hosts function NAME (so a "
            "multi-model endpoint does not load every class/model). Repeatable "
            "(--function a --function b) or comma-separated (--function a,b); the "
            "union of their owning classes is booted. Default (omitted): boot "
            "every @inference class."
        ),
    )
    p.add_argument(
        "--list-functions", dest="list_functions", action="store_true",
        help=(
            "List the endpoint's routable function names (and their hosting "
            "class) and exit — without booting/loading any model. Use the "
            "printed names with --function or 'gen-worker invoke <name>'."
        ),
    )
    p.add_argument(
        "--json", dest="json_output", action="store_true",
        help=(
            "With --list-functions, emit JSON — one object per function with its "
            "name, hosting class, kind, and input JSON Schema (from the msgspec "
            "payload type) — instead of the text listing. No model is loaded."
        ),
    )
    p.add_argument(
        "--eager", dest="eager", action="store_true",
        help=(
            "Run setup() for every booted class at startup (load all their "
            "models up front) instead of lazily on first invoke. Useful for "
            "fail-fast / pre-warming; default is lazy per-function loading."
        ),
    )
    p.add_argument(
        "--idle-timeout", dest="idle_timeout", type=float, default=0.0,
        metavar="SECONDS",
        help=(
            "Self-shut-down after SECONDS with no request (frees VRAM when a "
            "warm serve goes unused). 0 (default) = run until SIGINT / stdin EOF. "
            "The idle clock starts after boot and resets on every request."
        ),
    )
    p.set_defaults(_handler=_handle_serve)


def _filter_candidates_by_function(
    candidates: List["run_mod._SelectedFunction"],
    wanted: Optional[List[str]],
) -> List["run_mod._SelectedFunction"]:
    """Restrict to the classes that host the requested function name(s).

    Each ``--function NAME`` names a routable ``fn_name``. We keep every
    candidate whose owning class hosts at least one requested name — so the
    full hosting class (and only that class) is instantiated and ``setup()``'d,
    and its sibling functions stay routable. Other classes are dropped, so
    their models never load. Empty ``wanted`` returns all candidates.
    """
    if not wanted:
        return candidates

    # Accept both repeated flags (--function a --function b) and a
    # comma-separated list (--function a,b,c), so either spelling works.
    names: List[str] = []
    for raw in wanted:
        names.extend(part.strip() for part in str(raw or "").split(",") if part.strip())
    if not names:
        return candidates

    known = {c.fn_name for c in candidates}
    missing = [n for n in names if n not in known]
    if missing:
        raise run_mod._UsageError(
            f"--function {missing[0]!r} not found; available functions: "
            f"{sorted(known)}"
        )

    # Classes that host any requested function.
    wanted_cls_ids = {id(c.cls) for c in candidates if c.fn_name in names}
    return [c for c in candidates if id(c.cls) in wanted_cls_ids]


# --------------------------------------------------------------------------
# GPU serialization — mirror worker.py's BoundedSemaphore sizing
# --------------------------------------------------------------------------

def _make_gpu_semaphore() -> threading.BoundedSemaphore:
    """Size a GPU semaphore the same way the production Worker does.

    One slot per physical GPU; ``max(1, ...)`` so a no-GPU host still gets a
    real primitive. For local serve sequential handling is fine, but reusing
    the same primitive keeps a local ``compiled`` warmup serializing exactly
    like prod.
    """
    gpu_count = 0
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = int(torch.cuda.device_count() or 0)
    except Exception:
        gpu_count = 0
    return threading.BoundedSemaphore(max(1, gpu_count or 1))


# --------------------------------------------------------------------------
# The booted endpoint — instances held warm, functions indexed by name
# --------------------------------------------------------------------------

class _ServedFunction:
    """A booted, ready-to-invoke function: its selection + live instance."""

    def __init__(self, selected: "run_mod._SelectedFunction", instance: Any) -> None:
        self.selected = selected
        self.instance = instance


class _Endpoint:
    """One booted endpoint: every (selected) @inference class instantiated and
    its @inference.function methods indexed by routable name.

    Model residency is driven by the SAME ``ModelCache`` the production worker
    uses (``worker.py``: ``self._model_cache = ModelCache()``). ``setup()`` is
    LAZY by default — a class's setup (which BUILDS its pipeline) runs on the
    FIRST invoke of one of its functions, then the built pipeline is registered
    with the cache (``mark_loaded_to_vram``) after asking the cache to make room
    (``_evict_lru_for_space`` — which DEMOTES the LRU model to the CPU-RAM warm
    tier). On EVERY subsequent dispatch the cache ensures the invoked model is
    VRAM-resident (``_promote_from_cpu`` when it was demoted), so setup() runs
    ONCE per model and thereafter the cache moves models VRAM<->CPU instead of
    re-running setup. ``eager=True`` runs all setups at boot.

    Only the TRANSPORT differs from production (local stdin/UDS vs gRPC); model
    loading + residency + offload go through the identical ``ModelCache`` path.
    """

    def __init__(self, *, offline: bool, allow_publish: bool) -> None:
        self.offline = offline
        self.allow_publish = allow_publish
        self.gpu_semaphore = _make_gpu_semaphore()
        # function_name -> _ServedFunction
        self.functions: Dict[str, _ServedFunction] = {}
        # all live instances (for shutdown), de-duped by id()
        self._instances: List[Any] = []
        self._dispatch_lock = threading.Lock()
        # Lazy-setup bookkeeping, keyed by id(instance) (one instance per class).
        self._setup_done: set[int] = set()
        self._setup_locks: Dict[int, threading.Lock] = {}
        # PRODUCTION GPU memory manager — drives model residency (LRU eviction /
        # VRAM<->CPU demote+promote). Sized from real VRAM so the local card's
        # budget (e.g. an 8GB 4070) over-subscribes exactly like prod would.
        self._model_cache = _build_model_cache()
        # id(instance) -> model_id used as that instance's ModelCache key, and
        # the built pipeline object the cache moves between tiers.
        self._model_id_by_inst: Dict[int, str] = {}
        self._pipeline_by_inst: Dict[int, Any] = {}
        # Wall-clock of the last dispatched request; drives --idle-timeout.
        self.last_activity = time.time()
        # Per-request cancellation registry — request_id -> live RequestContext,
        # mirroring the production worker's ``self._active_requests`` (worker.py).
        # A cancel (socket control frame or server SIGINT) looks the ctx up here
        # and calls the canonical ``ctx.cancel()``; tenant code observes it via
        # ``raise_if_canceled()`` / ``cancel_event`` exactly as in production.
        self._active: Dict[str, Any] = {}
        self._active_lock = threading.Lock()

    def interrupt_request(self, request_id: Optional[str]) -> bool:
        """Cancel ONE in-flight request by id via the canonical ``ctx.cancel()``.

        Returns True if a matching active request was found. The server keeps
        running — only that request is canceled (#352).
        """
        if not request_id:
            return False
        with self._active_lock:
            ctx = self._active.get(request_id)
        if ctx is None:
            return False
        ctx.cancel()
        return True

    def cancel_all(self) -> int:
        """Cancel every in-flight request (server SIGINT/SIGTERM drain, #353).

        Returns how many were signaled. Each tenant handler observes the cancel
        and unwinds cooperatively; teardown then runs shutdown().
        """
        with self._active_lock:
            ctxs = list(self._active.values())
        for ctx in ctxs:
            try:
                ctx.cancel()
            except Exception:
                pass
        return len(ctxs)

    def boot(
        self,
        candidates: List["run_mod._SelectedFunction"],
        *,
        eager: bool = False,
    ) -> None:
        """Instantiate every (selected) class once and index its functions.

        setup() is deferred to first invoke per class (see ``_ensure_resident``)
        unless ``eager`` is set, in which case every class is set up + registered
        in the cache now (fail-fast / pre-warm-all).
        """
        instances_by_cls: Dict[int, Any] = {}
        for sel in candidates:
            cls_id = id(sel.cls)
            inst = instances_by_cls.get(cls_id)
            if inst is None:
                inst = run_mod.instantiate_class(sel.cls)
                instances_by_cls[cls_id] = inst
                self._instances.append(inst)
                self._setup_locks[id(inst)] = threading.Lock()
                # Stable per-class ModelCache key (the class qualname is unique
                # within an endpoint and maps 1:1 to one held instance/pipeline).
                self._model_id_by_inst[id(inst)] = (
                    f"{sel.cls.__module__}.{sel.cls.__qualname__}"
                )
            fn_name = sel.fn_name
            if fn_name in self.functions:
                # Function names are unique within an endpoint by contract
                # (#340). Defensive: surface the collision instead of silently
                # shadowing.
                raise run_mod._UsageError(
                    f"duplicate @inference.function name {fn_name!r} "
                    f"(hosted by {sel.cls.__name__}.{sel.attr_name})"
                )
            self.functions[fn_name] = _ServedFunction(sel, inst)
        if eager:
            for served in self.functions.values():
                self._ensure_resident(served)

    def _ensure_resident(self, served: "_ServedFunction") -> None:
        """Make the invoked function's model VRAM-resident via the ModelCache.

        First invoke of a class: run setup() ONCE to BUILD the pipeline, then
        register it in the cache — asking the cache for space first
        (``_evict_lru_for_space``, which demotes the LRU VRAM model to the
        CPU-RAM warm tier). Subsequent invokes: if the model was demoted to CPU
        (because another model needed the VRAM), promote it back
        (``_promote_from_cpu``, which itself demotes/evicts others as needed);
        otherwise just ``_touch`` it for LRU recency. setup() therefore runs
        exactly once per model; the cache moves models VRAM<->CPU thereafter.

        Static model bindings (Repo/HFRepo) are resolved + loaded inside
        setup(); Dispatch bindings (payload-dependent) still resolve per-request
        inside ``dispatch_request``.
        """
        inst = served.instance
        iid = id(inst)
        lock = self._setup_locks.setdefault(iid, threading.Lock())
        with lock:
            cache = self._model_cache
            model_id = self._model_id_by_inst.get(iid) or f"inst:{iid}"

            if iid not in self._setup_done:
                # COLD: build the pipeline via setup(), then register with the
                # cache so it participates in LRU residency from now on.
                resolved = _resolve_static_models(
                    served.selected.bindings, offline=self.offline,
                )
                run_mod.run_setup(inst, resolved)
                self._setup_done.add(iid)

                pipeline = _find_pipeline_object(inst)
                self._pipeline_by_inst[iid] = pipeline
                if cache is not None and pipeline is not None:
                    size_gb = _estimate_size_gb(pipeline)
                    # Make room BEFORE registering — demotes the LRU VRAM model
                    # to the CPU-RAM warm tier when this one won't otherwise fit.
                    try:
                        cache._evict_lru_for_space(size_gb)
                    except Exception as exc:  # noqa: BLE001
                        sys.stderr.write(
                            f"gen-worker serve: cache make-room failed for "
                            f"{model_id}: {exc}\n"
                        )
                    cache.mark_loaded_to_vram(model_id, pipeline, size_gb)
                    sys.stderr.write(
                        f"gen-worker serve: registered {model_id} in ModelCache "
                        f"(size={size_gb:.2f}GB, tier={cache.residency_tier(model_id)}, "
                        f"vram_models={cache.get_vram_models()})\n"
                    )
                    sys.stderr.flush()
                return

            # WARM: setup already ran. Ensure the model is back in VRAM.
            if cache is None:
                return
            tier = cache.residency_tier(model_id)
            if tier == "VRAM":
                cache._touch(model_id)
                return
            if tier == "RAM":
                # Demoted to the CPU-RAM warm tier on a prior eviction — promote
                # it back (evicting/demoting others as needed) for a fast PCIe
                # swap-in instead of re-running setup().
                sys.stderr.write(
                    f"gen-worker serve: promoting {model_id} RAM->VRAM "
                    f"(vram_models before={cache.get_vram_models()})\n"
                )
                ok = cache._promote_from_cpu(model_id, device="cuda")
                sys.stderr.write(
                    f"gen-worker serve: promote {model_id} ok={ok} "
                    f"tier={cache.residency_tier(model_id)} "
                    f"vram_models={cache.get_vram_models()}\n"
                )
                sys.stderr.flush()
            else:
                # ABSENT/DISK (dropped from RAM under memory pressure) — the
                # object is still held on the instance; re-register it (the
                # cache will make room first).
                pipeline = self._pipeline_by_inst.get(iid) or _find_pipeline_object(inst)
                if pipeline is not None:
                    size_gb = _estimate_size_gb(pipeline)
                    cache.mark_loaded_to_vram(model_id, pipeline, size_gb)
                    sys.stderr.write(
                        f"gen-worker serve: re-registered {model_id} in VRAM "
                        f"(tier={cache.residency_tier(model_id)})\n"
                    )
                    sys.stderr.flush()

    def function_names(self) -> List[str]:
        return sorted(self.functions.keys())

    def dispatch(
        self,
        function_name: str,
        payload_obj: Any,
        request_id: Optional[str] = None,
        on_event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run one request. Returns the (terminal) response envelope dict.

        Never raises — all errors are mapped into ``{"ok": false, "error": ...}``
        so the transport loop keeps serving. ``request_id`` registers the
        request's ctx in the cancellation registry so a concurrent cancel frame
        (or server SIGINT) can trip ``ctx.cancel()`` while the handler runs.

        ``on_event`` (streaming, #344): if given, each event is delivered to it
        as a frame ``{"event","value","request_id"}`` AS PRODUCED, and the return
        value is the terminal envelope (``{"ok":true,"done":true}`` or an error).
        If absent, events are buffered and returned in ``{"ok":true,"events":[...]}``.
        ``on_event`` may raise (e.g. on client disconnect) to abort the handler.
        """
        self.last_activity = time.time()  # reset the --idle-timeout clock
        served = self.functions.get(function_name)
        if served is None:
            return _error_envelope(
                "not_found",
                f"no function named {function_name!r}; available: "
                f"{self.function_names()}",
            )
        selected = served.selected

        raw_bytes = _encode_payload_bytes(payload_obj)

        events: List[Dict[str, Any]] = []

        def _write_event(kind: str, value: Any) -> None:
            body: Dict[str, Any] = {"event": kind}
            try:
                body["value"] = msgspec.to_builtins(value)
            except Exception:
                body["value"] = value
            if request_id:
                body["request_id"] = request_id
            if on_event is not None:
                on_event(body)  # stream as produced (may raise to abort)
            else:
                events.append(body)

        ctx = build_local_context(
            kind=selected.kind,
            allow_publish=self.allow_publish,
        )
        # Register BEFORE acquiring the dispatch lock so a cancel for a request
        # still queued behind a slow one trips its ctx too (the handler then
        # bails at its first raise_if_canceled / yield check).
        if request_id:
            with self._active_lock:
                self._active[request_id] = ctx

        try:
            # Serialize dispatch + acquire the GPU semaphore around the handler,
            # exactly like the production worker (sequential local handling).
            with self._dispatch_lock:
                self.gpu_semaphore.acquire()
                try:
                    # Drive residency through the ModelCache under the GPU
                    # semaphore (same as prod): first invoke builds the pipeline
                    # via setup() and registers it; later invokes promote it back
                    # into VRAM if the cache demoted it to the CPU-RAM warm tier.
                    self._ensure_resident(served)
                    run_mod.dispatch_request(
                        selected=selected,
                        instance=served.instance,
                        ctx=ctx,
                        raw_bytes=raw_bytes,
                        offline=self.offline,
                        emit=_stderr_emitter,
                        write_event=_write_event,
                        on_resolved=None,  # setup handled by _ensure_setup above
                    )
                except run_mod._UsageError as e:
                    return _error_envelope("usage", str(e))
                except run_mod._ModelResolutionError as e:
                    return _error_envelope("model_resolution", str(e))
                except CanceledError as e:
                    return _error_envelope("canceled", str(e))
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc(file=sys.stderr)
                    return _error_envelope("user_exception", str(e))
                finally:
                    try:
                        self.gpu_semaphore.release()
                    except ValueError:
                        pass
        finally:
            if request_id:
                with self._active_lock:
                    self._active.pop(request_id, None)

        if on_event is not None:
            return {"ok": True, "done": True}
        return {"ok": True, "events": events}

    def shutdown(self) -> None:
        """Call shutdown() on each instance that was actually set up (per #339).

        Lazy: an instance whose setup() never ran (its function was never
        invoked) has nothing to tear down — and its shutdown() may assume setup
        state — so skip it.
        """
        for inst in self._instances:
            if id(inst) not in self._setup_done:
                continue
            fn = getattr(inst, "shutdown", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass


def _resolve_static_models(
    bindings: Dict[str, Any], *, offline: bool,
) -> Dict[str, str]:
    """Resolve bindings that DON'T depend on the request payload, at boot.

    Dispatch bindings (which pick a model from a payload field) cannot be
    resolved without a request; they are skipped here and resolved per-request
    inside ``dispatch_request``. Static Repo / HFRepo / CivitaiRepo bindings
    are resolved so weights land in VRAM during ``setup()``.
    """
    from ..api.binding import CivitaiRepo, HFRepo, Repo

    static = {
        k: v for k, v in (bindings or {}).items()
        if isinstance(v, (HFRepo, CivitaiRepo, Repo))
    }
    if not static:
        return {}
    return run_mod._resolve_models_for_setup(
        bindings=static,
        payload=None,
        overrides={},
        offline=offline,
        emit=_stderr_emitter,
    )


# --------------------------------------------------------------------------
# ModelCache wiring — the SAME production residency manager, driven locally
# --------------------------------------------------------------------------

def _build_model_cache() -> Optional[ModelCache]:
    """Construct the production ``ModelCache`` sized from real local VRAM.

    Identical to the worker's ``ModelCache()`` except we size ``max_vram_gb``
    explicitly from ``inference_memory.get_total_vram_gb`` minus a margin so the
    local card's true budget governs residency (on an 8GB card two SDXL/SD1.5
    pipelines deliberately can't both stay VRAM-resident, forcing demote/promote
    through the cache). Returns ``None`` on a CPU-only host — there is no VRAM to
    manage, so serve falls back to plain warm-instance behavior.
    """
    try:
        from ..inference_memory import get_total_vram_gb

        total = float(get_total_vram_gb() or 0.0)
    except Exception:
        total = 0.0
    if total <= 0.0:
        return None
    try:
        return ModelCache(max_vram_gb=None)  # auto: total - default safety margin
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"gen-worker serve: ModelCache init failed: {exc}\n")
        return None


def _estimate_size_gb(pipeline: Any) -> float:
    """Best-effort VRAM footprint (GB) of a built pipeline, via inference_memory.

    Falls back to a conservative non-zero size so the cache still tracks the
    model (and can evict it) when the parameter probe can't see the weights
    (e.g. a fully offloaded pipeline reporting 0)."""
    try:
        from ..inference_memory import estimate_pipeline_size_gb

        size = float(estimate_pipeline_size_gb(pipeline) or 0.0)
    except Exception:
        size = 0.0
    if size <= 0.1:
        size = 5.0  # mirror worker.py's floor for un-probeable pipelines
    return size


# diffusers/transformers pipeline-ish duck type: has .to() AND at least one of
# the common module slots / a components mapping.
_PIPELINE_SLOTS = (
    "unet", "transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3",
)


def _looks_like_pipeline(obj: Any) -> bool:
    if obj is None:
        return False
    if not callable(getattr(obj, "to", None)):
        return False
    comps = getattr(obj, "components", None)
    if isinstance(comps, dict) and comps:
        return True
    return any(getattr(obj, slot, None) is not None for slot in _PIPELINE_SLOTS)


def _find_pipeline_object(instance: Any) -> Optional[Any]:
    """Find the diffusers pipeline an endpoint class built in ``setup()``.

    The convention (and what every stable-diffusion class uses) is to store the
    built pipeline on ``self.pipeline``; we prefer that. As a generic fallback
    for other endpoints we scan the instance ``__dict__`` for the first
    pipeline-shaped object. Returns ``None`` when the class holds no movable
    pipeline (then serve just keeps the instance warm without cache residency).
    """
    pref = getattr(instance, "pipeline", None)
    if _looks_like_pipeline(pref):
        return pref
    inst_dict = getattr(instance, "__dict__", {}) or {}
    for value in inst_dict.values():
        if _looks_like_pipeline(value):
            return value
    return None


# --------------------------------------------------------------------------
# Wire helpers
# --------------------------------------------------------------------------

def _error_envelope(kind: str, message: str) -> Dict[str, Any]:
    return {"ok": False, "error": {"kind": kind, "message": message}}


def _encode_payload_bytes(payload_obj: Any) -> bytes:
    """Re-serialize the decoded JSON payload to bytes for dispatch_request,
    which expects raw JSON bytes (it strips ``_models`` and msgspec-decodes).
    """
    if payload_obj is None:
        return b"{}"
    return json.dumps(payload_obj, separators=(",", ":")).encode("utf-8")


def _parse_frame(line: bytes) -> Dict[str, Any]:
    """Parse one NDJSON frame into a tagged dict.

    Kinds:
      - ``{"kind":"request","function","payload","request_id"}``
      - ``{"kind":"cancel","request_id"}``  (control frame: ``{"cancel":{...}}``)
      - ``{"kind":"error","message"}``
    """
    try:
        obj = json.loads(line.decode("utf-8"))
    except Exception as e:
        return {"kind": "error", "message": f"request is not valid JSON: {e}"}
    if not isinstance(obj, dict):
        return {"kind": "error", "message": "request must be a JSON object"}
    if "cancel" in obj:
        c = obj.get("cancel") or {}
        rid = c.get("request_id") if isinstance(c, dict) else None
        return {"kind": "cancel", "request_id": rid}
    fn = obj.get("function")
    if not isinstance(fn, str) or not fn:
        return {"kind": "error", "message": "request.function (string) is required"}
    return {
        "kind": "request",
        "function": fn,
        "payload": obj.get("payload", {}),
        "request_id": obj.get("request_id"),
        "stream": bool(obj.get("stream")),
    }


def _write_response_line(write: Any, envelope: Dict[str, Any]) -> None:
    line = json.dumps(envelope, separators=(",", ":"), default=str)
    write(line + "\n")


# --------------------------------------------------------------------------
# Transports
# --------------------------------------------------------------------------

def _serve_stdin(endpoint: _Endpoint, stop: threading.Event) -> None:
    """Read NDJSON requests from stdin, write NDJSON results to stdout."""
    def _emit_stdout(s: str) -> None:
        sys.stdout.write(s)
        sys.stdout.flush()

    for raw in sys.stdin.buffer:
        if stop.is_set():
            break
        line = raw.strip()
        if not line:
            continue
        frame = _parse_frame(line)
        if frame["kind"] == "error":
            _write_response_line(_emit_stdout, _error_envelope("usage", frame["message"]))
            continue
        if frame["kind"] == "cancel":
            # stdin requests are sequential — nothing is in-flight to cancel.
            found = endpoint.interrupt_request(frame.get("request_id"))
            _write_response_line(_emit_stdout, {"ok": True, "canceled": found})
            continue
        rid = frame.get("request_id")
        if frame.get("stream"):
            terminal = endpoint.dispatch(
                frame["function"], frame["payload"], request_id=rid,
                on_event=lambda ev: _write_response_line(_emit_stdout, ev),
            )
            _write_response_line(_emit_stdout, terminal)
        else:
            envelope = endpoint.dispatch(frame["function"], frame["payload"], request_id=rid)
            _write_response_line(_emit_stdout, envelope)


def _sidecar_path(listen_spec: str) -> Path:
    """Where the machine-readable serve handle lives (#349).

    Adjacent to the Unix socket (``<sock>.json``) so several serves don't
    collide; in cwd for TCP (no socket file).
    """
    addr = transport.parse_addr(listen_spec)
    if addr[0] == "unix":
        return Path(str(addr[1]) + ".json")
    return Path.cwd() / ".gen-worker.serve.json"


def _write_sidecar(listen_spec: str, endpoint: _Endpoint, idle_timeout: float) -> Path:
    """Write the serve sidecar so a host (cozy) reads pid/listen/functions
    instead of reimplementing pidfile/socket/ready guesswork."""
    path = _sidecar_path(listen_spec)
    doc = {
        "protocol_version": PROTOCOL_VERSION,
        "gen_worker_version": gen_worker_version(),
        "pid": os.getpid(),
        "listen": transport.display(listen_spec),
        "ready_at": time.time(),
        "idle_timeout": idle_timeout or 0.0,
        "functions": endpoint.function_names(),
    }
    try:
        path.write_text(json.dumps(doc), encoding="utf-8")
    except OSError:
        pass
    return path


def _serve_socket(
    endpoint: _Endpoint, listen_spec: str, stop: threading.Event,
    idle_timeout: float = 0.0,
) -> None:
    """Accept connections (Unix socket or TCP); one NDJSON request each.

    Each connection is handled on its own thread so the accept loop stays free
    to receive a cancel control frame WHILE a request runs.
    """
    srv = transport.create_listener(listen_spec, backlog=8)
    srv.settimeout(0.5)  # so we can poll `stop`

    sidecar = _write_sidecar(listen_spec, endpoint, idle_timeout)
    sys.stderr.write(
        f"gen-worker serve: listening on {transport.display(listen_spec)} "
        f"(functions: {', '.join(endpoint.function_names())})\n"
    )
    sys.stderr.write("gen-worker serve: ready\n")
    sys.stderr.flush()

    try:
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(
                target=_handle_conn_safe, args=(endpoint, conn), daemon=True,
            )
            t.start()
    finally:
        try:
            srv.close()
        finally:
            transport.cleanup_listener(listen_spec)
            try:
                sidecar.unlink()
            except OSError:
                pass


def _handle_conn_safe(endpoint: _Endpoint, conn: socket.socket) -> None:
    """Thread target: handle one connection, always closing the socket."""
    try:
        _handle_conn(endpoint, conn)
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _handle_conn(endpoint: _Endpoint, conn: socket.socket) -> None:
    """Read one NDJSON frame; dispatch a request OR apply a cancel; respond.

    A cancel control frame (``{"cancel":{"request_id"}}``) is the CLI analog of
    the orchestrator's ``interrupt_job_cmd``: it trips ``ctx.cancel()`` for the
    named in-flight request on ANOTHER connection's dispatch thread and returns
    immediately — the server keeps running (#352).
    """
    buf = bytearray()
    # Bound only the request-LINE read (the client sends it immediately); the
    # long cold-model wait happens later inside dispatch(), not here.
    conn.settimeout(30.0)
    try:
        while b"\n" not in buf:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf.extend(chunk)
    except (socket.timeout, OSError):
        return
    if not buf:
        return
    line = bytes(buf).split(b"\n", 1)[0].strip()
    if not line:
        return
    frame = _parse_frame(line)

    def _send(env: Dict[str, Any]) -> None:
        conn.sendall((json.dumps(env, separators=(",", ":"), default=str) + "\n").encode("utf-8"))

    if frame["kind"] == "cancel":
        found = endpoint.interrupt_request(frame.get("request_id"))
        try:
            _send({"ok": True, "canceled": found, "request_id": frame.get("request_id")})
        except OSError:
            pass
        return
    if frame["kind"] == "error":
        try:
            _send(_error_envelope("usage", frame["message"]))
        except OSError:
            pass
        return

    rid = frame.get("request_id")
    if frame.get("stream"):
        # Stream each event as its own frame as produced; a write failure means
        # the client is gone -> cancel the in-flight request (the disconnect
        # backstop deferred from #352) and let the handler unwind.
        def _on_event(ev: Dict[str, Any]) -> None:
            try:
                _send(ev)
            except OSError:
                endpoint.interrupt_request(rid)
                raise CanceledError("client disconnected")

        terminal = endpoint.dispatch(
            frame["function"], frame["payload"], request_id=rid, on_event=_on_event,
        )
        if rid is not None:
            terminal.setdefault("request_id", rid)
        try:
            _send(terminal)
        except OSError:
            pass
        return

    envelope = endpoint.dispatch(frame["function"], frame["payload"], request_id=rid)
    if rid is not None:
        envelope.setdefault("request_id", rid)
    try:
        _send(envelope)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Handler
# --------------------------------------------------------------------------

def _drain_active(endpoint: _Endpoint, timeout: float) -> None:
    """Wait (bounded) for in-flight requests to unwind after cancel_all().

    Cooperative handlers return promptly once canceled; a handler that ignores
    cancellation is bounded by ``timeout`` so teardown still proceeds.
    """
    deadline = time.time() + max(0.0, timeout)
    while time.time() < deadline:
        with endpoint._active_lock:
            if not endpoint._active:
                return
        time.sleep(0.05)


def _handle_serve(args: argparse.Namespace) -> int:
    try:
        return _serve_inner(args)
    except run_mod._UsageError as e:
        sys.stderr.write(f"gen-worker serve: {e}\n")
        return run_mod.EXIT_USAGE
    except run_mod._ModelResolutionError as e:
        sys.stderr.write(f"gen-worker serve: model resolution failed: {e}\n")
        return run_mod.EXIT_MODEL_RESOLUTION
    except KeyboardInterrupt:
        sys.stderr.write("gen-worker serve: interrupted\n")
        return run_mod.EXIT_OK


def _serve_inner(args: argparse.Namespace) -> int:
    # 1. Load + discover.
    _root, mod = run_mod.load_endpoint_module(
        config_path=args.config_path, module=args.module,
    )
    candidates = run_mod.discover_candidates(mod)

    # --list-functions: print routable names + hosting class, exit; no boot.
    if getattr(args, "list_functions", False):
        if getattr(args, "json_output", False):
            # Thin alias of `gen-worker describe`'s functions array — one shared
            # builder, one shape (#349).
            from .describe import function_entries

            sys.stdout.write(
                json.dumps({"functions": function_entries(candidates)}) + "\n"
            )
            sys.stdout.flush()
            return run_mod.EXIT_OK
        rows = sorted(
            ((c.fn_name, getattr(c.cls, "__name__", "?")) for c in candidates),
            key=lambda r: r[0],
        )
        if not rows:
            sys.stderr.write("gen-worker serve: no functions found in endpoint\n")
            return run_mod.EXIT_OK
        width = max(len(name) for name, _ in rows)
        for name, cls_name in rows:
            sys.stdout.write(f"{name.ljust(width)}  ({cls_name})\n")
        sys.stdout.flush()
        return run_mod.EXIT_OK

    # --function NAME: boot only the class(es) hosting the named function(s),
    # so a multi-model endpoint doesn't load every class/model.
    candidates = _filter_candidates_by_function(
        candidates, getattr(args, "functions", None),
    )

    if args.device:
        os.environ["GEN_WORKER_LOCAL_DEVICE"] = args.device

    # 2. Boot the endpoint — setup() once per class, hold instances warm.
    endpoint = _Endpoint(
        offline=bool(args.offline),
        allow_publish=bool(args.allow_publish),
    )
    endpoint.boot(candidates, eager=bool(getattr(args, "eager", False)))

    # Listen address: --listen overrides --socket. Resolve a Unix path to an
    # absolute form so teardown unlinks the right file regardless of cwd.
    listen_spec = getattr(args, "listen", None) or args.socket_path
    if transport.is_unix(listen_spec):
        _, _p = transport.parse_addr(listen_spec)
        listen_spec = str(Path(_p).resolve())
    stop = threading.Event()

    # 3. SIGINT / SIGTERM -> cancel in-flight requests, then clean teardown.
    #    Stopping the worker and cancelling a request both funnel through the
    #    same ctx.cancel() (#353): here we cancel ALL in-flight requests so each
    #    tenant handler unwinds cooperatively, then drain + shutdown(). SIGTERM
    #    (k8s/orchestrator graceful stop) maps to the identical drain path.
    #    SIGKILL is uncatchable — it bypasses all cleanup (no shutdown / GPU
    #    free); the bounded drain below exists so SIGKILL is rarely needed.
    def _on_signal(signum: int, _frame: Any) -> None:
        name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        n = endpoint.cancel_all()
        sys.stderr.write(
            f"\ngen-worker serve: received {name}; canceling {n} in-flight "
            "request(s) and shutting down\n"
        )
        sys.stderr.flush()
        stop.set()

    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGINT, _on_signal)
    except (ValueError, OSError):
        prev_int = None
    try:
        signal.signal(signal.SIGTERM, _on_signal)
    except (ValueError, OSError):
        prev_term = None

    idle_timeout = float(getattr(args, "idle_timeout", 0.0) or 0.0)

    # 4. Run the socket listener on a background thread; the foreground either
    #    reads stdin (default) or blocks until stop (--no-stdin / detached).
    sock_thread = threading.Thread(
        target=_serve_socket, args=(endpoint, listen_spec, stop, idle_timeout),
        daemon=True,
    )
    sock_thread.start()

    # Read interactive NDJSON from stdin ONLY when stdin is a real TTY. When
    # stdin is backgrounded / redirected (`serve &`, `serve </dev/null`) or a
    # pipe, treat it as detached and serve the socket until SIGINT — otherwise
    # an immediate stdin EOF would tear the socket down before any `invoke`
    # could connect (the common footgun).
    detached = args.no_stdin or not getattr(sys.stdin, "isatty", lambda: True)()
    # The idle clock starts now (after boot), so model load time isn't counted.
    endpoint.last_activity = time.time()
    try:
        if detached:
            while not stop.is_set():
                stop.wait(0.5)
                if idle_timeout > 0 and (time.time() - endpoint.last_activity) > idle_timeout:
                    sys.stderr.write(
                        f"gen-worker serve: idle for >{idle_timeout:g}s; shutting down\n"
                    )
                    sys.stderr.flush()
                    stop.set()
        else:
            _serve_stdin(endpoint, stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        # Cancel any in-flight requests (idempotent with the signal handler) and
        # give them a bounded window to unwind cooperatively before teardown.
        endpoint.cancel_all()
        _drain_active(endpoint, timeout=10.0)
        sock_thread.join(timeout=2.0)
        for sig, prev in ((signal.SIGINT, prev_int), (signal.SIGTERM, prev_term)):
            if prev is not None:
                try:
                    signal.signal(sig, prev)
                except Exception:
                    pass
        endpoint.shutdown()
        # Belt-and-suspenders: ensure a Unix socket file is gone (no-op for TCP).
        transport.cleanup_listener(listen_spec)

    sys.stderr.write("gen-worker serve: stopped\n")
    sys.stderr.flush()
    return run_mod.EXIT_OK
