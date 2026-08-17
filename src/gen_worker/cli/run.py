"""``gen-worker run`` — execute one endpoint function against a local Python
interpreter.

Two inputs: which function to call, what payload to send. Everything else
(model resolution, payload validation, context wiring) derives from the
``@endpoint`` declarations and pyproject's ``[tool.gen_worker]`` — exactly
the way the production worker does it.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import signal
import sys
import time
import traceback
import types
import typing
import uuid
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple,
)

import msgspec

if TYPE_CHECKING:  # the arming import stays deferred at runtime —
    # `gen_worker.local_serve` pulls the whole compile stack, and `cozy run`
    # on a payload-only endpoint must not pay for it.
    from .. import local_serve

from ..api.errors import CanceledError
from .. import serve_posture
from ..models import provision
from .local_context import build_local_context
from ..discovery.project import load_project_config
from gen_worker.registry import collect_from_namespace
from .args import ArgError, build_payload
from ..api.slot import resolve_slot
from ..models import execution_lanes as lanespec
from ..api.binding import rebind_pick
from .sockaddr import DEFAULT_SOCKET_PATH
from . import invoke as invoke_mod
from .local_context import _stderr_emitter
from ..hostfacts import cuda_ready


# Exit codes.
EXIT_OK = 0
EXIT_USER_EXCEPTION = 1
EXIT_USAGE = 2
EXIT_MODEL_RESOLUTION = 3
EXIT_SIGINT = 130


# --------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------

def add_subparser(sub: argparse._SubParsersAction[Any]) -> None:
    """Register the ``run`` subcommand on the top-level parser."""
    p = sub.add_parser(
        "run",
        help="Run an endpoint method against a local JSON payload.",
        description=(
            "Execute one endpoint function in the local Python "
            "interpreter against a JSON payload. Mirrors production behavior "
            "for model resolution, payload validation, and context wiring. "
            "Result on stdout (msgspec-JSON encoded); events on stderr "
            "(JSON lines)."
        ),
    )
    p.add_argument(
        "--class", dest="cls_name", default=None,
        help="Class name to invoke (inferred when only one is registered).",
    )
    p.add_argument(
        "--method", dest="method_name", default=None,
        help="Method name to invoke (inferred when only one is registered).",
    )
    p.add_argument(
        "--payload", dest="payload", default=None,
        help="Inline JSON payload string. Mutually exclusive with --payload-file.",
    )
    p.add_argument(
        "--payload-file", dest="payload_file", default=None,
        help="Read JSON payload from this file. Mutually exclusive with --payload.",
    )
    p.add_argument(
        "--config", dest="config_path", default=None,
        help="Path to the endpoint's pyproject.toml (defaults to ./pyproject.toml).",
    )
    p.add_argument(
        "--module", dest="module", default=None,
        help="Python module path to import (overrides [tool.gen_worker] main).",
    )
    p.add_argument(
        "--offline", action="store_true",
        help=(
            "Use only the local CAS — fail with exit 3 instead of fetching "
            "missing model weights from tensorhub / huggingface."
        ),
    )
    p.add_argument(
        "--eager-only", dest="eager_only", action="store_true",
        help=(
            "Serve EAGER ONLY: never arm a compiled cell and never mint one "
            "(DESIGN-RULINGS §4.32). One invocation is where a cold machine "
            "would otherwise spend minutes minting."
        ),
    )
    p.add_argument(
        "--device", dest="device", default=None,
        help="Override the torch device (e.g. 'cuda:0', 'cpu').",
    )
    p.add_argument(
        "--allow-publish", action="store_true",
        help=(
            "Allow ConversionContext.materialize_blob to call the real "
            "tensorhub APIs. Default: stubbed against the local CAS."
        ),
    )
    p.add_argument(
        "--source-path", dest="source_path", default=None,
        help=(
            "Local snapshot directory to use as the reserved `source` "
            "payload's materialized checkpoint: populates ctx.source_path "
            "exactly like the production worker does after Hub-resolve, so "
            "reserved-source producer functions run without a hub."
        ),
    )
    p.add_argument(
        "--lane", dest="execution_lane", default="",
        help=(
            "Execution lane (th#913 dual-form): a family 'bf16'|'fp8', or a "
            "full descriptor like 'fp8-w8a8-dynamic+compiled'. Default: auto "
            "(the binding as declared). 'fp8' maps to the local cast lane. "
            "Stored-quant lanes (fp8-w8a8, 4bit) are not selectable locally: "
            "§1.32(d) deleted the within-tag-group selector, so bind the "
            "checkpoint you mean by digest ('owner/repo@sha256:<hex>')."
        ),
    )
    p.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the stdout result with newlines + 2-space indent.",
    )
    p.add_argument(
        "--attach", action="store_true",
        help=(
            "Dispatch through a warm `gen-worker serve` socket when one is "
            "listening (endpoint already loaded, setup() already run)."
        ),
    )
    p.add_argument(
        "--list", dest="list_functions", action="store_true",
        help="Print a JSON description of the endpoint's functions and exit "
             "(no model load).",
    )
    p.add_argument(
        "fields", nargs="*", metavar="FIELD=VALUE",
        help=(
            "Ergonomic payload args instead of (or merged over) --payload: "
            "'field=value' (coerced to the field's type), 'field:=<json>' (raw "
            "JSON), 'field@path' (from file), or a bare value for the primary "
            "field. E.g. gen-worker run \"a cat\" seed=42 hires=true"
        ),
    )
    p.set_defaults(_handler=_handle_run)


# --------------------------------------------------------------------------
# endpoint.toml + module loading
# --------------------------------------------------------------------------

def _load_project_main(config_path: Optional[str]) -> Tuple[Path, str]:
    """Return ``(project_root, main_module)`` from pyproject [tool.gen_worker]."""

    try:
        cfg = load_project_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        raise _UsageError(str(e)) from e
    return cfg.root, cfg.main


def _ensure_sys_path(root: Path) -> None:
    """Match discover.py's sys.path priming so relative imports resolve."""
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


# --------------------------------------------------------------------------
# function discovery — collect live class+method bindings (not just manifest)
# --------------------------------------------------------------------------

class _SelectedFunction:
    """Live handle to one selected endpoint function.

    Carries the actual class object, bound method, payload type, and the
    endpoint-level binding map (``models={...}``). The run pipeline uses
    these to instantiate the class, call ``setup(**resolved_models)``, and
    dispatch with ``(ctx, payload)``.
    """

    def __init__(
        self,
        *,
        cls: Optional[type],
        attr_name: str,
        method: Callable[..., Any],
        fn_name: str,
        kind: str,
        payload_type: type,
        output_type: Optional[type],
        is_generator: bool,
        bindings: Dict[str, Any],
        resources: Any = None,
        slots: Optional[Dict[str, Any]] = None,
        handles: Tuple[str, ...] = (),
        spec: Any = None,
    ) -> None:
        self.cls = cls
        self.attr_name = attr_name
        self.method = method
        self.fn_name = fn_name
        self.kind = kind
        self.payload_type = payload_type
        self.output_type = output_type
        self.is_generator = is_generator
        self.bindings = bindings
        self.resources = resources
        self.slots = slots or {}
        self.handles = tuple(handles)
        # The full EndpointSpec (defaults_type / slot_family / payload_axes)
        # for the hub-less resolution chain.
        self.spec = spec


def _collect_class_methods(mod: Any) -> List[_SelectedFunction]:
    """Collect every routable function on every @endpoint object in a module
    namespace. Delegates signature inspection to ``gen_worker.registry`` — the
    one walker shared with the worker runtime and build-time discovery."""

    return [
        _SelectedFunction(
            cls=es.cls,
            attr_name=es.attr_name,
            method=es.method,
            fn_name=es.name,
            kind=es.kind,
            payload_type=es.payload_type,
            output_type=es.output_type,
            is_generator=es.output_mode == "stream",
            bindings=dict(es.models),
            resources=es.resources,
            slots=dict(es.slots),
            handles=es.handles,
            spec=es,
        )
        for es in collect_from_namespace(mod)
    ]


def _select_function(
    candidates: List[_SelectedFunction],
    *,
    cls_name: Optional[str],
    method_name: Optional[str],
    default_name: Optional[str] = None,
) -> _SelectedFunction:
    """Filter candidates by --class / --method and exit-2 on ambiguity.

    ``default_name`` (the endpoint package name) breaks a no-filter tie when
    exactly one candidate's function name matches it — so an endpoint whose
    primary function is named after the package runs with zero flags.
    """
    def _label(c: _SelectedFunction) -> str:
        owner = c.cls.__name__ + "." + c.attr_name if c.cls is not None else "<function>"
        return f"{owner} (fn_name={c.fn_name!r})"

    matches = list(candidates)
    if cls_name:
        matches = [
            c for c in matches
            if c.cls is not None and c.cls.__name__ == cls_name
        ]
    if method_name:
        # fn_name is the canonical slug; accept the python attr name or
        # either slug spelling.
        from gen_worker.discovery.names import slugify_name

        wanted_slug = slugify_name(method_name)
        matches = [
            c for c in matches
            if c.attr_name == method_name or c.fn_name == wanted_slug
        ]

    if not matches:
        available = "\n  - " + "\n  - ".join(_label(c) for c in candidates) if candidates else " (none)"
        filt_desc = []
        if cls_name:
            filt_desc.append(f"--class={cls_name!r}")
        if method_name:
            filt_desc.append(f"--method={method_name!r}")
        filt_msg = " ".join(filt_desc) or "(no filters)"
        raise _UsageError(
            f"no endpoint function matches {filt_msg}.\n"
            f"available:{available}"
        )
    if len(matches) > 1:
        from gen_worker.discovery.names import slugify_name

        if method_name:
            # An exact fn_name match wins over attr-name matches.
            exact = [m for m in matches if m.fn_name == slugify_name(method_name)]
            if len(exact) == 1:
                return exact[0]
        if not cls_name and not method_name and default_name:
            wanted = slugify_name(default_name)
            defaults = [m for m in matches if m.fn_name == wanted]
            if len(defaults) == 1:
                return defaults[0]
        listing = "\n  - " + "\n  - ".join(_label(c) for c in matches)
        raise _UsageError(
            f"ambiguous: {len(matches)} methods match the given filters; "
            f"specify --class and/or --method.\nmatches:{listing}"
        )
    return matches[0]


# --------------------------------------------------------------------------
# Payload decoding (+ _models override + payload constraints)
# --------------------------------------------------------------------------

def _read_payload_bytes(
    *, inline: Optional[str], path: Optional[str],
) -> bytes:
    if inline is not None and path is not None:
        raise _UsageError("--payload and --payload-file are mutually exclusive")
    if inline is None and path is None:
        # Empty payload: tenants with payload-required fields will get a
        # msgspec.ValidationError that already names the missing field path.
        return b"{}"
    if inline is not None:
        return inline.encode("utf-8")
    p = Path(path)  # type: ignore[arg-type]
    if not p.exists():
        raise _UsageError(f"--payload-file not found: {p}")
    return p.read_bytes()


def _apply_field_tokens(
    raw_bytes: bytes, fields: Optional[List[str]], payload_type: type,
) -> bytes:
    """Merge ergonomic ``field=value`` tokens over the base JSON payload.

    No tokens -> ``raw_bytes`` unchanged. Coercion uses ``payload_type`` so
    types/bounds match the real decode. Raises ``_UsageError`` on a bad token.
    """
    if not fields:
        return raw_bytes

    try:
        base = json.loads(raw_bytes.decode("utf-8") or "{}")
    except json.JSONDecodeError as e:
        raise _UsageError(f"--payload is not valid JSON: {e}") from e
    if not isinstance(base, dict):
        base = {}
    try:
        merged = build_payload(list(fields), payload_type, base=base)
    except ArgError as e:
        raise _UsageError(str(e)) from e
    return json.dumps(merged, separators=(",", ":")).encode("utf-8")


def _inject_default_source(
    raw_bytes: bytes, payload_type: type, source_path: Path,
) -> bytes:
    """When ``--source-path`` is given and the payload type declares a
    reserved ``source`` field the user didn't fill in, synthesize a local ref
    so decode passes. An explicit ``source`` in the payload is left alone."""
    try:
        field_names = {f.name for f in msgspec.structs.fields(payload_type)}
    except Exception:
        return raw_bytes
    if "source" not in field_names:
        return raw_bytes
    try:
        base = json.loads(raw_bytes.decode("utf-8") or "{}") if raw_bytes else {}
    except json.JSONDecodeError:
        return raw_bytes
    if not isinstance(base, dict) or base.get("source"):
        return raw_bytes
    base["source"] = {"ref": f"local/{source_path.name}"}
    return json.dumps(base, separators=(",", ":")).encode("utf-8")


def _decode_payload(
    payload_bytes: bytes, payload_type: type,
) -> Any:
    """Decode the JSON payload into the typed msgspec.Struct.

    The decode IS the whole contract: msgspec's own validation is what the
    live worker runs (`executor` decodes with the same `payload_type`), so
    there is deliberately no post-decode clamping pass here.
    """
    try:
        decoded: Any = msgspec.json.decode(payload_bytes, type=payload_type)
    except msgspec.ValidationError as e:
        # msgspec.ValidationError already carries field path + expected type;
        # re-raise as a usage error so the cli sets exit 2.
        raise _UsageError(f"payload validation failed: {e}") from e
    except msgspec.DecodeError as e:
        raise _UsageError(f"payload is not valid JSON: {e}") from e
    return decoded


# --------------------------------------------------------------------------
# Model resolution — the shared hub-less resolver (models/provision.py)
# --------------------------------------------------------------------------

# Exit-3 error type; aliased so serve/invoke keep one import site.
_ModelResolutionError = provision.ModelResolutionError


# --------------------------------------------------------------------------
# SIGINT handling
# --------------------------------------------------------------------------

class _SigintHandler:
    """Install a two-stage SIGINT handler.

    First Ctrl-C: trip cancellation so user code observes via
    ``ctx.cancelled / raise_if_cancelled()``.
    Second Ctrl-C within 2s: hard-exit 130.
    """

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self._last_at: float = 0.0
        self._prev_handler: Any = signal.SIG_DFL
        self._installed = False

    def install(self) -> None:
        try:
            self._prev_handler = signal.signal(signal.SIGINT, self._on_sigint)
            self._installed = True
        except (ValueError, OSError):
            # Non-main thread or signal not supported — best-effort.
            self._installed = False

    def restore(self) -> None:
        if not self._installed:
            return
        try:
            signal.signal(signal.SIGINT, self._prev_handler)
        except Exception:
            pass
        self._installed = False

    def _on_sigint(self, _signum: int, _frame: types.FrameType | None) -> None:
        now = time.monotonic()
        if self._last_at and (now - self._last_at) < 2.0:
            # Second hit within 2s — hard exit.
            sys.stderr.write(
                "\ngen-worker run: received SIGINT again; aborting (exit 130).\n"
            )
            sys.stderr.flush()
            os._exit(EXIT_SIGINT)
        self._last_at = now
        sys.stderr.write(
            "\ngen-worker run: received SIGINT; signaling cooperative "
            "cancellation. Press Ctrl-C again within 2s to hard-exit.\n"
        )
        sys.stderr.flush()
        try:
            # Same entry point the production worker uses for an
            # orchestrator cancel.
            self._ctx._cancel()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Result encoding
# --------------------------------------------------------------------------

def _write_stdout_event(event_kind: str, value: Any, *, pretty: bool) -> None:
    """Emit one ``{"event": <kind>, "value": ...}`` line on stdout."""
    body: Dict[str, Any] = {"event": event_kind}
    try:
        body["value"] = msgspec.to_builtins(value)
    except Exception:
        body["value"] = value
    if pretty:
        line = json.dumps(body, indent=2, default=str)
    else:
        line = json.dumps(body, separators=(",", ":"), default=str)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class _UsageError(Exception):
    """User-supplied argument was invalid. Maps to exit 2."""


# --------------------------------------------------------------------------
# Shared loading + dispatch (reused by `run` and `serve`)
# --------------------------------------------------------------------------

def load_endpoint_module(
    *, config_path: Optional[str], module: Optional[str],
) -> Tuple[Path, types.ModuleType]:
    """Resolve pyproject [tool.gen_worker] (or ``--module``), prime sys.path, import ``main``.

    Returns ``(project_root, imported_module)``. Shared by ``run`` and
    ``serve`` so both discover endpoint classes the same way. Raises
    ``_UsageError`` on import / config failure (exit 2).
    """
    if module:
        root = Path.cwd().resolve()
        main_module = module
    else:
        root, main_module = _load_project_main(config_path)
    _ensure_sys_path(root)
    try:
        mod = importlib.import_module(main_module)
    except Exception as e:
        raise _UsageError(
            f"failed to import module {main_module!r}: {e}"
        ) from e
    return root, mod


def discover_candidates(mod: types.ModuleType) -> List[_SelectedFunction]:
    """Collect every routable function, exit-2 if none exist."""
    candidates = _collect_class_methods(mod)
    if not candidates:
        raise _UsageError(
            f"no @endpoint functions or classes found in module "
            f"{getattr(mod, '__name__', '?')!r}"
        )
    return candidates


def instantiate_class(cls: Optional[type]) -> Any:
    """Instantiate a discovered endpoint class (no setup yet); None for
    function-shaped endpoints."""
    return cls() if cls is not None else None


def run_setup(
    instance: Any, resolved_models: Dict[str, str], *, device: str = "",
    arm_compile: bool = True, return_loaded: bool = False,
    structure_only: Sequence[str] = (), place: bool = True,
    selected: Optional["_SelectedFunction"] = None,
) -> Optional[Dict[str, Any]]:
    """Call ``instance.setup(...)`` once, passing exactly the resolved model
    slots its signature declares.

    ``arm_compile=False`` loads the slots WITHOUT arming any compiled path: the
    mint child drives its own cold arm + capture and must not have a cell armed
    under it. ``return_loaded=True`` returns the loaded slot objects so a caller
    can reach the pipeline it just built.

    ``selected`` is the routable function these slots belong to, and it is what
    makes ``arm_compile=True`` able to MINT: a delegated child rediscovers the
    endpoint from its declaring module and re-resolves the parent's slots.
    Absent, the arm is adopt-only and says so.

    ``place=False`` loads the slots without running the worker's serving
    placement ladder at all, while ``device`` still says which device the
    virtual structure is BUILT on — the two are one knob only for a load that
    intends to serve. The boot-trace child sets it: it needs shapes and a graph,
    never a servable pipeline, and the ladder would move the slot's REAL
    non-target components onto the card its serving parent owns.

    ``structure_only`` names the components every slot must build from CODE +
    CONFIG instead of from the checkpoint — the mint child's compile targets.
    They are constructed by ``models.structure_only.build_component`` and
    injected through the same ``components=`` seam a preloaded shared component
    uses, so the pipeline the endpoint receives is composed by its own loader
    either way. A component the slot's tree does not declare is skipped (a
    two-slot family's auxiliary slot has no denoiser); a component that CANNOT
    be built from config refuses typed (``StructureOnlyUnsupported``) rather
    than silently loading its weights.

    A bare ``def setup(self)`` is legitimate (model-selectable endpoints receive
    their model per-request in the handler instead), so unclaimed slots are
    fine. But a setup() that declares parameters we CANNOT satisfy is an
    authoring error and must fail loudly rather than being swallowed by a
    blanket ``except TypeError: setup_fn()``.
    """
    setup_fn = getattr(instance, "setup", None)
    if instance is None or setup_fn is None:
        return {} if return_loaded else None
    try:
        params = inspect.signature(setup_fn).parameters
    except (TypeError, ValueError):
        # Odd callables without an inspectable signature (decorated,
        # C-implemented, wrapped). pgw#1307: this used to swallow the TypeError
        # and re-call `setup_fn()` with NO models, silently dropping every
        # resolved slot — the exact blanket arm this function's docstring
        # forbids. A setup we cannot satisfy is an authoring error; let it
        # raise, naming the models we resolved.
        setup_fn(**resolved_models)
        return {} if return_loaded else None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        setup_fn(**resolved_models)
        return {} if return_loaded else None
    wanted = {
        name for name, p in params.items()
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    missing = sorted(
        name for name, p in params.items()
        if name in wanted
        and p.default is inspect.Parameter.empty
        and name not in resolved_models
    )
    if missing:
        raise _UsageError(
            f"setup() declares parameter(s) {missing} but the resolved model "
            f"slots are {sorted(resolved_models) or '(none)'}; each setup "
            f"parameter must match a key in the @endpoint models={{...}} "
            f"dict (static bindings are injected into setup; dispatch "
            f"bindings are injected into the handler per-request)."
        )
    # Each claimed slot loads through the production core (provision.load_slot):
    # a setup(pipeline: StableDiffusionXLPipeline) receives a constructed +
    # placed pipeline, str/Path slots receive the snapshot path.
    try:
        hints = typing.get_type_hints(setup_fn)
    except (TypeError, ValueError, NameError):
        hints = {}
    decl = getattr(type(instance), _ENDPOINT_ATTR, None)
    # The mint context is built from the WHOLE resolution, once, before any slot
    # loads — the child re-runs setup, so it needs every slot this endpoint
    # declares and not just the one whose load happens to reach the arm. Built
    # even when `arm_compile` is False so the two paths cannot drift into
    # resolving slots differently.
    mint = _local_mint_context(
        instance, resolved_models, wanted, selected=selected)
    # A mint this load OWES is collected, not run. Arming from the local store
    # still happens inside the load (the endpoint's `setup()` must receive an
    # armed pipeline), but the MINT waits until `setup()` and `warmup()` have
    # returned, because proving the endpoint's handler needs an instance whose
    # pipeline is bound. Same order the fleet boots in.
    deferred: List[Any] = []
    loaded = {
        k: _load_injected_model(
            hints.get(k), v, decl=decl, slot=k, device=device,
            arm_compile=arm_compile, place=place,
            structure_only=tuple(structure_only), mint=mint,
            defer=deferred)
        for k, v in resolved_models.items()
        if k in wanted
    }
    setup_fn(**loaded)
    # warmup() runs once after setup, as in the production worker.
    warmup_fn = getattr(instance, "warmup", None)
    if callable(warmup_fn):
        warmup_fn()
    if deferred and selected is not None:
        from .. import local_serve

        local_serve.run_deferred(
            deferred, instance=instance, specs=[selected.spec])
    return loaded if return_loaded else None


_INJECTED_CACHE: Dict[Tuple[str, str], Any] = {}
_ENDPOINT_ATTR = "__gen_worker_endpoint__"


def _local_mint_context(
    instance: Any,
    resolved_models: Dict[str, str],
    wanted: Any,
    *,
    selected: Optional[_SelectedFunction],
) -> Optional["local_serve.LocalMintContext"]:
    """This run's resolution, in the shape a delegated mint child reads.

    ``None`` when the caller did not name the function (``mint_child`` and
    ``boot_trace_child`` both call ``run_setup`` without one, and both pass
    ``arm_compile=False`` — a child that opened its own mint would be the
    recursion this must not have).
    """
    if selected is None:
        return None
    from .. import local_serve

    module = str(getattr(selected.spec, "module", "") or "").strip()
    if not module:
        # The endpoint was discovered from a file path or an interactive
        # namespace, so no importable module name exists for a child to
        # rediscover it in. Adopt-only, honestly.
        module = str(getattr(selected.cls, "__module__", "") or "").strip()
        if module in ("", "__main__"):
            module = ""
    return local_serve.mint_context(
        function=selected.fn_name,
        module=module,
        slots=local_serve.slot_map(
            {k: v for k, v in resolved_models.items() if k in wanted},
            selected.bindings,
        ),
    )


def _assert_structure_honored(
    pipe: Any, injected: Dict[str, Any], *,
    requested: Sequence[str] = (), slot: str = "",
) -> None:
    """The composed pipeline must actually CARRY the structure-only modules.

    ``components=`` is a request, and a pipeline class is free to ignore an
    unexpected keyword — ``**kwargs`` swallows it and the class loads the
    checkpoint itself. That failure is SILENT: the mint would hold every weight
    and still report a weightless child. So it is checked, on the object that
    came back, and a miss raises ``StructureNotHonored`` — a DISTINCT type from
    the buildable strand, because the component WAS built weight-free and the
    pipeline threw it away. The mint child FAILS CLOSED on it rather than
    falling back to a real-weight export.

    ``requested`` is what the CALLER asked to be weight-free, and it is checked
    separately because the loop below can only see components that were BUILT.
    A target the builder skipped — ``model_index.json`` does not name it, or the
    tree has no readable index at all, in which case ``model_index_components``
    returns the empty set and EVERY target is skipped — never enters
    ``injected``, so the loop would iterate nothing and pass while the pipeline
    loaded that target's checkpoint. Such a target is the buildable-strand
    refusal (``StructureOnlyUnsupported``), NOT the swallowed one: the mint
    child's fallback to a real-weight export stays correct and stays RECORDED,
    and the boot trace — which may not fall back at all — refuses.
    """
    from ..models.structure_only import (
        STAMP, StructureNotHonored, StructureOnlyUnsupported, target_module,
    )

    for component in sorted({str(c).strip() for c in requested if str(c).strip()}):
        # A target is an attribute PATH: `vae.decode`'s weights live in the
        # `vae` the builder injected, so the root is what `injected` is keyed
        # on and what decides whether this target was covered.
        if component in injected or component.split(".")[0] in injected:
            continue
        module = target_module(pipe, component)
        if module is None:
            continue  # this slot does not carry it — legitimately absent
        raise StructureOnlyUnsupported(
            component=component, cls_name=type(module).__name__,
            lacks=(
                f"the pipeline for slot {slot or '?'} DOES carry this "
                f"component and no structure-only module was built for it, so "
                f"it was composed from the checkpoint — the tree's "
                f"model_index.json does not name it (or has none), which is "
                f"what the builder skips on"))

    for component, module in sorted(injected.items()):
        got = getattr(pipe, component, None)
        if got is module or getattr(got, STAMP, False):
            continue
        raise StructureNotHonored(
            component=component,
            cls_name=type(pipe).__name__,
            lacks=(
                f"the composed pipeline for slot {slot or '?'} does not carry "
                f"the injected structure-only module — it built "
                f"{type(got).__name__} instead, so this composition either "
                f"ignores `components=` or rebuilds the target from the "
                f"checkpoint"))


def _structure_device(device: str) -> str:
    """Where a structure-only component's VIRTUAL tensors claim to live.

    AOTInductor codegens for the device the traced tensors report, so a
    structure built as "cpu" on a CUDA pod would compile a CPU cell. Fake
    tensors allocate nothing, so claiming the card costs nothing.
    """
    want = (device or "").strip().lower()
    if want:
        return want
    return "cuda" if cuda_ready() else "cpu"


def _load_injected_model(
    annotation: Any, local_path: str, *, decl: Any = None, slot: str = "",
    device: str = "", arm_compile: bool = True, place: bool = True,
    structure_only: Sequence[str] = (),
    mint: Optional["local_serve.LocalMintContext"] = None,
    defer: Optional[List[Any]] = None,
) -> Any:
    """Load one setup slot via the shared core, with a per-process warm cache
    (so ``serve`` and repeated local dispatches never reload weights).

    """
    key = (
        str(annotation),
        str(local_path)
        # A structure-only composition is a DIFFERENT object from the
        # real-weight one; sharing a cache entry between them would hand a
        # weightless pipeline to a caller that asked for weights.
        + ("|structure=" + ",".join(sorted(structure_only))
           if structure_only else "")
        # ...and an UNPLACED composition is a different object again: handing
        # it to a caller that asked for a servable one would serve from host
        # memory.
        + ("" if place else "|unplaced"),
    )
    if key in _INJECTED_CACHE:
        return _INJECTED_CACHE[key]
    binding = (getattr(decl, "models", None) or {}).get(slot) if decl is not None else None
    injected: Dict[str, Any] = {}
    if structure_only:
        from ..models.loading import model_index_components
        from ..models.structure_only import build_component

        declared = model_index_components(local_path)
        for comp in sorted(set(structure_only)):
            if comp not in declared:
                # An auxiliary slot of a multi-slot family (a refiner, a
                # second pipeline) legitimately has no denoiser of this name.
                continue
            module, _facts = build_component(
                local_path, comp, device=_structure_device(device),
                dtype=str(getattr(binding, "dtype", "") or ""))
            injected[comp] = module
    sl = provision.load_slot(
        annotation, local_path, binding=binding, slot=slot, device=device,
        place=place, components=injected or None,
    )
    if not sl.is_pipeline:
        return sl.obj
    if structure_only:
        _assert_structure_honored(
            sl.obj, injected, requested=structure_only, slot=slot)
    compile_cfg = getattr(decl, "compile", None) if decl is not None else None
    if compile_cfg is not None:
        # The compile machinery consumes the enriched CompileCell
        # (decorator-level lora_bucket + declared shape contract), never the
        # raw Compile. Warm guidance stays empty locally — the local mint
        # and its verify both compute from this same object, so the store
        # is self-consistent.
        from ..registry import CompileCell

        # ONE constructor, shared with the registry's path: a must-survive field
        # missed here (e.g. the declared numerics band) would silently judge
        # every locally minted cell at a default nobody chose.
        compile_cfg = CompileCell.from_declaration(
            compile_cfg, lora_bucket=int(getattr(decl, "lora_bucket", 0) or 0))
    if (
        arm_compile
        and compile_cfg is not None
        and device.strip().lower() != "cpu"
    ):
        from .. import local_serve
        from ..models.cache_paths import tensorhub_cas_dir

        # The ONE arming brain, with NO sink: delivered artifact, then THIS
        # MACHINE's own ck1-keyed cell store, then a delegated AOT mint whose
        # result lands in that store — so the second run of this endpoint on
        # this machine arms from disk with no mint, no hub and no network.
        local_serve.enable_compiled(
            sl.obj, compile_cfg, Path(tensorhub_cas_dir()), mint=mint,
            defer=defer)
    _INJECTED_CACHE[key] = sl.obj
    return sl.obj


def _handler_kwargs(bound_method: Any, resolved_models: Dict[str, str]) -> Dict[str, Any]:
    """Per-call model injection, matching the executor's ``_handler_kwargs``:
    handler parameters (after ctx, payload) whose names match model slots
    receive the local snapshot path (``Path`` when annotated ``Path``)."""
    try:
        sig = inspect.signature(bound_method)
        hints = typing.get_type_hints(bound_method)
    except (TypeError, ValueError, NameError):
        return {}
    kwargs: Dict[str, Any] = {}
    for name in list(sig.parameters)[2:]:  # skip ctx, payload (self already bound)
        if name in resolved_models:
            p = resolved_models[name]
            kwargs[name] = Path(p) if hints.get(name) is Path else str(p)
    return kwargs


def _resolve_ctx_slots(ctx: Any, selected: "_SelectedFunction") -> None:
    """Hub-less ``ctx.slots``: no hub means no catalog recipe JSON
    ever arrives, so every Slot resolves to the NEUTRAL schema defaults of
    the handler's derived config schema (``RequestContext[D]``) — exactly
    the hub's own neutral stamp, so hub-less and production agree."""

    spec = getattr(selected, "spec", None)
    defaults_cls = getattr(spec, "defaults_type", None)
    slot_family = getattr(spec, "slot_family", {}) or {}
    resolved: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    declared: List[str] = []  # FIXED slots only (see warmup)
    for name, slot in selected.slots.items():
        try:
            resolved[name] = resolve_slot(
                name, slot, ref=selected.bindings.get(name),
                defaults_cls=defaults_cls,
                family=slot_family.get(name, ""),
            )
        except ValueError as exc:
            errors[name] = str(exc)
            if not getattr(slot, "selected_by", ""):
                declared.append(name)
    set_slots = getattr(ctx, "_set_resolved_slots", None)
    if callable(set_slots):
        root = ""
        for name, slot in selected.slots.items():
            if getattr(slot, "root", False):
                root = name
                break
        set_slots(
            resolved, errors, root_slot=root,
            declared_slot_errors=tuple(declared))


def _local_executing_execution_lane(
    bindings: Dict[str, Any], execution_lane_str: str, handles: Tuple[str, ...]
) -> str:
    """ctx.lane for local runs: a --lane the endpoint DECLARES wins (author
    kernels execute it); otherwise the most-quantized binding's lane (the local
    twin of Executor._served_lane, eager execution)."""

    body = ""
    if execution_lane_str and handles:
        try:
            req = lanespec.parse_execution_lane_spec(execution_lane_str)
            if req.execution_lane is not None and lanespec.execution_lane_body_id(req.execution_lane) in handles:
                body = lanespec.execution_lane_body_id(req.execution_lane)
        except ValueError:
            pass
    if not body:
        body = lanespec.most_quantized_body(
            lanespec.execution_lane_body_of_binding(
                getattr(b, "storage_dtype", "") or "")
            for b in (bindings or {}).values())
    # A local run compiles nothing, so the execution axis is eager — including
    # for a compiled-only body, whose PLAN is not what happened here.
    return lanespec.execution_lane_id(
        lanespec.observed_execution_lane(body, False))


def _apply_execution_lane_to_bindings(bindings: Dict[str, Any], execution_lane_str: str) -> Dict[str, Any]:
    """Fold a --lane choice into the declared bindings.

    bf16 = the declared base (no transform); fp8 family = the local cast lane
    (per-layer fp8 storage). STORED-quant lanes (fp8-w8a8, 4bit) are refused —
    there is no within-tag-group selector; locally, bind the stored-quant
    checkpoint by digest instead."""

    try:
        req = lanespec.parse_execution_lane_spec(execution_lane_str)
    except ValueError as e:
        raise _UsageError(f"--lane: {e}") from None
    if req.is_zero or req.family == lanespec.FAMILY_BF16:
        return bindings
    if req.family == lanespec.FAMILY_4BIT or (
        req.execution_lane is not None
        and req.execution_lane.activation == lanespec.ACT_W8A8
    ):
        raise _UsageError(
            f"--lane {execution_lane_str!r}: a STORED-quant lane is not "
            "selectable locally. It used to fold a `#flavor` onto the "
            "binding, and §1.32(d)/th#1803 deleted that selector "
            "(pgw#1148) — bind the stored-quant checkpoint directly, e.g. "
            "'owner/repo@sha256:<hex>'.")
    out: Dict[str, Any] = {}
    for name, binding in bindings.items():
        try:
            out[name] = rebind_pick(binding, cast="fp8")
        except (ValueError, TypeError):
            out[name] = binding  # non-foldable source — declared as-is
    return out


def _discard(_value: Any) -> None:
    """``run_setup`` returns the loaded slots; ``on_resolved``'s contract is a
    None-returning callback."""
    return None


def dispatch_request(
    *,
    selected: _SelectedFunction,
    instance: Any,
    ctx: Any,
    raw_bytes: bytes,
    offline: bool,
    emit: Callable[[Dict[str, Any]], None],
    write_event: Callable[[str, Any], None],
    on_resolved: Optional[Callable[[Dict[str, str]], None]] = None,
) -> int:
    """Decode payload → resolve models → (optional setup) → invoke → encode.

    This is the dispatch half of ``_run_inner`` factored out so ``serve``
    reuses the exact code path. After model resolution, ``on_resolved`` is
    invoked with the ``{param: local_path}`` map — ``run`` uses it to call
    ``setup(**resolved)`` per-invocation; ``serve`` passes ``None`` because it
    already ran ``setup`` once at boot. ``write_event(kind, value)`` emits one
    result / yield event.

    Returns an EXIT_* code. Raises ``_UsageError`` / ``_ModelResolutionError``
    / ``CanceledError`` for the caller's error mapping.
    """
    payload = _decode_payload(raw_bytes, selected.payload_type)

    resolved_models = provision.resolve_bindings(
        selected.bindings, offline=offline, emit=emit,
        slots=selected.slots, payload=payload,
    )
    if selected.slots:
        _resolve_ctx_slots(ctx, selected)
    if on_resolved is not None:
        on_resolved(resolved_models)

    bound_method = (
        selected.method if instance is None
        else getattr(instance, selected.attr_name)
    )
    extra_kwargs = _handler_kwargs(bound_method, resolved_models)
    result = bound_method(ctx, payload, **extra_kwargs)

    def _emit_item(item: Any) -> None:
        if ctx.cancelled:
            raise CanceledError("canceled")
        write_event("yield", item)

    if inspect.isasyncgen(result):
        import asyncio

        async def _drain() -> int:
            count = 0
            async for item in result:
                _emit_item(item)
                count += 1
            return count

        write_event("result", {"yielded": asyncio.run(_drain())})
    elif inspect.iscoroutine(result):
        import asyncio

        write_event("result", asyncio.run(result))
    elif selected.is_generator or inspect.isgenerator(result):
        count = 0
        for item in result:
            _emit_item(item)
            count += 1
        write_event("result", {"yielded": count})
    else:
        write_event("result", result)
    return EXIT_OK


# --------------------------------------------------------------------------
# Handler
# --------------------------------------------------------------------------

def _handle_run(args: argparse.Namespace) -> int:
    # Must be applied before anything can arm or mint: `run` does its setup()
    # per invocation, so the posture has to stand before the first one.
    if bool(getattr(args, "eager_only", False)):
        serve_posture.apply_command(
            True, actor="cozy-local-cli", reason="--eager-only")
    try:
        return _run_inner(args)
    except _UsageError as e:
        sys.stderr.write(f"gen-worker run: {e}\n")
        return EXIT_USAGE
    except _ModelResolutionError as e:
        sys.stderr.write(f"gen-worker run: model resolution failed: {e}\n")
        return EXIT_MODEL_RESOLUTION
    except CanceledError as e:
        sys.stderr.write(f"gen-worker run: canceled: {e}\n")
        return EXIT_SIGINT
    except KeyboardInterrupt:
        sys.stderr.write("gen-worker run: interrupted\n")
        return EXIT_SIGINT


def _warm_serve_socket() -> Optional[Path]:
    """Return the default serve socket path if a warm ``gen-worker serve``
    is listening, else None. Used by the explicit ``--attach`` flag."""

    sock = Path(DEFAULT_SOCKET_PATH).resolve()
    try:
        is_sock = sock.exists() and sock.is_socket()
    except OSError:
        is_sock = False
    return sock if is_sock else None


def _run_via_warm_serve(
    args: argparse.Namespace, sock: Path, raw_bytes: bytes
) -> int:
    """Dispatch one request through a running ``gen-worker serve``.

    Reuses ``cli/invoke.py``'s socket client. ``run`` addresses by
    class/method, but serve addresses by FUNCTION NAME — so we still import the
    module to resolve the selected function's wire name (cheap; no model load).
    """

    _root, mod = load_endpoint_module(
        config_path=args.config_path, module=args.module,
    )
    candidates = discover_candidates(mod)
    selected = _select_function(
        candidates, cls_name=args.cls_name, method_name=args.method_name,
        default_name=getattr(mod, "__name__", "").split(".", 1)[0],
    )

    raw_bytes = _apply_field_tokens(
        raw_bytes, getattr(args, "fields", None), selected.payload_type,
    )
    try:
        payload_obj = json.loads(raw_bytes.decode("utf-8") or "{}")
    except json.JSONDecodeError as e:
        raise _UsageError(f"--payload is not valid JSON: {e}") from e

    request = {
        "request_id": uuid.uuid4().hex,
        "function": selected.fn_name,
        "payload": payload_obj,
        "stream": True,  # forward events to stdout as produced, like the cold path
    }

    def _on_frame(ev: Dict[str, Any]) -> None:
        _write_stdout_event(
            ev.get("event", "result"), ev.get("value"), pretty=bool(args.pretty)
        )

    resp = invoke_mod._send_request(sock, request, on_frame=_on_frame)

    if not resp.get("ok", False):
        err = resp.get("error") or {}
        kind = err.get("kind", "error")
        msg = err.get("message", "unknown error")
        sys.stderr.write(f"gen-worker run: {kind}: {msg}\n")
        if kind in ("usage", "not_found"):
            return EXIT_USAGE
        if kind == "model_resolution":
            return EXIT_MODEL_RESOLUTION
        if kind == "canceled":
            return EXIT_SIGINT
        return EXIT_USER_EXCEPTION

    # Streaming already wrote each event; a buffered fallback is handled too.
    for ev in resp.get("events") or []:
        _write_stdout_event(
            ev.get("event", "result"), ev.get("value"), pretty=bool(args.pretty)
        )
    return EXIT_OK


def _run_inner(args: argparse.Namespace) -> int:
    # --list: print the endpoint description (folded-in `describe`) and exit.
    if getattr(args, "list_functions", False):
        from .listing import build_description

        _root, mod = load_endpoint_module(
            config_path=args.config_path, module=args.module,
        )
        candidates = discover_candidates(mod)
        doc = build_description(
            main_module=getattr(mod, "__name__", args.module or "?"),
            candidates=candidates,
        )
        indent = 2 if args.pretty else None
        sys.stdout.write(json.dumps(doc, indent=indent, default=str) + "\n")
        return EXIT_OK

    # 0. Read the payload up front (decode happens later / on the server).
    raw_bytes = _read_payload_bytes(inline=args.payload, path=args.payload_file)

    # Explicit --attach: dispatch through a warm `gen-worker serve` socket
    # (endpoint already loaded + setup run) instead of cold-loading. Errors
    # loudly when no server is listening — attaching silently would change
    # semantics (--device/--offline don't apply to the serve process).
    if getattr(args, "attach", False):
        warm_sock = _warm_serve_socket()
        if warm_sock is None:
            raise _UsageError(
                "--attach: no warm `gen-worker serve` socket found; start "
                "`gen-worker serve` first or drop --attach."
            )
        return _run_via_warm_serve(args, warm_sock, raw_bytes)

    # 1. Load endpoint.toml + import the main module.
    _root, mod = load_endpoint_module(
        config_path=args.config_path, module=args.module,
    )

    # 2. Discover routable functions on every @endpoint object.
    candidates = discover_candidates(mod)
    selected = _select_function(
        candidates,
        cls_name=args.cls_name,
        method_name=args.method_name,
        default_name=getattr(mod, "__name__", "").split(".", 1)[0],
    )

    # Optional human lane choice, mapped onto the bindings before resolution.
    if getattr(args, "execution_lane", ""):
        selected.bindings = _apply_execution_lane_to_bindings(selected.bindings, args.execution_lane)

    # Ergonomic `field=value` tokens -> payload bytes (coerced via the function's
    # msgspec type), merged over any --payload base.
    raw_bytes = _apply_field_tokens(
        raw_bytes, getattr(args, "fields", None), selected.payload_type,
    )

    # Reserved-source contract, local mode: --source-path stands in for the
    # worker's Hub-resolve + materialize step. If the payload declares a
    # `source` field and none was provided, synthesize one so validation
    # passes; ctx.source_path is set on the context below.
    source_path: Optional[Path] = None
    if getattr(args, "source_path", None):
        source_path = Path(args.source_path).expanduser().resolve()
        if not source_path.exists():
            raise _UsageError(f"--source-path does not exist: {source_path}")
        raw_bytes = _inject_default_source(
            raw_bytes, selected.payload_type, source_path,
        )

    # 4. Build the local context for this kind.
    ctx = build_local_context(
        kind=selected.kind,
        allow_publish=bool(args.allow_publish),
        # The hub-write declaration travels into the local run (pgw#1294), so
        # an author who forgot publishes=True meets the refusal in the dev
        # loop instead of on a rented pod.
        publishes=bool(getattr(selected.spec, "publishes", False)),
    )
    ctx._set_execution_lane(_local_executing_execution_lane(
        selected.bindings, getattr(args, "execution_lane", ""), selected.handles))
    if source_path is not None:
        set_source = getattr(ctx, "_set_source_path", None)
        if not callable(set_source):
            raise _UsageError(
                f"--source-path: {selected.kind!r} endpoints have no reserved "
                "source contract (conversion/dataset/training only)."
            )
        set_source(str(source_path))

    # 5. Instantiate the user class. `run` calls setup() per-invocation, once
    #    the models have been resolved against this request's payload — wired
    #    via the on_resolved callback below.
    instance = instantiate_class(selected.cls)

    def _write_event(kind: str, value: Any) -> None:
        _write_stdout_event(kind, value, pretty=bool(args.pretty))

    # 6. Install SIGINT handler around the user call + dispatch.
    device = str(getattr(args, "device", "") or "")
    sig = _SigintHandler(ctx)
    sig.install()
    try:
        return dispatch_request(
            selected=selected,
            instance=instance,
            ctx=ctx,
            raw_bytes=raw_bytes,
            offline=bool(args.offline),
            emit=_stderr_emitter,
            write_event=_write_event,
            on_resolved=lambda resolved: _discard(
                run_setup(
                    instance, resolved, device=device, selected=selected)),
        )
    except CanceledError:
        raise
    except _UsageError:
        raise
    except _ModelResolutionError:
        raise
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return EXIT_USER_EXCEPTION
    finally:
        sig.restore()
        shutdown = getattr(instance, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                # Best-effort cleanup.
                pass
