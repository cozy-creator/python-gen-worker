"""``gen-worker run`` — execute one endpoint method against a local Python
interpreter.

Two inputs: which function to call, what payload to send. Everything else
(model resolution, payload validation, context wiring) derives from the
decorator declarations and the endpoint.toml — exactly the way the
production worker does it.

The full design and rationale live in ``progress.json`` issue #12.
"""

from __future__ import annotations

import argparse
import collections.abc as cabc
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
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import msgspec

from ..api.binding import CivitaiRepo, Dispatch, HFRepo, Repo
from ..api.errors import CanceledError
from .local_context import build_local_context


# Exit codes — tracked here for one-line reference. Mirrors progress.json #12.
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
            "Execute one @inference.function method in the local Python "
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
        help="Path to endpoint.toml (defaults to ./endpoint.toml).",
    )
    p.add_argument(
        "--module", dest="module", default=None,
        help="Python module path to import (overrides endpoint.toml `main`).",
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
        help=(
            "Allow ConversionContext.publish_repo_revision / materialize_blob "
            "to call the real tensorhub APIs. Default: stubbed (print payload "
            "to stderr, return a fake response)."
        ),
    )
    p.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the stdout result with newlines + 2-space indent.",
    )
    p.set_defaults(_handler=_handle_run)


# --------------------------------------------------------------------------
# endpoint.toml + module loading
# --------------------------------------------------------------------------

def _load_endpoint_toml_main(config_path: Optional[str]) -> Tuple[Path, str]:
    """Return ``(project_root, main_module)`` from the endpoint.toml.

    ``main_module`` defaults to the toml file's ``main`` field. The caller
    may override with ``--module`` (handled at the call site).
    """
    from ..discovery.toml_manifest import load_endpoint_toml

    if config_path:
        path = Path(config_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"endpoint.toml not found: {path}")
        root = path.parent
    else:
        root = Path.cwd().resolve()
        path = root / "endpoint.toml"
        if not path.exists():
            raise FileNotFoundError(
                f"endpoint.toml not found at {path}; pass --config or run from "
                "the endpoint root."
            )
    et = load_endpoint_toml(path)
    return root, et.main


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
    """Live handle to one selected @inference.function method.

    Carries the actual class object, bound method, payload type, and the
    endpoint-level binding map (``models={...}``). The run pipeline uses
    these to instantiate the class, call ``setup(**resolved_models)``, and
    dispatch with ``(ctx, payload)``.
    """

    def __init__(
        self,
        *,
        cls: type,
        attr_name: str,
        method: Callable[..., Any],
        fn_name: str,
        kind: str,
        payload_type: type,
        output_type: Optional[type],
        is_generator: bool,
        bindings: Dict[str, Any],
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


def _collect_class_methods(mod: Any) -> List[_SelectedFunction]:
    """Walk a module dict and collect every @inference.function method on
    every endpoint-decorated class (sync + async). Skips non-class endpoints
    — the SDK is class-shape after the 0.7.6 refactor (#322).
    """
    out: List[_SelectedFunction] = []
    seen_classes: set[int] = set()

    for name, obj in mod.__dict__.items():
        if not inspect.isclass(obj):
            continue
        spec = getattr(obj, "__gen_worker_endpoint_spec__", None)
        if spec is None:
            continue
        # Avoid double-counting re-exported classes.
        if id(obj) in seen_classes:
            continue
        seen_classes.add(id(obj))

        function_methods = getattr(obj, "__gen_worker_function_methods__", None) or []
        bindings = dict(getattr(spec, "models", {}) or {})
        for attr_name, method, fn_spec in function_methods:
            hints = typing.get_type_hints(method, include_extras=False)
            sig = inspect.signature(method)
            params = [p for p in sig.parameters.values() if p.name != "self"]
            if len(params) < 2:
                # Validated by decorator; defensive only.
                continue
            payload_param = params[1]
            payload_type = hints.get(payload_param.name)
            ret = hints.get("return")
            origin = typing.get_origin(ret)
            is_gen = origin in (
                typing.Iterator,
                typing.Iterable,
                typing.AsyncIterator,
                typing.AsyncIterable,
                cabc.Iterator,
                cabc.Iterable,
                cabc.AsyncIterator,
                cabc.AsyncIterable,
            )
            output_type: Optional[type] = None
            if is_gen:
                args = typing.get_args(ret) or ()
                if args:
                    output_type = args[0]
            else:
                output_type = ret if isinstance(ret, type) else None

            out.append(_SelectedFunction(
                cls=obj,
                attr_name=attr_name,
                method=method,
                fn_name=str(fn_spec.name or attr_name),
                kind=str(getattr(spec, "kind", "inference") or "inference"),
                payload_type=payload_type,
                output_type=output_type,
                is_generator=bool(is_gen),
                bindings=bindings,
            ))
    return out


def _select_function(
    candidates: List[_SelectedFunction],
    *,
    cls_name: Optional[str],
    method_name: Optional[str],
) -> _SelectedFunction:
    """Filter candidates by --class / --method and exit-2 on ambiguity."""
    def _label(c: _SelectedFunction) -> str:
        return f"{c.cls.__name__}.{c.attr_name} (fn_name={c.fn_name!r})"

    matches = list(candidates)
    if cls_name:
        matches = [c for c in matches if c.cls.__name__ == cls_name]
    if method_name:
        matches = [
            c for c in matches
            if c.attr_name == method_name or c.fn_name == method_name
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
            f"no @inference.function method matches {filt_msg}.\n"
            f"available:{available}"
        )
    if len(matches) > 1:
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


def _extract_models_override(raw_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """Pull the reserved ``_models`` field out of a JSON payload.

    Returns ``(stripped_bytes, overrides_dict)``. Mirrors the orchestrator's
    pre-dispatch strip — payload structs cannot declare ``_models``, so
    msgspec.decode on the stripped payload always sees a clean shape.
    """
    try:
        obj = json.loads(raw_bytes.decode("utf-8") or "{}")
    except json.JSONDecodeError as e:
        raise _UsageError(f"--payload is not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        # msgspec will reject; let it produce the typed error.
        return raw_bytes, {}
    overrides_raw = obj.pop("_models", None)
    if not overrides_raw:
        return json.dumps(obj, separators=(",", ":")).encode("utf-8"), {}
    if not isinstance(overrides_raw, dict):
        raise _UsageError(
            f"payload._models must be an object mapping param -> ref/spec; "
            f"got {type(overrides_raw).__name__}"
        )
    normalized: Dict[str, Any] = {}
    for k, v in overrides_raw.items():
        normalized[str(k)] = _normalize_one_override(str(k), v)
    return json.dumps(obj, separators=(",", ":")).encode("utf-8"), normalized


def _normalize_one_override(param: str, value: Any) -> Dict[str, str]:
    """Accept either a string shorthand or a structured ``{ref, tag, flavor}``.

    Grammar (same as production): ``owner/repo[:tag][#flavor]``.
    """
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise _UsageError(f"_models[{param!r}] is empty")
        ref = s
        tag = "prod"
        flavor = ""
        if "#" in ref:
            ref, flavor = ref.split("#", 1)
            flavor = flavor.strip()
        if ":" in ref:
            ref, tag = ref.rsplit(":", 1)
            tag = tag.strip() or "prod"
        return {"ref": ref.strip(), "tag": tag, "flavor": flavor}
    if isinstance(value, dict):
        ref = str(value.get("ref") or "").strip()
        if not ref:
            raise _UsageError(f"_models[{param!r}].ref is required")
        return {
            "ref": ref,
            "tag": str(value.get("tag") or "prod").strip() or "prod",
            "flavor": str(value.get("flavor") or "").strip(),
        }
    raise _UsageError(
        f"_models[{param!r}] must be a string shorthand or "
        f"{{ref, tag, flavor}} object; got {type(value).__name__}"
    )


def _decode_payload(
    payload_bytes: bytes, payload_type: type,
) -> Any:
    """Decode the JSON payload into the typed msgspec.Struct, then apply
    any post-decode Clamp constraints exactly the way the live worker does.
    """
    try:
        decoded = msgspec.json.decode(payload_bytes, type=payload_type)
    except msgspec.ValidationError as e:
        # msgspec.ValidationError already carries field path + expected type;
        # re-raise as a usage error so the cli sets exit 2.
        raise _UsageError(f"payload validation failed: {e}") from e
    except msgspec.DecodeError as e:
        raise _UsageError(f"payload is not valid JSON: {e}") from e
    try:
        from ..api.payload_constraints import apply_payload_constraints
        _ = apply_payload_constraints(decoded)
    except Exception:
        # Best-effort; failure here is non-fatal (matches live worker).
        pass
    return decoded


# --------------------------------------------------------------------------
# Model resolution
# --------------------------------------------------------------------------

class _ModelResolutionError(Exception):
    """Raised when a model binding cannot be resolved (exit 3)."""


def _resolve_binding_to_ref(
    *,
    param_name: str,
    binding: Any,
    payload: Any,
    overrides: Dict[str, Dict[str, str]],
) -> Tuple[str, str]:
    """Return ``(model_ref, provider)`` for one binding.

    Resolution order matches the live worker:
      1. payload._models[param] override (already validated upstream).
      2. Binding default — Repo / HFRepo / CivitaiRepo / Dispatch.
    """
    over = overrides.get(param_name)
    if over:
        if not getattr(binding, "_allow_override", False):
            raise _UsageError(
                f"_models[{param_name!r}]: binding has no .allow_override() "
                "declared — override rejected (production would 400 with "
                "model_override_not_allowed)."
            )
        ref = over["ref"]
        tag = over.get("tag", "prod")
        flavor = over.get("flavor", "")
        out = ref
        if tag and tag != "prod":
            out = f"{out}:{tag}"
        if flavor:
            out = f"{out}#{flavor}"
        # The override carries its own ref + optional provider. Absent
        # provider on the override payload defaults to "tensorhub"
        # (consistent with the wire-format contract: absence = tensorhub).
        # Callers who want to override with a non-tensorhub provider must
        # pass the structured form with an explicit "provider" key.
        return out, over.get("provider", "tensorhub") if isinstance(over, dict) else "tensorhub"

    if isinstance(binding, (HFRepo, CivitaiRepo, Repo)):
        provider = binding.provider
        ref = binding.ref
        out = ref
        if binding._tag and binding._tag != "prod":
            out = f"{out}:{binding._tag}"
        if binding._flavor:
            out = f"{out}#{binding._flavor}"
        return out, provider

    if isinstance(binding, Dispatch):
        try:
            chosen = getattr(payload, binding.field)
        except AttributeError:
            raise _UsageError(
                f"dispatch field {binding.field!r} not found on payload"
            ) from None
        if not isinstance(chosen, str):
            raise _UsageError(
                f"dispatch field {binding.field!r} must be a string, "
                f"got {type(chosen).__name__}"
            )
        pick = binding.table.get(chosen.strip())
        if pick is None:
            raise _UsageError(
                f"dispatch key {chosen.strip()!r} not in table; "
                f"allowed: {sorted(binding.table.keys())}"
            )
        out = pick.ref
        if pick._tag and pick._tag != "prod":
            out = f"{out}:{pick._tag}"
        if pick._flavor:
            out = f"{out}#{pick._flavor}"
        return out, pick.provider

    raise _UsageError(
        f"unknown binding type for param {param_name!r}: {type(binding).__name__}"
    )


def _resolve_local_path(
    *, ref: str, provider: str, offline: bool, emit: Callable[[Dict[str, Any]], None],
    allow_patterns: tuple[str, ...] = (),
) -> str:
    """Resolve one model ref to a local snapshot dir / loader-ready string.

    Order matches the live worker:
      1. local CAS lookup via ``_try_find_existing_cozy_snapshot_dir``.
      2. HF refs → ``HuggingFaceHubDownloader`` (auto-fetches from HF).
      3. Cozy refs missing from CAS:
         - ``--offline`` → exit 3.
         - otherwise → exit 3 with a warm-cache pointer (cozy refs need
           an orchestrator-resolved presigned manifest, which the CLI
           doesn't have wired up to talk to tensorhub directly yet —
           tracked separately).
    """
    from ..models.cache_paths import tensorhub_cas_dir
    from ..models.refs import parse_model_ref

    cache_dir = Path(os.getenv("TENSORHUB_CAS_DIR", "")) or tensorhub_cas_dir()
    if not isinstance(cache_dir, Path):
        cache_dir = Path(cache_dir)

    # Decode the bare ref into typed parts using the explicit provider.
    # No string-prefix sniffing — provider is the source of truth.
    try:
        parsed = parse_model_ref(ref, provider=provider)
    except Exception as e:
        raise _ModelResolutionError(
            f"failed to parse model ref {ref!r} (provider={provider!r}): {e}"
        ) from e

    if parsed.provider == "tensorhub" and parsed.tensorhub and parsed.tensorhub.digest:
        digest = parsed.tensorhub.digest
        snap_dir = cache_dir / "snapshots" / digest
        if snap_dir.exists():
            return str(snap_dir)

    # HF refs: fall through to HuggingFaceHubDownloader.
    if parsed.provider == "hf" and parsed.hf is not None:
        if offline:
            # Best-effort: check the HF cache (huggingface_hub manages this
            # itself; a cache hit returns a path, miss raises).
            try:
                from huggingface_hub import snapshot_download as _hf_snap
                p = _hf_snap(
                    repo_id=parsed.hf.repo_id,
                    revision=parsed.hf.revision,
                    local_files_only=True,
                    cache_dir=os.getenv("HF_HOME") or None,
                    token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None,
                    allow_patterns=list(allow_patterns) or None,
                )
                return str(p)
            except Exception as e:
                raise _ModelResolutionError(
                    f"--offline: huggingface ref {parsed.hf.canonical()} not "
                    f"in local cache ({e}); warm the cache by running without "
                    "--offline first."
                ) from e

        emit({"kind": "model_fetch.started", "ref": parsed.hf.canonical()})
        try:
            from ..models.hf_downloader import HuggingFaceHubDownloader
            dl = HuggingFaceHubDownloader(
                hf_home=os.getenv("HF_HOME") or None,
                hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None,
            )
            res = dl.download(parsed.hf, allow_patterns=list(allow_patterns) or None)
        except Exception as e:
            raise _ModelResolutionError(
                f"failed to fetch huggingface ref {parsed.hf.canonical()}: {e}"
            ) from e
        emit({
            "kind": "model_fetch.completed",
            "ref": parsed.hf.canonical(),
            "local_dir": str(res.local_dir),
        })
        return str(res.local_dir)

    # ModelScope refs: fetch directly via modelscope.snapshot_download. This is
    # file-oriented (allow_patterns) and has NO diffusers-layout requirement, so
    # it handles ComfyUI/DiffSynth split checkpoints the HF resolver rejects.
    if parsed.provider == "modelscope" and parsed.modelscope is not None:
        try:
            from modelscope import snapshot_download as _ms_snap
        except Exception as e:
            raise _ModelResolutionError(
                f"modelscope is required for modelscope refs ({parsed.modelscope.canonical()}): {e}"
            ) from e
        kwargs: Dict[str, Any] = {}
        if parsed.modelscope.revision:
            kwargs["revision"] = parsed.modelscope.revision
        if allow_patterns:
            kwargs["allow_patterns"] = list(allow_patterns)
        if offline:
            kwargs["local_files_only"] = True
        emit({"kind": "model_fetch.started", "ref": parsed.modelscope.canonical(), "provider": "modelscope"})
        try:
            local = _ms_snap(model_id=parsed.modelscope.repo_id, **kwargs)
        except Exception as e:
            raise _ModelResolutionError(
                f"failed to fetch modelscope ref {parsed.modelscope.canonical()}: {e}"
            ) from e
        emit({"kind": "model_fetch.completed", "ref": parsed.modelscope.canonical(), "local_dir": str(local)})
        return str(local)

    # Cozy refs that miss the CAS: not yet wired to tensorhub directly from
    # the CLI (requires the presigned-manifest fetch that's owned by the
    # orchestrator in production). Exit 3 with a clear pointer.
    if parsed.provider == "tensorhub" and parsed.tensorhub is not None:
        digest = parsed.tensorhub.digest or "(unresolved tag)"
        if offline:
            raise _ModelResolutionError(
                f"--offline: tensorhub ref {parsed.tensorhub.canonical()} not in local "
                f"CAS ({cache_dir}); warm the cache by running this endpoint "
                "via the orchestrator at least once (or set TENSORHUB_CAS_DIR "
                "to a path with the snapshot pre-seeded)."
            )
        raise _ModelResolutionError(
            f"tensorhub ref {parsed.tensorhub.canonical()} not in local CAS "
            f"({cache_dir}). gen-worker run cannot fetch cozy refs from "
            "tensorhub directly yet — invoke this endpoint via the "
            "orchestrator once to populate the cache, then re-run."
        )

    raise _ModelResolutionError(
        f"unsupported model ref: {ref!r} (provider={provider!r})"
    )


def _resolve_models_for_setup(
    *,
    bindings: Dict[str, Any],
    payload: Any,
    overrides: Dict[str, Dict[str, str]],
    offline: bool,
    emit: Callable[[Dict[str, Any]], None],
) -> Dict[str, str]:
    """Resolve every binding in ``bindings`` to a local path / loader string.

    Skips bindings that don't carry a static ref (pure Dispatch with a
    discriminator handled at invoke time would otherwise blow up here;
    Dispatch IS resolved here against the decoded payload).
    """
    out: Dict[str, str] = {}
    for param_name, binding in bindings.items():
        ref, provider = _resolve_binding_to_ref(
            param_name=param_name,
            binding=binding,
            payload=payload,
            overrides=overrides,
        )
        # ModelScopeRepo carries file-selection (allow_patterns) as binding
        # metadata; thread it through so only the requested files download.
        allow_patterns = tuple(getattr(binding, "_allow_patterns", ()) or ())
        out[param_name] = _resolve_local_path(
            ref=ref, provider=provider, offline=offline, emit=emit,
            allow_patterns=allow_patterns,
        )
    return out


# --------------------------------------------------------------------------
# SIGINT handling
# --------------------------------------------------------------------------

class _SigintHandler:
    """Install a two-stage SIGINT handler.

    First Ctrl-C: trip ``ctx._canceled`` so user code observes via
    ``ctx.is_canceled() / raise_if_canceled()``.
    Second Ctrl-C within 2s: hard-exit 130.
    """

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self._last_at: float = 0.0
        self._prev_handler: signal.Handlers | Callable[[int, types.FrameType | None], Any] = signal.SIG_DFL
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
            self._ctx._canceled = True
            self._ctx._cancel_event.set()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Result encoding
# --------------------------------------------------------------------------

def _encode_struct(value: Any) -> bytes:
    """Encode a msgspec.Struct (or anything msgspec can serialize) to JSON."""
    try:
        return msgspec.json.encode(value)
    except Exception:
        # Last-resort: stdlib json on a builtins shape.
        return json.dumps(msgspec.to_builtins(value), default=str).encode("utf-8")


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
    """Resolve endpoint.toml (or ``--module``), prime sys.path, import `main`.

    Returns ``(project_root, imported_module)``. Shared by ``run`` and
    ``serve`` so both discover endpoint classes the same way. Raises
    ``_UsageError`` on import / config failure (exit 2).
    """
    if module:
        root = Path.cwd().resolve()
        main_module = module
    else:
        root, main_module = _load_endpoint_toml_main(config_path)
    _ensure_sys_path(root)
    try:
        mod = importlib.import_module(main_module)
    except Exception as e:
        raise _UsageError(
            f"failed to import module {main_module!r}: {e}"
        ) from e
    return root, mod


def discover_candidates(mod: types.ModuleType) -> List[_SelectedFunction]:
    """Collect every @inference.function method, exit-2 if none exist."""
    candidates = _collect_class_methods(mod)
    if not candidates:
        raise _UsageError(
            f"no @inference / @training / @conversion / @dataset classes "
            f"with @inference.function methods found in module "
            f"{getattr(mod, '__name__', '?')!r}"
        )
    return candidates


def instantiate_class(cls: type) -> Any:
    """Instantiate a discovered endpoint class (no setup yet)."""
    return cls()


def run_setup(instance: Any, resolved_models: Dict[str, str]) -> None:
    """Call ``instance.setup(...)`` once, passing exactly the resolved model
    slots its signature declares.

    A bare ``def setup(self)`` is legitimate (#337 model-selectable endpoints
    receive their model per-request in the handler instead), so unclaimed
    slots are fine. But a setup() that declares parameters we CANNOT satisfy
    is an authoring error and must fail loudly -- the previous blanket
    ``except TypeError: setup_fn()`` swallowed signature mismatches and let
    them resurface as confusing downstream failures.
    """
    setup_fn = getattr(instance, "setup", None)
    if setup_fn is None:
        return
    try:
        params = inspect.signature(setup_fn).parameters
    except (TypeError, ValueError):
        # Odd callables without an inspectable signature: keep the legacy
        # best-effort behavior for them only.
        try:
            setup_fn(**resolved_models)
        except TypeError:
            setup_fn()
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        setup_fn(**resolved_models)
        return
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
            f"parameter must match a key in the @inference models={{...}} "
            f"dict (static bindings are injected into setup; dispatch "
            f"bindings are injected into the handler per-request)."
        )
    setup_fn(**{k: v for k, v in resolved_models.items() if k in wanted})


_INJECTED_CACHE: Dict[Tuple[str, str], Any] = {}


def _build_injected_kwargs(bound_method: Any, resolved_models: Dict[str, str]) -> Dict[str, Any]:
    """Local #337 shim: inject per-request model slots into the handler.

    Production assembles SharedBase variants in the worker; locally we load a
    complete pipeline from the resolved snapshot using the parameter's type
    annotation (from_pretrained for diffusers-layout dirs, from_single_file
    for single-checkpoint repos).
    """
    try:
        sig = inspect.signature(bound_method)
        hints = typing.get_type_hints(bound_method)
    except (TypeError, ValueError, NameError):
        return {}
    kwargs: Dict[str, Any] = {}
    for name in list(sig.parameters)[2:]:  # skip ctx, payload (self already bound)
        if name in resolved_models:
            kwargs[name] = _load_injected_model(hints.get(name), resolved_models[name])
    return kwargs


def _load_injected_model(annotation: Any, local_path: str) -> Any:
    key = (str(annotation), str(local_path))
    if key in _INJECTED_CACHE:
        return _INJECTED_CACHE[key]
    import torch
    cls = annotation if isinstance(annotation, type) else None
    if cls is None or not hasattr(cls, "from_pretrained"):
        from diffusers import DiffusionPipeline
        cls = DiffusionPipeline
    p = Path(local_path)
    if p.is_dir() and (p / "model_index.json").exists():
        pipe = cls.from_pretrained(str(p), torch_dtype=torch.float16)
    else:
        single = p if p.is_file() else next(iter(sorted(p.glob("*.safetensors"))), None)
        if single is None or not hasattr(cls, "from_single_file"):
            raise _ModelResolutionError(
                f"cannot materialize injected model slot from {p} "
                f"(no model_index.json, no single-file checkpoint)"
            )
        pipe = cls.from_single_file(str(single), torch_dtype=torch.float16)
    _INJECTED_CACHE[key] = pipe
    return pipe


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
    stripped_bytes, overrides = _extract_models_override(raw_bytes)
    payload = _decode_payload(stripped_bytes, selected.payload_type)

    resolved_models = _resolve_models_for_setup(
        bindings=selected.bindings,
        payload=payload,
        overrides=overrides,
        offline=offline,
        emit=emit,
    )
    if on_resolved is not None:
        on_resolved(resolved_models)

    bound_method = getattr(instance, selected.attr_name)
    extra_kwargs = _build_injected_kwargs(bound_method, resolved_models)
    result = bound_method(ctx, payload, **extra_kwargs)
    if selected.is_generator or inspect.isgenerator(result):
        count = 0
        for item in result:
            if ctx.is_canceled():
                raise CanceledError("canceled")
            write_event("yield", item)
            count += 1
        write_event("result", {"yielded": count})
    else:
        write_event("result", result)
    return EXIT_OK


# --------------------------------------------------------------------------
# Handler
# --------------------------------------------------------------------------

def _handle_run(args: argparse.Namespace) -> int:
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
    """Return the default serve socket path if a warm ``gen-worker serve`` is
    listening, else None (#340).

    A ``gen-worker serve`` at the default ``./.gen-worker.sock`` already has the
    endpoint loaded and ``setup()`` run. When present, ``run`` attaches to it
    rather than paying the cold-load cost again. Importing ``serve`` lazily
    keeps it off the hot import path.
    """
    from .serve import DEFAULT_SOCKET_PATH

    sock = Path(DEFAULT_SOCKET_PATH).resolve()
    try:
        is_sock = sock.exists() and sock.is_socket()
    except OSError:
        is_sock = False
    return sock if is_sock else None


def _run_via_warm_serve(
    args: argparse.Namespace, sock: Path, raw_bytes: bytes
) -> int:
    """Dispatch one request through a running ``gen-worker serve`` (#340).

    Reuses ``cli/invoke.py``'s socket client. ``run`` addresses by
    class/method, but serve addresses by FUNCTION NAME — so we still import the
    module to resolve the selected function's wire name (cheap; no model load).
    """
    from . import invoke as invoke_mod

    _root, mod = load_endpoint_module(
        config_path=args.config_path, module=args.module,
    )
    candidates = discover_candidates(mod)
    selected = _select_function(
        candidates, cls_name=args.cls_name, method_name=args.method_name,
    )

    try:
        payload_obj = json.loads(raw_bytes.decode("utf-8") or "{}")
    except json.JSONDecodeError as e:
        raise _UsageError(f"--payload is not valid JSON: {e}") from e

    request = {"function": selected.fn_name, "payload": payload_obj}
    resp = invoke_mod._send_request(sock, request)

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

    for ev in resp.get("events") or []:
        _write_stdout_event(
            ev.get("event", "result"), ev.get("value"), pretty=bool(args.pretty)
        )
    return EXIT_OK


def _run_inner(args: argparse.Namespace) -> int:
    # 0. Read the payload up front (decode happens later / on the server).
    raw_bytes = _read_payload_bytes(inline=args.payload, path=args.payload_file)

    # #340 auto-attach: if a warm `gen-worker serve` is listening on the
    # default socket, dispatch through it (endpoint already loaded + setup run)
    # instead of cold-loading. Falls back to the one-shot path when no server
    # is up, preserving all of `run`'s existing flags/behavior.
    warm_sock = _warm_serve_socket()
    if warm_sock is not None:
        return _run_via_warm_serve(args, warm_sock, raw_bytes)

    # 1. Load endpoint.toml + import the main module.
    _root, mod = load_endpoint_module(
        config_path=args.config_path, module=args.module,
    )

    # 2. Discover @inference.function methods on every endpoint class.
    candidates = discover_candidates(mod)
    selected = _select_function(
        candidates,
        cls_name=args.cls_name,
        method_name=args.method_name,
    )

    from .local_context import _stderr_emitter

    # 4. Build the local context for this kind.
    ctx = build_local_context(
        kind=selected.kind,
        allow_publish=bool(args.allow_publish),
    )
    # Honor --device by stashing it on the context. (RequestContext doesn't
    # currently expose a `.device` setter; tenants typically read it via
    # torch directly. We surface the override via env so user code can
    # `os.getenv("GEN_WORKER_LOCAL_DEVICE")` if it wants.)
    if args.device:
        os.environ["GEN_WORKER_LOCAL_DEVICE"] = args.device

    # 5. Instantiate the user class. `run` calls setup() per-invocation, once
    #    the models have been resolved against this request's payload — wired
    #    via the on_resolved callback below.
    instance = instantiate_class(selected.cls)

    def _write_event(kind: str, value: Any) -> None:
        _write_stdout_event(kind, value, pretty=bool(args.pretty))

    # 6. Install SIGINT handler around the user call + dispatch.
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
            on_resolved=lambda resolved: run_setup(instance, resolved),
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
