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
from typing import Any, Callable, Dict, List, Optional, Tuple

import msgspec

from ..api.binding import BINDING_TYPES, wire_ref
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
    from ..discovery.project import load_project_config

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
    """Collect every routable function on every @endpoint object in a module
    namespace. Delegates signature inspection to ``gen_worker.registry`` — the
    one walker shared with the worker runtime and build-time discovery."""
    from gen_worker.registry import collect_from_namespace

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
            # Variant fan-out: the base fn and every variant share one attr
            # name. An exact fn_name match wins over the attr-name matches.
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
    from .args import ArgError, build_payload

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


def _resolve_binding_to_ref(*, param_name: str, binding: Any) -> Tuple[str, str]:
    """Return ``(model_ref, provider)`` for one binding."""
    if isinstance(binding, BINDING_TYPES):
        return wire_ref(binding), binding.provider
    raise _UsageError(
        f"unknown binding type for param {param_name!r}: {type(binding).__name__}"
    )


def _resolve_local_path(
    *, ref: str, provider: str, offline: bool, emit: Callable[[Dict[str, Any]], None],
    allow_patterns: tuple[str, ...] = (),
    civitai_version_id: str = "",
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
            from ..models.download import download_hf

            local_dir = download_hf(
                parsed.hf,
                hf_home=os.getenv("HF_HOME") or None,
                hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None,
                allow_patterns=tuple(allow_patterns),
            )
        except Exception as e:
            raise _ModelResolutionError(
                f"failed to fetch huggingface ref {parsed.hf.canonical()}: {e}"
            ) from e
        emit({
            "kind": "model_fetch.completed",
            "ref": parsed.hf.canonical(),
            "local_dir": str(local_dir),
        })
        return str(local_dir)

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

    # Civitai refs: download the model-version files directly. Auth (for gated
    # creators) comes from CIVITAI_API_KEY; public models need none.
    if parsed.provider == "civitai" and parsed.civitai is not None:
        if offline:
            raise _ModelResolutionError(
                f"--offline: civitai ref {ref!r} not available offline (no local "
                "civitai cache); run once online to fetch it."
            )
        from ..models.download import (
            download_civitai,
            fetch_civitai_model,
            parse_civitai_version_id,
        )
        api_key = os.getenv("CIVITAI_API_KEY", "") or os.getenv("CIVITAI_TOKEN", "")

        if civitai_version_id:
            # Explicit version pin via Civitai(version="<id>"). The pinned id
            # IS a model-VERSION id, so use it directly — no model lookup.
            try:
                version_id = parse_civitai_version_id(civitai_version_id)
            except Exception as e:
                raise _ModelResolutionError(
                    f"bad civitai version pin {civitai_version_id!r} on ref {ref!r}: {e}"
                ) from e
        else:
            # Civitai's ref is a MODEL id by convention; map it to its latest
            # version id. No silent fallback: if the lookup fails or the model
            # has no versions, the ref is wrong (e.g. a bare version id was
            # passed where a model id was expected) — surface it rather than
            # guessing and downloading an unrelated model.
            try:
                model_id = parse_civitai_version_id(parsed.civitai.model_id)
            except Exception as e:
                raise _ModelResolutionError(f"bad civitai ref {ref!r}: {e}") from e
            try:
                model = fetch_civitai_model(model_id, api_key=api_key)
            except Exception as e:
                raise _ModelResolutionError(
                    f"failed to resolve civitai model {model_id} for ref {ref!r}: {e}; "
                    "Civitai's ref must be a MODEL id (pin a specific version "
                    'with .version("<version_id>")).'
                ) from e
            versions = model.get("modelVersions") or []
            version_id = int(versions[0].get("id") or 0) if versions else 0
            if version_id <= 0:
                raise _ModelResolutionError(
                    f"civitai model {model_id} (ref {ref!r}) has no published "
                    'version to download (pin one with .version("<version_id>")).'
                )
        out_dir = cache_dir / "civitai" / str(version_id)
        emit({"kind": "model_fetch.started", "ref": ref, "provider": "civitai"})
        try:
            local = download_civitai(version_id, out_dir, api_key=api_key)
        except Exception as e:
            raise _ModelResolutionError(
                f"failed to fetch civitai ref {ref!r} (resolved version {version_id}): {e}"
            ) from e
        emit({"kind": "model_fetch.completed", "ref": ref, "local_dir": str(local)})
        return str(local)

    raise _ModelResolutionError(
        f"unsupported model ref: {ref!r} (provider={provider!r})"
    )


def _resolve_models_for_setup(
    *,
    bindings: Dict[str, Any],
    offline: bool,
    emit: Callable[[Dict[str, Any]], None],
) -> Dict[str, str]:
    """Resolve every binding in ``bindings`` to a local path / loader string."""
    out: Dict[str, str] = {}
    for param_name, binding in bindings.items():
        ref, provider = _resolve_binding_to_ref(param_name=param_name, binding=binding)
        out[param_name] = _resolve_local_path(
            ref=ref, provider=provider, offline=offline, emit=emit,
            allow_patterns=tuple(getattr(binding, "files", ()) or ()),
            civitai_version_id=str(getattr(binding, "version", "") or ""),
        )
    return out


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
            # Same entry point the production worker uses for an
            # orchestrator cancel (#352).
            self._ctx._cancel()
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
    if instance is None or setup_fn is None:
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
            f"parameter must match a key in the @endpoint models={{...}} "
            f"dict (static bindings are injected into setup; dispatch "
            f"bindings are injected into the handler per-request)."
        )
    # Load each claimed slot through the same loader the handler-injection
    # path uses: a setup(pipeline: StableDiffusionXLPipeline) must receive a
    # constructed pipeline, not the snapshot path string (production executor
    # behavior; previously setup received raw paths -> 'str' is not callable).
    try:
        hints = typing.get_type_hints(setup_fn)
    except (TypeError, ValueError, NameError):
        hints = {}
    loaded = {
        k: _load_injected_model(hints.get(k), v)
        for k, v in resolved_models.items()
        if k in wanted
    }
    setup_fn(**loaded)


_INJECTED_CACHE: Dict[Tuple[str, str], Any] = {}


def _build_injected_kwargs(bound_method: Any, resolved_models: Dict[str, str]) -> Dict[str, Any]:
    """Local #337 shim: inject per-request model slots into the handler.

    Locally we load a
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
    from gen_worker.models.loading import load_from_pretrained

    cls = annotation if isinstance(annotation, type) else None
    if cls is None or not hasattr(cls, "from_pretrained"):
        from diffusers import DiffusionPipeline
        cls = DiffusionPipeline
    # fp16 kernels are CUDA-only for several ops; on a CPU device run use fp32.
    device = (os.getenv("GEN_WORKER_LOCAL_DEVICE") or "").strip().lower()
    # Same loader the production executor uses: handles diffusers-layout dirs,
    # module-layout dirs (root config.json, e.g. a bare VAE/UNet repo), and
    # single-file checkpoints.
    pipe = load_from_pretrained(cls, local_path, dtype="fp32" if device == "cpu" else "fp16")
    if device != "cpu":
        # Worker-owned placement (executor parity): endpoints never write
        # device/offload code, so the cold `run` path must place the loaded
        # pipeline too — full CUDA residency or the offload ladder as VRAM
        # allows. Without this the pipeline stays on CPU while ctx.generator()
        # hands out CUDA generators.
        from gen_worker.models.memory import place_pipeline

        place_pipeline(pipe)
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
    payload = _decode_payload(raw_bytes, selected.payload_type)

    resolved_models = _resolve_models_for_setup(
        bindings=selected.bindings,
        offline=offline,
        emit=emit,
    )
    if on_resolved is not None:
        on_resolved(resolved_models)

    bound_method = (
        selected.method if instance is None
        else getattr(instance, selected.attr_name)
    )
    extra_kwargs = _build_injected_kwargs(bound_method, resolved_models)
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

    from .local_context import _stderr_emitter

    # 4. Build the local context for this kind.
    ctx = build_local_context(
        kind=selected.kind,
        allow_publish=bool(args.allow_publish),
    )
    if source_path is not None:
        set_source = getattr(ctx, "_set_source_path", None)
        if not callable(set_source):
            raise _UsageError(
                f"--source-path: {selected.kind!r} endpoints have no reserved "
                "source contract (conversion/dataset/training only)."
            )
        set_source(str(source_path))
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
