from __future__ import annotations

import asyncio
import contextvars
from pathlib import Path
from typing import Any, Coroutine, Iterable, Mapping, Optional

from .cozy_snapshot_v2 import ensure_snapshot_async
from .downloader import ModelDownloader
from .hf_downloader import HuggingFaceHubDownloader
from .refs import ParsedModelRef, parse_model_ref
from .unsafe_format import UnsafeFileFormat, assert_safe_weight_format

# Per-request resolved manifests provided by gen-orchestrator (issue #92).
# Shape: {canonical_model_id: ResolvedRepo-like object}
_resolved_repos_by_id: contextvars.ContextVar[Optional[Mapping[str, Any]]] = contextvars.ContextVar(
    "resolved_repos_by_id", default=None
)


def set_resolved_repos_by_id(mapping: Optional[Mapping[str, Any]]) -> contextvars.Token:
    return _resolved_repos_by_id.set(mapping)


def reset_resolved_repos_by_id(token: contextvars.Token) -> None:
    _resolved_repos_by_id.reset(token)


# Per-worker provider index, built from endpoint.lock at boot (issue #17).
# Shape: {bare_ref_string: provider} where provider is "tensorhub" | "hf" | "civitai".
# The downloader and other parse-time sites consult this to recover the
# provider for a bare ref, since the runtime wire format (EndpointConfig,
# JobExecutionRequest, etc.) currently carries refs as bare strings without
# a side-channel provider field. Refs not in the index default to
# ``provider="tensorhub"`` — same as the wire-format contract.
_provider_by_ref: contextvars.ContextVar[Optional[Mapping[str, str]]] = contextvars.ContextVar(
    "provider_by_ref", default=None
)


def set_provider_by_ref(mapping: Optional[Mapping[str, str]]) -> contextvars.Token:
    return _provider_by_ref.set(mapping)


def reset_provider_by_ref(token: contextvars.Token) -> None:
    _provider_by_ref.reset(token)


# Issue #18: per-request set of override ref keys. When the worker
# dispatches a JobExecutionRequest whose `resolved_models` map carries
# invoker-supplied overrides, it records the corresponding canonical model
# id strings here so the downloader can run the safetensors-only gate on
# the resulting snapshot. Binding-default refs are NOT in this set —
# they already passed the build-time validator.
_override_ref_keys: contextvars.ContextVar[Optional[frozenset[str]]] = contextvars.ContextVar(
    "override_ref_keys", default=None
)


def set_override_ref_keys(keys: Optional[Iterable[str]]) -> contextvars.Token:
    if keys is None:
        return _override_ref_keys.set(None)
    return _override_ref_keys.set(frozenset(k for k in keys if k))


def reset_override_ref_keys(token: contextvars.Token) -> None:
    _override_ref_keys.reset(token)


# Process-global fallback index: the worker sets this once at boot from the
# baked endpoint.lock manifest. Used by ``lookup_provider_for_ref`` when the
# contextvar (per-request scope) is unset — contextvars don't reliably
# propagate across threads spawned by gRPC stream handlers etc., so a global
# fallback is needed for the binding-default code paths.
_provider_by_ref_global: Mapping[str, str] = {}


def set_provider_by_ref_global(mapping: Optional[Mapping[str, str]]) -> None:
    """Install the process-global fallback provider index. Worker calls this
    once at boot from the loaded endpoint.lock manifest. Per-request override
    indexes still go through the contextvar (``set_provider_by_ref``).
    """
    global _provider_by_ref_global
    _provider_by_ref_global = mapping or {}


def lookup_provider_for_ref(ref: str, *, default: str = "tensorhub") -> str:
    """Return the provider tag for ``ref`` from the active context, falling
    back to the process-global index, then to ``default``.

    Key variants tried (in order) on each mapping layer:
      1. raw ``ref``
      2. ``ref.strip()`` (in case of stray whitespace)
      3. ``ref`` with the tag segment removed (``owner/repo:tag#flavor`` ->
         ``owner/repo#flavor``) — runtime payloads may stamp ``:latest`` on
         HF refs because the canonicalizer's contextvar wasn't set in the
         thread that built them; the index keys are bare so we strip to
         match. Tag is meaningless for HF/civitai providers.
    """
    if not ref:
        return default

    def _try(mapping: Mapping[str, str]) -> Optional[str]:
        if not mapping:
            return None
        if ref in mapping:
            return mapping[ref]
        stripped = ref.strip()
        if stripped and stripped != ref and stripped in mapping:
            return mapping[stripped]
        # Strip ``:tag`` between the repo path and any ``#flavor`` suffix.
        # Only the first ``:`` after the last ``/`` is the tag separator;
        # everything before that is the repo identity.
        base = stripped or ref
        if "/" in base:
            slash = base.rfind("/")
            head = base[:slash]
            tail = base[slash:]
            colon = tail.find(":")
            if colon >= 0:
                hash_idx = tail.find("#")
                if hash_idx < 0:
                    no_tag = head + tail[:colon]
                else:
                    no_tag = head + tail[:colon] + tail[hash_idx:]
                if no_tag != base and no_tag in mapping:
                    return mapping[no_tag]
        return None

    ctx_mapping = _provider_by_ref.get()
    if ctx_mapping:
        hit = _try(ctx_mapping)
        if hit is not None:
            return hit
    hit = _try(_provider_by_ref_global)
    if hit is not None:
        return hit
    return default


class ModelRefDownloader(ModelDownloader):
    """Composite downloader for typed model refs.

    Supported providers (provider tracked separately from the bare ref):
      - Tensorhub (default): bare owner/repo[:tag] or owner/repo@sha256:<digest>.
        Requires orchestrator-provided presigned URLs.
      - Hugging Face: bare owner/repo[@revision].

    Returns a local directory path for both providers.
    """

    def __init__(
        self,
        hf_home: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self._hf = HuggingFaceHubDownloader(hf_home=hf_home, hf_token=hf_token)

    async def _download_async(self, parsed: ParsedModelRef, dest_dir: Path) -> Path:
        if parsed.provider == "hf" and parsed.hf is not None:
            # Workers download HuggingFace refs directly from HF. Any
            # Cozy-Hub mirroring of HF repos is orchestrator-side
            # (pre-resolved into resolved_repos_by_id as a cozy: ref).
            return self._hf.download(parsed.hf).local_dir

        if parsed.provider == "tensorhub" and parsed.tensorhub is not None:
            canonical = parsed.tensorhub.canonical()
            resolved_mapping = _resolved_repos_by_id.get()
            resolved_entry = resolved_mapping.get(canonical) if resolved_mapping is not None else None

            if resolved_entry is None:
                # Workers never resolve directly against tensorhub. The
                # orchestrator pre-resolves every cozy: ref a job needs and
                # ships the manifest + presigned URLs via
                # JobExecutionRequest.resolved_repos_by_id. Missing entry
                # here means the orchestrator didn't pre-resolve this ref
                # — that's an orchestrator-side bug, not a worker fallback.
                raise RuntimeError(
                    f"tensorhub ref {canonical!r} not in resolved_repos_by_id "
                    "— orchestrator must pre-resolve before dispatching the job"
                )

            return await ensure_snapshot_async(
                base_dir=dest_dir,
                ref=parsed.tensorhub,
                resolved=resolved_entry,
            )

        raise ValueError("invalid parsed model ref")

    def download(self, model_ref: str, dest_dir: str, filename: Optional[str] = None) -> str:
        # Issue #17: bare ref strings on the wire don't carry their provider,
        # so consult the per-worker provider index built from endpoint.lock.
        # Missing entries (e.g. invoker-supplied overrides not in the build-
        # time manifest) default to "tensorhub", matching the wire contract.
        provider = lookup_provider_for_ref(model_ref)
        parsed = parse_model_ref(model_ref, provider=provider)
        base = Path(dest_dir)
        # We ignore filename; snapshots/refs already define structure.
        local: Optional[str] = None
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Nested loop scenario; run in a new loop in a thread.
                local = _run_in_thread(self._download_async(parsed, base))
        except RuntimeError:
            pass
        if local is None:
            local = asyncio.run(self._download_async(parsed, base)).as_posix()

        # Issue #18: belt-and-braces safetensors-only gate for override
        # downloads. Binding-default refs already passed the build-time
        # validator; the orchestrator runs the same check pre-dispatch in
        # gen-orchestrator #358. This is the worker-side last line of
        # defense if either of those slipped.
        override_keys = _override_ref_keys.get()
        if override_keys and model_ref in override_keys:
            assert_safe_weight_format(Path(local), ref=model_ref)
        return local


def _run_in_thread(coro: Coroutine[Any, Any, Path]) -> str:
    out: dict[str, str] = {}
    err: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            out["v"] = asyncio.run(coro).as_posix()
        except BaseException as e:
            err["e"] = e

    import threading

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "e" in err:
        raise err["e"]
    return out["v"]
