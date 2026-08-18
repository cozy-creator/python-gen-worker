"""The endpoint serve host (pgw#1372): eager-first boot, adopt via ctx.compile.

Boot, in order (Paul's flow, pgw#1367 + the 2026-08-18 imperative-marking
ruling): when release metadata is at hand, build the torchcg ``AdoptSession``
FIRST — its constructor runs the exact-env audit, so a mismatched pod
refuses loudly before any author code runs — then instantiate the author's
class and call ``setup(ctx)``. The author's own ``ctx.compile(module)``
calls are where compiled graphs swap in: hit -> armed, miss -> eager + an
ordered :class:`Hole` for the background mint (pgw#1371). No trace, no
derivation, no compile happens here, ever.

Telemetry: ``setup`` records a ``model_load`` span, and an adoption-carrying
boot records the ``adopt_pull`` span (``graphs_from=release`` + per-outcome
counts) — the new-flow replacement for the deleted keyset/derive span.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import msgspec

from .. import boot_stages
from .context import DeployBinding, ServeContext
from .loader import LoadedEndpoint

logger = logging.getLogger(__name__)


class ServeDispatchError(RuntimeError):
    """A request names a function this endpoint does not serve."""


class EndpointHost:
    """One loaded endpoint, booted and routable.

    The host owns worker-side state only: the deploy binding (mutable hub
    state — ``rebind`` swaps it), the active lane (the deploy's pick), the
    author's live instance, and the adoption session. Author state lives on
    the author's instance, untouched.
    """

    def __init__(
        self,
        loaded: LoadedEndpoint,
        binding: DeployBinding,
        *,
        lane_contract: str = "",
        output_dir: Optional[Path] = None,
        context_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.loaded = loaded
        self.binding = binding
        self.lane = loaded.lane(lane_contract)
        self.instance: Any = None
        self.adoption: Any = None
        self._output_dir = output_dir
        self._context_kwargs = dict(context_kwargs or {})

    # -- contexts -----------------------------------------------------------

    def make_context(
        self, request_id: str, *, is_trace: bool = False, **overrides: Any
    ) -> ServeContext:
        kwargs: Dict[str, Any] = dict(self._context_kwargs)
        if self._output_dir is not None:
            kwargs.setdefault("local_output_dir", str(self._output_dir))
        kwargs.update(overrides)
        return ServeContext(
            request_id,
            binding=self.binding,
            lane=self.lane,
            is_trace=is_trace,
            **kwargs,
        )

    # -- boot ---------------------------------------------------------------

    def setup(
        self,
        *,
        store: Any = None,
        document: Any = None,
        sm: str = "",
        loader: Any = None,
        artifacts_dir: Optional[Path] = None,
        installed: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Instantiate the author's class and run its ``setup(ctx)``.

        With no ``document`` this is the EAGER bridge: ``ctx.compile`` is a
        transparent pass-through and the endpoint serves after setup,
        unconditionally. With release metadata (``document`` + ``sm``), the
        adopt session forms FIRST — the exact-env audit refuses loudly
        before any author code runs — and the author's own ``ctx.compile``
        calls arm what the store holds; ``self.holes`` afterwards is the
        ordered mint work-list. ``store`` may be ``None`` with a document:
        every claimed graph is then a hole (mint everything).
        """
        from .._vendor.torchcg import EnvironmentMismatch
        from .._vendor.torchcg.adopt import AdoptSession
        from .._vendor.torchcg.graph_identity import installed_closure

        started = time.monotonic()

        def span(**attrs: object) -> None:
            boot_stages.record_ending_now(
                boot_stages.Stage.ADOPT_PULL,
                duration_ms=int((time.monotonic() - started) * 1000),
                label=self.loaded.module_name,
                sm=sm,
                **attrs,
            )

        session = None
        installed_map: Optional[Mapping[str, str]] = None
        if document is not None and getattr(document, "eager_permanent", False):
            span(graphs_from="eager_permanent")
            document = None
        if document is not None:
            if self.lane is None:
                raise RuntimeError(
                    "setup(): release metadata offered but this endpoint "
                    "declared no lanes; an eager-permanent endpoint has "
                    "nothing to adopt"
                )
            lane_contract = str(getattr(self.lane, "contract"))
            installed_map = dict(installed) if installed is not None else installed_closure()
            try:
                session = AdoptSession(
                    store,
                    document,
                    lane_contract,
                    sm,
                    loader=loader,
                    artifacts_dir=artifacts_dir
                    or Path(".compiled-graphs"),
                    installed=installed_map,
                )
            except EnvironmentMismatch as exc:
                # The audit fired BEFORE any author code ran — a
                # build-system bug surfacing, recorded then re-raised loudly.
                span(graphs_from="release", lane=lane_contract,
                     refusal="environment_mismatch")
                logger.error("adopt refused: %s", exc)
                raise

        load_started = time.monotonic()
        self.instance = self.loaded.cls()
        ctx = self.make_context(
            "boot-setup", boot_warmup=True,
            compile_sink=session.adopt if session is not None else None,
        )
        self.instance.setup(ctx)
        boot_stages.record_ending_now(
            boot_stages.Stage.MODEL_LOAD,
            duration_ms=int((time.monotonic() - load_started) * 1000),
            label=self.loaded.module_name,
            checkpoint=self.binding.checkpoint_ref,
        )
        if session is not None:
            # The mint-written requirements manifest is an AUDIT assertion
            # (exact-env ruling): every adopted artifact restates what its
            # mint linked, and a divergence is the build system contradicting
            # itself — refuse loudly, never adopt-and-hope.
            if store is not None and installed_map is not None:
                for record in session.adopted:
                    manifest = store.get_manifest(record.graph, session.env)
                    if manifest is not None:
                        manifest.assert_environment(installed_map, sm=sm)
            span(
                graphs_from="release",
                lane=str(getattr(self.lane, "contract")),
                artifact_from_store=len(session.adopted),
                artifact_from_eager=len(session.holes),
                ambiguous=len(session.ambiguous),
                unclaimed=len(session.unclaimed),
            )
        self.adoption = session

    def teardown(self) -> None:
        """Run the author's ``teardown(ctx)`` hook (base-class contract)."""
        if self.instance is None:
            return
        self.instance.teardown(self.make_context("boot-teardown"))
        self.instance = None

    def rebind(self, binding: DeployBinding) -> None:
        """Swap the deploy binding (hub deploy state changed). Contexts made
        after this read the new binding; release identity is untouched —
        graphs a rebind introduces are holes the mint fills (partial-hit)."""
        self.binding = binding

    @property
    def holes(self) -> tuple[Any, ...]:
        """The ordered mint work-list (pgw#1371's input): torchcg ``Hole``
        rows in canonical document order, each carrying its full
        ``GraphRecord`` (graph hash + ingress) and reason. The mint arms
        each landed artifact via ``self.adoption.arm(record, path)``."""
        return tuple(self.adoption.holes) if self.adoption is not None else ()

    # -- serving ------------------------------------------------------------

    def dispatch(
        self,
        function: str,
        payload: Any,
        *,
        request_id: str,
        ctx: Optional[ServeContext] = None,
    ) -> Any:
        """Route one request to the named handler — function name IS the route.

        ``payload`` is the wire mapping (or an already-typed struct); it
        decodes against the handler's own msgspec schema, so a payload that
        does not fit is a typed ``msgspec.ValidationError`` naming the
        field, before author code runs.
        """
        if self.instance is None:
            raise RuntimeError("dispatch(): boot the endpoint first (setup())")
        handler = self.loaded.handlers.get(function)
        if handler is None:
            raise ServeDispatchError(
                f"{self.loaded.module_name} serves no function {function!r} "
                f"(functions: {sorted(self.loaded.handlers)})"
            )
        if isinstance(payload, handler.payload_type):
            decoded = payload
        elif isinstance(payload, (bytes, bytearray)):
            decoded = msgspec.json.decode(payload, type=handler.payload_type)
        else:
            decoded = msgspec.convert(payload, type=handler.payload_type)
        if ctx is None:
            ctx = self.make_context(request_id)
        return handler.fn(self.instance, ctx, decoded)


__all__ = ["EndpointHost", "ServeDispatchError"]
