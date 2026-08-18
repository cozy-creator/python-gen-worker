"""The endpoint serve host (pgw#1372): eager-first boot, adopt as a bolt-on.

Boot, in order (Paul's flow, pgw#1367): instantiate the author's class, call
``setup(ctx)`` — the author's own code loads its pipeline from
``ctx.checkpoint_dir`` and IS the serve host — then, when release metadata
and a store exist, ``adopt()`` swaps compiled graphs in for the active
lane's target modules on the author's own objects. Holes stay eager and are
handed to the background mint (pgw#1371) as the ordered
``LaneAdoption.holes`` list. No trace, no derivation, no compile happens
here, ever; the exact-env audit refuses loudly BEFORE any artifact is
touched.

Telemetry: ``setup`` records a ``model_load`` span and ``adopt`` records the
``adopt_pull`` span (``graphs_from=release`` + per-outcome counts) — the
new-flow replacement for the deleted keyset/derive span.
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
    author's live instance, and the adoption report. Author state lives on
    the author's instance, untouched.
    """

    def __init__(
        self,
        loaded: LoadedEndpoint,
        binding: DeployBinding,
        *,
        lane_name: str = "",
        output_dir: Optional[Path] = None,
        context_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.loaded = loaded
        self.binding = binding
        self.lane = loaded.lane(lane_name)
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

    def setup(self) -> None:
        """Instantiate the author's class and run its ``setup(ctx)``.

        The EAGER path ends here: after setup the endpoint serves, compiled
        or not — everything below is a bolt-on.
        """
        started = time.monotonic()
        self.instance = self.loaded.cls()
        ctx = self.make_context("boot-setup", boot_warmup=True)
        self.instance.setup(ctx)
        boot_stages.record_ending_now(
            boot_stages.Stage.MODEL_LOAD,
            duration_ms=int((time.monotonic() - started) * 1000),
            label=self.loaded.module_name,
            checkpoint=self.binding.checkpoint_ref,
        )

    def rebind(self, binding: DeployBinding) -> None:
        """Swap the deploy binding (hub deploy state changed). Contexts made
        after this read the new binding; release identity is untouched —
        graphs a rebind introduces are holes the mint fills (partial-hit)."""
        self.binding = binding

    def roots(self) -> Dict[str, Any]:
        """The author's namespace for lane target paths.

        The convention discovery uses at publish time, restated at adopt
        time: a pipeline-shaped attribute (anything with a ``components``
        mapping) contributes its components at top level — ``"unet"`` is
        ``pipe.unet`` — and bare module attributes contribute by name. A
        name two attributes both claim is refused: adoption must not guess
        which module the author meant.
        """
        import torch

        if self.instance is None:
            raise RuntimeError("roots(): boot the endpoint first (setup())")
        roots: Dict[str, Any] = {}

        def claim(name: str, value: Any, source: str) -> None:
            if name in roots and roots[name] is not value:
                raise RuntimeError(
                    f"root {name!r} is claimed twice (latest by {source}); "
                    f"adoption cannot guess which module the author meant"
                )
            roots[name] = value

        for attr, value in vars(self.instance).items():
            components = getattr(value, "components", None)
            if isinstance(components, Mapping):
                for name, component in components.items():
                    claim(str(name), component, f"{attr}.components")
            elif isinstance(value, torch.nn.Module):
                claim(attr, value, "instance attribute")
        return roots

    # -- adoption -----------------------------------------------------------

    def adopt(
        self,
        store: Any,
        document: Any,
        sm: str,
        *,
        loader: Any,
        artifacts_dir: Path,
        installed: Optional[Mapping[str, str]] = None,
    ) -> Any:
        """Adopt-first boot, after ``setup``: pull ``[release x sm]``, swap in.

        ``document`` is the release's stamped graph metadata
        (``GraphSetDocument``); ``None`` or an eager-permanent document is a
        clean no-op — the endpoint stays on the eager bridge. The exact-env
        audit runs INSIDE ``torchcg.adopt_lane`` and its
        ``EnvironmentMismatch`` propagates loudly (a build-system bug is not
        a compat decision); the refusal is recorded on the span before it
        leaves. Everything else is partial-hit: what exists arms, the rest
        is the ordered ``holes`` handoff to the pgw#1371 mint.
        """
        from .._vendor.torchcg import EnvironmentMismatch
        from .._vendor.torchcg.adopt import adopt_lane
        from .._vendor.torchcg.graph_identity import EnvIdentity, installed_closure

        if self.instance is None:
            raise RuntimeError("adopt(): boot the endpoint first (setup())")
        started = time.monotonic()

        def span(**attrs: object) -> None:
            boot_stages.record_ending_now(
                boot_stages.Stage.ADOPT_PULL,
                duration_ms=int((time.monotonic() - started) * 1000),
                label=self.loaded.module_name,
                sm=sm,
                **attrs,
            )

        if document is None or getattr(document, "eager_permanent", False):
            span(graphs_from="absent" if document is None else "eager_permanent")
            self.adoption = None
            return None
        if self.lane is None:
            raise RuntimeError(
                "adopt(): this endpoint declared no lanes; an eager-permanent "
                "endpoint has nothing to adopt"
            )
        lane_name = str(getattr(self.lane, "name"))
        installed_map = dict(installed) if installed is not None else installed_closure()
        try:
            adoption = adopt_lane(
                store,
                document,
                lane_name,
                self.roots(),
                sm,
                loader=loader,
                artifacts_dir=artifacts_dir,
                installed=installed_map,
            )
            # The mint-written requirements manifest is an AUDIT assertion
            # (exact-env ruling): every adopted artifact restates what its
            # mint linked, and a divergence is the build system contradicting
            # itself — refuse loudly, never adopt-and-hope.
            env = EnvIdentity(closure=document.closure, sm=sm)
            for record in adoption.adopted:
                manifest = store.get_manifest(record.graph, env)
                if manifest is not None:
                    manifest.assert_environment(installed_map, sm=sm)
        except EnvironmentMismatch as exc:
            span(graphs_from="release", lane=lane_name, refusal="environment_mismatch")
            logger.error("adopt refused: %s", exc)
            raise
        span(
            graphs_from="release",
            lane=lane_name,
            artifact_from_store=len(adoption.adopted),
            artifact_from_eager=len(adoption.holes),
            ambiguous=len(adoption.ambiguous),
        )
        self.adoption = adoption
        return adoption

    @property
    def holes(self) -> tuple[Any, ...]:
        """The ordered mint work-list (pgw#1371's input): torchcg ``Hole``
        rows in canonical document order, each carrying its full
        ``GraphRecord`` (graph hash + target path + ingress) and reason."""
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
