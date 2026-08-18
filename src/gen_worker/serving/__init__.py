"""The ship-code-as-is serving layer (pgw#1372, program pgw#1367).

Authors ship code AS-IS: the endpoint class from ``endpoint.toml``'s
``main =`` module is the whole serve surface — no catalog, no ModelSpec, no
family registry, no codegen. The worker loads the class, calls
``setup(ctx)``, and routes requests to handlers by function name. The EAGER
path works standalone and first; adoption of compiled graphs is a bolt-on
that swaps module forwards on the author's own objects (torchcg's
``adopt_lane``) and hands the ordered hole list to the background mint
(pgw#1371).

Seams, deliberately narrow:

* ``RequestContext`` here is the serving surface ``main_v2.py`` uses
  (``ctx.checkpoint_dir``, ``ctx.lane``, ``ctx.checkpoint_defaults(T)``,
  ``ctx.adapter``, ``ctx.is_trace`` + the base survivors).
* Endpoint classes inherit the REQUIRED minimal base
  (:class:`~gen_worker.serving.endpoint.Endpoint` — typed setup/teardown
  hooks only; framework capabilities arrive via ctx ONLY, by ruling).
  Module marking is imperative: ``self.pipe.unet = ctx.compile(self.pipe.unet)``.
* The author-surface lane owns the ``@endpoint`` decorator (DATA: lanes);
  this package only reads what it stamps
  (``loader.ENDPOINT_ATTR`` -> :class:`~gen_worker.serving.loader.EndpointDeclaration`).
* The hub is one ``GraphStore`` implementation behind th#2133's route
  (:mod:`~gen_worker.serving.hub_store`); cozy-local is another
  (torchcg's ``LocalGraphStore``). Same boot code, any store.
"""

from .context import BoundAdapter, DeployBinding, ServeContext
from .endpoint import Endpoint
from .host import EndpointHost, ServeDispatchError
from .loader import (
    ENDPOINT_ATTR,
    EndpointDeclaration,
    EndpointLoadError,
    Handler,
    LoadedEndpoint,
    load_endpoint,
    load_endpoint_module,
)

__all__ = [
    "ENDPOINT_ATTR",
    "BoundAdapter",
    "DeployBinding",
    "Endpoint",
    "EndpointDeclaration",
    "EndpointHost",
    "EndpointLoadError",
    "Handler",
    "LoadedEndpoint",
    "ServeContext",
    "ServeDispatchError",
    "load_endpoint",
    "load_endpoint_module",
]
