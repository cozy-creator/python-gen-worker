"""Removed in gen-worker 0.7.0 (decorator-table-model-bindings).

The ``Annotated[T, ModelRef(...)]`` injection pattern has been replaced by
the ``models={...}`` kwarg on the ``@inference`` class decorator, carrying
:class:`gen_worker.Repo` and :class:`gen_worker.Dispatch` bindings.

Migration::

    # Before (0.6.x):
    from gen_worker import ModelRef, ModelRefSource as Src

    @inference(resources=...)
    def generate(
        ctx,
        pipeline: Annotated[FluxPipeline, ModelRef(Src.FIXED, ref="owner/repo", flavor="nf4")],
        payload: GenerateInput,
    ) -> GenerateOutput: ...

    # After (0.7.0):
    from gen_worker import Repo, inference

    flux = Repo("owner/repo")

    @inference(resources=..., models={"pipeline": flux.flavor("nf4")})
    def generate(ctx, pipeline: FluxPipeline, payload: GenerateInput) -> GenerateOutput: ...

See ``progress.json`` issue #9 (decorator-table-model-bindings) for the full
binding model and migration notes for ``Src.PAYLOAD`` / ``Src.PAYLOAD_REF``.
"""

from __future__ import annotations

_MIGRATION_MESSAGE = (
    "gen_worker.api.injection (ModelRef / ModelRefSource / Src / parse_injection / "
    "InjectionSpec) was removed in gen-worker 0.7.0. Use the models={...} kwarg on "
    "@inference with Repo / Dispatch bindings instead."
)


def __getattr__(name: str):
    raise ImportError(_MIGRATION_MESSAGE)
