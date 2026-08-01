"""pgw#828 — the delegated mint child runs the endpoint's warm job with the
SAME RequestContext the serving path builds.

Measured on a real L4 (gen-worker 0.85.0, release ``4f32ea53dd81adc1d8501e14``,
flag DISARMED so this is the dynamo route, pod ``zs8p1k4x0jlzp5``, lane
``w8a8-lora64``): the child LOADS the pipeline in **16.45 s** — pgw#816's fix
holding on this route too — and then dies in the endpoint's own warm job:

    jit_compile     child:load  completed  seconds=16.45
    jit_compile     aborted     attempt=1 status=crashed total_s=16.58
                                phases={'load': 16.455, 'warmup_forward': 0.0}
    self_mint_abort delegated_crashed (phase=warmup_forward, deterministic)

      File ".../mint_child.py", line 339, in _run_warm_job
        out = bound(**kwargs)
      File ".../sdxl/main.py", line 326, in generate
        resolved = ctx.slots["pipeline"]
      KeyError: 'pipeline'

The child hand-rolled its own ``RequestContext`` with **no slots, no models
and no root slot**::

    ctx = RequestContext(request_id=f"mint-child-{spec.name}",
                         local_output_dir=tmp, boot_warmup=True)

so ``ctx.slots`` was an empty table and the endpoint's one declared slot was
a ``KeyError``. Nothing was missing from the wire: the child holds the SPEC,
and warm-shape slot resolution (``run=None``) needs only the spec.

With pgw#827 blocking the AOT route, this blocked the other one — between
them no mint route on the platform could publish a cell.

The fix is one construction (``warmup.warm_context``) called by the executor's
boot-warm path, the executor's background-mint seed, and the child. The
equivalence test below is the part that stops the next drift: it asserts the
two are the SAME construction, not merely that both are non-empty. The graphs
the child traces are the graphs the parent must later hit, so a child that
resolved DIFFERENT slot defaults would trace different shapes and the parent's
proof would miss — the "mint succeeded, no artifact" class (pgw#815).
"""

from __future__ import annotations

import tempfile
from typing import Any, List

import pytest

from gen_worker import mint_child, registry, warmup
from gen_worker.request_context import RequestContext

from harness import toy_endpoints


def _spec(name: str) -> Any:
    specs = registry.collect_from_namespace(toy_endpoints)
    for spec in specs:
        if spec.name == name:
            return spec
    raise AssertionError(f"{name!r} not in {[s.name for s in specs]}")


@pytest.fixture
def spec() -> Any:
    """A real endpoint spec whose handler reads ``ctx.slots["pipeline"]`` —
    the exact sdxl shape (``sdxl/main.py:326``)."""
    return _spec("warm-slot-echo")


# ---------------------------------------------------------------------------
# THE RED TEST — the child's own warm job, at the real seam
# ---------------------------------------------------------------------------


def test_the_childs_warm_job_resolves_the_endpoints_declared_slots(spec) -> None:
    """RED at HEAD with the pod's exact sentence, ``KeyError: 'pipeline'``."""
    instance = spec.cls()
    instance.pipeline_path = "/tmp/does-not-matter"
    jobs = mint_child._warm_jobs([spec])
    job = next(j for j in jobs if j.spec.name == spec.name)

    # No exception is the assertion: the handler dereferences
    # `ctx.slots["pipeline"]` and returns.
    mint_child._run_warm_job(instance, job, {}, "w8a8")


def test_the_child_and_the_executor_build_the_SAME_context(spec) -> None:
    """The structural pin. Two constructions of one thing were free to drift,
    and did — four times now (pgw#816, #822, #825, #827) in this exact class.
    """
    with tempfile.TemporaryDirectory() as tmp:
        child = warmup.warm_context(
            spec, request_id="mint-child-x", local_output_dir=tmp,
            lane="w8a8", config={})
        served = warmup.warm_context(
            spec, request_id="boot-warmup-x", local_output_dir=tmp,
            lane="w8a8", config={})
    assert sorted(child.slots) == sorted(served.slots)
    assert sorted(child.slots) == sorted(spec.slots)
    for name in spec.slots:
        assert child.slots[name].ref == served.slots[name].ref
        assert child.slots[name].defaults == served.slots[name].defaults
    assert child.models == served.models
    assert child.models, "the child's ctx.models was empty too"


def test_the_root_slot_and_defaults_are_reachable_in_the_child(spec) -> None:
    """The two latent siblings of the reported crash: ``ctx.models`` was also
    empty, and ``ctx.defaults`` raised ``ValueError`` on 0 resolved slots."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = warmup.warm_context(
            spec, request_id="mint-child-x", local_output_dir=tmp)
    assert ctx.defaults is not None
    assert warmup.spec_root_slot(spec) == "pipeline"


def test_the_executor_keeps_ONE_implementation() -> None:
    """The executor's own names are aliases of the factory's, so a future
    edit cannot fix one path and leave the other behind."""
    from gen_worker import executor

    assert executor._resolve_slots_kwargs is warmup.resolved_slots_kwargs
    assert executor._spec_root_slot is warmup.spec_root_slot


def test_a_slotless_endpoint_still_warms(spec) -> None:
    """An endpoint that declares no slots must not gain a resolution error —
    the empty table is the correct answer there, and always was."""
    slotless = _spec("echo")
    with tempfile.TemporaryDirectory() as tmp:
        ctx: RequestContext[Any] = warmup.warm_context(
            slotless, request_id="mint-child-x", local_output_dir=tmp)
    assert dict(ctx.slots) == {}


def test_the_child_never_imports_the_executor() -> None:
    """``warm_context`` lives in ``warmup`` on purpose: the child must not
    pull the whole arming brain in to build a context (mint_child.py's own
    standing rule, and the reason the constants there are literals)."""
    import inspect

    source = inspect.getsource(mint_child)
    offenders: List[str] = [
        line.strip() for line in source.splitlines()
        if "import" in line and "executor" in line]
    assert offenders == [], offenders
