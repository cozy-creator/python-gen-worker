"""pgw#1346 K11 — the EAGER tier gets a runtime binding, so `inst.tuned` exists.

B5 landed the declaration half: `eager_model_v1` and `render_eager` emit a real
`Model` subclass for a model that declares no graph classes. What no code did
was BUILD one on the serving path — `residency.instance_for` refused any model
with neither an eager module nor an armed cell, and a runner-less model has
neither by construction. So every eager model was unconstructible, `inst.tuned`
was unreachable, and no `ctx.defaults` read on an eager endpoint could migrate.
That single refusal gated anima (5 reads), hidream (14), the auxiliary-model
class, and all 11 weight-bearing boundary endpoints at once.

The tier also needed a SERVING SURFACE. Measured across the boundary batch, 8 of
11 hand a local path to an external binary and 3 use an object the worker built
— so `inst.path` and `inst.pipeline` are what an eager instance is FOR, the way
a typed runner call is what a graph instance is for.
"""

from __future__ import annotations

import pytest

from gen_worker.model.backing import BackingKind
from gen_worker.model.catalog import Joycaption, Musicgen, Sdxl
from gen_worker.model.errors import ModelError, ModelRefusal
from gen_worker.model.residency import instance_for


def test_an_eager_model_is_constructible_at_all() -> None:
    """The K11 unblock, stated as the smallest possible claim.

    Before this, `instance_for` refused every eager model — the model declares
    no runners, so it has no eager module, so the "neither backing" refusal
    fired unconditionally.
    """

    inst = instance_for(Musicgen, ref="hub:cozy/musicgen")
    assert inst.ref == "hub:cozy/musicgen"


def test_the_catalog_stamp_reaches_an_eager_instance() -> None:
    """`inst.tuned` is the whole replacement for `ctx.defaults`, so an eager
    model that cannot be constructed cannot migrate a single defaults read.
    """

    inst = instance_for(Musicgen, ref="hub:cozy/musicgen")
    assert inst.tuned is not None
    assert isinstance(inst.tuned, Musicgen.Tuned)


def test_a_runner_less_model_reports_a_backing_of_NONE_not_a_missing_one() -> None:
    """"Nothing to arm, by declaration" and "not armed yet" are different
    answers, and only one of them is a defect. The backing kind says which.
    """

    inst = instance_for(Joycaption, ref="hub:cozy/joycaption")
    assert inst.backing.kind is BackingKind.NONE


# ---------------------------------------------------------------------------
# The eager serving surface: path (8 of 11) and pipeline (3 of 11)
# ---------------------------------------------------------------------------


def test_the_local_path_is_what_an_external_binary_model_serves_through() -> None:
    inst = instance_for(
        Musicgen, ref="hub:cozy/musicgen", path="/cache/blobs/musicgen"
    )
    assert str(inst.path) == "/cache/blobs/musicgen"


def test_asking_for_a_path_that_was_never_materialized_refuses_by_name() -> None:
    """A handler that shells out with `""` fails somewhere far away and
    confusingly. The answer is only useful at the point it stops being true.
    """

    inst = instance_for(Musicgen, ref="hub:cozy/musicgen")
    with pytest.raises(ModelError) as caught:
        inst.path
    assert caught.value.reason is ModelRefusal.BACKING_MISSING
    assert "no local path" in str(caught.value)


def test_a_constructed_object_rides_the_instance_for_the_models_that_have_one() -> None:
    built = object()
    inst = instance_for(Musicgen, ref="hub:cozy/musicgen", tree=built)
    assert inst.pipeline is built


def test_asking_for_a_pipeline_that_was_never_built_refuses_by_name() -> None:
    inst = instance_for(Musicgen, ref="hub:cozy/musicgen", path="/cache/x")
    with pytest.raises(ModelError) as caught:
        inst.pipeline
    assert caught.value.reason is ModelRefusal.BACKING_MISSING
    assert "carries no constructed object" in str(caught.value)


# ---------------------------------------------------------------------------
# The refusals the tier still owes
# ---------------------------------------------------------------------------


def test_calling_a_runner_on_a_model_that_declares_none_says_exactly_that() -> None:
    """Not "no eager module bound for runner 'x'; it has []", which would send
    the reader hunting for a registration that was never supposed to happen.
    """

    from gen_worker.model.backing import NoGraphBacking

    with pytest.raises(ModelError) as caught:
        NoGraphBacking().invoke("denoiser", None, (), {})  # type: ignore[arg-type]
    assert "declares NO runners" in str(caught.value)
    assert "inst.path" in str(caught.value)


def test_a_GRAPH_model_with_no_backing_still_refuses_as_it_did_before() -> None:
    """The K11 relaxation is scoped to models that declare no runners. A
    declared graph with nothing to run it is still a pod-level failure, and
    quietly handing back an instance would move the error somewhere useless.
    """

    with pytest.raises(ModelError) as caught:
        instance_for(Sdxl, ref="hub:cozy/sdxl", tree=None)
    assert caught.value.reason is ModelRefusal.BACKING_MISSING
    assert "neither an eager module nor an armed cell" in str(caught.value)
