"""Paul 2026-08-02: declare how BOUNDED an endpoint's weight set is.

    "A serving inference endpoint, like SDXL or Z-Image, uses the same set of
     weights every time, and hence can benefit from an NFS drive a lot. A
     conversion-endpoint ... downloads different sets of weights per request,
     so an NFS drive provides no value. We need some way for the orchestrator
     to figure out what endpoints benefit from NFS drives, and which ones do
     not."

WHY IT IS NOT A `use_nfs: bool`. The honest fact is the endpoint's
relationship to its weights, and it decides more than caching. A cached volume
lives in **exactly one datacenter**, so attaching one to an endpoint that
cannot use it does not merely waste disk — it collapses that endpoint's
placement to a single datacenter. Measured: a wan conversion failed sixteen
times against one datacenter while three others had capacity. The annotation
removes a placement constraint that should never have applied.

WHY IT IS DECLARED RATHER THAN INTROSPECTED — with evidence, because
"we could infer it" will be proposed by someone who has not checked:

* **`kind=` cannot answer it.** `kind="inference"` does not imply fixed
  weights: sdxl, z-image, ltx-video-2.3, krea-2 and hidream all declare
  ``Slot.selected_by="model"``, i.e. the REQUEST picks which checkpoint runs.
* **Bindings answer it only by proxy.** A conversion endpoint names weights as
  ordinary payload strings (`payload.huggingface_repo` -> `hf_hub_download`),
  so inferring from bindings means recognising *"a plain string field that
  happens to be a model ref"* — implicit meaning of exactly the kind that has
  already burned this program twice today.
* **The boundedness itself is not in the bindings.** "Bound" is about the
  WORKING SET being finite and reused; a catalog with 500 entries is bound and
  useless to a cache. No binding says that.

WHY UNSET IS NOT A VALUE. Attaching a volume nobody can use costs capacity;
withholding one costs cold-boot seconds. Those are not symmetric, so the
conservative branch is "no volume" — but *undeclared* must stay distinguishable
from a declared ``per_request``, or the gap reads as a fact and nobody ever
annotates it.
"""

from __future__ import annotations

import msgspec
import pytest

from gen_worker import RequestContext
from gen_worker.api.decorators import ATTR, WEIGHT_SETS, endpoint
from gen_worker.discovery.discover import _extract_entries


class _In(msgspec.Struct):
    x: int = 0


class _Out(msgspec.Struct):
    y: int = 0


def _decl(cls):
    return getattr(cls, ATTR)


# ---------------------------------------------------------------------------
# 1. THE VOCABULARY — a kind, not a boolean
# ---------------------------------------------------------------------------


def test_the_three_values_name_the_relationship_not_the_storage() -> None:
    assert WEIGHT_SETS == ("bound", "per_request", "none")


@pytest.mark.parametrize("value", WEIGHT_SETS)
def test_every_declared_value_is_accepted(value: str) -> None:
    @endpoint(weight_set=value)
    class E:
        def go(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out()

    assert _decl(E).weight_set == value


def test_an_undeclared_value_is_refused_and_the_message_teaches() -> None:
    with pytest.raises(ValueError) as excinfo:
        @endpoint(weight_set="use_nfs")
        class E:
            def go(self, ctx: RequestContext, data: _In) -> _Out:
                return _Out()

    msg = str(excinfo.value)
    assert "use_nfs" in msg
    assert "bound" in msg and "per_request" in msg
    # The refusal must say what the axis IS, or the next author guesses again.
    assert "deploy bindings" in msg
    assert "do not guess" in msg


# ---------------------------------------------------------------------------
# 2. THE DEFAULT — undeclared is its own state
# ---------------------------------------------------------------------------


def test_undeclared_stays_None_and_is_never_silently_a_value() -> None:
    """`None` must not collapse into `per_request` just because they lead to
    the same placement today. A consumer has to be able to tell "the author
    said" from "nobody said" — otherwise the fleet looks fully annotated and
    nobody ever finishes annotating it."""
    @endpoint
    class Plain:
        def go(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out()

    assert _decl(Plain).weight_set is None


def test_the_manifest_OMITS_an_undeclared_weight_set() -> None:
    """Omitted, not `null`. The placement side must read absence as absence."""
    @endpoint
    class Plain:
        def go(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out()

    (fn,) = _extract_entries(Plain, "tests.weightset")
    assert "weight_set" not in fn


def test_the_manifest_CARRIES_a_declared_weight_set() -> None:
    """The whole point: the orchestrator reads this off discovery."""
    @endpoint(kind="conversion", weight_set="per_request")
    class Convert:
        def go(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out()

    (fn,) = _extract_entries(Convert, "tests.weightset")
    assert fn["weight_set"] == "per_request"
    assert fn["kind"] == "conversion"


# ---------------------------------------------------------------------------
# 3. IT IS AN ENDPOINT FACT, NOT A GRAPH FACT — identity cannot see it
# ---------------------------------------------------------------------------


def test_weight_set_is_not_a_Compile_field_so_it_cannot_reach_the_cell_key() -> None:
    """Structural, not a promise: the field lives on `@endpoint`, and the cell
    contract is built from `Compile`. Keeping placement off the AOT
    declaration is what makes "it cannot re-key a cell" true by construction.

    The fleet-wide half ran off-tree: all six annotated declarations
    fingerprinted BYTE-IDENTICAL (full serialised form plus derived entries,
    fork coordinates and dynamic rows).
    """
    from gen_worker import Compile
    from gen_worker.compile_cache import declared_contract_facts

    assert "weight_set" not in Compile.__struct_fields__

    @endpoint(weight_set="bound")
    class E:
        def go(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out()

    facts = declared_contract_facts(type("_Cfg", (), {
        "shapes": ((1024, 1024),), "targets": ("unet",), "text_lens": (77,),
        "dynamic": (), "regional": False, "lora_bucket": 0,
        "guidance_scales": (),
    })())
    assert "weight_set" not in facts


# ---------------------------------------------------------------------------
# 4. THE FLEET SHAPES IT HAS TO EXPRESS
# ---------------------------------------------------------------------------


def test_a_bound_endpoint_may_still_let_the_REQUEST_choose() -> None:
    """The counter-example that killed introspection-by-kind: sdxl, z-image,
    ltx and others use `Slot.selected_by="model"`, so an inference endpoint
    DOES pick weights per request — from a bound catalog. Choosing among
    bound weights is still `bound`, and the vocabulary must not force such an
    endpoint to call itself `per_request`."""
    from gen_worker import Hub

    @endpoint(model=Hub("harness/base"), weight_set="bound")
    class Serving:
        def setup(self, model: str) -> None:
            self._m = model

        def go(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out()

    assert _decl(Serving).weight_set == "bound"
    # The real fleet shape this stands for, verified in source rather than
    # constructed here: sdxl, z-image, ltx-video-2.3, krea-2 and
    # hidream-o1-image all declare `Slot(..., selected_by="model")`, so an
    # inference endpoint DOES pick weights per request — from a bound
    # catalog. That is why `kind=` cannot infer this field.


def test_an_endpoint_with_no_weights_at_all_can_say_so() -> None:
    """`none` is not a synonym for `per_request`. They place identically today
    and are different facts — a `none` endpoint needs no weight cache AND no
    model-repo read grants, and recording it stops someone "fixing" it by
    adding a binding."""
    @endpoint(kind="dataset", weight_set="none")
    class Corpus:
        def go(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out()

    assert _decl(Corpus).weight_set == "none"
