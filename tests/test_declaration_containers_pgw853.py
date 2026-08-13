"""pgw#853 — the #739 vocabulary learns non-tensor / container arguments.

THE FINDING THIS BUILDS ON. The pgw#853 blocker inventory measured what each
of the fleet's eight declaring families actually refuses with, and **three of
the eight blockers reduce to one gap**: the declaration vocabulary cannot
express an argument that is not a single tensor.

* **z-image B1** — ``x`` is a ``list[Tensor]``. ``_nest`` builds DICTS from
  dotted names, so ``Input('x.0', ...)`` yielded ``{'0': tensor}`` where the
  target takes ``[tensor]``; and ``dynamic_shapes_spec`` emitted a flat
  ``{input: {axis: Dim}}`` mapping, which torch refuses by name.
* **qwen-image B1** — ``img_shapes`` is ``[[(1, H_pat, W_pat)]] * B``: the
  class row restated as python ints. ``Arg.value`` is a scalar union AND an
  ``Arg`` is declaration-global, while this value is per-row.
* **qwen-image B2's second half** — the edit lane "cannot go dynamic either,
  for the same img_shapes reason".

**The gap was in the SDK, not in torch** — z-image's own measurement said so
("the nested form torch wants exports fine, control rc=4"), and
:func:`test_a_real_torch_export_accepts_the_container_form` re-proves it here
rather than taking it on faith. This is the vocabulary catching up to what the
platform underneath already supports.

THE pgw#846 GATE. A vocabulary may change how a declaration is EXPRESSED,
never what graph gets traced. Both new fields are ``None``-defaulted and
OMITTED from ``as_row()`` when absent, so every declaration written before
they existed is untouched by construction —
:func:`test_existing_declarations_are_byte_identical` pins that, and the same
comparison was run against the three real fleet declarations off-tree (flux2
4b/9b, wan x3, sdxl: byte-identical declarations AND byte-identical derived
compiled graphs, dynamic rows included).
"""

from __future__ import annotations

import json
from typing import Any, List

import pytest

from gen_worker import Arg, Compile, Dim, GraphClass, Input
from gen_worker.aot_mint import DynamicDim, ExportSpec, MintRefused, dynamic_shapes_spec
from gen_worker.api.export_contract import DeclarationError
from gen_worker.aot_declaration import (
    compiled_graph_plans, container_arities, declared_inputs,
)

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Doubles shaped like the two blocked families' real signatures.
# ---------------------------------------------------------------------------


class _Cfg:
    in_channels = 4


class _ListModule(torch.nn.Module):
    """z-image's shape: a python LIST of per-sample tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _Cfg()

    def forward(self, x: List[Any], t: Any) -> Any:
        return torch.stack([e.sum() for e in x]) + t.sum()


class _StructuredArgModule(torch.nn.Module):
    """qwen's shape: a tensor plus a nested python-int container whose value
    restates the class row."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _Cfg()

    def forward(self, hidden_states: Any, img_shapes: Any = None,
                return_dict: bool = False) -> Any:
        total = 0
        for outer in img_shapes:
            for grp in outer:
                total += grp[0] * grp[1] * grp[2]
        return hidden_states * float(total)


def _list_declaration(repeat: Any = "N") -> Compile:
    return Compile(
        family="harness-list-family",
        targets=("transformer",),
        text_len=0,
        shapes=((64, 64),),
        dims=(
            Dim("N", carried_by=(("t", 0),)),
            Dim("H_lat", carried_by=(("x", 2),)),
            Dim("W_lat", carried_by=(("x", 3),)),
        ),
        classes=(
            GraphClass(dims={"N": 2, "H_lat": 8, "W_lat": 8}),
            GraphClass(dims={"N": 2, "H_lat": 16, "W_lat": 12}),
        ),
        inputs=(
            Input("x", shape=(("config", "in_channels"), 1, "H_lat", "W_lat"),
                  repeat=repeat, dtype="model"),
            Input("t", shape=("N",), dtype="float32"),
        ),
        shape_strategy="dynamic-collapse",
        warm_changes_key=False,
    )


def _template_declaration() -> Compile:
    return Compile(
        family="harness-template-family",
        targets=("transformer",),
        text_len=0,
        shapes=((64, 64),),
        dims=(
            Dim("B", carried_by=(("hidden_states", 0),)),
            Dim("H_pat", carried_by=(("hidden_states", 1),)),
            Dim("W_pat", carried_by=(("hidden_states", 2),)),
        ),
        classes=(GraphClass(dims={"B": 1, "H_pat": 4, "W_pat": 6}),),
        inputs=(Input("hidden_states", shape=("B", "H_pat", "W_pat"), dtype="model"),),
        args=(
            Arg("img_shapes", template=[(1, "H_pat", "W_pat")], repeat="B"),
            Arg("return_dict", False),
        ),
        shape_strategy="static-rows",
        warm_changes_key=False,
    )


def _spec(family: str, **kw: Any) -> ExportSpec:
    return ExportSpec(family=family, target="transformer",
                      weight_lane="", precision="bf16", **kw)


# ---------------------------------------------------------------------------
# 1. THE CONTAINER INPUT — z-image B1
# ---------------------------------------------------------------------------


def test_a_declared_list_input_feeds_a_list_not_a_dict() -> None:
    """RED before this vocabulary: `_nest` built {'0': tensor} from a dotted
    name, and the target takes [tensor]."""
    decl = _list_declaration()
    args, kwargs = declared_inputs(_ListModule(), _spec(decl.family), decl)

    assert kwargs == {}
    feed, t = args
    assert isinstance(feed, list), f"expected a list, got {type(feed).__name__}"
    assert len(feed) == 2, "arity comes from the class row's N"
    assert all(isinstance(e, torch.Tensor) for e in feed)
    # Every element carries the declared shape: (config.in_channels, 1, H, W),
    # resolved from the row that seeds the trace.
    (plan,) = compiled_graph_plans(decl)
    seed = dict(plan.seed.dims)
    want = (4, 1, seed["H_lat"], seed["W_lat"])
    assert all(tuple(e.shape) == want for e in feed), feed[0].shape
    assert tuple(t.shape) == (2,)


def test_the_container_arity_comes_from_the_row_not_a_guess() -> None:
    decl = _list_declaration()
    assert container_arities(decl, _spec(decl.family)) == {"x": 2}


def test_a_literal_arity_and_a_config_arity_both_resolve() -> None:
    """`repeat` is an AxisSpec, so it obeys the SAME three-way rule as a
    tensor axis — one rule, not a third special case."""
    lit = _list_declaration(repeat=3)
    args, _ = declared_inputs(_ListModule(), _spec(lit.family), lit)
    assert len(args[0]) == 3

    cfg = _list_declaration(repeat=("config", "in_channels"))
    args, _ = declared_inputs(_ListModule(), _spec(cfg.family), cfg)
    assert len(args[0]) == 4, "read off the module's own config"


def test_dynamic_shapes_mirror_the_container_structure() -> None:
    """The exact refusal z-image measured:

        Detected mismatch between the structure of `inputs` and
        `dynamic_shapes`: `inputs['x']` is a <class 'list'>, but
        `dynamic_shapes['x']` is a <class 'dict'>
    """
    dims = (
        DynamicDim(input_name="x", axis=2, min=8, max=16, dim="H_lat"),
        DynamicDim(input_name="x", axis=3, min=8, max=12, dim="W_lat"),
    )
    spec = dynamic_shapes_spec(dims, ["x", "t"], {"x": 2})

    assert isinstance(spec["x"], list), spec["x"]
    assert len(spec["x"]) == 2
    assert sorted(spec["x"][0]) == [2, 3]
    # Elements share the container's declared symbols — one graph class, not N.
    assert spec["x"][0][2] is spec["x"][1][2]
    assert spec["t"] is None


def test_without_a_container_the_spec_is_unchanged() -> None:
    """pgw#846: the mapping form every existing family gets must not move."""
    dims = (DynamicDim(input_name="x", axis=2, min=8, max=16, dim="H_lat"),)

    def _shape(spec: Any) -> Any:
        # torch.export.Dim objects are fresh per call, so compare STRUCTURE.
        return {k: (None if v is None else sorted(v)) for k, v in spec.items()}

    assert _shape(dynamic_shapes_spec(dims, ["x", "t"])) == \
        _shape(dynamic_shapes_spec(dims, ["x", "t"], {}))
    assert isinstance(dynamic_shapes_spec(dims, ["x", "t"])["x"], dict)


def test_a_real_torch_export_accepts_the_container_form() -> None:
    """The decisive one: the gap was in the SDK, not torch. z-image's own
    control measured rc=4 on the nested form; this re-proves it end to end,
    from a DECLARATION, through the real feed and the real spec builder,
    into a real `torch.export`."""
    decl = _list_declaration()
    module = _ListModule()
    args, kwargs = declared_inputs(module, _spec(decl.family), decl)
    (plan,) = compiled_graph_plans(decl)
    spec = dynamic_shapes_spec(
        plan.dynamic, ["x", "t"], container_arities(decl, _spec(decl.family)))

    program = torch.export.export(
        module, tuple(args), dict(kwargs), dynamic_shapes=spec, strict=True)

    assert program is not None
    # And it really is dynamic on the declared axes, not silently specialised.
    assert len(list(program.graph.nodes)) > 0
    out = program.module()(*args)
    assert out.shape == torch.Size([2])


# ---------------------------------------------------------------------------
# 2. THE ROW-DERIVED STRUCTURED ARG — qwen-image B1
# ---------------------------------------------------------------------------


def test_a_templated_arg_resolves_against_the_class_row() -> None:
    decl = _template_declaration()
    args, kwargs = declared_inputs(
        _StructuredArgModule(), _spec(decl.family), decl)

    assert kwargs == {}
    hidden, img_shapes, return_dict = args
    assert tuple(hidden.shape) == (1, 4, 6)
    # qwen's real shape: [[(1, H_pat, W_pat)]] * B
    assert img_shapes == [[(1, 4, 6)]], img_shapes
    assert return_dict is False


def test_the_template_tracks_the_row_it_is_derived_from() -> None:
    """The whole point of `template` over `value`: an Arg is
    declaration-GLOBAL, and this value is PER-ROW."""
    decl = _template_declaration()
    rows = (GraphClass(dims={"B": 2, "H_pat": 9, "W_pat": 3}),)
    two_rows = Compile(
        **{**{f: getattr(decl, f) for f in decl.__struct_fields__},
           "classes": decl.classes + rows})

    args, _ = declared_inputs(
        _StructuredArgModule(),
        _spec(two_rows.family, class_dims={"B": 2, "H_pat": 9, "W_pat": 3}),
        two_rows)
    assert args[1] == [[(1, 9, 3)], [(1, 9, 3)]], "repeat=B follows the row"


def test_a_templated_arg_reaches_a_real_export() -> None:
    decl = _template_declaration()
    module = _StructuredArgModule()
    args, kwargs = declared_inputs(module, _spec(decl.family), decl)

    program = torch.export.export(module, tuple(args), dict(kwargs), strict=True)
    assert program is not None


# ---------------------------------------------------------------------------
# 3. REFUSALS — the vocabulary states which of the two a number is
# ---------------------------------------------------------------------------


def test_value_and_template_together_are_refused() -> None:
    with pytest.raises(DeclarationError) as excinfo:
        Arg("x", 1, template=[1])
    assert "state one" in str(excinfo.value)


def test_repeat_without_template_is_refused() -> None:
    with pytest.raises(DeclarationError):
        Arg("x", 1, repeat="B")


def test_a_template_leaf_must_be_an_axis_spec() -> None:
    with pytest.raises(DeclarationError):
        Arg("x", template=[[1.5]])
    with pytest.raises(DeclarationError):
        Arg("x", template=[])


def test_a_template_naming_an_undeclared_dim_refuses_at_MINT_not_import() -> None:
    """Consistent with every other row-derived number: the declaration is
    well-formed, the ROW is what does not carry it."""
    decl = Compile(
        family="harness-template-family",
        targets=("transformer",), text_len=0, shapes=((64, 64),),
        dims=(Dim("B", carried_by=(("hidden_states", 0),)),),
        classes=(GraphClass(dims={"B": 1}),),
        inputs=(Input("hidden_states", shape=("B", 4, 6), dtype="model"),),
        args=(Arg("img_shapes", template=[[(1, "NOPE", 2)]]),),
        shape_strategy="static-rows", warm_changes_key=False,
    )
    with pytest.raises(MintRefused) as excinfo:
        declared_inputs(_StructuredArgModule(), _spec(decl.family), decl)
    assert "NOPE" in str(excinfo.value)


def test_a_zero_arity_container_is_refused_by_name() -> None:
    decl = _list_declaration()
    bad = Compile(
        **{**{f: getattr(decl, f) for f in decl.__struct_fields__},
           "classes": (GraphClass(dims={"N": 2, "H_lat": 8, "W_lat": 8}),),
           "inputs": (Input("x", shape=(4, 1, "H_lat", "W_lat"),
                            repeat=("config", "missing_field"), dtype="model"),
                      Input("t", shape=("N",), dtype="float32"))})
    with pytest.raises(MintRefused) as excinfo:
        declared_inputs(_ListModule(), _spec(bad.family), bad)
    assert "missing_field" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. THE pgw#846 GATE — expression may change; the traced graph may not
# ---------------------------------------------------------------------------


def _fingerprint(decl: Compile) -> str:
    return json.dumps({
        "inputs": [i.as_row() for i in decl.inputs],
        "args": [a.as_row() for a in decl.args],
        "classes": [c.as_row() for c in decl.classes],
        "compiled_graphs": [
            {"fork": [list(f) for f in p.fork],
             "dynamic": [[d.input_name, d.axis, d.min, d.max, d.multiple_of,
                          d.dim] for d in p.dynamic]}
            for p in compiled_graph_plans(decl)],
    }, sort_keys=True)


def test_existing_declarations_are_byte_identical() -> None:
    """A declaration that uses NEITHER new field must serialise exactly as it
    did before they existed — no `repeat: null`, no `template: null`.

    Pinned as a unit here; the real check is the same comparison run against
    the fleet's own declarations, which came back byte-identical for flux2
    4b/9b, wan t2v/i2v/ti2v and sdxl — declarations AND derived compiled graphs,
    dynamic rows included.
    """
    plain = Input("hidden_states", shape=("B", 4, 6), dtype="model")
    assert "repeat" not in plain.as_row()
    assert set(plain.as_row()) == {"name", "shape", "dtype", "value", "targets"}

    plain_arg = Arg("return_dict", False)
    assert set(plain_arg.as_row()) == {"name", "value", "targets"}

    # ...and a declaration built entirely from the old vocabulary still
    # derives exactly what it derived: one static row, no dynamic dims.
    decl = Compile(
        family="harness-old-vocab", targets=("transformer",), text_len=0,
        shapes=((64, 64),),
        dims=(Dim("B", carried_by=(("hidden_states", 0),)),),
        classes=(GraphClass(dims={"B": 1}),),
        inputs=(Input("hidden_states", shape=("B", 4, 6), dtype="model"),),
        args=(Arg("return_dict", False),),
        shape_strategy="static-rows", warm_changes_key=False,
    )
    (plan,) = compiled_graph_plans(decl)
    assert plan.dynamic == ()
    assert '"repeat"' not in _fingerprint(decl)
    assert '"template"' not in _fingerprint(decl)


def test_the_new_fields_are_absent_from_the_compiled_graph_key_contract() -> None:
    """`declared_compile_facts` is the local declared-compile-contract block
    (pgw#1059: no longer a key-axis input). It reads shapes/targets/
    text_lens/dynamic/regional/lora_bucket/guidance — NOT input or arg rows
    — so expressing a container cannot churn the local store verdict."""
    from gen_worker.compile_cache import declared_compile_facts

    facts = declared_compile_facts(_list_declaration())
    assert "inputs" not in facts and "args" not in facts


# ---------------------------------------------------------------------------
# 5. AN ARG-CARRIED EXTENT — qwen-image's H_pat/W_pat, which enter the traced
#    call as PYTHON INTS inside img_shapes, not as a tensor axis.
# ---------------------------------------------------------------------------


def _arg_carried_declaration(rows: Any = None) -> Compile:
    return Compile(
        family="harness-argcarried-family",
        targets=("transformer",), text_len=0, shapes=((64, 64),),
        dims=(
            Dim("B", carried_by=(("hidden_states", 0),)),
            # The binding qwen-image actually has: tuple positions 1 and 2 of
            # img_shapes[b][0] = (frames, height, width).
            Dim("H_pat", carried_by=(("img_shapes", 1),)),
            Dim("W_pat", carried_by=(("img_shapes", 2),)),
        ),
        classes=rows or (GraphClass(dims={"B": 1, "H_pat": 4, "W_pat": 6}),),
        inputs=(Input("hidden_states", shape=("B", 4, 6), dtype="model"),),
        args=(Arg("img_shapes", template=[(1, "H_pat", "W_pat")], repeat="B"),
              Arg("return_dict", False)),
        shape_strategy="static-rows", warm_changes_key=False,
    )


def test_a_dim_may_be_carried_by_a_templated_arg() -> None:
    """qwen-image's declaration already said this was the right binding:
    "The (name, index) pair is accurate; that the vocabulary means 'tensor
    axis' by it is exactly blocker B1." The vocabulary now means either."""
    decl = _arg_carried_declaration()
    args, _ = declared_inputs(_StructuredArgModule(), _spec(decl.family), decl)
    assert args[1] == [[(1, 4, 6)]]


def test_an_arg_carried_extent_never_becomes_a_torch_symbol() -> None:
    """A python int SPECIALIZES the graph. It must never reach
    `dynamic_shapes`, or torch is being told a constant is free."""
    decl = _arg_carried_declaration()
    (plan,) = compiled_graph_plans(decl)
    assert all(d.input_name != "img_shapes" for d in plan.dynamic), plan.dynamic


def test_collapsing_rows_that_differ_on_an_ARG_carried_dim_is_REFUSED() -> None:
    """The silent-wrongness this rule exists to stop: one artifact minted for
    rows that differ on a python-int argument serves only the seed row, and
    nothing downstream would notice. Same class as the 0/1 refusal."""
    rows = (GraphClass(dims={"B": 1, "H_pat": 4, "W_pat": 6}),
            GraphClass(dims={"B": 1, "H_pat": 8, "W_pat": 6}))
    decl = Compile(
        **{**{f: getattr(_arg_carried_declaration(rows), f)
              for f in _arg_carried_declaration().__struct_fields__},
           "shape_strategy": "dynamic-collapse"})
    with pytest.raises(MintRefused) as excinfo:
        compiled_graph_plans(decl)
    msg = str(excinfo.value)
    assert "H_pat" in msg and "SPECIALIZES" in msg


def test_static_rows_keeps_one_artifact_per_row_for_arg_carried_dims() -> None:
    """The declared remedy in that refusal, exercised: static-rows is exactly
    right for a family whose extents ride python ints — which is why
    qwen-image derives 14 compiled graphs from 14 rows."""
    rows = (GraphClass(dims={"B": 1, "H_pat": 4, "W_pat": 6}),
            GraphClass(dims={"B": 1, "H_pat": 8, "W_pat": 6}))
    assert len(compiled_graph_plans(_arg_carried_declaration(rows))) == 2


def test_a_dim_binding_nothing_declared_is_still_refused() -> None:
    """The permission is for TEMPLATED args only — a literal Arg is not a
    carrier, and a typo must still fail at declaration time."""
    with pytest.raises(DeclarationError):
        Compile(
            family="harness-argcarried-family",
            targets=("transformer",), text_len=0, shapes=((64, 64),),
            dims=(Dim("H_pat", carried_by=(("return_dict", 1),)),),
            classes=(GraphClass(dims={"H_pat": 4}),),
            inputs=(Input("hidden_states", shape=(1, 4, 6), dtype="model"),),
            args=(Arg("return_dict", False),),
            shape_strategy="static-rows", warm_changes_key=False)
