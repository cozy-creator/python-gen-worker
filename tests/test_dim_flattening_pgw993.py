"""pgw#993 — a ``Dim`` carried by a ``repeat=`` container is RESOLVABLE.

THE FIELD RECORD. gen-worker 0.93.2, pod `7evvazd2xplzml` (A100-SXM4-80GB),
\\$0.4655: the z-image AOT mint passed every earlier gate, entered the AOT
recipe, and then refused in `trace_graph` — exit=2, `deterministic`, four
identical attempts::

    aot mint refused: entry 'transformer/adapter=true,cfg=true':
    declared-range gate: declared dynamic dim names input 'x', which is not a
    user input of the exported program (inputs: ['cap_feats_0',
    'cap_feats_1', 'lora_a', 'lora_b', 't', 'x_0', 'x_1'])

THE MECHANISM — two SDK features that each work and could not compose.
`Input.repeat` containers (pgw#853) are FLATTENED by `torch.export` into one
positional user input per element, suffixed `_0`, `_1`, …; `Dim.carried_by`
names its input by the DECLARED name. The declared-range gate resolved the
declared name against the exported program and refused on the miss, which
made a `Dim` carried by a repeated container unsatisfiable by construction —
no declaration edit inside the vocabulary could fix it. z-image is the only
family in the fleet using the list vocabulary, so a rented GPU was the only
thing that could find it.

THE INVARIANT. `carried_by` resolves through the SAME flattening
`dynamic_shapes_spec` mirrors the structure with:
`aot_mint.exported_input_names`, fed the arity map the example feed was built
from. One expansion rule, both consumers. Two independent spellings of one
name mapping is the defect class, so the sweep covers `lifted_input_gaps`
too.

NO GPU IS NEEDED to hold this. The first half of this file drives the gate
against a program DOUBLE whose user inputs are the pod's, verbatim; the
second half exports for real on CPU and re-proves the whole path from a
declaration, for both arms of the fork (N=1 and N=2).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import pytest

from gen_worker import Compile, Dim, GraphClass, Input
from gen_worker.aot_contract import DynamicDim, ExportSpec
from gen_worker.aot_declaration import (
    cell_plans, container_arities, declared_inputs,
)
from gen_worker.aot_mint import (
    declared_range_gaps, dynamic_shapes_spec, exported_input_names,
    lifted_input_gaps,
)

# Section 3 exports for real; sections 1 and 2 drive doubles. Module-level, as
# the rest of this suite does it — torch is a dev dependency and CI installs it,
# so this never fires there.
torch = pytest.importorskip("torch")

# The pod's sentence, verbatim. Not paraphrased: this file is the RED proof
# and the RED has to be the thing that was measured.
FIELD_REFUSAL = (
    "declared dynamic dim names input 'x', which is not a user input of the "
    "exported program (inputs: ['cap_feats_0', 'cap_feats_1', 'lora_a', "
    "'lora_b', 't', 'x_0', 'x_1'])")


# ---------------------------------------------------------------------------
# An ExportedProgram double: user inputs are all that this gate reads.
# ---------------------------------------------------------------------------


class _Expr:
    """A sympy-shaped expression carrying one free symbol."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.free_symbols: Tuple[Any, ...] = (self,)

    def __str__(self) -> str:
        return self.name


class _Node:
    def __init__(self, expr: _Expr) -> None:
        self.expr = expr
        self.shape_env = None


class _SymInt:
    def __init__(self, expr: _Expr) -> None:
        self.node = _Node(expr)

    def __str__(self) -> str:
        return str(self.node.expr)


class _Interval:
    def __init__(self, lower: int, upper: int) -> None:
        self.lower = lower
        self.upper = upper


class _Val:
    def __init__(self, shape: Tuple[Any, ...]) -> None:
        self.shape = shape


class _Placeholder:
    def __init__(self, name: str, val: _Val) -> None:
        self.op = "placeholder"
        self.name = name
        self.meta: Dict[str, Any] = {"val": val}


class _Program:
    """Enough ExportedProgram for `_placeholder_shapes` + the range checks."""

    def __init__(self, shapes: Dict[str, Tuple[Any, ...]],
                 ranges: Dict[Any, _Interval]) -> None:
        self.graph_signature = type(
            "_Sig", (), {"user_inputs": tuple(shapes)})()
        graph = type("_Graph", (), {"nodes": [
            _Placeholder(name, _Val(shape)) for name, shape in shapes.items()
        ]})()
        self.graph_module = type("_GM", (), {"graph": graph})()
        self.range_constraints = ranges


def _zimage_arm(arity: int) -> _Program:
    """The pod's program, parameterised by the CFG fork's resolved arity.

    N=2 reproduces `['cap_feats_0', 'cap_feats_1', 'lora_a', 'lora_b', 't',
    'x_0', 'x_1']` exactly; N=1 is the `cfg=false` arm, where torch still
    suffixes — a one-element container exports as `x_0`, never `x`
    (measured, and re-measured on a real export at the bottom of this file).
    """
    height, width, caption = _Expr("s0"), _Expr("s1"), _Expr("s2")
    shapes: Dict[str, Tuple[Any, ...]] = {}
    for index in range(arity):
        shapes[f"x_{index}"] = (
            4, 1, _SymInt(height), _SymInt(width))
        shapes[f"cap_feats_{index}"] = (_SymInt(caption), 1024)
    shapes["lora_a"] = (128, 64)
    shapes["lora_b"] = (64, 128)
    shapes["t"] = (arity,)
    return _Program(shapes, {
        height: _Interval(16, 128),
        width: _Interval(16, 128),
        caption: _Interval(1, 512),
    })


ZIMAGE_DIMS: Tuple[DynamicDim, ...] = (
    DynamicDim("x", 2, 16, 128, multiple_of=2, dim="H_lat"),
    DynamicDim("x", 3, 16, 128, multiple_of=2, dim="W_lat"),
    DynamicDim("cap_feats", 0, 1, 512, dim="T_cap"),
)


def _arities(arity: int) -> Dict[str, int]:
    return {"x": arity, "cap_feats": arity}


# ---------------------------------------------------------------------------
# 1. THE REFUSAL, AND THE FIX
# ---------------------------------------------------------------------------


def test_the_double_reproduces_the_pods_refusal_verbatim() -> None:
    """The double is faithful: without the arity map the gate still says the
    sentence the mint died on, character for character."""
    gaps = declared_range_gaps(_zimage_arm(2), ZIMAGE_DIMS)

    assert FIELD_REFUSAL in gaps, gaps


def test_a_dim_carried_by_a_container_resolves_on_the_cfg_doubled_arm() -> None:
    """RED before pgw#993: the gate had no arity map at all, so `x` was looked
    up under its declared name and every z-image mint refused."""
    assert declared_range_gaps(_zimage_arm(2), ZIMAGE_DIMS, _arities(2)) == []


def test_the_same_dim_resolves_on_the_single_element_arm() -> None:
    """Both arms of the fork, because a fix for N=2 that breaks N=1 is not a
    fix — `cfg=false` is the arm the declaration edit would have had to
    sacrifice."""
    assert declared_range_gaps(_zimage_arm(1), ZIMAGE_DIMS, _arities(1)) == []


def test_a_container_element_that_specialized_is_still_refused() -> None:
    """The gate must check EVERY element, not merely find one. A per-element
    pin is exactly the pgw#704 B2 defect wearing a container."""
    program = _zimage_arm(2)
    shape = list(program.graph_module.graph.nodes[2].meta["val"].shape)
    assert program.graph_module.graph.nodes[2].name == "x_1"
    shape[2] = 64
    program.graph_module.graph.nodes[2].meta["val"].shape = tuple(shape)

    gaps = declared_range_gaps(program, ZIMAGE_DIMS, _arities(2))

    assert len(gaps) == 1, gaps
    assert gaps[0].startswith("x_1[2] exported as the STATIC value 64"), gaps


def test_a_name_that_is_genuinely_absent_still_refuses() -> None:
    """The gate keeps its teeth: expansion resolves declared names, it does
    not stop resolving them."""
    gaps = declared_range_gaps(
        _zimage_arm(2), (DynamicDim("nope", 0, 2, 8),), _arities(2))

    assert gaps and "not a user input" in gaps[0], gaps
    gaps = declared_range_gaps(
        _zimage_arm(2), (DynamicDim("x", 2, 16, 128),), {"x": 3})
    assert any("flattened element 'x_2'" in gap for gap in gaps), gaps


def test_a_plain_input_is_untouched_by_the_expansion() -> None:
    """pgw#846: every declaration written before containers existed resolves
    exactly as it did — a non-container name is its own exported name."""
    program = _Program(
        {"sample": (2, 4, _SymInt(_Expr("s9")), 64)},
        {})
    program.range_constraints = {
        program.graph_module.graph.nodes[0].meta["val"].shape[2].node.expr:
            _Interval(8, 32)}
    dims = (DynamicDim("sample", 2, 8, 32),)

    assert declared_range_gaps(program, dims) == []
    assert declared_range_gaps(program, dims, {"x": 2}) == []


# ---------------------------------------------------------------------------
# 2. ONE EXPANSION RULE, SHARED BY BOTH CONSUMERS
# ---------------------------------------------------------------------------


def test_the_expansion_rule_is_the_one_torch_uses() -> None:
    assert exported_input_names("x") == ("x",)
    assert exported_input_names("x", {}) == ("x",)
    assert exported_input_names("x", {"other": 2}) == ("x",)
    assert exported_input_names("x", {"x": 1}) == ("x_0",)
    assert exported_input_names("x", {"x": 3}) == ("x_0", "x_1", "x_2")


def test_the_spec_builder_and_the_gate_expand_identically() -> None:
    """The invariant, asserted rather than described: the structure
    `dynamic_shapes_spec` mirrors and the names the gate resolves come out of
    the SAME call. Two spellings of one mapping is what pgw#993 was."""
    dims = (DynamicDim("x", 2, 8, 16, dim="H_lat"),)
    for arity in (1, 2, 5):
        spec = dynamic_shapes_spec(dims, ["x", "t"], {"x": arity})
        assert isinstance(spec["x"], list)
        assert len(spec["x"]) == len(exported_input_names("x", {"x": arity}))


def test_a_lifted_input_declared_as_a_container_resolves_too() -> None:
    """The sweep (pgw#993 acceptance 4): the lifted-input gate resolved names
    the same wrong way. Nothing declares a repeated adapter today, which is
    precisely the argument — z-image's `carried_by` had no user either, until
    it cost \\$0.4655 to find out."""
    spec = ExportSpec(
        family="harness", target="transformer", weight_lane="",
        precision="bf16", lifted_inputs=("lora_a",))
    program = _Program({"lora_a_0": (128, 64), "lora_a_1": (128, 64)}, {})

    assert lifted_input_gaps(program, spec, {"lora_a": 2}) == []
    gaps = lifted_input_gaps(program, spec)
    assert gaps and "not a user input" in gaps[0], gaps


# ---------------------------------------------------------------------------
# 3. THE SAME THING ON A REAL EXPORT (CPU, no GPU)
# ---------------------------------------------------------------------------


class _ListModule(torch.nn.Module):
    """z-image's shape: a python LIST of per-sample tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.config = type("_Cfg", (), {"in_channels": 4})()

    def forward(self, x: List[Any], t: Any) -> Any:
        return torch.stack([e.sum() for e in x]) + t.sum()


def _list_declaration(repeat: Any) -> Compile:
    return Compile(
        family="harness-pgw993",
        targets=("transformer",),
        text_len=0,
        shapes=((64, 64),),
        dims=(
            Dim("H_lat", carried_by=(("x", 2),)),
            Dim("W_lat", carried_by=(("x", 3),)),
        ),
        classes=(
            GraphClass(dims={"H_lat": 8, "W_lat": 8}),
            GraphClass(dims={"H_lat": 16, "W_lat": 12}),
        ),
        inputs=(
            Input("x", shape=(("config", "in_channels"), 1, "H_lat", "W_lat"),
                  repeat=repeat),
            Input("t", shape=(2,), dtype="float32"),
        ),
        shape_strategy="dynamic-collapse",
        warm_changes_key=False,
    )


def _export(arity: int) -> Tuple[Any, Sequence[DynamicDim], Dict[str, int]]:
    decl = _list_declaration(arity)
    spec = ExportSpec(family=decl.family, target="transformer",
                      weight_lane="", precision="bf16")
    module = _ListModule()
    args, kwargs = declared_inputs(module, spec, decl)
    (plan,) = cell_plans(decl)
    arities = container_arities(decl, spec, module)
    program = torch.export.export(
        module, tuple(args), dict(kwargs), strict=True,
        dynamic_shapes=dynamic_shapes_spec(plan.dynamic, ["x", "t"], arities))
    return program, plan.dynamic, arities


@pytest.mark.parametrize("arity", [1, 2])
def test_a_real_export_flattens_and_the_gate_follows(arity: int) -> None:
    """End to end from a declaration, on both arms: torch really does emit
    `x_0`/`x_1`, and the mint's own gate passes the program it produced."""
    program, dims, arities = _export(arity)

    assert list(program.graph_signature.user_inputs) == \
        [f"x_{i}" for i in range(arity)] + ["t"]
    assert arities == {"x": arity}
    assert declared_range_gaps(program, dims, arities) == []
    # And the declared-name spelling is the refusal the pod measured.
    assert any("not a user input" in gap
               for gap in declared_range_gaps(program, dims))
