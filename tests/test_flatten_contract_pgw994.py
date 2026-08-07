"""pgw#994 — the ingress contract records WHERE a leaf lives, and the serve
side replays that identity instead of guessing.

THE DEFECT. `aot_package.input_contract` numbers `position` over the EXPORTED
(flattened) graph inputs, while `aot_serve.bind_call_inputs` matched that
number against the caller's args, which are the call BEFORE flattening. A
container argument occupies ONE caller slot and produces N graph inputs, so
for the z-image shape `x.0` bound the whole list, `x.1` bound the next
argument, and `t` then refused `input_missing` — measured off-GPU:

    row x.0  position 0 | row x.1  position 1 | row t  position 2
    marshal_positional(contract, (x_list, t), {}) ->
      IngressContractError: declared input 't' (position 2) is absent

sdxl escaped only because diffusers passes its dict by KEYWORD (so the
positional branch never fired) and a nested search then found `text_embeds`
inside `added_cond_kwargs`. Two accidents, not a rule.

TWO MORE THINGS THAT WERE LUCK, both measured here rather than argued:

* `flat_input_names` sorted mapping keys "because that is what torch's pytree
  does". It does NOT — torch flattens dicts in INSERTION order. A dict whose
  insertion order is not alphabetical had every leaf recorded with another
  leaf's dtype and shape. sdxl's `text_embeds` < `time_ids` hid it.
* A plain input sitting AFTER a container has a flat position its argument
  does not have, so even a container-free row cannot always be resolved from
  `position` alone.

THE FIX. `aot_flatten` owns the walk and the naming, and every consumer reads
it: the mint flattens the example feed with it, the contract records each
row's `(param, param_position, path)`, the serve side resolves by replaying
that identity, and pgw#993's `exported_input_names` is a special case of the
same naming rule. The nested search is DELETED — nothing needs to hunt for a
value whose location the contract states.

No GPU: every export below is a tiny CPU trace.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import aot_flatten, aot_mint, aot_package, aot_serve
from gen_worker.aot_flatten import Leaf

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Doubles: the three real shapes in the fleet.
# ---------------------------------------------------------------------------


class _ListModule(torch.nn.Module):
    """z-image: a python LIST of per-sample tensors, then a plain tensor."""

    def forward(self, x: List[Any], t: Any) -> Any:
        return torch.stack([e.sum() for e in x]) + t.sum()


class _DictModule(torch.nn.Module):
    """sdxl: a tensor plus a dict of conditioning tensors."""

    def forward(self, sample: Any, added_cond_kwargs: Dict[str, Any]) -> Any:
        return (sample.sum() + added_cond_kwargs["zeta"].sum()
                + added_cond_kwargs["alpha"].sum())


class _MixedModule(torch.nn.Module):
    """qwen: a tensor, a nested python-int container, a bool, a tensor."""

    def forward(self, hidden_states: Any, img_shapes: List[Any],
                return_dict: bool, extra: Any) -> Any:
        total = 0
        for outer in img_shapes:
            for grp in outer:
                total += grp[0] * grp[1]
        return hidden_states * float(total) + extra.sum()


def _contract(
    module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any],
) -> Tuple[Any, aot_serve.ArtifactContract]:
    program = torch.export.export(module, args, kwargs, strict=True)
    leaves = aot_mint.flat_input_leaves(module, args, kwargs)
    rows, symbols = aot_package.input_contract(program, leaves)
    return program, aot_serve.contract_from_meta(
        {"inputs": rows, "symbols": symbols})


def _shapes(values: List[Any]) -> List[Tuple[int, ...]]:
    return [tuple(v.shape) for v in values]


# ---------------------------------------------------------------------------
# 1. THE DEFECT — a container family binds a real call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arity", [1, 2])
def test_a_container_call_binds_element_for_element(arity: int) -> None:
    """RED before pgw#994: N=1 bound the LIST where element 0 belonged, and
    N=2 refused `input_missing: t` outright."""
    module = _ListModule()
    x = [torch.randn(4, 1, 8, 8) for _ in range(arity)]
    t = torch.randn(2)
    _, contract = _contract(module, (x, t), {})

    feeds = aot_serve.marshal_positional(contract, (x, t), {})

    assert _shapes(feeds) == _shapes(list(x) + [t])
    # Identity, not luck: each element is ITS element object.
    assert all(a is b for a, b in zip(feeds, list(x) + [t]))


def test_the_input_after_a_container_keeps_its_own_value() -> None:
    """The half a path-only fix would miss: `t` is flat position 2 and
    ARGUMENT position 1, so a row with no path still needs its identity."""
    module = _ListModule()
    x = [torch.randn(4, 1, 8, 8), torch.randn(4, 1, 8, 8)]
    t = torch.randn(2)
    _, contract = _contract(module, (x, t), {})

    row = next(s for s in contract.inputs if s.name == "t")

    assert row.position == 2, "flat position, after two container leaves"
    assert row.call_position == 1, "argument position, which is what binds"
    assert aot_serve.bind_call_inputs(contract, (x, t), {})["t"] is t


def test_a_container_call_that_is_short_refuses_by_name() -> None:
    """The gate keeps its teeth: resolution names the argument and the path
    it walked, rather than an index into a call that has been reshaped."""
    module = _ListModule()
    x = [torch.randn(4, 1, 8, 8), torch.randn(4, 1, 8, 8)]
    _, contract = _contract(module, (x, torch.randn(2)), {})

    with pytest.raises(aot_serve.IngressContractError) as caught:
        aot_serve.marshal_positional(contract, ([x[0]], torch.randn(2)), {})

    assert caught.value.reason == "input_missing"
    assert "argument 'x'" in str(caught.value)
    assert "[1]" in str(caught.value)


# ---------------------------------------------------------------------------
# 2. THE DICT CASE — through the rule, with the search deleted
# ---------------------------------------------------------------------------


def test_dict_leaves_pair_with_their_own_tensors() -> None:
    """RED before pgw#994: the walk sorted keys, torch does not. With
    insertion order `zeta, alpha` the sorted walk recorded `alpha` with
    zeta's [2, 1280] and `zeta` with alpha's [2, 6]."""
    module = _DictModule()
    sample = torch.randn(2, 4)
    cond = {"zeta": torch.randn(2, 1280), "alpha": torch.randn(2, 6)}
    program, contract = _contract(module, (sample,),
                                  {"added_cond_kwargs": cond})

    assert list(program.graph_signature.user_inputs) == [
        "sample", "added_cond_kwargs_zeta", "added_cond_kwargs_alpha"]
    by_name = {s.name: s for s in contract.inputs}
    assert by_name["zeta"].shape == (2, 1280)
    assert by_name["alpha"].shape == (2, 6)


def test_the_dict_case_binds_because_the_contract_says_where() -> None:
    """The nested search is gone (pgw#994): a decoy kwarg carrying the same
    key must not be able to decide the bind, and the real one still resolves
    because its identity names its argument."""
    module = _DictModule()
    sample = torch.randn(2, 4)
    cond = {"zeta": torch.randn(2, 1280), "alpha": torch.randn(2, 6)}
    _, contract = _contract(module, (sample,), {"added_cond_kwargs": cond})

    decoy = {"zeta": torch.randn(9, 9), "alpha": torch.randn(9, 9)}
    bound = aot_serve.bind_call_inputs(
        contract, (sample,), {"decoy": decoy, "added_cond_kwargs": cond})

    assert bound["zeta"] is cond["zeta"]
    assert bound["alpha"] is cond["alpha"]
    # And with the real argument absent it REFUSES rather than finding the
    # decoy's key — which is exactly what the old search would have done.
    with pytest.raises(aot_serve.IngressContractError):
        aot_serve.bind_call_inputs(contract, (sample,), {"decoy": decoy})


def test_a_non_tensor_argument_still_lines_the_positions_up() -> None:
    """qwen's shape: `img_shapes` and `return_dict` produce constant leaves
    that occupy flat slots but no contract rows."""
    module = _MixedModule()
    args = (torch.randn(1, 4, 6), [[(1, 4, 6)]], False, torch.randn(3))
    _, contract = _contract(module, args, {})

    bound = aot_serve.bind_call_inputs(contract, args, {})

    assert bound["hidden_states"] is args[0]
    assert bound["extra"] is args[3]


# ---------------------------------------------------------------------------
# 3. ONE RULE — the naming, and who reads it
# ---------------------------------------------------------------------------


def test_exported_name_is_what_torch_actually_emits() -> None:
    """Measured against real exports, for all three container shapes at once
    — the rule is not allowed to be a guess about torch."""
    class _All(torch.nn.Module):
        def forward(self, sample: Any, cond: Dict[str, Any], x: List[Any],
                    nested: List[Any]) -> Any:
            return (sample.sum() + cond["text_embeds"].sum()
                    + cond["time_ids"].sum()
                    + torch.stack([e.sum() for e in x]).sum()
                    + nested[0][0].sum())

    args = (torch.randn(2, 4),
            {"text_embeds": torch.randn(2, 1280), "time_ids": torch.randn(2, 6)},
            [torch.randn(1, 8), torch.randn(1, 8)],
            [[torch.randn(2)]])
    program = torch.export.export(_All(), args, {}, strict=True)
    leaves = aot_mint.flat_input_leaves(_All(), args, {})

    assert [leaf.exported_name for leaf in leaves] == \
        [str(n) for n in program.graph_signature.user_inputs]
    assert aot_flatten.exported_name("x", (0,)) == "x_0"
    assert aot_flatten.exported_name("cond", ("time_ids",)) == "cond_time_ids"
    assert aot_flatten.exported_name("nested", (0, 0)) == "nested_0_0"


def test_the_pgw993_gate_rule_is_the_same_naming_rule() -> None:
    """pgw#993 fixed the declaration side by expanding declared names into
    exported ones. That expansion is now this module's naming rule applied to
    the paths a container has, not a second copy of it."""
    assert aot_mint.exported_input_names("x", {"x": 3}) == tuple(
        aot_flatten.exported_name("x", (i,)) for i in range(3))
    assert aot_mint.exported_input_names("x") == (
        aot_flatten.exported_name("x"),)


def test_the_serve_side_name_spelling_is_unchanged() -> None:
    """pgw#790's spelling is load-bearing — it is what published contracts are
    keyed by — so the identity is recorded NEXT to it, never instead of it."""
    assert Leaf("x", 0, (0,)).name == "x.0"
    assert Leaf("cond", 1, ("text_embeds",)).name == "text_embeds"
    assert Leaf("img_shapes", 1, (0, 0, 2)).name == "img_shapes.0.0.2"
    assert Leaf("cond", 1, ("k", 0)).name == "k.0"
    assert Leaf("sample", 0, ()).name == "sample"


# ---------------------------------------------------------------------------
# 4. NO LIVE CELL IS RE-KEYED
# ---------------------------------------------------------------------------


def _meta(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"inputs": rows, "symbols": {"s0": [8, 64]}}


def test_a_trivial_identity_does_not_move_the_range_digest() -> None:
    """`range_digest` feeds ck6. Spelling out an identity that says exactly
    what its absence says must not re-key the fleet's live checkpoints."""
    bare = [
        {"name": "sample", "position": 0, "dtype": "bfloat16",
         "shape": [2, 4, "s0"], "optional": False},
        {"name": "t", "position": 1, "dtype": "bfloat16", "shape": [2],
         "optional": False},
    ]
    spelled = [dict(row, param=row["name"], param_position=row["position"],
                    path=[]) for row in bare]

    assert aot_serve.range_digest(_meta(bare)) == \
        aot_serve.range_digest(_meta(spelled))


def test_a_container_identity_DOES_move_the_range_digest() -> None:
    """The other direction, because it is the reason the field is keyed at
    all: two classes taking the same tensors in different argument structures
    are different graphs and must not share a cell key."""
    flat = [{"name": "x.0", "position": 0, "dtype": "bfloat16",
             "shape": [4, "s0"], "optional": False}]
    carried = [dict(flat[0], param="x", param_position=0, path=[0])]

    assert aot_serve.range_digest(_meta(flat)) != \
        aot_serve.range_digest(_meta(carried))


def test_metadata_without_the_fields_still_loads() -> None:
    """Every artifact published before pgw#994 has rows with no identity, and
    they mean the trivial one."""
    contract = aot_serve.contract_from_meta(json.loads(json.dumps(_meta([
        {"name": "sample", "position": 0, "dtype": "bfloat16",
         "shape": [2, 4], "optional": False}]))))
    (row,) = contract.inputs

    assert row.call_param == "sample"
    assert row.call_position == 0
    assert row.path == ()
    assert row.trivial_identity
