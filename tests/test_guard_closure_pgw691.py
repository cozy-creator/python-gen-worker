"""pgw#691: guard-closure classifier fixes, red-verified on real dynamo trees.

The offline audit (tracker 4b37ee8) drove the shipped classifier against 546
real torch-2.13.0 guard rows and proved the closed world was not closed:

* P0 — the RelationalGuard family (NO_TENSOR_ALIASING, OBJECT_ALIASING,
  STORAGE_OVERLAPPING) is attached to INPUT guard managers, but the
  classifier dispatched on source root first, so every graph with >=2
  tensor inputs leaked — i.e. every sdxl mint refused.
* P1 — the enumerated vocabulary mixed python GuardBuilder names into the
  C++ leaf-class universe the walk yields: 7 real classes unnameable,
  2 phantoms, SYMBOLIC_SHAPE_GUARD (torch 2.13's shape-env facts) always a
  LEAK so declared dynamic dims were unmintable.
* P1 — EQUALS_MATCH against torch singletons (``L['dt'] == torch.float32``)
  failed ``ast.literal_eval`` and false-positive leaked.

All probes here are CPU ``backend="eager"`` (dynamo's guard machinery is
production-identical; only inductor codegen is skipped) — no GPU, no
inference.
"""

from __future__ import annotations

import enum
import threading
from typing import Any, Callable, Dict, Iterator, Tuple

import pytest

torch = pytest.importorskip("torch")

from gen_worker import compile_cache as cc
from gen_worker import guard_closure as gc
from gen_worker.registry import CompileCell


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> Iterator[None]:
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _cfg(**overrides: Any) -> CompileCell:
    base: Dict[str, Any] = dict(
        shapes=((64, 64),), targets=("unet",), family="sdxl",
        regional=False, text_len=77, dynamic=(), lora_bucket=0,
        guidance_scales=(5.0,), text_lens=(),
    )
    base.update(overrides)
    return CompileCell(**base)


def _arm_marker(pipe: Any, module: Any, fn: Any) -> None:
    setattr(pipe, cc._MARKER_ATTR, {
        "targets": ["unet"],
        "shapes": [(64, 64)],
        "cache": True,
        "originals": [(module, "forward", fn)],
        "regional_mods": [],
        "failure_signal": {
            "callback": None, "lock": threading.Lock(),
            "successful_calls": 0, "cache_hits": 0, "cache_misses": 0,
            "router": None,
        },
    })


def _sdxl_shaped_pipe(forward: Callable[..., Any]) -> Tuple[Any, Any]:
    """An armed pipe whose target has the sdxl unet ingress ARGUMENT
    TOPOLOGY: several tensor inputs + a tensor dict + None/bool/str kwargs +
    a float guidance scalar (the audit's probe shape)."""

    class _Unet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)

    _Unet.forward = forward  # type: ignore[method-assign]

    class _Pipe:
        pass

    pipe = _Pipe()
    mod = _Unet()
    pipe.unet = mod  # type: ignore[attr-defined]
    _arm_marker(pipe, mod, mod.forward)
    return pipe, mod.forward


def _sdxl_forward(
    self: Any, sample: Any, timestep: Any, encoder_hidden_states: Any,
    added_cond_kwargs: Any, cross_attention_kwargs: Any = None,
    guidance_scale: float = 5.0, return_dict: bool = False,
    mode: str = "default",
) -> Any:
    out = self.lin(sample) + timestep + encoder_hidden_states
    out = out + added_cond_kwargs["time_ids"]
    if guidance_scale != 1.0:
        out = out * guidance_scale
    return out


def _sdxl_inputs() -> Dict[str, Any]:
    return dict(
        sample=torch.randn(2, 4, 8),
        timestep=torch.randn(2, 1, 8),
        encoder_hidden_states=torch.randn(2, 4, 8),
        added_cond_kwargs={"time_ids": torch.randn(2, 1, 8)},
        cross_attention_kwargs=None,
        return_dict=False,
        mode="default",
    )


# ---------------------------------------------------------------------------
# P0: the RelationalGuard family is input-attached, judged by TYPE
# ---------------------------------------------------------------------------


def test_multi_tensor_input_aliasing_guards_are_structural() -> None:
    """>=2 tensor inputs => dynamo emits NO_TENSOR_ALIASING on the INPUT
    managers (one row per participating tensor). Pre-fix these hit the
    terminal input-root LEAK; they are call-topology facts — STRUCTURAL."""

    def two_tensor(a: Any, b: Any) -> Any:
        return a + b

    compiled = torch.compile(two_tensor, backend="eager", dynamic=None)
    compiled(torch.randn(2, 8), torch.randn(2, 8))

    graphs = gc.extract_target_guards(two_tensor, "t", _cfg())
    assert len(graphs) == 1
    rows = graphs[0].guards
    alias = [r for r in rows if r.guard_type == "NO_TENSOR_ALIASING"]
    assert alias, "the >=2-tensor trigger must emit the relational family"
    assert {r.verdict for r in alias} == {gc.STRUCTURAL}
    assert all(r.source.startswith("L[") for r in alias)  # input-attached
    assert gc.LEAK not in {r.verdict for r in rows}


def test_relational_family_judged_by_type_regardless_of_root() -> None:
    pins = gc.contract_pins(_cfg())
    for guard_type in ("NO_TENSOR_ALIASING", "OBJECT_ALIASING",
                       "STORAGE_OVERLAPPING", "DUPLICATE_INPUT"):
        for source in ("", "L['sample']", "L['added']['time_ids']"):
            verdict, _axis = gc.classify(guard_type, source, "x", pins)
            assert verdict == gc.STRUCTURAL, (guard_type, source)


# ---------------------------------------------------------------------------
# Acceptance: the sdxl-shaped mint gate passes closed, refuses named
# ---------------------------------------------------------------------------


def test_sdxl_shaped_mint_gate_passes_closed() -> None:
    """The definitive acceptance: the classifier is leak-free on a real
    multi-tensor-input compiled module with the sdxl-shaped contract.
    Pre-fix: refused 100% (NO_TENSOR_ALIASING).

    pgw#1181 reads it off `audit_armed` rather than `closure_manifest`, the
    mint-side wrapper deleted with the `torch-inductor-cache` format. Same
    classifier, same verdicts, one fewer serialization step."""
    pipe, fn = _sdxl_shaped_pipe(_sdxl_forward)
    compiled = torch.compile(fn, backend="eager", dynamic=None)
    compiled(guidance_scale=5.0, **_sdxl_inputs())

    report = gc.audit_armed(pipe, _cfg())
    assert report.leaks == ()
    assert report.graphs
    types = {r.guard_type for g in report.graphs for r in g.guards}
    assert "NO_TENSOR_ALIASING" in types  # the P0 family really was present


def test_sdxl_shaped_mint_records_the_undeclared_variable() -> None:
    """A genuinely undeclared scalar is still detected and named — the
    pgw#691 fix widens nothing, and pgw#756 only removed the veto, so the
    finding lands in the manifest instead of killing the mint."""
    pipe, fn = _sdxl_shaped_pipe(_sdxl_forward)
    compiled = torch.compile(fn, backend="eager", dynamic=None)
    compiled(guidance_scale=7.5, **_sdxl_inputs())  # 7.5 is NOT declared

    leaks = "\n".join(gc.audit_armed(pipe, _cfg()).leaks)
    assert "L['guidance_scale']" in leaks
    assert "7.5" in leaks


# ---------------------------------------------------------------------------
# P1: SYMBOLIC_SHAPE_GUARD is torch 2.13's shape-env fact
# ---------------------------------------------------------------------------


def test_symbolic_shape_guard_rides_declared_dynamic_dims() -> None:
    """Real SYMBOLIC_SHAPE_GUARD rows (enable_cpp_symbolic_shape_guards)
    arrive with synthetic tuple sources like ``(0, L['x'].size()[0])`` —
    pre-fix both the ambient and the "other"-root paths leaked them, making
    every declared-dynamic contract unmintable."""
    flag = "enable_cpp_symbolic_shape_guards"
    old = getattr(torch._dynamo.config, flag)
    setattr(torch._dynamo.config, flag, True)
    try:
        def d(x: Any) -> Any:
            if x.shape[0] > 2:
                return x * 2
            return x

        xd = torch.randn(5, 3)
        torch._dynamo.mark_dynamic(xd, 0)
        torch.compile(d, backend="eager")(xd)

        dyn_cfg = _cfg(dynamic=(
            type("D", (), {"dim": "batch", "min": 2, "max": 64})(),))
        rows = gc.extract_target_guards(d, "t", dyn_cfg)[0].guards
        ssg = [r for r in rows if r.guard_type == "SYMBOLIC_SHAPE_GUARD"]
        assert ssg, "cpp symbolic shape guards must be emitted on 2.13"
        assert {r.verdict for r in ssg} == {gc.CONTRACT_SHAPE}

        # The SAME graph without declared dynamism is a named LEAK.
        undeclared = gc.extract_target_guards(d, "t", _cfg())[0].guards
        ssg_leaks = [
            r for r in undeclared if r.guard_type == "SYMBOLIC_SHAPE_GUARD"]
        assert {r.verdict for r in ssg_leaks} == {gc.LEAK}
        assert all("dynamic" in r.axis for r in ssg_leaks)
    finally:
        setattr(torch._dynamo.config, flag, old)


# ---------------------------------------------------------------------------
# P1: EQUALS_MATCH against torch singletons
# ---------------------------------------------------------------------------


def test_equals_match_torch_objects_are_covered() -> None:
    """``L['dt'] == torch.float32`` is not a python literal; pre-fix it
    leaked "unparseable". Dtype at the boundary is pinned by weight_lane +
    the ingress dtype memo; memory_format rides the same canonical pin."""

    def g(t: Any, dt: Any, mf: Any) -> Any:
        if dt == torch.float16:
            return t.half()
        if mf == torch.channels_last:
            return t
        return t + 1

    compiled = torch.compile(g, backend="eager", dynamic=None)
    compiled(torch.randn(2, 3), torch.float32, torch.contiguous_format)

    rows = gc.extract_target_guards(g, "t", _cfg())[0].guards
    eq = [r for r in rows
          if r.guard_type == "EQUALS_MATCH" and r.source.startswith("L[")]
    assert {r.source for r in eq} == {"L['dt']", "L['mf']"}
    assert {r.verdict for r in eq} == {gc.CONTRACT_SHAPE}
    assert all(r.axis == "weight_lane + ingress dtype memo" for r in eq)
    assert gc.LEAK not in {r.verdict for r in rows}


def test_equals_match_device_literal_and_garbage() -> None:
    pins = gc.contract_pins(_cfg())
    verdict, axis = gc.classify(
        "EQUALS_MATCH", "L['dev']",
        "L['dev'] == device(type='cuda', index=0)", pins)
    assert verdict == gc.CONTRACT_SHAPE
    assert "device" in axis
    # Non-torch unparseables still leak — the fix widens only torch objects.
    verdict, axis = gc.classify(
        "EQUALS_MATCH", "L['obj']", "L['obj'] == SomeClass(1)", pins)
    assert verdict == gc.LEAK
    assert "unparseable" in axis
    # torch attrs that are NOT dtype/layout/memory_format stay leaks.
    verdict, _axis = gc.classify(
        "EQUALS_MATCH", "L['fn']", "L['fn'] == torch.relu", pins)
    assert verdict == gc.LEAK


# ---------------------------------------------------------------------------
# P1: input-rooted object identity of call-path constants
# ---------------------------------------------------------------------------


class _Mode(enum.Enum):
    A = 1
    B = 2


def test_id_match_call_path_constants_are_code_constants() -> None:
    """Enum members / callables bound by the fixed endpoint call path emit
    input-rooted ID_MATCH; pre-fix any object-valued argument was
    unmintable. Identity of a call-path constant cannot vary per request;
    a cross-process identity miss recompiles via the guard-miss path —
    never wrongness."""

    def h(t: Any, mode: Any, hook: Any) -> Any:
        if mode is _Mode.A:
            return hook(t)
        return t

    compiled = torch.compile(h, backend="eager", dynamic=None)
    compiled(torch.randn(2, 3), _Mode.A, torch.relu)

    rows = gc.extract_target_guards(h, "t", _cfg())[0].guards
    idm = [r for r in rows
           if r.guard_type == "ID_MATCH" and r.source.startswith("L[")]
    assert {r.source for r in idm} >= {"L['mode']", "L['hook']"}
    assert {r.verdict for r in idm} == {gc.CODE_CONSTANT}
    assert gc.LEAK not in {r.verdict for r in rows}


# ---------------------------------------------------------------------------
# Vocabulary hygiene: C++ names in, phantoms out
# ---------------------------------------------------------------------------


def test_cpp_leaf_names_are_classified() -> None:
    pins = gc.contract_pins(_cfg())
    assert gc.classify("DUAL_LEVEL_MATCH", "", "x", pins)[0] == gc.RUNTIME_STATE
    assert gc.classify("FAKE_SCRIPT_TYPE_MATCH", "L['x']", "x", pins)[0] \
        == gc.STRUCTURAL
    assert gc.classify("DICT_VERSION", "L['cfg']", "x", pins)[0] \
        == gc.CODE_CONSTANT


def test_phantom_names_are_leaks() -> None:
    """AUTOCAST_STATE and KEYS_MATCH exist in NEITHER torch 2.13 universe
    (C++ leaf classes or GuardBuilder names) — if one ever arrives it is by
    definition unknown, and the closed world leaks it."""
    pins = gc.contract_pins(_cfg())
    assert gc.classify("AUTOCAST_STATE", "", "x", pins)[0] == gc.LEAK
    assert gc.classify("KEYS_MATCH", "L['d']", "x", pins)[0] == gc.LEAK
