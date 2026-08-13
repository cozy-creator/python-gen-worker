"""pgw#681: guard-closure gate + boundary canonicalization.

Real-dynamo tapes (CPU, ``backend="eager"`` — dynamo's guard machinery is
identical to production's; only inductor codegen is skipped): dynamo's guard
set IS the exhaustive dependency list of a compiled graph, so the gate can
machine-check closure over the declared contract, and the ingress can pin
the input boundary so serving inputs HIT the minted guards.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict, Iterator, List, Tuple

import pytest

torch = pytest.importorskip("torch")

from gen_worker import compile_cache as cc
from gen_worker import guard_closure as gc
from gen_worker.registry import CompileCompiledGraph


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> Iterator[None]:
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _cfg(**overrides: Any) -> CompileCompiledGraph:
    base: Dict[str, Any] = dict(
        shapes=((64, 64),), targets=("transformer",), family="toyfam",
        regional=False, text_len=None, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=(),
    )
    base.update(overrides)
    return CompileCompiledGraph(**base)


def _compiled_graphs(fn: Any) -> int:
    from torch._dynamo.eval_frame import _debug_get_cache_compiled_graph_list

    code = getattr(getattr(fn, "__func__", fn), "__code__", None)
    return len(_debug_get_cache_compiled_graph_list(code))


def _arm_marker(pipe: Any, module: Any, fn: Any) -> None:
    """A real apply()-shaped marker (apply() itself refuses on CPU hosts)."""
    setattr(pipe, cc._MARKER_ATTR, {
        "targets": ["transformer"],
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


def _compiled_toy(
    forward: Callable[..., Any],
) -> Tuple[Any, Any, Callable[..., Any]]:
    """(pipe, module, bound_forward) with a marker armed and the forward
    really compiled through dynamo (guards live on the code object)."""

    class _Mod(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)

    _Mod.forward = forward  # type: ignore[method-assign]

    class _Pipe:
        pass

    pipe = _Pipe()
    mod = _Mod()
    pipe.transformer = mod  # type: ignore[attr-defined]
    _arm_marker(pipe, mod, mod.forward)
    return pipe, mod, mod.forward


# ---------------------------------------------------------------------------
# Extraction + classification
# ---------------------------------------------------------------------------


def test_extraction_walks_real_dynamo_guard_tree() -> None:
    def forward(self: Any, x: Any, scale: float) -> Any:
        return self.lin(x) * scale

    pipe, _mod, fn = _compiled_toy(forward)
    compiled = torch.compile(fn, backend="eager", dynamic=None)
    compiled(torch.randn(2, 4, 8), 7.5)

    report = gc.extract(pipe, _cfg(guidance_scales=(7.5,)))
    assert len(report.graphs) == 1
    rows = report.graphs[0].guards
    by_type = {(r.guard_type, r.source): r for r in rows}
    tensor = by_type[("TENSOR_MATCH", "L['x']")]
    assert tensor.verdict == gc.CANONICALIZED
    scalar = by_type[("EQUALS_MATCH", "L['scale']")]
    assert scalar.verdict == gc.CONTRACT_SCALAR
    verdicts = {r.verdict for r in rows}
    assert gc.MODULE_STRUCTURE in verdicts    # L['self'] weights/structure
    assert gc.RUNTIME_STATE in verdicts       # ambient global state
    assert gc.LEAK not in verdicts
    assert report.closed


def test_manifest_is_deterministic_and_scrubs_aslr_ids() -> None:
    def forward(self: Any, x: Any) -> Any:
        return self.lin(x) + 1

    pipe, _mod, fn = _compiled_toy(forward)
    compiled = torch.compile(fn, backend="eager", dynamic=None)
    compiled(torch.randn(2, 4, 8))

    cfg = _cfg()
    one = gc.extract(pipe, cfg).manifest()
    two = gc.extract(pipe, cfg).manifest()
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
    dump = json.dumps(one)
    assert "<id>" in dump
    # No raw process addresses survive in check_obj_id/check_type_id.
    assert not any(
        "_id(" in r["expr"] and r["expr"].split("_id(")[1].rstrip(")").split(", ")[-1].isdigit()
        for g in one["graphs"] for r in g["guards"]
    )


def test_classifier_closed_world() -> None:
    pins = gc.contract_pins(_cfg(guidance_scales=(7.5,), text_lens=(256,)))
    # Module + global roots: covered by construction.
    assert gc.classify("ID_MATCH", "L['self']._modules['lin']", "x", pins)[0] \
        == gc.MODULE_STRUCTURE
    assert gc.classify("ID_MATCH", "G['mod'].F.linear", "x", pins)[0] \
        == gc.CODE_IDENTITY
    # Inputs: closed world.
    assert gc.classify("EQUALS_MATCH", "L['n']", "L['n'] == 256", pins)[0] \
        == gc.CONTRACT_SCALAR
    assert gc.classify("EQUALS_MATCH", "L['n']", "L['n'] == 999", pins) \
        == (gc.LEAK, "int 999 is not a declared contract pin")
    # pgw#691: input-rooted ID_MATCH is the object identity of a call-path
    # constant (enum member / callable bound by the endpoint call path) —
    # covered, not a leak.
    assert gc.classify("ID_MATCH", "L['mask']", "___check_obj_id(...)", pins)[0] \
        == gc.CODE_CONSTANT
    assert gc.classify("WEIRD_NEW_GUARD", "L['x']", "whatever", pins)[0] == gc.LEAK
    # Ambient: known types covered, unknown leak.
    assert gc.classify("GLOBAL_STATE", "", "___check_global_state()", pins)[0] \
        == gc.RUNTIME_STATE
    assert gc.classify("MYSTERY", "", "x", pins)[0] == gc.LEAK
    # Shape-env relations only pass under declared dynamism.
    rel = "2 <= L['x'].size()[0] <= 64"
    assert gc.classify("LAMBDA_GUARD", "L['x']", rel, pins)[0] == gc.LEAK
    dyn_pins = gc.contract_pins(_cfg(dynamic=(
        type("D", (), {"dim": "batch", "min": 2, "max": 64})(),)))
    assert gc.classify("LAMBDA_GUARD", "L['x']", rel, dyn_pins)[0] \
        == gc.CONTRACT_SHAPE


def test_noncanonical_minted_stride_is_a_leak() -> None:
    """A TENSOR_MATCH whose stride is not the ingress-canonical layout can
    never HIT once serving canonicalizes — the gate refuses the mint."""
    verdict, axis = gc.classify(
        "TENSOR_MATCH", "L['x']",
        "check_tensor(L['x'], Tensor, ..., size=[4, 1, 8], stride=[8, 999, 1])",
        gc.contract_pins(_cfg()))
    assert verdict == gc.LEAK
    assert "non-canonical stride" in axis


# ---------------------------------------------------------------------------
# Acceptance: an out-of-envelope scalar leak is RECORDED, not refused
# (pgw#756 — the classifier lost its veto; see test_guard_gate_advisory_pgw756)
# ---------------------------------------------------------------------------


def test_out_of_envelope_scalar_is_recorded_naming_the_variable() -> None:
    def forward(self: Any, x: Any, scale: float) -> Any:
        return self.lin(x) * scale

    pipe, _mod, fn = _compiled_toy(forward)
    compiled = torch.compile(fn, backend="eager", dynamic=None)
    compiled(torch.randn(2, 4, 8), 3.25)  # 3.25 is NOT declared

    # pgw#1181: read the classification off `audit_armed`, the live view of an
    # armed pipeline, instead of `closure_manifest` — the MINT-side wrapper
    # that embedded it in a `torch-inductor-cache` compiled graph and is deleted with
    # that format. The classifier under test is the same one; only the
    # serialize-into-a-compiled graph step is gone.
    leaks = "\n".join(gc.audit_armed(pipe, _cfg()).leaks)
    assert "L['scale']" in leaks
    assert "3.25" in leaks

    # The other direction still works: the SAME graph classifies clean once
    # the contract declares the pin — classification keys on the contract,
    # not the code shape.
    assert gc.audit_armed(pipe, _cfg(guidance_scales=(3.25,))).leaks == ()


# pgw#1181 REMOVED `test_gate_refuses_when_nothing_is_extractable` and
# `test_the_packed_compiled_graph_records_the_manifest_leaking_or_clean`. Both are about
# `closure_manifest` — the mint-side wrapper that refused an empty capture and
# wrote the classification into a compiled graph's metadata. It is deleted with the
# `torch-inductor-cache` format, whose last writer died in pgw#1178: there is
# no compiled graph for a guard manifest to ride on and no mint for the empty-capture
# refusal to fail. The CLASSIFIER those rows reached through it is untouched
# and is driven directly above, via `audit_armed`.


def _perturbed_inputs() -> List[Any]:
    base = torch.randn(4, 1, 8)
    return [
        # ie#544 trap: size-1 dim keeps is_contiguous() true under a garbage
        # stride, so .contiguous() alone is a no-op.
        base.as_strided((4, 1, 8), (8, 999, 1)),
        # Genuinely non-contiguous layout of the same shape.
        torch.randn(8, 4, 1).permute(1, 2, 0),
    ]


def test_stride_perturbed_input_recompiles_without_the_ingress() -> None:
    """RED half: the raw compiled callable pays one recompile per stride
    perturbation (live-confirmed dynamo behavior the ingress exists for)."""

    def f(x: Any) -> Any:
        return x * 2 + 1

    compiled = torch.compile(f, backend="eager", dynamic=None)
    compiled(torch.randn(4, 1, 8))
    assert _compiled_graphs(f) == 1
    for t in _perturbed_inputs():
        compiled(t)
    assert _compiled_graphs(f) == 3


def test_stride_perturbed_input_canonicalizes_and_hits() -> None:
    def f(x: Any) -> Any:
        return x * 2 + 1

    compiled = gc.canonical_ingress(
        torch.compile(f, backend="eager", dynamic=None), "transformer")
    warm = torch.randn(4, 1, 8)
    compiled(warm)
    assert _compiled_graphs(f) == 1
    for t in _perturbed_inputs():
        out = compiled(t)
        assert torch.equal(out, t.contiguous() * 2 + 1)
    assert _compiled_graphs(f) == 1  # zero recompiles: both perturbations HIT


def test_ingress_pins_size1_dim_strides_not_just_contiguous() -> None:
    seen: List[Tuple[int, ...]] = []

    def f(x: Any) -> Any:
        seen.append(tuple(x.stride()))
        return x

    wrapped = gc.canonical_ingress(f, "t")
    weird = torch.randn(4, 1, 8).as_strided((4, 1, 8), (8, 999, 1))
    assert weird.is_contiguous() and weird.contiguous() is weird
    wrapped(weird)
    assert seen == [gc.canonical_strides((4, 1, 8))]


def test_ingress_canonicalizes_nested_containers_and_kwargs() -> None:
    seen: List[Tuple[int, ...]] = []

    def f(xs: Any, *, cond: Any = None) -> Any:
        seen.append(tuple(xs[0].stride()))
        seen.append(tuple(cond["c"].stride()))
        return xs[0]

    wrapped = gc.canonical_ingress(f, "t")
    nc = torch.randn(8, 4, 1).permute(1, 2, 0)
    wrapped([nc], cond={"c": nc.clone()})
    assert seen == [gc.canonical_strides((4, 1, 8))] * 2


def test_ingress_dtype_drift_raises_named() -> None:
    def f(x: Any) -> Any:
        return x

    wrapped = gc.canonical_ingress(f, "transformer")
    wrapped(torch.randn(2, 8))
    with pytest.raises(gc.GuardBoundaryError) as excinfo:
        wrapped(torch.randn(2, 8, dtype=torch.float64))
    message = str(excinfo.value)
    assert "args[0]" in message
    assert "float64" in message and "float32" in message
    assert "transformer" in message


# ---------------------------------------------------------------------------
# Fleet consolidation: the N-cold-pod zero-miss closure check
# ---------------------------------------------------------------------------


def _toy_manifest(**mutate: Any) -> Dict[str, Any]:
    guards = [
        {"type": "TENSOR_MATCH", "source": "L['x']",
         "expr": "check_tensor(size=[4, 1, 8], stride=[8, 8, 1])",
         "verdict": gc.CANONICALIZED, "axis": "ingress"},
        {"type": "GLOBAL_STATE", "source": "",
         "expr": '___check_global_state() against {"num_threads":24}',
         "verdict": gc.RUNTIME_STATE, "axis": "runtime"},
    ]
    manifest: Dict[str, Any] = {
        "v": 1,
        "graphs": [{"target": "transformer", "code": "forward", "compiled_graph": 0,
                    "guards": guards}],
        "verdicts": {}, "leaks": [],
    }
    manifest.update(mutate)
    return manifest


# pgw#1181 REMOVED the three `consolidate` rows and
# `test_cli_zero_miss_check_exit_codes`. `guard_closure.consolidate`,
# `load_manifest`, `FleetAudit` and the `python -m gen_worker.guard_closure`
# CLI compared the guard manifests of N compiled graphs — a cross-pod audit over a block
# that no compiled graph carries any more, since `closure_manifest` was its only writer.
# `load_manifest` could only ever raise "carries no guard manifest", so the CLI
# was a tool that could not succeed. Deleted with their subject (§4.34).
