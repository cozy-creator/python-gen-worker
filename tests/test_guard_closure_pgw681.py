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
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

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
        shapes=((64, 64),), targets=("transformer",), family="toyfam",
        regional=False, text_len=None, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=(),
    )
    base.update(overrides)
    return CompileCell(**base)


def _entries(fn: Any) -> int:
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list

    code = getattr(getattr(fn, "__func__", fn), "__code__", None)
    return len(_debug_get_cache_entry_list(code))


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

    manifest = gc.closure_manifest(pipe, _cfg(), label="toyfam")
    leaks = "\n".join(manifest["leaks"])
    assert "L['scale']" in leaks
    assert "3.25" in leaks
    assert manifest[gc.GATE_KEY] == gc.GATE_ADVISORY

    # The other direction still works: the SAME graph classifies clean once
    # the contract declares the pin — classification keys on the contract,
    # not the code shape.
    manifest = gc.closure_manifest(
        pipe, _cfg(guidance_scales=(3.25,)), label="toyfam")
    assert manifest["leaks"] == []


def test_gate_refuses_when_nothing_is_extractable() -> None:
    """A mint that compiled NOTHING is a proven defect (the cell would be
    empty) — this refusal survives the pgw#756 demotion."""

    class _Mod:
        def forward(self, x: Any) -> Any:  # pragma: no cover - never called
            return x

    class _Pipe:
        pass

    pipe = _Pipe()
    mod = _Mod()
    pipe.transformer = mod  # type: ignore[attr-defined]
    _arm_marker(pipe, mod, mod.forward)
    with pytest.raises(gc.GuardClosureError, match="nothing was compiled"):
        gc.closure_manifest(pipe, _cfg())


def test_the_packed_cell_records_the_manifest_leaking_or_clean(
    tmp_path: Path,
) -> None:
    """The audit rides the packed cell: a leaking capture PACKS anyway
    (pgw#756) carrying its leak, and a clean capture packs with an empty leak
    list — both with the manifest riding metadata.json.

    pgw#1010: assembled the way the surviving producer (`mint_artifact`) does,
    because `finish_fleet_mint` sealed a dynamo cell and is deleted with it."""

    def forward(self: Any, x: Any, scale: float) -> Any:
        return self.lin(x) * scale

    pipe, _mod, fn = _compiled_toy(forward)
    compiled = torch.compile(fn, backend="eager", dynamic=None)
    compiled(torch.randn(2, 4, 8), 3.25)

    capture = tmp_path / "capture"
    fx_entry = capture / "inductor" / "fxgraph" / "aa" / "bb"
    fx_entry.mkdir(parents=True)
    (fx_entry / "entry").write_bytes(b"fx")
    target = tmp_path / "cell.tar.gz"

    def _packed(cfg: Any, out: Path) -> dict:
        meta = cc.artifact_metadata(
            family="toyfam", source_ref="self-mint",
            shapes=cfg.shapes, targets=cfg.targets,
            guidance_scales=getattr(cfg, "guidance_scales", ()))
        meta[gc.MANIFEST_KEY] = gc.closure_manifest(pipe, cfg, label="toyfam")
        cc.pack(capture, out, meta)
        return meta

    leaky = tmp_path / "leaky.tar.gz"
    meta = _packed(_cfg(), leaky)
    assert leaky.exists(), "an undeclared scalar must no longer refuse the mint"
    assert any("L['scale']" in row for row in meta[gc.MANIFEST_KEY]["leaks"])
    assert gc.load_manifest(leaky)["leaks"] == meta[gc.MANIFEST_KEY]["leaks"]

    meta = _packed(_cfg(guidance_scales=(3.25,)), target)
    assert meta[gc.MANIFEST_KEY]["leaks"] == []
    assert target.exists()
    loaded = gc.load_manifest(target)
    assert loaded == meta[gc.MANIFEST_KEY]


# ---------------------------------------------------------------------------
# Acceptance: stride-perturbed input canonicalizes and HITS
# ---------------------------------------------------------------------------


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
    assert _entries(f) == 1
    for t in _perturbed_inputs():
        compiled(t)
    assert _entries(f) == 3


def test_stride_perturbed_input_canonicalizes_and_hits() -> None:
    def f(x: Any) -> Any:
        return x * 2 + 1

    compiled = gc.canonical_ingress(
        torch.compile(f, backend="eager", dynamic=None), "transformer")
    warm = torch.randn(4, 1, 8)
    compiled(warm)
    assert _entries(f) == 1
    for t in _perturbed_inputs():
        out = compiled(t)
        assert torch.equal(out, t.contiguous() * 2 + 1)
    assert _entries(f) == 1  # zero recompiles: both perturbations HIT


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
        "graphs": [{"target": "transformer", "code": "forward", "entry": 0,
                    "guards": guards}],
        "verdicts": {}, "leaks": [],
    }
    manifest.update(mutate)
    return manifest


def test_consolidate_identical_manifests_is_closed() -> None:
    audit = gc.consolidate({"pod-a": _toy_manifest(), "pod-b": _toy_manifest()})
    assert audit.ok
    assert audit.guard_total == 4


def test_consolidate_tolerates_per_host_ambient_content() -> None:
    """GLOBAL_STATE content is per-host (num_threads); its PRESENCE is
    compared, its content stays in the dump."""
    other = _toy_manifest()
    other["graphs"][0]["guards"][1]["expr"] = (
        '___check_global_state() against {"num_threads":8}')
    audit = gc.consolidate({"pod-a": _toy_manifest(), "pod-b": other})
    assert audit.ok


def test_consolidate_names_divergence_and_leaks() -> None:
    missing = _toy_manifest()
    missing["graphs"][0]["guards"] = missing["graphs"][0]["guards"][:1]
    audit = gc.consolidate({"pod-a": _toy_manifest(), "pod-b": missing})
    assert not audit.ok
    assert any("GLOBAL_STATE" in d and "pod-b" in d for d in audit.divergence)

    leaky = _toy_manifest(leaks=["target=transformer EQUALS_MATCH L['s']: ..."])
    audit = gc.consolidate({"pod-a": leaky})
    assert audit.leaks and "pod-a" in audit.leaks[0]


def test_cli_zero_miss_check_exit_codes(tmp_path: Path) -> None:
    good_a = tmp_path / "a.json"
    good_b = tmp_path / "b.json"
    good_a.write_text(json.dumps(_toy_manifest()))
    good_b.write_text(json.dumps(_toy_manifest()))
    assert gc.main([str(good_a), str(good_b)]) == 0

    leaky = tmp_path / "leak.json"
    leaky.write_text(json.dumps(_toy_manifest(leaks=["leak row"])))
    assert gc.main([str(good_a), str(leaky)]) == 2

    divergent = _toy_manifest()
    divergent["graphs"][0]["guards"] = divergent["graphs"][0]["guards"][:1]
    div = tmp_path / "div.json"
    div.write_text(json.dumps(divergent))
    assert gc.main([str(good_a), str(div)]) == 3

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"graphs": []}))
    assert gc.main([str(good_a), str(empty)]) == 3  # unproven cell never passes
