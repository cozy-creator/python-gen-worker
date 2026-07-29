"""Derived guard sources must resolve their EMBEDDED root (pgw#733).

The fleet-wide P0: `_source_root` prefix-matched `^L\\['name'\\]`, but torch 2.13
emits class/code-structure guards whose sources CONTAIN a root without starting
with it — `dict(type(L['self']).__mro__[1].__dict__)` for a base-class
`@property` (diffusers `ConfigMixin.config`, read in every model's forward) and
`type(L['self']._modules['x']).__dict__['forward'].__defaults__` for a submodule
whose forward has a default argument. Those fell through to the "other" root,
which LEAKS *before* the type dispatch that already covers them — so every
self-mint on every family was refused at pack and no cell was ever published.

The 26 existing tapes stayed green because their fixtures are plain modules with
plain attributes and no default args. The last test here is the net that was
missing: a REAL module hierarchy with both triggers, compiled for real.
"""

from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch")

from gen_worker import compile_cache as cc  # noqa: E402
from gen_worker import guard_closure as gc  # noqa: E402


def _pins(**over):
    cfg = type("_Cfg", (), {
        "shapes": ((64, 64),), "targets": ("unet",), "family": "sdxl",
        "regional": False, "text_len": 77, "dynamic": (), "lora_bucket": 0,
        "guidance_scales": (5.0,), "text_lens": (),
    })()
    for k, v in over.items():
        setattr(cfg, k, v)
    return gc.contract_pins(cfg)


# --------------------------------------------------------------------------
# The two universal triggers, at the classifier.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source", [
    "dict(type(L['self']).__mro__[1].__dict__)",
    "dict(type(L['self']).__mro__[2].__dict__)['config'].fget.__code__",
    "type(L['self']._modules['time_embedding']).__dict__['forward'].__defaults__",
    "L['self']._modules['conv_in'].__class__",
])
def test_derived_self_sources_resolve_to_self(source):
    assert gc._source_root(source) == "self"


@pytest.mark.parametrize("guard_type", [
    "DICT_CONTAINS", "LENGTH_CHECK", "TYPE_MATCH", "NONE_MATCH", "ID_MATCH",
])
def test_derived_self_guards_are_covered_not_leaked(guard_type):
    """Every one of these types is already in the structural/identity
    dispatch — the root resolution was the only thing standing between them
    and a covering verdict."""
    verdict, _axis = gc.classify(
        guard_type,
        "dict(type(L['self']).__mro__[1].__dict__)",
        "not ___dict_contains('config', dict(type(L['self']).__mro__[1].__dict__))",
        _pins(),
    )
    assert verdict != gc.LEAK


def test_unrecognizable_source_is_still_a_leak():
    """The closed world is unchanged: resolving embedded roots must not become
    a licence to wave through sources with no root at all."""
    assert gc._source_root("some_synthetic_thing[0]") == "other"
    verdict, axis = gc.classify(
        "TYPE_MATCH", "some_synthetic_thing[0]", "whatever", _pins())
    assert verdict == gc.LEAK and "unrecognized guard source" in axis


# --------------------------------------------------------------------------
# Freevars: same dispatch, honest diagnosis.
# --------------------------------------------------------------------------

def test_freevar_code_facts_are_covered():
    """A wrapper's captured code object IS code — its identity and structure
    are pinned by code_closure + toolchain, exactly like a global."""
    pins = gc.contract_pins(_pins_cfg := type("_C", (), {
        "shapes": ((64, 64),), "text_lens": (), "text_len": 77,
        "lora_bucket": 0, "dynamic": (), "guidance_scales": (5.0,),
    })(), ("forward_fn", "kwargs_name"))
    assert gc._source_root("L['forward_fn'].__defaults__", pins.freevars) == "freevar"
    verdict, _ = gc.classify(
        "LENGTH_CHECK", "L['forward_fn'].__defaults__",
        "len(L['forward_fn'].__defaults__) == 10", pins)
    assert verdict != gc.LEAK
    verdict, _ = gc.classify(
        "EQUALS_MATCH", "L['kwargs_name']",
        "L['kwargs_name'] == 'cross_attention_kwargs'", pins)
    assert verdict != gc.LEAK


def test_leaking_freevar_is_named_as_a_freevar_not_a_call_input():
    """A captured runtime scalar still leaks — the fix is a declaration, not a
    different request, so the reason must say which it is."""
    pins = gc.contract_pins(type("_C", (), {
        "shapes": ((64, 64),), "text_lens": (), "text_len": 77,
        "lora_bucket": 0, "dynamic": (), "guidance_scales": (5.0,),
    })(), ("outer_scale",))
    verdict, axis = gc.classify(
        "EQUALS_MATCH", "L['outer_scale']",
        "L['outer_scale'] == 0.30000001", pins)
    assert verdict == gc.LEAK
    assert "closure freevar" in axis and "outer_scale" in axis
    assert "0.30000001" in axis


# --------------------------------------------------------------------------
# The net the toy tapes did not provide: a real hierarchy, really compiled.
# --------------------------------------------------------------------------

class _Base(torch.nn.Module):
    @property
    def config(self):  # the ConfigMixin.config shape, resolved via the MRO
        return {"kind": "test"}


class _Sub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x, condition=None):  # default arg: the second trigger
        return self.lin(x) if condition is None else self.lin(x) + condition


class _Unet(_Base):
    def __init__(self) -> None:
        super().__init__()
        self.time_embedding = _Sub()
        self.out = torch.nn.Linear(8, 8)

    def forward(self, x):
        assert self.config["kind"] == "test"   # base-class property read
        return self.out(self.time_embedding(x))


def _cfg():
    from gen_worker.registry import CompileCell

    return CompileCell(
        shapes=((64, 64),), targets=("unet",), family="sdxl", regional=False,
        text_len=77, dynamic=(), lora_bucket=0, guidance_scales=(5.0,),
        text_lens=())


def test_a_real_hierarchy_with_both_triggers_closes(monkeypatch):
    """Both triggers in one module, compiled for real (CPU, backend='eager').
    Pre-fix this produces LEAK rows — which, while the classifier still held
    a veto, refused every production self-mint (the bug that motivated the
    pgw#756 demotion). The rows must classify clean regardless."""
    unet = _Unet().eval()

    class _Pipe:
        pass

    pipe = _Pipe()
    pipe.unet = unet
    setattr(pipe, cc._MARKER_ATTR, {
        "targets": ["unet"], "shapes": [(64, 64)], "cache": True,
        "originals": [(unet, "forward", unet.forward)], "regional_mods": [],
        "failure_signal": {
            "callback": None, "lock": threading.Lock(),
            "successful_calls": 0, "cache_hits": 0, "cache_misses": 0,
            "router": None,
        },
    })

    torch._dynamo.reset()
    try:
        compiled = torch.compile(unet.forward, backend="eager", dynamic=None)
        with torch.no_grad():
            compiled(torch.randn(2, 8))
        report = gc.audit_armed(pipe, _cfg())
        assert not report.leaks, (
            "an in-contract call on a real module hierarchy leaked: "
            + "; ".join(str(leak) for leak in report.leaks))
        gc.closure_manifest(pipe, _cfg(), label="sdxl")
        counts = report.verdict_counts()
        assert counts.get(gc.MODULE_STRUCTURE, 0) > 0, (
            "the derived self-rooted guards must land as module structure, "
            f"got {counts}")
    finally:
        torch._dynamo.reset()
