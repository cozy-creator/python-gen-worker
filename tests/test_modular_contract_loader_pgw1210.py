"""pgw#1210: a component declaring a layout contract loads through that
contract's registered loader — on BOTH entry points, or it is a typed refusal.

THE FAULT (filed by the ie#681/ie#699 lane; minimax-h3 0.4.34, two releases,
three pods, same signature):

    ModularHydrationError: … Failed to create component transformer:
      modular_pipeline_utils.py load -> modeling_utils.py from_pretrained
      -> config["quantization_config"] = DiffusersAutoQuantizer.merge_…

The artifact is a PROVEN `cozy.fp8-rowwise@1` tree and the SDK ships the
declaring loader for exactly that layout (`w8a8.load_w8a8_denoiser`,
`@implements_contract`). The NON-modular path dispatches to it. A diffusers
MODULAR pipeline hydrates through `ComponentSpec.load()` -> plain
`from_pretrained`, which never consults it — so the same bytes that serve fine
on one path kill the pod on the other, forever (`deterministic_fault_loop`).

THE INVARIANT, and why the fix is a shared dispatch rather than a second copy:
one artifact shape has two entry points, and two dispatches that agree today
are two dispatches that will disagree later. `contract_loaded_component` is now
the only place that decides, and both callers ask it.

wan-2.2's fp8 lane is the sibling consumer — it binds a produced fp8
tree on the non-modular path — which is why "keep both entry points on one
dispatch" was the filing's own requirement.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"
if str(MICRO_SRC) not in sys.path:
    sys.path.insert(0, str(MICRO_SRC))

from gen_worker.models import loading, w8a8  # noqa: E402


def _w8a8_component(root: Path, component: str = "transformer") -> Path:
    """A produced fp8-rowwise component tree: fp8 weights + their scales.

    Written with real safetensors because detection reads real headers — a
    stub would prove the test's own fiction rather than the contract.
    """
    from micro_diffusion.model import MicroConfig, MicroDenoiser

    comp = root / component
    comp.mkdir(parents=True)
    denoiser = MicroDenoiser(MicroConfig())
    tensors = {}
    for name, param in denoiser.named_parameters():
        if name.endswith(".weight") and param.ndim == 2:
            tensors[name] = param.detach().to(torch.float8_e4m3fn)
            tensors[name[: -len("weight")] + "weight_scale"] = torch.ones(
                param.shape[0], 1, dtype=torch.float32)
        else:
            tensors[name] = param.detach().clone()
    save_file(tensors, str(comp / "model.safetensors"))
    (comp / "config.json").write_text(json.dumps(MicroConfig().as_dict()))
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "MicroW8a8Pipeline",
        component: ["micro_diffusion.model", "MicroDenoiser"],
    }))
    return comp


def _spec(cls: Any, path: Path) -> Any:
    return types.SimpleNamespace(
        type_hint=cls, pretrained_model_name_or_path=str(path),
        subfolder="", default_creation_method="from_pretrained")


def test_the_tree_really_is_a_contract_artifact(tmp_path: Path) -> None:
    """The fixture must be the thing, or everything below proves nothing."""
    _w8a8_component(tmp_path)
    art = w8a8.detect_w8a8_artifact(tmp_path)
    assert art is not None and art.component == "transformer"
    assert art.quantized, "no fp8 weight/weight_scale pairs were detected"


def test_a_MODULAR_component_routes_through_the_contract_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE issue. Before this, modular hydration reached plain
    `from_pretrained` and died inside `DiffusersAutoQuantizer`.

    Asserted by WHICH LOADER RAN, not by the leaf classes it produced: whether
    `Fp8ScaledLinear` actually replaces a leaf depends on `torch._scaled_mm`
    availability on the box, so a leaf-class assertion would be a CUDA test
    wearing a routing test's name. The property under test is the ROUTING.
    """
    from micro_diffusion.model import MicroDenoiser

    comp = _w8a8_component(tmp_path)
    called: list[tuple[str, str]] = []
    from gen_worker.models.w8a8 import load_w8a8_denoiser as _real_loader

    real = _real_loader

    def _spy(root: Any, art: Any, **kw: Any) -> Any:
        called.append((str(root), art.component))
        return real(root, art, **kw)

    # Patched by dotted PATH, not by attribute: `loading` re-exports
    # `load_w8a8_denoiser` without listing it in `__all__`, and strict mypy
    # (correctly) refuses an implicit re-export. Widening the production
    # surface to suit a test would be the wrong direction.
    monkeypatch.setattr(
        "gen_worker.models.loading.load_w8a8_denoiser", _spy)

    built = loading._contract_component_for_spec(
        _spec(MicroDenoiser, comp), "transformer", tmp_path)

    assert built is not None, (
        "the modular path did not route a `cozy.fp8-rowwise@1` tree through "
        "its registered loader — this is the h3 fault exactly")
    assert called, (
        "a module came back but `load_w8a8_denoiser` never ran — the tree was "
        "loaded generically, which is the defect wearing a passing assertion")
    assert called[0][1] == "transformer"


def test_a_PLAIN_tree_is_left_to_the_generic_loader(tmp_path: Path) -> None:
    """`None` means 'no contract here' and the caller's own load is correct —
    the dispatch must not capture every component it is shown."""
    from micro_diffusion.model import MicroConfig, MicroDenoiser

    comp = tmp_path / "transformer"
    comp.mkdir(parents=True)
    denoiser = MicroDenoiser(MicroConfig())
    save_file({n: p.detach().clone() for n, p in denoiser.named_parameters()},
              str(comp / "model.safetensors"))
    (comp / "config.json").write_text(json.dumps(MicroConfig().as_dict()))

    assert loading._contract_component_for_spec(
        _spec(MicroDenoiser, comp), "transformer", comp) is None


def test_BOTH_entry_points_ask_the_SAME_dispatch(tmp_path: Path) -> None:
    """One artifact shape, two entry points. Two dispatches that agree today
    are two dispatches that will disagree later — so there is only one, and
    this asserts both callers reach it rather than trusting they do."""
    import inspect

    src = inspect.getsource(loading)
    # the non-modular path
    assert "contracted = contract_loaded_component(" in src
    # the modular path, via the spec-typed shim
    assert "_contract_component_for_spec(specs[n], n, comp_src)" in src
    assert "return contract_loaded_component(src, name, cls=cls, src=src)" in src


def test_a_recognised_lane_with_NO_component_loader_REFUSES(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the invariant, and the reason this is a dispatch and
    not a lookup: a layout we RECOGNISE but cannot build at component level is
    a typed refusal naming our own code — never a silent fall-through to
    generic loading, which is what produced numerics that could not serve."""
    from micro_diffusion.model import MicroDenoiser

    comp = tmp_path / "transformer"
    comp.mkdir(parents=True)
    monkeypatch.setattr(
        loading, "detect_gguf_snapshot", lambda root: {"gguf": True})

    with pytest.raises(loading.ComponentExecutionLaneUnsupported) as caught:
        loading.contract_loaded_component(
            tmp_path, "transformer", cls=MicroDenoiser, src=comp)

    said = str(caught.value)
    assert "transformer" in said and "GGUF" in said
    assert "no component-level production loader" in said
