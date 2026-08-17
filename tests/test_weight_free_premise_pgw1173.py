"""The weight-free premise is FENCED on the trace path.

The whole compile-loop design rests on one sentence — *compilation is
weight-free* — and every VRAM number downstream is unsound if the trace path
can silently materialise real tensors. Three guards stand on that path and each
can read clean on a composition holding a real checkpoint:

* ``off_host_tensors`` walks for tensors that are NOT on the host, and the boot
  trace loads ``place=False``, so a real target sits on the host and the walk
  returns ``[]``;
* ``structure_only_components`` is an ANY check, so a family declaring two
  targets passes with one of them stranded on real weights;
* ``_assert_structure_honored`` iterates only what the builder BUILT, so a
  target the builder skipped — the tree's ``model_index.json`` does not name
  it, or there is no readable index at all, in which case
  ``model_index_components`` returns the empty set and every target is skipped
  — never enters its loop and the guard passes over an empty dict.

The third one is the defect class this whole redesign is about: a guard that
cannot fire for the case it is named for, whose silence is then read as proof.

These rows fence the premise on the path where §4.27 step 1 forbids weights
outright.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import msgspec
import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import boot_key, boot_trace_child  # noqa: E402
from gen_worker.cli import run as cli_run  # noqa: E402
from gen_worker.models import structure_only  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"

#: Every compile-target vocabulary the fleet's endpoint declarations actually
#: use, read out of `inference-endpoints/*/src/*/main.py` on 2026-08-12. The
#: fence is exercised against each because the hole is per-SHAPE-of-target: a
#: plain component name, a nested attribute path, and a bound method on a
#: component are three different resolutions and the pre-fence guards handled
#: only the first.
FLEET_TARGETS: Dict[str, tuple] = {
    # transformer families: flux.1-dev/-schnell, flux.2-klein-4b/-9b,
    # qwen-image, z-image, wan-2.2, ltx-video-2.3, krea-2, ernie
    "transformer-families": ("transformer",),
    # sdxl, sd15
    "unet-families": ("unet",),
    # qwen-image's documented boundary form (`transformer.denoise`)
    "nested-attribute": ("transformer.denoise",),
    # a decoder target, the form minimax-h3's source names
    "component-method": ("vae.decode",),
    # a MULTI-target declaration: the arity the ANY check could not see
    "multi-target": ("transformer", "vae"),
}


# ---------------------------------------------------------------------------
# Fixtures: what production actually composes
# ---------------------------------------------------------------------------


class _Composed:
    """A diffusers-shaped composition, the same shape
    `test_trace_child_placement_pgw1124` uses: `.components` is the walk's
    entry point and every part is also a plain attribute."""

    def __init__(self, **parts: Any) -> None:
        self._parts: Dict[str, Any] = dict(parts)
        for name, part in parts.items():
            setattr(self, name, part)

    @property
    def components(self) -> Dict[str, Any]:
        return dict(self._parts)


class _Denoiser(nn.Module):
    """A component carrying a nested compile target, as qwen-image's boundary
    documents (`targets=("transformer.denoise",)`)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, dtype=torch.bfloat16)

    def forward(self, x: Any) -> Any:  # pragma: no cover — never called
        return self.proj(x)

    def denoise(self, x: Any) -> Any:  # pragma: no cover — never called
        return self.proj(x)


class _Vae(nn.Module):
    """A component carrying a bound-method compile target (`vae.decode`)."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(8, 8, dtype=torch.bfloat16)

    def forward(self, x: Any) -> Any:  # pragma: no cover — never called
        return self.up(x)

    def decode(self, x: Any) -> Any:  # pragma: no cover — never called
        return self.up(x)


#: The component class each declared target root needs in order to RESOLVE.
#: A plain `nn.Linear` has no `.decode`, so a `vae.decode` row built on one
#: resolves to nothing and would pass the fence by being invisible to it —
#: which is the failure this file exists to rule out, not to reproduce.
_ROOT_CLASS = {"vae": _Vae, "transformer": _Denoiser}


def _root_module(root: str) -> nn.Module:
    return _ROOT_CLASS.get(root, lambda: nn.Linear(8, 8))()


def _virtual(module: nn.Module | None = None, device: str = "cuda") -> nn.Module:
    """A compile target built the way the boot trace builds it — through the
    production `virtualize`, not a hand-written `device="meta"` module, which
    is a different object with a different failure."""
    mod = nn.Linear(8, 8) if module is None else module
    structure_only.virtualize(mod, device=device, dtype=torch.bfloat16)
    return mod


def _real(module: nn.Module | None = None, device: str = "cpu") -> nn.Module:
    """A component holding a real checkpoint, wherever it was placed."""
    mod = nn.Linear(8, 8, dtype=torch.bfloat16) if module is None else module
    return mod.to(device)


class _TinyTrace(nn.Module):
    def forward(self, value: Any) -> Any:
        return value.sin()


def _traced_class(aot_mint: Any) -> Any:
    """One real export crossing the same TCG declaration seam as production."""
    from gen_worker._vendor.torch_compiled_graphs import CallIngress, CallInput

    example = torch.ones(2)
    program = torch.export.export(_TinyTrace(), (example,))
    ingress = CallIngress(
        parameters=("value",),
        flat_arity=1,
        inputs=(
            CallInput(
                "value", 0, "value", 0, (), "value", "float32", (2,),
            ),
        ),
    )
    return aot_mint.TracedClass(
        name="transformer",
        block={
            "target": "transformer",
            "fork": [],
            "class_dims": [],
            "graph": {
                "v": 3,
                "lifted_inputs": [],
                "pytree": {
                    "in": "leaf",
                    "out": "leaf",
                    "ingress": ingress.as_dict(),
                },
                "specialization": {},
            },
        },
        nodes=len(program.graph_module.graph.nodes),
        program=program,
        declared=1,
    )


# ---------------------------------------------------------------------------
# The premise, and the fixture that states it
# ---------------------------------------------------------------------------


def test_a_virtualized_target_holds_zero_real_parameter_bytes() -> None:
    """If this ever stops holding, every row below is testing a different
    object and should fail HERE first."""
    pipe = _Composed(transformer=_virtual())

    assert structure_only.weight_free_breaches(pipe, ("transformer",)) == ()


def test_real_buffers_on_a_virtual_target_are_NOT_a_breach() -> None:
    """A structure-only component's buffers stay real by construction — they
    are config-derived tables and a literal-bearing family ships them inside
    the cell. Counting them would make the fence refuse every correct trace."""
    target = nn.Linear(8, 8)
    target.register_buffer("rope", torch.ones(64, dtype=torch.bfloat16))
    # On the host: a buffer is REALLY moved by `virtualize`, so building this
    # one on the card would need a card, and the property under test is
    # device-independent.
    pipe = _Composed(transformer=_virtual(target, device="cpu"))

    real = [b for _n, b in pipe.transformer.named_buffers()
            if type(b).__name__ != "FakeTensor"]
    assert real, "the fixture must carry a REAL buffer or it proves nothing"
    assert structure_only.weight_free_breaches(pipe, ("transformer",)) == ()


# ---------------------------------------------------------------------------
# Hole 1 — the ANY check, and the host-resident weights nobody was walking for
# ---------------------------------------------------------------------------


def test_a_stranded_SECOND_target_is_a_breach_though_both_old_guards_are_green(
) -> None:
    """THE hole, stated with its two green guards next to it.

    A two-target family whose first target virtualized and whose second did
    not. `off_host_tensors` sees nothing because `place=False` left the real
    weights on the HOST; `structure_only_components` is satisfied by the one
    virtual component. Both green, and the pipeline about to be traced is
    holding a checkpoint.
    """
    pipe = _Composed(transformer=_virtual(), vae=_real())

    assert boot_trace_child.off_host_tensors(pipe) == [], (
        "the off-host walk must be GREEN here — that is the point of the row")
    assert structure_only.structure_only_components(pipe) == ("transformer",), (
        "the ANY check must be GREEN here too")

    breaches = structure_only.weight_free_breaches(pipe, ("transformer", "vae"))
    assert [b.component for b in breaches] == ["vae"]
    assert breaches[0].reason == "not_structure_only"
    assert breaches[0].real_param_bytes > 0
    assert breaches[0].devices == ("cpu",)


def test_the_fence_raises_StructureNotHonored_and_names_the_component() -> None:
    pipe = _Composed(transformer=_virtual(), vae=_real())

    with pytest.raises(structure_only.StructureNotHonored) as caught:
        structure_only.assert_weight_free(pipe, ("transformer", "vae"))

    text = str(caught.value)
    assert "vae" in text and "REAL parameters" in text
    # The typed contract the mint path relies on: this is the FAIL-CLOSED
    # strand, never the buildable one a caller may fall back from.
    assert isinstance(caught.value, structure_only.StructureOnlyUnsupported)


@pytest.mark.parametrize("label", sorted(FLEET_TARGETS))
def test_the_fence_fires_for_every_target_shape_the_fleet_declares(
    label: str,
) -> None:
    """Per-family, by the shape of the target each family declares.

    `transformer` / `unet` are plain components; `transformer.denoise` is a
    nested attribute path and `vae.decode` a bound method on a component —
    both resolve through the ONE target authority and neither is a
    `getattr(pipe, name)`. A fence with its own weaker resolver would read the
    last two as "not carried" and skip silently, which is the failure mode
    being fenced against.
    """
    targets = FLEET_TARGETS[label]
    roots = {t.split(".")[0] for t in targets}

    clean = _Composed(**{
        root: _virtual(_root_module(root)) for root in sorted(roots)})
    assert structure_only.weight_free_breaches(clean, targets) == (), (
        f"{label}: an all-virtual composition must pass")
    structure_only.assert_weight_free(clean, targets)

    dirty = _Composed(**{
        root: _real(_root_module(root)) for root in sorted(roots)})
    breaches = structure_only.weight_free_breaches(dirty, targets)
    assert {b.component for b in breaches} == set(targets), (
        f"{label}: every declared target holding a checkpoint must be named "
        f"— got {[b.component for b in breaches]}")
    with pytest.raises(structure_only.StructureNotHonored):
        structure_only.assert_weight_free(dirty, targets)


def test_a_target_this_slot_does_not_carry_is_not_a_breach() -> None:
    """A multi-slot family's auxiliary slot (a refiner, a second pipeline)
    legitimately has no denoiser of the primary's name. Refusing there would
    make the fence unusable rather than strict."""
    pipe = _Composed(transformer=_virtual())

    assert structure_only.weight_free_breaches(
        pipe, ("transformer", "unet")) == ()


def test_a_pipeline_carrying_NO_declared_target_is_refused() -> None:
    """A trace with no target is not a weight-free trace, it is a trace of
    nothing — and reporting it clean is how a derivation looks proven and
    means nothing."""
    pipe = _Composed(text_encoder=_real())

    with pytest.raises(structure_only.StructureNotHonored) as caught:
        structure_only.assert_weight_free(pipe, ("transformer",))
    assert "carries none of the declared compile target" in str(caught.value)


def test_a_stamped_target_that_still_holds_real_parameters_is_a_breach() -> None:
    """The stamp is not the property. A module virtualized and then handed a
    real parameter — a submodule swapped in after the build, the shape a
    partial re-materialisation leaves — is stamped and weightful."""
    target = _virtual()
    target.register_parameter(
        "late", nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16),
                             requires_grad=False))
    pipe = _Composed(transformer=target)

    breaches = structure_only.weight_free_breaches(pipe, ("transformer",))
    assert [b.reason for b in breaches] == ["real_parameters"]


# ---------------------------------------------------------------------------
# Hole 2 — the guard that iterated an empty dict
# ---------------------------------------------------------------------------


def test_a_requested_target_the_builder_SKIPPED_is_a_typed_refusal() -> None:
    """`_assert_structure_honored` used to iterate only what was BUILT.

    With `injected == {}` — which is what a tree whose `model_index.json` is
    absent or does not name the target produces, since
    `model_index_components` returns the empty set and the builder loop
    `continue`s over every target — the loop ran zero times and the guard
    passed over a pipeline that had loaded the checkpoint.

    The typed outcome is the BUILDABLE strand, not the swallowed one: nothing
    was built, so the mint child's fallback to a real-weight export stays
    correct and, crucially, stays RECORDED, while the boot trace — which may
    not fall back at all — refuses.
    """
    pipe = _Composed(transformer=_real())

    with pytest.raises(structure_only.StructureOnlyUnsupported) as caught:
        cli_run._assert_structure_honored(
            pipe, {}, requested=("transformer",), slot="pipeline")

    assert not isinstance(caught.value, structure_only.StructureNotHonored), (
        "nothing was BUILT here, so this is the strand a mint may fall back "
        "from — typing it as the fail-closed one would turn every "
        "index-less tree into a dead mint")
    assert "model_index.json" in str(caught.value)


def test_the_swallowed_case_still_types_as_StructureNotHonored() -> None:
    """The pre-existing guard is unchanged where it could already fire: the
    component WAS built weight-free and the pipeline threw it away."""
    built = _virtual()
    pipe = _Composed(transformer=_real())

    with pytest.raises(structure_only.StructureNotHonored):
        cli_run._assert_structure_honored(
            pipe, {"transformer": built}, requested=("transformer",),
            slot="pipeline")


def test_a_requested_target_this_slot_does_not_carry_stays_silent() -> None:
    pipe = _Composed(vae=_real())

    cli_run._assert_structure_honored(
        pipe, {}, requested=("transformer",), slot="refiner")


def test_a_dotted_target_covered_by_its_injected_ROOT_is_not_refused() -> None:
    """`vae.decode`'s weights live in the `vae` the builder injected, and
    `injected` is keyed on the component. Refusing on the string mismatch
    would fail every family that declares a method target."""
    built = _virtual()
    pipe = _Composed(vae=built)

    cli_run._assert_structure_honored(
        pipe, {"vae": built}, requested=("vae.decode",), slot="pipeline")


# ---------------------------------------------------------------------------
# The real vehicle: `boot_trace_child.run` over micro-diffusion
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def micro_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("accelerate")
    pytest.importorskip("diffusers")
    if str(MICRO_SRC) not in sys.path:
        sys.path.insert(0, str(MICRO_SRC))
    from micro_diffusion.weights import SEED, materialize

    return materialize(tmp_path_factory.mktemp("micro-tree"), seed=SEED)


@pytest.fixture
def micro_declaration(micro_tree: Path) -> None:
    from gen_worker.api import export_contract as ec

    import micro_diffusion.aot_declaration as decl

    ec.register_export_declaration(decl.DECLARATION, replace=True)


def _job(micro_tree: Path, report: Path, *, extra_targets: tuple = ()) -> Any:
    from gen_worker.api.binding import ModelRef
    from gen_worker.child_contract import CompileSpec, MintSlot
    from gen_worker.registry import collect_endpoints

    specs = collect_endpoints(["harness.rig_runtime", "micro_diffusion.main"])
    spec = next(s for s in specs if s.name == "generate")
    cell = spec.compile_cell()
    cfg = CompileSpec(
        shapes=tuple(
            tuple(int(v) for v in row) for row in (cell.shapes or ())),
        targets=tuple(str(t) for t in (cell.targets or ())) + extra_targets,
        family=str(cell.family or ""),
        lora_bucket=int(cell.lora_bucket or 0),
        guidance_scales=tuple(float(v) for v in (cell.guidance_scales or ())),
        text_lens=tuple(int(v) for v in (cell.text_lens or ())),
    )
    return boot_key.TraceJob(
        function="generate",
        modules=("harness.rig_runtime", "micro_diffusion.main"),
        family=cfg.family,
        cfg=cfg,
        slots={"pipeline": MintSlot(
            ref=ModelRef(source="tensorhub", path="cozy/micro-diffusion",
                         release="prod"),
            path=str(micro_tree))},
        report=str(report),
        code_digest=boot_key.CODE_DIGEST,
    )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    job: Any,
    report_path: Path,
    *,
    seen: list[Any] | None = None,
) -> Any:
    from gen_worker import aot_mint
    from gen_worker.models import provision

    # `_load_injected_model` keeps a PROCESS-GLOBAL warm cache keyed on
    # (annotation, path, overrides, structure targets, placement). Two rows
    # composing the same slot the same way would otherwise share one object
    # and the second would be asserting about the first row's pipeline.
    cli_run._INJECTED_CACHE.clear()
    monkeypatch.setattr(
        provision, "place_pipeline", lambda pipe, **kw: {"mode": "off"})

    def _traced(pipeline: Any, spec: Any, decl: Any, **_: Any) -> Any:
        traced = _traced_class(aot_mint)
        if seen is not None:
            seen.append(traced)
        return iter([traced])

    monkeypatch.setattr(aot_mint, "trace_for_key", _traced)
    rc = boot_trace_child.run(job)
    return rc, msgspec.json.decode(
        report_path.read_bytes(), type=boot_key.TraceReport)


def test_the_real_child_still_derives_when_the_premise_HOLDS(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, micro_tree: Path,
    micro_declaration: None,
) -> None:
    """The fence must not cost a correct derivation — the control for the row
    below, run over the same real loader and the same real checkpoint."""
    if str(MICRO_SRC) not in sys.path:
        sys.path.insert(0, str(MICRO_SRC))
    monkeypatch.syspath_prepend(str(REPO / "tests"))

    report_path = tmp_path / "ok.json"
    rc, report = _run(monkeypatch, _job(micro_tree, report_path), report_path)

    assert report.ok and rc == boot_key.EXIT_OK, (
        f"the trace child refused {report.reason!r}: {report.detail[:400]}")


def test_a_tcg_declaration_refusal_releases_the_exported_program(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    micro_tree: Path,
    micro_declaration: None,
) -> None:
    from gen_worker import aot_mint

    class RefusingSpec:
        def declare(self) -> Any:
            raise ValueError("declaration refused")

    monkeypatch.syspath_prepend(str(REPO / "tests"))
    monkeypatch.setattr(
        aot_mint,
        "tcg_graph_class_spec",
        lambda *_args, **_kwargs: RefusingSpec(),
    )
    report_path = tmp_path / "refused.json"
    seen: list[Any] = []

    rc, report = _run(
        monkeypatch,
        _job(micro_tree, report_path),
        report_path,
        seen=seen,
    )

    assert rc == boot_key.EXIT_REFUSED
    assert report.reason == "trace_refused"
    assert len(seen) == 1 and seen[0].program is None


def test_the_real_child_REFUSES_a_second_target_that_loaded_real_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, micro_tree: Path,
    micro_declaration: None,
) -> None:
    """The hole on the production vehicle.

    A second declared target (`vae`) that the structure-only builder SKIPS,
    reproduced the way production produces it: `model_index_components` does
    not name it. That is not a contrivance — the function returns the EMPTY
    SET for any tree with no readable `model_index.json` ("single-file
    checkpoints, transformers layouts", its own docstring), in which case
    every target is skipped and `injected` is empty.

    Only the INDEX is patched. The subject is untouched: the real `run_setup`,
    the real pgw#1080 build of the real target, and the real
    `provision.load_slot` composing micro-diffusion's other components from
    its actual checkpoint — including the `vae` this now loads with weights.
    Before the fence this reported `ok=True`, with `_assert_structure_honored`
    iterating a dict the skipped target was never in.
    """
    if str(MICRO_SRC) not in sys.path:
        sys.path.insert(0, str(MICRO_SRC))
    monkeypatch.syspath_prepend(str(REPO / "tests"))

    from gen_worker.models import loading

    real_index = loading.model_index_components
    monkeypatch.setattr(
        loading, "model_index_components",
        lambda path: {c for c in real_index(path) if c != "decoder"})

    report_path = tmp_path / "breach.json"
    job = _job(micro_tree, report_path, extra_targets=("decoder",))
    rc, report = _run(monkeypatch, job, report_path)

    assert not report.ok and rc == boot_key.EXIT_REFUSED
    assert report.reason in (
        "structure_not_honored", "structure_unsupported"), report.reason
    assert "decoder" in report.detail, report.detail[:400]


def test_the_real_child_REFUSES_a_target_that_is_STAMPED_and_weightful(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, micro_tree: Path,
    micro_declaration: None,
) -> None:
    """The case only the boot-trace fence can catch — its call site's proof.

    Here the builder DOES run, the pipeline DOES carry the module it built,
    and the module is stamped structure-only — so `_assert_structure_honored`
    is satisfied on every question it asks (`got is module`) and cannot see
    that the parameters are real. That is what a partial virtualization leaves
    (a submodule swapped in after the build, a re-materialisation that only
    half completed), and it is why the loader's guard is not sufficient on
    this path: only `assert_weight_free` reads the TENSORS.

    Deleting the `assert_weight_free` call from `boot_trace_child.run` turns
    this row red and nothing else in the file.
    """
    if str(MICRO_SRC) not in sys.path:
        sys.path.insert(0, str(MICRO_SRC))
    monkeypatch.syspath_prepend(str(REPO / "tests"))

    real_build = structure_only.build_component

    def _half_virtual(tree: Any, comp: str, **kw: Any) -> Any:
        module, facts = real_build(tree, comp, **kw)
        # Stamped, injected, honored — and holding a real checkpoint tensor.
        module.register_parameter(
            "_late", nn.Parameter(torch.ones(64, 64, dtype=torch.bfloat16),
                                  requires_grad=False))
        return module, facts

    monkeypatch.setattr(structure_only, "build_component", _half_virtual)

    report_path = tmp_path / "stamped.json"
    rc, report = _run(monkeypatch, _job(micro_tree, report_path), report_path)

    assert not report.ok and rc == boot_key.EXIT_REFUSED, (
        "a stamped target holding real parameters reached the export")
    assert report.reason == "structure_not_honored", report.reason
    assert "REAL parameters" in report.detail, report.detail[:400]


def test_the_refusal_token_is_in_the_boot_adopt_vocabulary() -> None:
    """pgw#1116's rule: a refusal nobody can enumerate is the next silent
    one."""
    from gen_worker import boot_adopt

    assert "structure_not_honored" in boot_adopt.REASONS
    assert "structure_unsupported" in boot_adopt.REASONS
