"""A boot-trace child holds no card, and a VIRTUAL structure is not a
placement miss.

Two independent defects, either of which makes every boot-trace child of a
family fail `CUDA OOM left the pipeline mixed-device` deterministically:

1. the child must NOT run the serving PLACEMENT ladder — that pushes the slot's
   real non-target components (a 15.5 GiB text encoder) onto the card the
   serving parent already occupies;
2. the CPU rollback that OOM demotes through must not count the
   deliberately-virtual compile target as "not on cpu", or it can never
   succeed and a recoverable ladder step becomes FATAL.

The fixtures here are what production actually composes: the compile target is
built by `structure_only.virtualize` (the function `build_component` ends in)
on the compute device, so its parameters are FAKE and its module declines
`_apply` — not a hand-written `nn.Linear(device="meta")`, which is a different
object with a different failure and would have proved nothing about either.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import msgspec
import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import boot_adopt, boot_key, boot_trace_child  # noqa: E402
from gen_worker.models import memory, provision, structure_only  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"


# ---------------------------------------------------------------------------
# The composition, as the loader hands it back
# ---------------------------------------------------------------------------


class _Composed:
    """A diffusers-shaped composition: `.components` is the walk's own entry
    point, and `.to()` moves each component exactly as the pipeline class
    does — including declining to move the frozen structure-only one."""

    def __init__(self, **parts: Any) -> None:
        self._parts: Dict[str, Any] = dict(parts)
        for name, part in parts.items():
            setattr(self, name, part)

    @property
    def components(self) -> Dict[str, Any]:
        return dict(self._parts)

    def to(self, device: str) -> "_Composed":
        if str(device).startswith("cuda"):
            # The card the serving parent is already resident on.
            raise torch.cuda.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 15.50 GiB")
        for part in self._parts.values():
            part.to(device)
        return self


def _virtual(device: str = "cuda") -> nn.Module:
    """The compile target, built the way the boot trace builds it."""
    module = nn.Linear(8, 8)
    structure_only.virtualize(module, device=device, dtype=torch.bfloat16)
    return module


def _real(device: str = "cpu") -> nn.Module:
    """A non-target component: real weights, wherever they were placed."""
    return nn.Linear(8, 8, dtype=torch.bfloat16).to(device)


def test_the_target_is_fake_on_the_compute_device_which_is_the_premise() -> None:
    """Everything below is about a component whose parameters claim the card
    and allocate nothing. If that ever stops being what production builds,
    these rows are testing a different object and should fail here first."""
    param = next(_virtual("cuda").parameters())
    assert type(param).__name__ == "FakeTensor"
    assert param.device.type == "cuda"


# ---------------------------------------------------------------------------
# Defect 2 — the fatal-maker
# ---------------------------------------------------------------------------


def test_a_virtual_compile_target_is_not_a_placement_miss() -> None:
    pipe = _Composed(transformer=_virtual("cuda"), text_encoder=_real("cpu"))

    assert memory.device_mismatches(pipe, "cpu") == [], (
        "the structure-only target is virtual BY DESIGN — counting it as a "
        "misplaced tensor is what made the CPU rollback unsatisfiable")


def test_the_cpu_rollback_of_a_virtual_composition_actually_succeeds() -> None:
    pipe = _Composed(transformer=_virtual("cuda"), text_encoder=_real("cpu"))

    # Never raises (`comp.to()` on a frozen structure is a no-op, on a meta
    # tensor a NotImplementedError) and reports nothing left behind.
    assert memory.repair_device_placement(pipe, "cpu") == []


def test_a_meta_tensor_outside_a_virtual_component_is_STILL_a_miss() -> None:
    """The exemption is scoped, not blanket: an unmaterialized load is the
    thing `meta_tensors` exists to report and must keep reporting."""
    pipe = _Composed(
        transformer=_virtual("cuda"), vae=nn.Linear(8, 8, device="meta"))

    assert sorted({c for c, _, _ in memory.device_mismatches(pipe, "cpu")}) \
        == ["vae"]
    assert sorted({c for c, _ in memory.meta_tensors(pipe)}) == ["vae"]


def test_a_trace_child_oom_demotes_instead_of_fataling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE census failure, end to end through the real `place_pipeline`.

    Deliberately does NOT stub `_move_pipeline_to_cpu` / `repair_device_
    placement`: those two ARE the defect, and stubbing them is why the OOM
    ladder looked proven while every qwen-image boot fataled on it. (The suite
    that did stub them, `test_place_pipeline_strict_vram_th1043`, went with
    `strict_vram` in th#1867.)
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(memory, "select_auto_mode", lambda **_: "off")
    monkeypatch.setattr(memory, "flush_memory", lambda: None)
    monkeypatch.setattr(memory, "get_available_vram_gb", lambda: 1.0)

    rungs: List[str] = []

    def _apply(pipeline: Any, *, mode: str, logger: Any = None) -> Dict[str, Any]:
        rungs.append(mode)
        return {"mode": mode}

    monkeypatch.setattr(memory, "apply_low_vram_config", _apply)

    pipe = _Composed(transformer=_virtual("cuda"), text_encoder=_real("cpu"))
    applied = memory.place_pipeline(pipe, mode="auto", ref="qwen-image/t2i")

    assert applied["mode"] == "model_offload"
    assert applied["oom_demotions"] == 1
    # The `off` rung OOMs in `pipeline.to("cuda")` itself — before any config
    # is applied — which is exactly where the fleet's children died.
    assert rungs == ["model_offload"], (
        "the OOM is a ladder transition (gw#463); it became "
        "`CUDA OOM left the pipeline mixed-device and CPU rollback failed` "
        "only because the rollback walk could not be satisfied")


# ---------------------------------------------------------------------------
# Defect 1 — the trigger
# ---------------------------------------------------------------------------


def test_off_host_tensors_names_real_residents_and_nothing_virtual() -> None:
    clean = _Composed(transformer=_virtual("cuda"), text_encoder=_real("cpu"))
    assert boot_trace_child.off_host_tensors(clean) == []

    placed = _Composed(
        transformer=_virtual("cuda"),
        text_encoder=nn.Linear(8, 8, device="meta"))  # a real, off-host module
    assert [c for c, _, _ in boot_trace_child.off_host_tensors(placed)] \
        == ["text_encoder", "text_encoder"]


def test_the_refusal_is_in_the_boot_adopt_vocabulary() -> None:
    """pgw#1116's rule: a refusal nobody can enumerate is the next silent
    one. (`test_boot_adopt_observability_pgw1116` reads the site out of the
    tree; this states the token the site names.)"""
    assert "real_weights_resident" in boot_adopt.REASONS


# ---------------------------------------------------------------------------
# Defect 1, on the real vehicle: `boot_trace_child.run` over micro-diffusion
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
    """See `test_boot_adopt_observability_pgw1116.micro_declaration`: sibling
    files empty the process-global declaration registry, and by then
    `micro_diffusion.main` is already imported, so nothing re-registers."""
    from gen_worker.api import export_contract as ec

    import micro_diffusion.aot_declaration as decl

    ec.register_export_declaration(decl.DECLARATION, replace=True)


def _job(micro_tree: Path, report: Path) -> Any:
    from gen_worker.api.binding import ModelRef
    from gen_worker.child_contract import CompileSpec, MintSlot
    from gen_worker.registry import collect_endpoints

    specs = collect_endpoints(["harness.rig_runtime", "micro_diffusion.main"])
    spec = next(s for s in specs if s.name == "generate")
    cell = spec.compile_cell()
    cfg = CompileSpec(
        shapes=tuple(
            tuple(int(v) for v in row) for row in (cell.shapes or ())),
        targets=tuple(str(t) for t in (cell.targets or ())),
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
                         tag="prod"),
            path=str(micro_tree))},
        report=str(report),
        code_digest=boot_key.CODE_DIGEST,
    )


def test_the_boot_trace_child_never_runs_the_serving_placement_ladder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, micro_tree: Path,
    micro_declaration: None,
) -> None:
    """The REAL child over the REAL loader, on the vehicle the pods ran.

    Everything up to the export is production: `run_setup` -> the pgw#1080
    structure-only build of the compile target -> `provision.load_slot`
    composing micro-diffusion's other components from its actual checkpoint.
    `place_pipeline` is where qwen-image's 15.5 GiB went onto the parent's
    card, so the property is that the child never reaches it — not that it
    survives it.

    Only `trace_for_key` is stubbed, and not to make the row pass: called
    in-process it is `mi.guard`'s own `actionable_only` heuristic that breaks
    (torch.export's small real allocations are attributable to the pytest
    frame, which on a pod is `<frozen runpy>` and therefore ignored). What is
    under test here is the composition, and the composition is real.
    """
    if str(MICRO_SRC) not in sys.path:
        sys.path.insert(0, str(MICRO_SRC))
    monkeypatch.syspath_prepend(str(REPO / "tests"))

    placements: List[Tuple[str, str]] = []

    def _spy(pipe: Any, **kwargs: Any) -> Dict[str, Any]:
        placements.append((type(pipe).__name__, str(kwargs.get("ref", ""))))
        return {"mode": "off"}

    monkeypatch.setattr(provision, "place_pipeline", _spy)

    from gen_worker import aot_mint

    composed: List[Any] = []

    def _traced(pipeline: Any, spec: Any, decl: Any, **_: Any) -> Any:
        composed.append(pipeline)
        return iter([aot_mint.TracedClass(
            name="transformer", block={"entry": "transformer"}, nodes=1,
            program=None, declared=1)])

    monkeypatch.setattr(aot_mint, "trace_for_key", _traced)

    report_path = tmp_path / "report.json"
    rc = boot_trace_child.run(_job(micro_tree, report_path))
    report = msgspec.json.decode(
        report_path.read_bytes(), type=boot_key.TraceReport)

    assert report.ok and rc == boot_key.EXIT_OK, (
        f"the trace child refused {report.reason!r}: {report.detail[:400]}")
    assert report.structure_only, (
        "the target must still be composed structure-only — skipping "
        "placement must not have changed WHAT the child composes")
    assert placements == [], (
        "the boot-trace child entered the serving placement ladder; on a pod "
        "that is the moment the slot's real non-target components land on the "
        f"card the serving parent owns ({placements})")
    assert boot_trace_child.off_host_tensors(composed[0]) == [], (
        "the invariant, on a real composition: every real component of a "
        "trace child's pipeline stays on the host")
