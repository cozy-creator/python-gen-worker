"""pgw#689: the swap benchmark must load what SERVING loads, a broken
diagnostic must not look like a broken release, and a re-exported endpoint
must never be dropped silently.

Four tapes, all red before the fix:

1. ``benchmarks.swap_latency._load_component`` on a REAL modelopt-shaped
   w8a8 tree — the flavor the fleet actually serves. Pre-fix it called
   ``cls.from_pretrained`` itself and died reconstructing
   ``NVIDIAModelOptConfig`` (evidence tape below pins that exact mechanism
   on the pinned diffusers); post-fix it goes through
   ``models.loading.load_component``, so the module comes back off the w8a8
   artifact lane.
2. ``bench_load`` end to end over that tree with a fake-CUDA seam (the
   real-GPU numbers are a pod's job; the LOADABILITY is CI's).
3. The diagnostics handler converts every benchmark-domain failure into a
   typed outcome and returns normally — the job succeeds, so no
   ``model_load_failure_streak`` signal is ever emitted for it.
4. Discovery's out-of-package skip is a WARNING naming the class.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any, List

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")


@pytest.fixture(scope="module")
def tiny_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minimal REAL diffusers tree (unet + scheduler)."""
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

    root = tmp_path_factory.mktemp("pgw689") / "src"
    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"), norm_num_groups=8,
    )
    DDPMPipeline(unet=unet, scheduler=DDPMScheduler()).save_pretrained(str(root))
    return root


@pytest.fixture(scope="module")
def w8a8_tree(tiny_tree: Path) -> Path:
    """The same tree quantized to the gw#534 w8a8 contract — carrying the
    modelopt ``quantization_config`` block that broke the benchmark."""
    import json

    from gen_worker.models.w8a8 import quantize_tree_w8a8

    tree = quantize_tree_w8a8(tiny_tree, tiny_tree.parent / "w8a8")
    cfg = json.loads((tree / "unet" / "config.json").read_text())
    assert cfg["quantization_config"] == {
        "quant_method": "modelopt", "quant_algo": "FP8"}
    return tree


# ---------------------------------------------------------------------------
# 1: the evidence — a bare from_pretrained on a modelopt tree is the fatal
# ---------------------------------------------------------------------------


def test_bare_from_pretrained_on_a_modelopt_tree_is_the_recorded_fatal(
    w8a8_tree: Path,
) -> None:
    """The mechanism, reproduced on the pinned stack (diffusers 0.39.0 —
    the release's own version): diffusers rebuilds ``NVIDIAModelOptConfig``
    from the stored block, whose constructor demands a ``quant_type`` the
    block does not carry. This call IS what the benchmark used to make per
    component, and the message below is verbatim the ie#546 fatal. If a
    diffusers bump ever makes this call succeed, this tape is the thing to
    update — the fix does not depend on it."""
    from diffusers import UNet2DModel

    with pytest.raises(TypeError) as excinfo:
        UNet2DModel.from_pretrained(
            str(w8a8_tree / "unet"), torch_dtype=torch.bfloat16)
    assert "NVIDIAModelOptConfig.__init__()" in str(excinfo.value)
    assert "quant_type" in str(excinfo.value)
    # ...and dropping the torch_dtype kwarg (the retry the benchmark did)
    # fails IDENTICALLY — the fallback never could have helped.
    with pytest.raises(TypeError, match="quant_type"):
        UNet2DModel.from_pretrained(str(w8a8_tree / "unet"))


# ---------------------------------------------------------------------------
# 2: the fix — the benchmark loads through the production component path
# ---------------------------------------------------------------------------


def test_benchmark_loads_a_quantized_component_via_the_serve_path(
    tiny_tree: Path, w8a8_tree: Path,
) -> None:
    from diffusers import UNet2DModel

    from gen_worker.benchmarks import swap_latency as bench
    from gen_worker.models import w8a8

    if hasattr(w8a8.w8a8_gemm_mode, "cache_clear"):
        w8a8.w8a8_gemm_mode.cache_clear()

    unet = bench._load_component(w8a8_tree, "unet")

    # The w8a8 artifact lane ran — the same loader load_from_pretrained
    # routes a served pipeline through — not a bare from_pretrained.
    assert getattr(unet, "_cozy_w8a8_mode", None) in (
        "rowwise", "pertensor", "dequant")
    # ...and it materialized the real weights, to fp8 rounding.
    ref = UNet2DModel.from_pretrained(str(tiny_tree / "unet"))
    name = w8a8.detect_w8a8_artifact(w8a8_tree).quantized[0] + ".weight"
    a = ref.state_dict()[name].float()
    b = unet.state_dict()[name].float()
    rel = ((a - b).abs() / a.abs().clamp(min=1e-3)).max().item()
    assert rel < 0.13


def test_unquantized_siblings_still_load_and_honor_the_lane_dtype(
    w8a8_tree: Path,
) -> None:
    """A w8a8 tree's NON-denoiser components take the ordinary path — and
    inherit the quant lane's bf16 compute default, not the tree's majority
    on-disk sniff (pgw#675/pgw#683's rule, now shared by construction)."""
    from gen_worker.models.loading import load_component

    sched = load_component(w8a8_tree, "scheduler")
    assert type(sched).__name__ == "DDPMScheduler"


def test_bench_load_walks_a_quantized_tree_end_to_end(
    w8a8_tree: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `load` case over a quantized tree. CUDA is faked (this box has
    none, and the real numbers are the pod's run) — what CI proves is that
    every component of a served flavor is loadable at all, which is exactly
    what was broken."""
    from gen_worker.benchmarks import swap_latency as bench

    monkeypatch.setattr(
        torch.nn.Module, "to", lambda self, *a, **k: self, raising=True)
    monkeypatch.setattr(bench, "_sync", lambda: None)
    monkeypatch.setattr(bench, "_cuda_bytes", lambda: 0)

    rows: List[bench.Row] = []
    loaded = bench.bench_load(w8a8_tree, rows.append)

    assert "unet" in loaded
    assert getattr(loaded["unet"], "_cozy_w8a8_mode", None) is not None
    labels = {r.label for r in rows}
    assert f"{w8a8_tree.name}/unet" in labels
    assert f"{w8a8_tree.name}/TOTAL" in labels
    assert any(r.bytes > 0 for r in rows)


def test_lanes_without_a_component_loader_refuse_by_name(
    tmp_path: Path,
) -> None:
    """svdq builds its denoiser inside the pipeline load. Handing back a
    plain from_pretrained module would be measuring something serving never
    runs — refuse, named."""
    import json

    from gen_worker.models.loading import (
        ComponentLaneUnsupported, load_component,
    )

    tree = tmp_path / "svdq"
    (tree / "transformer").mkdir(parents=True)
    (tree / "model_index.json").write_text(json.dumps({
        "_class_name": "FluxPipeline",
        "transformer": ["diffusers", "FluxTransformer2DModel"],
    }))
    from safetensors.torch import save_file

    from gen_worker.models.svdq import SVDQ_METHOD, detect_svdq_artifact

    save_file(
        {"x": torch.zeros(2)},
        str(tree / "transformer" / "svdq-int4_r32-model.safetensors"),
        metadata={"model_class": "NunchakuFluxTransformer2dModel",
                  "quantization_config": json.dumps({
                      "method": SVDQ_METHOD, "rank": 32,
                      "weight": {"dtype": "int4"}})},
    )
    assert detect_svdq_artifact(tree) is not None
    with pytest.raises(ComponentLaneUnsupported, match="svdq"):
        load_component(tree, "transformer")


def test_the_identical_typeerror_retry_is_gone() -> None:
    """Defect 2: the retry could not help (it re-ran the failing path) and
    it destroyed the evidence. A loader TypeError now propagates, and the
    torch_dtype question is answered by INSPECTION instead."""
    import inspect as _inspect

    from gen_worker.models import loading

    src = _inspect.getsource(loading.load_component)
    assert "except TypeError" not in src

    class _NoDtype:
        @staticmethod
        def from_pretrained(path: str) -> str:
            return path

    class _AnyKwargs:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> str:
            return path

    assert loading._accepts_kwarg(_NoDtype.from_pretrained, "torch_dtype") is False
    assert loading._accepts_kwarg(_AnyKwargs.from_pretrained, "torch_dtype") is True


# ---------------------------------------------------------------------------
# 3: a broken DIAGNOSTIC is not a broken release (defect 3)
# ---------------------------------------------------------------------------


def _handler() -> Any:
    from gen_worker.diagnostics import SwapLatencyDiagnostics

    obj = SwapLatencyDiagnostics()
    obj.setup(checkpoint="/nonexistent/tree", to="")
    return obj


def test_a_load_fatal_inside_the_benchmark_never_escapes_the_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact ie#546 fatal. Pre-fix it left the handler as a fast fatal,
    the hub counted it, and two of them marked the release's function broken
    — recycling a warm pod. It must come back as data."""
    from gen_worker.benchmarks import swap_latency as bench
    from gen_worker.diagnostics import SwapLatencyInput

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise TypeError(
            "NVIDIAModelOptConfig.__init__() missing 1 required positional "
            "argument: 'quant_type'")

    monkeypatch.setattr(bench, "run_cases", _boom)
    out = _handler().swap_latency(None, SwapLatencyInput())

    assert out.status == "failed"
    assert "quant_type" in out.error
    assert "NVIDIAModelOptConfig" in out.traceback_text
    assert out.rows == []


def test_off_pod_refusal_is_an_outcome_not_a_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker.diagnostics import SwapLatencyInput

    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "1")
    out = _handler().swap_latency(None, SwapLatencyInput())

    assert out.status == "refused"
    assert "weights-locality" in out.error


def test_a_bad_case_request_is_an_outcome_not_a_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker.diagnostics import SwapLatencyInput

    monkeypatch.delenv("GEN_WORKER_FORBID_CPU_OFFLOAD", raising=False)
    # 'swap' with no distinct `to` binding: a caller error, still a
    # measurement outcome — never a release-health signal.
    out = _handler().swap_latency(None, SwapLatencyInput(cases=("swap",)))
    assert out.status == "failed"
    assert "swap case" in out.error


def test_the_family_less_slot_convention_is_dispatchable() -> None:
    """The hub cannot put a curated policy on a family-less string slot, a
    fixed slot rejects supplied values, and msgspec cannot omit a required
    field — so an all-defaults payload MUST be valid or the diagnostics
    function is uninvokable (the pgw#689 ``checkpoint: ""`` workaround, now
    the declared contract)."""
    import msgspec

    from gen_worker.diagnostics import SwapLatencyInput

    payload = msgspec.json.decode(b"{}", type=SwapLatencyInput)
    assert payload.checkpoint == "" and payload.to == ""
    assert "stage" in payload.cases


# ---------------------------------------------------------------------------
# 4: the other final-cycle trap — a silent discovery skip
# ---------------------------------------------------------------------------


def _pkg(root: Path, name: str, body: str) -> None:
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(textwrap.dedent(body))


def test_reexported_endpoint_skip_is_loud_and_names_the_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from gen_worker.discovery.walk import find_endpoints

    monkeypatch.syspath_prepend(str(tmp_path))
    _pkg(tmp_path, "ep_pgw689_reexport", """
        from gen_worker.diagnostics import SwapLatencyDiagnostics  # noqa: F401
    """)

    with caplog.at_level(logging.WARNING, logger="gen_worker.discovery.walk"):
        found = find_endpoints(["ep_pgw689_reexport"])

    assert found == []
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "an endpoint dropped from the route table must be LOUD"
    text = warnings[0].getMessage()
    assert "SwapLatencyDiagnostics" in text
    assert "ep_pgw689_reexport" in text
    assert "SUBCLASS" in text


def test_subclassing_in_the_endpoint_package_actually_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fix the warning names has to work."""
    from gen_worker.discovery.walk import find_endpoints

    monkeypatch.syspath_prepend(str(tmp_path))
    _pkg(tmp_path, "ep_pgw689_subclass", """
        from gen_worker.diagnostics import SwapLatencyDiagnostics

        class SwapLatency(SwapLatencyDiagnostics):
            pass
    """)

    found = find_endpoints(["ep_pgw689_subclass"])
    assert [f.qualname for f in found] == ["SwapLatency"]
    assert found[0].module == "ep_pgw689_subclass"
