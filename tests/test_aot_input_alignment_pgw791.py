"""pgw#791 — the AOT serve path must satisfy the artifact's ALIGNED-input
contract at ingress, once, instead of letting AOTInductor copy per call.

Measured on an RTX 4090 (WARM-INFERENCE-MATRIX §2c, production `w8a8-lora64`,
gw 0.76.8 / torch 2.13.0+cu130, real StableDiffusionXLPipeline, 28 steps): the
armed `.pt2` is 2.8% FASTER per forward than the equivalent dynamo compiled graph and
0.9% SLOWER per request, because its residual over 28x(per-forward) is 196 ms
against the dynamo compiled graph's 77 ms. The cause names itself 28+ times per request
on the worker's stderr — a surface hub-spawned pods do not expose:

    Input 1 was compiled as 16-bytes aligned, but it is not aligned at run
    time. Copying to an aligned tensor to guarantee correctness, but expect a
    performance hit. (function run_impl)

Input 1 is `timestep`: diffusers passes `timesteps[i]`, a scalar VIEW at an
odd element offset into the scheduler's timestep tensor.

Real codepaths throughout: a real `torch.export` + real AOTI compile on CPU,
called through the real `aot_serve.ArtifactRunner`. The red half runs the same
package through the same unaligned view WITHOUT the ingress pass and asserts
the runner itself reports the misalignment — so the fix is measured against
the defect, not against a description of it.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import activity, aot_serve  # noqa: E402


class TinyDenoiser(nn.Module):
    """One linear plus a timestep the graph actually consumes — the timestep
    must reach the arithmetic or export would drop it and there would be no
    unaligned input to realign."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, sample: Any, timestep: Any) -> Any:
        return torch.tanh(self.lin(sample)) + timestep


def _export_package(tmp_path: Path) -> Tuple[Any, Any]:
    module = TinyDenoiser().eval()
    sample = torch.randn(2, 8)
    timestep = torch.tensor(1.0)
    with torch.no_grad():
        program = torch.export.export(module, (sample, timestep), {}, strict=False)
        path = torch._inductor.aoti_compile_and_package(
            program, package_path=str(tmp_path / "tiny.pt2"))
    return module, torch._inductor.aoti_load_package(str(path))


def _contract() -> aot_serve.ArtifactContract:
    return aot_serve.contract_from_meta({
        "inputs": [
            {"name": "sample", "position": 0, "dtype": "float32",
             "shape": [2, 8]},
            {"name": "timestep", "position": 1, "dtype": "float32",
             "shape": []},
        ],
        "symbols": {},
        "constants": [],
    })


def _runner(package: Any) -> aot_serve.ArtifactRunner:
    runner = aot_serve.ArtifactRunner(
        package=package, contract=_contract(), constants=(),
        module_name="unet", entry="unet/adapter=false", family="tiny791")
    # Constants are baked in a locally compiled package (this test is about
    # ingress, not B1), so the bind proof has nothing to prove here.
    runner.bound = True
    return runner


def _unaligned_timestep() -> Any:
    """The diffusers shape: a scalar VIEW at an odd element offset.

    Verified, not assumed — an aligned view would make every assertion below
    vacuously true.
    """
    timesteps = torch.arange(16, dtype=torch.float32)
    for index in range(1, 16):
        view = timesteps[index]
        if int(view.data_ptr()) % aot_serve.AOTI_ALIGNMENT:
            return view
    pytest.skip("no unaligned element view available on this allocator")


@pytest.fixture(scope="module")
def package(tmp_path_factory) -> Any:
    return _export_package(tmp_path_factory.mktemp("pt2_791"))[1]


# ---------------------------------------------------------------------------
# The gap itself
# ---------------------------------------------------------------------------


def test_a_scalar_element_view_is_out_of_the_aligned_contract() -> None:
    view = _unaligned_timestep()
    assert aot_serve.alignment_gap(view) == "unaligned_16b"
    # A freshly allocated tensor — what every microbenchmark feeds, which is
    # exactly why the microbenchmarks never saw this.
    assert aot_serve.alignment_gap(torch.tensor(1.0)) == ""


def test_non_contiguous_inputs_are_out_of_contract_too() -> None:
    strided = torch.randn(4, 8)[:, ::2]
    assert aot_serve.alignment_gap(strided) == "non_contiguous"


def test_non_tensor_feeds_are_not_alignment_candidates() -> None:
    assert aot_serve.alignment_gap(None) == ""
    assert aot_serve.alignment_gap(False) == ""


# ---------------------------------------------------------------------------
# RED: what the pre-fix serve path did — the runner copies, and says so
# where nobody can read it
# ---------------------------------------------------------------------------


def test_the_realign_threshold_is_the_compilers_own_constant() -> None:
    """Not a knob, and not a guess: inductor emits the check at
    ``GPU_ALIGN_BYTES``. If torch ever moves it, this fails here rather than
    silently under-realigning on a pod."""
    from torch._inductor.utils import GPU_ALIGN_BYTES

    assert aot_serve.AOTI_ALIGNMENT == GPU_ALIGN_BYTES


def test_red_the_package_itself_reports_the_unaligned_input(package, capfd) -> None:
    """Calling the package with the unaligned view — i.e. what
    `marshal_positional` alone produced — makes AOTInductor's own run_impl
    clone it and warn, per call. This is the defect, reproduced.

    Only the GPU wrapper emits it: the check is generated in
    ``codegen/cpp_wrapper_gpu.py::codegen_inputs`` and has no CPU counterpart,
    so a CPU package cannot show the red half. The GPU red/green is the pod
    measurement recorded in the tracker; here it skips honestly rather than
    passing vacuously.
    """
    sample = torch.randn(2, 8)
    view = _unaligned_timestep()
    capfd.readouterr()
    with torch.no_grad():
        package(sample, view)
    err = capfd.readouterr().err
    if "not aligned at run time" not in err:
        pytest.skip(
            "CPU AOTI wrapper emits no alignment check (GPU wrapper only); "
            "the pointer contract is asserted directly below")
    assert "Copying to an aligned tensor" in err


def test_green_the_runner_hands_the_package_an_aligned_pointer(
        package, capfd) -> None:
    runner = _runner(package)
    sample = torch.randn(2, 8)
    view = _unaligned_timestep()
    capfd.readouterr()
    with torch.no_grad():
        out = runner(sample, view)
    err = capfd.readouterr().err
    assert "not aligned at run time" not in err
    # And the answer is unchanged — realignment is a copy, not a computation.
    with torch.no_grad():
        direct = package(sample, view.clone())
    assert torch.allclose(out, direct)


def test_the_marshalled_feed_is_realigned_in_place() -> None:
    """`aligned_feeds` is the stage that stands between the marshal and the
    package; it realigns by POSITION and reports by NAME."""
    contract = _contract()
    view = _unaligned_timestep()
    feeds = aot_serve.marshal_positional(
        contract, (torch.randn(2, 8), view), {})
    assert aot_serve.alignment_gap(feeds[1]) == "unaligned_16b"
    seen: List[Tuple[str, str, str]] = []
    out = aot_serve.aligned_feeds(
        contract, feeds, aot_serve.FeedAligner(),
        lambda name, reason, event: seen.append((name, reason, event)))
    assert seen == [
        ("timestep", "unaligned_16b", aot_serve.REALIGN_EVENT)]
    assert aot_serve.alignment_gap(out[1]) == ""
    assert torch.equal(out[1], view)
    assert out[0] is feeds[0]      # an in-contract input is never copied


# ---------------------------------------------------------------------------
# "Align ONCE": the buffer is allocated once, not per call
# ---------------------------------------------------------------------------


def test_the_aligned_buffer_is_allocated_once_and_is_pointer_stable(
        package) -> None:
    runner = _runner(package)
    sample = torch.randn(2, 8)
    pointers = set()
    with torch.no_grad():
        for _ in range(6):
            feeds = aot_serve.aligned_feeds(
                runner.contract,
                aot_serve.marshal_positional(
                    runner.contract, (sample, _unaligned_timestep()), {}),
                runner.aligner)
            pointers.add(int(feeds[1].data_ptr()))
    assert len(pointers) == 1, "the staging buffer must not be reallocated"
    assert runner.aligner.buffered() == ("timestep",)


def test_a_freshly_allocated_buffer_is_within_the_contract() -> None:
    aligner = aot_serve.FeedAligner()
    staged = aligner.staged("timestep", _unaligned_timestep())
    assert int(staged.data_ptr()) % aot_serve.AOTI_ALIGNMENT == 0


# ---------------------------------------------------------------------------
# The typed event — the half that makes the tax visible off-pod
# ---------------------------------------------------------------------------


@pytest.fixture
def sink(monkeypatch) -> List[Any]:
    """The typed wire, captured at the emit seam the worker->hub stream binds
    to (same convention as the pgw#733 adopt-event tests)."""
    captured: List[Any] = []
    monkeypatch.setattr(activity, "_emit", captured.append)
    return captured


def _realign_events(sink: List[Any]) -> List[Any]:
    return [e for e in sink if e.kind == aot_serve.REALIGN_EVENT]


def test_realignment_emits_one_typed_event_naming_the_input(
        package, sink) -> None:
    runner = _runner(package)
    sample = torch.randn(2, 8)
    with torch.no_grad():
        for _ in range(5):
            runner(sample, _unaligned_timestep())
    events = _realign_events(sink)
    assert len(events) == 1, "one event per (entry, input, reason), not per call"
    detail = events[0].detail
    assert "input=timestep" in detail
    assert "entry=unet/adapter=false" in detail
    assert "family=tiny791" in detail
    assert events[0].phase == "unaligned_16b"
    # Coalesced, but never lost: the count carries every occurrence.
    assert runner.realigned == {"timestep/unaligned_16b": 5}


def test_an_in_contract_call_emits_nothing(package, sink) -> None:
    runner = _runner(package)
    with torch.no_grad():
        runner(torch.randn(2, 8), torch.tensor(1.0))
    assert _realign_events(sink) == []
    assert runner.realigned == {}


def test_realignments_surface_on_the_wrapped_pipeline(package, sink) -> None:
    """The fleet-visible accessor: a pod paying the tax is distinguishable
    from one that is not — which stderr could never make it."""
    module = TinyDenoiser().eval()
    pipe = types.SimpleNamespace(unet=module)
    runner = _runner(package)
    dispatch = aot_serve.EntryDispatch((("unet/adapter=false", runner),))
    aot_serve.wrap_module(module, dispatch, {"family": "tiny791"}, target="unet")
    setattr(pipe, "_cozy_aot", {
        "meta": {}, "targets": {"unet": {
            "module": module, "attr": "forward",
            "state": getattr(module, "_cozy_aot")["state"]}}})
    with torch.no_grad():
        module(torch.randn(2, 8), _unaligned_timestep())
        module(torch.randn(2, 8), _unaligned_timestep())
    assert aot_serve.realigned_inputs(pipe) == {"timestep/unaligned_16b": 2}
    assert aot_serve.served_entry_calls(pipe) == {"unet/adapter=false": 2}


def test_the_realignment_counter_is_not_a_refusal(package, sink) -> None:
    """A realigned input is IN contract once realigned: the artifact stays
    armed and the request is served compiled, not eagerly."""
    runner = _runner(package)
    with torch.no_grad():
        runner(torch.randn(2, 8), _unaligned_timestep())
    assert runner.refusals == {}
    assert runner.calls == 1
