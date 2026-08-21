from __future__ import annotations

import logging
import pathlib

import pytest

from gen_worker.models import rung as rung_mod

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402


def test_the_rung_sits_where_the_CARD_put_it_not_where_the_issue_predicted() -> None:
    names = [r.name for r in rung_mod.LADDER]
    assert names.index("model_offload") < names.index("partial_stream")
    assert names.index("partial_stream") < names.index("group_offload")
    assert rung_mod.PARTIAL_STREAM.touches_host_ram
    assert rung_mod.run_mode_of("partial_stream") == rung_mod.RUN_OFFLOAD


def test_the_rungs_price_was_measured() -> None:
    """THE instrument that cannot be forgotten."""
    assert rung_mod.PARTIAL_STREAM.latency > 0.0, (
        "partial_stream still carries the unmeasured placeholder price"
    )
    assert rung_mod.MODEL_OFFLOAD.latency < rung_mod.PARTIAL_STREAM.latency
    assert rung_mod.PARTIAL_STREAM.latency < rung_mod.GROUP_OFFLOAD.latency
    prices = [r.latency for r in rung_mod.LADDER]
    assert prices == sorted(prices), "the ladder must stay monotonic down"


def test_adding_the_rung_did_not_move_the_wire_price_of_offload() -> None:
    """Three rungs now project onto `offload`, and a run-mode-only price describes the shallowest PROACTIVELY reachable one."""
    assert rung_mod.price(rung_mod.RUN_OFFLOAD) == rung_mod.MODEL_OFFLOAD.latency
    assert rung_mod.price(rung_mod.RUN_CPU) == rung_mod.CPU.latency
    assert rung_mod.price("sequential") == rung_mod.SEQUENTIAL.latency
    assert rung_mod.price("partial_stream") == rung_mod.PARTIAL_STREAM.latency


def test_descending_from_the_rung_continues_down_the_measured_order() -> None:
    assert rung_mod.descend("partial_stream") is rung_mod.GROUP_OFFLOAD


def test_the_reactive_descent_never_enters_an_admission_only_rung() -> None:
    """A load-time OOM has no lease in hand at the moment it fires, so a rung whose budget would have to be invented there must not be on its path."""
    assert rung_mod.descend(None) is rung_mod.MODEL_OFFLOAD
    assert rung_mod.descend("off") is rung_mod.MODEL_OFFLOAD
    assert rung_mod.descend("vae_only") is rung_mod.MODEL_OFFLOAD
    assert rung_mod.floor_of("partial_stream", "model_offload") == "partial_stream"
    assert rung_mod.floor_of("partial_stream", "") == "partial_stream"


def test_select_auto_mode_can_never_return_it() -> None:
    """A proactive decider has no lease to read."""
    from gen_worker.models.memory import select_auto_mode

    class Empty:
        components: dict = {}

    seen = set()
    for avail in (0.0, 0.5, 4.0, 6.0, 8.0, 12.0, 24.0, 80.0):
        for size in (None, 0.0, 1.0, 7.0, 20.0, 60.0):
            for peak in (None, 0.0, 3.0, 40.0):
                for total in (None, 8.0, 24.0, 80.0):
                    seen.add(
                        select_auto_mode(
                            pipeline=Empty(),
                            available_vram_gb=avail,
                            model_size_gb=size,
                            peak_vram_gb=peak,
                            total_vram_gb=total,
                        )
                    )
    assert "partial_stream" not in seen
    assert seen, "the sweep must actually exercise the decider"


def test_the_rung_refuses_without_a_lease_budget() -> None:
    """Not a smaller version of the rung — a refusal."""
    from gen_worker.models.memory import apply_low_vram_config

    class Empty:
        components: dict = {}

    with pytest.raises(ValueError, match="RESIDENCY LEASE"):
        apply_low_vram_config(Empty(), mode="partial_stream")
    with pytest.raises(ValueError, match="RESIDENCY LEASE"):
        apply_low_vram_config(Empty(), mode="partial_stream", stream_budget_bytes=0)


def test_partial_stream_is_a_valid_mode_and_an_unknown_one_still_is_not() -> None:
    from gen_worker.models.memory import apply_low_vram_config

    class Empty:
        components: dict = {}

    with pytest.raises(ValueError, match="invalid low-VRAM mode"):
        apply_low_vram_config(Empty(), mode="stream")


class Pipe:
    """A diffusers-shaped pipeline: `components` is the vocabulary every rung in this module reads."""

    def __init__(self) -> None:
        torch.manual_seed(1497)
        self.unet = nn.Sequential(
            *[m for _ in range(8) for m in (nn.Linear(1024, 1024), nn.ReLU())]
        ).eval()
        self.vae = nn.Linear(1024, 1024).eval()

    @property
    def components(self) -> dict:
        return {"unet": self.unet, "vae": self.vae}


def test_arming_the_rung_splits_a_real_pipeline_at_the_budget() -> None:
    from gen_worker.models.memory import _apply_partial_stream, stream_residency_of

    pipe = Pipe()
    x = torch.randn(2, 1024)
    with torch.no_grad():
        want = pipe.unet(x).clone()

    applied: dict = {}
    total = sum(
        p.numel() * p.element_size()
        for m in pipe.components.values()
        for p in m.parameters()
    )
    ok = _apply_partial_stream(
        pipe,
        applied,
        budget_bytes=total // 2,
        log=logging.getLogger(__name__),
        device="cpu",
    )
    assert ok
    assert applied["partial_stream"] is True
    assert applied["stream_streamed_leaves"] >= 1
    assert applied["stream_window_bytes"] > 0
    assert applied["stream_resident_bytes"] + applied["stream_window_bytes"] <= (
        applied["stream_budget_bytes"]
    ), "the armed plan must fit the budget it was given, window included"

    residency = stream_residency_of(pipe)
    assert residency is not None and residency.plan is not None
    with torch.no_grad():
        assert torch.equal(pipe.unet(x), want), "the armed rung changed the answer"


def test_a_budget_that_holds_everything_arms_but_reports_no_degradation() -> None:
    """The rung armed and had nothing to do."""
    from gen_worker.models.memory import _apply_partial_stream

    pipe = Pipe()
    applied: dict = {}
    assert _apply_partial_stream(
        pipe,
        applied,
        budget_bytes=1 << 30,
        log=logging.getLogger(__name__),
        device="cpu",
    )
    assert applied["stream_streamed_leaves"] == 0
    assert applied["stream_streamed_bytes"] == 0
    assert applied["stream_window_bytes"] == 0


def test_the_unhookable_union_is_never_handed_to_the_ring() -> None:
    """The same union every other rung excludes: a dtype-fragile VAE and a content-shared module."""
    from gen_worker.models.memory import (
        SHARED_COMPONENT_ATTR,
        _apply_partial_stream,
        stream_residency_of,
        unhookable_components,
    )

    pipe = Pipe()
    setattr(pipe.vae, SHARED_COMPONENT_ATTR, True)
    assert "vae" in unhookable_components(pipe)

    applied: dict = {}
    assert _apply_partial_stream(
        pipe, applied, budget_bytes=0, log=logging.getLogger(__name__), device="cpu"
    )
    residency = stream_residency_of(pipe)
    assert residency is not None
    assert not any(name.startswith("vae") for name in residency.plan.streamed)
    assert any(name.startswith("unet") for name in residency.plan.streamed)


def test_the_manager_answers_the_budget_from_the_same_sizer_admission_uses() -> None:
    from gen_worker.serving.residency import ResidencyManager

    class Sizer:
        def resident_bytes(self, checkpoint_ref: str, lane: str) -> int:
            return 7 * (1 << 20) if checkpoint_ref == "known" else 0

        def activation_headroom_bytes(self, checkpoint_ref: str, lane: str) -> int:
            return 1 << 20

    manager = ResidencyManager(1 << 30, Sizer())
    assert manager.weight_budget_bytes("known", "lane") == 7 * (1 << 20)
    assert manager.weight_budget_bytes("unknown", "lane") == 0


def test_the_load_context_carries_the_budget_down_to_the_ladder() -> None:
    from gen_worker.serving.context import DeployBinding, LoadContext

    binding = DeployBinding(checkpoint_ref="ref", checkpoint_dir=pathlib.Path("/tmp"))
    assert LoadContext(binding=binding)._weight_budget_bytes == 0
    assert (
        LoadContext(binding=binding, weight_budget_bytes=1234)._weight_budget_bytes
        == 1234
    )
    assert (
        LoadContext(binding=binding, weight_budget_bytes=-1)._weight_budget_bytes == 0
    )


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cross-device placement needs a card"
)


class VocabPipe(Pipe):
    """A pipeline whose component vocabulary holds NON-MODULES, exactly as a real one does: sd1.5 answers `['vae','text_encoder','tokenizer','unet', 'scheduler','safety_checker','feature_extractor','image..."""

    class _Tokenizer:
        pass

    @property
    def components(self) -> dict:
        return {
            "unet": self.unet,
            "vae": self.vae,
            "tokenizer": self._Tokenizer(),
            "scheduler": object(),
        }


def test_a_non_module_in_the_component_vocabulary_does_not_unarm_the_rung() -> None:
    """MEASURED on the card: without the filter this raised `CLIPTokenizer has no attribute named_modules`, the rung reported False, and the pipeline served on `model_offload` — a placement nobody asked f..."""
    from gen_worker.models.memory import _apply_partial_stream, stream_residency_of

    pipe = VocabPipe()
    applied: dict = {}
    assert _apply_partial_stream(
        pipe, applied, budget_bytes=0, log=logging.getLogger(__name__), device="cpu"
    ), "a tokenizer in the vocabulary unarmed the whole rung"
    assert applied["stream_streamed_leaves"] >= 1
    residency = stream_residency_of(pipe)
    assert residency is not None
    assert not any(
        n.startswith(("tokenizer", "scheduler")) for n in residency.plan.streamed
    )


def test_an_unarmed_rung_confesses_on_the_typed_channel_not_just_the_log() -> None:
    """A fall-through to a coarser rung is a placement the operator asked for and did not get."""
    from gen_worker import activity as activity_mod
    from gen_worker.models.memory import (
        PARTIAL_STREAM_UNARMED_PHASE,
        _apply_partial_stream,
    )

    seen: list = []
    original = activity_mod.emit_event
    activity_mod.emit_event = lambda *a, **k: seen.append((a, k))  # type: ignore[assignment]
    try:

        class NoTree:
            components: dict = {}

        assert not _apply_partial_stream(
            NoTree(), {}, budget_bytes=1 << 20,
            log=logging.getLogger(__name__), device="cpu",
        )
    finally:
        activity_mod.emit_event = original  # type: ignore[assignment]

    phases = [k.get("phase") for _a, k in seen]
    assert PARTIAL_STREAM_UNARMED_PHASE in phases, (
        "the rung fell through silently — a log line is not a confession"
    )
    detail = " ".join(str(k.get("detail", "")) for _a, k in seen)
    assert "COARSER" in detail


@CUDA
def test_tensors_owned_by_a_module_WITH_children_are_placed_on_the_device() -> None:
    """MEASURED: CLIP's `position_ids` hangs off `CLIPTextEmbeddings`, which has children, so it is never a leaf and nothing else in this rung touched it."""
    from gen_worker.models.stream_residency import StreamedResidency, module_roots

    class WithChildren(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("position_ids", torch.arange(77).unsqueeze(0))
            self.child = nn.Linear(2048, 2048)

    model = WithChildren().eval()
    assert model.position_ids.device.type == "cpu"
    residency = StreamedResidency(
        module_roots(model), device="cuda", budget_bytes=0, min_stream_bytes=1
    )
    plan = residency.engage()
    assert any(n.endswith("child") for n in plan.streamed)
    assert model.position_ids.device.type == "cuda", (
        "a parent-owned tensor was left on the host"
    )
    residency.release()


@CUDA
def test_an_excluded_component_is_kept_ON_the_device_not_left_behind() -> None:
    """An exclusion is a statement about HOOKS, never about residency."""
    from gen_worker.models.memory import (
        SHARED_COMPONENT_ATTR,
        _apply_partial_stream,
        unhookable_components,
    )

    pipe = Pipe()
    setattr(pipe.vae, SHARED_COMPONENT_ATTR, True)
    assert "vae" in unhookable_components(pipe)
    assert next(pipe.vae.parameters()).device.type == "cpu"

    assert _apply_partial_stream(
        pipe, {}, budget_bytes=0, log=logging.getLogger(__name__), device="cuda"
    )
    assert next(pipe.vae.parameters()).device.type == "cuda", (
        "the excluded component was kept out of the ring AND off the card"
    )


@CUDA
def test_the_pipeline_still_answers_with_its_execution_device() -> None:
    """`DiffusionPipeline.device` answers with the first parameter it finds, so a parked tail makes it report the HOST — and the pipeline then builds its input ids there, which is how sd1.5 died at a 5% b..."""
    from diffusers import StableDiffusionPipeline

    from gen_worker.models.memory import (
        STREAM_RESIDENCY_ATTR,
        install_execution_device_fallback,
    )
    from gen_worker.models.stream_residency import StreamedResidency

    snapshots = sorted(
        pathlib.Path.home().glob(
            ".cache/huggingface/hub/models--hf-internal-testing--"
            "tiny-stable-diffusion-pipe/snapshots/*"
        )
    )
    if not snapshots:
        pytest.skip("the tiny diffusers fixture is not cached on this host")
    pipe = StableDiffusionPipeline.from_pretrained(
        str(snapshots[-1]), safety_checker=None, requires_safety_checker=False,
        local_files_only=True,
    )
    roots = [
        (n, m) for n, m in pipe.components.items() if hasattr(m, "named_modules")
    ]
    residency = StreamedResidency(
        roots, device="cuda", budget_bytes=0, min_stream_bytes=1
    )
    residency.engage()
    assert residency.plan is not None and residency.plan.streamed
    assert pipe.device.type == "cpu", (
        "with no repair installed the parked tail must still be visible as the "
        "host — otherwise this test proves nothing"
    )

    setattr(pipe, STREAM_RESIDENCY_ATTR, residency)
    install_execution_device_fallback()
    assert pipe.device.type == "cuda", "pipeline.device still lies about the host"
    assert pipe._execution_device.type == "cuda"
    residency.release()


def test_the_budget_is_a_pair_and_the_ram_half_is_reported_not_enforced() -> None:
    from gen_worker.models.stream_residency import (
        LeafCost,
        MemoryBudget,
        plan_residency,
    )

    costs = [LeafCost(f"m{i}", 4 << 20) for i in range(6)]
    bare = plan_residency(costs, budget_bytes=8 << 20, min_stream_bytes=1)
    assert bare.ram_budget_bytes == 0
    assert bare.host_fits, "an UNSTATED ram half is not a failed one"

    tight = plan_residency(
        costs,
        budget_bytes=MemoryBudget(vram_bytes=8 << 20, ram_bytes=1 << 20),
        min_stream_bytes=1,
    )
    assert tight.resident == bare.resident, (
        "the ram half must not silently change the VRAM split — enforcement is "
        "deferred, and a half-done enforcement is worse than a named one"
    )
    assert tight.host_bytes == tight.streamed_bytes > (1 << 20)
    assert not tight.host_fits, "an over-budget tail must be VISIBLE"


def test_a_rebudget_does_not_revoke_the_assigned_ram_half() -> None:
    from gen_worker.models.stream_residency import (
        MemoryBudget,
        StreamedResidency,
        module_roots,
    )

    model = Pipe().unet
    residency = StreamedResidency(
        module_roots(model),
        device="cpu",
        budget_bytes=MemoryBudget(vram_bytes=1 << 30, ram_bytes=7 << 20),
        min_stream_bytes=1,
    )
    residency.engage()
    plan = residency.rebudget(0)
    assert plan.ram_budget_bytes == 7 << 20
