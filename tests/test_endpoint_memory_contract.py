from __future__ import annotations

import torch

from gen_worker import hostfacts
from gen_worker.models import memory, residency


class _EmulatedFp8(torch.Tensor):

    @staticmethod
    def __new__(cls, inner: torch.Tensor) -> "_EmulatedFp8":
        out = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, inner.shape, dtype=torch.bfloat16, device=inner.device,
        )
        out._inner = inner  # type: ignore[attr-defined]
        return out

    def __tensor_flatten__(self) -> tuple[list[str], dict]:  # noqa: D105
        return ["_inner"], {}

    @staticmethod
    def __tensor_unflatten__(inner: dict, meta: dict, sizes: object, strides: object) -> "_EmulatedFp8":  # noqa: D105
        return _EmulatedFp8(inner["_inner"])

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]  # noqa: D105
        cls, func: object, types: object, args: tuple = (), kwargs: dict | None = None,
    ) -> object:
        if func in (
            torch.ops.aten.detach.default,
            torch.ops.aten.alias.default,
        ):
            return _EmulatedFp8(func(args[0]._inner))
        if func is torch.ops.aten._to_copy.default:
            kwargs = dict(kwargs or {})
            kwargs.pop("dtype", None)
            return _EmulatedFp8(func(args[0]._inner, **kwargs))
        raise NotImplementedError(f"{func} on a census-only fixture")


def test_tensor_storage_bytes_prices_the_storage_not_the_logical_dtype() -> None:
    inner = torch.zeros(64, 32, dtype=torch.uint8)
    quantized = _EmulatedFp8(inner)

    assert quantized.dtype is torch.bfloat16
    assert quantized.numel() * quantized.element_size() == 64 * 32 * 2

    assert memory.tensor_storage_bytes(quantized) == 64 * 32 * 1
    assert memory.tensor_storage_bytes(inner) == 64 * 32 * 1
    label = memory.tensor_dtype_label(quantized)
    assert "_EmulatedFp8" in label and "uint8" in label
    assert memory.tensor_dtype_label(inner) == "torch.uint8"


def test_size_estimates_follow_the_storage() -> None:
    """The census the offload ladder reads must not book an fp8 component at the bf16 checkpoint's bytes."""

    class Quantized(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                _EmulatedFp8(torch.zeros(1024, 1024, dtype=torch.uint8)),
                requires_grad=False,
            )

    class Holder:
        def __init__(self) -> None:
            self.transformer = Quantized()

    gb = memory.estimate_pipeline_size_gb(Holder())
    assert abs(gb - 1024 * 1024 / float(1024**3)) < 1e-9


def test_resident_census_is_empty_off_gpu() -> None:
    class Holder:
        def __init__(self) -> None:
            self.transformer = torch.nn.Linear(8, 8)

    assert memory.resident_census(Holder()) == []


class _StuckModule(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
        self.register_buffer("bias", torch.zeros(4))

    def _apply(self, *args, **kwargs):  # noqa: ANN002, ANN003, D102
        return self


def _devices(module: torch.nn.Module) -> set[str]:
    return {t.device.type for t in list(module.parameters()) + list(module.buffers())}


def test_repair_escalates_past_a_no_op_to() -> None:
    class Holder:
        def __init__(self) -> None:
            self.transformer = _StuckModule()

    holder = Holder()
    holder.transformer.to("meta")
    assert _devices(holder.transformer) == {"cpu"}
    assert memory.device_mismatches(holder, "meta")

    assert memory.repair_device_placement(holder, "meta") == []
    assert _devices(holder.transformer) == {"meta"}


def test_repair_still_moves_a_cooperative_component() -> None:
    class Holder:
        def __init__(self) -> None:
            self.transformer = torch.nn.Linear(4, 4)

    holder = Holder()
    assert memory.repair_device_placement(holder, "meta") == []
    assert _devices(holder.transformer) == {"meta"}


def test_move_verified_is_a_free_function_and_proves_the_landing() -> None:
    module = torch.nn.Linear(4, 4)
    assert residency.move_verified(module, "meta", label="transformer") is True
    assert _devices(module) == {"meta"}


def test_move_verified_survives_a_component_that_ignores_to() -> None:
    module = _StuckModule()
    assert residency.move_verified(module, "meta", label="text_encoder") is True
    assert _devices(module) == {"meta"}


def test_registry_move_delegates_to_the_free_function() -> None:
    """`Residency` keeps its bookkeeping; the mechanism is not a second copy."""
    seen: list[tuple[str, str]] = []

    def mover(obj: object, device: str) -> None:
        seen.append((type(obj).__name__, device))
        obj.to(device)  # type: ignore[attr-defined]

    registry = residency.Residency(move_fn=mover)
    module = torch.nn.Linear(4, 4)
    assert registry._move_verified(module, "meta", ref="transformer") is True
    assert seen == [("Linear", "meta")]


def test_process_ceiling_is_a_named_formula() -> None:
    ceiling = hostfacts.process_ceiling_bytes()
    if hostfacts.cuda_ready():
        assert ceiling is None or ceiling >= 0
    else:
        assert ceiling is None


def test_process_ceiling_vram_names_its_zero_cause() -> None:
    reading = memory.process_ceiling_vram()
    if hostfacts.cuda_ready():
        return
    assert reading.gb == 0.0
    assert reading.measured is False
    assert reading.reason == memory.VRAM_NO_CUDA
    assert reading.reason != memory.VRAM_UNREADABLE


def test_release_cached_vram_does_not_reset_the_peak() -> None:
    """The stage-boundary flush an activation measurement can survive."""
    import inspect

    source = inspect.getsource(memory.release_cached_vram)
    assert "empty_cache" in source
    assert "reset_peak_memory_stats" not in source
    assert "gc.collect" not in source
    memory.release_cached_vram()


def test_module_storage_bytes_asks_the_module_itself() -> None:
    """`estimate_pipeline_size_gb` enumerates the COMPONENTS OF what it is handed, so a bare denoiser loses every parameter held on its root."""

    class Denoiser(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pos_embed = torch.nn.Parameter(
                torch.zeros(1000, dtype=torch.float32), requires_grad=False
            )
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(10, 10, bias=False)])

    module = Denoiser()
    expected = 1000 * 4 + 10 * 10 * 4
    assert memory.module_storage_bytes(module) == expected
    assert memory.estimate_pipeline_size_gb(module) * float(1024**3) < expected


def test_module_storage_bytes_counts_a_shared_storage_once() -> None:
    shared = torch.zeros(256, dtype=torch.float32)

    class Twin(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Parameter(shared, requires_grad=False)
            self.b = torch.nn.Parameter(shared, requires_grad=False)

    assert memory.module_storage_bytes(Twin()) == 256 * 4


def test_module_storage_bytes_is_zero_for_a_non_module() -> None:
    assert memory.module_storage_bytes(None) == 0
    assert memory.module_storage_bytes(object()) == 0
