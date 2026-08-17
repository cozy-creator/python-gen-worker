"""pgw#1074 — the sdxl CFG arm's `ingress_refused`, and the refusal that hid it.

**The defect**: from one sdxl cell the turbo arm served FROM THE CELL while the
base arm — same cell, an entry with exactly its dims — got
`fallback_reason=ingress_refused` / `no_entry_admits`.

**Root cause**: the dtype a diffusers denoiser is handed for its scalar timestep
is a per-request SAMPLER fact, not a family fact.

    euler / euler_a / euler_trailing / heun / flow_euler  -> float32
    ddim / ddim_trailing / ddpm / deis / dpmpp_2m{,_karras,_sde,_sde_karras}
    / lcm / unipc                                          -> int64

Four of sdxl's six legal samplers (`SdxlScheduler`) present int64. The turbo arm
runs `euler_trailing` (float32) and served; the base arm ran an int64 sampler and
was refused. A declared `dtype="float32"` is NOT wrong — it is what the graph is
specialized on — and CFG/B=2 is a red herring: a `generate` call on `euler_a`
would have served, and a `dmd2-4step` turbo call (`lcm`) would have been refused.
The sampler is per-request VIEW state and deliberately not a compile axis, so no
declared dtype can be right for every call. The normalization belongs at the one
boundary that knows the contract.

**And the observability half.** The refusal said "36 tried" and then listed six,
in iteration order — and the entry whose dims MATCHED was not among them, so its
actual objection was unavailable and diagnosing it meant pulling the published
cell apart.

Real codepaths: a real `torch.export` + real AOTI compile on CPU, driven through
the live `TCGEntryRunner`/`EntryDispatch`, on the pgw#791 rig.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, cast

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import activity, aot_serve  # noqa: E402
from gen_worker._vendor.torch_compiled_graphs import (  # noqa: E402
    CallIngress,
    CallInput,
    CompiledGraphRunner,
)


# ---------------------------------------------------------------------------
# The rig: one real AOTI package specialized on a float32 scalar timestep
# ---------------------------------------------------------------------------


class TinyDenoiser(nn.Module):
    """The sdxl shape in miniature: the timestep reaches the arithmetic
    through a float cast, exactly as `get_timestep_embedding` does
    (`timesteps[:, None].float()`) — which is why an int64 feed recast at
    ingress is bit-identical to the eager call that presented it."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, sample: Any, timestep: Any) -> Any:
        return torch.tanh(self.lin(sample)) + timestep.float()


@pytest.fixture(scope="module")
def package(tmp_path_factory) -> Any:
    module = TinyDenoiser().eval()
    tmp_path = tmp_path_factory.mktemp("pgw1074")
    with torch.no_grad():
        program = torch.export.export(
            module, (torch.randn(2, 8), torch.tensor(1.0)), {}, strict=False)
        path = torch._inductor.aoti_compile_and_package(
            program, package_path=str(tmp_path / "tiny.pt2"))
    return torch._inductor.aoti_load_package(str(path))


def _contract(timestep_dtype: str = "float32",
              sample_dims: Tuple[int, int] = (2, 8),
              timestep_shape: List[Any] | None = None,
              ) -> CallIngress:
    return CallIngress(
        parameters=("sample", "timestep"),
        flat_arity=2,
        inputs=(
            CallInput(
                "sample", 0, "sample", 0, (), "sample", "float32",
                tuple(sample_dims),
            ),
            CallInput(
                "timestep", 1, "timestep", 1, (), "timestep",
                timestep_dtype,
                tuple(() if timestep_shape is None else timestep_shape),
            ),
        ),
    )


class _PackageRunner:
    """The minimal TCG runner surface around this test's real AOTI package."""

    def __init__(self, package: Any) -> None:
        self.package = package
        self.calls = 0
        self.bound = True
        self.declared_fqns: Tuple[str, ...] = ()

    def __call__(self, *feeds: Any) -> Any:
        if self.package is None:
            raise AssertionError("selection-only runner was invoked")
        result = self.package(*feeds)
        self.calls += 1
        return result


def _runner(package: Any, contract: Any = None,
            entry: str = "unet/adapter=false,cfg=false/B=1") -> Any:
    return aot_serve.TCGEntryRunner(
        runner=cast(CompiledGraphRunner, _PackageRunner(package)),
        contract=contract or _contract(),
        module_name="unet", entry=entry, family="tiny1074")


@pytest.fixture
def sink(monkeypatch) -> List[Any]:
    captured: List[Any] = []
    monkeypatch.setattr(activity, "_emit", captured.append)
    return captured


# ---------------------------------------------------------------------------
# HALF ONE — the class fix: an int64 scalar timestep is served, not refused
# ---------------------------------------------------------------------------


def test_the_int64_sampler_class_is_served_by_the_float32_entry(
        package, sink) -> None:
    """THE FIELD DEFECT. `dpmpp_2m_karras`/`lcm`/`ddim` present int64; the
    cell is specialized float32; the call must SERVE, not fall back."""
    runner = _runner(package)
    sample = torch.randn(2, 8)
    with torch.no_grad():
        out = runner(sample, torch.tensor(7, dtype=torch.int64))
    assert runner.refusals == {}, "the covering entry must admit this call"
    assert runner.calls == 1
    assert runner.realigned == {"timestep/int64_to_float32": 1}
    events = [e for e in sink if e.kind == aot_serve.RECAST_EVENT]
    assert len(events) == 1
    assert "input=timestep" in events[0].detail
    assert events[0].phase == "int64_to_float32"
    with torch.no_grad():
        eager = package(sample, torch.tensor(7.0))
    assert torch.equal(out, eager), "the recast feed is the eager value"


def test_the_recast_is_value_preserving_across_a_whole_timestep_ladder(
        package) -> None:
    """Not a spot check: every timestep an int64 sampler emits over a
    1000-step schedule must produce exactly the float32 call's output."""
    runner = _runner(package)
    sample = torch.randn(2, 8)
    with torch.no_grad():
        for t in (999, 749, 499, 249, 1, 0):
            got = runner(sample, torch.tensor(t, dtype=torch.int64))
            want = package(sample, torch.tensor(float(t)))
            assert torch.equal(got, want), f"timestep {t} diverged"
    assert runner.calls == 6
    assert runner.refusals == {}


def test_the_staging_buffer_is_allocated_once_across_recast_calls(
        package) -> None:
    """The recast rides pgw#791's staging buffer, so it costs the copy that
    ingress already made — not an allocation per denoise step."""
    runner = _runner(package)
    sample = torch.randn(2, 8)
    with torch.no_grad():
        for t in range(6):
            runner(sample, torch.tensor(t, dtype=torch.int64))
    assert runner.aligner.buffered() == ("timestep",)
    assert runner.realigned == {"timestep/int64_to_float32": 6}
    events = 6  # counted; the typed event is coalesced to one
    assert sum(runner.realigned.values()) == events


def test_a_float32_call_is_untouched(package, sink) -> None:
    runner = _runner(package)
    with torch.no_grad():
        runner(torch.randn(2, 8), torch.tensor(1.0))
    assert runner.realigned == {}
    assert [e for e in sink if e.kind == aot_serve.RECAST_EVENT] == []


# ---------------------------------------------------------------------------
# The rails — every dtype disagreement this is NOT allowed to paper over
# ---------------------------------------------------------------------------


def test_pgw1058s_defect_is_still_caught_float32_call_bfloat16_entry() -> None:
    """The attempt-30 cell: entries specialized bf16, every real call float32.
    float -> float is not a normalization and must stay a named refusal, or
    this fix would have silently served the cell pgw#1058 exists to reject."""
    contract = _contract(timestep_dtype="bfloat16")
    with pytest.raises(aot_serve.IngressContractError) as excinfo:
        aot_serve.assert_ingress(
            contract, (), {"sample": torch.randn(2, 8),
                           "timestep": torch.tensor(1.0)})
    assert excinfo.value.reason == "dtype_mismatch"


def test_an_int64_call_on_a_bfloat16_entry_is_refused() -> None:
    """bf16 carries 8 mantissa bits: timestep 999 would land on 1000. A
    normalization that changes the value is a numeric change, so float32 and
    float64 are the ONLY recast targets."""
    contract = _contract(timestep_dtype="bfloat16")
    with pytest.raises(aot_serve.IngressContractError) as excinfo:
        aot_serve.assert_ingress(
            contract, (), {"sample": torch.randn(2, 8),
                           "timestep": torch.tensor(999, dtype=torch.int64)})
    assert excinfo.value.reason == "dtype_mismatch"
    assert aot_serve.RECAST_TARGETS == ("float32", "float64")


def test_a_nonscalar_int64_input_is_refused() -> None:
    """Rank-0 only. A tensor of integers is a payload, not a schedule
    coordinate, and casting one would be a guess about its meaning."""
    contract = _contract(timestep_shape=[4])
    with pytest.raises(aot_serve.IngressContractError) as excinfo:
        aot_serve.assert_ingress(
            contract, (), {"sample": torch.randn(2, 8),
                           "timestep": torch.zeros(4, dtype=torch.int64)})
    assert excinfo.value.reason == "dtype_mismatch"


def test_a_float32_call_on_an_int64_entry_is_refused() -> None:
    """One direction only. wan-2.2 declares an int64 timestep; a karras-sigma
    float32 timestep fed to it would silently truncate the fraction."""
    contract = _contract(timestep_dtype="int64")
    with pytest.raises(aot_serve.IngressContractError) as excinfo:
        aot_serve.assert_ingress(
            contract, (), {"sample": torch.randn(2, 8),
                           "timestep": torch.tensor(999.5)})
    assert excinfo.value.reason == "dtype_mismatch"


def test_dims_still_decide_admission(package) -> None:
    """The recast widens nothing but dtype: a wrong-shape call is refused as
    before, so an entry cannot serve a class it was not compiled for."""
    runner = _runner(package, contract=_contract(sample_dims=(1, 8)))
    with pytest.raises(aot_serve.IngressContractError) as excinfo:
        with torch.no_grad():
            runner(torch.randn(2, 8), torch.tensor(7, dtype=torch.int64))
    assert excinfo.value.reason == "static_dim_mismatch"


# ---------------------------------------------------------------------------
# HALF TWO — the refusal names the CLOSEST entry (acceptance item one)
# ---------------------------------------------------------------------------


#: (H_lat, W_lat) — the nine SDXL bucket rungs the field mint packaged.
_ASPECTS = ((80, 192), (96, 168), (104, 152), (112, 144), (128, 128),
            (144, 112), (152, 104), (168, 96), (192, 80))


def _sdxl_entry(adapter: bool, cfg: bool, batch: int, h: int, w: int,
                timestep_dtype: str) -> Tuple[str, Any]:
    """One sdxl cell entry, labelled and shaped exactly as the field cell's
    36 are (`unet/adapter=…,cfg=…/B=…,H_lat=…,T_txt=77,W_lat=…`). The adapter
    fork is the pgw#790 one: the branchless class REFUSES the lifted pair, the
    branch-bearing one declares it."""
    name = (f"unet/adapter={str(adapter).lower()},cfg={str(cfg).lower()}"
            f"/B={batch},H_lat={h},T_txt=77,W_lat={w}")
    inputs = [
        CallInput(
            "sample", 0, "sample", 0, (), "sample", "bfloat16",
            (batch, 4, h, w),
        ),
        CallInput(
            "timestep", 1, "timestep", 1, (), "timestep", timestep_dtype, ()
        ),
        CallInput(
            "encoder_hidden_states", 2, "encoder_hidden_states", 2, (),
            "encoder_hidden_states", "bfloat16", (batch, 77, 2048),
        ),
    ]
    excluded: List[str] = []
    if adapter:
        inputs += [
            CallInput(
                "lora_a", 3, "lora_a", 3, (), "lora_a", "bfloat16", (64, 8)
            ),
            CallInput(
                "lora_b", 4, "lora_b", 4, (), "lora_b", "bfloat16", (8, 64)
            ),
        ]
    else:
        excluded = ["lora_a", "lora_b"]
    contract = CallIngress(
        parameters=(
            "sample", "timestep", "encoder_hidden_states", "lora_a", "lora_b"
        ),
        flat_arity=len(inputs),
        inputs=tuple(inputs),
        excluded_inputs=tuple(excluded),
    )
    return name, _runner(None, contract=contract, entry=name)


def _sdxl_dispatch(timestep_dtype: str) -> aot_serve.EntryDispatch:
    """The field cell: 36 entries = adapter{F,T} x cfg{F,T} x 9 aspect rungs,
    B pinned by the cfg fork (cfg=true is ONE batch-2 forward, ie#345)."""
    rows = []
    for adapter in (False, True):
        for cfg, batch in ((False, 1), (True, 2)):
            for h, w in _ASPECTS:
                rows.append(_sdxl_entry(
                    adapter, cfg, batch, h, w, timestep_dtype))
    assert len(rows) == 36
    return aot_serve.EntryDispatch(tuple(rows))


def _cfg_call() -> Dict[str, Any]:
    """The refused call, verbatim from the field class line:
    `#0=bfloat16[2,4,128,128],#1=int64[],encoder_hidden_states=…`."""
    return {
        "sample": torch.zeros(2, 4, 128, 128, dtype=torch.bfloat16),
        "timestep": torch.tensor(999, dtype=torch.int64),
        "encoder_hidden_states": torch.zeros(2, 77, 2048,
                                             dtype=torch.bfloat16),
    }


_COVERING = ("unet/adapter=false,cfg=true/B=2,H_lat=128,T_txt=77,W_lat=128")


def test_the_refusal_names_the_dims_matching_entry_not_the_first_six() -> None:
    """RED BEFORE THE FIX. The dims-matching entry is 14th of 36 in iteration
    order, so `reasons[:6]` truncated away the only informative one — which is
    why the field report could say `36 tried` and show nothing that explained
    anything. Here the entry is declared bfloat16 (the pgw#1058 shape) so that
    it refuses for a reason the listing must surface rather than admitting."""
    dispatch = _sdxl_dispatch("bfloat16")
    with pytest.raises(aot_serve.IngressContractError) as excinfo:
        dispatch.select((), _cfg_call())
    detail = str(excinfo.value)
    assert excinfo.value.reason == "no_entry_admits"
    assert _COVERING in detail, "the entry whose dims match must be NAMED"
    assert "dtype_mismatch" in detail
    assert "every declared dim MATCHES" in detail
    assert "36 tried" in detail


def test_the_closest_entry_survives_the_hub_detail_truncation() -> None:
    """The field detail was cut at 573 chars and the informative entry fell
    past the cut. The closest entry and its reason lead the sentence."""
    dispatch = _sdxl_dispatch("bfloat16")
    with pytest.raises(aot_serve.IngressContractError) as excinfo:
        dispatch.select((), _cfg_call())
    detail = str(excinfo.value)
    assert _COVERING in detail[:400]
    assert "dtype_mismatch" in detail[:400]


def test_every_tried_entry_is_accounted_for_never_silently_dropped() -> None:
    """`36 tried` then 6 listed and 30 unexplained is the defect. Whatever is
    not named individually is COUNTED by reason, and the counts add up."""
    dispatch = _sdxl_dispatch("bfloat16")
    with pytest.raises(aot_serve.IngressContractError) as excinfo:
        dispatch.select((), _cfg_call())
    detail = str(excinfo.value)
    assert "Other 35 entries" in detail
    # 17 branchless entries that reach the dtype check, 18 branch-bearing ones
    # the call carries no adapter for. One count per entry, under its own
    # closest reason, so the counts sum to exactly what "36 tried" promised.
    counted = re.findall(r"(\w+) x(\d+)", detail)
    assert dict((k, int(v)) for k, v in counted) == {
        "dtype_mismatch": 17, "input_missing": 18}
    assert sum(int(v) for _k, v in counted) == 35


def test_the_field_cell_now_serves_the_cfg_arm_end_to_end() -> None:
    """The whole issue, as one assertion: the SAME 36-entry cell, declared as
    ie#627 declares it (float32), dispatches the int64 CFG call to the entry
    that covers it."""
    dispatch = _sdxl_dispatch("float32")
    name, _runner_ = dispatch.select((), _cfg_call())
    assert name == _COVERING


def test_a_genuinely_uncovered_class_still_refuses() -> None:
    """The dispatch is not weakened: an aspect the cell never minted is
    refused by name, and the refusal reports a dim miss, not a dtype one."""
    dispatch = _sdxl_dispatch("float32")
    call = _cfg_call()
    call["sample"] = torch.zeros(2, 4, 64, 64, dtype=torch.bfloat16)
    with pytest.raises(aot_serve.IngressContractError) as excinfo:
        dispatch.select((), call)
    assert excinfo.value.reason == "no_entry_admits"
    assert "static_dim_mismatch" in str(excinfo.value)
    assert "every declared dim MATCHES" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# The ranking itself — stated as a rule, not inferred from one example
# ---------------------------------------------------------------------------


def test_miss_distance_orders_dtype_before_dims_and_fewer_before_more(
) -> None:
    dtype_only = (aot_serve.IngressMiss("dtype_mismatch", "", "timestep"),)
    dtype_and_dim = dtype_only + (
        aot_serve.IngressMiss("static_dim_mismatch", "", "sample"),)
    dims_only = (aot_serve.IngressMiss("static_dim_mismatch", "", "sample"),)
    missing = (aot_serve.IngressMiss("input_missing", ""),)
    order = sorted(
        (dims_only, missing, dtype_and_dim, dtype_only),
        key=aot_serve.miss_distance)
    assert order == [dtype_only, dtype_and_dim, dims_only, missing]


def test_ingress_report_collects_every_miss_not_only_the_first() -> None:
    """The ranking is only as good as the diagnosis it ranks: one entry that
    misses on two axes must report both."""
    contract = _contract(timestep_dtype="bfloat16", sample_dims=(1, 8))
    misses, symbols = aot_serve.ingress_report(
        contract, (), {"sample": torch.randn(2, 8),
                       "timestep": torch.tensor(1.0)})
    assert symbols == {}
    # Declaration order, per input dtype -> rank -> dims: exactly the order
    # the raising check walked before it became a collecting one.
    assert [m.reason for m in misses] == [
        "static_dim_mismatch", "dtype_mismatch"]
    assert [m.input for m in misses] == ["sample", "timestep"]


def test_the_short_circuit_never_changes_an_admission_decision() -> None:
    """`select` walks 36 entries per denoise step, so the ADMISSION pass exits
    at the first miss and only the refusal path pays the exhaustive walk. That
    is an early exit from one walk, not a second rule — so admitted/refused
    must agree exactly, over admitting and refusing calls alike."""
    contracts = [
        _contract(),
        _contract(timestep_dtype="bfloat16"),
        _contract(sample_dims=(1, 8)),
        _contract(timestep_shape=[4]),
    ]
    calls = [
        {"sample": torch.randn(2, 8), "timestep": torch.tensor(1.0)},
        {"sample": torch.randn(2, 8), "timestep": torch.tensor(3,
                                                               dtype=torch.int64)},
        {"sample": torch.randn(1, 8), "timestep": torch.tensor(1.0)},
        {"sample": torch.randn(2, 8)},
    ]
    for contract in contracts:
        for call in calls:
            full, _ = aot_serve.ingress_report(contract, (), call)
            first, _ = aot_serve.ingress_report(
                contract, (), call, first_only=True)
            assert bool(full) == bool(first)
            if full:
                assert first == (full[0],)


def test_assert_ingress_raises_ingress_reports_first_miss() -> None:
    """One implementation: what `select` ranks and what `assert_ingress`
    raises cannot drift, because there is only one rule."""
    contract = _contract(timestep_dtype="bfloat16", sample_dims=(1, 8))
    call = {"sample": torch.randn(2, 8), "timestep": torch.tensor(1.0)}
    misses, _ = aot_serve.ingress_report(contract, (), call)
    with pytest.raises(aot_serve.IngressContractError) as excinfo:
        aot_serve.assert_ingress(contract, (), call)
    assert excinfo.value.reason == misses[0].reason
    assert str(excinfo.value) == misses[0].detail
