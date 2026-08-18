"""th#1871 P1's worker half: the posture is TYPED and it rides `JobMetrics`.

The end-to-end assertions here run over the REAL gRPC terminal path — the same
`hub_double` a request actually takes — because the property being proven is
that the hub RECEIVES this, not that a Python object exists. A unit test of the
ledger would have passed on every day of the window this issue describes: the
posture was being computed the whole time and reaching nobody.

THE DEFECT, restated so a future reader knows what a red here means (ie#707,
DESIGN-RULINGS §1.36): a family served on `sdpa` because flash-attn was absent
from its image while its lane declared flash, and every number taken from those
runs was filed as a measurement OF the flash lane. tensorhub now keys
`endpoint_measurements` on the posture digest, so with this on the wire those
are two comparable rows instead of one silent overwrite.
"""

from __future__ import annotations

import msgspec
import pytest

from gen_worker import measured_posture as mp
from gen_worker.models import attention_modes
from gen_worker.models import provision
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.hub_double import hub_double, is_ready, is_result_for
from harness.posture_endpoints_pgw1225 import PostureIn

_MODULES = ("harness.posture_endpoints_pgw1225",
            "harness.posture_clean_endpoint_pgw1225",
            "harness.posture_required_compile_pgw1225")


def _run(
    request_id: str, function_name: str, *, lane: str = "",
) -> "pb.JobResult":
    with hub_double(modules=_MODULES) as (scheduler, _h):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id=request_id, attempt=1, function_name=function_name,
            lane=lane,
            input_payload=msgspec.msgpack.encode(PostureIn(prompt="x"))))
        return conn.wait_for(is_result_for(request_id)).job_result


# ---------------------------------------------------------------------------
# The wire
# ---------------------------------------------------------------------------

def test_the_silent_attention_fallback_reaches_the_hub_typed() -> None:
    """THE red test. On the pre-fix tree this posture does not exist on the
    wire at all — `JobMetrics` had no field for it, and the fact lived only in
    `worker_activity_events.detail` as prose the reducer refuses to parse."""
    res = _run("r-1225-fallback", "render")
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    posture = res.metrics.posture

    assert posture.attention_backend == "sdpa"
    # The half that makes it a FINDING rather than a reading: what was asked
    # for. Equal values would be a clean measurement of the sdpa lane; these
    # are evidence the pair is mis-ranked.
    assert posture.attention_backend_wanted == "fa2"
    assert [t.name for t in posture.applied] == ["attention_fallback"]
    # And the lane is carried with it, so the record is self-contained.
    assert posture.execution_lane == res.metrics.lane


def test_a_clean_run_reports_a_clean_posture_not_a_silent_one() -> None:
    """`wanted == applied` is a DELIBERATE configuration and a clean
    measurement. It must not read as the fallback above, and it must not read
    as unreported either — three distinct things."""
    res = _run("r-1225-clean", "render-clean")
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    posture = res.metrics.posture
    assert posture.attention_backend == "sdpa"
    assert posture.attention_backend_wanted == "sdpa"
    assert list(posture.applied) == []


def test_the_posture_is_the_same_reading_as_the_lane_and_serving_mode() -> None:
    """ie#655's rule, extended to the posture: one axis, one reading.

    Not "they agree" — they are stamped from the same `ServedIdentity` at the
    same instant, so they CANNOT disagree. A test that only compared them would
    pass on the day two derivations happened to coincide.
    """
    res = _run("r-1225-axis", "render")
    m = res.metrics
    assert m.posture.execution_lane == m.lane
    expected = "eager" if m.serving_mode == "eager" else "compiled"
    assert m.posture.compile_state == expected, (
        f"posture says compile_state={m.posture.compile_state!r} while "
        f"serving_mode={m.serving_mode!r} — a second reading of the execution "
        f"axis is exactly what ie#655 deleted")


def test_the_component_posture_carries_applied_versus_bound() -> None:
    """`applied=fp8-w8a8-dynamic bound=bf16-w16a16` was prose on a channel with
    no consumer. It is the per-component fact that decides whether two numbers
    describe the same thing."""
    res = _run("r-1225-components", "render")
    components = {c.component: c for c in res.metrics.posture.components}
    assert "transformer" in components, list(components)
    assert components["transformer"].applied_quant == "fp8-w8a8-dynamic"
    assert components["transformer"].bound_quant


def test_a_declared_compiled_lane_that_serves_eager_says_so() -> None:
    """minimax-h3's specimen: `wanted=compiled, applied=eager`.

    The hub instructs `+compiled`; this worker has no compiled graph and serves eager. The
    posture must carry BOTH values — a record holding only the resolved one
    makes the more urgent fact unrepresentable at the moment it matters most,
    which is how hours of eager serving at the shape that serves most went
    unnoticed. It is also the ONE axis where the hub can corroborate the worker
    from its own durable column (`served_eager_fallback`), so a disagreement
    here is detectable rather than merely wrong.
    """
    res = _run("r-1225-declared", "render-declared",
               lane="fp8-w8a8-dynamic+compiled")
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    posture = res.metrics.posture
    assert posture.compile_state_wanted == "compiled"
    assert posture.compile_state == "eager"
    assert res.metrics.serving_mode == "eager"


def test_an_uninstructed_lane_claims_nothing_on_the_compile_axis() -> None:
    """No instruction and no `required_compile` fence: `wanted` is EMPTY.

    Not "eager". An unstated expectation is not a satisfied one — a mismatch
    needs a stated expectation to violate, and inventing one here would report
    every policy-dispatched request as a clean compiled-vs-eager agreement.
    """
    res = _run("r-1225-uninstructed", "render-declared")
    assert res.metrics.posture.compile_state_wanted == ""
    assert res.metrics.posture.compile_state == "eager"


# ---------------------------------------------------------------------------
# The public reporter (th#1871 §6.6 item 4)
# ---------------------------------------------------------------------------

def test_the_backend_reporter_accepts_the_value_that_started_all_this() -> None:
    """`sdpa` — the token `report_applied_attention` raises on, which is why
    the fleet reported nothing. It is a different AXIS, not a stricter grammar,
    so it gets a reporter rather than a widened vocabulary."""
    with provision.AppliedAttentionScope() as scope:
        assert provision.report_attention_backend("transformer", "sdpa")
    assert [e.backend for e in scope.applied] == ["sdpa"]

    with pytest.raises(ValueError):
        # Still refused on the SPARSITY reporter, and that is correct.
        provision.report_applied_attention("transformer", "sdpa")


@pytest.mark.parametrize("spelling,canonical", [
    ("flash_attention_2", "fa2"),
    ("flash-attn", "fa2"),
    ("FlashAttention_3", "fa3"),
    ("torch_sdpa", "sdpa"),
    ("xformers", "xformers"),
    ("math", "eager"),
])
def test_ecosystem_spellings_are_normalized_not_refused(
    spelling: str, canonical: str,
) -> None:
    """An author reporting `flash_attention_2` is being precise. Refusing them
    is how the previous reporter ended up with no callers."""
    assert mp.normalize_backend(spelling) == canonical


def test_an_unknown_backend_is_refused_at_the_reporter() -> None:
    """A value tensorhub does not share is not an unknown backend — it is a
    FOURTH vocabulary, and th#1871 §1.3 measured what those cost: three writers
    keying one relation three ways, so no two sources could ever join."""
    with pytest.raises(ValueError, match="not one of"):
        mp.normalize_backend("triton_flash_v9")


def test_a_backend_only_report_does_not_claim_dense_sparsity() -> None:
    """The two axes share one record and one scope. They do not share a
    default: `attention_mode` reports "" (unreported) when only a backend was
    reported, and "" is what the hub reads as unknown."""
    entry = attention_modes.AppliedAttention(component="transformer", backend="sdpa")
    assert entry.mode == ""
    assert "backend=sdpa" in entry.detail()


# ---------------------------------------------------------------------------
# The ledger's rules
# ---------------------------------------------------------------------------

def test_levers_keep_the_order_they_were_reached_for() -> None:
    """ORDER IS MEANING: group-offload AFTER model-offload failed is not
    group-offload instead of it, and the hub does not sort this list."""
    ledger = mp.PostureLedger()
    ledger.technique(mp.TECHNIQUE_MODEL_OFFLOAD, component="transformer")
    ledger.technique(mp.TECHNIQUE_GROUP_OFFLOAD, component="transformer")
    posture = ledger.snapshot()
    assert [t.name for t in posture.applied] == [
        mp.TECHNIQUE_MODEL_OFFLOAD, mp.TECHNIQUE_GROUP_OFFLOAD]


def test_a_repeated_lever_is_not_a_deeper_descent() -> None:
    """The placement path is re-entered on retries. A lever recorded twice
    reads as a descent that went one rung further than it did."""
    ledger = mp.PostureLedger()
    for _ in range(3):
        ledger.technique(mp.TECHNIQUE_GROUP_OFFLOAD, component="transformer")
    assert len(ledger.snapshot().applied) == 1


def test_components_are_not_pre_sorted_by_the_producer() -> None:
    """The hub sorts before digesting. A producer that pre-sorts HIDES a
    producer that does not — and the next producer is the one that breaks it."""
    ledger = mp.PostureLedger()
    ledger.component("vae", applied_quant="bf16-w16a16")
    ledger.component("transformer", applied_quant="fp8-w8a8-dynamic")
    order = [c.component for c in ledger.snapshot().components]
    assert order == ["vae", "transformer"], (
        "the ledger sorted its components; the hub's sort is then unexercised "
        "by every test that goes through this path")


def test_the_deepest_shortfall_wins() -> None:
    """A descent short by 2 GiB and then, one rung down, by 9 is described by
    the 9. The magnitudes are data — the hub excludes them from the key — but
    the wrong one is still the wrong diagnostic."""
    ledger = mp.PostureLedger()
    ledger.shortfall(mp.ResourceShortfall.from_gb("vram", 26.0, 24.0))
    ledger.shortfall(mp.ResourceShortfall.from_gb("vram", 33.0, 24.0))
    short = ledger.snapshot().shortfall
    assert short is not None
    assert short.short_by_bytes == 9 * (1 << 30)


def test_a_resident_placement_is_not_a_degradation_and_vae_only_is_resident() -> None:
    """pgw#750: `off` and `vae_only` are BOTH fully resident — the refinement
    only toggles VAE slicing. Reading `vae_only` as an offload rung would file
    every sliced decode as a moved-weights measurement."""
    assert mp.residency_for_placement("off") == mp.RESIDENCY_ALL_RESIDENT
    assert mp.residency_for_placement("vae_only") == mp.RESIDENCY_ALL_RESIDENT
    assert mp.residency_for_placement("group_offload") == mp.TECHNIQUE_GROUP_OFFLOAD


def test_an_unprepped_pipeline_reports_unknown_residency_not_resident() -> None:
    """Unknown must never render as fine — the same rule the nullable
    measurement columns follow."""
    assert mp.residency_for_placement("") == ""
    assert not mp.PostureLedger().snapshot().observed


def test_an_unobserved_posture_is_not_sent() -> None:
    """The emit guard. An all-empty posture is the ABSENCE of a report, and the
    hub keys it differently on purpose; sending one claims "measured, nothing
    applied" on behalf of a worker that never looked."""
    lane_only = mp.PostureLedger().snapshot(
        execution_lane="fp8-w8a8-dynamic+eager", compile_state="eager")
    assert not lane_only.observed
    observed = mp.PostureLedger()
    observed.residency("off")
    assert observed.snapshot().observed


def test_the_ladder_projection_names_the_rung_not_the_wire_token() -> None:
    """th#1871 §6.6 item 5: `model_offload`, `group_offload` and `sequential`
    stop sharing the token `offload`. Their prices differ by 60%, and the hub
    could not tell them apart at all."""
    assert mp.technique_for_run_mode("offload", "group_offload") == \
        mp.TECHNIQUE_GROUP_OFFLOAD
    assert mp.technique_for_run_mode("offload", "sequential") == \
        mp.TECHNIQUE_SEQUENTIAL
    assert mp.technique_for_run_mode("fp8_storage", "") == mp.TECHNIQUE_FP8_STORAGE
    # A transition that named no rung: the coarse token is all there is, and
    # guessing which of the three it was would be a fabricated fact.
    assert mp.technique_for_run_mode("offload", "") == mp.TECHNIQUE_MODEL_OFFLOAD


def test_the_compile_axis_reads_both_cell_kinds_as_compiled() -> None:
    """`jit_cell` and `aot_cell` are both COMPILED. The artifact kind is a
    different axis (`metrics.serving_mode` carries it) and §1.30 keeps it out
    of the identity."""
    assert mp.compile_axis("eager") == mp.COMPILE_EAGER
    assert mp.compile_axis("jit_cell") == mp.COMPILE_COMPILED
    assert mp.compile_axis("aot_cell") == mp.COMPILE_COMPILED


def test_the_declared_compile_axis_comes_off_the_lane_and_nothing_else() -> None:
    """What the hub ASKED for. Reading it off the served lane would make every
    run trivially self-consistent — which is how minimax-h3 served eager on a
    declared-compiled lane for hours with nothing able to say so."""
    assert mp.compile_axis_of_lane("fp8-w8a8-dynamic+compiled") == "compiled"
    assert mp.compile_axis_of_lane("fp8-w8a8-dynamic+eager") == "eager"
    assert mp.compile_axis_of_lane("fp8-w8a8-dynamic") == ""
    assert mp.compile_axis_of_lane("") == ""
