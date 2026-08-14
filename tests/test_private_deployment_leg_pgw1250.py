"""pgw#1250 — the private-deployment mint leg, as a row.

RUNS IN CI, entirely. Pure Python: no GPU, no card, no hub, no network, no
money. That is the point of the seam — the driver talks to a `DeploymentAPI`,
and `ContractModel` implements the settled contract in process, so the whole
leg (create -> invoke -> read -> stop) is provable red/green before tensorhub's
invoke route and settlement have merged.

What a green here does NOT mean: it is not evidence about tensorhub. It proves
the DRIVER reads the contract correctly. Every assertion the model cannot reach
is asserted to be reported BLOCKED — naming the merge that unblocks it — rather
than skipped, because a skipped assertion on a money-bearing path is how a
report ends up claiming more than it checked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from private_deployment_leg import (  # noqa: E402
    BLOCKED,
    SKIPPED,
    BLOCKER_COMPILED_GRAPH_INVENTORY,
    BLOCKER_INVOKE_ROUTE,
    BLOCKER_RECONCILER,
    BLOCKER_SETTLEMENT,
    CHOOSER_PRIVATE_DEPLOYMENT,
    FAILED,
    OK,
    STATE_ACTIVE,
    STATE_STOPPED,
    STATE_STOPPING,
    ContractModel,
    Leg,
    compiled_graph_inventory_path,
    decoded_pixel_digest,
    coherence_error,
    parse_pair,
    sku_slug,
    run_leg,
)


def _leg(**overrides: Any) -> Leg:
    base: Dict[str, Any] = dict(
        org="tensorhub",
        endpoint="tensorhub/sdxl",
        release_id="rel-sdxl-0240",
        pair=("a40", "bf16-w16a16+compiled"),
        function="generate",
        payload={"prompt": "a red fox", "seed": 1929},
        invocations=1,
        spend_limit_usd=1.0,
    )
    base.update(overrides)
    return Leg(**base)


def _run(model: ContractModel, leg: Leg) -> Any:
    # The model's clock is the leg's clock, so pod-seconds accrue without the
    # test sleeping and the stall budget measures MODEL time.
    return run_leg(model, leg, poll_s=0.0, stall_budget_s=10_000.0,
                   clock=lambda: float(model.now), sleep=lambda _s: None)


def _finding(result: Any, ident: str) -> Any:
    for finding in result.findings:
        if finding.ident == ident:
            return finding
    raise AssertionError(f"no finding {ident!r} among {[f.ident for f in result.findings]}")


def test_leg_on_a_fully_merged_hub_is_green_end_to_end() -> None:
    model = ContractModel(provisioning=True, invoke_route=True, settlement=True)
    result = _run(model, _leg())

    assert result.status == OK, [
        (f.ident, f.status, f.detail) for f in result.findings if f.status != OK
    ]
    assert [p.name for p in result.phases] == ["create", "provision", "invoke", "read", "stop"]
    assert result.usage["settlement"]["available"] is True
    assert result.usage["settlement"]["provider_micros"] > 0
    assert result.deployment["state"] in (STATE_STOPPING, STATE_STOPPED)
    assert result.blockers == []
    # The product check ran, and it is what makes this leg an acceptance proof
    # rather than a latency measurement.
    for ident in ("seal.rows", "seal.sku_matches_rented_card", "seal.artifact_ref",
                  "seal.graph_keys_exact", "seal.artifact_refs_exact",
                  "seal.worker_versions_exact", "seal.not_quarantined",
                  "seal.sm_recorded", "seal.no_failed_publish"):
        assert _finding(result, ident).status == OK, _finding(result, ident).detail
    for ident in (
        "evidence.status[0]",
        "evidence.outcome[0]",
        "evidence.refusal[0]",
        "evidence.graph_key[0]",
        "evidence.artifact_ref[0]",
        "evidence.receipt_ref[0]",
        "evidence.versions[0]",
        "evidence.serving_pid[0]",
        "evidence.no_serving_compile[0]",
        "evidence.hashrepo_refs[0]",
        "evidence.hashrepo_objects[0]",
        "evidence.bind_fqns[0]",
        "evidence.bind_calls[0]",
        "evidence.runner_calls[0]",
        "evidence.pixel_digest[0]",
        "evidence.publisher_compile_count[0]",
        "evidence.publisher_spawn_count[0]",
        "evidence.compile_child[0]",
    ):
        assert _finding(result, ident).status == OK, _finding(result, ident).detail


def test_leg_today_is_blocked_and_names_every_missing_merge() -> None:
    """th#1926 merged, th#1927's reconciler and invoke route and th#1928 not.

    The leg must still create, stop cleanly, and produce a report in which every
    unreachable assertion is BLOCKED with its issue named.
    """
    model = ContractModel()
    result = _run(model, _leg())

    assert result.status == BLOCKED
    for blocker in (BLOCKER_RECONCILER, BLOCKER_INVOKE_ROUTE, BLOCKER_SETTLEMENT):
        assert blocker in result.blockers, result.blockers
    # The half that IS merged is fully asserted, not deferred with the rest.
    for ident in ("create.accepted", "create.state_active", "create.row_coherent",
                  "create.release_pinned", "create.pair_roundtrip",
                  "create.generation_genesis", "create.owner_recorded",
                  "read.history_genesis", "settle.honest_absence",
                  "stop.accepted", "stop.row_coherent", "stop.idempotent",
                  "stop.history_source"):
        assert _finding(result, ident).status == OK, _finding(result, ident).detail


def test_leg_with_pods_but_no_invoke_route_separates_the_two_halves() -> None:
    """The state the epic is in the day th#1927's fences merge but its follow-up
    has not: pods exist, the workload cannot be submitted."""
    model = ContractModel(provisioning=True)
    result = _run(model, _leg())

    assert _finding(result, "invoke.route").status == BLOCKED
    assert _finding(result, "invoke.route").blocker == BLOCKER_INVOKE_ROUTE
    # Provisioning must read GREEN here, or the report cannot tell "the
    # reconciler works" from "nothing works".
    assert _finding(result, "provision.ready").status == OK
    chooser = [f for f in result.findings if f.ident.startswith("provision.chooser")]
    assert chooser and all(f.status == OK for f in chooser)


@pytest.mark.parametrize(
    "break_invariant, ident_fragment",
    [
        ("coherence", "stop.row_coherent"),
        ("chooser", "provision.chooser"),
        ("phantom_figures", "settle.no_phantom_figures"),
        ("margin_drift", "margin_frozen"),
        ("stop_not_idempotent", "stop.idempotent"),
    ],
)
def test_every_money_bearing_assertion_is_red_provable(break_invariant: str, ident_fragment: str) -> None:
    """A guard that has never been seen fail is not a guard.

    Each case severs exactly one invariant in the model and requires the
    matching assertion to go red — and the leg with it.
    """
    merged = break_invariant != "phantom_figures"
    model = ContractModel(provisioning=merged, invoke_route=merged, settlement=merged,
                          break_invariant=break_invariant)
    result = _run(model, _leg())

    reds = [f for f in result.findings if ident_fragment in f.ident and f.status == FAILED]
    assert reds, (
        f"severing {break_invariant!r} turned no {ident_fragment!r} assertion red; "
        f"leg status={result.status}"
    )
    assert result.status == FAILED


def test_the_rental_is_always_stopped_whatever_the_leg_does() -> None:
    """The money property. The deployment id is the entire kill set — there is no
    demand to cancel, no endpoint_tags query, and no conditional terminate."""
    model = ContractModel(provisioning=True)  # no invoke route: the leg exits early
    result = _run(model, _leg())

    assert result.deployment_id
    row, _pods = model.get("tensorhub", result.deployment_id)
    assert row["state"] != STATE_ACTIVE, row
    assert coherence_error(row["state"], row["stop_reason"], row["stopped_at"]) is None


def test_a_refused_leg_creates_nothing() -> None:
    model = ContractModel(provisioning=True, invoke_route=True, settlement=True)
    refusals: List[Tuple[Dict[str, Any], str]] = [
        ({"release_id": ""}, "release_id is required"),
        ({"pair": ("", "bf16-w16a16+compiled")}, "must name both a GPU and a lane"),
        ({"pod_count": 0}, "pod_count must be at least 1"),
        ({"access_mode": "everyone"}, "must be owner or org"),
        ({"on_pod_failure": "ignore"}, "must be replace or stop"),
    ]
    for bad, expected in refusals:
        with pytest.raises(ValueError) as caught:
            _run(model, _leg(**bad))
        assert expected in str(caught.value)
    assert not model.rows, "a refused leg created a rental"


def test_a_rental_may_not_name_a_bare_lane() -> None:
    assert parse_pair("a40:bf16-w16a16+compiled") == ("a40", "bf16-w16a16+compiled")
    assert parse_pair("  A40 : BF16-W16A16+COMPILED ") == ("a40", "bf16-w16a16+compiled")
    for bad in ("bf16-w16a16+compiled", ":lane", "a40:", ""):
        with pytest.raises(ValueError):
            parse_pair(bad)


def test_coherence_mirrors_the_schema_checks_exactly() -> None:
    stamp = "2026-08-14T12:00:00Z"
    cases: List[Any] = [
        ((STATE_ACTIVE, "", None), True),
        ((STATE_STOPPING, "owner_stop", None), True),
        ((STATE_STOPPED, "owner_stop", stamp), True),
        # The two CHECKs admit a stopped row with an EMPTY stop_reason, and this
        # mirror admits exactly what they admit. Refusing an unnamed stop is the
        # DRIVER's separate `stop.reason` assertion, not a schema invariant.
        ((STATE_STOPPED, "", stamp), True),
        ((STATE_ACTIVE, "owner_stop", None), False),
        ((STATE_ACTIVE, "", stamp), False),
        ((STATE_STOPPING, "owner_stop", stamp), False),
        ((STATE_STOPPED, "owner_stop", None), False),
    ]
    for (state, reason, stopped_at), ok in cases:
        assert (coherence_error(state, reason, stopped_at) is None) is ok, (state, reason, stopped_at)


def test_a_leg_with_no_invocations_is_a_lifecycle_proof_not_a_silent_pass() -> None:
    """`invocations=0` is what a mint proof that only needs the pod to BOOT looks
    like. It must read SKIPPED, which is distinct from OK and from BLOCKED."""
    model = ContractModel(provisioning=True, invoke_route=True, settlement=True)
    result = _run(model, _leg(invocations=0))

    invoke = _finding(result, "invoke.workload")
    assert invoke.status == "skipped"
    assert _finding(result, "provision.ready").status == OK
    assert result.pods and result.pods[0]["placement_chooser"] == CHOOSER_PRIVATE_DEPLOYMENT


@pytest.mark.parametrize(
    "break_invariant, ident",
    [
        ("seal_sku_mismatch", "seal.sku_matches_rented_card"),
        ("seal_no_artifact", "seal.artifact_ref"),
        ("seal_failed_publish", "seal.no_failed_publish"),
        ("seal_wrong_release", "seal.rows"),
    ],
)
def test_the_product_check_is_red_provable(break_invariant: str, ident: str) -> None:
    """Each case is a way a mint leg could have reported success while producing
    nothing usable: sealed on a card it never rented, a key with no artifact
    behind it, a failed publish phase, or a graph minted for another release."""
    model = ContractModel(provisioning=True, invoke_route=True, settlement=True,
                          break_invariant=break_invariant)
    result = _run(model, _leg())

    assert _finding(result, ident).status == FAILED, (
        f"severing {break_invariant!r} left {ident} at {_finding(result, ident).status}"
    )
    assert result.status == FAILED


def test_missing_seal_routes_are_blocked_not_green() -> None:
    model = ContractModel(provisioning=True, invoke_route=True, settlement=True,
                          seal_evidence=False)
    result = _run(model, _leg())

    assert _finding(result, "seal.route").status == BLOCKED
    assert _finding(result, "seal.route").blocker == BLOCKER_COMPILED_GRAPH_INVENTORY
    assert result.status == BLOCKED


def test_a_leg_that_is_not_a_mint_proof_claims_no_product() -> None:
    model = ContractModel(provisioning=True, invoke_route=True, settlement=True)
    result = _run(model, _leg(mint_proof=False))

    assert not [f for f in result.findings if f.ident.startswith("seal.")]
    assert result.status == OK


def test_sku_slug_bridges_the_two_vocabularies() -> None:
    """worker_pods.gpu_class holds the provider's CATALOGUE ID; the graph store
    keys on the compilecache SKU SLUG. Comparing them raw is always false — a
    vacuously RED assertion, as useless as a vacuously green one and harder to
    spot, because red looks like it is working."""
    assert sku_slug("NVIDIA A40") == "a40"
    assert sku_slug("NVIDIA GeForce RTX 4090") == "rtx-4090"
    assert sku_slug("NVIDIA RTX A4000") == "rtx-a4000"
    assert sku_slug("NVIDIA H100 80GB HBM3") == "h100-80gb-hbm3"
    assert sku_slug("a40") == "a40"
    # The model records the provider display name exactly as a real hub does,
    # so the bridge is load-bearing in every seal test above.
    model = ContractModel(provisioning=True, invoke_route=True, settlement=True)
    result = _run(model, _leg())
    assert result.pods[0]["gpu_class"] == "NVIDIA A40"


def test_a_mint_proof_that_never_invoked_reports_no_false_red() -> None:
    """The trap this closes: on TODAY's hub the invoke route is absent, so a
    mint-proof leg completes no request. Asserting an empty graph store there
    would report "nothing was published" as a defect of the mint, when the real
    answer is upstream and the invoke phase already names it. That is a false
    red, and a false red costs exactly as much trust as a false green."""
    model = ContractModel()  # resource only: no reconciler, no invoke route
    result = _run(model, _leg())

    assert _finding(result, "seal.workload").status == SKIPPED
    assert not [f for f in result.findings
                if f.ident.startswith("seal.") and f.status == FAILED]
    assert result.status == BLOCKED
    assert BLOCKER_INVOKE_ROUTE in result.blockers


def test_admin_inventory_is_a_hard_compiled_graph_cut() -> None:
    assert compiled_graph_inventory_path("rel 1") == (
        "/v1/admin/compiled-graphs?view=compiled_graphs&release=rel+1&limit=200"
    )


def test_decoded_pixel_digest_binds_shape_mode_and_raw_pixels() -> None:
    raw = bytes((0, 17, 34, 51, 68, 85))
    baseline = decoded_pixel_digest("RGB", 2, 1, raw)

    assert baseline.startswith("sha256:")
    assert len(baseline) == len("sha256:") + 64
    assert decoded_pixel_digest("RGBA", 2, 1, raw) != baseline
    assert decoded_pixel_digest("RGB", 1, 2, raw) != baseline
    assert decoded_pixel_digest("RGB", 2, 1, raw[:-1] + b"\x56") != baseline


def test_second_request_proves_same_graph_reuse_without_compile_or_spawn() -> None:
    model = ContractModel(provisioning=True, invoke_route=True, settlement=True)
    result = _run(model, _leg(invocations=2))

    assert result.status == OK, [
        (f.ident, f.status, f.detail) for f in result.findings if f.status != OK
    ]
    published = result.requests[0]["compiled_graph_evidence"]
    reused = result.requests[1]["compiled_graph_evidence"]
    assert reused["outcome"] == "reused"
    assert reused["compiled_graph_key"] == published["compiled_graph_key"]
    assert reused["artifact_ref"] == published["artifact_ref"]
    assert reused["receipt_ref"] == published["receipt_ref"]
    assert reused["decoded_pixel"] == published["decoded_pixel"]
    for ident in (
        "evidence.reuse_compile_count[1]",
        "evidence.reuse_spawn_count[1]",
        "evidence.reuse_no_child[1]",
    ):
        assert _finding(result, ident).status == OK, _finding(result, ident).detail


@pytest.mark.parametrize(
    "break_invariant, invocations, ident",
    [
        ("evidence_status", 1, "evidence.status[0]"),
        ("evidence_outcome", 1, "evidence.outcome[0]"),
        ("evidence_refusal", 1, "evidence.refusal[0]"),
        ("evidence_graph_key", 1, "evidence.graph_key[0]"),
        ("evidence_artifact_ref", 1, "evidence.artifact_ref[0]"),
        ("evidence_receipt_ref", 1, "evidence.receipt_ref[0]"),
        ("evidence_versions", 1, "evidence.versions[0]"),
        ("evidence_serving_pid", 1, "evidence.serving_pid[0]"),
        ("evidence_serving_compile", 1, "evidence.no_serving_compile[0]"),
        ("evidence_manifest_ref", 1, "evidence.hashrepo_manifest_ref[0]"),
        ("evidence_materialized_root", 1, "evidence.hashrepo_materialized_root[0]"),
        ("evidence_nonempty_cache", 1, "evidence.empty_cache[0]"),
        ("evidence_ref_count", 1, "evidence.hashrepo_refs[0]"),
        ("evidence_object_count", 1, "evidence.hashrepo_objects[0]"),
        ("evidence_bind_fqns", 1, "evidence.bind_fqns[0]"),
        ("evidence_bind_calls", 1, "evidence.bind_calls[0]"),
        ("evidence_runner_calls", 1, "evidence.runner_calls[0]"),
        ("evidence_pixel_shape", 1, "evidence.pixel_shape[0]"),
        ("evidence_pixel_digest", 1, "evidence.pixel_digest[0]"),
        ("evidence_publish_compile", 1, "evidence.publisher_compile_count[0]"),
        ("evidence_publish_spawn", 1, "evidence.publisher_spawn_count[0]"),
        ("evidence_compile_child", 1, "evidence.compile_child[0]"),
        ("evidence_reuse_compile", 2, "evidence.reuse_compile_count[1]"),
        ("evidence_reuse_spawn", 2, "evidence.reuse_spawn_count[1]"),
        ("evidence_reuse_child", 2, "evidence.reuse_no_child[1]"),
    ],
)
def test_every_compiled_graph_evidence_gate_is_red_provable(
        break_invariant: str, invocations: int, ident: str) -> None:
    model = ContractModel(
        provisioning=True,
        invoke_route=True,
        settlement=True,
        break_invariant=break_invariant,
    )
    result = _run(model, _leg(invocations=invocations))

    assert _finding(result, ident).status == FAILED, (
        f"severing {break_invariant!r} left {ident} at {_finding(result, ident).status}"
    )
    assert result.status == FAILED
