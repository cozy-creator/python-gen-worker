"""The per-COMPONENT cold-boot decomposition, off the wire.

A leg-grade cold-boot number ("6.2 min") answers nothing: how long is the
per-class trace, the toolchain derivation, the memo delta, the
download-vs-admission split of an adopt?

WHY IT LOOKS LIKE THIS
----------------------
Same discipline as `test_boot_span_ladder_pgw797.py`: a boot instrument whose
unit tests call the emitters directly passes while the production path emits
NOTHING. So this file drives a REAL boot — `harness.hub_double` runs an actual
`Worker` against an actual gRPC server, weights are real bytes over a real HTTP
blob host — and reads the phase rows OFF THE WIRE. Nothing here calls a boot
emitter. If a production call site stops reaching one, these go red.

The fixture snapshot is deliberately MULTI-COMPONENT (transformer /
text_encoder / vae / root). With one weight file, "is the decomposition per
component" cannot fail, and an unfalsifiable assertion is not coverage.

THE RED PROOF
-------------
`test_the_completeness_verdict_goes_red_when_a_phase_stops_emitting` deletes
one phase's rows from the REAL boot's captured ladder and asserts the verdict
turns red and NAMES the deleted phase. A green completeness assertion proves
nothing until it is known it could have gone red — and the failure mode this
whole issue exists to prevent is precisely a phase that quietly stops emitting.
"""

from __future__ import annotations

from typing import List

import pytest

from gen_worker import boot_phases
from gen_worker.api.binding import wire_ref
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.blob_host import BlobHost
from harness.cold_boot_endpoints_pgw1087 import COMPILE_MODEL, COMPONENTS
from harness.hub_double import hub_double, is_ready


def _boot_rows(conn) -> List[pb.BootPhase]:
    return [
        m.boot_phase for m in list(conn.received)
        if m.WhichOneof("msg") == "boot_phase"
    ]


def _terminal(rows: List[pb.BootPhase], phase: str) -> List[pb.BootPhase]:
    return [r for r in rows if r.phase == phase and r.terminal]


def _one(rows: List[pb.BootPhase], phase: str) -> pb.BootPhase:
    got = _terminal(rows, phase)
    assert got, (
        f"no terminal {phase!r} row on the wire — the ladder has: "
        f"{sorted({r.phase for r in rows})}"
    )
    return got[0]


#: The shape this fixture drives. `hub_double` runs a real `Worker` IN
#: PROCESS, so it is an EMBEDDED boot: it never calls `env_seal.establish`,
#: which is an entrypoint/mint-child call. Asking it for `env_establish` would
#: be asking for a phase this boot shape structurally cannot have — the
#: entrypoint shape is proven where an entrypoint actually runs
#: (`tests_v2/test_boot.py::test_real_entrypoint_seals_dials_and_dumps_stacks`).
#: It also has no CUDA, so the compiled graph half is out of scope here and is covered by
#: the local micro-mint rig legs.
EXPECTED_SHAPE = boot_phases.SHAPE_EAGER


@pytest.fixture(scope="module")
def ladder(tmp_path_factory) -> List[pb.BootPhase]:
    """ONE real hub-delivered boot; every assertion below reads its ladder.

    One boot rather than one per test, for the reason pgw#797 documents:
    `boot_phases` state is process-global, so a second boot in the same
    interpreter starts with `in_boot()` already False and records nothing.
    Booting once is also what production gives a reader — one boot's table,
    whole.
    """
    boot_phases.reset_for_tests()
    tmp_path = tmp_path_factory.mktemp("pgw1087")
    blobs = BlobHost(tmp_path)
    model_ref = wire_ref(COMPILE_MODEL)
    files = [
        blobs.file(
            f"{name}-weights",
            f"pgw1087-{name}-bytes".encode() * (256 * (i + 1)),
            path_in_snapshot=f"{name}/weights.safetensors",
        )
        for i, name in enumerate(COMPONENTS)
    ]
    files.append(blobs.file(
        "model-index", b'{"_class_name": "MultiComponentPipeline"}',
        path_in_snapshot="model_index.json"))
    model_snap = blobs.snapshot("snap-multi", files)
    try:
        with hub_double(
            modules=("harness.cold_boot_endpoints_pgw1087",),
        ) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1,
                    disk_refs=[model_ref],
                    snapshots={model_ref: model_snap},
                    hot=[pb.DesiredInstance(
                        function_name="compile-echo",
                        models=[pb.ModelBinding(slot="model", ref=model_ref)],
                    )],
                ),
            ))
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and "compile-echo" in m.state_delta.available_functions
            )
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "boot_phase"
                and m.boot_phase.phase == boot_phases.PHASE_FIRST_REQUEST_SERVABLE
                and m.boot_phase.terminal
            )
            rows = _boot_rows(conn)
    finally:
        blobs.shutdown()
    return rows


# ---------------------------------------------------------------------------
# The acceptance criterion: the table is COMPLETE and it SUMS
# ---------------------------------------------------------------------------


def test_the_boot_phase_table_is_complete_and_sums_to_the_wall(ladder) -> None:
    """pgw#1087's acceptance, stated once: no phase of the driven shape is
    missing, and what the phases explain plus the NAMED segments is within 5%
    of the boot's wall clock, with whatever is left reported rather than
    smeared."""
    verdict = boot_phases.completeness(EXPECTED_SHAPE, rows=ladder)
    assert verdict.complete, (
        verdict.explain() + "\n\n"
        + boot_phases.render_phase_table(ladder))


def test_the_residual_is_named_not_a_lump(ladder) -> None:
    """"Unmeasured" and "zero" are different answers. The single biggest
    unmeasured window of every previous boot — interpreter start plus `import
    torch` plus endpoint discovery — is now the `sdk_ready` milestone and is
    subtracted from the residual as a NAMED segment, not left in it."""
    recon = boot_phases.reconciliation(ladder)
    assert recon["named_ms"] > 0, (
        "no named residual segment: the pre-SDK window is back inside the "
        f"unexplained lump ({recon})")
    assert recon["named.sdk_import_ms"] > 0
    assert recon["accounted_pct"] >= 95, recon


def test_every_declared_phase_of_this_shape_has_a_production_producer(
    ladder,
) -> None:
    """The pgw#924 rule, re-applied to the names pgw#1087 adds. A declared
    phase with no producer is a name every reader learns and the system can
    never emit — which is how `cell_discover` survived pgw#924's own audit."""
    emitted = {r.phase for r in ladder if r.terminal}
    for phase in EXPECTED_SHAPE:
        assert phase in emitted, (
            f"{phase} is declared but a REAL boot of this shape produced no "
            f"row; emitted: {sorted(emitted)}")


# ---------------------------------------------------------------------------
# The per-COMPONENT half — the actual subject of the issue
# ---------------------------------------------------------------------------


def test_weights_download_decomposes_per_component(ladder) -> None:
    """`weights_fetch` said how long the ref took and nothing about which
    component owned it. Four components, four rows, each with its own bytes."""
    comps = _terminal(ladder, boot_phases.PHASE_COMPONENT_FETCH)
    names = sorted({r.function for r in comps})
    assert names == sorted([*COMPONENTS, "(root)"]), (
        f"component rows {names} do not cover the snapshot's components "
        f"{sorted([*COMPONENTS, '(root)'])}")
    assert all(r.bytes > 0 for r in comps if r.function in COMPONENTS), (
        f"a weight component reported no bytes: "
        f"{[(r.function, r.bytes, r.source) for r in comps]}")
    assert all(r.source for r in comps), (
        "a component named no source — a cold pull and a warm CAS hit are the "
        f"same phase at wildly different rates: {[(r.function, r.source) for r in comps]}")


def test_component_rows_nest_under_the_weights_fetch_they_decompose(
    ladder,
) -> None:
    """A decomposition that does not nest double-counts: `weights_fetch` and
    its components would each charge the same seconds, and the ladder would
    stop reconciling — the exact defect pgw#797 fixed for `warmup`."""
    fetch = _one(ladder, boot_phases.PHASE_WEIGHTS_FETCH)
    comps = _terminal(ladder, boot_phases.PHASE_COMPONENT_FETCH)
    assert comps, "no component rows under a boot that fetched weights"
    assert all(r.parent_ordinal == fetch.ordinal for r in comps), (
        "component rows are not children of weights_fetch: "
        f"{[(r.function, r.parent_ordinal) for r in comps]} vs {fetch.ordinal}")
    table = {r.ordinal: r for r in boot_phases.phase_table(ladder)}
    assert table[fetch.ordinal].exclusive_ms <= fetch.duration_ms


def test_component_intervals_make_overlap_readable(ladder) -> None:
    """The reason a component row carries START and END rather than a
    duration: four 180 s components inside a 200 s fetch is a completely
    different finding from four sequential 50 s ones, and only intervals can
    tell them apart. Assert the interval is well-formed and inside its
    parent's — the arithmetic an overlap query depends on."""
    rows = {
        r.function: r for r in boot_phases.phase_table(ladder)
        if r.phase == boot_phases.PHASE_COMPONENT_FETCH
    }
    fetch = next(
        r for r in boot_phases.phase_table(ladder)
        if r.phase == boot_phases.PHASE_WEIGHTS_FETCH)
    for name, row in rows.items():
        assert row.end_ms >= row.start_ms, name
        assert row.end_ms <= fetch.end_ms + 1, (
            f"component {name} ends after its weights_fetch parent "
            f"({row.end_ms} > {fetch.end_ms})")


# ---------------------------------------------------------------------------
# The two user-visible timestamps, and the toolchain derivation
# ---------------------------------------------------------------------------


def test_eager_ready_precedes_the_boot_close_and_is_cumulative(ladder) -> None:
    """`eager_ready` is the first instant a request could have been answered
    at all; `first_request_servable` additionally requires the hub to have been
    told. Both measure from process start, so the first cannot follow the
    second."""
    eager = _one(ladder, boot_phases.PHASE_EAGER_READY)
    servable = _one(ladder, boot_phases.PHASE_FIRST_REQUEST_SERVABLE)
    assert eager.cumulative and eager.parent_ordinal == 0, (
        "a cumulative milestone charged against a span drives that span's "
        "exclusive time to zero (pgw#797)")
    assert eager.duration_ms <= servable.duration_ms


def test_sdk_ready_is_cumulative_and_precedes_every_span(ladder) -> None:
    """The import wall. No span can cover it — the recorder is part of what is
    being imported — so it is a milestone, and it must bound the ladder."""
    sdk = _one(ladder, boot_phases.PHASE_SDK_READY)
    assert sdk.cumulative and sdk.parent_ordinal == 0
    assert sdk.duration_ms > 0, (
        "sdk_ready measured 0 ms: the interpreter+import window cannot be free")


def test_the_entrypoint_shape_is_a_superset_and_is_proven_elsewhere() -> None:
    """The seal derivation belongs to the ENTRYPOINT shape, not to every boot.

    Stated here as a structural assertion so the split cannot silently
    collapse: an embedded worker must not be asked for `env_establish`, and a
    POD boot must still be. The pod half is asserted, off the wire, in
    `tests_v2/test_boot.py::test_real_entrypoint_seals_dials_and_dumps_stacks`.
    """
    assert boot_phases.SHAPE_EAGER < boot_phases.SHAPE_ENTRYPOINT
    assert boot_phases.SHAPE_ENTRYPOINT - boot_phases.SHAPE_EAGER == {
        boot_phases.PHASE_ENV_ESTABLISH, boot_phases.PHASE_LIB_MEMO}
    assert boot_phases.SHAPE_ENTRYPOINT < boot_phases.SHAPE_ADOPT


# ---------------------------------------------------------------------------
# THE RED PROOF
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dropped", sorted(EXPECTED_SHAPE))
def test_the_completeness_verdict_goes_red_when_a_phase_stops_emitting(
    ladder, dropped: str,
) -> None:
    """Delete one phase's rows from the REAL boot and watch the verdict fail.

    This is the assertion's own red proof, and it is parametrized over every
    phase of the shape rather than one convenient name: a completeness check
    that could only ever fail on `weights_fetch` would let the other eight
    rot silently, which is the exact history of this vocabulary.
    """
    survivors = [r for r in ladder if r.phase != dropped]
    verdict = boot_phases.completeness(EXPECTED_SHAPE, rows=survivors)
    assert not verdict.complete, (
        f"removing every {dropped!r} row left the table reading COMPLETE — "
        "the assertion cannot detect the failure it exists for")
    assert dropped in verdict.missing or not verdict.reconciles, (
        f"{dropped} vanished without being named: {verdict.explain()}")


def test_a_ladder_with_no_boot_close_is_never_complete(ladder) -> None:
    """A pod that died mid-boot has no wall clock to reconcile against, and a
    verdict of "complete" on it would be a default read as a fact."""
    survivors = [
        r for r in ladder
        if r.phase != boot_phases.PHASE_FIRST_REQUEST_SERVABLE]
    verdict = boot_phases.completeness(EXPECTED_SHAPE, rows=survivors)
    assert not verdict.complete
    assert "never closed" in verdict.explain()


# ---------------------------------------------------------------------------
# The artifact this issue exists to produce
# ---------------------------------------------------------------------------


def test_the_phase_table_renders(ladder, capsys) -> None:
    """The decomposition, printed. Not decoration: pgw#1087's third acceptance
    box is that campaign runbooks cite PHASE NAMES rather than leg aggregates,
    and this is the shape they cite."""
    text = boot_phases.render_phase_table(ladder)
    assert boot_phases.PHASE_COMPONENT_FETCH in text
    assert "residual_ms" in text and "accounted_pct" in text
    with capsys.disabled():
        print("\n--- pgw#1087 cold-boot decomposition (real boot) ---")
        print(text)
