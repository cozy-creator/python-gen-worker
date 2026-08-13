"""pgw#947 — the mint MEASURES which serving-kernel lane wins on the target
card, records the verdict into the cell, and serving adopts it.

Integration-style over mocks: the tests drive the REAL selection rule, the
REAL envelope/evidence round-trip, and the REAL adoption path (a packed tar
on disk, read by the code the executor calls). The only thing stubbed is the
pair of lanes themselves — a benchmark needs a GPU and two compiled graphs,
and what has to be proven here is the DECISION, not that torch can time a
kernel.

The decisive check is the last one: the mechanism, handed the recorded
pgw#862/#863 numbers, independently reproduces the two answers we already
know are right — fused on the 5090, baseline on the B200.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from gen_worker import kernel_path as kl

GB = 1000 ** 3  # the benchmark tables are in decimal GB
GiB = 1 << 30


def _m(execution_lane: str, ms: float, peak_gb: float) -> kl.Measurement:
    return kl.Measurement(
        execution_lane=execution_lane, ms_per_step=ms, peak_bytes=int(peak_gb * GB),
        samples_ms=(ms,))


def _on_card(
    monkeypatch: pytest.MonkeyPatch, total: int, name: str, sm: str,
) -> None:
    """Serve as if this process were on that card. Adoption re-applies the
    fit rule against the LOCAL device, so a test of adoption has to say which
    device it is adopting on — that is the whole point of the mechanism."""
    monkeypatch.setattr(kl, "device_facts", lambda: (total, name, sm))


@pytest.fixture(autouse=True)
def _clear_pin():
    kl.clear()
    yield
    kl.clear()


# --- the rule --------------------------------------------------------------


def test_fastest_that_fits_wins_even_when_it_is_the_larger_execution_lane() -> None:
    """(a) Speed is the objective. B200 shape: the baseline lane is 9.5 GB
    BIGGER and 35% faster, and the card has the room — so it wins."""
    verdict = kl.select(
        [_m("baseline+dense", 228.0, 44.1), _m("fused+packed", 350.0, 35.1)],
        device_total_bytes=180 * GB, sm="sm_100")
    assert verdict.winner == "baseline+dense"
    assert verdict.binding == kl.BIND_SPEED
    assert "228.0" in verdict.detail and "350.0" in verdict.detail


def test_a_execution_lane_that_does_not_fit_is_excluded_even_when_faster() -> None:
    """(b) Fit is a CONSTRAINT applied before ranking. Same two lanes, on a
    48 GB card: the baseline's 44.1 GB peak plus its allowance cannot fit, so
    the 35%-slower fused lane wins outright."""
    verdict = kl.select(
        [_m("baseline+dense", 228.0, 44.1), _m("fused+packed", 350.0, 35.1)],
        device_total_bytes=48 * GB, sm="sm_100")
    assert verdict.winner == "fused+packed"
    assert verdict.binding == kl.BIND_FIT
    assert "baseline+dense" in verdict.detail


def test_headroom_allowance_excludes_a_execution_lane_that_bare_peak_would_admit() -> None:
    """The allowance does real work: a 40 GB peak fits a 44 GB card by bare
    measurement and does NOT fit once the activation-spike + fragmentation
    allowance is applied — which is the point of stating one."""
    row = _m("baseline+dense", 100.0, 40.0)
    assert row.peak_bytes < 44 * GB
    assert not kl.fits(row, 44 * GB)
    assert row.required_bytes() == int(40 * GB * 1.20) + GiB


def test_within_margin_ties_fall_to_the_smaller_execution_lane() -> None:
    """(c) VRAM breaks a tie and ONLY a tie. 2% apart is noise, so the
    smaller peak wins; 6% apart is a real win, so speed does."""
    tie = kl.select(
        [_m("baseline+dense", 300.0, 44.0), _m("fused+packed", 306.0, 35.0)],
        device_total_bytes=180 * GB)
    assert tie.winner == "fused+packed"
    assert tie.binding == kl.BIND_VRAM_TIEBREAK

    win = kl.select(
        [_m("baseline+dense", 300.0, 44.0), _m("fused+packed", 319.0, 35.0)],
        device_total_bytes=180 * GB)
    assert win.winner == "baseline+dense"
    assert win.binding == kl.BIND_SPEED


def test_margin_makes_the_verdict_deterministic_across_mints() -> None:
    """Two mints on one card must not disagree. The measurement jitters; the
    verdict may not, because a win under the margin is not a win."""
    verdicts = {
        kl.select([_m("baseline+dense", 300.0 + jitter, 44.0),
                   _m("fused+packed", 297.0 - jitter, 35.0)],
                  device_total_bytes=180 * GB).winner
        for jitter in (-3.0, -1.0, 0.0, 1.0, 3.0)
    }
    assert verdicts == {"fused+packed"}  # within margin every time => smaller peak


def test_unmeasurable_candidate_drops_out_with_its_reason() -> None:
    verdict = kl.select(
        [_m("baseline+dense", 400.0, 20.0),
         kl.Measurement(execution_lane="fused+packed", unavailable="triton compile failed")],
        device_total_bytes=180 * GB)
    assert verdict.winner == "baseline+dense"
    assert verdict.binding == kl.BIND_SOLE_CANDIDATE
    assert verdict.measurement("fused+packed").unavailable == "triton compile failed"


def test_nothing_fits_names_itself_and_takes_the_smallest() -> None:
    verdict = kl.select(
        [_m("baseline+dense", 228.0, 44.1), _m("fused+packed", 350.0, 35.1)],
        device_total_bytes=24 * GB)
    assert verdict.binding == kl.BIND_NO_FIT
    assert verdict.winner == "fused+packed"


def test_nothing_measured_is_the_declared_default() -> None:
    verdict = kl.select([], device_total_bytes=180 * GB)
    assert verdict.winner == kl.DEFAULT_EXECUTION_LANE
    assert verdict.binding == kl.BIND_NO_FIT


# --- the probe loop --------------------------------------------------------


def test_probe_measures_every_candidate_and_a_build_failure_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The A/B harness end to end with a stubbed timer: both candidates are
    built, one blows up, the survivor wins and the failure is recorded."""
    timings = {"baseline+dense": (250.0, 40 * GB),
               "fused+packed": (300.0, 30 * GB)}

    def _measure(execution_lane: str, step):
        step()
        ms, peak = timings[execution_lane]
        return kl.Measurement(execution_lane=execution_lane, ms_per_step=ms, peak_bytes=peak,
                              samples_ms=(ms,))

    monkeypatch.setattr(kl, "measure", _measure)
    built = []

    def _build(execution_lane: str):
        if execution_lane == "fused+dense":
            raise RuntimeError("no kernels here")
        built.append(execution_lane)
        return lambda: None

    verdict = kl.probe(
        ("baseline+dense", "fused+dense", "fused+packed"), _build,
        device_total_bytes=180 * GB, device_name="stub", sm="sm_100")
    assert built == ["baseline+dense", "fused+packed"]
    assert verdict.winner == "baseline+dense"
    assert ("no kernels here"
            in verdict.measurement("fused+dense").unavailable)


# --- recording -------------------------------------------------------------


def test_envelope_is_discrete_and_evidence_round_trips() -> None:
    """(e) The verdict rides the packed envelope; the NUMBERS ride the
    published evidence block and come back out unchanged.

    The split is not cosmetic: the #699 double-mint byte-compare forbids wall
    clocks inside the artifact, so metadata.json carries only facts a second
    mint would reproduce."""
    verdict = kl.select(
        [_m("baseline+dense", 228.0, 44.1), _m("fused+packed", 350.0, 35.1)],
        device_total_bytes=180 * GB, device_name="NVIDIA B200", sm="sm_100")

    envelope = kl.envelope_block(verdict)
    assert envelope == {
        "schema": kl.SCHEMA, "winner": "baseline+dense",
        "rule": "fit_constrained_speed", "binding": kl.BIND_SPEED,
        "margin_fraction": kl.MARGIN_FRACTION,
        "candidates": ["baseline+dense", "fused+packed"],
        # The FIT half of the rule travels with the artifact, because a card
        # of the same SM but a different size has to re-apply it. Quantized
        # BYTES only — discrete, reproducible, and not a wall clock.
        "fit": {
            "quantum_bytes": kl.PEAK_QUANTUM_BYTES,
            "activation_spike_fraction": kl.ACTIVATION_SPIKE_FRACTION,
            "fragmentation_headroom_bytes": kl.FRAGMENTATION_HEADROOM_BYTES,
            "peaks": {
                "baseline+dense": kl.quantize_peak(int(44.1 * GB)),
                "fused+packed": kl.quantize_peak(int(35.1 * GB)),
            },
            "order": ["baseline+dense", "fused+packed"],
        },
    }
    # No wall clocks, no timings, JSON-clean.
    flat = json.dumps(envelope)
    assert "ms" not in flat and "measured_at" not in flat

    evidence = json.loads(json.dumps(kl.evidence_block(verdict)))
    back = kl.verdict_from_evidence(evidence)
    assert back.winner == verdict.winner
    assert back.binding == verdict.binding
    assert back.device_name == "NVIDIA B200"
    assert back.measurement("fused+packed").ms_per_step == 350.0
    assert back.measurement("baseline+dense").peak_bytes == int(44.1 * GB)
    # Re-running the RULE over the recorded evidence reproduces the verdict —
    # which is what makes a recorded decision auditable rather than asserted.
    assert kl.select(
        back.measurements,
        device_total_bytes=back.device_total_bytes).winner == verdict.winner


# --- adoption --------------------------------------------------------------


def _packed(tmp_path: Path, meta: dict) -> Path:
    body = tmp_path / "metadata.json"
    body.write_text(json.dumps(meta))
    artifact = tmp_path / "cell.tar.gz"
    with tarfile.open(artifact, "w:gz") as tar:
        tar.add(body, arcname="metadata.json")
    return artifact


def test_serving_adopts_the_execution_lane_the_cell_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _on_card(monkeypatch, 32 * GB, "NVIDIA RTX 5090", "sm_120")
    verdict = kl.select(
        [_m("baseline+dense", 1189.0, 22.4), _m("fused+packed", 975.0, 15.6)],
        device_total_bytes=32 * GB, sm="sm_120")
    artifact = _packed(tmp_path, {
        "kind": "aot-inductor",
        kl.META_KEY: kl.envelope_block(verdict),
    })
    assert kl.adopt_from_artifact(artifact) == "fused+packed"
    execution_lane, reason = kl.pinned()
    assert execution_lane == "fused+packed"
    assert kl.REASON_ADOPTED in reason


def test_cell_without_a_verdict_is_the_default_with_a_typed_reason(
    tmp_path: Path,
) -> None:
    """(d) A pre-pgw#947 cell records nothing. That is the declared default
    and it SAYS which case it was — never a silent fall-through."""
    artifact = _packed(tmp_path, {"kind": "aot-inductor"})
    assert kl.adopt_from_artifact(artifact) == kl.DEFAULT_EXECUTION_LANE
    assert kl.REASON_ABSENT in kl.pinned()[1]


def test_no_cell_at_all_is_the_default_with_its_own_reason() -> None:
    assert kl.adopt(None) == kl.DEFAULT_EXECUTION_LANE
    assert kl.REASON_NO_CELL in kl.pinned()[1]


def test_unreadable_artifact_degrades_and_names_the_failure(
    tmp_path: Path,
) -> None:
    junk = tmp_path / "not-a-tar.tar.gz"
    junk.write_bytes(b"definitely not a tarball")
    assert kl.adopt_from_artifact(junk) == kl.DEFAULT_EXECUTION_LANE
    assert kl.REASON_UNREADABLE in kl.pinned()[1]


def test_a_execution_lane_this_worker_does_not_implement_is_refused_by_name() -> None:
    execution_lane, reason = kl.execution_lane_from_metadata(
        {kl.META_KEY: {"winner": "tcgen05-v2", "binding": "speed"}})
    assert execution_lane == kl.DEFAULT_EXECUTION_LANE
    assert kl.REASON_UNKNOWN_EXECUTION_LANE in reason


def test_pin_refuses_a_execution_lane_outside_the_vocabulary() -> None:
    with pytest.raises(kl.ExecutionLaneProbeError):
        kl.pin("made-up", "test")


def test_verdict_survives_the_real_pack_unpack_round_trip(
    tmp_path: Path,
) -> None:
    """(e) Through the ACTUAL artifact writer and reader the mint and the
    worker use — not a hand-rolled tar."""
    pytest.importorskip("torch")
    from gen_worker import aot_serve

    verdict = kl.select(
        [_m("baseline+dense", 228.0, 44.1), _m("fused+packed", 350.0, 35.1)],
        device_total_bytes=180 * GB, sm="sm_100")
    content = tmp_path / "work"
    content.mkdir()
    (content / aot_serve.PACKAGE_NAME).write_bytes(b"stub")
    artifact = aot_serve.pack(
        content, tmp_path / "cell.tar.gz",
        {"kind": "aot-inductor", kl.META_KEY: kl.envelope_block(verdict)})

    meta = aot_serve.unpack_metadata(artifact)
    assert meta[kl.META_KEY]["winner"] == "baseline+dense"
    assert kl.adopt_from_artifact(artifact) == "baseline+dense"


# --- the same SM, a different card -----------------------------------------
#
# Cell keys are keyed on SM and the lane is deliberately NOT a key axis, so a
# 96 GB RTX PRO 6000 and a 32 GB RTX 5090 are ONE cell key. Paul: "we cannot
# guarantee that two separate GPUs with the same sm_x capability will both
# want to use the same set of kernels." Serving therefore re-applies the FIT
# half of the rule against its own detected total instead of obeying the
# minting card.

# The workload the PRO 6000 minted: the fast lane is 48 GB resident and the
# small lane is 20 GB. Both fit 96 GB; only one fits 32 GB.
_BIG = _m("baseline+dense", 1063.0, 48.0)
_SMALL = _m("fused+packed", 1240.0, 20.0)


def _minted_on_the_big_card(**extra) -> dict:
    """A cell as the 96 GB card would have packed it."""
    verdict = kl.select(
        [_BIG, _SMALL], device_total_bytes=96 * GB,
        device_name="NVIDIA RTX PRO 6000", sm="sm_120")
    assert verdict.winner == "baseline+dense"  # correct ON THAT CARD
    meta = {"kind": "aot-inductor", kl.META_KEY: kl.envelope_block(verdict)}
    meta.update(extra)
    return meta


def test_a_verdict_from_a_bigger_card_of_the_same_sm_is_refit_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE gap this closes. The 96 GB card measured `baseline+dense` fastest
    and recorded it. The 32 GB 5090 shares the cell key and would have pinned
    a lane whose measured peak cannot fit its card — an OOM, or a silent
    slide into CPU-offload degraded mode, because the kernels' self-checks
    are CORRECTNESS checks and know nothing about memory.

    Serving re-applies the rule here and falls to the fastest recorded
    candidate that does fit, loudly."""
    _on_card(monkeypatch, 32 * GB, "NVIDIA RTX 5090", "sm_120")
    assert kl.adopt(_minted_on_the_big_card()) == "fused+packed"

    execution_lane, reason = kl.pinned()
    assert execution_lane == "fused+packed"
    assert kl.REASON_REFIT_LOCAL in reason
    # It names the recorded winner, this card, and the binding term.
    assert "baseline+dense" in reason
    assert str(32 * GB) in reason and "NVIDIA RTX 5090" in reason
    assert f"binding={kl.BIND_FIT}" in reason


def test_the_fast_path_is_unchanged_when_the_recorded_winner_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same cell, adopted on a card that CAN hold the recorded winner: the
    verdict is obeyed exactly as before, and nothing calls itself a re-fit."""
    _on_card(monkeypatch, 96 * GB, "NVIDIA RTX PRO 6000", "sm_120")
    assert kl.adopt(_minted_on_the_big_card()) == "baseline+dense"

    reason = kl.pinned()[1]
    assert kl.REASON_ADOPTED in reason
    assert kl.REASON_REFIT_LOCAL not in reason
    assert kl.REASON_REFIT_NO_FIT not in reason
    assert kl.REASON_FIT_UNVERIFIED not in reason


def test_nothing_fitting_locally_takes_the_smallest_not_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 16 GB card holds neither lane. Obeying the verdict would OOM; so
    would "fall back to the declared default", because the default carries
    the DENSE modulation and is the LARGER of the two. The smallest recorded
    peak is pinned instead, with its own typed reason, and the degrade path
    owns what happens next."""
    _on_card(monkeypatch, 16 * GB, "NVIDIA RTX 5080", "sm_120")
    assert kl.adopt(_minted_on_the_big_card()) == "fused+packed"

    execution_lane, reason = kl.pinned()
    assert kl.REASON_REFIT_NO_FIT in reason
    assert execution_lane != kl.DEFAULT_EXECUTION_LANE  # the default is the 48 GB lane here
    assert f"binding={kl.BIND_NO_FIT}" in reason


def test_the_recorded_winner_survives_when_it_is_the_one_that_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-regression case: a verdict minted ON a small card names the
    small lane, and adoption on that same class of card must not invent a
    re-fit or drift to anything else."""
    verdict = kl.select(
        [_BIG, _SMALL], device_total_bytes=32 * GB,
        device_name="NVIDIA RTX 5090", sm="sm_120")
    assert verdict.winner == "fused+packed"
    _on_card(monkeypatch, 32 * GB, "NVIDIA RTX 5090", "sm_120")
    meta = {kl.META_KEY: kl.envelope_block(verdict)}
    assert kl.adopt(meta) == "fused+packed"
    assert kl.REASON_ADOPTED in kl.pinned()[1]
    assert kl.REASON_REFIT_LOCAL not in kl.pinned()[1]


def test_full_evidence_reruns_the_whole_rule_against_this_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the caller holds the PUBLISHED metadata, the evidence block has
    both lanes' ms/step AND peaks — so the local re-fit is not a reduced
    rule, it is `select()` itself re-run against this card's total. The
    detail therefore carries the recorded timings, which the envelope-only
    path cannot know."""
    _on_card(monkeypatch, 32 * GB, "NVIDIA RTX 5090", "sm_120")
    # Three lanes, so the local answer is a genuine SPEED decision among the
    # survivors rather than "the one thing left".
    mid = _m("baseline+packed", 1150.0, 22.0)
    verdict = kl.select(
        [_BIG, mid, _SMALL], device_total_bytes=96 * GB, sm="sm_120")
    assert verdict.winner == "baseline+dense"
    meta = json.loads(json.dumps({
        kl.META_KEY: kl.envelope_block(verdict),
        kl.EVIDENCE_KEY: kl.evidence_block(verdict),
    }))
    assert kl.adopt(meta) == "baseline+packed"
    reason = kl.pinned()[1]
    assert kl.REASON_REFIT_LOCAL in reason
    assert kl.EVIDENCE_KEY in reason  # the provenance of the numbers used
    # `select()`'s own speed detail — the envelope-only path has no timings
    # and could not produce this sentence.
    assert f"binding={kl.BIND_SPEED}" in reason
    assert "1150.0" in reason and "1240.0" in reason


def test_a_cell_with_no_recorded_peaks_is_adopted_but_marked_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DECIDED: a cell that records no per-candidate peaks (a verdict minted
    before the fit block existed, or a `sole_candidate` verdict that measured
    nothing) is ADOPTED, not dropped — and it says the fit is unverified
    across cards.

    Falling to the declared default here would not be the conservative
    choice: the default is `baseline+dense`, and DENSE is the larger
    residency of the two modulation values, so "be safe, take the default"
    would trade a possibly-too-big lane for a certainly-bigger one. The
    numerics self-checks still gate arming on the box, and every verdict
    minted from here on carries its peaks."""
    _on_card(monkeypatch, 32 * GB, "NVIDIA RTX 5090", "sm_120")
    meta = _minted_on_the_big_card()
    meta[kl.META_KEY].pop("fit")  # a pre-fit-block pgw#947 cell

    assert kl.adopt(meta) == "baseline+dense"
    reason = kl.pinned()[1]
    assert kl.REASON_ADOPTED in reason
    assert kl.REASON_FIT_UNVERIFIED in reason
    assert "baseline+dense" in reason


def test_a_sole_candidate_verdict_records_no_peaks_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A card with one buildable lane pays for no benchmark, so there is
    nothing to re-fit — and the envelope carries no fit block rather than an
    empty one that would look like a checked constraint."""
    _on_card(monkeypatch, 24 * GB, "NVIDIA L4", "sm_89")
    verdict = kl.sole("baseline+dense", "sm_89 is not Blackwell")
    envelope = kl.envelope_block(verdict)
    assert "fit" not in envelope
    assert kl.adopt({kl.META_KEY: envelope}) == "baseline+dense"
    assert kl.REASON_FIT_UNVERIFIED in kl.pinned()[1]


def test_a_worker_that_cannot_detect_its_card_adopts_and_says_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _on_card(monkeypatch, 0, "", "")
    assert kl.adopt(_minted_on_the_big_card()) == "baseline+dense"
    assert kl.REASON_FIT_UNVERIFIED in kl.pinned()[1]


def test_the_refit_runs_through_the_packed_artifact_the_executor_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end on the path the executor actually calls: a tar on disk,
    read by `adopt_from_artifact`, re-fit against this card."""
    _on_card(monkeypatch, 32 * GB, "NVIDIA RTX 5090", "sm_120")
    artifact = _packed(tmp_path, _minted_on_the_big_card())
    assert kl.adopt_from_artifact(artifact, source="unit") == "fused+packed"
    assert kl.REASON_REFIT_LOCAL in kl.pinned()[1]


def test_recorded_peaks_survive_the_double_mint_byte_compare() -> None:
    """(c) Peak BYTES may ride the packed envelope; raw ones may not.

    The #699 gate compares two mints' `metadata.json` byte for byte. A raw
    `max_memory_allocated()` is a measurement — an autotuned kernel picking a
    different workspace moves it — so the envelope carries the peak QUANTIZED
    up to a coarse grain, which is the same trick `MARGIN_FRACTION` already
    plays on the winner. Here two mints disagree by 40 MB of peak and 3 ms of
    step time and produce byte-identical envelopes."""
    def _mint(peak_jitter: float, ms_jitter: float) -> str:
        verdict = kl.select(
            [_m("baseline+dense", 1063.0 + ms_jitter, 48.0 + peak_jitter),
             _m("fused+packed", 1240.0 - ms_jitter, 20.0 - peak_jitter)],
            device_total_bytes=96 * GB, sm="sm_120")
        return json.dumps(kl.envelope_block(verdict), sort_keys=True)

    assert _mint(0.0, 0.0) == _mint(0.04, 3.0) == _mint(-0.04, -3.0)
    # And the quantization only ever rounds UP, so the re-applied constraint
    # is never more permissive than the measurement it came from.
    assert kl.quantize_peak(int(48.0 * GB)) >= int(48.0 * GB)
    assert kl.quantize_peak(1) == kl.PEAK_QUANTUM_BYTES
    assert kl.quantize_peak(0) == 0


def test_the_fallback_order_is_the_ranking_and_the_winner_leads_it() -> None:
    """The order is what a serving worker falls THROUGH when the winner does
    not fit, so it has to be the rule's own ranking — and it has to be
    discrete, or it could not ride the envelope. Speed ranks it; ties inside
    the margin are ordered by the smaller peak; the recorded winner leads."""
    rows = [
        _m("baseline+dense", 228.0, 44.1),
        _m("baseline+packed", 231.0, 34.6),   # within the 5% margin
        _m("fused+dense", 350.0, 44.6),
        _m("fused+packed", 350.0, 35.1),
    ]
    verdict = kl.select(rows, device_total_bytes=180 * GB, sm="sm_100")
    order = kl.refit_order(rows, winner=verdict.winner)
    assert order[0] == verdict.winner == "baseline+packed"
    assert order == ("baseline+packed", "baseline+dense",
                     "fused+packed", "fused+dense")


def test_the_refit_never_pins_a_execution_lane_this_worker_cannot_implement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cell from a newer worker may record candidates this one has no
    vocabulary for. They are dropped from the re-fit rather than pinned, and
    `pin()` would refuse them anyway."""
    _on_card(monkeypatch, 32 * GB, "NVIDIA RTX 5090", "sm_120")
    meta = _minted_on_the_big_card()
    meta[kl.META_KEY]["fit"]["peaks"]["tcgen05-v2"] = 1
    meta[kl.META_KEY]["fit"]["order"].insert(0, "tcgen05-v2")
    assert kl.adopt(meta) == "fused+packed"
    assert kl.pinned()[0] in kl.EXECUTION_LANES


# --- the mint-side A/B -----------------------------------------------------


def test_mint_probes_every_candidate_and_mints_the_winner_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mint driver end to end with two stubbed lanes of known timing:
    each candidate is loaded onto an empty card and measured, the winner is
    pinned, and the pipeline handed to the exporter is a FRESH load of the
    winner — never a probe pipeline the benchmark already compiled."""
    from gen_worker import aot_mint, mint_child

    timings = {  # the recorded B200 pair
        "baseline+dense": (228.0, int(44.1 * GB)),
        "fused+packed": (350.0, int(35.1 * GB)),
    }
    loads: list[str] = []

    def _load(execution_lane: str):
        loads.append(execution_lane)
        assert kl.pinned()[0] == execution_lane, "the lane must be pinned BEFORE loading"
        # the endpoint INSTANCE rides out with the pipeline — the AOT
        # recipe proves the handler runs before it exports, and the handler is
        # a method on it.
        return f"obj:{execution_lane}", f"pipe:{execution_lane}", f"spec:{execution_lane}"

    monkeypatch.setattr(
        kl, "candidates_here", lambda: ("baseline+dense", "fused+packed"))
    monkeypatch.setattr(
        kl, "device_facts", lambda: (180 * GB, "NVIDIA B200", "sm_100"))
    monkeypatch.setattr(aot_mint, "bench_step", lambda pipe, spec: (pipe, spec))
    monkeypatch.setattr(mint_child, "frame", lambda **kw: None)
    monkeypatch.setattr(mint_child, "_release", lambda: None)

    def _measure(execution_lane: str, step):
        assert step == (f"pipe:{execution_lane}", f"spec:{execution_lane}")
        ms, peak = timings[execution_lane]
        return kl.Measurement(execution_lane=execution_lane, ms_per_step=ms, peak_bytes=peak,
                              samples_ms=(ms,))

    monkeypatch.setattr(kl, "measure", _measure)

    verdict, obj, pipe, spec = mint_child.execution_lane_verdict_for(_load)
    assert verdict.winner == "baseline+dense"
    assert verdict.binding == kl.BIND_SPEED
    assert loads == ["baseline+dense", "fused+packed", "baseline+dense"]
    assert (obj, pipe, spec) == (
        "obj:baseline+dense", "pipe:baseline+dense", "spec:baseline+dense")
    assert kl.pinned()[0] == "baseline+dense"
    # Both lanes' numbers are in the record, not just the winner's.
    assert verdict.measurement("fused+packed").ms_per_step == 350.0
    assert verdict.measurement("baseline+dense").peak_bytes == int(44.1 * GB)


def test_mint_records_a_typed_verdict_when_only_one_execution_lane_can_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A card with no rival lane gets a real verdict, not an absence — and
    pays for no benchmark, because there is nothing to compare."""
    from gen_worker import mint_child

    monkeypatch.setattr(kl, "candidates_here", lambda: ("baseline+dense",))
    monkeypatch.setattr(kl, "candidate_axes", lambda: (
        {kl.AXIS_LINEAR: ("baseline",), kl.AXIS_MODULATION: ("dense",)},
        {kl.AXIS_LINEAR: "sm_89 is not Blackwell",
         kl.AXIS_MODULATION: "triton unavailable"}))
    monkeypatch.setattr(kl, "device_facts", lambda: (24 * GB, "L4", "sm_89"))
    monkeypatch.setattr(mint_child, "frame", lambda **kw: None)
    monkeypatch.setattr(
        kl, "measure",
        lambda *a: pytest.fail("no benchmark for a sole candidate"))

    verdict, _obj, pipe, _spec = mint_child.execution_lane_verdict_for(
        lambda execution_lane: (
            f"obj:{execution_lane}", f"pipe:{execution_lane}",
            f"spec:{execution_lane}"))
    assert verdict.winner == "baseline+dense"
    assert verdict.binding == kl.BIND_SOLE_CANDIDATE
    assert "sm_89 is not Blackwell" in verdict.detail
    assert "triton unavailable" in verdict.detail
    assert pipe == "pipe:baseline+dense"


def test_a_execution_lane_that_cannot_be_built_never_fails_the_mint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker import aot_mint, mint_child

    def _load(execution_lane: str):
        if execution_lane == "fused+packed":
            raise RuntimeError("triton kernels will not compile here")
        return f"obj:{execution_lane}", f"pipe:{execution_lane}", f"spec:{execution_lane}"

    monkeypatch.setattr(
        kl, "candidates_here", lambda: ("baseline+dense", "fused+packed"))
    monkeypatch.setattr(kl, "device_facts", lambda: (180 * GB, "B200", "sm_100"))
    monkeypatch.setattr(aot_mint, "bench_step", lambda pipe, spec: lambda: None)
    monkeypatch.setattr(mint_child, "frame", lambda **kw: None)
    monkeypatch.setattr(mint_child, "_release", lambda: None)
    monkeypatch.setattr(
        kl, "measure",
        lambda execution_lane, step: kl.Measurement(
            execution_lane=execution_lane, ms_per_step=228.0, peak_bytes=int(44.1 * GB)))

    verdict, _obj, pipe, _spec = mint_child.execution_lane_verdict_for(_load)
    assert verdict.winner == "baseline+dense"
    assert ("will not compile here"
            in verdict.measurement("fused+packed").unavailable)
    assert pipe == "pipe:baseline+dense"


# --- the replay ------------------------------------------------------------


def test_replays_the_recorded_campaigns() -> None:
    """THE acceptance: handed the numbers the hand-run campaigns produced,
    the rule independently reaches the answers the two SM tuples were
    hand-edited to encode.

    B200 (pgw#863 run b200-r3, normalized 1024^2/30, COMPILED posture):
    baseline linears 228 ms @ 44.1 GB vs fused linears 350 ms @ 35.1 GB on a
    180 GB card. Speed wins; the ~9 GB the fused linear saves buys nothing
    there.

    5090 (pgw#862 final table, pod mus8neq4kk7b6k, EAGER posture — the only
    posture where BOTH 5090 lanes were measured; the compiled baseline was
    never run there, which is exactly the hole this mechanism closes):
    baseline 1189 ms @ 22.4 GB vs fused 975 ms @ 15.6 GB on a 32 GB card.
    Fused wins on speed, and it would still win if the baseline's 22.4 GB
    peak were excluded on fit.

    Both campaigns varied the LINEAR axis with the modulation axis held
    fixed, because at the time the two shared one switch — which is the
    pgw#863 complaint.
    """
    b200 = kl.select(
        [_m("baseline+packed", 228.0, 44.1), _m("fused+packed", 350.0, 35.1)],
        device_total_bytes=180 * GB, device_name="NVIDIA B200", sm="sm_100")
    assert (b200.winner, b200.binding) == ("baseline+packed", kl.BIND_SPEED)

    rtx5090 = kl.select(
        [_m("baseline+packed", 1189.0, 22.4), _m("fused+packed", 975.0, 15.6)],
        device_total_bytes=32 * GB, device_name="NVIDIA RTX 5090", sm="sm_120")
    assert (rtx5090.winner, rtx5090.binding) == ("fused+packed", kl.BIND_SPEED)

    # And the fit constraint is not decorative on a consumer card: put the
    # same 5090 pair on a 24 GB card and the baseline lane is excluded before
    # speed is ever consulted.
    tight = kl.select(
        [_m("baseline+packed", 1189.0, 22.4), _m("fused+packed", 975.0, 15.6)],
        device_total_bytes=24 * GB, sm="sm_120")
    assert (tight.winner, tight.binding) == ("fused+packed", kl.BIND_FIT)


def test_the_rule_derives_the_pgw863_split_without_a_hand_tuple() -> None:
    """THE pgw#863 acceptance: the answer that needed TWO hand-maintained SM
    tuples — `baseline` linears WITH `packed` modulation on sm_100 — falls
    out of one rule ranking all four combinations, and nobody writes it down.

    The linear pair is measured (run b200-r3 above). The modulation delta is
    the recorded pgw#864/#863 finding — 22.8 -> 13.3 GB transformer-resident
    on B200, i.e. -9.5 GB, and speed-NEUTRAL — applied to each linear lane to
    complete the 2x2 the old single switch could not express. That derivation
    is the whole claim under test: given those numbers, the rule must pick
    the combination pgw#863 had to hand-edit a tuple pair to reach.

    It gets there by the VRAM tiebreak, which is exactly right: the packed
    modulation buys no speed, so it can only win where speed is a tie — and
    it does, on both linear lanes at once.
    """
    packed_saving = 9.5
    four = [
        _m("baseline+dense", 228.0, 44.1),
        _m("baseline+packed", 228.0, 44.1 - packed_saving),
        _m("fused+dense", 350.0, 35.1 + packed_saving),
        _m("fused+packed", 350.0, 35.1),
    ]
    b200 = kl.select(
        four, device_total_bytes=180 * GB, device_name="NVIDIA B200",
        sm="sm_100")
    assert b200.winner == "baseline+packed"
    assert b200.binding == kl.BIND_VRAM_TIEBREAK
    # The 19% step-time penalty the old single switch charged sm_100 for its
    # residency win is simply not paid: the winner is the FAST linear lane.
    assert kl.linear_of(b200.winner) == kl.LINEAR_BASELINE
    assert kl.modulation_of(b200.winner) == kl.MOD_PACKED
    assert b200.winner in kl.EXECUTION_LANES

    # And the residency win is load-bearing, not decorative: shrink the card
    # until neither DENSE lane fits and the packed ones are the only
    # survivors. The winner is unchanged, but it is now the only reason the
    # card can serve this checkpoint at all.
    tight = kl.select(four, device_total_bytes=45 * GB, sm="sm_100")
    assert tight.winner == "baseline+packed"
    assert sorted(
        r.execution_lane for r in four if not kl.fits(r, 45 * GB)
    ) == ["baseline+dense", "fused+dense"]
