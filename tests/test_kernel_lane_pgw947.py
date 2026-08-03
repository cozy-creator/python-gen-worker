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

from gen_worker import kernel_lane as kl

GB = 1000 ** 3  # the benchmark tables are in decimal GB
GiB = 1 << 30


def _m(lane: str, ms: float, peak_gb: float) -> kl.Measurement:
    return kl.Measurement(
        lane=lane, ms_per_step=ms, peak_bytes=int(peak_gb * GB),
        samples_ms=(ms,))


@pytest.fixture(autouse=True)
def _clear_pin():
    kl.clear()
    yield
    kl.clear()


# --- the rule --------------------------------------------------------------


def test_fastest_that_fits_wins_even_when_it_is_the_larger_lane() -> None:
    """(a) Speed is the objective. B200 shape: the baseline lane is 9.5 GB
    BIGGER and 35% faster, and the card has the room — so it wins."""
    verdict = kl.select(
        [_m("baseline+dense", 228.0, 44.1), _m("fused+packed", 350.0, 35.1)],
        device_total_bytes=180 * GB, sm="sm_100")
    assert verdict.winner == "baseline+dense"
    assert verdict.binding == kl.BIND_SPEED
    assert "228.0" in verdict.detail and "350.0" in verdict.detail


def test_a_lane_that_does_not_fit_is_excluded_even_when_faster() -> None:
    """(b) Fit is a CONSTRAINT applied before ranking. Same two lanes, on a
    48 GB card: the baseline's 44.1 GB peak plus its allowance cannot fit, so
    the 35%-slower fused lane wins outright."""
    verdict = kl.select(
        [_m("baseline+dense", 228.0, 44.1), _m("fused+packed", 350.0, 35.1)],
        device_total_bytes=48 * GB, sm="sm_100")
    assert verdict.winner == "fused+packed"
    assert verdict.binding == kl.BIND_FIT
    assert "baseline+dense" in verdict.detail


def test_headroom_allowance_excludes_a_lane_that_bare_peak_would_admit() -> None:
    """The allowance does real work: a 40 GB peak fits a 44 GB card by bare
    measurement and does NOT fit once the activation-spike + fragmentation
    allowance is applied — which is the point of stating one."""
    row = _m("baseline+dense", 100.0, 40.0)
    assert row.peak_bytes < 44 * GB
    assert not kl.fits(row, 44 * GB)
    assert row.required_bytes() == int(40 * GB * 1.20) + GiB


def test_within_margin_ties_fall_to_the_smaller_lane() -> None:
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
         kl.Measurement(lane="fused+packed", unavailable="triton compile failed")],
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
    assert verdict.winner == kl.DEFAULT_LANE
    assert verdict.binding == kl.BIND_NO_FIT


# --- the probe loop --------------------------------------------------------


def test_probe_measures_every_candidate_and_a_build_failure_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The A/B harness end to end with a stubbed timer: both candidates are
    built, one blows up, the survivor wins and the failure is recorded."""
    timings = {"baseline+dense": (250.0, 40 * GB),
               "fused+packed": (300.0, 30 * GB)}

    def _measure(lane: str, step):
        step()
        ms, peak = timings[lane]
        return kl.Measurement(lane=lane, ms_per_step=ms, peak_bytes=peak,
                              samples_ms=(ms,))

    monkeypatch.setattr(kl, "measure", _measure)
    built = []

    def _build(lane: str):
        if lane == "fused+dense":
            raise RuntimeError("no kernels here")
        built.append(lane)
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
    }
    # No wall clocks, no measurements, JSON-clean.
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


def test_serving_adopts_the_lane_the_cell_names(tmp_path: Path) -> None:
    verdict = kl.select(
        [_m("baseline+dense", 1189.0, 22.4), _m("fused+packed", 975.0, 15.6)],
        device_total_bytes=32 * GB, sm="sm_120")
    artifact = _packed(tmp_path, {
        "kind": "aot-inductor",
        kl.META_KEY: kl.envelope_block(verdict),
    })
    assert kl.adopt_from_artifact(artifact) == "fused+packed"
    lane, reason = kl.pinned()
    assert lane == "fused+packed"
    assert kl.REASON_ADOPTED in reason


def test_cell_without_a_verdict_is_the_default_with_a_typed_reason(
    tmp_path: Path,
) -> None:
    """(d) A pre-pgw#947 cell records nothing. That is the declared default
    and it SAYS which case it was — never a silent fall-through."""
    artifact = _packed(tmp_path, {"kind": "aot-inductor"})
    assert kl.adopt_from_artifact(artifact) == kl.DEFAULT_LANE
    assert kl.REASON_ABSENT in kl.pinned()[1]


def test_no_cell_at_all_is_the_default_with_its_own_reason() -> None:
    assert kl.adopt(None) == kl.DEFAULT_LANE
    assert kl.REASON_NO_CELL in kl.pinned()[1]


def test_unreadable_artifact_degrades_and_names_the_failure(
    tmp_path: Path,
) -> None:
    junk = tmp_path / "not-a-tar.tar.gz"
    junk.write_bytes(b"definitely not a tarball")
    assert kl.adopt_from_artifact(junk) == kl.DEFAULT_LANE
    assert kl.REASON_UNREADABLE in kl.pinned()[1]


def test_a_lane_this_worker_does_not_implement_is_refused_by_name() -> None:
    lane, reason = kl.lane_from_metadata(
        {kl.META_KEY: {"winner": "tcgen05-v2", "binding": "speed"}})
    assert lane == kl.DEFAULT_LANE
    assert kl.REASON_UNKNOWN_LANE in reason


def test_pin_refuses_a_lane_outside_the_vocabulary() -> None:
    with pytest.raises(kl.LaneProbeError):
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

    def _load(lane: str):
        loads.append(lane)
        assert kl.pinned()[0] == lane, "the lane must be pinned BEFORE loading"
        return f"pipe:{lane}", f"spec:{lane}"

    monkeypatch.setattr(
        kl, "candidates_here", lambda: ("baseline+dense", "fused+packed"))
    monkeypatch.setattr(
        kl, "device_facts", lambda: (180 * GB, "NVIDIA B200", "sm_100"))
    monkeypatch.setattr(aot_mint, "bench_step", lambda pipe, spec: (pipe, spec))
    monkeypatch.setattr(mint_child, "frame", lambda **kw: None)
    monkeypatch.setattr(mint_child, "_release", lambda: None)

    def _measure(lane: str, step):
        assert step == (f"pipe:{lane}", f"spec:{lane}")
        ms, peak = timings[lane]
        return kl.Measurement(lane=lane, ms_per_step=ms, peak_bytes=peak,
                              samples_ms=(ms,))

    monkeypatch.setattr(kl, "measure", _measure)

    verdict, pipe, spec = mint_child.lane_verdict_for(_load)
    assert verdict.winner == "baseline+dense"
    assert verdict.binding == kl.BIND_SPEED
    assert loads == ["baseline+dense", "fused+packed", "baseline+dense"]
    assert (pipe, spec) == ("pipe:baseline+dense", "spec:baseline+dense")
    assert kl.pinned()[0] == "baseline+dense"
    # Both lanes' numbers are in the record, not just the winner's.
    assert verdict.measurement("fused+packed").ms_per_step == 350.0
    assert verdict.measurement("baseline+dense").peak_bytes == int(44.1 * GB)


def test_mint_records_a_typed_verdict_when_only_one_lane_can_be_built(
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

    verdict, pipe, _spec = mint_child.lane_verdict_for(
        lambda lane: (f"pipe:{lane}", f"spec:{lane}"))
    assert verdict.winner == "baseline+dense"
    assert verdict.binding == kl.BIND_SOLE_CANDIDATE
    assert "sm_89 is not Blackwell" in verdict.detail
    assert "triton unavailable" in verdict.detail
    assert pipe == "pipe:baseline+dense"


def test_a_lane_that_cannot_be_built_never_fails_the_mint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker import aot_mint, mint_child

    def _load(lane: str):
        if lane == "fused+packed":
            raise RuntimeError("triton kernels will not compile here")
        return f"pipe:{lane}", f"spec:{lane}"

    monkeypatch.setattr(
        kl, "candidates_here", lambda: ("baseline+dense", "fused+packed"))
    monkeypatch.setattr(kl, "device_facts", lambda: (180 * GB, "B200", "sm_100"))
    monkeypatch.setattr(aot_mint, "bench_step", lambda pipe, spec: lambda: None)
    monkeypatch.setattr(mint_child, "frame", lambda **kw: None)
    monkeypatch.setattr(mint_child, "_release", lambda: None)
    monkeypatch.setattr(
        kl, "measure",
        lambda lane, step: kl.Measurement(
            lane=lane, ms_per_step=228.0, peak_bytes=int(44.1 * GB)))

    verdict, pipe, _spec = mint_child.lane_verdict_for(_load)
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
    assert b200.winner in kl.LANES

    # And the residency win is load-bearing, not decorative: shrink the card
    # until neither DENSE lane fits and the packed ones are the only
    # survivors. The winner is unchanged, but it is now the only reason the
    # card can serve this checkpoint at all.
    tight = kl.select(four, device_total_bytes=45 * GB, sm="sm_100")
    assert tight.winner == "baseline+packed"
    assert sorted(
        r.lane for r in four if not kl.fits(r, 45 * GB)
    ) == ["baseline+dense", "fused+dense"]
