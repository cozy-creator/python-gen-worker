"""pgw#848: an ABANDONED mint discarded 29 minutes of measurement.

Attempt sixteen ran on the `8559140` cap fix and proved it — 29 minutes with
no memory error, where attempts 14 and 15 died at +11.6 and +11.0 min against
the 11.09 GiB ceiling. Then the worker's endpoint instances were torn down
under the drain path and the mint was abandoned, and the ENTIRE phase table
for those 29 minutes was one row::

    status=abandoned total_s=1741.33 — no cell produced

**Zero `entry:` rows. No `pool` row.** K, its binding constraint, every
per-entry timing and every peak were measured and thrown away, and the
K-and-binding answer had to be re-bought with another pod.

The mechanism: `report.json` is written ONCE, at a terminus the child reaches
under its own power. A child that is group-killed reaches no terminus, raises
nothing, and writes nothing — so `f9c1b2d`'s work on the *aborted* path (the
failed attempt teaches the retry instead of discarding what it measured) never
applied here. Same code, different exit.

The fix is a snapshot on disk, rewritten atomically on every beat, because a
file is the only thing that survives a signal — the same principle the pgw#848
resume design keys on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker import aot_compile_pool as pool
from gen_worker import aot_mint, mint_delegate, mint_process
from gen_worker.cell_adopt import AdoptOutcome

_GIB = 1 << 30


def _progress_midflight(tmp_path: Path) -> aot_mint.MintProgress:
    """A mint that has finished some entries and is inside another — exactly
    the state attempt sixteen was killed in."""
    width = pool.entry_workers(
        36, vcpus=16, available_bytes=64 * _GIB, free_vram_bytes=0,
        device_lock=True, limit=4)
    progress = aot_mint.MintProgress()
    progress.t_mint = 0.0
    progress.width = width
    progress.timings.update({"export_all_s": 61.2})
    progress.pool_ledger = {
        "pool_workers": width.workers, "pool_efficiency": 0.97,
        "peak_child_rss_bytes": 3 * _GIB, "peak_concurrency": width.workers,
    }
    progress.at = {
        "phase": aot_mint.PHASE_INDUCTOR_COMPILE, "step": 30, "total": 36,
        "note": "unet/adapter=true/1024x1024"}
    return progress


def test_a_killed_mints_measurements_are_on_disk_before_it_dies(
    tmp_path: Path,
) -> None:
    """The snapshot exists, is complete, and is written atomically.

    Atomicity is not decoration: the parent reads this file the instant after
    it kills the child, and a half-written table is a table nobody can use.
    """
    snapshot = tmp_path / mint_process.PHASES_SNAPSHOT_NAME
    progress = _progress_midflight(tmp_path)

    aot_mint.write_phase_snapshot(snapshot, progress)
    table = json.loads(snapshot.read_text())

    assert table["pool"]["entry_workers"] == progress.width.workers
    assert table["pool"]["binding"] == progress.width.binding
    assert table["pool"]["peak_child_rss_bytes"] == 3 * _GIB
    assert table["at"]["step"] == 30, (
        "the entry a mint DIES ON is the one a reader most needs named")
    assert table["terminus"] == "in_flight"
    assert not list(tmp_path.glob("*.tmp")), (
        "the atomic write must leave no temp file behind")


def test_nothing_measured_writes_nothing(tmp_path: Path) -> None:
    """"No measurement" and "zero" must not read the same."""
    snapshot = tmp_path / mint_process.PHASES_SNAPSHOT_NAME
    aot_mint.write_phase_snapshot(snapshot, aot_mint.MintProgress())
    assert not snapshot.exists()
    assert aot_mint.partial_phase_table(aot_mint.MintProgress()) == {}


def test_an_abandoned_outcome_emits_the_rows_it_measured(
    tmp_path: Path,
) -> None:
    """THE REGRESSION, over the real parent-side relay.

    A child that wrote no report at all — which is every abandoned and every
    killed mint — must still put its entry rows and its pool row on the wire.
    """
    snapshot = tmp_path / mint_process.PHASES_SNAPSHOT_NAME
    progress = _progress_midflight(tmp_path)
    aot_mint.write_phase_snapshot(snapshot, progress)

    request = mint_process.MintRequest(
        function="f", modules=(), family="sdxl", cell_key="k",
        target=str(tmp_path / "cell.tar.gz"), capture=str(tmp_path),
        report=str(tmp_path / "report.json"),
        cfg=mint_process.CompileCellSpec(),
        phases_snapshot=str(snapshot))
    recovered = mint_process._read_phase_snapshot(request.phases_snapshot)
    assert recovered, "the parent could not read what the child wrote"

    outcome = mint_process.MintOutcome(
        status=mint_process.ABANDONED,
        detail="background mint abandoned (shutdown: worker shutdown)",
        report=None, elapsed_s=1741.33, partial_phases=recovered)

    emitted: list[Dict[str, Any]] = []

    def _capture(**kwargs: Any) -> None:
        emitted.append(kwargs)

    original = aot_mint.emit_phase_events
    aot_mint.emit_phase_events = _capture  # type: ignore[assignment]
    try:
        mint_delegate._emit_aot_phases(outcome, family="sdxl", execution_lane="w8a8")
    finally:
        aot_mint.emit_phase_events = original  # type: ignore[assignment]

    assert emitted, (
        "an abandoned mint emitted NO phase table — this is attempt sixteen, "
        "29 minutes reported as one row")
    table = emitted[0]["table"]
    assert table["pool"]["entry_workers"] == progress.width.workers, (
        "the K-and-binding answer is the one the coordinator had to re-buy "
        "with another pod")
    assert table["pool"]["binding"] == progress.width.binding
    assert emitted[0]["terminus"] == "abandoned", (
        "an abandoned mint must not be relabelled as an ordinary abort — the "
        "cause is a co-tenancy decision, not a mint failure")
    assert table["recovered_from"] == "phase_snapshot", (
        "a recovered table must say it was recovered; a reader must never "
        "mistake it for one the child wrote at its own terminus")


def test_a_report_beats_a_snapshot_when_both_exist(tmp_path: Path) -> None:
    """The child reaching its own terminus is better evidence than the last
    beat before it got there. The snapshot is a fallback, never an override."""
    outcome = mint_process.MintOutcome(
        status=mint_process.REFUSED, elapsed_s=10.0,
        report=mint_process.MintReport(
            status="refused", elapsed_s=10.0,
            mint_phases={"v": 1, "terminus": "aborted",
                         "pool": {"entry_workers": 7}}),
        partial_phases={"v": 1, "pool": {"entry_workers": 99}})

    emitted: list[Dict[str, Any]] = []
    original = aot_mint.emit_phase_events
    aot_mint.emit_phase_events = (
        lambda **kw: emitted.append(kw))  # type: ignore[assignment]
    try:
        mint_delegate._emit_aot_phases(outcome, family="sdxl", execution_lane="w8a8")
    finally:
        aot_mint.emit_phase_events = original  # type: ignore[assignment]

    assert emitted[0]["table"]["pool"]["entry_workers"] == 7
    assert "recovered_from" not in emitted[0]["table"]


def test_the_snapshot_path_reaches_the_child(tmp_path: Path) -> None:
    """The wiring. Every measurement above is worthless if the child is never
    told where to write."""
    import inspect

    assert "phases_snapshot=str(" in inspect.getsource(
        mint_delegate.build_request)
    from gen_worker import mint_child

    assert "phase_snapshot=(" in inspect.getsource(mint_child._mint_aot)


def test_an_unreadable_snapshot_never_changes_an_outcome(
    tmp_path: Path,
) -> None:
    """Telemetry must not be able to fail a mint, in either direction."""
    assert mint_process._read_phase_snapshot("") == {}
    assert mint_process._read_phase_snapshot(str(tmp_path / "nope")) == {}
    junk = tmp_path / "junk.json"
    junk.write_text("{not json")
    assert mint_process._read_phase_snapshot(str(junk)) == {}
    listy = tmp_path / "listy.json"
    listy.write_text("[1, 2, 3]")
    assert mint_process._read_phase_snapshot(str(listy)) == {}


def test_the_retry_decision_is_untouched_by_recovered_telemetry() -> None:
    """`retryable` branches on ``report is None``. Carrying the recovered
    table in a SEPARATE field rather than a synthesized report is what keeps
    a telemetry fix from silently changing a retry policy."""
    crashed = mint_process.MintOutcome(
        status=mint_process.CRASHED, partial_phases={"v": 1})
    assert crashed.report is None
    assert crashed.retryable is True
    abandoned = mint_process.MintOutcome(
        status=mint_process.ABANDONED, partial_phases={"v": 1})
    assert abandoned.retryable is False, (
        "abandonment is not a failure and must never be retried into a "
        "second billed compile")


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_pool_ledger_is_live_not_end_of_run(tmp_path: Path) -> None:
    """A ledger written only at the end is a ledger an abandoned mint never
    gets — so this asserts it BEHAVIOURALLY, from inside the run.

    Was a source-text assertion, and that was the wrong instrument twice over:
    a wiring claim proven by reading the file is not proven, and
    `inspect.getsource` goes stale the moment the file is edited under a
    running session (which is exactly how it failed — I edited `aot_mint.py`
    mid-gate). This drives the REAL `_compile_entries_parallel` over a REAL
    two-entry pool and reads `progress.pool_ledger` from inside the per-entry
    callback: if it is already populated when entry 1 lands, the ledger is
    live, and no amount of refactoring can make that pass falsely.
    """
    import torch

    from gen_worker.aot_mint import _MintedEntry

    class Tiny(torch.nn.Module):
        def __init__(self, seed: int) -> None:
            super().__init__()
            self.a = torch.nn.Linear(64, 64)
            self.seed = seed

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.a(x)) * (1.0 + self.seed)

    minted = [
        _MintedEntry(
            name=f"unet/row={i}", spec=None, module=None, owner=None,
            program=torch.export.export(Tiny(i), (torch.randn(4, 64),)),
            input_names=(), flat_names=(), files=[], timings={})
        for i in range(2)
    ]
    width = pool.entry_workers(
        2, vcpus=16, available_bytes=64 * _GIB, free_vram_bytes=0,
        device_lock=True, limit=2)
    progress = aot_mint.MintProgress()
    seen: list[Dict[str, Any]] = []

    aot_mint._compile_entries_parallel(
        minted, tmp_path / "work", width, progress=progress,
        inductor_configs={"compile_threads": 2},
        on_entry=lambda name, done, total: seen.append(
            dict(progress.pool_ledger)))

    assert seen, "the pool never reported a completed entry"
    assert seen[0], (
        "the ledger was still EMPTY when the first entry landed — it is being "
        "written at the end of the run, which is the one moment an abandoned "
        "mint never reaches")
    assert seen[0].get("pool_workers") == width.workers
    assert "peak_child_rss_bytes" in seen[0]


# ---------------------------------------------------------------------------
# pgw#848 long-fuse sweep: the pod-side reaper's progress signal had no producer
# ---------------------------------------------------------------------------


def test_the_mint_feeds_the_pod_side_reapers_progress_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """podguard's own docstring: both its layers "kill on liveness +
    progress-staleness" — Paul's rule, implemented. The pod-side layer reads a
    token file that `podguard-progress` writes, and **nothing in the SDK has
    ever written it** (zero references to podguard in gen_worker).

    So the pod-side progress path had no producer. SCOPE: `PODGUARD_STATE` is
    injected by `podguard.arm()`, which runs only when podguard creates the
    pod — so this is live on lane-rented pods and INERT on hub-created ones,
    which is most of them and will include forge pods. It did not cause
    attempt sixteen (verdict UNREACHABLE -> reaped on renter liveness alone)
    and would not have prevented it; it closes the gap wherever the oracle is
    reachable, and CP5's correction records what closes the rest.
    """
    state = tmp_path / "podguard"
    monkeypatch.setenv(aot_mint.PODGUARD_STATE_ENV, str(state))

    aot_mint._touch_pod_progress("aot_mint inductor_compile 3/36 unet/x")
    token_a = (state / "progress").read_text()
    aot_mint._touch_pod_progress("aot_mint inductor_compile 4/36 unet/y")
    token_b = (state / "progress").read_text()

    assert "3/36" in token_a and "4/36" in token_b
    assert token_a != token_b, (
        "the watchdog compares the token's CONTENT, so a value that does not "
        "change reads as NO progress however often the file is rewritten")


def test_the_progress_signal_is_inert_off_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset everywhere but a podguard-rented pod, and a mint must never fail
    because a telemetry file could not be written."""
    monkeypatch.delenv(aot_mint.PODGUARD_STATE_ENV, raising=False)
    aot_mint._touch_pod_progress("nothing should happen")
    assert not list(tmp_path.iterdir())

    # An unwritable state dir is survivable, not fatal.
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory")
    monkeypatch.setenv(aot_mint.PODGUARD_STATE_ENV, str(blocked))
    aot_mint._touch_pod_progress("still must not raise")


def test_every_mint_beat_feeds_both_survivors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two things that must outlive a killed mint are fed by the SAME
    beat: the phase snapshot (what it measured) and the pod-side progress
    token (that it was working). Neither may depend on the other running.

    Was `inspect.getsource(aot_mint.mint)`, and that failed TWICE in a
    release gate — once because I edited the file mid-run, once because a
    SIBLING LANE did. That is the finding, not the accident: on a shared
    chaos worktree the source file is not a stable object, so a source-text
    assertion tests the file rather than the behaviour and can go red without
    the behaviour changing. Driven through the real `mint()` entrypoint
    instead: `_mint_cell` is replaced with one that beats once and raises, so
    the REAL beat wrapper `mint()` installs is what runs.
    """
    snapshot = tmp_path / mint_process.PHASES_SNAPSHOT_NAME
    state = tmp_path / "podguard"
    monkeypatch.setenv(aot_mint.PODGUARD_STATE_ENV, str(state))

    def _one_beat(pipeline, spec, out_dir, **kw):  # type: ignore[no-untyped-def]
        progress = kw["progress"]
        progress.width = pool.entry_workers(
            2, vcpus=16, available_bytes=64 * _GIB, free_vram_bytes=0,
            device_lock=True, limit=2)
        progress.timings["export_all_s"] = 1.0
        progress.beat(aot_mint.PHASE_INDUCTOR_COMPILE, 1, 36, "unet/row=0")
        raise aot_mint.MintRefused("stop here — the beat is what is under test")

    monkeypatch.setattr(aot_mint, "_mint_cell", _one_beat)
    with pytest.raises(aot_mint.MintRefused):
        aot_mint.mint(None, None, tmp_path / "out", phase_snapshot=snapshot)

    assert snapshot.exists(), (
        "the beat did not write the phase snapshot — a killed mint keeps "
        "nothing it measured")
    assert (state / "progress").exists(), (
        "the beat did not touch the pod-side progress token — the reaper is "
        "told nothing about work that is happening")
    assert "1/36" in (state / "progress").read_text()


# ---------------------------------------------------------------------------
# pgw#848 CP10: the worker never looked at its own credential's expiry
# ---------------------------------------------------------------------------


def _jwt(exp: float) -> str:
    import base64
    import json as _json

    def seg(d: Any) -> str:
        raw = _json.dumps(d).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{seg({'alg': 'none'})}.{seg({'exp': int(exp)})}.x"


def _transport_probe(token: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Drive the REAL `_report_credential_age` and capture what it said."""

    from gen_worker import transport as tp

    said: list[tuple[str, str]] = []

    class _Probe:
        _last_credential_left = None
        _report_credential_age = tp.Transport._report_credential_age

    import gen_worker.activity as activity_mod

    real = activity_mod.emit_event
    activity_mod.emit_event = (  # type: ignore[assignment]
        lambda kind, detail, **kw: said.append((kind, detail)))
    logs: list[str] = []
    try:
        _Probe()._report_credential_age(token)  # type: ignore[arg-type]
    finally:
        activity_mod.emit_event = real  # type: ignore[assignment]
    return logs, said


def test_an_expiring_worker_jwt_is_announced_before_the_pod_dies() -> None:
    """MEASURED (hub pod_events, attempts 16 and 17): the worker JWT expired
    at T+32.4 and T+31.2 minutes and the worker said NOTHING. Ten minutes of
    silence later the hub recorded "silent death mid-activity" and destroyed a
    mint it described as reporting fresh progress.

    The worker can always know this locally — it holds the token and the token
    carries `exp`. It just never looked.
    """
    import time as _time

    _, said = _transport_probe(_jwt(_time.time() + 120))
    assert said, "an expiring credential produced no typed event"
    kind, detail = said[0]
    assert kind == "worker_credential"
    assert "worker_jwt_expiring" in detail

    _, said = _transport_probe(_jwt(_time.time() - 90))
    assert "worker_jwt_expired" in said[0][1]


def test_a_healthy_credential_is_silent() -> None:
    """A warning every connect would be noise, and noise is how the real one
    gets missed."""
    import time as _time

    _, said = _transport_probe(_jwt(_time.time() + 3600))
    assert said == []
    # An unreadable or exp-less token is not an event either.
    _, said = _transport_probe("not-a-jwt")
    assert said == []


def test_every_hub_dial_reads_ONE_refreshable_credential() -> None:
    """pgw#848: the defect was a SPLIT, not a value.

    `transport._worker_jwt` is rotated; `Settings.worker_jwt` is frozen at pod
    create and updated by nothing — and the attestation carrier, which opens
    its OWN gRPC Connect every 300 s forever, read the frozen one. Past T+30
    min every one of those dials is a fresh `worker_token_expired`; three
    wedge the pod. That is what killed attempts 16 and 17, with the scheduler
    stream healthy throughout.

    Not a mint-only concern: ANY worker that reports past its TTL burns
    strikes. A mint is just the first workload that reliably lives that long.
    """
    import inspect

    from gen_worker import hardware_report, worker_credential

    worker_credential.reset()
    try:
        # Before any rotation: the boot token is the honest answer.
        assert worker_credential.current() == "" or isinstance(
            worker_credential.current(), str)
        worker_credential.install("rotated-token", 1234.0)
        assert worker_credential.current() == "rotated-token"
        assert worker_credential.expires_at() == 1234.0
        # An empty install must not erase a good credential.
        worker_credential.install("")
        assert worker_credential.current() == "rotated-token"

        # The dialer must READ that source, not the frozen settings value.
        # pgw#848 follow-up (Paul: "why would we have two credentials?"): the
        # fallback is GONE, not merely deprioritised, and the settings field is
        # renamed so a stale read is an AttributeError rather than a stale
        # string. There is one source now.
        src = inspect.getsource(hardware_report)
        assert src.count("token = worker_credential.current()") == 2
        assert "settings.worker_jwt" not in src
    finally:
        worker_credential.reset()


# ---------------------------------------------------------------------------
# pgw#848 CP12 -> pgw#868: the honest ABSENCE has been replaced by the GATE
# ---------------------------------------------------------------------------


def test_the_unchecked_announcement_is_gone_because_the_gate_landed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """CP12 shipped `cell_numerics phase=unchecked` — an arm that stated, on
    the wire, that nobody had checked it. That was the right stopgap and the
    wrong end state, and pgw#868 replaced it with the measurement.

    Two things are pinned here so the stopgap cannot creep back:

    * `_announce_unchecked_numerics` no longer exists. A cell that armed while
      announcing it was unchecked is exactly what the gate refuses now.
    * `arm_aot` FAILS CLOSED when there is nothing to measure. The old shape
      of this test drove `arm_aot` with a stub `enable` and asserted True;
      today that same call arms NOTHING (no marker, no runner, no eager
      reference), so it must refuse — `phase=unmeasurable`, never `unchecked`.

    The gate's own behaviour lives in `tests/test_numerics_gate_pgw868.py`,
    which drives the real arm path against a real packed artifact.
    """
    import gen_worker.activity as activity_mod
    from gen_worker import aot_serve
    from gen_worker.models import provision

    assert not hasattr(provision, "_announce_unchecked_numerics")

    monkeypatch.setattr(
        aot_serve, "enable", lambda *a, **k: AdoptOutcome.hit())
    said: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, **kw: said.append(
            (kind, detail, str(kw.get("phase", "")))))

    cfg = type("Cfg", (), {"family": "sdxl", "numerics_floor": 0.995,
                           "numerics_warn": 0.999, "lora_bucket": 0,
                           "targets": ()})()
    assert provision.arm_aot(
        object(), cfg, None, Path("cell.pt2"), 0).armed is False
    rows = [(d, p) for k, d, p in said if k == activity_mod.KIND_CELL_NUMERICS]
    assert rows, "an arm that could not be measured said nothing"
    detail, phase = rows[-1]
    assert phase == "unmeasurable"
    assert "unchecked" not in [p for _d, p in rows]
    assert "not a pass" in detail




def test_exactly_ONE_module_may_read_the_bootstrap_credential() -> None:
    """pgw#848 / Paul: *"why would we have two credentials? that makes no
    sense."* It was never two credentials — it was ONE with two storage
    locations and no source of truth, and nothing ever superseded the boot copy
    once rotation began.

    The rename makes an accidental read an AttributeError rather than a stale
    string, which is the "make it impossible, not detectable" half. This is the
    other half: a reader-role check, because the rule is sharp — exactly one
    module may read the field to AUTHENTICATE, and everything else is a bug.

    pgw#849's unreached-surface guard is structurally blind to this class: the
    field IS read, and by production code. Only a role check catches it.
    """
    root = Path(pool.PACKAGE_ROOT) / "gen_worker"
    allowed = {
        # the single source of truth
        "worker_credential.py": "owns the field; every other reader goes through it",
        # identity-only reads: `sub`/`release_id`, which rotation never changes,
        # taken before any stream exists. Justified in-line at both sites.
        "lifecycle.py": "boot identity claims",
        "procsplit/parent.py": "boot identity claims",
        # the loader maps the env name onto the field
        "config/loader.py": "env->field mapping",
        "config/settings.py": "the declaration itself",
    }
    offenders = []
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        if rel in allowed:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if "bootstrap_worker_jwt" in code:
                offenders.append(f"{rel}:{n}: {line.strip()[:90]}")
    assert not offenders, (
        "these read the BOOTSTRAP credential directly — it is frozen at pod "
        "create and updated by nothing, so a long-lived pod authenticates with "
        "a dead token while other paths work fine (pgw#846 attempts 16/17). "
        "Read `worker_credential.current()`:\n  " + "\n  ".join(offenders))


def test_the_old_name_is_gone_so_a_stale_read_cannot_compile() -> None:
    """The rename is the enforcement. If `Settings.worker_jwt` still existed,
    every one of the readers fixed above would still be *valid Python* that
    silently returns a frozen token — which is exactly how the attestation
    carrier acquired it."""
    from gen_worker.config.settings import Settings

    assert not hasattr(Settings(), "worker_jwt"), (
        "the old field name is back — a call site can read the frozen token "
        "again and nothing will refuse it")
    assert hasattr(Settings(), "bootstrap_worker_jwt")


def test_no_reader_defeats_the_rename_with_a_getattr_default() -> None:
    """pgw#876 §2. The rename makes a stale read raise — UNLESS the reader
    supplies a default, and two did, for months, undetected.

    `getattr(settings, "worker_jwt", "")` is valid Python that returns "" now
    that the field is gone, so the AttributeError never happens and the caller
    reads an empty credential instead of a frozen one. `executor.py` built its
    whole `worker_jwt_provider` this way, and `aot_mint._publisher_from_settings`
    used it to authorize `--publish`. The sibling sweep above could not see
    either one: it searches for the NEW name, and these name the OLD one.
    """
    root = Path(pool.PACKAGE_ROOT) / "gen_worker"
    offenders = []
    pattern = re.compile(r"""getattr\(\s*[A-Za-z_.]*settings\s*,\s*["']worker_jwt["']""")
    for path in sorted(root.rglob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if pattern.search(code):
                offenders.append(
                    f"{path.relative_to(root)}:{n}: {line.strip()[:90]}")
    assert not offenders, (
        "these defeat the pgw#848 rename with a getattr default — the field "
        "does not exist, so each silently yields the empty string instead of "
        "raising. Read `worker_credential.current()`:\n  "
        + "\n  ".join(offenders))
