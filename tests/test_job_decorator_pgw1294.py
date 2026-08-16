"""pgw#1294 ⇄ th#2049 (JOBS program, issue 2 of 8): ``@job`` — run-once
functions with a PORTABLE body, a merged ``JobContext``, and liveness that is
POSITION, not pulse.

What each section proves, and the one-line edit that turns it RED:

1. **Portability is a tested requirement, not a style.** ONE function object
   carries BOTH declarations and runs green under BOTH harnesses — the real
   ``gen-worker run`` endpoint CLI and the real ``gen-worker job run`` job CLI,
   both driving ``cli.main`` end to end against a package on disk. RED by
   changing either decorator's ``(ctx, payload) -> Struct`` contract, or by
   giving ``JobContext`` less than the producer-endpoint context has.
2. **The run-once lifecycle is enforced at DECORATION.** A class, a method, a
   generator, the wrong arity, an unannotated payload or return — each is a
   typed refusal naming the rule. RED by deleting the matching branch in
   ``api/jobs._validate_job_shape``.
3. **The hub-write declaration is ONE surface for both decorators.** An
   ``@endpoint`` function and a ``@job`` each calling ``save_checkpoint``
   without ``publishes=True`` get ``PublishNotDeclaredError``; the SAME bodies
   with the declaration get past the gate. RED by deleting either
   ``_require_publish_declaration`` call site.
4. **Position is monotonic and load-bearing.** A position that goes backwards
   in a phase raises; a new phase restarts the count. RED by deleting the
   comparison in ``RequestContext._advance_position``.
5. **pgw#1287's class cannot bank as a success.** A transfer loop that reports
   NOTHING for its whole duration fails the job with
   ``JobProgressStalledError``; the byte-identical loop that reports position
   succeeds. RED by deleting ``ProgressWatch.check()`` from ``execute_job`` —
   the stalled run then returns "succeeded", which is exactly the defect
   th#2014 measured against a pod that was downloading correctly.
6. **The manifest carries jobs beside functions, and ``publishes`` on both row
   shapes, deterministically.** RED by dropping the ``jobs`` block, by making
   ``publishes`` omit-when-false, or by not sorting the block.

Everything here drives real production code: the real decorators, the real
registry walk, the real discovery manifest builder, the real
``jobs.execute_job`` harness and the real CLI entry point. The only doubles are
a temp package on disk and a payload.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict

import msgspec
import pytest

from gen_worker import JobContext, Resources, endpoint, job
from gen_worker.api.errors import (
    JobProgressStalledError,
    NonMonotonicProgressError,
    PublishNotDeclaredError,
)
from gen_worker.jobs import (
    DEFAULT_PHASE_BUDGET_S, JobDispatch, ProgressWatch, execute_job,
)
from gen_worker.registry import extract_job_spec, extract_specs
from harness.progress_wait import Cadence, await_progress


class BakeIn(msgspec.Struct):
    rung: str = "w8a8"


class BakeOut(msgspec.Struct):
    rung: str
    ctx_class: str


# ---- 1. ONE body, BOTH declarations ---------------------------------------
# Stacked deliberately: `@job` and `@endpoint` each attach their own marker to
# the SAME function object, so this is not two bodies that happen to agree —
# it is one body, and `bake_both is spec.method` below asserts exactly that.
@job(publishes=True, resources=Resources(vcpus=2))
@endpoint(kind="conversion", publishes=True, name="bake")
def bake_both(ctx: JobContext, spec: BakeIn) -> BakeOut:
    ctx.progress(position=1, total=2, phase="bake")
    ctx.metric({"cosine": 0.999}, step=1, total=2)
    ctx.progress(position=2, total=2, phase="bake")
    return BakeOut(rung=spec.rung, ctx_class=type(ctx).__mro__[1].__name__)


def test_one_body_carries_both_declarations() -> None:
    """The portability property, stated on the objects themselves."""
    job_spec = extract_job_spec(bake_both)
    (endpoint_spec,) = extract_specs(bake_both)
    assert job_spec is not None
    # Same function object under both harnesses — not a copy, not a wrapper.
    assert job_spec.method is bake_both
    assert endpoint_spec.method is bake_both
    assert job_spec.payload_type is endpoint_spec.payload_type is BakeIn
    assert job_spec.output_type is endpoint_spec.output_type is BakeOut
    assert job_spec.publishes and endpoint_spec.publishes


def test_job_context_is_a_superset_of_the_producer_endpoint_context() -> None:
    """Promotion must be a redeploy, not a rewrite: every name a producer
    endpoint handler may use has to exist on JobContext, under that name."""
    from gen_worker.request_context import ConversionContext, JobContext as JC

    # The three producer contexts merged (they are aliases until th#2052).
    assert ConversionContext is JC
    for name in (
        "mktemp", "checkpoint_dir", "resolve_dataset", "dataset_paths",
        "save_checkpoint", "open_checkpoint_stream", "cancelled",
        "call_endpoint", "progress", "metric", "training_metric", "log",
    ):
        assert hasattr(JC, name), name


PKG_SRC = """
    import msgspec
    from gen_worker import JobContext, Resources, endpoint, job

    class BakeIn(msgspec.Struct):
        rung: str = "w8a8"

    class BakeOut(msgspec.Struct):
        rung: str
        positions: int

    @job(publishes=True, resources=Resources(vcpus=2))
    @endpoint(kind="conversion", publishes=True, name="bake")
    def bake(ctx: JobContext, spec: BakeIn) -> BakeOut:
        ctx.progress(position=1, total=2, phase="bake")
        ctx.progress(position=2, total=2, phase="bake")
        return BakeOut(rung=spec.rung, positions=int(ctx.position("bake") or 0))
"""


@pytest.fixture()
def both_harness_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("GEN_WORKER_LOCAL_OUTPUT_DIR", str(tmp_path / "out"))
    pkg = tmp_path / "portable_job"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent(PKG_SRC))
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "portable-job"\nversion = "0.0.0"\n'
        '[tool.gen_worker]\nmain = "portable_job.main"\n'
    )
    return tmp_path


def test_the_same_body_runs_green_under_both_harnesses(
    both_harness_pkg: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """THE charter obligation (th#2049 constraint 1). Both CLIs, end to end."""
    import json

    from gen_worker.cli import main

    payload = both_harness_pkg / "payload.json"
    payload.write_text('{"rung": "w4a4"}')
    cfg = str(both_harness_pkg / "pyproject.toml")

    # (a) the ENDPOINT harness. `run` wraps the handler's return in its
    # {"event": "result", "value": ...} stdout envelope.
    assert main(["run", "--config", cfg, "--payload", '{"rung": "w4a4"}']) == 0
    envelope = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert envelope["event"] == "result"
    endpoint_out = envelope["value"]

    # (b) the JOB harness — same body, same payload, no hub either way
    assert main(
        ["job", "run", "bake", "--config", cfg, "--payload", str(payload)]
    ) == 0
    job_out = json.loads(capsys.readouterr().out)

    assert endpoint_out["rung"] == job_out["rung"] == "w4a4"
    # Both harnesses ran the monotonic-position surface, under the same name.
    assert endpoint_out["positions"] == job_out["positions"] == 2


# ---- 2. run-once refusals, at decoration ----------------------------------

def test_a_job_may_not_be_a_class() -> None:
    with pytest.raises(TypeError, match="module-level FUNCTION"):
        job(  # type: ignore[call-overload]
            resources=Resources()
        )(type("Trainer", (), {}))


def test_a_job_may_not_live_inside_a_class() -> None:
    with pytest.raises(TypeError, match="declared inside class"):
        class Holder:
            @job
            def train(self, ctx: JobContext, spec: BakeIn) -> BakeOut:
                raise AssertionError("never runs")


def test_a_job_may_not_be_a_generator() -> None:
    with pytest.raises(TypeError, match="must not be a generator"):
        @job
        def streamer(ctx: JobContext, spec: BakeIn) -> Any:
            yield BakeOut(rung="x", ctx_class="")


def test_a_job_takes_exactly_ctx_and_payload() -> None:
    with pytest.raises(TypeError, match=r"exactly \(ctx, payload\)"):
        @job
        def three(ctx: JobContext, spec: BakeIn, extra: int) -> BakeOut:
            raise AssertionError("never runs")


def test_a_job_declares_struct_payload_and_result() -> None:
    with pytest.raises(TypeError, match="must be annotated with a msgspec"):
        @job
        def loose(ctx: JobContext, spec: dict) -> BakeOut:  # type: ignore[type-arg]
            raise AssertionError("never runs")

    with pytest.raises(TypeError, match="return type must be a msgspec.Struct"):
        @job
        def loose_out(ctx: JobContext, spec: BakeIn) -> dict:  # type: ignore[type-arg]
            raise AssertionError("never runs")


def test_visibility_is_private_by_default_and_a_closed_vocabulary() -> None:
    @job
    def defaulted(ctx: JobContext, spec: BakeIn) -> BakeOut:
        raise AssertionError("never runs")

    spec = extract_job_spec(defaulted)
    assert spec is not None and spec.visibility == "private"
    with pytest.raises(ValueError, match="visibility must be one of"):
        job(visibility="world-readable")


# ---- 3. the hub-write declaration, RED under BOTH harnesses ---------------

def _ctx(*, publishes: bool, kind: str = "job") -> JobContext:
    return JobContext(
        request_id="r-1294", job_id="j-1294", publishes=publishes,
        execution_hints={"kind": kind},
    )


def test_undeclared_publish_is_refused_typed_under_both_harnesses(
    tmp_path: Path,
) -> None:
    """One declaration surface, so one refusal — reached from the @job side
    and from the @endpoint side of the SAME publisher method."""
    blob = tmp_path / "adapter.safetensors"
    blob.write_bytes(b"\x00" * 16)

    for kind in ("job", "inference"):
        ctx = _ctx(publishes=False, kind=kind)
        with pytest.raises(PublishNotDeclaredError) as caught:
            ctx.save_checkpoint("org/repo", blob)
        assert caught.value.surface == "save_checkpoint"
        assert "publishes=True" in str(caught.value)
        with pytest.raises(PublishNotDeclaredError):
            ctx.open_checkpoint_stream("org/repo")


def test_publish_flavors_refuses_undeclared_before_it_reads_anything() -> None:
    from gen_worker.convert.publish import publish_flavors

    with pytest.raises(PublishNotDeclaredError) as caught:
        publish_flavors(_ctx(publishes=False), [], destination_repo="org/repo")
    assert caught.value.surface == "publish_flavors"


def test_a_declared_publisher_gets_past_the_declaration_gate(
    tmp_path: Path,
) -> None:
    """The same call with the declaration is no longer refused HERE — it goes
    on to the repo-scope/transport checks, which is the correct next gate."""
    blob = tmp_path / "adapter.safetensors"
    blob.write_bytes(b"\x00" * 16)
    ctx = _ctx(publishes=True)
    ctx.open_checkpoint_stream("org/repo")  # no PublishNotDeclaredError


def test_producer_kinds_still_publish_but_are_told_they_are_on_borrowed_time(
    tmp_path: Path,
) -> None:
    """TRANSITIONAL (th#2052 deletes it): kind still implies write authority
    for the un-migrated fleet, and every such call CONFESSES so the admission
    is never silent."""
    emitted: list[Dict[str, Any]] = []
    ctx = JobContext(
        request_id="r-legacy", job_id="j-legacy", publishes=False,
        execution_hints={"kind": "conversion"}, emitter=emitted.append,
    )
    ctx.open_checkpoint_stream("org/repo")  # admitted by kind, not declaration
    warnings = [e for e in emitted if e["type"] == "request.log"]
    assert warnings and "th#2052" in warnings[0]["payload"]["message"]


class PubIn(msgspec.Struct):
    pass


class PubOut(msgspec.Struct):
    declared: bool


@job(publishes=True, name="declared-publisher")
def declared_publisher(ctx: JobContext, spec: PubIn) -> PubOut:
    return PubOut(declared=ctx.publishes)


@job(name="silent-job")
def silent_job(ctx: JobContext, spec: PubIn) -> PubOut:
    return PubOut(declared=ctx.publishes)


def _run(fn: Any, ctx: JobContext) -> Any:
    spec = extract_job_spec(fn)
    assert spec is not None
    return execute_job(
        JobDispatch(job_name=spec.name, payload=msgspec.msgpack.encode(PubIn())),
        jobs={spec.name: spec}, ctx=ctx, reraise=True,
    )


def test_execute_job_stamps_the_declaration_so_a_dispatch_head_cannot_forget() -> None:
    """The JobSpec is the ONE home for `publishes`; the context flag is a
    projection of it. A dispatch head that built the context without it would
    hand a job a perfectly valid capability token and then have the SDK refuse
    the job's own publish — a wiring bug that reads as a broken feature. So the
    one place holding both halves stamps it."""
    ctx = _ctx(publishes=False)          # head forgot to pass it
    outcome = _run(declared_publisher, ctx)
    assert msgspec.msgpack.decode(outcome.result)["declared"] is True
    assert ctx.publishes is True


def test_a_caller_may_not_grant_authority_the_release_never_declared() -> None:
    """The opposite direction is a refusal, not a silent downgrade: the hub
    minted no write grant for an undeclared job, so the claim could only fail
    later and further from its cause."""
    with pytest.raises(ValueError, match="declaration is the release's"):
        _run(silent_job, _ctx(publishes=True))


# ---- 4. position is monotonic ---------------------------------------------

def test_position_going_backwards_raises_rather_than_lying() -> None:
    ctx = _ctx(publishes=False)
    ctx.progress(position=10, total=100, phase="download")
    ctx.progress(position=10, total=100, phase="download")   # flat is legal
    ctx.progress(position=64, total=100, phase="download")
    with pytest.raises(NonMonotonicProgressError) as caught:
        ctx.progress(position=63, total=100, phase="download")
    assert caught.value.phase == "download"
    assert (caught.value.last, caught.value.attempted) == (64.0, 63.0)
    # A NEW phase restarts the count — that is how a job says "next stage".
    ctx.progress(position=0, total=100, phase="upload")
    assert ctx.position("download") == 64.0
    assert ctx.position("upload") == 0.0


def test_both_spellings_of_one_quantity_may_not_disagree() -> None:
    ctx = _ctx(publishes=False)
    with pytest.raises(ValueError, match="position= and step="):
        ctx.progress(position=1, step=2, phase="p")
    with pytest.raises(ValueError, match="phase= and stage="):
        ctx.progress(position=1, phase="a", stage="b")


def test_the_emitted_payload_is_the_shape_the_HUB_parses() -> None:
    """RECONCILED against th#2050's landed `forkJobProgress` +
    `runtimestore.ParseRequestProgressPayload`, which read `step` / `stage` /
    `total` and nothing else. A position that only reached `position` would be
    invisible to the liveness sweep that CONDEMNS the pod — the precise way
    pgw#1287 killed pods that were working."""
    emitted: list[Dict[str, Any]] = []
    ctx = JobContext(request_id="r", emitter=emitted.append)
    ctx.progress(position=4096.5, total=31_000_000, phase="download")
    payload = emitted[-1]["payload"]
    assert payload["step"] == 4096          # what the hub reads
    assert payload["stage"] == "download"   # what the hub reads
    assert payload["total"] == 31_000_000   # what the hub reads
    assert payload["position"] == 4096.5    # the exact value, beside it


def test_metric_emits_the_name_value_rows_the_hub_ingests() -> None:
    """th#2050's job-metric arm reads `name` and `value` off the payload and
    DROPS anything else, so one event per named scalar is the contract — a
    `{values: {...}}` envelope would have been silently discarded."""
    emitted: list[Dict[str, Any]] = []
    ctx = JobContext(request_id="r", emitter=emitted.append)
    ctx.metric({"loss": 0.31, "cosine": 0.998}, step=120, total=2000)
    rows = [e["payload"] for e in emitted if e["type"] == "request.metric"]
    assert [(r["name"], r["value"]) for r in rows] == [
        ("cosine", 0.998), ("loss", 0.31),
    ]
    assert all(r["step"] == 120 and r["total"] == 2000 for r in rows)


def test_positions_ride_the_CTX_EVENT_channel_never_the_output_channel() -> None:
    """th#2050 landed `JobProgress` as a STREAMING OUTPUT chunk and put job
    liveness on the ctx-event envelope instead: `forkJobProgress` parses only
    frames whose `content_type` is `application/x-request-event+json` and DROPS
    every raw output chunk. So a position emitted on the output channel would
    be silently invisible to the freshness window and to th#2014's brake —
    indistinguishable from a pod doing nothing.

    Two halves, both asserted here: the ctx-event channel this context's
    progress/metric/log events ride is EXACTLY the hub's constant, and every
    streamed-output content type is a different string.
    """
    from gen_worker.executor import EVENT_CONTENT_TYPE

    # Verbatim from tensorhub `runtimestore.RequestEventContentType`.
    assert EVENT_CONTENT_TYPE == "application/x-request-event+json"

    # The output channel's content types, from executor._encode_chunk. None of
    # them may collide with the ctx-event channel, or an output chunk would be
    # parsed as a position (and vice versa).
    output_content_types = {
        "text/plain", "application/x-batch-item+msgpack", "application/json",
    }
    assert EVENT_CONTENT_TYPE not in output_content_types

    # And a job structurally has NO output-chunk path at all: a generator body
    # is refused at decoration, so there is nothing that could stream one.
    with pytest.raises(TypeError, match="must not be a generator"):
        @job
        def streams(ctx: JobContext, spec: BakeIn) -> Any:
            yield BakeOut(rung="x", ctx_class="")


def test_the_request_spelling_is_untouched() -> None:
    """Portability cuts both ways: an existing endpoint body must keep
    working, byte for byte."""
    emitted: list[Dict[str, Any]] = []
    ctx = JobContext(request_id="r", emitter=emitted.append)
    ctx.progress(0.5, "denoise", step=5, total=20)
    payload = emitted[-1]["payload"]
    assert payload["progress"] == 0.5
    assert payload["stage"] == "denoise"
    assert payload["step"] == 5 and payload["total"] == 20
    # ...and the same call fed the position ledger the stall watch reads.
    assert ctx.position("denoise") == 5.0


def test_the_positional_job_form_is_refused_instead_of_reinterpreted() -> None:
    """`ctx.progress(4096, 31_000_000, "download")` would have to guess which
    quantity the second argument is. Guessing is what makes an instrument a
    liar, so it refuses and names the keyword form."""
    ctx = _ctx(publishes=False)
    # Called through getattr: these are DELIBERATE misuses, and a type checker
    # rejecting them statically is half the guarantee — the runtime refusal
    # below is the other half, for callers with no type checker.
    misuse = getattr(ctx, "progress")
    # Two positional args: the SDK's own refusal, naming the keyword form.
    with pytest.raises(TypeError, match="second positional argument"):
        misuse(4096, 31_000_000)
    # Three: Python refuses it before we get a chance to, which is the same
    # answer — the point is that nothing reinterprets the argument by type.
    with pytest.raises(TypeError):
        misuse(4096, 31_000_000, "download")


# ---- 5. pgw#1287's class: silence cannot bank as success -------------------

class SilentIn(msgspec.Struct):
    report: bool = False


class SilentOut(msgspec.Struct):
    moved: int


@job(name="transfer")
def transfer(ctx: JobContext, spec: SilentIn) -> SilentOut:
    """A transfer loop. The ONLY difference between the two arms is whether it
    reports position — which is the whole of pgw#1287.

    The silent arm waits for the WATCH'S OWN VERDICT (it cancels the context)
    rather than for a clock, then returns NORMALLY. That is the property under
    test: a run that stalled must not bank as a success even though its body
    completed. Waiting on the verdict is also what makes the arm deterministic
    — nothing here races the scheduler.
    """
    moved = 0
    for _chunk in range(4):
        moved += 64
        if spec.report:
            ctx.progress(position=moved, total=256, phase="download")
    if not spec.report:
        await_progress(
            lambda: ctx.cancelled,
            lambda seen: bool(seen),
            what="the progress watch to cancel the stalled run",
            cadence=Cadence(),
        )
    return SilentOut(moved=moved)


_TRANSFER_JOBS = {"transfer": extract_job_spec(transfer)}


def _run_transfer(*, report: bool, budget_s: float) -> Any:
    ctx = _ctx(publishes=False)
    dispatch = JobDispatch(
        job_name="transfer",
        payload=msgspec.msgpack.encode(SilentIn(report=report)),
        phase_budget_s=budget_s,
    )
    return execute_job(dispatch, jobs=_TRANSFER_JOBS, ctx=ctx)  # type: ignore[arg-type]


def test_a_transfer_that_reports_nothing_fails_the_job() -> None:
    """Silence for the whole budget is the fault, and the body RETURNING
    afterwards does not launder it."""
    outcome = _run_transfer(report=False, budget_s=0.05)
    assert outcome.status == "failed"
    assert outcome.error_type == "JobProgressStalledError"


def test_the_same_loop_reporting_position_succeeds() -> None:
    """The identical body, reporting, under the PRODUCTION default budget.

    Deliberately not the silent arm's 0.05 s: a budget equal to the loop's own
    cadence asserts that today's runner is fast enough, not that the code is
    right — and it flaked exactly that way on CI (run 31983078681) before this
    was fixed. What this arm owes is that a loop which REPORTS is never
    condemned under the budget production actually ships.
    """
    outcome = _run_transfer(report=True, budget_s=DEFAULT_PHASE_BUDGET_S)
    assert outcome.status == "succeeded"
    assert msgspec.msgpack.decode(outcome.result)["moved"] == 256


def test_a_stalled_context_is_cancelled_and_the_error_names_the_phase() -> None:
    ctx = _ctx(publishes=False)
    ctx.progress(position=64, total=256, phase="download")
    with ProgressWatch(ctx, budget_s=0.05, poll_s=0.01) as watch:
        # Waited on PROGRESS, never on a clock (pgw#795): the only success is
        # the verdict arriving, and a wait that never advances dies at the
        # harness floor naming what it last saw.
        await_progress(
            lambda: watch.stalled,
            lambda seen: seen is not None,
            what="the progress watch to judge the phase stalled",
            cadence=Cadence(),
            render=lambda seen: "no verdict yet" if seen is None else str(seen),
        )
    assert watch.stalled is not None
    assert watch.stalled.phase == "download"
    assert ctx.cancelled  # production-owned: the run is told, not just logged
    with pytest.raises(JobProgressStalledError):
        watch.check()


def test_run_once_is_stated_as_data_on_the_outcome() -> None:
    """Nothing survives in-process between jobs by contract, so the outcome
    tells its driver to recycle the child rather than leaving it to a comment."""
    assert _run_transfer(report=True, budget_s=0.05).recycle_child is True


# ---- 6. the manifest ------------------------------------------------------

@pytest.fixture()
def manifest_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg = tmp_path / "manifest_job"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent("""
        import msgspec
        from gen_worker import JobContext, RequestContext, Resources, endpoint, job

        class In_(msgspec.Struct):
            steps: int = 10

        class Out_(msgspec.Struct):
            ok: bool

        @job(resources=Resources(vcpus=4), env=("HF_TOKEN",),
             resumable=True, publishes=True)
        def zebra_bake(ctx: JobContext, spec: In_) -> Out_:
            return Out_(ok=True)

        @job
        def alpha_plan(ctx: JobContext, spec: In_) -> Out_:
            return Out_(ok=True)

        @endpoint
        class Gen:
            def generate(self, ctx: RequestContext, p: In_) -> Out_:
                return Out_(ok=True)
    """))
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "manifest-job"\nversion = "0.0.0"\n'
        '[tool.gen_worker]\nmain = "manifest_job.main"\n'
    )
    return tmp_path


def test_jobs_ride_the_manifest_beside_functions(manifest_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_jobs

    jobs = discover_jobs(manifest_pkg, main_module="manifest_job.main")
    # Sorted by name: the manifest is a published artifact, so the block is
    # byte-stable across runs.
    assert [j["name"] for j in jobs] == ["alpha-plan", "zebra-bake"]
    assert jobs == discover_jobs(manifest_pkg, main_module="manifest_job.main")

    bake = jobs[1]
    assert bake["resources"]["vcpus"] == 4
    assert bake["env"] == ["HF_TOKEN"]
    assert bake["resumable"] is True
    assert bake["visibility"] == "private"
    assert bake["publishes"] is True
    assert bake["payload_schema_sha256"] and bake["output_schema_sha256"]
    # A POINTER into the release tarball, never a copy of the bytes
    # (RECONCILED to th#2049's landed correction 6: a `source` text field would
    # be a second copy that can only drift).
    assert bake["source_file"] == "manifest_job/main.py"
    assert "source" not in bake
    # A job declares no lanes, no compile cell, no slots — deliberately.
    assert not {"execution_lanes", "compile", "slots"} & set(bake)

    assert jobs[0]["publishes"] is False   # never omitted; see below


def test_publishes_is_emitted_on_the_function_row_too(manifest_pkg: Path) -> None:
    """Both row shapes, ALWAYS emitted: the hub mints a write grant off this,
    so 'absent' must mean 'wheel too old to have the concept' and nothing
    else."""
    from gen_worker.discovery.discover import discover_functions

    fns = discover_functions(manifest_pkg, main_module="manifest_job.main")
    assert [f["publishes"] for f in fns] == [False]


def test_the_full_manifest_carries_a_jobs_block_beside_functions(
    manifest_pkg: Path,
) -> None:
    """One package may carry BOTH; publish once, submit as needed."""
    from gen_worker.discovery.discover import discover_manifest

    manifest = discover_manifest(manifest_pkg)
    assert [j["name"] for j in manifest["jobs"]] == ["alpha-plan", "zebra-bake"]
    assert [f["name"] for f in manifest["functions"]] == ["generate"]
    assert manifest["functions"][0]["publishes"] is False
