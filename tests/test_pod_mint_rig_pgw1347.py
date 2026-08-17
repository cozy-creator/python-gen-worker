"""pgw#1347 — the pod mint-rig, with the provider faked.

WHY THE REST LAYER IS FAKED HERE AND NOWHERE ELSE. This workspace's standing
rule is that mocks hide the bugs integration tests catch, and a pod driver is
exactly the kind of code where that is true. The split this file takes is:

  * everything the driver DECIDES — the mandatory rail, the kill-set written
    before the create call, stall detection with no clock in it, the three
    teardown verdicts, the create-window name sweep, the re-roll verdict — is
    tested here against a fake provider, because those are decisions about
    money and about leaks and they must be proven on the paths that are hard to
    reach with a real pod (a create that half-happens; a listing that still
    carries a deleted pod);
  * everything the driver ASSERTS ABOUT THE WIRE is proven by the real run
    recorded on pgw#1347/#1331, not by a mock.

The fakes below therefore imitate the PROVIDER, never the rig.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from mint_rig import cards as cards_mod  # noqa: E402
from mint_rig.driver import CENSUS, SSHD, Rig, _count, _section  # noqa: E402
from mint_rig.killset import KillSet  # noqa: E402
from mint_rig.progress import Failed, Gate, Observation, Stuck  # noqa: E402
from mint_rig.rail import Rail, RailTripped  # noqa: E402
from mint_rig.row import Teardown  # noqa: E402
from mint_rig.runpod import PodNotFound, RunpodRest  # noqa: E402
from mint_rig.transport import Result, SshTransport  # noqa: E402
from mint_rig.workload import (  # noqa: E402
    POD_REPO,
    POD_ROOT,
    Upload,
    Workload,
    install_sdist,
    mint_model,
    sm_probe,
)

# --------------------------------------------------------------------------- #
# fakes: the PROVIDER, not the rig
# --------------------------------------------------------------------------- #


class FakePods:
    """A RunPod account with one pod in it, and every failure mode we care about."""

    def __init__(
        self,
        *,
        rate: float = 0.4,
        create_raises: Exception | None = None,
        creates_anyway: bool = False,
        sticky_listing: bool = False,
    ) -> None:
        self.rate = rate
        self.create_raises = create_raises
        self.creates_anyway = creates_anyway
        self.sticky_listing = sticky_listing
        self.pods: dict[str, dict[str, Any]] = {}
        self.created_bodies: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.get_calls = 0
        self._n = 0

    def create(self, body: Mapping[str, Any]) -> dict[str, Any]:
        self.created_bodies.append(dict(body))
        if self.create_raises is not None:
            if self.creates_anyway:
                self._register(str(body.get("name", "")))
            raise self.create_raises
        return self._register(str(body.get("name", "")))

    def _register(self, name: str) -> dict[str, Any]:
        self._n += 1
        pod_id = f"pod{self._n}"
        record = {
            "id": pod_id,
            "name": name,
            "costPerHr": self.rate,
            "desiredStatus": "RUNNING",
            "publicIp": "10.0.0.1",
            "portMappings": {"22": 40022},
            "lastStatusChange": "rented: 2026-08-17",
            "machine": {"gpuTypeId": "NVIDIA A40"},
        }
        self.pods[pod_id] = record
        return record

    def get(self, pod_id: str) -> dict[str, Any]:
        self.get_calls += 1
        if pod_id not in self.pods:
            raise PodNotFound(pod_id)
        return dict(self.pods[pod_id])

    def list_pods(self) -> list[dict[str, Any]]:
        if self.sticky_listing:
            # The failure the third verdict exists for: DELETE answered, the
            # per-pod GET 404s, and the account listing still carries it.
            return [dict(r) for r in ({**p} for p in self._all())]
        return [dict(p) for p in self.pods.values()]

    def _all(self) -> list[dict[str, Any]]:
        return list(self.pods.values()) + [{"id": pid, "name": "ghost"} for pid in self.deleted]

    def delete(self, pod_id: str) -> None:
        self.deleted.append(pod_id)
        self.pods.pop(pod_id, None)

    def registry_auth(self, name: str, username: str, password: str) -> str:
        return "auth1"



class FakeGuard:
    """podguard, reduced to the two calls the driver makes of it."""

    def __init__(self) -> None:
        self.armed: list[dict[str, Any]] = []
        self.released: list[tuple[str, bool, str]] = []

    def rent(
        self,
        api_key: str,
        body: Mapping[str, Any],
        *,
        lane: str,
        lease_seconds: float,
        orig_cmd: Sequence[str],
        post: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        armed = dict(body)
        env = dict(armed.get("env") or {})
        env["PODGUARD_CONFIG_B64"] = "ZmFrZQ=="
        armed["env"] = env
        armed["dockerStartCmd"] = list(orig_cmd)
        self.armed.append(armed)
        return post(armed)

    def release(self, pod_id: str, *, confirmed_dead: bool, reason: str) -> None:
        self.released.append((pod_id, confirmed_dead, reason))


class FakeTransport:
    """A pod that answers. Scripted by substring, because the driver's scripts
    are the thing under test and matching them exactly would just restate them."""

    def __init__(
        self,
        *,
        ssh_ready_after: int = 0,
        rigboot_rc: int = 0,
        rigboot_path: str = "native",
        setup_fails_at: int | None = None,
        ticks_to_done: int = 3,
        workload_fails: bool = False,
        frozen: bool = False,
    ) -> None:
        self.ssh_ready_after = ssh_ready_after
        self.rigboot_rc, self.rigboot_path = rigboot_rc, rigboot_path
        self.setup_fails_at = setup_fails_at
        self.ticks_to_done, self.workload_fails, self.frozen = ticks_to_done, workload_fails, frozen
        self.scripts: list[str] = []
        self.puts: list[tuple[list[str], str]] = []
        self.fetched: list[str] = []
        self._ssh_calls = 0
        self._probe_calls = 0
        self._setup_calls = 0

    def run(self, script: str, *, timeout_s: float = 300.0, env: Mapping[str, str] | None = None) -> Result:
        self.scripts.append(script)
        if script.strip() == "true":
            self._ssh_calls += 1
            if self._ssh_calls <= self.ssh_ready_after:
                return Result(255, "ssh: connect to host 10.0.0.1 port 40022: Connection refused")
            return Result(0, "")
        if "rigboot.py" in script:
            # Shaped like the real one: the record, then the bootlog section.
            body = json.dumps({"path": self.rigboot_path, "driver": "580.65.06"})
            return Result(0, f"RIG_RC={self.rigboot_rc}\n{body}\n--RIG-BOOTLOG--\n{{not json}}\n")
        if "nvidia-smi" in script:
            return Result(0, "NVIDIA A40, 580.65.06, 8.6\n")
        if "--RIG-SIZE--" in script:
            self._probe_calls += 1
            size = 0 if self.frozen else self._probe_calls * 1_048_576
            done = not (self.workload_fails or self.frozen) and self._probe_calls >= self.ticks_to_done
            fail = self.workload_fails and self._probe_calls >= self.ticks_to_done
            if fail:
                tail = "Traceback (most recent call last)"
            elif self.frozen:
                # A retry loop reprinting the same line forever: the log has
                # BYTES but no progress. This is the shape a wall-clock gate
                # cannot tell apart from a slow compile.
                tail = "waiting for the hub to answer"
            else:
                tail = f"[{self._probe_calls}/6] minting clip"
            # `grep -c` prints its count and exits 1 when the count is zero.
            # The fake reproduces that shape EXACTLY, because a fake that only
            # ever emitted a clean "0" is what let the real false-green ship.
            return Result(
                0,
                f"--RIG-SIZE--\n{size}\n--RIG-BYTES--\n{100 if self.frozen else size}\n"
                f"--RIG-MARK--\n{1 if done else 0}\n"
                f"--RIG-FAIL--\n{1 if fail else 0}\n0\n"
                f"--RIG-TAIL--\n{tail}\n",
            )
        if "setsid nohup" in script:
            return Result(0, "RIG_LAUNCHED 1234")
        if script.startswith("mkdir -p") and "&&" not in script:
            return Result(0, "")
        self._setup_calls += 1
        if self.setup_fails_at is not None and self._setup_calls > self.setup_fails_at:
            return Result(1, "ERROR: could not install")
        return Result(0, "ok")

    def put(self, local: Sequence[Path], remote_dir: str, *, timeout_s: float = 1800.0) -> Result:
        self.puts.append(([str(p) for p in local], remote_dir))
        return Result(0, "")

    def fetch(self, remote: str, local_dir: Path, *, timeout_s: float = 1800.0) -> Result:
        self.fetched.append(remote)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / Path(remote).name).write_text("artifact\n")
        return Result(0, "")


@pytest.fixture(autouse=True)
def _killset_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "killset"
    monkeypatch.setenv("MINT_RIG_KILLSET_DIR", str(root))
    monkeypatch.setattr("mint_rig.killset.KILLSET_DIR", root)
    return root


def _rig(tmp_path: Path, pods: FakePods, transport: FakeTransport, **kw: Any) -> Rig:
    guard = kw.pop("guard", None) or FakeGuard()
    return Rig(
        rail=kw.pop("rail", None) or Rail(max_usd=2.0),
        lane=kw.pop("lane", "test"),
        api=pods,
        guard=guard,
        api_key="k",
        out_dir=tmp_path / "runs",
        transport_factory=lambda host, port: transport,
        # A FIXED key. Reading ~/.ssh/runpod.pub made every row in this file
        # depend on the developer's home directory; CI has no such file and said
        # so, 30 rows at a time.
        public_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5 pgw1347-test",
        sleep=lambda _s: None,
        log=lambda _m: None,
        tick_s=0.0,
        **kw,
    )


# --------------------------------------------------------------------------- #
# 1. the rail is mandatory, and it is money, not a clock
# --------------------------------------------------------------------------- #


def test_rail_refuses_to_exist_without_a_declared_cap() -> None:
    with pytest.raises(TypeError):
        Rail()  # type: ignore[call-arg]
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="declare a spend rail"):
            Rail(max_usd=bad)


def test_rig_refuses_to_exist_without_a_rail(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="mandatory"):
        Rig(rail=None, lane="x", api=FakePods(), guard=FakeGuard())  # type: ignore[arg-type]


def test_rail_wall_is_derived_from_the_observed_rate_not_a_constant() -> None:
    rail = Rail(max_usd=2.0)
    # No rate observed yet: the rig must not invent one.
    assert rail.wall_seconds == float("inf")
    rail.observe_rate(0.40)
    assert rail.wall_seconds == pytest.approx(5.0 * 3600.0)
    rail.observe_rate(5.98)  # a B200: the SAME cap is a much shorter wall
    assert rail.wall_seconds == pytest.approx(2.0 / 5.98 * 3600.0)


def test_rail_headroom_refuses_to_start_work_it_cannot_finish() -> None:
    rail = Rail(max_usd=1.0, start_headroom=0.85)
    rail.observe_rate(1.0)
    rail.clock_started(at=0.0)
    rail.may_start("ship", now=3000.0)  # $0.83 spent — still inside the headroom
    with pytest.raises(RailTripped, match="headroom exhausted"):
        rail.may_start("ship", now=3100.0)
    rail.check("workload", now=3100.0)  # mid-stage the cap itself is the bound
    with pytest.raises(RailTripped, match="reached the declared"):
        rail.check("workload", now=3700.0)


# --------------------------------------------------------------------------- #
# 2. no magic timeouts: stuck is a frozen token, never an elapsed clock
# --------------------------------------------------------------------------- #


def test_gate_returns_when_the_goal_is_reached() -> None:
    seen: list[int] = []

    def probe() -> Observation:
        seen.append(len(seen))
        return Observation(reached=len(seen) >= 3, token=len(seen))

    result = Gate("s", probe, stall_ticks=2, tick_s=0.0, sleep=lambda _s: None).wait()
    assert result.reached and len(seen) == 3


def test_gate_never_stops_work_that_is_still_moving_however_long_it_takes() -> None:
    """The whole rule, as a test: a thousand observations, all slow, none stuck.

    A wall-clock gate would have fired long before tick 1000. This one cannot,
    because the token advanced every time.
    """
    calls = {"n": 0}

    def probe() -> Observation:
        calls["n"] += 1
        return Observation(reached=calls["n"] >= 1000, token=calls["n"])

    Gate("compile", probe, stall_ticks=3, tick_s=0.0, sleep=lambda _s: None).wait()
    assert calls["n"] == 1000


def test_gate_is_stuck_when_the_token_freezes_and_names_the_token() -> None:
    with pytest.raises(Stuck) as caught:
        Gate(
            "compile",
            lambda: Observation(token="47 bytes", note="waiting for hub"),
            stall_ticks=4,
            tick_s=0.0,
            sleep=lambda _s: None,
        ).wait()
    assert "47 bytes" in str(caught.value)
    assert caught.value.ticks == 4


def test_gate_stall_counter_resets_on_any_progress() -> None:
    tokens = ["a", "a", "a", "b", "a", "a", "a", "c"]
    steps = iter(tokens)

    def probe() -> Observation:
        return Observation(token=next(steps))

    # stall_ticks=3 would fire on a naive counter; it must not, because the
    # token changed at index 3 and again at 7.
    with pytest.raises(StopIteration):
        Gate("s", probe, stall_ticks=3, tick_s=0.0, sleep=lambda _s: None).wait()


def test_gate_reports_a_failure_marker_immediately() -> None:
    with pytest.raises(Failed, match="boom"):
        Gate(
            "s",
            lambda: Observation(failed=True, note="boom"),
            tick_s=0.0,
            sleep=lambda _s: None,
        ).wait()


def test_gate_defers_to_the_rail() -> None:
    rail = Rail(max_usd=0.5)
    rail.observe_rate(10.0)
    rail.clock_started(at=0.0)

    with pytest.raises(RailTripped):
        Gate(
            "s",
            lambda: Observation(token=object()),
            tick_s=0.0,
            sleep=lambda _s: None,
            rail_check=lambda stage: rail.check(stage, now=1e9),
        ).wait()


def test_bring_up_is_bounded_by_MONEY_because_it_has_no_progress_signal() -> None:
    """MEASURED 2026-08-17: RunPod reports `desiredStatus: RUNNING` from the
    instant of rent and exposes `publicIp`/`portMappings` only when the
    container starts — so a 3 GB image pull and a wedged host produce identical
    records. Staleness cannot separate them, and a tick count pretending to
    would be the magic timeout this package refuses."""
    rail = Rail(max_usd=2.0)
    rail.observe_rate(0.74)
    rail.clock_started(at=0.0)
    # 15% of $2.00 is $0.30, which at $0.74/hr is ~24 minutes. DERIVED: the same
    # fraction on a cheaper card buys proportionally longer.
    rail.check_sub("endpoint", 0.15, now=1400.0)
    with pytest.raises(RailTripped, match="no progress signal to be stuck on"):
        rail.check_sub("endpoint", 0.15, now=1500.0)


def test_a_gate_with_no_staleness_rule_waits_as_long_as_the_rail_allows() -> None:
    calls = {"n": 0}

    def probe() -> Observation:
        calls["n"] += 1
        return Observation(reached=calls["n"] >= 500, token="frozen solid")

    Gate("endpoint", probe, stall_ticks=0, tick_s=0.0, sleep=lambda _s: None,
         rail_check=lambda _stage: None).wait()
    assert calls["n"] == 500, "a frozen token must not stop a gate that has no staleness rule"


def test_a_gate_with_no_staleness_rule_and_no_rail_is_REFUSED() -> None:
    """Unbounded is worse than a timeout, so it is not reachable."""
    with pytest.raises(ValueError, match="must carry a rail_check"):
        Gate("x", lambda: Observation(), stall_ticks=0, tick_s=0.0, sleep=lambda _s: None).wait()


# --------------------------------------------------------------------------- #
# 3. the kill-set exists before the pod does
# --------------------------------------------------------------------------- #


def test_killset_is_written_before_the_create_call_leaves(tmp_path: Path, _killset_home: Path) -> None:
    seen: dict[str, Any] = {}

    class WatchingPods(FakePods):
        def create(self, body: Mapping[str, Any]) -> dict[str, Any]:
            # At the instant the POST is made, what is on disk?
            seen["records"] = KillSet.open_records(_killset_home)
            return super().create(body)

    pods = WatchingPods()
    rig = _rig(tmp_path, pods, FakeTransport())
    rig.run(cards_mod.pick("a40"), Workload(name="w", command="echo RIG_DONE"))

    records = seen["records"]
    assert len(records) == 1, "the kill-set must be on disk BEFORE the create call"
    assert records[0]["state"] == "PENDING"
    assert records[0]["pod_id"] == "", "the id cannot exist yet — that is the point"
    # And it is stoppable by the one identifier that DOES exist: the name.
    assert records[0]["pod_name"] in records[0]["kill_by_name"]
    assert "terminate --name" in records[0]["kill_by_name"]
    assert records[0]["kill_by_name"].split()[1].startswith("/"), (
        "the kill command must be absolute — whoever reads it is not in our cwd"
    )


def test_killset_closes_on_a_verified_teardown(tmp_path: Path, _killset_home: Path) -> None:
    rig = _rig(tmp_path, FakePods(), FakeTransport())
    rig.run(cards_mod.pick("a40"), Workload(name="w", command="echo RIG_DONE"))
    assert KillSet.open_records(_killset_home) == [], "a released rental leaves no open record"


def test_a_create_that_half_happened_is_swept_by_name(tmp_path: Path, _killset_home: Path) -> None:
    """The create-window leak: the pod exists, the response never landed.

    Nobody knows the id — but we chose the name, and the account listing has it.
    """
    pods = FakePods(create_raises=RuntimeError("connection reset"), creates_anyway=True)
    rig = _rig(tmp_path, pods, FakeTransport())
    row = rig.run(cards_mod.pick("a40"), Workload(name="w", command="echo RIG_DONE"))

    assert row.verdict == "refused" and row.failed_stage == "create"
    assert pods.deleted == ["pod1"], "the orphan must be found by name and killed"
    assert KillSet.open_records(_killset_home) == []


def test_a_create_that_truly_failed_leaves_nothing_open(tmp_path: Path, _killset_home: Path) -> None:
    pods = FakePods(create_raises=RuntimeError("no capacity"))
    row = _rig(tmp_path, pods, FakeTransport()).run(
        cards_mod.pick("a40"), Workload(name="w", command="echo RIG_DONE")
    )
    assert row.verdict == "refused"
    assert pods.deleted == []
    assert KillSet.open_records(_killset_home) == []


def test_dry_run_writes_the_killset_and_rents_nothing(tmp_path: Path) -> None:
    pods = FakePods()
    row = _rig(tmp_path, pods, FakeTransport(), dry_run=True).run(
        cards_mod.pick("a40"), Workload(name="w", command="echo RIG_DONE")
    )
    assert row.verdict == "refused" and pods.created_bodies == []
    assert Path(row.killset_path).is_file()


# --------------------------------------------------------------------------- #
# 4. the whole cycle, and what the row must carry
# --------------------------------------------------------------------------- #


def test_a_green_run_records_a_row_a_matrix_can_use(tmp_path: Path) -> None:
    pods, transport = FakePods(rate=0.40), FakeTransport(ssh_ready_after=2)
    guard = FakeGuard()
    workload = mint_model(
        "gen_worker.model.catalog.flux1_dev:FLUX1_DEV",
        runners=("clip",),
        install=("pip install -q x",),
        uploads=(Upload(local=tmp_path / "d.whl"),),
    )
    (tmp_path / "d.whl").write_text("wheel bytes")

    row = _rig(tmp_path, pods, transport, guard=guard, lane="pgw1331-clip").run(
        cards_mod.pick("a40"), workload
    )

    assert row.verdict == "green", row.detail
    # asked vs observed are SEPARATE columns — a row that conflates them
    # measures an intention, not a machine.
    assert row.asked_gpu == "a40" and row.asked_gpu_type_ids == ["NVIDIA A40"]
    assert row.observed_gpu == "NVIDIA A40" and row.observed_sm == "8.6"
    assert row.sm_expected == "8.6" and row.driver_version == "580.65.06"
    assert row.cuda_path == "native"
    # cost is runtime x the rate the PROVIDER returned
    assert row.rate_per_hr == 0.40
    assert row.est_cost_usd == round(0.40 * row.pod_seconds / 3600.0, 4)
    assert row.rail_usd == 2.0 and row.rail_tripped is False
    # digests, so "which code ran" survives the month
    assert len(row.workload_digest) == 64
    assert row.uploads[0]["sha256"] == Upload(local=tmp_path / "d.whl").sha256
    assert any(a["remote"].endswith("minted.json") and a["fetched"] for a in row.artifacts)
    assert all(a["sha256"] for a in row.artifacts if a["fetched"])
    # teardown, three ways
    assert row.teardown.delete_issued and row.teardown.get_404 and row.teardown.absent_from_list
    assert guard.released == [("pod1", True, "mint-rig green")]
    # and the row is on disk under a stable name
    banked = json.loads((tmp_path / "runs" / f"{row.pod_name}.row.json").read_text())
    assert banked["verdict"] == "green"


def test_the_ssh_gate_waits_for_the_pod_to_ANSWER_not_for_rest_to_say_running(tmp_path: Path) -> None:
    transport = FakeTransport(ssh_ready_after=5)
    row = _rig(tmp_path, FakePods(), transport).run(
        cards_mod.pick("a40"), Workload(name="w", command="echo RIG_DONE")
    )
    assert row.verdict == "green"
    assert sum(1 for s in transport.scripts if s.strip() == "true") == 6


def test_artifacts_are_captured_even_when_the_workload_fails(tmp_path: Path) -> None:
    transport = FakeTransport(workload_fails=True)
    workload = Workload(name="mint", command="x", artifacts=(f"{POD_ROOT}/minted.json",))
    row = _rig(tmp_path, FakePods(), transport).run(cards_mod.pick("a40"), workload)

    assert row.verdict == "red" and row.failed_stage.startswith("workload")
    assert f"{POD_ROOT}/minted.json" in transport.fetched
    assert f"{POD_ROOT}/mint.log" in transport.fetched, "the failed run's log is the valuable part"
    assert row.teardown.confirmed


def test_a_frozen_workload_is_stuck_and_the_pod_still_dies(tmp_path: Path) -> None:
    pods = FakePods()
    row = _rig(tmp_path, pods, FakeTransport(frozen=True), stall_ticks=4).run(
        cards_mod.pick("a40"), Workload(name="w", command="x")
    )
    assert row.verdict == "stuck"
    assert pods.deleted == ["pod1"], "a stuck workload must not leave a pod billing"


def test_a_setup_failure_is_red_and_names_the_line(tmp_path: Path) -> None:
    workload = Workload(name="w", command="x", setup=("pip install a", "pip install b"))
    row = _rig(tmp_path, FakePods(), FakeTransport(setup_fails_at=0)).run(cards_mod.pick("a40"), workload)
    assert row.verdict == "red" and row.failed_stage == "setup"
    assert "pip install a" in row.detail


def test_a_too_old_host_driver_is_a_REROLL_verdict_not_a_torch_problem(tmp_path: Path) -> None:
    """RIG-ENV §3c: 'driver too old' is a HOST fact. Do not downgrade torch."""
    pods = FakePods()
    row = _rig(tmp_path, pods, FakeTransport(rigboot_rc=91, rigboot_path="reroll")).run(
        cards_mod.pick("a40"), Workload(name="w", command="x")
    )
    assert row.verdict == "reroll" and row.failed_stage == "preflight"
    assert "re-roll" in row.detail
    assert pods.deleted == ["pod1"], "a re-roll kills the host immediately — that is the saving"


def test_a_pod_that_never_answers_is_a_REROLL_not_a_budget_overrun(tmp_path: Path) -> None:
    """A matrix lane can retry a `reroll`; it must not retry a `railed`, which
    means the operator's money is genuinely gone. MEASURED: two pods on the same
    image answered in ~100s while a third sat silent for six minutes."""

    class NeverAnswers(FakePods):
        def _register(self, name: str) -> dict[str, Any]:
            record = super()._register(name)
            record["publicIp"], record["portMappings"] = "", None
            return record

    pods = NeverAnswers(rate=3_600_000.0)  # $1000/s: the sub-cap trips at once
    row = _rig(tmp_path, pods, FakeTransport(), rail=Rail(max_usd=1.0)).run(
        cards_mod.pick("sm89"), Workload(name="w", command="x")
    )
    assert row.verdict == "reroll" and row.failed_stage == "preflight"
    assert "re-roll the host" in row.detail
    assert pods.deleted == ["pod1"]


def test_a_reroll_takes_another_HOST_but_never_another_BUDGET(tmp_path: Path) -> None:
    """MEASURED 2026-08-17: roughly half of twelve rentals on one evening came up
    on a host that could not run the fleet CUDA line, or never exposed a port. A
    matrix that gives up on the first bad machine measures the provider's weather
    rather than the code — and one that re-rolls on a fresh budget each time has
    no budget at all."""
    pods = FakePods(rate=0.74)
    rail = Rail(max_usd=1.0)
    rig = _rig(tmp_path, pods, FakeTransport(rigboot_rc=91), rail=rail)
    row = rig.run(cards_mod.pick("sm89"), Workload(name="w", command="x"), rerolls=2)

    assert row.verdict == "reroll"
    assert len(pods.created_bodies) == 3, "one attempt plus two re-rolls"
    assert pods.deleted == ["pod1", "pod2", "pod3"], "every bad host is torn down"
    # The rail carried each dead pod's spend forward rather than restarting.
    assert rail.banked_usd > 0.0
    assert any(s["stage"] == "reroll" for s in row.stage_trail)


def test_rerolling_stops_when_the_budget_is_gone_not_when_the_count_runs_out(
    tmp_path: Path,
) -> None:
    pods = FakePods(rate=3_600_000.0)  # $1000/s — the first pod eats the rail
    rail = Rail(max_usd=0.05)
    rig = _rig(tmp_path, pods, FakeTransport(rigboot_rc=91), rail=rail)
    row = rig.run(cards_mod.pick("sm89"), Workload(name="w", command="x"), rerolls=5)
    assert len(pods.created_bodies) < 6, "it stopped on money, not on the counter"
    assert "no budget left" in row.detail


def test_no_driver_at_all_is_also_a_reroll(tmp_path: Path) -> None:
    row = _rig(tmp_path, FakePods(), FakeTransport(rigboot_rc=92)).run(
        cards_mod.pick("a40"), Workload(name="w", command="x")
    )
    assert row.verdict == "reroll" and "no NVIDIA driver" in row.detail


def test_the_rail_tears_the_pod_down_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Money running out AFTER the pod came up is `railed`, not `reroll`: the
    host was fine, the budget is gone, and a matrix lane must not retry it.

    Boot is made affordable and the third check is not, stated directly rather
    than staged through a rate — the property under test is WHICH verdict a
    mid-run exhaustion produces, and the arithmetic that decides WHEN is covered
    above.
    """
    calls = {"n": 0}

    def late_check(self: Any, stage: str, now: float | None = None) -> None:
        calls["n"] += 1
        if calls["n"] > 2:
            raise RailTripped(f"cap reached during {stage!r}")

    monkeypatch.setattr(Rail, "check", late_check)
    monkeypatch.setattr(Rail, "check_sub", lambda *a, **k: None)

    pods = FakePods()
    row = _rig(tmp_path, pods, FakeTransport(), rail=Rail(max_usd=1.0)).run(
        cards_mod.pick("sm89"), Workload(name="w", command="x")
    )
    assert row.verdict == "railed" and row.rail_tripped is True
    assert pods.deleted == ["pod1"]


# --------------------------------------------------------------------------- #
# 5. teardown: three verdicts, because each one alone has lied
# --------------------------------------------------------------------------- #


def test_teardown_is_unconfirmed_while_the_account_listing_still_carries_the_pod(tmp_path: Path) -> None:
    pods = FakePods(sticky_listing=True)
    row = _rig(tmp_path, pods, FakeTransport()).run(cards_mod.pick("a40"), Workload(name="w", command="x"))

    assert row.teardown.delete_issued and row.teardown.get_404
    assert row.teardown.absent_from_list is False
    assert row.teardown.confirmed is False, "404 alone is not proof the pod stopped billing"
    assert row.teardown.attempts == 30, "an unconfirmed teardown is retried, not accepted"


def test_teardown_confirmed_needs_both_the_404_and_the_listing() -> None:
    assert Teardown(delete_issued=True, get_404=True, absent_from_list=True).confirmed
    assert not Teardown(delete_issued=True, get_404=True).confirmed
    assert not Teardown(delete_issued=True, absent_from_list=True).confirmed
    assert not Teardown(delete_issued=True).confirmed


def test_terminate_by_name_finds_the_pod_and_kills_it(tmp_path: Path) -> None:
    pods = FakePods()
    pods.create({"name": "mintrig-orphan-1"})
    rig = _rig(tmp_path, pods, FakeTransport())
    row = rig.terminate(name="mintrig-orphan-1")
    assert row.teardown.confirmed and pods.deleted == ["pod1"]


def test_sweep_flags_a_live_pod_NOBODY_attends(tmp_path: Path) -> None:
    pods = FakePods()
    pods.create({"name": "someone-elses-pod"})
    report = _rig(tmp_path, pods, FakeTransport()).sweep(leases={})
    assert report["unattended"] == ["pod1"]
    assert report["live_pods"][0]["known_to_mint_rig"] is False


def test_sweep_does_not_cry_wolf_over_a_sibling_lanes_attended_pod(tmp_path: Path) -> None:
    """An alarm that fires on every healthy pod is one nobody reads."""
    pods = FakePods()
    pods.create({"name": "another-lanes-pod"})
    report = _rig(tmp_path, pods, FakeTransport()).sweep(leases={"pod1": "svdq-bench-b200"})
    assert report["unrecorded_live"] == ["pod1"], "it is still not OURS, and the row says so"
    assert report["unattended"] == [], "but podguard attends it, so it is not a leak"
    assert report["live_pods"][0]["podguard_lane"] == "svdq-bench-b200"


# --------------------------------------------------------------------------- #
# 6. the workload is a value, and the scripts it generates say what they mean
# --------------------------------------------------------------------------- #


def test_the_workload_digest_covers_the_uploads_not_just_the_command(tmp_path: Path) -> None:
    dist = tmp_path / "gen_worker-0.1.whl"
    dist.write_text("one")
    a = Workload(name="w", command="c", uploads=(Upload(local=dist),))
    first = a.digest()
    dist.write_text("two")
    assert a.digest() != first, "different bytes shipped is a different workload"
    assert Workload(name="w", command="c").digest() != first


def test_the_command_is_launched_detached_and_sources_the_compat_shim() -> None:
    script = Workload(name="mint", command="gen-worker model mint X").launch_script()
    assert "setsid nohup" in script and script.rstrip().endswith("echo RIG_LAUNCHED $!")
    # RIG-ENV §3c: a detached job inherits no login shell, so a forward-compat
    # libcuda repair is invisible to it unless it is sourced explicitly.
    assert "/etc/profile.d/zz-cuda-compat.sh" in script


def test_a_ZERO_COUNT_IS_NOT_A_MATCH_however_the_shell_spells_it() -> None:
    """THE WORST DEFECT THIS RIG HAS HAD, as a row.

    `grep -c X f` prints its count AND exits 1 when the count is zero, so the
    original `grep -c X f || echo 0` emitted "0\n0" for a log with no match. The
    done check compared the section against the string "0", did not match, and
    reported GREEN — for a real run whose command died at its first line and
    produced no artifact at all. A rig that can fabricate a success is worse
    than no rig.

    Fixed at both ends: the script no longer emits the second value, and the
    parse SUMS whatever it gets, so no future variation can resurrect it.
    """
    assert _count("0") == 0
    assert _count("0\n0") == 0, "the exact shape that shipped a false green"
    assert _count("") == 0
    assert _count("0\n0\n0") == 0
    assert _count("1") == 1
    assert _count("0\n3") == 3
    assert _count("grep: no such file") == 0


def test_the_probe_script_never_emits_a_second_zero() -> None:
    script = Workload(name="w", command="x").probe_script()
    assert "|| echo 0" not in script, "grep already prints its own zero"
    assert script.count("|| true") >= 3


def test_a_command_that_dies_at_line_one_is_NOT_green(tmp_path: Path) -> None:
    """The regression, end to end: no marker, no artifact, therefore not green."""

    class DiesImmediately(FakeTransport):
        def run(self, script: str, *, timeout_s: float = 300.0,
                env: Mapping[str, str] | None = None) -> Result:
            if "--RIG-SIZE--" in script:
                # A tiny log holding a refusal, no marker, and — as the real
                # shell does — a zero count that grep exits 1 on.
                return Result(
                    0,
                    "--RIG-SIZE--\n0\n--RIG-BYTES--\n213\n--RIG-MARK--\n0\n"
                    "--RIG-FAIL--\n0\n0\n--RIG-TAIL--\nno fleet-line authority is reachable\n",
                )
            return super().run(script, timeout_s=timeout_s, env=env)

    workload = mint_model("m:F", runners=("clip",))
    row = _rig(tmp_path, FakePods(), DiesImmediately(), stall_ticks=3).run(
        cards_mod.pick("sm89"), workload
    )
    assert row.verdict != "green", "a log with no marker must never read as success"
    assert row.verdict == "stuck"


def test_a_marker_without_the_artifact_is_still_not_green(tmp_path: Path) -> None:
    """The second, independent statement of success. The marker says the command
    believed it worked; the artifact says it left the thing behind."""

    class NoArtifact(FakeTransport):
        def fetch(self, remote: str, local_dir: Path, *, timeout_s: float = 1800.0) -> Result:
            self.fetched.append(remote)
            if remote.endswith("minted.json"):
                return Result(1, "scp: /root/rig/minted.json: No such file or directory")
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / Path(remote).name).write_text("x\n")
            return Result(0, "")

    row = _rig(tmp_path, FakePods(), NoArtifact()).run(
        cards_mod.pick("sm89"), mint_model("m:F", runners=("clip",))
    )
    assert row.verdict == "red" and row.failed_stage == "artifacts"
    assert "minted.json" in row.detail


def test_a_mint_that_DID_produce_its_row_is_green(tmp_path: Path) -> None:
    row = _rig(tmp_path, FakePods(), FakeTransport()).run(
        cards_mod.pick("sm89"), mint_model("m:F", runners=("clip",))
    )
    assert row.verdict == "green", row.detail


def test_the_probe_is_ONE_round_trip_and_parses_into_the_gate_observation() -> None:
    workload = Workload(name="mint", command="x", progress_paths=("/root/rig/cells",))
    script = workload.probe_script()
    for section in ("SIZE", "BYTES", "MARK", "FAIL", "TAIL"):
        assert f"--RIG-{section}--" in script
    assert "/root/rig/cells" in script
    sample = "--RIG-SIZE--\n123\n--RIG-BYTES--\n45\n--RIG-MARK--\n0\n--RIG-FAIL--\n0\n--RIG-TAIL--\nminting clip\n"
    assert _section(sample, "SIZE") == "123"
    assert _section(sample, "TAIL") == "minting clip"
    assert _section(sample, "NOPE") == ""


def test_a_quiet_non_zero_exit_leaves_a_MARK_and_does_not_cost_a_stall_budget() -> None:
    """MEASURED: the mint's rigcheck leg aborted with a one-line refusal and no
    traceback. The log simply stopped growing, so the only signal left was the
    frozen token — and the gate paid a full stall budget of RENTED POD to learn
    what the exit code had already said."""
    script = Workload(name="w", command="do_thing").launch_script()
    assert "RIG_FAIL rc=$rc" in script
    assert "RIG_FAIL" in Workload(name="w", command="x").fail_markers[1]


def test_the_launch_payload_survives_BOTH_shells_it_crosses() -> None:
    """It crosses the shell ssh starts on the pod AND the `bash -lc` that shell
    launches. A double-quoted payload lets the FIRST expand the SECOND's
    variables: measured, `rc=$?` arrived as `rc=` and the guard `[ -eq 0 ]`
    errored, printing RIG_FAIL after every SUCCESSFUL run — a fail marker that
    always fires is a fail marker that means nothing."""
    script = Workload(name="mint", command="echo hi && echo RIG_DONE").launch_script()
    payload = script.split("bash -lc ", 1)[1].split(" > ", 1)[0]
    assert payload.startswith("'") and payload.endswith("'"), (
        "the payload must be SINGLE-quoted; single quotes suppress every expansion"
    )
    assert "$rc" in payload and "$?" in payload, "which is why the variables survive verbatim"
    # `$!` is deliberately OUTSIDE the quotes: it is for the outer shell.
    assert script.endswith("echo RIG_LAUNCHED $!")


def test_the_mint_ships_the_fleet_line_authority_because_the_sdk_is_not_one(tmp_path: Path) -> None:
    """RIG-ENV §2: rigcheck reads endpoint.toml / fleet-floors.toml / ENDPOINT
    dist metadata and deliberately refuses gen-worker's own requirement — an SDK
    certifying its own floor makes every rig pass. This repo ships none of those
    files, so a pod carrying only its wheel aborts FleetLineUnknown."""
    authority = tmp_path / "fleet-floors.toml"
    authority.write_text("[floors]\ntorch = \"2.13.0\"\n")
    workload = mint_model("m:F", runners=("clip",), fleet_line=authority)
    assert workload.env["GEN_WORKER_FLEET_LINE_FILE"] == f"{POD_ROOT}/fleet-floors.toml"
    assert any(u.local == authority for u in workload.uploads)


def test_the_mint_workload_asserts_the_fleet_line_before_it_compiles() -> None:
    workload = mint_model("m:F", runners=("clip",))
    assert workload.command.startswith("python3 -m gen_worker.rigcheck && ")
    assert "--runner clip" in workload.command
    # A compile's progress lives in the inductor cache long before the log moves.
    assert "/root/.cache/torchinductor_root" in workload.progress_paths
    assert workload.env["TORCHINDUCTOR_CACHE_DIR"] == "/root/.cache/torchinductor_root"


def test_the_preflight_reads_the_record_and_not_the_bootlog_after_it(tmp_path: Path) -> None:
    """The first real run reported an EMPTY driver beside a perfectly good
    probe: rigboot prints its record AND writes it to --json, so a naive
    first-brace-to-last-brace parse spanned two documents and yielded nothing."""
    row = _rig(tmp_path, FakePods(), FakeTransport()).run(
        cards_mod.pick("sm89"), Workload(name="w", command="x")
    )
    assert row.cuda_path == "native"


def test_pip_breaks_the_system_because_the_pod_IS_the_system() -> None:
    """MEASURED: the fleet base image's interpreter is Debian
    EXTERNALLY-MANAGED, so PEP 668 refuses the install and recommends a venv —
    which would shadow or re-resolve the fleet's own torch, RIG-ENV §3a's single
    most common way a rig drifts."""
    lines = install_sdist(Path("/tmp/gen_worker-9.whl"))
    assert "--break-system-packages" in lines[1]


def test_the_torch_pin_is_read_from_the_installed_interpreter_not_hardcoded() -> None:
    lines = install_sdist(Path("/tmp/gen_worker-9.whl"))
    assert "torch.__version__" in lines[0], "RIG-ENV §2: the numbers are READ, never hardcoded"
    assert "13.0" not in "".join(lines) and "2.13" not in "".join(lines)
    assert f"-c {POD_ROOT}/constraints.txt" in lines[1]


def test_the_start_command_installs_an_sshd_because_the_fleet_image_ships_none() -> None:
    # RIG-ENV §3a. Also: the fleet image must be the PUBLIC upstream tag —
    # RunPod exits a pod ~1s after rent, with no logs, for an unpullable one.
    assert "openssh-server" in SSHD and "sshd -D" in SSHD
    assert cards_mod.FLEET_IMAGE == (
        "pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime"  # oci-image: the asserted tag
    )
    assert "tensorhub/endpoints" not in cards_mod.FLEET_IMAGE


def test_the_census_reads_the_card_from_the_pod(tmp_path: Path) -> None:
    assert "nvidia-smi" in CENSUS and "compute_cap" in CENSUS


# --------------------------------------------------------------------------- #
# 7. the create body, and the th#1327 gate
# --------------------------------------------------------------------------- #


def test_the_armed_start_command_is_moved_onto_the_entrypoint(tmp_path: Path) -> None:
    """RunPod treats dockerEntrypoint=[] as unset and lets the IMAGE win, so an
    armed dockerStartCmd on an image with its own entrypoint never runs."""
    pods = FakePods()
    _rig(tmp_path, pods, FakeTransport()).run(cards_mod.pick("a40"), Workload(name="w", command="x"))
    body = pods.created_bodies[0]
    assert body["dockerEntrypoint"] == ["/bin/bash", "-lc", SSHD]
    assert "dockerStartCmd" not in body
    assert body["gpuTypeIds"] == ["NVIDIA A40"] and body["ports"] == ["22/tcp"]
    assert body["cloudType"] == "SECURE"
    assert body["env"]["PODGUARD_CONFIG_B64"], "the guard must have armed it"


def test_the_rest_layer_refuses_an_unarmed_create() -> None:
    """th#1327's gate, on OUR http path — podguard installs it on its own."""

    class Refuser:
        class Unarmed(RuntimeError):
            pass

        def assert_armed(self, body: Mapping[str, Any]) -> None:
            if not (body.get("env") or {}).get("PODGUARD_CONFIG_B64"):
                raise self.Unarmed("unarmed")

    rest = RunpodRest("key", guard=Refuser())
    with pytest.raises(RuntimeError, match="unarmed"):
        rest.create({"name": "x", "env": {}})


def test_the_real_podguard_gate_is_the_one_we_call() -> None:
    """Proves the assertion we rely on is podguard's, not a lookalike.

    Skipped where the tracker checkout is absent (CI): the CONTRACT is that a
    body without PODGUARD_CONFIG_B64 is refused, and the test above proves the
    rig honours it. This one proves the real module agrees.
    """
    from mint_rig.runpod import load_podguard

    try:
        podguard = load_podguard()
    except RuntimeError:
        # A STABLE reason string: the real exception names an absolute path,
        # which would make this row's scripts/skip_census.txt key differ per
        # machine and defeat the census that reads it.
        pytest.skip("pgw#1347: podguard is a tracker-checkout peer, absent here")
    with pytest.raises(podguard.Unarmed):
        podguard.assert_armed({"env": {}})
    podguard.assert_armed({"env": {"PODGUARD_CONFIG_B64": "x"}})


def test_ssh_argv_keeps_the_agent_socket_reachable() -> None:
    """The runpod key on the box is passphrase-protected: the agent holds the
    only usable copy, so a transport that sanitised the environment would fail
    in a way that looks exactly like a slow boot."""
    seen: dict[str, Any] = {}

    def runner(argv: Sequence[str], timeout_s: float) -> Result:
        seen["argv"] = list(argv)
        return Result(0, "")

    SshTransport("1.2.3.4", 40022, runner=runner).run("echo hi", env={"HF_TOKEN": "t"})
    argv = seen["argv"]
    assert argv[0] == "ssh" and "root@1.2.3.4" in argv and "40022" in argv
    assert 'export HF_TOKEN="t"; echo hi' == argv[-1]


def test_cloud_type_is_omitted_entirely_when_the_provider_should_choose(tmp_path: Path) -> None:
    pods = FakePods()
    _rig(tmp_path, pods, FakeTransport(), cloud_type="").run(
        cards_mod.pick("sm86"), Workload(name="w", command="x")
    )
    # Omitted, not sent empty: RunPod reads "" as a cloud type it does not have.
    assert "cloudType" not in pods.created_bodies[0]


def test_the_sm86_set_asks_for_every_ampere_sku_not_just_one(tmp_path: Path) -> None:
    """MEASURED: `["NVIDIA A40"]` alone answered HTTP 500 out-of-capacity, and
    the REST API has no availability query, so the SET is the whole strategy."""
    card = cards_mod.pick("sm86")
    assert len(card.gpu_type_ids) >= 4 and "NVIDIA A40" in card.gpu_type_ids
    assert card.sm_expected == "8.6"


def test_a_missing_ssh_key_is_a_REFUSAL_with_a_row_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an authorized key the pod boots unreachable — a rental that could
    never be used — so it is refused before the create call, with a banked row."""
    monkeypatch.setenv("RUNPOD_SSH_PUBLIC_KEY_FILE", str(tmp_path / "nope.pub"))
    pods = FakePods()
    rig = _rig(tmp_path, pods, FakeTransport())
    rig.public_key = ""
    row = rig.run(cards_mod.pick("sm89"), Workload(name="w", command="x"))
    assert row.verdict == "refused" and row.failed_stage == "create"
    assert "no RunPod ssh public key" in row.detail
    assert pods.created_bodies == [], "nothing was rented"


# --------------------------------------------------------------------------- #
# 8. pgw#1352 / pgw#1348 rows 1-6 — the sm-clearance probe
# --------------------------------------------------------------------------- #


def test_the_probe_ships_a_TREE_and_proves_it_arrived_before_paying_for_a_compile(
    tmp_path: Path,
) -> None:
    """`micro_mint_rig` puts `<repo>/src` and `<repo>/tests` ahead of
    site-packages, so the code under test is the TREE; the wheel beside it only
    satisfies dependencies."""
    archive = tmp_path / "repo.tar.gz"
    archive.write_bytes(b"tarball")
    workload = sm_probe(archive)
    assert any(u.local == archive for u in workload.uploads)
    untar = [line for line in workload.setup if "tar -xzf" in line]
    assert untar and POD_REPO in untar[0]
    guard = [line for line in workload.setup if line.startswith("test -f")]
    assert guard and "micro_mint_rig.py" in guard[0] and "examples/micro-diffusion" in guard[0]
    # A cell that was never written is not a cleared sm, whatever the log said.
    assert workload.required_artifacts == (f"{POD_ROOT}/probe.json",)


def test_the_probe_needs_no_hub_which_is_why_these_rows_are_buyable_today(tmp_path: Path) -> None:
    """The micro vehicle publishes through an IN-PROCESS `LocalCellHub` and
    adopts in a second OS process on the same pod, so one rental proves compile,
    re-use and parity with nothing external to be blocked on. pgw#1348 gates
    every other leg A on a hub wire proof; this row is exempt by construction."""
    workload = sm_probe(tmp_path / "repo.tar.gz")
    assert "--vehicle micro" in workload.command and "--device cuda" in workload.command
    assert "--json" in workload.command, "the row IS the report; no report, no row"
    assert "nice -n 19" in workload.command
    for path in ("/root/.cache/torchinductor_root", POD_REPO):
        assert path in workload.progress_paths


def test_every_sm_class_the_gauntlet_names_has_a_card_set() -> None:
    """pgw#1348 rows 1-6 are sm_86, sm_89, sm_80, sm_90, sm_100, sm_120."""
    for slug, sm in (("sm86", "8.6"), ("sm89", "8.9"), ("sm80", "8.0"),
                     ("sm90", "9.0"), ("sm100", "10.0"), ("sm120", "12.0")):
        card = cards_mod.pick(slug)
        assert card.sm_expected == sm
        assert card.gpu_type_ids, "a pick names a SET; there is no capacity query to consult"


def test_unknown_card_names_the_ones_that_exist() -> None:
    with pytest.raises(KeyError, match="a40"):
        cards_mod.pick("nope")
