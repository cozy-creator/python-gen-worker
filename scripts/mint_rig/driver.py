"""The driver — rent, ship, run, capture, tear down, bank the row.

Read :mod:`mint_rig` first for the four rules. This module is where they are
actually enforced, in this order:

    rail declared            -> or the Rig cannot be constructed
    kill-set written         -> BEFORE the create call leaves
    podguard armed           -> or the create call is refused (th#1327)
    endpoint gate            -> pod REST record advancing
    ssh gate                 -> the pod answers, and every retry prints WHY
    host preflight           -> rigboot: native | compat | RE-ROLL (RIG-ENV 3c)
    census                   -> observed gpu / sm / driver, not the asked ones
    ship + setup             -> uploads digested, each setup line checked
    workload gate            -> log marker + artifact bytes as the progress token
    capture                  -> artifacts fetched whether it ended green or red
    teardown                 -> DELETE, GET 404, absent from the listing
    row                      -> written under every exit, including the bad ones

THE ONE STRUCTURAL RULE. Everything from the create call onward is inside a
`try/finally` whose `finally` tears down and writes the row. There is no early
`return` between them and no exception type that skips them. A rig whose
teardown is reachable only on the happy path is the th#1323 incident with extra
steps.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .cards import FLEET_CUDA, FLEET_IMAGE, Card
from .guard import Guard, real_guard
from .killset import ENTRYPOINT as KILLSET_ENTRYPOINT
from .killset import KillSet
from .progress import Failed, Gate, Observation, Stuck
from .rail import Rail, RailTripped
from .row import Artifact, RigRow
from .runpod import PodApi, PodNotFound, RunpodRest, dotenv
from .transport import Result, SshTransport, Transport
from .workload import POD_ROOT, Upload, Workload

#: RIG-ENV §3a: `pytorch/pytorch:*-runtime` ships no sshd and an entrypoint that
#: exits, so a rig must install one and block on it. Idempotent, because a pod
#: that restarts its container must come back reachable.
SSHD = (
    "set -e; mkdir -p /root/.ssh && chmod 700 /root/.ssh; "
    'printf "%s\\n" "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && '
    "chmod 600 /root/.ssh/authorized_keys; "
    "if [ ! -x /usr/sbin/sshd ] && ! command -v sshd >/dev/null 2>&1; then "
    "apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
    "apt-get install -y -qq openssh-server; fi; "
    "ssh-keygen -A 2>/dev/null || true; "
    "mkdir -p /run/sshd; exec /usr/sbin/sshd -D -e"
)

#: `nvidia-smi` is the only authority for what card we actually got.
CENSUS = (
    "nvidia-smi --query-gpu=name,driver_version,compute_cap "
    "--format=csv,noheader 2>/dev/null | head -1"
)


def _flushing_print(message: str) -> None:
    print(message, flush=True)


def _rest_token(record: Mapping[str, Any]) -> str:
    """The pod's REST record reduced to the fields that MOVE during bring-up.

    Whole-record equality would make an unrelated field's churn read as
    progress; a single field would make a pod that is pulling a 3 GB image read
    as stuck. These five are the ones that change exactly when something has.
    """
    machine = record.get("machine") or {}
    return json.dumps(
        {
            "desiredStatus": record.get("desiredStatus"),
            "lastStatusChange": record.get("lastStatusChange"),
            "publicIp": record.get("publicIp"),
            "ports": sorted((record.get("portMappings") or {}).items()),
            "gpu": machine.get("gpuTypeId"),
        },
        sort_keys=True,
    )


def _endpoint(record: Mapping[str, Any]) -> tuple[str, int] | None:
    ip = record.get("publicIp")
    port = (record.get("portMappings") or {}).get("22")
    return (str(ip), int(port)) if ip and port else None


def _section(text: str, name: str) -> str:
    """Pull one `--RIG-<name>--` section out of the probe's single round trip."""
    marks = list(re.finditer(r"^--RIG-([A-Z]+)--$", text, re.M))
    for index, mark in enumerate(marks):
        if mark.group(1) == name:
            end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
            return text[mark.end() : end].strip()
    return ""


@dataclass
class Rig:
    """One rental at a time. A matrix is a loop over this, not a feature of it."""

    rail: Rail
    lane: str
    api: PodApi | None = None
    guard: Guard | None = None
    api_key: str = ""
    issue: str = ""
    out_dir: Path = field(default_factory=lambda: Path.cwd() / "rig-runs")
    #: podguard's Layer-A staleness bound: how long the pod tolerates the renter
    #: not renewing before it reclaims ITSELF. Not a lifetime — the keeper
    #: renews every 60 s for as long as this process lives.
    lease_seconds: float = 900.0
    #: Injected so the whole driver is testable without a network.
    transport_factory: Callable[[str, int], Transport] | None = None
    sleep: Callable[[float], None] = time.sleep
    #: FLUSHED. A long rental is normally run with its stdout redirected to a
    #: file, and block buffering made the console silent for the whole run —
    #: which is indistinguishable from a hung driver at exactly the moment an
    #: operator is deciding whether to kill a pod.
    log: Callable[[str], None] = _flushing_print
    tick_s: float = 15.0
    stall_ticks: int = 12
    dry_run: bool = False
    #: SECURE | COMMUNITY | "" (let the provider choose). MEASURED 2026-08-17:
    #: SECURE has almost no sm_86 Ampere capacity — five SKUs in one create call
    #: all answered "this machine does not have the resources". A compile proof
    #: carries no weights and no tenant data, so COMMUNITY is a legitimate place
    #: to buy one; a lane that needs SECURE says so and pays for the scarcity.
    cloud_type: str = "SECURE"
    #: Fraction of the rail bring-up may consume before the pod has answered.
    #: See Rail.check_sub for why this phase is bounded by money and not by
    #: staleness: RunPod's REST record cannot distinguish an image pull from a
    #: wedged host, so there is no progress signal to be stuck on.
    boot_budget: float = 0.15

    def __post_init__(self) -> None:
        if not isinstance(self.rail, Rail):
            raise TypeError(
                "pgw#1347: Rig(rail=Rail(max_usd=...)) is mandatory. A rig that can start "
                "without a declared budget will eventually run without one."
            )
        if self.api is None:
            self.api_key = self.api_key or dotenv(("RUNPOD_API_KEY",)).get("RUNPOD_API_KEY", "")
            self.api = RunpodRest(self.api_key)
        if self.guard is None:
            self.guard = real_guard()
        if self.transport_factory is None:
            self.transport_factory = lambda host, port: SshTransport(host, port)

    # ---- the one public verb -------------------------------------------------

    def run(self, card: Card, workload: Workload, *, image: str = "", pod_name: str = "") -> RigRow:
        api, guard = self._api, self._guard
        name = pod_name or f"mintrig-{self.lane}-{int(time.time())}"
        image = image or FLEET_IMAGE
        row = RigRow(
            lane=self.lane,
            issue=self.issue,
            pod_name=name,
            image=image,
            asked_gpu=card.slug,
            asked_gpu_type_ids=list(card.gpu_type_ids),
            sm_expected=card.sm_expected,
            command=workload.command,
            workload_name=workload.name,
            workload_digest=workload.digest(),
            uploads=[u.record() for u in workload.uploads],
            rail_usd=self.rail.max_usd,
        )

        # RULE 2: the kill-set lands BEFORE the create call. If this box dies
        # between here and the response, the pod is still nameable.
        kill = KillSet(
            lane=self.lane, pod_name=name, gpu=card.slug, image=image, rail_usd=self.rail.max_usd
        ).save()
        row.killset_path = str(kill.path)
        self.log(f"[killset] {kill.path}\n[killset] stop it with: {kill.kill_by_name}")

        if self.dry_run:
            row.verdict, row.detail = "refused", "dry run: nothing was rented"
            # Nothing to tear down is a CONFIRMED teardown, not an unconfirmed
            # one: a dry run that printed "teardown=UNCONFIRMED" would train the
            # operator to ignore the one line that must never be ignored.
            row.teardown.get_404 = row.teardown.absent_from_list = True
            row.teardown.note = "dry run: no pod was created"
            kill.close(confirmed_dead=True, note="dry run")
            return self._bank(row)

        body = self._create_body(card, image, name)
        try:
            created = guard.rent(
                self.api_key,
                body,
                lane=f"mintrig-{self.lane}",
                lease_seconds=self.lease_seconds,
                orig_cmd=["/bin/bash", "-lc", SSHD],
                post=self._post_pod,
            )
        except Exception as exc:  # noqa: BLE001 — a create that half-happened is the point
            row.verdict, row.failed_stage, row.detail = "refused", "create", f"{type(exc).__name__}: {exc}"
            # The create-window leak. The response never landed, so the id is
            # unknown — but the NAME is ours, and the account listing has it.
            leaked = self._reconcile_by_name(name)
            kill.close(confirmed_dead=not leaked, note=f"create failed; leaked={leaked}")
            row.detail += f"; name-sweep found {'A LEAKED POD' if leaked else 'nothing'}"
            return self._bank(row)

        pod_id = str(created.get("id") or "")
        rate = float(created.get("costPerHr") or 0.0)
        row.pod_id, row.rate_per_hr = pod_id, rate
        row.started_at = time.time()
        self.rail.observe_rate(rate)
        self.rail.clock_started(row.started_at)
        kill.arm(pod_id, rate_per_hr=rate)
        wall = self.rail.wall_seconds
        self.log(
            f"[pod] {pod_id} ({name}) rate=${rate}/hr — the ${self.rail.max_usd:.2f} rail is "
            f"{wall / 60:.0f} min at that rate\n[pod] stop it with: {kill.kill_by_id}"
        )

        try:
            self._bring_up_and_run(api, row, workload, card)
        except RailTripped as exc:
            row.verdict, row.rail_tripped, row.detail = "railed", True, str(exc)
            row.failed_stage = row.failed_stage or "rail"
        except Stuck as exc:
            row.verdict, row.failed_stage, row.detail = "stuck", exc.stage, str(exc)
        except Failed as exc:
            row.verdict, row.failed_stage, row.detail = "red", exc.stage, str(exc)
        except _Reroll as exc:
            row.verdict, row.failed_stage, row.detail = "reroll", "preflight", str(exc)
        except Exception as exc:  # noqa: BLE001
            row.verdict = "red"
            row.failed_stage = row.failed_stage or "driver"
            row.detail = f"{type(exc).__name__}: {exc}"
        finally:
            # RULE 4, and the structural rule: teardown is reachable from every
            # exit, including the ones nobody predicted.
            self._teardown(api, row)
            kill.close(confirmed_dead=row.teardown.confirmed, note=row.verdict)
            self._guard.release(pod_id, confirmed_dead=row.teardown.confirmed, reason=f"mint-rig {row.verdict}")
            row.price()
            self._bank(row)
        return row

    # ---- terminate / sweep, for the CLI and for recovery ---------------------

    def terminate(self, *, pod_id: str = "", name: str = "") -> RigRow:
        row = RigRow(lane=self.lane, pod_id=pod_id, pod_name=name, verdict="refused")
        if not pod_id and name:
            found = self._find_by_name(name)
            pod_id = row.pod_id = found or ""
        if not pod_id:
            row.detail = "no such pod on the account"
            row.verdict = "green"  # nothing to stop is the outcome we wanted
            row.teardown.get_404 = row.teardown.absent_from_list = True
            return row
        self._teardown(self._api, row)
        row.verdict = "green" if row.teardown.confirmed else "red"
        self._guard.release(pod_id, confirmed_dead=row.teardown.confirmed, reason="mint-rig terminate")
        for record in KillSet.open_records():
            if record.get("pod_id") == pod_id or record.get("pod_name") == name:
                KillSet(lane=str(record.get("lane", "")), pod_name=str(record.get("pod_name", ""))).close(
                    confirmed_dead=row.teardown.confirmed, note="terminated by hand"
                )
        return row

    def sweep(self, *, leases: Mapping[str, str] | None = None) -> dict[str, Any]:
        """What is running, and what anyone's records think is running.

        The interesting output is the DISAGREEMENT. `unattended` is the alarm:
        a live pod that NEITHER this package's kill-sets NOR podguard's lease
        records name — the th#1323 shape, a pod burning money with no owner.

        A pod another lane rented through podguard is attended and is reported
        as such, not as a leak. Reading only our own records would make every
        sibling lane's healthy pod look like an emergency, and an alarm that
        cries wolf is one nobody reads.
        """
        live = {str(p.get("id")): p for p in self._api.list_pods()}
        records = KillSet.open_records()
        recorded = {str(r.get("pod_id")) for r in records if r.get("pod_id")}
        held = dict(leases) if leases is not None else _podguard_leases()
        return {
            "live_pods": [
                {
                    "pod_id": pid,
                    "name": p.get("name"),
                    "status": p.get("desiredStatus"),
                    "rate_per_hr": p.get("costPerHr"),
                    "known_to_mint_rig": pid in recorded,
                    "podguard_lane": held.get(pid, ""),
                }
                for pid, p in sorted(live.items())
            ],
            "open_killsets": records,
            "unrecorded_live": [pid for pid in live if pid not in recorded],
            "unattended": [pid for pid in live if pid not in recorded and pid not in held],
            "open_but_dead": [str(r.get("pod_id")) for r in records if str(r.get("pod_id")) not in live],
        }

    # ---- internals -----------------------------------------------------------

    @property
    def _api(self) -> PodApi:
        assert self.api is not None
        return self.api

    @property
    def _guard(self) -> Guard:
        assert self.guard is not None
        return self.guard

    def _post_pod(self, armed: Mapping[str, Any]) -> dict[str, Any]:
        """Create, moving podguard's start command onto the ENTRYPOINT.

        RunPod treats `dockerEntrypoint: []` as unset and lets the IMAGE's
        entrypoint win, so an armed `dockerStartCmd` on an image with its own
        entrypoint runs the image's command and never our sshd. pod_run.py
        learned this; the fix belongs in the shared primitive, not in each lane.
        """
        body = dict(armed)
        start = body.pop("dockerStartCmd", None)
        if start:
            body["dockerEntrypoint"] = start
        return self._api.create(body)

    def _create_body(self, card: Card, image: str, name: str) -> dict[str, Any]:
        public_key = (Path.home() / ".ssh" / "runpod.pub").read_text().strip()
        return {
            "name": name,
            "imageName": image,
            "gpuTypeIds": list(card.gpu_type_ids),
            "gpuCount": 1,
            "containerDiskInGb": card.disk_gb,
            **({"cloudType": self.cloud_type} if self.cloud_type else {}),
            "supportPublicIp": True,
            "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": public_key},
        }

    def _gate(
        self,
        stage: str,
        probe: Callable[[], Observation],
        *,
        stall_ticks: int | None = None,
        rail_check: Callable[[str], None] | None = None,
    ) -> Observation:
        return Gate(
            stage=stage,
            probe=probe,
            stall_ticks=self.stall_ticks if stall_ticks is None else stall_ticks,
            tick_s=self.tick_s,
            rail_check=rail_check or self.rail.check,
            sleep=self.sleep,
            on_tick=lambda n, obs: self.log(f"[{stage}] {n:3d} {obs.note[:150]}"),
        ).wait()

    def _boot_check(self, stage: str) -> None:
        self.rail.check_sub(stage, self.boot_budget)

    def _bring_up_and_run(self, api: PodApi, row: RigRow, workload: Workload, card: Card) -> None:
        row.stage("endpoint")
        endpoint = self._wait_for_endpoint(api, row)
        transport = self._transport(*endpoint)

        row.stage("ssh")
        self._wait_for_ssh(api, row, transport)

        row.stage("preflight")
        self._preflight(transport, row)

        row.stage("census")
        self._census(transport, row, card)

        row.stage("ship")
        self.rail.may_start("ship")
        self._ship(transport, workload, row)

        row.stage("setup")
        self.rail.may_start("setup")
        self._setup(transport, workload, row)

        row.stage("launch")
        self.rail.may_start(f"workload:{workload.name}")
        launched = transport.run(workload.launch_script())
        if "RIG_LAUNCHED" not in launched.out:
            raise Failed("launch", f"rc={launched.rc} {launched.out[-500:]}")
        self.log(f"[launch] {launched.out.strip()[-200:]}")

        row.stage("workload")
        try:
            self._watch(transport, workload, row)
            row.verdict = "green"
        finally:
            # Capture under EVERY outcome. A failed compile's log is the most
            # valuable thing the rental produced, and it is on a pod that is
            # about to be deleted.
            row.stage("capture")
            self._capture(transport, workload, row)

    def _transport(self, host: str, port: int) -> Transport:
        assert self.transport_factory is not None
        return self.transport_factory(host, port)

    def _wait_for_endpoint(self, api: PodApi, row: RigRow) -> tuple[str, int]:
        found: dict[str, tuple[str, int]] = {}

        def probe() -> Observation:
            record = api.get(row.pod_id)
            endpoint = _endpoint(record)
            if endpoint:
                found["ep"] = endpoint
            return Observation(
                reached=bool(endpoint),
                token=_rest_token(record),
                note=f"status={record.get('desiredStatus')} ip={record.get('publicIp')} "
                f"ports={(record.get('portMappings') or {})}",
            )

        # Staleness DISABLED, money is the bound — see Rail.check_sub.
        self._gate("endpoint", probe, stall_ticks=0, rail_check=self._boot_check)
        return found["ep"]

    def _wait_for_ssh(self, api: PodApi, row: RigRow, transport: Transport) -> None:
        """The pod ANSWERING is the liveness signal — not the REST status.

        The progress token pairs the REST record with ssh's own stderr, because
        during a slow image pull the REST record is what moves, and after the
        container starts it is the ssh error that changes ("Connection refused"
        -> "Connection closed by remote host" -> success). A token built from
        only one of them goes stale during the other's phase.
        """

        def probe() -> Observation:
            result = transport.run("true", timeout_s=45)
            record = api.get(row.pod_id)
            reason = result.out.strip().splitlines()[-1][:120] if result.out.strip() else ""
            return Observation(
                reached=result.ok,
                token=(_rest_token(record), reason),
                # Print the reason EVERY tick. A retry loop that hides ssh's
                # stderr cannot tell "still booting" from "wrong key".
                note=f"rc={result.rc} {reason}",
            )

        self._gate("ssh", probe, stall_ticks=0, rail_check=self._boot_check)

    def _preflight(self, transport: Transport, row: RigRow) -> None:
        """RIG-ENV §3c — probe the HOST driver before anything is downloaded.

        RunPod's driver version is per-host. A 570 host imports torch+cu130
        perfectly and then dies on the first allocation, twenty minutes in.
        `rigboot` is stdlib-only, so it ships as one file and runs before any
        install: exit 0 usable (path native|compat), 91 RE-ROLL, 92 no driver.
        """
        boot = Path(__file__).resolve().parents[2] / "src" / "gen_worker" / "rigboot.py"
        if not boot.is_file():  # pragma: no cover - a checkout without src/
            self.log("[preflight] rigboot.py not found in this checkout — SKIPPED")
            return
        transport.run(f"mkdir -p {POD_ROOT}")
        shipped = transport.put([boot], POD_ROOT)
        if not shipped.ok:
            raise Failed("preflight", f"could not ship rigboot.py: {shipped.out[-400:]}")
        # rigboot prints its record AND writes it to --json, so a naive
        # "first { to last }" parse spans two documents and silently yields
        # nothing — which is how the first real run reported an empty driver
        # version next to a perfectly good probe. Send its console to a file and
        # read the record from exactly one place.
        result = transport.run(
            f"cd {POD_ROOT} && python3 rigboot.py --cuda {FLEET_CUDA} "
            f"--json {POD_ROOT}/rigboot.json > {POD_ROOT}/rigboot.out 2>&1; "
            f"echo RIG_RC=$?; cat {POD_ROOT}/rigboot.json 2>/dev/null; "
            f"echo '--RIG-BOOTLOG--'; tail -20 {POD_ROOT}/rigboot.out",
            timeout_s=900,
        )
        rc_match = re.search(r"RIG_RC=(\d+)", result.out)
        rc = int(rc_match.group(1)) if rc_match else result.rc
        head, _, bootlog = result.out.partition("--RIG-BOOTLOG--")
        record: dict[str, Any] = {}
        brace = head.find("{")
        if brace >= 0:
            try:
                record = json.loads(head[brace : head.rfind("}") + 1])
            except json.JSONDecodeError:
                record = {}
        row.cuda_path = str(record.get("path") or ("native" if rc == 0 else "reroll"))
        row.driver_version = str(record.get("driver") or record.get("driver_version") or "")
        self.log(f"[preflight] rc={rc} path={row.cuda_path} driver={row.driver_version}")
        if rc == 91:
            raise _Reroll(
                f"host driver {row.driver_version} has no usable path to CUDA {FLEET_CUDA} "
                "(RIG-ENV §3c: re-roll the host, do NOT downgrade torch)"
                f"\n{bootlog[-500:]}"
            )
        if rc == 92:
            raise _Reroll("no NVIDIA driver on this host — wrong pod type")
        if rc != 0:
            raise Failed("preflight", f"rigboot rc={rc}: {bootlog[-800:] or result.out[-800:]}")

    def _census(self, transport: Transport, row: RigRow, card: Card) -> None:
        """What we ACTUALLY got. Evidence, never a gate.

        A probe that cannot read a version must not be able to delete a rented
        pod — so a failed census is a blank column, not a teardown.
        """
        result = transport.run(CENSUS, timeout_s=120)
        line = result.out.strip().splitlines()[-1] if result.out.strip() else ""
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            row.observed_gpu, row.driver_version, row.observed_sm = parts[0], parts[1], parts[2]
        row.env_line = [line]
        self.log(f"[census] {line}")
        if row.observed_sm and row.observed_sm != card.sm_expected:
            # NOT a failure: the pick names a card SET and the provider chose.
            # It is a fact the row must carry so nobody reads the asked sm.
            self.log(
                f"[census] asked {card.slug} (sm {card.sm_expected}) — got sm {row.observed_sm}. "
                "The row records both."
            )

    def _ship(self, transport: Transport, workload: Workload, row: RigRow) -> None:
        by_dir: dict[str, list[Path]] = {}
        for upload in workload.uploads:
            by_dir.setdefault(upload.remote_dir, []).append(upload.local)
        for remote_dir, paths in sorted(by_dir.items()):
            transport.run(f"mkdir -p {remote_dir}")
            result = self._retry_transfer(lambda: transport.put(paths, remote_dir), f"ship->{remote_dir}")
            if not result.ok:
                raise Failed("ship", f"scp to {remote_dir} rc={result.rc}: {result.out[-500:]}")
        self.log(f"[ship] {len(workload.uploads)} upload(s)")

    def _retry_transfer(self, call: Callable[[], Result], label: str) -> Result:
        """A freshly booted sshd drops connections; one broken pipe must not
        cost a rented pod. Bounded by attempts and by the rail, never by a wall
        clock, and every attempt prints its own reason."""
        result = Result(1, "")
        for attempt in range(5):
            self.rail.check(label)
            result = call()
            if result.ok:
                return result
            self.log(f"[{label}] attempt {attempt} rc={result.rc}: {result.out.strip()[-200:]}")
            self.sleep(self.tick_s)
        return result

    def _setup(self, transport: Transport, workload: Workload, row: RigRow) -> None:
        for index, line in enumerate(workload.setup):
            self.rail.may_start(f"setup[{index}]")
            self.log(f"[setup {index}] {line[:160]}")
            result = transport.run(f"mkdir -p {workload.workdir} && cd {workload.workdir} && {line}", timeout_s=3600)
            if not result.ok:
                raise Failed("setup", f"line {index} rc={result.rc}: {line}\n{result.out[-800:]}")

    def _watch(self, transport: Transport, workload: Workload, row: RigRow) -> None:
        script = workload.probe_script()

        def probe() -> Observation:
            result = transport.run(script, timeout_s=180)
            if not result.ok:
                # An ssh hiccup is NOT evidence about the compile. Return the
                # previous token unchanged-but-marked so a run of them still
                # trips the stall detector while a single one does not.
                return Observation(token=("ssh-error", result.out[-80:]), note=f"probe rc={result.rc}")
            size = _section(result.out, "SIZE") or "0"
            log_bytes = _section(result.out, "BYTES") or "0"
            done = (_section(result.out, "MARK") or "0").strip() not in ("", "0")
            failed = any(part.strip() not in ("", "0") for part in _section(result.out, "FAIL").split())
            tail = _section(result.out, "TAIL")
            last = tail.strip().splitlines()[-1] if tail.strip() else ""
            return Observation(
                reached=done,
                failed=failed and not done,
                token=(size.strip(), log_bytes.strip(), last),
                note=f"{int(size or 0) / 2**20:.0f} MiB artifacts, {log_bytes}B log | {last[:110]}",
            )

        self._gate(f"workload:{workload.name}", probe)

    def _capture(self, transport: Transport, workload: Workload, row: RigRow) -> None:
        destination = self.out_dir / f"{row.pod_name}"
        destination.mkdir(parents=True, exist_ok=True)
        for remote in (*workload.artifacts, workload.log):
            artifact = Artifact(remote=remote)
            result = transport.fetch(remote, destination)
            base = destination / Path(remote).name
            artifact.fetched = result.ok and base.exists()
            artifact.local = str(base) if artifact.fetched else ""
            artifact.note = "" if artifact.fetched else result.out.strip()[-200:]
            if artifact.fetched and base.is_file():
                from .workload import sha256_file

                artifact.bytes = base.stat().st_size
                artifact.sha256 = sha256_file(base)
            row.artifacts.append(artifact.__dict__)
        self.log(f"[capture] {sum(1 for a in row.artifacts if a['fetched'])}/{len(row.artifacts)} -> {destination}")

    def _teardown(self, api: PodApi, row: RigRow) -> None:
        """DELETE, then GET 404, then absent from the listing. All three.

        Each one alone has lied: DELETE answers 200 for a pod still winding
        down, and a single GET can be answered from a cache while the account
        listing still carries the pod that is still being billed.
        """
        teardown = row.teardown
        if not row.pod_id:
            teardown.note = "no pod id — nothing was created"
            teardown.get_404 = teardown.absent_from_list = True
            return
        try:
            api.delete(row.pod_id)
            teardown.delete_issued = True
        except Exception as exc:  # noqa: BLE001
            teardown.note = f"DELETE raised {type(exc).__name__}: {exc}"
        for attempt in range(30):
            teardown.attempts = attempt + 1
            try:
                api.get(row.pod_id)
            except PodNotFound:
                teardown.get_404 = True
            except Exception as exc:  # noqa: BLE001
                teardown.note = f"GET raised {type(exc).__name__}: {exc}"
            try:
                teardown.absent_from_list = not any(
                    str(p.get("id")) == row.pod_id for p in api.list_pods()
                )
            except Exception as exc:  # noqa: BLE001
                teardown.note = f"list raised {type(exc).__name__}: {exc}"
            if teardown.confirmed:
                break
            self.sleep(self.tick_s)
        self.log(
            f"[teardown] {row.pod_id} delete={teardown.delete_issued} 404={teardown.get_404} "
            f"absent={teardown.absent_from_list} after {teardown.attempts} check(s)"
        )
        if not teardown.confirmed:
            self.log(
                f"[teardown] ⚠ UNCONFIRMED — the pod may still be billing. "
                f"Stop it: python3 {KILLSET_ENTRYPOINT} terminate --pod {row.pod_id}"
            )

    def _find_by_name(self, name: str) -> str:
        for pod in self._api.list_pods():
            if str(pod.get("name")) == name:
                return str(pod.get("id"))
        return ""

    def _reconcile_by_name(self, name: str) -> bool:
        """The create-window sweep. Returns True when a leaked pod was found."""
        try:
            pod_id = self._find_by_name(name)
        except Exception as exc:  # noqa: BLE001
            self.log(f"[create] name sweep failed ({exc}) — VERIFY {name} BY HAND")
            return True
        if not pod_id:
            return False
        self.log(f"[create] the create call failed but pod {pod_id} EXISTS under our name — killing it")
        row = RigRow(lane=self.lane, pod_id=pod_id, pod_name=name)
        self._teardown(self._api, row)
        return not row.teardown.confirmed

    def _bank(self, row: RigRow) -> RigRow:
        path = self.out_dir / f"{row.pod_name or row.lane}.row.json"
        row.write(path)
        self.log(f"[row] {path}\n{row.one_line()}")
        return row


def _podguard_leases() -> dict[str, str]:
    """Live podguard leases as {pod_id: lane}. Absent podguard means {}: the
    sweep degrades to our own records rather than refusing to report."""
    try:
        from .runpod import load_podguard

        podguard = load_podguard()
    except Exception:  # noqa: BLE001 — a missing peer must not silence the sweep
        return {}
    out: dict[str, str] = {}
    for lease in podguard.Lease.all():
        if lease.state == "LIVE":
            out[str(lease.pod_id)] = str(lease.lane)
    return out


class _Reroll(RuntimeError):
    """The host cannot run the fleet CUDA line. Kill it and create another."""


def uploads_for(paths: Sequence[Path], remote_dir: str = POD_ROOT) -> tuple[Upload, ...]:
    return tuple(Upload(local=p, remote_dir=remote_dir) for p in paths)
