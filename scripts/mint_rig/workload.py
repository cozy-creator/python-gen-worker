"""What to run on the pod — a VALUE, so a new lane writes data, not a driver.

A :class:`Workload` says: what to put on the pod, how to make the code under
test importable there, the one named command to run, which paths prove progress
while it runs, which marker means done, and what to bring home. Nothing about
renting, waiting or paying appears here.

THE THREE DELIVERY MODES, and why all three exist.

``image``   the code is already installed in the image. This is the route
            pgw#1346's migrated endpoints take: the endpoint image IS the
            subject, and installing anything over it would measure a different
            artifact than the fleet runs.
``wheel``   ``pip install gen-worker==X`` from PyPI. The route for proving a
            RELEASE — what a pod built from a published wheel actually does.
``sdist``   a dist file built from the worktree and shipped. The route for
            proving a LANE, and it is not a convenience: pgw#1337 has the wheel
            cut blocked, so a lane whose surface is newer than PyPI has no
            other honest way to reach a card. The row records the dist's sha256,
            so "which code ran" is a digest rather than a claim.

WHAT A PROGRESS PATH IS FOR. :mod:`mint_rig.progress` needs a token that
advances for the slowest legitimate work. For a compile the log alone is not
enough — inductor can print nothing for ten minutes while it writes objects —
so `progress_paths` names the trees whose byte size is polled alongside the
log's tail. A path that does not exist yet contributes ``0`` and is not an
error: an artifact directory appearing IS progress.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

#: Where the rig works on the pod. One root, so `--artifacts` and the sweep's
#: recovery instructions can name it without a per-lane convention.
POD_ROOT = "/root/rig"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Upload:
    """One local path shipped to the pod, with the digest of what was shipped."""

    local: Path
    remote_dir: str = POD_ROOT

    @property
    def sha256(self) -> str:
        return sha256_file(self.local) if self.local.is_file() else ""

    def record(self) -> dict[str, str]:
        return {"local": str(self.local), "remote_dir": self.remote_dir, "sha256": self.sha256}


@dataclass(frozen=True)
class Workload:
    """One named command and everything it needs."""

    name: str
    #: The named command. Exactly one, run detached and watched.
    command: str
    #: Shell lines run before the command, in order. Each is checked; the first
    #: non-zero rc fails the setup stage and names the line.
    setup: tuple[str, ...] = ()
    uploads: tuple[Upload, ...] = ()
    #: Remote paths fetched back after the command ends — whether it ended
    #: green or red. A failed compile's log is the most valuable thing the
    #: rental produced.
    artifacts: tuple[str, ...] = ()
    #: Artifacts whose ABSENCE makes the run red no matter what the log said.
    #: A second, independent statement of "it worked": the done marker says the
    #: command believed it succeeded, this says it left the thing behind. They
    #: fail independently, which is the whole point — the first real green this
    #: rig ever printed was a command that died at line one.
    required_artifacts: tuple[str, ...] = ()
    #: Remote paths whose byte size is polled as a progress token.
    progress_paths: tuple[str, ...] = ()
    #: Printed by the command when it has finished successfully.
    done_marker: str = "RIG_DONE"
    #: Any of these in the log ends the gate as FAILED, immediately.
    fail_markers: tuple[str, ...] = ("Traceback (most recent call last)", "RIG_FAIL")
    env: Mapping[str, str] = field(default_factory=dict)
    workdir: str = POD_ROOT

    @property
    def log(self) -> str:
        return f"{self.workdir}/{self.name}.log"

    def with_uploads(self, *uploads: Upload) -> "Workload":
        return replace(self, uploads=self.uploads + uploads)

    # ---- the launch line
    def launch_script(self) -> str:
        """Start the command DETACHED and return immediately.

        `setsid nohup … &` is not incidental: a command run in the foreground of
        an ssh session dies with the session, and an ssh session on a freshly
        booted pod dies for reasons that have nothing to do with the work. The
        gate then watches the log, so the control box may lose its connection
        for any length of time without costing the compile.
        """
        exports = "".join(f"export {k}={json.dumps(v)}; " for k, v in sorted(self.env.items()))
        # A NON-ZERO EXIT MUST LEAVE A MARK. Measured 2026-08-17: the mint
        # command's `rigcheck` leg aborted with a one-line refusal and no
        # traceback, so the log simply stopped growing — and a gate that can only
        # see "the token froze" then paid a full stall budget of rented pod to
        # learn what the exit code had already said. The wrapper turns every
        # non-zero exit into the FAIL marker the gate reads on its next tick.
        wrapped = f"{{ {self.command}; }}; rc=$?; [ \"$rc\" -eq 0 ] || echo RIG_FAIL rc=$rc"
        return (
            f"mkdir -p {self.workdir} && cd {self.workdir} && "
            # The forward-compat libcuda binds only at process start and a
            # detached job inherits no login shell — RIG-ENV §3c.
            "{ [ -f /etc/profile.d/zz-cuda-compat.sh ] && . /etc/profile.d/zz-cuda-compat.sh; }; "
            f"{exports}"
            # SINGLE-quoted, not json.dumps'd. This string crosses TWO shells —
            # the one ssh starts on the pod, and the `bash -lc` it launches — and
            # a double-quoted payload lets the FIRST one expand the SECOND one's
            # variables. Measured: `rc=$?` arrived as `rc=` (the outer shell had
            # already substituted an unset `$rc`), so the guard read
            # `[ -eq 0 ]`, errored, and printed RIG_FAIL after every successful
            # run. Single quotes suppress all expansion, so the payload reaches
            # bash exactly as written. `$!` below is OUTSIDE the quotes and is
            # meant for the outer shell.
            f"setsid nohup bash -lc {shlex.quote(wrapped)} > {self.log} 2>&1 & "
            "echo RIG_LAUNCHED $!"
        )

    def probe_script(self, tail_lines: int = 3) -> str:
        """One round trip that returns everything a gate observation needs.

        Deliberately ONE call: three separate ssh round-trips per tick is three
        chances to mistake a flaky connection for a stalled compile.

        **`|| true`, never `|| echo 0`.** `grep -c` prints its count AND exits 1
        when the count is zero, so `grep -c X f || echo 0` emits the two lines
        "0\n0" for a log with no match. That is the bug that made this rig report
        GREEN for a run whose command died at its first line: the done check read
        the section as a single value, saw "0\n0" != "0", and called it a match.
        A rig that can fabricate a success is worse than no rig, so the shape is
        fixed at BOTH ends — the script never emits a second value, and
        :func:`mint_rig.driver._count` sums whatever it gets.
        """
        paths = " ".join(f"'{p}'" for p in (self.progress_paths or (self.workdir,)))
        return (
            f"echo '--RIG-SIZE--'; du -sb {paths} 2>/dev/null | awk '{{s+=$1}} END {{print s+0}}'; "
            f"echo '--RIG-BYTES--'; {{ wc -c < '{self.log}' 2>/dev/null || true; }}; "
            f"echo '--RIG-MARK--'; {{ grep -c -F '{self.done_marker}' '{self.log}' 2>/dev/null || true; }}; "
            f"echo '--RIG-FAIL--'; "
            + "; ".join(
                f"{{ grep -c -F {json.dumps(m)} '{self.log}' 2>/dev/null || true; }}"
                for m in self.fail_markers
            )
            + f"; echo '--RIG-TAIL--'; tail -{tail_lines} '{self.log}' 2>/dev/null || true"
        )

    def digest(self) -> str:
        """A digest of the whole intent, uploads included.

        Two rows with the same `workload_digest` ran the same thing. That is the
        property a matrix needs and the one a prose 'command' column cannot give.
        """
        body: dict[str, Any] = {
            "name": self.name,
            "command": self.command,
            "setup": list(self.setup),
            "artifacts": list(self.artifacts),
            "progress_paths": list(self.progress_paths),
            "done_marker": self.done_marker,
            "env": dict(sorted(self.env.items())),
            "workdir": self.workdir,
            "uploads": [u.record() for u in self.uploads],
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


# --- the delivery modes ------------------------------------------------------


#: Generates a pip constraint file from the torch that is ALREADY INSTALLED.
#: RIG-ENV §3a: "anything that lists torch as a dependency is installed
#: --no-deps or from the SAME cu130 index. This is the single most common way a
#: rig drifts." Deriving the constraint from the interpreter is stronger than
#: either — the pin cannot be stale, because it is read from the thing it pins.
PIN_TORCH = (
    "python3 -c \"import torch,pathlib;"
    f"pathlib.Path('{POD_ROOT}/constraints.txt')"
    ".write_text('torch=='+torch.__version__+chr(10))\""
)


def _pip_install(target: str, index_args: str = "") -> str:
    # `--no-input` so a prompt is a failure rather than a hang, and `-q` so the
    # log's byte growth means work rather than progress bars.
    # `--break-system-packages`: MEASURED 2026-08-17 on
    # pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime — its interpreter is a
    # Debian EXTERNALLY-MANAGED one, so PEP 668 refuses the install outright and
    # tells you to build a venv. A venv is the wrong answer here: the image's
    # site-packages is where the fleet's own torch lives, and an isolated venv
    # would either shadow it or re-resolve it from an index, which is RIG-ENV
    # §3a's single most common way a rig drifts. The pod is disposable and
    # single-purpose; breaking "the system" is the intended outcome.
    parts = [
        "pip install -q --no-input --break-system-packages",
        f"-c {POD_ROOT}/constraints.txt",
    ]
    if index_args:
        parts.append(index_args)
    # QUOTED: an extras suffix (`…whl[torch]`) is a glob pattern to bash, and an
    # unmatched glob is passed through only by accident of the shell's settings.
    parts.append(f"'{target}'")
    return " ".join(parts)


def install_sdist(dist: Path, extras: str = "", index_args: str = "") -> tuple[str, ...]:
    """Install the code under test from a locally built dist file.

    `--no-deps` is deliberately NOT used: a model DECLARATION builds its
    architecture with the model library (diffusers is allowed inside `build`),
    so an install that skipped dependencies could not trace anything.
    """
    target = f"{POD_ROOT}/{dist.name}" + (f"[{extras}]" if extras else "")
    return (PIN_TORCH, _pip_install(target, index_args))


def install_wheel(spec: str, index_args: str = "") -> tuple[str, ...]:
    """Install a PUBLISHED distribution — the route that proves a RELEASE.

    `spec` is a pip requirement verbatim, e.g. ``gen-worker[torch]==0.121.0``,
    so the caller states extras and version in the one place pip already has a
    grammar for.
    """
    return (PIN_TORCH, _pip_install(spec, index_args))


#: pgw#1352 / pgw#1348 rows 1-6. Where the shipped tree lands on the pod.
POD_REPO = f"{POD_ROOT}/repo"


def sm_probe(
    archive: Path,
    *,
    vehicle: str = "micro",
    install: tuple[str, ...] = (),
    uploads: tuple[Upload, ...] = (),
    name: str = "probe",
) -> Workload:
    """One sm-clearance probe — [[pgw#1348]] rows 1-6, on the bare-pod route.

    WHY THIS VEHICLE NEEDS NO HUB, which is the whole reason these six rows are
    buyable today while the other fifty are gated. `scripts/micro_mint_rig.py
    --vehicle micro` runs the FULL production cycle against
    `examples/micro-diffusion` — resolve, `MintSlot` handoff, a real child
    interpreter, warmup, `torch.export` + AOTInductor, seal, publish, and a
    SECOND OS process adopting the exact named cell and comparing every arm to
    eager — and its publish leg goes to `harness.cell_hub.LocalCellHub`,
    IN-PROCESS. So one pod proves compile AND re-use AND parity on that card,
    with nothing external to be blocked on.

    It ships the repository's own tree rather than a wheel, because the rig
    inserts `<repo>/src` and `<repo>/tests` ahead of site-packages: the code
    under test is the TREE, and the wheel is installed only to satisfy its
    dependencies. Both digests land on the row.
    """
    root = f"{POD_ROOT}/micro-root"
    report = f"{POD_ROOT}/{name}.json"
    # `--force-load`, with the reason the flag asks for: `micro_mint_rig`'s load
    # gate refuses above 1-minute load 24 because THIS BOX is shared with other
    # agent sessions. A rented pod is the opposite of that — it is dedicated,
    # single-purpose and disposable, and its load is high precisely BECAUSE the
    # install we just ran finished seconds ago. MEASURED on pgw#1348 row 2: the
    # gate refused at load 26.2 on a pod nobody else could touch, after four
    # re-rolls had already been paid for to reach a host that worked.
    command = (
        f"cd {POD_REPO} && nice -n 19 python3 scripts/micro_mint_rig.py "
        f"--vehicle {vehicle} --device cuda --clean --force-load --root {root} "
        f"--json {report} && echo RIG_DONE"
    )
    return Workload(
        name=name,
        command=command,
        setup=install
        + (
            f"mkdir -p {POD_REPO} && tar -xzf {POD_ROOT}/{archive.name} -C {POD_REPO}",
            # Proof the tree arrived whole, before a compile is paid for.
            f"test -f {POD_REPO}/scripts/micro_mint_rig.py "
            f"&& test -d {POD_REPO}/examples/micro-diffusion "
            f"&& test -d {POD_REPO}/tests/harness",
        ),
        uploads=uploads + (Upload(local=archive),),
        artifacts=(report, f"{POD_ROOT}/{name}.log", f"{POD_ROOT}/rigboot.json"),
        # A cell that was never written is not a cleared sm, whatever the log says.
        required_artifacts=(report,),
        progress_paths=(root, "/root/.cache/torchinductor_root", "/root/.triton", POD_REPO),
        env={"TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torchinductor_root"},
    )


def mint_model(
    target: str,
    *,
    runners: tuple[str, ...] = (),
    install: tuple[str, ...] = (),
    uploads: tuple[Upload, ...] = (),
    fleet_line: Path | None = None,
    name: str = "mint",
) -> Workload:
    """pgw#1331's owed leg as a workload value.

    `gen-worker model mint` needs a GPU and a real toolchain and needs NO
    weights and NO network for the model — cell identity is checkpoint-free
    (§4.27) and the constants arrive at arm time from the store. That is what
    makes a per-model mint pod cheap enough to be routine, and it is why this
    workload ships no checkpoint machinery at all.
    """
    only = " ".join(f"--runner {r}" for r in runners)
    out = f"{POD_ROOT}/cells"
    # THE FLEET-LINE AUTHORITY HAS TO BE SHIPPED, and finding that out cost a
    # pod. RIG-ENV §2: rigcheck reads endpoint.toml / fleet-floors.toml /
    # ENDPOINT dist metadata, and deliberately does NOT accept `gen-worker`'s own
    # requirement — an SDK certifying its own floor makes every rig pass. This
    # repository contains none of those files, so a bare pod carrying only
    # gen-worker aborts `FleetLineUnknown` before it compiles anything. The
    # workspace's authority is ~/cozy/serverless-endpoints/fleet-floors.toml (and
    # a per-endpoint endpoint.toml, which is the one that also declares CUDA).
    env: dict[str, str] = {"TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torchinductor_root"}
    if fleet_line is not None:
        uploads = uploads + (Upload(local=fleet_line),)
        env["GEN_WORKER_FLEET_LINE_FILE"] = f"{POD_ROOT}/{fleet_line.name}"
    command = (
        # rigcheck FIRST: a mint measured off the fleet line is RIG-ENV §5's
        # two false verdicts, and its exit 90/91 is a cheaper answer than a
        # compile that succeeds against the wrong torch.
        "python3 -m gen_worker.rigcheck && "
        f"gen-worker model mint {target} --out-dir {out} {only} "
        f"--json {POD_ROOT}/minted.json && echo RIG_DONE"
    )
    return Workload(
        name=name,
        command=command,
        setup=install,
        uploads=uploads,
        artifacts=(f"{POD_ROOT}/minted.json", f"{POD_ROOT}/{name}.log", f"{POD_ROOT}/rigboot.json"),
        # A mint that produced no row produced nothing, whatever the log claims.
        required_artifacts=(f"{POD_ROOT}/minted.json",),
        # The AOTI object cache and the packed cells both grow throughout the
        # compile; the log can be silent for minutes while they do.
        progress_paths=(out, f"{POD_ROOT}/work", "/root/.cache/torchinductor_root", "/root/.triton"),
        env=env,
    )
