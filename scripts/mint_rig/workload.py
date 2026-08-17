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
        wrapped = f"{{ {self.command}; }}; rc=$?; [ $rc -eq 0 ] || echo RIG_FAIL rc=$rc"
        return (
            f"mkdir -p {self.workdir} && cd {self.workdir} && "
            # The forward-compat libcuda binds only at process start and a
            # detached job inherits no login shell — RIG-ENV §3c.
            "{ [ -f /etc/profile.d/zz-cuda-compat.sh ] && . /etc/profile.d/zz-cuda-compat.sh; }; "
            f"{exports}"
            f"setsid nohup bash -lc {json.dumps(wrapped)} > {self.log} 2>&1 & "
            "echo RIG_LAUNCHED $!"
        )

    def probe_script(self, tail_lines: int = 3) -> str:
        """One round trip that returns everything a gate observation needs.

        Deliberately ONE call: three separate ssh round-trips per tick is three
        chances to mistake a flaky connection for a stalled compile.
        """
        paths = " ".join(f"'{p}'" for p in (self.progress_paths or (self.workdir,)))
        return (
            f"echo '--RIG-SIZE--'; du -sb {paths} 2>/dev/null | awk '{{s+=$1}} END {{print s+0}}'; "
            f"echo '--RIG-BYTES--'; wc -c < '{self.log}' 2>/dev/null || echo 0; "
            f"echo '--RIG-MARK--'; grep -c -F '{self.done_marker}' '{self.log}' 2>/dev/null || echo 0; "
            f"echo '--RIG-FAIL--'; "
            + "; ".join(f"grep -c -F {json.dumps(m)} '{self.log}' 2>/dev/null || echo 0" for m in self.fail_markers)
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

    `--no-deps` is deliberately NOT used: a family DECLARATION builds its
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


def mint_family(
    target: str,
    *,
    runners: tuple[str, ...] = (),
    install: tuple[str, ...] = (),
    uploads: tuple[Upload, ...] = (),
    fleet_line: Path | None = None,
    name: str = "mint",
) -> Workload:
    """pgw#1331's owed leg as a workload value.

    `gen-worker family mint` needs a GPU and a real toolchain and needs NO
    weights and NO network for the model — cell identity is checkpoint-free
    (§4.27) and the constants arrive at arm time from the store. That is what
    makes a per-family mint pod cheap enough to be routine, and it is why this
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
        f"gen-worker family mint {target} --out-dir {out} {only} "
        f"--json {POD_ROOT}/minted.json && echo RIG_DONE"
    )
    return Workload(
        name=name,
        command=command,
        setup=install,
        uploads=uploads,
        artifacts=(f"{POD_ROOT}/minted.json", f"{POD_ROOT}/{name}.log", f"{POD_ROOT}/rigboot.json"),
        # The AOTI object cache and the packed cells both grow throughout the
        # compile; the log can be silent for minutes while they do.
        progress_paths=(out, f"{POD_ROOT}/work", "/root/.cache/torchinductor_root", "/root/.triton"),
        env=env,
    )
