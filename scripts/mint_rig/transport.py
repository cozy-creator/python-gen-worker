"""Getting bytes and commands onto the pod, behind a Protocol.

Same reason as :mod:`mint_rig.runpod`: the driver's logic is tested against a
fake that records what was shipped and answers scripted output, so a real pod is
only ever proving the wire.

:class:`SshTransport` carries the three things this workspace has already paid
to learn:

  * **Do not drop `SSH_AUTH_SOCK`.** The runpod key on the box is
    passphrase-protected, so `-i` alone cannot authenticate and the agent holds
    the only usable copy. A driver that sanitises the environment gets a
    failure that looks exactly like a slow boot (pod_run.py's opening note).
  * **Print the reason on every retry.** A retry loop that swallows ssh's
    stderr cannot tell "still booting" from "wrong key", and the difference is
    a rented pod.
  * **A freshly booted sshd drops connections for a while**, so a transfer
    retries rather than failing the run — but the retry is bounded by the same
    progress rule as everything else, not by a wall clock.

The per-call `timeout_s` bounds the SUBPROCESS, not the work: every long pod
task is launched detached (`setsid nohup … &`) and watched by a
:class:`~mint_rig.progress.Gate`, so no command this module runs is ever
expected to take minutes.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

#: SSH is a control-plane round trip here (start a detached job, read a tail,
#: stat a directory). A call that has produced nothing in this long is broken.
SSH_CALL_TIMEOUT_S = 300.0
SCP_CALL_TIMEOUT_S = 1800.0

SSH_KEY = Path.home() / ".ssh" / "runpod"
SSH_OPTS = (
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "LogLevel=ERROR",
)


@dataclass(frozen=True)
class Result:
    rc: int
    out: str

    @property
    def ok(self) -> bool:
        return self.rc == 0


class Transport(Protocol):
    """Two verbs. Everything the rig does to a pod is one of them."""

    def run(
        self, script: str, *, timeout_s: float = SSH_CALL_TIMEOUT_S, env: Mapping[str, str] | None = None
    ) -> Result: ...

    def put(self, local: Sequence[Path], remote_dir: str, *, timeout_s: float = SCP_CALL_TIMEOUT_S) -> Result: ...

    def fetch(self, remote: str, local_dir: Path, *, timeout_s: float = SCP_CALL_TIMEOUT_S) -> Result: ...


@dataclass
class SshTransport:
    """The real one."""

    host: str
    port: int
    user: str = "root"
    key: Path = SSH_KEY
    #: Injected so tests can drive the argv builder without a network.
    runner: "_Runner" = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.runner is None:
            self.runner = _subprocess_runner

    @property
    def _opts(self) -> list[str]:
        return ["-i", str(self.key), *SSH_OPTS]

    def run(
        self, script: str, *, timeout_s: float = SSH_CALL_TIMEOUT_S, env: Mapping[str, str] | None = None
    ) -> Result:
        # json.dumps, not shlex.quote: the value crosses a shell on THIS box and
        # a shell on the pod, and a double-quoted JSON string survives both.
        prelude = "".join(f"export {k}={json.dumps(v)}; " for k, v in (env or {}).items())
        return self.runner(
            ["ssh", *self._opts, "-p", str(self.port), f"{self.user}@{self.host}", prelude + script],
            timeout_s,
        )

    def put(self, local: Sequence[Path], remote_dir: str, *, timeout_s: float = SCP_CALL_TIMEOUT_S) -> Result:
        if not local:
            return Result(0, "")
        return self.runner(
            [
                "scp",
                *self._opts,
                "-P",
                str(self.port),
                *[str(p) for p in local],
                f"{self.user}@{self.host}:{remote_dir}",
            ],
            timeout_s,
        )

    def fetch(self, remote: str, local_dir: Path, *, timeout_s: float = SCP_CALL_TIMEOUT_S) -> Result:
        local_dir.mkdir(parents=True, exist_ok=True)
        return self.runner(
            ["scp", *self._opts, "-r", "-P", str(self.port), f"{self.user}@{self.host}:{remote}", str(local_dir)],
            timeout_s,
        )


class _Runner(Protocol):
    def __call__(self, argv: Sequence[str], timeout_s: float) -> Result: ...


def _subprocess_runner(argv: Sequence[str], timeout_s: float) -> Result:
    # NOTE: the environment is INHERITED, deliberately — SSH_AUTH_SOCK is the
    # only usable copy of the passphrase-protected runpod key on this box.
    try:
        proc = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        parts = [exc.stdout or b"", exc.stderr or b""]
        text = "".join(p.decode(errors="replace") if isinstance(p, bytes) else p for p in parts)
        return Result(124, f"{text}\n[transport] subprocess produced nothing for {timeout_s}s")
    return Result(proc.returncode, (proc.stdout or "") + (proc.stderr or ""))
