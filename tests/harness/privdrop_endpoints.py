"""pgw#858 probes: TENANT CODE going after the pod's credentials via /proc.

Every handler here is an attack the threat model says untrusted endpoint code
can run — it is imported into the compute child, so all of it is reachable. The
guards assert each one comes back denied while the drop is in effect, and the
SAME handlers are re-run with the drop removed, where they must succeed.

Separate file from procsplit_endpoints.py so this lane owns its fixture outright
(shared-worktree etiquette).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import msgspec

from gen_worker import RequestContext, endpoint


class ProbeIn(msgspec.Struct):
    text: str = ""


class ProbeOut(msgspec.Struct):
    response: str


def _read_environ(pid: int) -> dict:
    """Try to steal another process's environment through /proc.

    Reports the OUTCOME, never raises: a guard that cannot distinguish "denied"
    from "the probe crashed" is not a guard."""
    path = f"/proc/{pid}/environ"
    try:
        raw = Path(path).read_bytes()
    except PermissionError as exc:
        return {"outcome": "denied", "error": f"{type(exc).__name__}: {exc}"}
    except OSError as exc:
        return {"outcome": "error", "error": f"{type(exc).__name__}: {exc}"}
    keys = sorted(
        item.split("=", 1)[0]
        for item in raw.decode("utf-8", "replace").split("\0")
        if "=" in item
    )
    return {"outcome": "read", "bytes": len(raw), "keys": keys}


@endpoint
class PrivProbe:
    def report_identity(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """Who this process actually is, and whether it can climb back."""
        try:
            os.setuid(0)
            regained = True
        except OSError:
            regained = False
        return ProbeOut(response=json.dumps({
            "uid": os.getuid(),
            "euid": os.geteuid(),
            "gid": os.getgid(),
            "groups": sorted(os.getgroups()),
            "regained_root": regained,
            "ppid": os.getppid(),
        }))

    def escalation_surface(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """A stock base image ships setuid binaries (su, mount, passwd...), so
        "no setuid escalation path" is a property we IMPOSE, not one we inherit:
        NoNewPrivs is what makes them harmless to a dropped child."""
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        flags = dict(
            (line.split(":", 1)[0].strip(), line.split(":", 1)[1].strip())
            for line in status.splitlines() if ":" in line
        )
        setuid = []
        for candidate in ("/usr/bin/su", "/usr/bin/mount", "/usr/bin/passwd"):
            p = Path(candidate)
            if p.exists() and p.stat().st_mode & 0o4000:
                setuid.append(candidate)
        return ProbeOut(response=json.dumps({
            "no_new_privs": flags.get("NoNewPrivs", ""),
            "setuid_binaries_present": setuid,
        }))

    def steal_pid1_environ(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """THE ATTACK th#1380 measured: RunPod's account-authority key lives in
        PID 1's environment and cannot be suppressed at the create call."""
        return ProbeOut(response=json.dumps(_read_environ(1)))

    def steal_parent_environ(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """THE ATTACK that defeated the delta-1 strip: the parent still holds
        WORKER_JWT in its own environment, one /proc read away at a shared uid."""
        return ProbeOut(response=json.dumps(_read_environ(os.getppid())))

    def own_environ_keys(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """What the strip actually left behind in THIS process."""
        return ProbeOut(response=json.dumps(sorted(os.environ)))

    def write_probe(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """The positive control for the grant list: the child must still be
        able to write every path it was given. `data.text` is a path.

        pgw#1349: it CREATES A SUBDIRECTORY too, because that is the operation
        that actually died in production. Every child-side writer under a
        granted root reaches it through
        ``path.parent.mkdir(parents=True, exist_ok=True)`` — the local compiled graph
        store's memo and sidecar writes are exactly that — and a probe that
        only writes a file into a directory the PARENT made would have gone
        green on the tree where the child could not make one."""
        root = Path(data.text)
        nested = root / "pgw858-probe-dir" / "nested"
        target = root / "pgw858-write-probe"
        try:
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "leaf").write_text("ok", encoding="utf-8")
            target.write_text("ok", encoding="utf-8")
            target.unlink()
            (nested / "leaf").unlink()
            nested.rmdir()
            nested.parent.rmdir()
            return ProbeOut(response="ok")
        except OSError as exc:
            return ProbeOut(response=f"{type(exc).__name__}: {exc}")

    def config_snapshot_probe(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """th#1087's config snapshot is the one child-side writer that RAISES
        on failure, and it lives in the root-owned image tree (/app/.tensorhub).
        Mirrors `_write_snapshot_locked` exactly — mkstemp in the SAME dir plus
        os.replace — and takes its path from the module, so the two cannot
        drift apart silently."""
        import tempfile

        from gen_worker import runtime_config

        path = (os.environ.get(runtime_config.SNAPSHOT_PATH_ENV)
                or runtime_config.DEFAULT_SNAPSHOT_PATH)
        d = os.path.dirname(path) or "."
        try:
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, prefix=".pgw858-")
            os.close(fd)
            os.replace(tmp, path + ".pgw858")
            os.unlink(path + ".pgw858")
            return ProbeOut(response=f"ok:{d}")
        except OSError as exc:
            return ProbeOut(response=f"{type(exc).__name__}: {exc} ({d})")

    def home_probe(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """`~` and getpass.getuser() must resolve for an account-less uid —
        HF cache, ~/.triton and inductor's default cache dir all depend on it."""
        import getpass

        return ProbeOut(response=json.dumps({
            "home": os.path.expanduser("~"),
            "user": getpass.getuser(),
            "tmpdir": os.environ.get("TMPDIR", ""),
        }))

    def read_root_home(self, ctx: RequestContext, data: ProbeIn) -> ProbeOut:
        """Anything the control parent leaves in root's home must be out of
        reach — th#1380 checked this on the real pod alongside /proc."""
        try:
            return ProbeOut(response="read:" + ",".join(sorted(os.listdir("/root"))))
        except OSError as exc:
            return ProbeOut(response=f"{type(exc).__name__}")
