from __future__ import annotations

import json
import os
from pathlib import Path

import msgspec

from gen_worker import RequestContext, entrypoint


class ProbeIn(msgspec.Struct):
    text: str = ""


class ProbeOut(msgspec.Struct):
    response: str


def _read_environ(pid: int) -> dict:
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


@entrypoint
def report_identity(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
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


@entrypoint
def escalation_surface(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """A stock base image ships setuid binaries (su, mount, passwd...), so "no setuid escalation path" is a property we IMPOSE, not one we inherit: NoNewPrivs is what makes them harmless to a dropped child."""
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


@entrypoint
def steal_pid1_environ(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    return ProbeOut(response=json.dumps(_read_environ(1)))


@entrypoint
def steal_parent_environ(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """THE ATTACK that defeated the delta-1 strip: the parent still holds WORKER_JWT in its own environment, one /proc read away at a shared uid."""
    return ProbeOut(response=json.dumps(_read_environ(os.getppid())))


@entrypoint
def own_environ_keys(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """What the strip actually left behind in THIS process."""
    return ProbeOut(response=json.dumps(sorted(os.environ)))


@entrypoint
def write_probe(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """The positive control for the grant list: the child must still be able to write every path it was given."""
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


@entrypoint
def config_snapshot_probe(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
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


@entrypoint
def home_probe(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    """`~` and getpass.getuser() must resolve for an account-less uid — HF cache, ~/.triton and inductor's default cache dir all depend on it."""
    import getpass

    return ProbeOut(response=json.dumps({
        "home": os.path.expanduser("~"),
        "user": getpass.getuser(),
        "tmpdir": os.environ.get("TMPDIR", ""),
    }))


@entrypoint
def read_root_home(ctx: RequestContext, data: ProbeIn) -> ProbeOut:
    try:
        return ProbeOut(response="read:" + ",".join(sorted(os.listdir("/root"))))
    except OSError as exc:
        return ProbeOut(response=f"{type(exc).__name__}")
