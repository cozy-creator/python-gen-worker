"""The compute child runs as an UNPRIVILEGED uid."""

from __future__ import annotations

import ctypes
import errno
import grp
import logging
import os
import pwd
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ENV_COMPUTE_UID = "GEN_WORKER_COMPUTE_UID"

DEFAULT_COMPUTE_USER = "cozy-compute"
DEFAULT_COMPUTE_UID = 10001
DEFAULT_COMPUTE_GID = 10001

_DEVICE_GROUPS: Tuple[str, ...] = ("video", "render")

_DEVICE_GLOBS: Tuple[str, ...] = (
    "/dev/nvidia[0-9]*",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm*",
    "/dev/nvidia-caps/*",
    "/dev/dri/*",
)

_PR_SET_PDEATHSIG = 1
_PR_SET_NO_NEW_PRIVS = 38


@dataclass(frozen=True)
class DropPlan:
    """The credential the compute child execs with, and the identity facts the child's env must carry so ``~``/``getpass.getuser()`` still resolve."""

    uid: int
    gid: int
    groups: Tuple[int, ...]
    user: str
    home: str

    def describe(self) -> str:
        return (
            f"{self.user}({self.uid}:{self.gid})"
            f" groups={','.join(str(g) for g in self.groups) or '-'}"
            f" home={self.home}"
        )


def _resolve_target(spec: str) -> Tuple[int, int, str]:
    spec = (spec or "").strip()
    if not spec:
        try:
            ent = pwd.getpwnam(DEFAULT_COMPUTE_USER)
            return int(ent.pw_uid), int(ent.pw_gid), ent.pw_name
        except KeyError:
            return DEFAULT_COMPUTE_UID, DEFAULT_COMPUTE_GID, DEFAULT_COMPUTE_USER
    try:
        uid = int(spec)
    except ValueError:
        ent = pwd.getpwnam(spec)
        return int(ent.pw_uid), int(ent.pw_gid), ent.pw_name
    if uid == 0:
        raise ValueError(
            f"{ENV_COMPUTE_UID}=0 would run tenant code as root. This name "
            "selects WHICH unprivileged uid the compute child uses; it is not "
            "a way to turn the boundary off (pgw#858)."
        )
    try:
        ent = pwd.getpwuid(uid)
        return uid, int(ent.pw_gid), ent.pw_name
    except KeyError:
        return uid, uid, DEFAULT_COMPUTE_USER


def _supplementary(user: str, gid: int) -> Tuple[int, ...]:
    got: List[int] = []
    try:
        got = [int(g) for g in os.getgrouplist(user, gid)]
    except (KeyError, OSError):
        got = [gid]
    for name in _DEVICE_GROUPS:
        try:
            got.append(int(grp.getgrnam(name).gr_gid))
        except KeyError:
            continue
    return tuple(sorted(set(got)))


def plan_drop(home: str) -> Optional[DropPlan]:
    """The plan, or ``None`` with the reason logged when there is nothing to drop (a parent that is not root cannot setuid, and on a developer box it already has no privilege to give away)."""
    if not hasattr(os, "geteuid"):
        logger.info("privilege drop unavailable on this platform")
        return None
    if os.geteuid() != 0:
        logger.info(
            "compute child keeps uid %d: the control parent is not root, so "
            "there is no privilege to drop (pgw#858)", os.geteuid(),
        )
        return None
    uid, gid, user = _resolve_target(os.environ.get(ENV_COMPUTE_UID, ""))
    _ensure_account(uid, gid, user, home)
    return DropPlan(
        uid=uid, gid=gid, groups=_supplementary(user, gid), user=user, home=home,
    )


def _ensure_account(uid: int, gid: int, user: str, home: str) -> None:
    try:
        pwd.getpwuid(uid)
        return
    except KeyError:
        pass
    try:
        grp.getgrgid(gid)
    except KeyError:
        _append_line("/etc/group", f"{user}:x:{gid}:\n")
    _append_line("/etc/passwd", f"{user}:x:{uid}:{gid}:cozy compute:{home}:/usr/sbin/nologin\n")


def _append_line(path: str, line: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
        logger.info("added %s entry for the compute child", path)
    except OSError as exc:
        logger.info(
            "could not add a %s entry (%s); the compute child runs on a "
            "numeric uid, which is sufficient", path, exc,
        )


def child_env(plan: DropPlan) -> Dict[str, str]:
    """The env delta that makes an account-less uid behave like a user."""
    home = plan.home
    return {
        "HOME": home,
        "USER": plan.user,
        "LOGNAME": plan.user,
        "XDG_CACHE_HOME": os.path.join(home, ".cache"),
        "TMPDIR": os.path.join(home, "tmp"),
        "PYTHONPYCACHEPREFIX": os.path.join(home, "pycache"),
    }


def writable_paths(plan: DropPlan, extra: Iterable[str] = ()) -> List[str]:
    """Every directory the compute child must be able to write."""
    env = child_env(plan)
    paths = [plan.home, env["TMPDIR"], env["XDG_CACHE_HOME"], env["PYTHONPYCACHEPREFIX"]]
    paths.extend(p for p in extra if p)
    out: List[str] = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        if p not in out:
            out.append(p)
    return out


_NEVER_GRANT = frozenset({
    "/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib64", "/media",
    "/mnt", "/opt", "/proc", "/root", "/run", "/sbin", "/srv", "/sys", "/tmp",
    "/usr", "/var", "/var/tmp",
})


def grant_paths(plan: DropPlan, paths: Sequence[str]) -> List[str]:
    """mkdir + recursive chown, so the child owns what it must write."""
    granted: List[str] = []
    for path in paths:
        started = time.monotonic()
        if path in _NEVER_GRANT:
            logger.info(
                "not granting %s to the compute child: a recursive chown of a "
                "shared system directory is never the right answer", path,
            )
            continue
        try:
            if os.path.isdir(path) and os.stat(path).st_mode & 0o002:
                continue
        except OSError:
            pass
        try:
            os.makedirs(path, exist_ok=True)
            _ensure_traversable(path)
            _chown_tree(path, plan.uid, plan.gid)
        except OSError as exc:
            logger.warning("could not grant %s to the compute child: %s", path, exc)
            continue
        granted.append(path)
        took = time.monotonic() - started
        if took > 1.0:
            logger.info("granted %s to the compute child in %.1fs", path, took)
    return granted


def _ensure_traversable(path: str) -> None:
    parts = Path(path).resolve().parents
    for ancestor in reversed(list(parts)):
        try:
            mode = ancestor.stat().st_mode
        except OSError:
            continue
        if mode & 0o001:
            continue
        try:
            ancestor.chmod(mode | 0o001)
        except OSError:
            continue


def _chown_tree(root: str, uid: int, gid: int) -> None:
    os.chown(root, uid, gid)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            try:
                os.chown(os.path.join(dirpath, name), uid, gid, follow_symlinks=False)
            except OSError:
                continue


def grant_socket(plan: DropPlan, socket_path: str) -> bool:
    """The child's ONE channel to the parent."""
    try:
        _ensure_traversable(socket_path)
        os.chown(socket_path, plan.uid, plan.gid)
        os.chmod(socket_path, 0o600)
        return True
    except OSError as exc:
        logger.error(
            "could not hand the control socket %s to the compute child: %s — "
            "the child will not be able to connect", socket_path, exc,
        )
        return False


def grant_devices(plan: DropPlan, globs: Sequence[str] = _DEVICE_GLOBS) -> List[str]:
    """Make sure the tenant uid can reach the accelerator."""
    widened: List[str] = []
    for pattern in globs:
        for node in sorted(Path("/dev").glob(pattern[len("/dev/"):])):
            try:
                st = node.stat()
            except OSError:
                continue
            if _reachable(st, plan):
                continue
            try:
                node.chmod(st.st_mode | 0o006)
                widened.append(str(node))
            except OSError as exc:
                logger.warning(
                    "compute child cannot reach %s and it could not be widened "
                    "(%s) — GPU work will fail", node, exc,
                )
    if widened:
        logger.info("widened device nodes for the compute child: %s", ", ".join(widened))
    return widened


def _reachable(st: os.stat_result, plan: DropPlan) -> bool:
    need = 0o600
    if st.st_uid == plan.uid:
        return st.st_mode & need == need
    if st.st_gid == plan.gid or st.st_gid in plan.groups:
        return st.st_mode & 0o060 == 0o060
    return st.st_mode & 0o006 == 0o006


def _prctl(option: int, arg2: int = 0) -> None:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(option, arg2, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), f"prctl({option}) failed")


def set_pdeathsig() -> None:
    """SIGKILL this process when its parent dies, so a crashed control parent never strands G children each holding tens of GB of VRAM."""
    try:
        _prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL))
    except Exception:
        pass


def preexec(plan: Optional[DropPlan]) -> Callable[[], None]:
    """Build the preexec_fn for one spawn — post-fork, PRE-exec, so tenant code has never run in this process when the credential changes. Order is load-bearing: (1) no_new_privs FIRST, while still root, so no setuid binary in the image can hand privilege back after the drop; (2) supplementary groups, then gid, then uid — groups/gid can only be set while privileged, and setuid LAST makes the change irreversible (real, effective and saved move together); (3) ASSERT the drop took — a silent no-op drop is worse than none, and a failed check raises here, aborting the spawn before exec so tenant code never runs as root; (4) PR_SET_PDEATHSIG last, re-established after the credential change."""

    def _run() -> None:
        if plan is not None:
            _prctl(_PR_SET_NO_NEW_PRIVS, 1)
            os.setgroups(list(plan.groups))
            os.setgid(plan.gid)
            os.setuid(plan.uid)
            _assert_dropped(plan)
        set_pdeathsig()

    return _run


def _assert_dropped(plan: DropPlan) -> None:
    if os.getuid() != plan.uid or os.geteuid() != plan.uid:
        raise RuntimeError(
            f"privilege drop did not take: uid={os.getuid()} euid={os.geteuid()} "
            f"expected {plan.uid} (pgw#858)"
        )
    if os.getgid() != plan.gid or os.getegid() != plan.gid:
        raise RuntimeError(
            f"privilege drop did not take: gid={os.getgid()} egid={os.getegid()} "
            f"expected {plan.gid} (pgw#858)"
        )
    try:
        os.setuid(0)
    except OSError as exc:
        if exc.errno not in (errno.EPERM, errno.EINVAL):
            raise
    else:
        raise RuntimeError(
            "privilege drop is reversible — the compute child regained uid 0 "
            "(pgw#858); refusing to exec tenant code"
        )


__all__ = [
    "DEFAULT_COMPUTE_UID",
    "DEFAULT_COMPUTE_USER",
    "ENV_COMPUTE_UID",
    "DropPlan",
    "child_env",
    "grant_devices",
    "grant_paths",
    "grant_socket",
    "plan_drop",
    "preexec",
    "set_pdeathsig",
    "writable_paths",
]
