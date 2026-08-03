"""pgw#858 / th#1380: the compute child runs as an UNPRIVILEGED uid.

The parent/child split (pgw#763) already separates the process that holds the
worker's identity from the process that imports tenant code. Until this module
both processes ran as **root, in one PID namespace**, which made the separation
polite rather than enforced: `_CHILD_FORBIDDEN_ENVS` deletes ``WORKER_JWT`` from
the child's environment, and tenant code read it straight back out of
``/proc/<ppid>/environ``. The same mechanism handed it RunPod's injected
``RUNPOD_API_KEY`` from ``/proc/1/environ`` — a key th#1380 verified is
**account-scoped in authority** (it enumerates our whole fleet, reads our
balance, and lists 90 container-registry credential records) and which RunPod
gives us no way to suppress at the source.

A uid boundary closes it without the provider's cooperation. th#1380 verified
the end state on a real pod: an unprivileged uid got ``PermissionError`` on
``/proc/1/environ`` (mode 0400, root-owned), ``PermissionError`` on root's home,
an empty environment, and **401** from the RunPod API.

**Why the drop happens here — in the parent, at spawn — and not as a ``USER``
directive in the images.** A ``USER`` line must be written correctly in 9
Dockerfiles across 4 repos plus ``generate_dockerfile.go``, and remembered by
every future endpoint author; a control an author can forget is not a control.
It would also drop the *parent*, which needs root to prepare the child's paths
and to reap it, and it cannot express "privileged until the caches are ready,
unprivileged from the exec onward". The parent covers every image uniformly,
including images we do not build.

**What this is not.** It defends against *tenant code*, not against *container
escape*: GPU workloads need ``--gpus``, which rules out most syscall sandboxes.
See th#1380 §"Recommended design" item 6 before mistaking one for the other.
"""

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

# WHICH uid the compute child runs as. This is PLUMBING (like a cache path),
# not a behaviour switch: there is no value that turns the boundary off, and 0
# is refused. Platform-reserved hub-side like every other name in this package.
ENV_COMPUTE_UID = "GEN_WORKER_COMPUTE_UID"

DEFAULT_COMPUTE_USER = "cozy-compute"
DEFAULT_COMPUTE_UID = 10001
DEFAULT_COMPUTE_GID = 10001

# Groups that gate device access in some base images. Resolved by NAME and
# skipped when absent — never invented, never a numeric guess.
_DEVICE_GROUPS: Tuple[str, ...] = ("video", "render")

# Device nodes the compute child must reach. Under the NVIDIA container runtime
# these are already 0666 root:root; we verify rather than assume, and widen
# only the specific nodes that are not already reachable.
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
    """The credential the compute child execs with, and the identity facts the
    child's env must carry so ``~``/``getpass.getuser()`` still resolve."""

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


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def _resolve_target(spec: str) -> Tuple[int, int, str]:
    """(uid, gid, name) from a name, a number, or the default."""
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
        ent = pwd.getpwnam(spec)   # a name we cannot resolve is a hard error
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
    """The child's supplementary groups: the target's own list when the image
    declares the user, otherwise only the device groups that actually exist.

    Never the parent's groups — inheriting root's supplementary set is how a
    "drop" ends up still holding docker/sudo group membership."""
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
    """The plan, or ``None`` with the reason logged when there is nothing to
    drop (a parent that is not root cannot setuid, and on a developer box it
    already has no privilege to give away)."""
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
    """Best-effort ``/etc/passwd`` + ``/etc/group`` entry for a numeric target.

    Not required — the drop works on a uid with no account at all, and the
    child's env carries HOME/USER/LOGNAME so the common lookups do not need
    one. It exists so ``pwd.getpwuid()`` behaves for a library that calls it
    directly, and so no image has to remember to create the user."""
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


# ---------------------------------------------------------------------------
# granting what the child needs
# ---------------------------------------------------------------------------


def child_env(plan: DropPlan) -> Dict[str, str]:
    """The env delta that makes an account-less uid behave like a user.

    Every one of these has a concrete consumer: ``~`` (HF cache, ``~/.triton``,
    ``~/.nv``) resolves from HOME; torch inductor's default cache dir is named
    after ``getpass.getuser()``, which reads USER/LOGNAME before ``pwd``;
    TMPDIR moves temp/output staging off world-writable ``/tmp`` onto a dir the
    tenant uid owns; PYTHONPYCACHEPREFIX keeps ``.pyc`` writes out of a
    root-owned ``/app``."""
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
    """Every directory the compute child must be able to write.

    Deliberately explicit and small — the answer to a permission error is
    another entry here, never giving the child root back."""
    env = child_env(plan)
    paths = [plan.home, env["TMPDIR"], env["XDG_CACHE_HOME"], env["PYTHONPYCACHEPREFIX"]]
    paths.extend(p for p in extra if p)
    out: List[str] = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        if p not in out:
            out.append(p)
    return out


# Directories that must NEVER be handed to the compute uid, however they were
# derived. A grant is recursive, so a caller that resolves a default to "/tmp"
# (the post-mortem marker dir does exactly that when TENSORHUB_CACHE_DIR is
# unset) would otherwise chown a shared system tree to tenant code.
_NEVER_GRANT = frozenset({
    "/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib64", "/media",
    "/mnt", "/opt", "/proc", "/root", "/run", "/sbin", "/srv", "/sys", "/tmp",
    "/usr", "/var", "/var/tmp",
})


def grant_paths(plan: DropPlan, paths: Sequence[str]) -> List[str]:
    """mkdir + recursive chown, so the child owns what it must write.

    Recursive is cheap where it matters: a cold pod's CAS is empty, and a warm
    one is inode metadata only. Failures are logged per path and never fatal —
    the child's own error will name the path far more precisely than a guess
    here could."""
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
                continue   # already world-writable (a sticky /tmp-alike)
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
    """Owning a directory is useless if an ancestor cannot be entered.

    Adds ``o+x`` — traverse, NOT list and NOT write — to every ancestor that
    lacks it. On a production image the whole chain is already 0755 and this
    touches nothing; it matters where the parent created the chain itself with
    a root-only mode."""
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
    """The child's ONE channel to the parent.

    Connecting to a unix socket needs WRITE on the inode, so a root-created
    socket under the default umask (0755) is unreachable by the tenant uid.
    Owner-only 0600 on the tenant uid is also strictly tighter than what the
    split shipped with — before this, any process on the host side of the
    container that could reach the path had the same access root did."""
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
    """Make sure the tenant uid can reach the accelerator.

    Under the NVIDIA container runtime ``/dev/nvidia*`` is already 0666
    root:root, so this normally touches nothing. We verify instead of assuming,
    and widen ONLY the specific nodes that are not already reachable — never a
    blanket chmod over ``/dev``."""
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
    """rw for the target uid under the usual owner/group/other ladder."""
    need = 0o600
    if st.st_uid == plan.uid:
        return st.st_mode & need == need
    if st.st_gid == plan.gid or st.st_gid in plan.groups:
        return st.st_mode & 0o060 == 0o060
    return st.st_mode & 0o006 == 0o006


# ---------------------------------------------------------------------------
# the drop itself
# ---------------------------------------------------------------------------


def _prctl(option: int, arg2: int = 0) -> None:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(option, arg2, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), f"prctl({option}) failed")


def set_pdeathsig() -> None:
    """pgw#783: SIGKILL this process when its parent dies, so a crashed control
    parent never strands G children each holding tens of GB of VRAM.

    Called AFTER the credential change: the property must survive the drop, and
    establishing it second is how that is guaranteed rather than assumed."""
    try:
        _prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL))
    except Exception:
        # Best-effort: a platform without prctl keeps the pre-pgw#783 behaviour
        # (the container's own death took the child with it at G == 1).
        pass


def preexec(plan: Optional[DropPlan]) -> Callable[[], None]:
    """Build the ``preexec_fn`` for one spawn — post-fork, PRE-exec, so tenant
    code has never run in this process when the credential changes.

    Order is load-bearing:

    1. ``no_new_privs`` FIRST, while still root, so no setuid binary in the
       image can hand privilege back after the drop.
    2. supplementary groups, then gid, then uid — groups and gid can only be
       set while privileged, and ``setuid`` last is what makes the change
       irreversible (real, effective and saved all move together).
    3. **assert it worked.** A silent no-op drop is worse than no drop, so a
       failed check raises here — which aborts the spawn before ``exec``, so
       tenant code never runs at all rather than running as root.
    4. ``PR_SET_PDEATHSIG`` last, re-establishing pgw#783 after the change.
    """

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
    """Prove the credential actually changed and cannot be undone.

    Runs between fork and exec, where the only honest failure mode is to die:
    raising would be reported as a spawn failure, which is exactly right, and
    is what the caller's crash-loop typing already knows how to bound."""
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
