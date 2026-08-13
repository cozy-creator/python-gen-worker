"""Pytest session guard — fail fast if a STALE gen_worker shadows the source.

A user-global / stale wheel install of ``gen-worker`` (e.g. in ``~/.local``)
silently shadows the working tree, so the suite would pass while testing old
code. We import ``gen_worker`` and assert it resolves under this repo's ``src/``.

The supported way to run the suite is ``uv run --extra dev pytest`` (pytest is
declared in the ``dev`` optional-dependency group, not the default deps). See
issue #345.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# point the postmortem carriers off the HOST before anything imports
# `gen_worker.postmortem`, which resolves BOOT_RECORD_PATH (and its in-flight /
# crash-registry / fault-dump siblings) ONCE at import from this variable.
#
# Doing it here rather than only in a fixture covers the case a fixture cannot:
# tests that spawn a REAL child (`gen_worker.entrypoint`, the SIGSEGV boot in
# test_native_crash_streak_pgw676) inherit os.environ, and a child that records
# a native-crash streak into the shared /tmp registry refuses `generate` at
# every later boot in EVERY lane's suite. Harnesses that build a replacement
# env still set it explicitly (see tests/harness/subprocess_runner.py).
os.environ.setdefault(
    "GEN_WORKER_BOOT_RECORD",
    str(Path(tempfile.mkdtemp(prefix="pgw-postmortem-")) / "boot-record.json"),
)

# put THIS directory on `sys.path` before any test module is
# imported, so `from harness import ...` cannot depend on import ORDER.
#
# pytest's `prepend` import mode inserts a test file's rootdir into `sys.path`
# as it imports that file, so the first module in `tests/` to be imported is
# what makes `harness` importable for everyone after it. That is an ordering
# dependency nobody declared, and ~15 modules rely on it. It is fine right up
# until something changes the order or narrows what a process imports — and
# then it is a COLLECTION error, which fails a whole run rather than a test.
#
# conftest.py is the one module pytest guarantees to import first, in every
# mode including each xdist worker, so this is the single place the guarantee
# can be made rather than re-made per file (`test_compile_duration_th1322.py`
# already carries a private copy of this line; it is now redundant, not wrong).
#
# REPRODUCED at the mechanism level: exec'ing `test_mint_memory_fit_pgw848.py`
# with `src/` on the path but not `tests/` raises
# `ModuleNotFoundError: No module named 'harness'` — a collection error, from a
# file whose tests pass 5/5 standalone.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import gen_worker  # noqa: E402

# the suite runs under the DECLARED interpreter env — the exact
# imposition the entrypoint performs at boot (PYTHONHASHSEED=0; env_seal.
# establish refuses without it). CPython read the seed at interpreter start,
# so a pytest launched without it gets ONE re-exec, in pytest_configure
# below — import time is too late to own the terminal: global fd capture is
# already live, and an exec under it inherits the capture tmpfiles as
# stdout/stderr, so the whole re-exec'd run reports silently (observed:
# green run, zero bytes of output). Capture is stopped first, then exec.
from gen_worker.settings_authority import (  # noqa: E402
    _interpreter_env_diffs, ensure_interpreter_env, impose_process_env)

impose_process_env()  # children spawned by tests inherit the declared env


def pytest_configure(config):
    if not _interpreter_env_diffs():
        return
    capman = config.pluginmanager.getplugin("capturemanager")
    if capman is not None:
        capman.stop_global_capturing()
    ensure_interpreter_env()  # execs; never returns (or raises for -E)

# Deterministic CPU encode wherever the suite runs: never probe or
# engage NVENC from tests — CI has no GPU, and dev boxes that have one must
# not do GPU work from the unit suite. Selection tests override + refresh
# the cache explicitly.
os.environ.setdefault("GEN_WORKER_VIDEO_ENCODER", "x264")

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
_LOCATION = Path(gen_worker.__file__).resolve()

if _REPO_SRC not in _LOCATION.parents:
    raise RuntimeError(
        f"gen_worker is imported from {_LOCATION}, NOT this repo's src/ "
        f"({_REPO_SRC}). A stale global install is shadowing the working tree — "
        "tests would run against the wrong code. Fix: run via "
        "`uv run --extra dev pytest`, and remove any global install with "
        "`python3 -m pip uninstall --break-system-packages gen-worker`."
    )


import pytest  # noqa: E402

from gen_worker import config as gw_config  # noqa: E402
from gen_worker import worker_goals as gw_worker_goals  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_process_settings():
    """pgw#931: `get_settings()` is gone, and so is the cache this fixture used
    to clear. `Settings` are now PUBLISHED by a process entry
    (`config.install`), so a test starts from "nothing installed" and installs
    what it needs — the same shape production uses, instead of poking a cache.

    A test that reaches `config.current()` without installing gets a loud
    `SettingsNotInstalled` rather than a silent fresh read of the environment,
    which is the whole point of the change.

    The default install mirrors a bare pod: `load_settings()` over whatever env
    the test has monkeypatched, so tests that tune env still work.
    """
    gw_config.reset_for_test()
    gw_worker_goals.reset_for_test()
    gw_config.install(gw_config.load_settings())
    gw_worker_goals.install(gw_worker_goals.SERVE_ONLY)
    yield
    gw_config.reset_for_test()
    gw_worker_goals.reset_for_test()


@pytest.fixture(autouse=True)
def _fresh_learned_aot_keys():
    """`aot_serve.note_aot_key` learns into a process-global set. A key one
    test teaches must never reclassify another test's dynamo refs as AOT —
    the pgw#722 discovery suite's `ck1-999…` collided with the adopt suite's
    stubbed mint digest and silently flipped its whole proof lane."""
    from gen_worker import aot_serve

    with aot_serve._KNOWN_AOT_KEYS_LOCK:
        before = set(aot_serve._KNOWN_AOT_KEYS)
    yield
    with aot_serve._KNOWN_AOT_KEYS_LOCK:
        aot_serve._KNOWN_AOT_KEYS.clear()
        aot_serve._KNOWN_AOT_KEYS.update(before)


@pytest.fixture(autouse=True)
def _fresh_delivered_seed_flag():
    """The gw#608 delivered-cell seed latch is process-lifetime in
    production; tests seeding artifacts must not leak it into later
    self-mint tests."""
    from gen_worker import compile_cache as _cc

    _cc._DELIVERED_SEEDED = False
    yield
    _cc._DELIVERED_SEEDED = False


@pytest.fixture(autouse=True)
def _fresh_cell_ledgers():
    """pgw#672 process ledgers (quarantined identities, in-process finalized
    mints) are process-lifetime in production; clear them between tests so a
    proof failure in one test cannot poison another's arm/selection."""
    from gen_worker import compile_cache as _cc
    from gen_worker import fleet_cells as _fc

    def _clear() -> None:
        with _cc._PROVEN_CELLS_LOCK:
            getattr(_cc, "_QUARANTINED_CELLS", set()).clear()
        with _fc._PENDING_LOCK:
            getattr(_fc, "_FINALIZED", {}).clear()

    _clear()
    yield
    _clear()


@pytest.fixture(autouse=True)
def _fresh_report_sinks():
    """pgw#1024: `activity._sink` and `boot_phases._sink` are process globals
    that production binds ONCE (`Executor.ensure_setup`, `lifecycle`) and never
    unbinds — correct there, a leak here.

    A hub-double row binds the sink to a `Worker._send` whose queue stops being
    drained at teardown. Nothing resets it, so under `--dist loadfile` the NEXT
    FILE runs in the same worker process holding a sink nobody empties: every
    activity it reports is handed to a dead SendQueue, and `boot_phases`
    additionally REPLAYS the previous file's buffered rows into it on the next
    bind. That is cross-file state whose effect depends on which files a worker
    happened to get — i.e. it reshuffles greens whenever suite composition
    changes. One authority here, so no file has to remember; the private copies
    that predate it (`test_reconnect_episode_pgw803`, the `reset_for_tests()`
    fixtures in the boot-phase files) are redundant, not wrong.
    """
    from gen_worker import activity as _activity
    from gen_worker import boot_phases as _boot

    _activity.reset_for_tests()
    _boot.reset_for_tests()
    yield
    _activity.reset_for_tests()
    _boot.reset_for_tests()


@pytest.fixture(autouse=True)
def _fresh_boot_seal():
    """pgw#719 boot seal is process-lifetime in production (one boot, one
    environment); tests legitimately vary global torch state per test, so
    each test gets a fresh lazy boot — a mint in one test must never drift
    against a boot seal adopted under another test's transient flags."""
    from gen_worker import env_seal as _es

    _es._BOOT_READBACK = None
    yield
    _es._BOOT_READBACK = None


@pytest.fixture(autouse=True)
def _boot_isa_clamp():
    """pgw#754's codegen clamp, imposed the way every real boot imposes it.

    Production host-compiles ONLY ever happen in a process that ran
    ``env_seal.establish`` — ``entrypoint`` (line 262), ``mint_child`` (399)
    and ``aot_compile_child`` (74) all do, and ``establish`` calls
    ``host_isa.impose``. The suite host-compiles without any of those, so
    every real AOTI compile a test drove was built ``-march=native``: an
    unclamped, unportable object, silently unlike anything a pod produces.

    pgw#811's ``assert_command_is_clamped`` made that visible by refusing it
    at the argv level. The honest answer is to give the suite the boot
    precondition production has, not to soften the assert — pgw#754 is a
    SIGILL-class defect. Tests that exercise the clamp itself monkeypatch
    ``inductor_config.cpp`` directly and are unaffected (monkeypatch restores).
    """
    from gen_worker import host_isa as _isa

    try:
        _isa.impose()
    except Exception:  # torchless/non-x86 runner: nothing to clamp
        pass
    yield


@pytest.fixture(autouse=True)
def _fresh_pinned_pool_groups():
    """pgw#780 item 1 made `bind_topology` wire the PROCESS-GLOBAL pinned pool
    with the delivered group count. Correct in production (one process, one
    bind); poison in a suite: a G=4 executor in one test leaves every later
    test's pinned budget quartered. Reset to the solo share after each test."""
    yield
    from gen_worker.models import staging as _staging

    _staging.pinned_pool().set_group_count(1)


@pytest.fixture(autouse=True)
def _fresh_receipt_gate():
    """pgw#709's receipt gate is armed once at HelloAck and stays configured
    for the process — correct in production, poison in a suite: any test that
    walks a HelloAck path leaves the gate armed, and from then on EVERY later
    test's delivered artifact is refused (no receipt on a unit fixture) and
    silently dropped to ``artifact=None``.

    That is the whole mechanism behind the long-standing order-dependent
    flux cluster: `provision.enable_compiled` drops the refused artifact and
    falls through to the inductor lane, so a dispatch assertion sees
    ``{'cc': None}`` instead of the delivered artifact, and the executor-adopt
    hit-counter tests lose their delivered cell. Those tests passed in
    isolation and failed only after an earlier file armed the gate.
    """
    from gen_worker import receipts as _receipts

    _receipts.reset()
    yield
    _receipts.reset()


@pytest.fixture(scope="session")
def _postmortem_root(tmp_path_factory):
    """One private carrier directory per test process (per xdist worker)."""
    return tmp_path_factory.mktemp("postmortem")


@pytest.fixture(autouse=True)
def _postmortem_paths_off_the_host(_postmortem_root):
    """pgw#801: the postmortem carriers are HOST paths, and the suite wrote
    to the real ones.

    ``postmortem.BOOT_RECORD_PATH`` resolves at import to ``$TENSORHUB_CACHE_DIR``
    or ``/tmp``, and the in-flight marker, crash registry and fault dump are its
    siblings. On a pod that is correct — the point is to survive process death.
    On a dev box or CI runner ``/tmp`` is shared by every process, every lane and
    every run, so a suite that reaches it is asserting about the RUNNER.

    Measured 2026-07-30: ``/tmp/gen-worker-crash-streaks.json`` held
    ``{"generate": {"count": 2, ...}}`` written by another lane's run minutes
    earlier. ``NATIVE_CRASH_REFUSE_STREAK`` is 2, so from then on EVERY boot in
    EVERY lane's suite refused ``generate`` — 6 tests across
    ``test_boot_compile_deferral_gw584.py`` and ``test_resolution_rekey_gw494.py``
    failed for everyone, in isolation as well as in the full run, until someone
    deleted the file by hand. The registry is genuinely durable-by-design, which
    is exactly why the suite must never share the production carrier.

    The carrier directory is per-process, but the FILES are removed around
    every test: a streak one test records is that test's fact, and the crash
    gate is boot-scoped in production.
    """
    from gen_worker import postmortem as _pm

    names = ("BOOT_RECORD_PATH", "INFLIGHT_PATH",
             "CRASH_REGISTRY_PATH", "FAULT_DUMP_PATH")
    saved = {name: getattr(_pm, name) for name in names}
    redirected = [_postmortem_root / path.name for path in saved.values()]

    def _wipe() -> None:
        for path in redirected:
            path.unlink(missing_ok=True)

    for name, path in zip(names, redirected):
        setattr(_pm, name, path)
    _wipe()
    yield
    _wipe()
    for name, path in saved.items():
        setattr(_pm, name, path)


@pytest.fixture(autouse=True)
def _isolated_local_cell_store(tmp_path_factory):
    """pgw#1096: the local cell store defaults to ``~/.cache/cozy/compile-cells``.

    A suite that exercises the AOT self-mint now legitimately WRITES there —
    `local_keep_reason` keeps a cell whenever no publisher was wired, which is
    every unit fixture in the tree. Redirect the root per test, so the suite
    can never deposit fake cells in the developer's real store (or read one
    left by a previous run and call it a hit). Also isolates
    ``aot_resume.bank_root``, which is sited under the same root.

    **Deliberately does NOT take `monkeypatch`**, and this is not a style
    preference — it cost a CI red. An AUTOUSE fixture is set up before the
    explicitly-requested ones, so requesting `monkeypatch` here makes pytest
    build it early for EVERY test in the tree; fixtures finalize in reverse
    setup order, so `monkeypatch` then unwinds LAST — after the yield-fixture
    teardowns that run beside it. `test_seal_lib_memo_pgw832` and
    `test_seal_record_derivation_pgw1095` both `monkeypatch.setattr(env_seal,
    "_lib_digest", ...)` over an `lru_cache` and then call
    `env_seal._lib_digest.cache_clear()` in their own `finally` — which, with
    the order flipped, ran while the attribute was still a plain function
    (`AttributeError: 'function' object has no attribute 'cache_clear'`).
    Save and restore the variable by hand so this fixture adds no ordering
    edge to anything.
    """
    prior = os.environ.get("GEN_WORKER_LOCAL_CELLS_DIR")
    os.environ["GEN_WORKER_LOCAL_CELLS_DIR"] = str(
        tmp_path_factory.mktemp("local-cell-store"))
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("GEN_WORKER_LOCAL_CELLS_DIR", None)
        else:
            os.environ["GEN_WORKER_LOCAL_CELLS_DIR"] = prior
