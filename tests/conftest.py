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
from pathlib import Path

import gen_worker

# Deterministic CPU encode wherever the suite runs (gw#476): never probe or
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

from gen_worker.config.loader import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_settings_cache():
    """`get_settings()` caches process-wide; tests monkeypatch env vars, so
    every test starts (and ends) with a cleared cache."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_learned_aot_keys():
    """`aot_serve.note_aot_key` learns into a process-global set. A key one
    test teaches must never reclassify another test's dynamo refs as AOT —
    the pgw#722 discovery suite's `ck5-999…` collided with the adopt suite's
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
def _fresh_boot_seal():
    """pgw#719 boot seal is process-lifetime in production (one boot, one
    environment); tests legitimately vary global torch state per test, so
    each test gets a fresh lazy boot — a mint in one test must never drift
    against a boot seal adopted under another test's transient flags."""
    from gen_worker import env_seal as _es

    _es._BOOT_SEAL = None
    yield
    _es._BOOT_SEAL = None


@pytest.fixture(autouse=True)
def _fresh_receipt_gate():
    """pgw#709's receipt gate is armed once at HelloAck and stays configured
    for the process — correct in production, poison in a suite: any test that
    walks a HelloAck path leaves the gate armed, and from then on EVERY later
    test's delivered artifact is refused (no receipt on a unit fixture) and
    silently dropped to ``artifact=None``.

    That is the whole mechanism behind the long-standing order-dependent
    flux/trt cluster: `provision.enable_compiled` drops the refused artifact
    and falls through to the inductor lane, so `test_trt_engine`'s dispatch
    assertion sees ``{'cc': None}`` instead of the TRT engine, and the
    executor-adopt hit-counter tests lose their delivered cell. Those tests
    passed in isolation and failed only after an earlier file armed the gate.
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
