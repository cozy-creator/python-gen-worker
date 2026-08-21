"""Pytest session guard — fail fast if a STALE gen_worker shadows the source."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "GEN_WORKER_BOOT_RECORD",
    str(Path(tempfile.mkdtemp(prefix="pgw-postmortem-")) / "boot-record.json"),
)

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import gen_worker  # noqa: E402

from gen_worker.settings_authority import (  # noqa: E402
    _interpreter_env_diffs, ensure_interpreter_env, impose_process_env)

impose_process_env()


def pytest_configure(config):
    if not _interpreter_env_diffs():
        return
    capman = config.pluginmanager.getplugin("capturemanager")
    if capman is not None:
        capman.stop_global_capturing()
    ensure_interpreter_env()

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
    gw_config.reset_for_test()
    gw_worker_goals.reset_for_test()
    gw_config.install(gw_config.load_settings())
    gw_worker_goals.install(gw_worker_goals.SERVE_ONLY)
    yield
    gw_config.reset_for_test()
    gw_worker_goals.reset_for_test()


@pytest.fixture(autouse=True)
def _fresh_report_sinks():
    from gen_worker import activity as _activity
    from gen_worker import boot_phases as _boot

    _activity.reset_for_tests()
    _boot.reset_for_tests()
    yield
    _activity.reset_for_tests()
    _boot.reset_for_tests()


@pytest.fixture(autouse=True)
def _fresh_boot_seal():
    from gen_worker import env_seal as _es

    _es._BOOT_READBACK = None
    yield
    _es._BOOT_READBACK = None


@pytest.fixture(autouse=True)
def _boot_isa_clamp():
    from gen_worker import host_isa as _isa

    try:
        _isa.impose()
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _fresh_pinned_pool_groups():
    yield
    from gen_worker.models import staging as _staging

    _staging.pinned_pool().set_group_count(1)


@pytest.fixture(autouse=True)
def _fresh_receipt_gate():
    from gen_worker import receipts as _receipts

    def _rig_posture() -> None:
        _receipts.reset()
        _receipts.trust_local_store("pytest rig: the store is the test's tmpdir")

    _rig_posture()
    yield
    _rig_posture()


@pytest.fixture(scope="session", autouse=True)
def _inductor_cache_per_worker(tmp_path_factory):
    root = tmp_path_factory.mktemp("inductor-cache")
    before = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root)
    yield
    if before is None:
        os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
    else:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = before


@pytest.fixture(scope="session")
def _postmortem_root(tmp_path_factory):
    return tmp_path_factory.mktemp("postmortem")


@pytest.fixture(autouse=True)
def _postmortem_paths_off_the_host(_postmortem_root):
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
def _isolated_local_compiled_graph_store(tmp_path_factory):
    from gen_worker import config as _config

    prior_cache = os.environ.get("TENSORHUB_CACHE_DIR")
    os.environ["TENSORHUB_CACHE_DIR"] = str(
        tmp_path_factory.mktemp("worker-cache"))
    _config.reload_for_test()
    try:
        yield
    finally:
        for name, value in (("TENSORHUB_CACHE_DIR", prior_cache),):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
