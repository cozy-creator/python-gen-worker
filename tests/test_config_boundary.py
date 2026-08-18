"""The config boundary: who may read settings, and how a reconcile resolves.

Sections keep their incident id; the full narratives live in the tracker.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import msgspec
import pytest
from harness.hub_double import hub_double, is_ready, is_result_for
from harness.toy_endpoints import EchoIn

from gen_worker import config
from gen_worker.models import provision
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.request_context import RequestContext
from gen_worker.runtime_config import (
    SNAPSHOT_PATH_ENV,
    ConfigSnapshotWriteError,
    ConfigStore,
    read_snapshot,
)
from gen_worker.subproc import run_process

# ============================================================================
# pgw#931 — §1.18 — one config pipeline in, and the struct is passed.
# ============================================================================

def test_current_raises_when_nothing_was_installed() -> None:
    config.reset_for_test()
    try:
        with pytest.raises(config.SettingsNotInstalled):
            config.current()
    finally:
        config.reload_for_test()


def test_get_settings_is_gone() -> None:
    """pgw#931: The cached process-global accessor must not come back."""
    assert not hasattr(config, "get_settings")
    from gen_worker.config import loader

    assert not hasattr(loader, "get_settings")


def test_current_or_takes_its_fallback_as_a_value() -> None:
    """pgw#931: A standalone default must be visible AT THE CALL SITE, not be a silent env read."""
    config.reset_for_test()
    try:
        sentinel = config.Settings(tensorhub_url="https://standalone.example")
        assert config.current_or(sentinel).tensorhub_url == "https://standalone.example"
    finally:
        config.reload_for_test()


def test_install_returns_the_settings_so_a_bootstrap_is_one_line() -> None:
    settings = config.Settings(worker_id="w-1")
    assert config.install(settings) is settings
    assert config.current() is settings
    config.reload_for_test()


def test_a_typo_in_dotenv_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """pgw#931: The deliverable's own example."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TENSORHUB_CHACE_DIR=/oops\n")
    with pytest.raises(config.UnknownSettingError) as excinfo:
        config.load_settings()
    assert "TENSORHUB_CHACE_DIR" in str(excinfo.value), (
        "the refusal must NAME the key, or it is a wall of text an operator "
        "cannot act on")


def test_foreign_keys_in_dotenv_are_still_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """pgw#931: Scoped to the namespaces we own, deliberately."""
    monkeypatch.chdir(tmp_path)
    # A real environment variable outranks the dotenv, correctly — and pgw#1283
    # made the suite set this one for every test (the local cell store's bytes
    # live under it now, so it has to be isolated per run). Clear it so this
    # row still measures the dotenv it is about.
    monkeypatch.delenv("TENSORHUB_CACHE_DIR", raising=False)
    (tmp_path / ".env").write_text(
        "TENSORHUB_CACHE_DIR=/good\nSOME_OTHER_TOOL=1\nEDITOR=vim\n")
    assert config.load_settings().tensorhub_cache_dir == "/good"


def test_a_typo_in_a_secrets_dir_is_refused(tmp_path: Path) -> None:
    """A mounted secret nobody reads is a secret that did not arrive."""
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "TENSORHUB_TOKEEN").write_text("s3cret")
    from gen_worker.config import loader

    with pytest.raises(config.UnknownSettingError):
        loader._load_secrets_dir(str(secrets))


def test_a_correctly_named_secret_still_loads(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "TENSORHUB_TOKEN").write_text("s3cret\n")
    from gen_worker.config import loader

    assert loader._load_secrets_dir(str(secrets))["tensorhub_token"] == "s3cret"


def test_an_unknown_owned_env_does_not_refuse_the_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    """pgw#931: THE asymmetry, and it is load-bearing."""
    monkeypatch.setenv("GEN_WORKER_OOM_PROBE", "1")
    monkeypatch.setenv("GEN_WORKER_PROCESS_SPLIT", "1")
    config.load_settings()  # must not raise


def test_an_unknown_owned_env_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """pgw#931: Not refusing is not the same as saying nothing."""
    monkeypatch.setenv("GEN_WORKER_PREFR_AOT", "1")
    assert "GEN_WORKER_PREFR_AOT" in config.unrecognised_owned_env()


def test_a_known_env_is_not_reported_as_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEN_WORKER_C2PA_CERT_PATH", "/tmp/cert.pem")
    monkeypatch.setenv("GEN_WORKER_COMPUTE_CHILD", "1")   # IPC, deliberately unbound
    monkeypatch.setenv("GEN_WORKER_LOG_LEVEL", "DEBUG")   # library knob, ditto
    reported = config.unrecognised_owned_env()
    assert "GEN_WORKER_C2PA_CERT_PATH" not in reported
    assert "GEN_WORKER_COMPUTE_CHILD" not in reported
    assert "GEN_WORKER_LOG_LEVEL" not in reported


def test_a_foreign_env_is_never_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")  # 21 live releases set this
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    reported = config.unrecognised_owned_env()
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in reported
    assert "CUDA_VISIBLE_DEVICES" not in reported


def test_parent_control_installs_the_boot_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """pgw#931: The control parent holds the worker credential; the compute child is stripped of it and signs th..."""
    from gen_worker import worker_credential
    from gen_worker.procsplit.parent import ParentControl

    monkeypatch.setattr(worker_credential, "_TOKEN", "")
    monkeypatch.setattr(worker_credential, "_BOOTSTRAP", "")
    assert worker_credential.current() == ""

    ParentControl(config.Settings(bootstrap_worker_jwt="boot-jwt-xyz"))
    assert worker_credential.current() == "boot-jwt-xyz", (
        "building a control parent must install its boot credential — the "
        "child cannot sign through a parent that has none")


def test_the_retired_prefer_aot_env_is_reported_as_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """pgw#990: the gate is deleted, so a pod still being handed the env name must SAY SO."""
    monkeypatch.setenv("GEN_WORKER_PREFER_AOT", "1")
    assert "GEN_WORKER_PREFER_AOT" in config.unrecognised_owned_env()


# ============================================================================
# th#1087 — th#1087 stage D: worker reconcile of config-generation pushes.
# ============================================================================

def test_gen_bump_rewrites_snapshot_and_subprocess_sees_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snap_path = tmp_path / "cfg" / "runtime_config.msgpack"
    # ConfigStore exports the path into os.environ; route that through
    # monkeypatch so the export never leaks across tests.
    monkeypatch.setenv(SNAPSHOT_PATH_ENV, str(snap_path))
    store = ConfigStore(str(snap_path))

    # A desired-state push advertises gen 3; a dispatch stamps the values.
    assert store.observe(3, release_id="rel-1")
    assert store.stamp_function("config-echo", {"default_steps": 60}, 3)
    on_disk = read_snapshot(str(snap_path))
    assert on_disk.config_generation == 3
    assert on_disk.release_id == "rel-1"
    assert on_disk.parameters["config-echo"]["default_steps"] == 60

    # Stale/duplicate generations and unchanged stamps are ignored — file
    # untouched.
    before = snap_path.read_bytes()
    assert not store.observe(2)
    assert not store.observe(3)
    assert not store.stamp_function("config-echo", {"default_steps": 1}, 2)
    assert not store.stamp_function("config-echo", {"default_steps": 60}, 3)
    assert snap_path.read_bytes() == before

    # A real subprocess with an EXPLICIT env mapping still finds the
    # snapshot (run_process injects the known-path env var) and reads the
    # post-bump value.
    assert store.stamp_function("config-echo", {"default_steps": 75}, 4)
    lines: list[str] = []
    code = run_process(
        [
            sys.executable,
            "-c",
            "from gen_worker.runtime_config import read_snapshot; "
            "s = read_snapshot(); "
            "print(s.config_generation, s.parameters['config-echo']['default_steps'])",
        ],
        env={},
        on_line=lines.append,
    )
    assert code == 0, lines
    assert lines == ["4 75"]

    # A job already stamped at an older generation gets an immutable
    # per-invocation subprocess snapshot. A newer global push cannot change
    # the bytes that run_process(ctx=...) exposes to that child.
    ctx: Any = RequestContext("old-gen")
    old_values = {"default_steps": 40}
    ctx._set_config(
        old_values,
        snapshot=store.invocation_snapshot(
            "config-echo",
            old_values,
            4,
        ),
    )
    assert store.stamp_function("config-echo", {"default_steps": 100}, 5)
    lines = []
    code = run_process(
        [
            sys.executable,
            "-c",
            "from gen_worker.runtime_config import read_snapshot; "
            "s = read_snapshot(); "
            "print(s.config_generation, s.parameters['config-echo']['default_steps'])",
        ],
        ctx=ctx,
        env={},
        on_line=lines.append,
    )
    assert code == 0, lines
    assert lines == ["4 40"]
    assert read_snapshot(str(snap_path)).config_generation == 5


def test_failed_snapshot_write_does_not_advance_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snap_path = tmp_path / "runtime_config.msgpack"
    store = ConfigStore(str(snap_path))
    assert store.observe(1, release_id="rel-1")
    before = snap_path.read_bytes()

    def fail_replace(_source: str, _target: str) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(ConfigSnapshotWriteError):
        store.observe(2, release_id="rel-1")

    assert store.generation == 1
    assert snap_path.read_bytes() == before
    assert read_snapshot(str(snap_path)).config_generation == 1


def test_full_parameter_snapshot_is_atomic_and_release_fenced(
    tmp_path: Path,
) -> None:
    snap_path = tmp_path / "runtime_config.msgpack"
    store = ConfigStore(str(snap_path))
    raw = msgspec.msgpack.encode(
        {
            "config-echo": {"default_steps": 42, "scheduler": "euler"},
        }
    )
    assert store.apply_parameter_snapshot(
        raw,
        4,
        release_id="release-1",
    )
    assert read_snapshot(str(snap_path)).parameters == {
        "config-echo": {"default_steps": 42, "scheduler": "euler"},
    }
    before = snap_path.read_bytes()

    with pytest.raises(ConfigSnapshotWriteError, match="release_id mismatch"):
        store.apply_parameter_snapshot(
            raw,
            5,
            release_id="release-2",
        )
    assert store.generation == 4
    assert snap_path.read_bytes() == before


def _run_config_echo(
    conn: Any,
    request_id: str,
    *,
    generation: int = 0,
    params: dict[str, object] | None = None,
) -> str:
    conn.send(
        run_job=pb.RunJob(
            request_id=request_id,
            attempt=1,
            function_name="config-echo",
            input_payload=msgspec.msgpack.encode(EchoIn(text="x")),
            config_generation=generation,
            config_params=msgspec.msgpack.encode(params) if params is not None else b"",
        )
    )
    res = conn.wait_for(is_result_for(request_id)).job_result
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    return msgspec.msgpack.decode(res.inline)["response"]


def test_config_push_serves_next_request_pod_churn_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snap_path = tmp_path / "runtime_config.msgpack"
    monkeypatch.setenv(SNAPSHOT_PATH_ENV, str(snap_path))
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        assert _run_config_echo(conn, "r-before") == "ddim:30"

        # The hub's config-write push: a full-replace HelloAck advertises
        # the desired generation; the next RunJob carries this function's
        # effective values as msgpack.
        conn.send(
            hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                file_base_url=scheduler.file_base_url,
                desired_residency=pb.DesiredResidency(
                    release_id="rel-1",
                    config_generation=2,
                ),
            )
        )
        conn.wait_for(
            lambda m: (
                m.WhichOneof("msg") == "state_delta"
                and m.state_delta.observed_config_generation == 2
            )
        )

        # Same worker, same connection, no pod churn: the NEXT request
        # serves the pushed values, and the snapshot file tracked the push.
        values = {"default_steps": 90, "scheduler": "euler_a"}
        assert (
            _run_config_echo(
                conn,
                "r-after",
                generation=2,
                params=values,
            )
            == "euler_a:90"
        )
        snapshot = read_snapshot(str(snap_path))
        assert snapshot.config_generation == 2
        assert snapshot.release_id == "rel-1"
        assert snapshot.parameters["config-echo"] == values

        # Undeclared parameter names never leak into ctx.config OR the
        # subprocess snapshot.
        conn.send(
            hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                file_base_url=scheduler.file_base_url,
                desired_residency=pb.DesiredResidency(
                    release_id="rel-1",
                    config_generation=3,
                ),
            )
        )
        assert (
            _run_config_echo(
                conn,
                "r-undeclared",
                generation=3,
                params={**values, "bogus": True},
            )
            == "euler_a:90"
        )
        assert "bogus" not in read_snapshot(str(snap_path)).parameters["config-echo"]

        # A job already stamped at the older generation keeps its own
        # values, but cannot roll the worker's latest snapshot backward.
        assert (
            _run_config_echo(
                conn,
                "r-in-flight-old-gen",
                generation=2,
                params={"default_steps": 40, "scheduler": "ddim"},
            )
            == "ddim:40"
        )
        assert read_snapshot(str(snap_path)).config_generation == 3


# ============================================================================
# pgw#846 — pgw#846 retirement semantics: regional cells are RETIRED, and a
#   cell whose metadata still says ``mode='regional'`` is declined BY NAME —
#   never handed to the whole-graph ...
# ============================================================================

def test_arm_route_serves_only_the_whole_graph_mode() -> None:
    assert provision.arm_route("") == "aot_serve.enable"
    assert provision.arm_route("regional") is None
    assert provision.arm_route("some-future-recipe") is None


@pytest.mark.parametrize("mode", ["regional", "some-future-recipe"])
def test_a_cell_whose_mode_has_no_arm_is_declined_by_name_and_stays_eager(
    monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    from gen_worker import aot_serve

    def _never(*_a: Any, **_k: Any) -> bool:  # pragma: no cover - the defect
        raise AssertionError(
            f"a mode={mode!r} cell must never reach the whole-graph arm")

    monkeypatch.setattr(aot_serve, "enable", _never)

    class _Pipe:
        pass

    outcome = provision.arm_aot(
        _Pipe(), object(), None, Path("/nonexistent/cell.tar.gz"),
        0, {"mode": mode})
    assert outcome.armed is False
    # The decline is BY NAME, not a bare False.
    assert outcome.reason == "no_arm_for_mode"
