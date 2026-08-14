"""§1.18 — one config pipeline in, and the struct is passed.

Never load envs in the middle of code; load from the config pipeline and pass
the value around. Three properties:

1. `config.current()` RAISES when no process entry published `Settings`. A
   fallback that reads the environment there and then hands a module running
   before bootstrap config nobody validated, and leaves nothing in the tree able
   to say what a process was actually configured as.
2. A key inside a namespace we OWN, in an operator-authored source, is refused
   rather than accepted and ignored. A `_normalize_key` returning `None` makes
   every source layer silently skip it, so a typo'd `TENSORHUB_CHACE_DIR`
   evaporates with no diagnostic anywhere.
3. That refusal deliberately does NOT extend to the process environment, and
   the asymmetry is the whole design. Tensorhub injects owned-namespace names
   this worker has no reader for (`GEN_WORKER_OOM_PROBE`,
   `GEN_WORKER_PROCESS_SPLIT`), so refusing there turns a hub-side addition
   into a fleet of dead pods. It is REPORTED instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker import config


# ---------------------------------------------------------------------------
# 1. The struct is published, never found
# ---------------------------------------------------------------------------


def test_current_raises_when_nothing_was_installed() -> None:
    config.reset_for_test()
    try:
        with pytest.raises(config.SettingsNotInstalled):
            config.current()
    finally:
        config.reload_for_test()


def test_get_settings_is_gone() -> None:
    """The cached process-global accessor must not come back.

    Its absence is the deliverable, so it is asserted rather than assumed: a
    re-added `get_settings` would pass every other test in this suite while
    restoring exactly the defect pgw#931 removed.
    """
    assert not hasattr(config, "get_settings")
    from gen_worker.config import loader

    assert not hasattr(loader, "get_settings")


def test_current_or_takes_its_fallback_as_a_value() -> None:
    """A standalone default must be visible AT THE CALL SITE, not be a silent
    env read. That is what makes `models/` usable outside a worker bring-up
    without reintroducing a lazy loader."""
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


# ---------------------------------------------------------------------------
# 2. An owned key in a file source is refused, not ignored
# ---------------------------------------------------------------------------


def test_a_typo_in_dotenv_is_refused(tmp_path: Path, monkeypatch) -> None:
    """The deliverable's own example. `TENSORHUB_CHACE_DIR` used to be accepted
    and inert — the operator's intent evaporated silently."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TENSORHUB_CHACE_DIR=/oops\n")
    with pytest.raises(config.UnknownSettingError) as excinfo:
        config.load_settings()
    assert "TENSORHUB_CHACE_DIR" in str(excinfo.value), (
        "the refusal must NAME the key, or it is a wall of text an operator "
        "cannot act on")


def test_foreign_keys_in_dotenv_are_still_ignored(tmp_path: Path, monkeypatch) -> None:
    """Scoped to the namespaces we own, deliberately. A real `.env` carries
    hundreds of entries belonging to other tools; refusing those would make the
    worker unusable and the rule would be turned off within a day."""
    monkeypatch.chdir(tmp_path)
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


# ---------------------------------------------------------------------------
# 3. The process environment is REPORTED, never refused
# ---------------------------------------------------------------------------


def test_an_unknown_owned_env_does_not_refuse_the_boot(monkeypatch) -> None:
    """THE asymmetry, and it is load-bearing.

    `GEN_WORKER_OOM_PROBE` and `GEN_WORKER_PROCESS_SPLIT` are real names
    Tensorhub's source injects and this worker has no reader for. If the file
    sources' refusal were applied here, every pod receiving one would be dead
    on arrival.
    """
    monkeypatch.setenv("GEN_WORKER_OOM_PROBE", "1")
    monkeypatch.setenv("GEN_WORKER_PROCESS_SPLIT", "1")
    config.load_settings()  # must not raise


def test_an_unknown_owned_env_is_reported(monkeypatch) -> None:
    """Not refusing is not the same as saying nothing. A misspelled
    `GEN_WORKER_PREFR_AOT` in a release declaration is silently inert, so it is
    named at boot instead of vanishing."""
    monkeypatch.setenv("GEN_WORKER_PREFR_AOT", "1")
    assert "GEN_WORKER_PREFR_AOT" in config.unrecognised_owned_env()


def test_a_known_env_is_not_reported_as_unknown(monkeypatch) -> None:
    monkeypatch.setenv("GEN_WORKER_C2PA_CERT_PATH", "/tmp/cert.pem")
    monkeypatch.setenv("GEN_WORKER_COMPUTE_CHILD", "1")   # IPC, deliberately unbound
    monkeypatch.setenv("GEN_WORKER_LOG_LEVEL", "DEBUG")   # library knob, ditto
    reported = config.unrecognised_owned_env()
    assert "GEN_WORKER_C2PA_CERT_PATH" not in reported
    assert "GEN_WORKER_COMPUTE_CHILD" not in reported
    assert "GEN_WORKER_LOG_LEVEL" not in reported


def test_a_foreign_env_is_never_reported(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")  # 21 live releases set this
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    reported = config.unrecognised_owned_env()
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in reported
    assert "CUDA_VISIBLE_DEVICES" not in reported


# ---------------------------------------------------------------------------
# 4. Derived process facts are established WHERE THE SETTINGS ARRIVE
# ---------------------------------------------------------------------------


def test_parent_control_installs_the_boot_credential(monkeypatch) -> None:
    """The control parent holds the worker credential; the compute child is
    stripped of it and signs through the parent.

    pgw#931 first installed the boot token in `procsplit.parent.run_parent`
    ONLY. That is one of several ways a control parent gets built — the split
    harness and the group-process tests construct `ParentControl` directly — so
    every other path produced a parent with no credential and the mediated C2PA
    sign refused with *"this pod holds no worker JWT"*. Caught by
    `test_procsplit_security_pgw763` after the merge.

    The lesson is §4.22's: a fact and its carrier must be established together,
    at the seam the fact's owner is constructed, not at one convenient entry
    point. This asserts the seam rather than the entry point.
    """
    from gen_worker import worker_credential
    from gen_worker.procsplit.parent import ParentControl

    monkeypatch.setattr(worker_credential, "_TOKEN", "")
    monkeypatch.setattr(worker_credential, "_BOOTSTRAP", "")
    assert worker_credential.current() == ""

    ParentControl(config.Settings(bootstrap_worker_jwt="boot-jwt-xyz"))
    assert worker_credential.current() == "boot-jwt-xyz", (
        "building a control parent must install its boot credential — the "
        "child cannot sign through a parent that has none")


def test_the_retired_prefer_aot_env_is_reported_as_unknown(monkeypatch) -> None:
    """pgw#990: the gate is deleted, so a pod still being handed the env name
    must SAY SO. A stale endpoint_env row is the exact trap that hid the
    un-armed adoption path for three A1 attempts; silence about it is how it
    would hide again."""
    monkeypatch.setenv("GEN_WORKER_PREFER_AOT", "1")
    assert "GEN_WORKER_PREFER_AOT" in config.unrecognised_owned_env()
