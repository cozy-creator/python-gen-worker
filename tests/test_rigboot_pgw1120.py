"""pgw#1120 — the host-driver gate's own red/green.

Two lanes lost an evening to the same host fact on 2026-08-11: RunPod's driver is
per-host, ``570.211.01`` is CUDA 12.8, and a cu130 torch imports perfectly there
before failing on the first allocation ~20 minutes in — after the weight fetch,
looking like a torch bug. What is tested here is exactly the machinery that turns
that into a five-second refusal at bring-up:

* the ``nvidia-smi`` parsing (real output shapes, including the container image
  that prints no CUDA banner),
* the driver-vs-CUDA-line decision, whose one dangerous edge is that
  ``580.159.04`` is NEWER than the ``580.65.06`` floor — a float comparison, the
  obvious implementation, rejects the exact host that works,
* the three paths (``native`` / ``compat`` / ``reroll``) and the typed failure,
* ``rigcheck``'s assertion refusing a host where the wheel is right and the
  ALLOCATION is not, with the distinct exception type and exit code.

The allocation itself is substituted (there is no card in CI, and the point of
the module is that a card's presence is not the question); everything else —
parsing, comparison, path selection, message text — runs for real.
"""

from __future__ import annotations

import pytest

from gen_worker import rigboot

SMI_QUERY_H100 = "570.211.01, NVIDIA H100 80GB HBM3\n"
SMI_BANNER_570 = (
    "Tue Aug 11 09:44:54 2026\n"
    "+-----------------------------------------------------------------------------+\n"
    "| NVIDIA-SMI 570.211.01   Driver Version: 570.211.01   CUDA Version: 12.8     |\n"
)
SMI_QUERY_4X = (
    "580.159.04, NVIDIA H100 80GB HBM3\n"
    "580.159.04, NVIDIA H100 80GB HBM3\n"
)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def test_parses_driver_gpu_and_banner_cuda() -> None:
    probe = rigboot.parse_smi(SMI_QUERY_H100, SMI_BANNER_570)
    assert probe.driver == "570.211.01"
    assert probe.driver_cuda == "12.8"
    assert probe.gpus == ("NVIDIA H100 80GB HBM3",)
    assert probe.present


def test_parses_multi_gpu_and_survives_a_missing_banner() -> None:
    """Some container images print no banner; the driver version is never absent."""
    probe = rigboot.parse_smi(SMI_QUERY_4X, "")
    assert probe.driver == "580.159.04"
    assert probe.driver_cuda is None
    assert len(probe.gpus) == 2


def test_no_driver_is_reported_not_guessed() -> None:
    probe = rigboot.parse_smi("", "")
    assert probe.driver is None and not probe.present


# --------------------------------------------------------------------------- #
# the decision — the float trap is the whole test
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "driver,cuda,expected",
    [
        ("570.211.01", "13.0", False),  # the host that blocked pgw#1043
        ("580.159.04", "13.0", True),   # US-GA-2 on 2026-08-09 — NEWER than the floor
        ("580.65.06", "13.0", True),    # exactly the floor
        ("580.65.05", "13.0", False),   # one patch below it
        ("570.211.01", "12.8", True),   # the same host runs the CUDA it advertises
        ("525.60.13", "13.0", False),
    ],
)
def test_driver_vs_cuda_line(driver: str, cuda: str, expected: bool) -> None:
    assert rigboot.driver_supports_cuda(driver, cuda) is expected


def test_five_eighty_one_fifty_nine_is_newer_than_the_floor() -> None:
    """As floats 580.159 < 580.65, which would reject a host that works."""
    assert rigboot.driver_supports_cuda("580.159.04", "13.0") is True
    assert rigboot.min_driver_for_cuda("13.0") == (580, 65, 6)


def test_an_off_table_cuda_major_is_undecidable_not_a_guess() -> None:
    assert rigboot.driver_supports_cuda("580.159.04", "99.0") is None
    assert rigboot.min_driver_for_cuda("99.0") is None


def test_compat_package_name() -> None:
    assert rigboot.compat_package("13.0") == "cuda-compat-13-0"
    assert rigboot.compat_package("12.8") == "cuda-compat-12-8"
    assert rigboot.compat_package("nonsense") is None


# --------------------------------------------------------------------------- #
# the three paths
# --------------------------------------------------------------------------- #


def _host(monkeypatch, driver: str) -> None:
    monkeypatch.setattr(
        rigboot,
        "probe_host",
        lambda: rigboot.parse_smi(f"{driver}, NVIDIA H100 80GB HBM3\n", ""),
    )


def _allocations(monkeypatch, *verdicts: dict) -> list[int]:
    """Substitute the real allocation with a scripted sequence of verdicts."""
    calls = [0]
    queue = list(verdicts)

    def fake() -> dict:
        calls[0] += 1
        return queue.pop(0) if queue else verdicts[-1]

    monkeypatch.setattr(rigboot, "verify_allocation", fake)
    return calls


OK = {"ok": True, "device": "NVIDIA H100 80GB HBM3", "sm": "9.0", "vram_gib": 79.19}
TOO_OLD = {
    "ok": False,
    "error": "RuntimeError: CUDA initialization: driver too old (found version 12080)",
}


def test_native_path_does_not_install_anything(monkeypatch) -> None:
    _host(monkeypatch, "580.159.04")
    _allocations(monkeypatch, OK)
    monkeypatch.setattr(
        rigboot, "_install_compat", lambda *a, **k: pytest.fail("must not install")
    )
    record = rigboot.ensure_cuda_line("13.0")
    assert record["path"] == "native"
    assert record["native_ok"] is True


def test_compat_path_installs_repairs_and_reverifies(monkeypatch, tmp_path) -> None:
    """The pgw#1081 outcome: a 570 host made usable, and it SAYS it was."""
    _host(monkeypatch, "570.211.01")
    calls = _allocations(monkeypatch, TOO_OLD, OK)
    compat = tmp_path / "compat"
    compat.mkdir()
    (compat / "libcuda.so.580.65.06").write_text("", encoding="utf-8")
    installed: list[str] = []

    def fake_install(cuda: str, log) -> str:
        installed.append(cuda)
        return str(compat)

    monkeypatch.setattr(rigboot, "_install_compat", fake_install)
    monkeypatch.setattr(rigboot, "_persist_ld_path", lambda d, log: None)
    monkeypatch.setattr(rigboot, "_existing_compat_dir", lambda cuda: None)

    record = rigboot.ensure_cuda_line("13.0")
    assert installed == ["13.0"]
    assert record["path"] == "compat", record
    assert record["compat_dir"] == str(compat)
    assert record["native_ok"] is False
    # both allocations ran: the failing one is what triggered the repair, the
    # second is what proves it.
    assert calls[0] == 2
    assert record["allocation_first"]["ok"] is False


def test_reroll_when_compat_cannot_repair_it(monkeypatch) -> None:
    _host(monkeypatch, "570.211.01")
    _allocations(monkeypatch, TOO_OLD, TOO_OLD)
    monkeypatch.setattr(rigboot, "_install_compat", lambda cuda, log: "/usr/local/cuda/compat")
    monkeypatch.setattr(rigboot, "_persist_ld_path", lambda d, log: None)
    monkeypatch.setattr(rigboot, "_existing_compat_dir", lambda cuda: None)
    record = rigboot.ensure_cuda_line("13.0")
    assert record["path"] == "reroll"
    assert "still fails" in record["reason"]


def test_reroll_when_the_package_is_not_installable(monkeypatch) -> None:
    _host(monkeypatch, "570.211.01")
    _allocations(monkeypatch, TOO_OLD)
    monkeypatch.setattr(rigboot, "_install_compat", lambda cuda, log: None)
    monkeypatch.setattr(rigboot, "_existing_compat_dir", lambda cuda: None)
    record = rigboot.ensure_cuda_line("13.0")
    assert record["path"] == "reroll"
    assert "cuda-compat-13-0" in record["reason"]


def test_no_compat_reports_without_repairing(monkeypatch) -> None:
    _host(monkeypatch, "570.211.01")
    _allocations(monkeypatch, TOO_OLD)
    monkeypatch.setattr(
        rigboot, "_install_compat", lambda *a, **k: pytest.fail("must not install")
    )
    assert rigboot.ensure_cuda_line("13.0", allow_compat=False)["path"] == "reroll"


def test_assert_cuda_usable_is_typed_and_says_reroll(monkeypatch) -> None:
    _host(monkeypatch, "570.211.01")
    _allocations(monkeypatch, TOO_OLD)
    monkeypatch.setattr(rigboot, "_install_compat", lambda cuda, log: None)
    monkeypatch.setattr(rigboot, "_existing_compat_dir", lambda cuda: None)
    with pytest.raises(rigboot.DriverTooOld) as caught:
        rigboot.assert_cuda_usable("13.0")
    message = str(caught.value)
    assert "RE-ROLL THE HOST" in message
    assert "NOT a torch defect" in message


def test_no_gpu_at_all_is_a_different_mistake(monkeypatch) -> None:
    monkeypatch.setattr(rigboot, "probe_host", lambda: rigboot.parse_smi("", ""))
    with pytest.raises(rigboot.NoDriver):
        rigboot.ensure_cuda_line("13.0")


# --------------------------------------------------------------------------- #
# exit codes — the bring-up script's actual contract
# --------------------------------------------------------------------------- #


def test_cli_exit_91_means_reroll(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        rigboot,
        "ensure_cuda_line",
        lambda cuda, **k: {"path": "reroll", "driver": "570.211.01", "reason": "x"},
    )
    assert rigboot.main(["--cuda", "13.0"]) == 91
    assert "RIGBOOT_REROLL" in capsys.readouterr().err


def test_cli_exit_92_means_no_gpu(monkeypatch) -> None:
    def boom(cuda, **k):
        raise rigboot.NoDriver("no driver")

    monkeypatch.setattr(rigboot, "ensure_cuda_line", boom)
    assert rigboot.main(["--cuda", "13.0"]) == 92


def test_cli_exit_0_on_a_usable_host(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        rigboot, "ensure_cuda_line", lambda cuda, **k: {"path": "compat"}
    )
    out = tmp_path / "rigboot.json"
    assert rigboot.main(["--cuda", "13.0", "--json", str(out)]) == 0
    assert '"compat"' in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# rigcheck's half: the wheel can be perfect and the HOST still wrong
# --------------------------------------------------------------------------- #

from gen_worker import rigcheck  # noqa: E402 — read after the rigboot section


def _on_line_env(**overrides) -> dict:
    env = {
        "python": "3.12.3",
        "platform": "linux",
        "packages": {"gen-worker": "0.104.0", "torch": "2.13.0+cu130"},
        "torch": "2.13.0+cu130",
        "cuda": "13.0",
        "cudnn": 92000,
        "device": "NVIDIA H100 80GB HBM3",
        "sm": "9.0",
        "vram_gib": 79.19,
        "driver": "570.211.01",
        "cuda_usable": False,
        "cuda_unusable_reason": (
            "RuntimeError: CUDA initialization: driver too old (found version 12080)"
        ),
        "cuda_unusable_class": "driver_too_old",
    }
    env.update(overrides)
    return env


def _authority(tmp_path, monkeypatch):
    (tmp_path / "endpoint.toml").write_text(
        '[[build.profiles]]\nname = "default"\naccelerator = "cuda"\n'
        'cuda = "13.0"\ntorch = "2.13.0"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv(rigcheck.FLEET_LINE_FILE_ENV, str(tmp_path / "endpoint.toml"))


def test_perfect_versions_unusable_card_is_refused(tmp_path, monkeypatch) -> None:
    """The exact pgw#1043 trap: every string right, nothing allocates."""
    _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(rigcheck, "resolve_environment", _on_line_env)
    with pytest.raises(rigcheck.CudaUnusable) as caught:
        rigcheck.assert_fleet_line("recell", start=tmp_path)
    message = str(caught.value)
    assert "this HOST cannot run it" in message
    assert "driver_too_old" in message
    assert "cuda-compat-13-0" in message
    assert "gen_worker.rigboot" in message
    # it must NOT tell the reader to rebuild the environment: that is the wrong
    # instruction here, and following it costs another pod.
    assert "Rebuild the environment" not in message


def test_the_host_refusal_is_still_a_fleet_line_mismatch(tmp_path, monkeypatch) -> None:
    """Subclass, so every existing `except FleetLineMismatch` rig still aborts."""
    _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(rigcheck, "resolve_environment", _on_line_env)
    with pytest.raises(rigcheck.FleetLineMismatch):
        rigcheck.assert_fleet_line("recell", start=tmp_path)


def test_a_usable_card_on_the_line_passes(tmp_path, monkeypatch, capsys) -> None:
    _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(
        rigcheck,
        "resolve_environment",
        lambda: _on_line_env(cuda_usable=True, cuda_unusable_reason=None,
                             cuda_unusable_class=None),
    )
    env = rigcheck.assert_fleet_line("recell", start=tmp_path)
    assert env["cuda_usable"] is True
    assert "yes (real allocation)" in capsys.readouterr().err


def test_a_cardless_host_is_not_a_host_failure(tmp_path, monkeypatch) -> None:
    """No driver at all is a different mistake and must not be reported as one."""
    _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(
        rigcheck,
        "resolve_environment",
        lambda: _on_line_env(driver=None, cuda_usable=False),
    )
    rigcheck.assert_fleet_line("recell", start=tmp_path)


def test_wheel_mismatch_stays_a_plain_mismatch(tmp_path, monkeypatch) -> None:
    _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(
        rigcheck,
        "resolve_environment",
        lambda: _on_line_env(torch="2.9.1+cu129", cuda="12.9", cuda_usable=True),
    )
    with pytest.raises(rigcheck.FleetLineMismatch) as caught:
        rigcheck.assert_fleet_line("recell", start=tmp_path)
    assert not isinstance(caught.value, rigcheck.CudaUnusable)


def test_cli_exit_91_for_a_host_fault(tmp_path, monkeypatch) -> None:
    """90 means "rebuild the environment"; 91 means "repair or re-roll the host"."""
    _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(rigcheck, "resolve_environment", _on_line_env)
    assert rigcheck.main([str(tmp_path)]) == 91


def test_usability_is_measured_by_a_real_allocation(monkeypatch) -> None:
    """`resolve_environment` reports the probe's verdict, not a version string."""
    from gen_worker import cuda_probe

    monkeypatch.setattr(
        cuda_probe,
        "probe_cuda",
        lambda *a, **k: cuda_probe.CudaProbeResult(
            ok=False, reason="CUDA initialization: driver too old (found version 12080)"
        ),
    )
    out = rigcheck._usability({"torch": "2.13.0+cu130"})
    assert out == {
        "cuda_usable": False,
        "cuda_unusable_reason": (
            "CUDA initialization: driver too old (found version 12080)"
        ),
        "cuda_unusable_class": "driver_too_old",
    }
