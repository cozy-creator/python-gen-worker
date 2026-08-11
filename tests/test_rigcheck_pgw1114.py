"""pgw#1114 — the fleet-line preflight's own red/green.

The assertion is the thing standing between a stale rig and a false verdict, so
it is tested the way a rig exercises it: real authority FILES on disk (the same
``endpoint.toml`` / ``fleet-floors.toml`` shapes production reads), a real
resolved-environment dict, and the abort path proven to be typed and loud rather
than a log line.

No torch import is faked into ``sys.modules``: :func:`resolve_environment` is the
only torch toucher and the assertion consumes its output as data, so the version
axes are driven by substituting that reader. Everything else — parsing, authority
discovery, strictest-floor selection, the printed table — runs for real.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from gen_worker import rigcheck

ENDPOINT_TOML = textwrap.dedent(
    """
    schema_version = 1
    main = "demo.main"

    [runtime]
    language = "python"

    [[build.profiles]]
    name = "default"
    accelerator = "cuda"
    cuda = "13.0"
    python = "3.12"
    torch = "2.13.0"
    """
)

FLEET_FLOORS_TOML = textwrap.dedent(
    """
    [floors]
    gen-worker = "0.102.0"
    torch = "2.13.0"
    transformers = "5.13"
    """
)


def _env(torch: str | None, cuda: str | None) -> dict:
    return {
        "python": "3.12.8",
        "platform": "linux",
        "packages": {"gen-worker": "0.104.0", "torch": torch or "absent"},
        "torch": torch,
        "cuda": cuda,
        "cudnn": 91002,
        "cuda_available": True,
        "device": "NVIDIA H100 80GB HBM3",
        "sm": "9.0",
        "vram_gib": 79.19,
        "driver": "580.65.06",
    }


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A pod-shaped layout: `/src/<endpoint>/endpoint.toml` next to a rig dir."""
    src = tmp_path / "src" / "demo-endpoint"
    src.mkdir(parents=True)
    (src / "endpoint.toml").write_text(ENDPOINT_TOML, encoding="utf-8")
    rig_dir = tmp_path / "src" / "rig"
    rig_dir.mkdir()
    monkeypatch.delenv(rigcheck.FLEET_LINE_FILE_ENV, raising=False)
    monkeypatch.chdir(tmp_path / "src")
    return rig_dir


def _use_env(monkeypatch: pytest.MonkeyPatch, torch: str | None, cuda: str | None):
    monkeypatch.setattr(rigcheck, "resolve_environment", lambda: _env(torch, cuda))


# --------------------------------------------------------------------------- #
# the expectation is READ, not hardcoded
# --------------------------------------------------------------------------- #


def test_reads_the_line_from_endpoint_toml(rig: Path) -> None:
    line = rigcheck.resolve_fleet_line(start=rig)
    assert line.torch == (2, 13, 0)
    assert line.cuda == (13, 0)
    assert any("endpoint.toml" in a.source for a in line.authorities)


def test_the_line_moves_when_the_authority_moves(rig: Path) -> None:
    """No constant in the module: edit the file, the floor follows."""
    toml = rig.parent / "demo-endpoint" / "endpoint.toml"
    toml.write_text(
        ENDPOINT_TOML.replace('torch = "2.13.0"', 'torch = "2.15.2"').replace(
            'cuda = "13.0"', 'cuda = "14.1"'
        ),
        encoding="utf-8",
    )
    line = rigcheck.resolve_fleet_line(start=rig)
    assert line.torch == (2, 15, 2)
    assert line.cuda == (14, 1)


def test_fleet_floors_is_an_authority_too(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(rigcheck.FLEET_LINE_FILE_ENV, raising=False)
    (tmp_path / "fleet-floors.toml").write_text(FLEET_FLOORS_TOML, encoding="utf-8")
    line = rigcheck.resolve_fleet_line(start=tmp_path)
    assert line.torch == (2, 13, 0)
    assert line.cuda is None  # only endpoint.toml declares CUDA
    assert any("fleet-floors.toml" in a.source for a in line.authorities)


def test_strictest_authority_wins(rig: Path) -> None:
    """Two authorities disagreeing must not let the lenient one through."""
    (rig.parent / "fleet-floors.toml").write_text(
        FLEET_FLOORS_TOML.replace('torch = "2.13.0"', 'torch = "2.14.0"'),
        encoding="utf-8",
    )
    assert rigcheck.resolve_fleet_line(start=rig).torch == (2, 14, 0)


def test_explicit_authority_path_env(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "elsewhere.toml"
    path.write_text(ENDPOINT_TOML, encoding="utf-8")
    monkeypatch.setenv(rigcheck.FLEET_LINE_FILE_ENV, str(path))
    line = rigcheck.resolve_fleet_line(start=tmp_path)
    assert (line.torch, line.cuda) == ((2, 13, 0), (13, 0))


def test_gen_worker_metadata_is_the_last_resort() -> None:
    """gen-worker declares `torch>=…` in its own `torch` extra; that is readable."""
    authority = rigcheck._metadata_authority("gen-worker")
    assert authority is not None, "gen-worker must declare a torch floor"
    assert authority.torch is not None and authority.torch >= (2, 13, 0)


# --------------------------------------------------------------------------- #
# RED — a stale rig aborts
# --------------------------------------------------------------------------- #


def test_stale_torch_aborts(rig: Path, monkeypatch) -> None:
    """pgw#1081 §R2 verbatim: torch 2.9.1+cu129 against a 2.13/cu13 fleet."""
    _use_env(monkeypatch, "2.9.1+cu129", "12.9")
    with pytest.raises(rigcheck.FleetLineMismatch) as excinfo:
        rigcheck.assert_fleet_line("pgw1081-r2", start=rig)
    message = str(excinfo.value)
    assert "REFUSING TO MEASURE" in message
    assert "torch 2.9.1+cu129 is below the fleet line 2.13.0" in message
    assert "CUDA build 12.9 is below the fleet line 13.0" in message
    assert "endpoint.toml" in message  # the report names its own authority


def test_right_torch_wrong_cuda_aborts(rig: Path, monkeypatch) -> None:
    """The rig_gpu_env.sh shape: correct torch, cu126 build. Still not the fleet."""
    _use_env(monkeypatch, "2.13.0+cu126", "12.6")
    with pytest.raises(rigcheck.FleetLineMismatch) as excinfo:
        rigcheck.assert_fleet_line(start=rig)
    assert "CUDA build 12.6 is below" in str(excinfo.value)
    assert "torch 2.13.0+cu126 is below" not in str(excinfo.value)


def test_torchless_environment_aborts(rig: Path, monkeypatch) -> None:
    env = _env(None, None)
    env["torch_import_error"] = "ImportError: no module named torch"
    monkeypatch.setattr(rigcheck, "resolve_environment", lambda: env)
    with pytest.raises(rigcheck.FleetLineMismatch, match="torch does not import"):
        rigcheck.assert_fleet_line(start=rig)


def test_no_authority_aborts(tmp_path: Path, monkeypatch) -> None:
    """Nothing to compare against is not permission to measure."""
    monkeypatch.delenv(rigcheck.FLEET_LINE_FILE_ENV, raising=False)
    monkeypatch.setattr(rigcheck, "_collect_authorities", lambda *a, **k: [])
    with pytest.raises(rigcheck.FleetLineUnknown, match="no evidence"):
        rigcheck.assert_fleet_line(start=tmp_path)


def test_there_is_no_override(rig: Path, monkeypatch) -> None:
    """No env, argument or config turns a mismatch into a warning."""
    for name in ("GEN_WORKER_RIGCHECK", "GEN_WORKER_SKIP_RIGCHECK",
                 "GEN_WORKER_RIGCHECK_FORCE", "RIGCHECK_ALLOW_STALE"):
        monkeypatch.setenv(name, "1")
    _use_env(monkeypatch, "2.9.1+cu129", "12.9")
    with pytest.raises(rigcheck.FleetLineMismatch):
        rigcheck.assert_fleet_line(start=rig)


# --------------------------------------------------------------------------- #
# GREEN — an on-line rig passes and PRINTS the table
# --------------------------------------------------------------------------- #


def test_on_the_line_passes_and_prints(rig: Path, monkeypatch, capsys) -> None:
    _use_env(monkeypatch, "2.13.0+cu130", "13.0")
    env = rigcheck.assert_fleet_line("h3-rig", start=rig, stream=None)
    printed = capsys.readouterr().err
    assert env["torch"] == "2.13.0+cu130"
    assert "h3-rig: on the fleet line." in printed
    for expected in ("torch", "2.13.0+cu130", "fleet floor 2.13.0", "driver",
                     "580.65.06", "NVIDIA H100 80GB HBM3", "endpoint.toml"):
        assert expected in printed, expected


def test_newer_than_the_floor_passes(rig: Path, monkeypatch) -> None:
    _use_env(monkeypatch, "2.14.0+cu131", "13.1")
    assert rigcheck.assert_fleet_line(start=rig)["torch"] == "2.14.0+cu131"


def test_cli_exit_codes(rig: Path, monkeypatch) -> None:
    _use_env(monkeypatch, "2.13.0+cu130", "13.0")
    assert rigcheck.main([str(rig)]) == 0
    _use_env(monkeypatch, "2.9.1+cu129", "12.9")
    assert rigcheck.main([str(rig)]) == 90


# --------------------------------------------------------------------------- #
# parsing edges that decide a verdict
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2.13.0+cu130", (2, 13, 0)),
        ("2.9.1+cu129", (2, 9, 1)),
        ("2.13.0a0+git1234", (2, 13, 0)),
        ("13.0", (13, 0)),
        ("", None),
        (None, None),
    ],
)
def test_version_parsing(text, expected) -> None:
    assert rigcheck._version(text) == expected


def test_two_component_comparison_is_not_lexical() -> None:
    """`2.9` must not read as newer than `2.13`."""
    assert rigcheck._below((2, 9, 1), (2, 13, 0))
    assert not rigcheck._below((2, 13, 0), (2, 13, 0))
    assert not rigcheck._below((2, 13), (2, 13, 0))


@pytest.mark.parametrize(
    "requirement,expected",
    [
        ("torch>=2.13.0", (2, 13, 0)),
        ("torch>=2.13.0; extra == 'torch'", (2, 13, 0)),
        ("torch==2.12.1", (2, 12, 1)),
        ("torch~=2.13.0", (2, 13, 0)),
        ("torchvision>=0.28.0", None),
        ("torchaudio>=2.8,<3", None),
    ],
)
def test_requirement_floor(requirement, expected) -> None:
    assert rigcheck._requirement_floor(requirement, "torch") == expected
