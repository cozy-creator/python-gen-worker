"""The custom-Dockerfile path is checked by a REGISTRY, not by another patch.

pgw#1016, pgw#1017, pgw#1017's GAP A and pgw#1068 are four instances of one
defect: a step the synthesized Dockerfile takes, which a hand-written Dockerfile
does not, which nothing asks for until a pod pays. Each was closed one-off, and
pgw#1068 closed its instance by ALSO sweeping eight fleet Dockerfiles by hand.

`gen_worker.build_guarantees` is the mechanism that ends the pattern: one
enumerated table of the steps, each row naming where the platform itself
refuses, and a checker that runs on a source tree for $0.00 before a build
exists. These tests hold the mechanism to the two properties that make it one:

* a row cannot exist without a platform refusal behind it, and where that
  refusal is an in-image precondition it must be REACHABLE from the gate — an
  unwired verifier is exactly the "holds on one path only" defect, one level up;
* the registry, not a hard-coded string, is what the examples and the docs are
  checked against, so pgw#1017's instance is a CONSUMER of the mechanism rather
  than a second bespoke assertion.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from gen_worker import aot_preconditions as ap
from gen_worker import build_guarantees as bg
from gen_worker import cuda_root as cr

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
DOC = REPO / "docs" / "dockerfile.md"
MICRO = EXAMPLES / "micro-diffusion"
FENCED_DOCKERFILE = re.compile(r"```dockerfile\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# The registry itself — a row with no enforcement behind it is decoration
# ---------------------------------------------------------------------------

def test_the_registry_is_well_formed() -> None:
    ids = [row.id for row in bg.REGISTRY]
    assert len(ids) == len(set(ids)), "duplicate guarantee id"
    for row in bg.REGISTRY:
        assert row.id.islower() and " " not in row.id
        assert row.applies in (bg.ALWAYS, bg.DECLARES_AOT_EXPORT)
        assert (row.require is None) != (row.forbid is None), (
            f"{row.id}: a row states either what must be present or what must "
            f"not, never both and never neither")
        assert row.synthesized, f"{row.id}: name the synthesized step it mirrors"
        assert row.verified_by, (
            f"{row.id}: a guarantee with no platform refusal behind it is a "
            f"style preference. Name where the build or the publish dies")
        assert row.cost, f"{row.id}: say what skipping it costs"


@pytest.mark.parametrize(
    "row", [r for r in bg.REGISTRY if r.verified_by.startswith(bg.PRECONDITION)],
    ids=lambda r: r.id)
def test_a_precondition_backed_row_is_REACHABLE_from_the_gate(row, monkeypatch
                                                              ) -> None:
    """The registry may not claim a verifier the gate never emits.

    This is the mechanism applied to itself. `_cuda_root_row(...)` asserted in
    isolation proves a function's opinion and nothing about a build; a row that
    is never stamped into `endpoint.lock` is a guarantee that holds nowhere,
    which is the defect one level up from the one this module addresses.
    """
    name = row.verified_by[len(bg.PRECONDITION):]
    check = getattr(ap, name)

    monkeypatch.setattr(ap, "has_export_declaration", lambda _f: True)
    monkeypatch.setattr(ap, "declaration_import_failures", lambda: ())
    monkeypatch.setattr(
        ap, "export_declaration",
        lambda _f: type("D", (), {"classes": (), "targets": ()})())
    monkeypatch.setattr(ap, "_torch_is_cuda_build", lambda: True)
    monkeypatch.setattr(cr, "torch_cuda_home", lambda: "")

    rows = ap.static_mint_preconditions({"micro-diffusion": 0},
                                        torch_available=True,
                                        torch_version="2.13.0")
    assert check in {r.check for r in rows}, (
        f"{row.id} names precondition {check!r}, which the gate never emits")


# ---------------------------------------------------------------------------
# Every example that ships a Dockerfile — the fleet check, as a test
# ---------------------------------------------------------------------------

def _endpoint_dirs() -> list[Path]:
    return sorted(p for p in EXAMPLES.iterdir() if p.is_dir())


@pytest.mark.parametrize("path", _endpoint_dirs(), ids=lambda p: p.name)
def test_every_example_satisfies_its_build_guarantees(path: Path) -> None:
    findings = bg.check_endpoint(path)
    assert not findings, "\n".join(str(f) for f in findings)


def test_a_dockerfile_less_endpoint_is_asked_for_nothing() -> None:
    """`flux2-klein-image` declares an AOT export and ships no Dockerfile.

    It takes the synthesized path, where the hub writes every one of these
    layers itself. Asking it for the author-side lines would be the mirror
    image of the bug — a refusal for a step something else already took.
    """
    src = EXAMPLES / "flux2-klein-image"
    assert bg.declares_aot_export(src), "fixture drifted: no compile= here"
    assert not (src / "Dockerfile").exists()
    assert bg.check_endpoint(src) == []


# ---------------------------------------------------------------------------
# RED — each of the four historical instances, replayed against the mechanism
# ---------------------------------------------------------------------------

def _micro_dockerfile() -> str:
    return (MICRO / "Dockerfile").read_text()


def test_RED_the_pre_pgw1068_micro_dockerfile_is_REFUSED() -> None:
    """The fourth instance, exactly as it shipped, caught for $0.00.

    Until pgw#1068 (`8b05533f`) this file had no `cuda_root` step, and nothing
    in CI said so — the miss was found by a live build and fixed one-off, then
    the rest of the fleet was swept by hand. Dropping the line reproduces that
    tree, and the checker must refuse it by name.
    """
    without = "\n".join(
        line for line in _micro_dockerfile().splitlines()
        if "gen_worker.cuda_root" not in line)
    findings = bg.check_dockerfile(without, aot=True, why_aot="main_w8a8.py")
    assert [f.guarantee for f in findings] == ["cuda_root"]
    assert "RUN python -m gen_worker.cuda_root" in findings[0].message
    assert "aot precondition cuda_root" in findings[0].message, (
        "name the string the author will read in the failing build, not the "
        "constant the registry stores")
    assert "main_w8a8.py" in findings[0].message, (
        "the refusal must name WHY the row applies to this tree")


def test_RED_dropping_the_toolchain_layer_is_REFUSED() -> None:
    """pgw#1017's own instance, now a consumer of the registry."""
    without = _micro_dockerfile().replace("ca-certificates curl g++",
                                          "ca-certificates curl")
    findings = bg.check_dockerfile(without, aot=True)
    assert [f.guarantee for f in findings] == ["cxx_toolchain"]
    assert "pgw#823" in findings[0].message


def test_RED_a_new_family_copying_the_minimum_viable_dockerfile_is_REFUSED(
        tmp_path: Path) -> None:
    """pgw#1068's shape, generalized: a NEW hand-written family.

    This is the case the four one-off patches never covered — not the example
    that was fixed, but the next author who copies the smallest documented
    Dockerfile and declares a `compile=` export. Both AOT rows must fire, and
    neither may fire for the family next door that declares no export.
    """
    src = tmp_path / "new-family"
    (src / "src" / "newfam").mkdir(parents=True)
    (src / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "WORKDIR /app\n"
        "COPY . /app\n"
        "RUN pip install -e .\n"
        "RUN mkdir -p /app/.tensorhub \\\n"
        "    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock\n"
        'ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]\n')
    main = src / "src" / "newfam" / "main.py"
    main.write_text(
        "@endpoint(\n"
        '    compile=Compile(family="newfam", shapes=((512, 512),)),\n'
        ")\ndef generate():\n    ...\n")

    assert bg.declares_aot_export(src) == ["src/newfam/main.py"]
    assert sorted(f.guarantee for f in bg.check_endpoint(src)) == [
        "cuda_root", "cxx_toolchain"]

    main.write_text("@endpoint()\ndef generate():\n    ...\n")
    assert bg.check_endpoint(src) == [], (
        "a JIT-only family owes the AOT lane nothing — intake mode is ruled, "
        "and a CPU image must not start failing builds over a CUDA root it "
        "never needed")


def test_RED_the_always_rows_fire_for_a_family_with_no_export(tmp_path: Path
                                                              ) -> None:
    src = tmp_path / "bare"
    src.mkdir()
    (src / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY . /app\n")
    assert sorted(f.guarantee for f in bg.check_endpoint(src)) == [
        "discovery_lock", "worker_entrypoint"]


def test_RED_pgw1016s_cache_mount_is_the_same_table(tmp_path: Path) -> None:
    """pgw#1016 is not a separate assertion any more — it is one row."""
    poisoned = _micro_dockerfile().replace(
        "RUN uv export --no-cache",
        "RUN --mount=type=cache,target=/root/.cache uv export --no-cache")
    findings = bg.check_dockerfile(poisoned, aot=True)
    assert [f.guarantee for f in findings] == ["buildkit_cache_mount"]
    assert "invalid_tarball" in findings[0].message


def test_a_comment_ABOUT_a_step_is_not_the_step(tmp_path: Path) -> None:
    """The trap this exact example fell into for two days.

    Between pgw#1017 and pgw#1068 micro-diffusion's Dockerfile carried a
    paragraph explaining that it had no CUDA root and was therefore not
    pod-ready. A checker that greps raw bytes reads that paragraph as
    compliance and stays green over the defect it describes.
    """
    src = tmp_path / "commented"
    (src / "src").mkdir(parents=True)
    (src / "src" / "m.py").write_text('compile=Compile(family="x")\n')
    (src / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "# a mint on this image would die at CUDA_HOME: it needs\n"
        "# `python -m gen_worker.cuda_root`, which this file does not run,\n"
        "# and g++, which it does not install either.\n"
        "RUN mkdir -p /app/.tensorhub \\\n"
        "    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock\n"
        'ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]\n')
    assert sorted(f.guarantee for f in bg.check_endpoint(src)) == [
        "cuda_root", "cxx_toolchain"]


def test_the_BAN_reads_comments_because_the_hub_does() -> None:
    """The mirror rule, and it points the other way.

    tensorhub's validator matches the raw bytes of the whole Dockerfile, so a
    commented-out cache mount is refused exactly like a live one. `forbid` rows
    must therefore read comments even though `require` rows must not.
    """
    text = ("FROM python:3.12-slim\n"
            "# RUN --mount=type=cache,target=/root/.cache pip install .\n"
            "RUN python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock\n"
            'ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]\n')
    assert [f.guarantee for f in bg.check_dockerfile(text, aot=False)] == [
        "buildkit_cache_mount"]
    assert bg.guarantee("buildkit_cache_mount").reads_comments is True
    assert bg.guarantee("cuda_root").reads_comments is False


# ---------------------------------------------------------------------------
# The docs are a consumer too — pgw#1016's root cause was a doc nobody ran
# ---------------------------------------------------------------------------

def _complete_doc_dockerfiles() -> list[str]:
    return [b for b in FENCED_DOCKERFILE.findall(DOC.read_text())
            if "FROM" in b and "ENTRYPOINT" in b]


def test_the_doc_teaches_at_least_one_complete_AOT_capable_dockerfile() -> None:
    """An author with a `compile=` export must find a file they can copy.

    pgw#1016's root cause was a documented Dockerfile that no first-party
    family exercised. Running the registry over the doc's own blocks is what
    keeps the page and the platform from drifting apart again.
    """
    blocks = _complete_doc_dockerfiles()
    assert blocks, "docs/dockerfile.md teaches no complete Dockerfile"
    assert any(not bg.check_dockerfile(b, aot=True) for b in blocks), (
        "no documented Dockerfile satisfies the AOT rows; an author with a "
        "compile= export has nothing to copy that would build")


@pytest.mark.parametrize("block", _complete_doc_dockerfiles(),
                         ids=lambda b: b.splitlines()[0][:40])
def test_every_complete_doc_dockerfile_satisfies_the_always_rows(block: str
                                                                 ) -> None:
    findings = bg.check_dockerfile(block, aot=False, path="docs/dockerfile.md")
    assert not findings, "\n".join(str(f) for f in findings)


# ---------------------------------------------------------------------------
# The invocation — the author runs it, the SDK owns whether it is right
# ---------------------------------------------------------------------------

def test_the_checker_is_runnable_as_one_line_and_exits_nonzero(tmp_path: Path
                                                               ) -> None:
    src = tmp_path / "ep"
    src.mkdir()
    (src / "Dockerfile").write_text("FROM python:3.12-slim\n")
    proc = subprocess.run(
        [sys.executable, "-m", "gen_worker.build_guarantees", str(src)],
        capture_output=True, text=True, timeout=300, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 1
    assert "discovery_lock" in proc.stderr and "worker_entrypoint" in proc.stderr

    ok = subprocess.run(
        [sys.executable, "-m", "gen_worker.build_guarantees", str(MICRO)],
        capture_output=True, text=True, timeout=300, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert ok.returncode == 0, ok.stderr


def test_the_checker_says_when_it_checked_NOTHING(tmp_path: Path) -> None:
    """A pass because there was no Dockerfile must not read like a pass.

    A green line that means "I looked at nothing" is how a gate becomes
    decorative, and a repo wiring this into CI over the wrong directory would
    get exactly that.
    """
    empty = tmp_path / "nodockerfile"
    empty.mkdir()
    proc = subprocess.run(
        [sys.executable, "-m", "gen_worker.build_guarantees", str(empty)],
        capture_output=True, text=True, timeout=300, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0
    assert "no Dockerfile" in proc.stderr

    missing = subprocess.run(
        [sys.executable, "-m", "gen_worker.build_guarantees",
         str(tmp_path / "nope")],
        capture_output=True, text=True, timeout=300, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert missing.returncode == 2, "a mistyped path must not pass"


def test_a_dockerfile_path_checks_its_own_directory() -> None:
    """`... build_guarantees examples/x/Dockerfile` is the obvious mistake."""
    assert bg.check_endpoint(MICRO / "Dockerfile") == []


def test_the_registry_is_printable_for_an_author() -> None:
    text = bg.describe_registry()
    for row in bg.REGISTRY:
        assert row.id in text and row.synthesized in text
