"""The CUSTOM-Dockerfile path VERIFIES what the synthesized path establishes.

pgw#1017's audit walked `tensorhub/internal/builder/generate_dockerfile.go` line
by line and asked, of each invariant it writes into the generated file, whether
anything asks it of a Dockerfile an author wrote. Four did not. These are the
rows that close three of them; GAP C lives in the hub, because
`system_dependencies` never reaches this process.

The lesson the audit recorded, and the reason these tests are shaped this way:
enumerate what the synthesized file DOES, not what it INSTALLS. Two of the
three gaps below are not packages — one is a composed directory, one is a shell
wrapper around an exit code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from gen_worker import aot_preconditions as ap
from gen_worker import cuda_root as cr
from gen_worker.discovery.discover import BUILD_INPUT_FAILURE_MARKER

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# GAP A — the composed CUDA root
# ---------------------------------------------------------------------------

def test_missing_parts_names_each_of_the_three_measured_facts(tmp_path):
    """Absent root, flattened crt/, absent nv/target — each named separately.

    pgw#823 measured three independent reasons `cpp_extension` fails on the
    pytorch runtime bases. A predicate that collapses them into "no CUDA" sends
    the author to fix the wrong one.
    """
    root = tmp_path / "cuda"
    assert "the root itself" in " ".join(cr.missing_parts(root))

    (root / "include").mkdir(parents=True)
    (root / "lib64").mkdir()
    (root / "include" / "cuda_runtime.h").write_text("")
    gaps = " ".join(cr.missing_parts(root))
    assert "crt/host_defines.h" in gaps and "flattens crt/ away" in gaps
    assert "nv/target" in gaps and "CCCL" in gaps

    (root / "include" / "crt").mkdir()
    (root / "include" / "crt" / "host_defines.h").write_text("")
    (root / "include" / "nv").mkdir()
    (root / "include" / "nv" / "target").write_text("")
    assert cr.missing_parts(root) == []


def test_compose_leaves_a_preexisting_root_alone(tmp_path):
    """A devel base or a distro CUDA is not this recipe's to second-guess."""
    root = tmp_path / "cuda"
    root.mkdir()
    (root / "sentinel").write_text("do not touch")
    assert cr.compose(root).root == "preexisting"
    assert (root / "sentinel").read_text() == "do not touch"


def test_compose_reports_and_never_raises_when_there_is_no_wheel(tmp_path,
                                                                monkeypatch):
    """A CPU image has nothing to compose, and that is not a failure.

    The refusal is `aot_preconditions`' job. Keeping the composer silent-and-
    successful is what lets the gate be the single place a build dies.
    """
    monkeypatch.setattr(cr, "wheel_cuda_root", lambda: "")
    out = cr.compose(tmp_path / "cuda")
    assert out.root == ""
    assert any("nothing to compose" in n for n in out.notes)


def test_the_cuda_root_module_is_runnable_as_the_doc_says(tmp_path):
    """`python -m gen_worker.cuda_root` — the exact line the docs teach."""
    proc = subprocess.run(
        [sys.executable, "-m", "gen_worker.cuda_root", "--root", str(tmp_path / "c")],
        capture_output=True, text=True, timeout=300,
        cwd=str(REPO), env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "cuda_root:" in proc.stderr


def _rows(monkeypatch, *, cuda_home, gaps, cuda_build=True):
    monkeypatch.setattr(ap, "_torch_is_cuda_build", lambda: cuda_build)
    monkeypatch.setattr(cr, "torch_cuda_home", lambda: cuda_home)
    monkeypatch.setattr(cr, "missing_parts", lambda _root: list(gaps))
    return ap._cuda_root_row(["micro-diffusion"])


def test_a_cuda_image_with_no_root_REFUSES_and_names_the_one_line_fix(monkeypatch):
    """RED for GAP A: the row that did not exist before pgw#1017.

    This is the whole acceptance criterion. Before this row, an image with g++
    and no CUDA root built clean, booted a pod, loaded, exported — and died at
    CUDA_HOME. A free build-time refusal became a paid pod-time failure.
    """
    row = _rows(monkeypatch, cuda_home="", gaps=[])
    assert row.check == ap.CHECK_CUDA_ROOT
    assert row.verdict == ap.REFUSED
    assert "python -m gen_worker.cuda_root" in row.detail
    assert "micro-diffusion" in row.detail


def test_a_half_composed_root_REFUSES_too(monkeypatch):
    """CUDA_HOME resolving is not the property — compiling is.

    `cpp_extension` finding a directory says nothing about `crt/host_defines.h`
    being in it, and that is exactly the second of pgw#823's three facts.
    """
    row = _rows(monkeypatch, cuda_home="/usr/local/cuda",
                gaps=["include/crt/host_defines.h (the cu13 wheel flattens crt/ away)"])
    assert row.verdict == ap.REFUSED
    assert "crt/host_defines.h" in row.detail


def test_a_composed_root_passes_and_a_cpu_build_needs_none(monkeypatch):
    assert _rows(monkeypatch, cuda_home="/usr/local/cuda", gaps=[]).verdict == ap.OK
    cpu = _rows(monkeypatch, cuda_home="", gaps=["x"], cuda_build=False)
    assert cpu.verdict == ap.OK and "CPU build" in cpu.detail


# ---------------------------------------------------------------------------
# GAP B — one torch-family install, not two
# ---------------------------------------------------------------------------

class _Dist:
    def __init__(self, name):
        self.metadata = {"Name": name}


def test_a_duplicate_torch_REFUSES_with_the_cost_named(monkeypatch):
    """RED for GAP B: ~7 GB per pod pull, and a desynced CUDA runtime."""
    monkeypatch.setattr(
        "importlib.metadata.distributions",
        lambda: [_Dist("torch"), _Dist("Torch"), _Dist("torchvision")])
    row = ap._torch_singleton_row()
    assert row.verdict == ap.REFUSED
    assert "torch" in row.detail and "7 GB" in row.detail


def test_one_of_each_passes_and_underscores_normalize(monkeypatch):
    monkeypatch.setattr(
        "importlib.metadata.distributions",
        lambda: [_Dist("torch"), _Dist("torchvision"), _Dist("not-torch")])
    assert ap._torch_singleton_row().verdict == ap.OK


def test_the_torch_family_is_the_set_the_hub_asserts():
    """The synthesized file's own list — one spelling, not a parallel one."""
    assert ap.TORCH_FAMILY == ("torch", "torchvision", "torchaudio", "triton")


# ---------------------------------------------------------------------------
# GAP D — a source refusal is NON-RETRYABLE on both paths
# ---------------------------------------------------------------------------

def _discovery(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "gen_worker.discovery"],
        capture_output=True, text=True, timeout=600, cwd=str(root),
        env={"PYTHONPATH": f"{REPO / 'src'}:{root}", "PATH": "/usr/bin:/bin"},
    )


def test_a_missing_config_carries_the_non_retryable_marker(tmp_path):
    """RED for GAP D, and the sting the audit found.

    Tensorhub turns this marker into `ErrBuildInput` → `river.JobCancel`. It
    used to be emitted by a shell wrapper in the SYNTHESIZED Dockerfile only,
    so an identical refusal on a custom Dockerfile was RETRIED — and that
    included pgw#996's precondition refusals, whose entire value is deciding
    once, at build. The whole image build re-ran to reach the same verdict.

    A tarball with no `pyproject.toml` is the plainest immutable-source
    failure there is: no retry will grow one.
    """
    proc = _discovery(tmp_path)
    assert proc.returncode != 0
    assert "pyproject.toml not found" in proc.stderr
    assert BUILD_INPUT_FAILURE_MARKER in proc.stderr, proc.stderr


def test_a_broken_endpoint_module_carries_it_too(tmp_path):
    """The other shape: the config is fine and the endpoint package raises.

    Both reach `main()` by different routes — one through `discover_manifest`'s
    except arm, one through the validation gate — and the marker has to ride
    every one of them, not the first that got tested.
    """
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "brokenep"\nversion = "0.1.0"\n'
        '[tool.gen_worker]\nmain = "brokenep"\n')
    pkg = tmp_path / "brokenep"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "raise RuntimeError('this endpoint module is broken')")
    proc = _discovery(tmp_path)
    assert proc.returncode != 0
    assert BUILD_INPUT_FAILURE_MARKER in proc.stderr, proc.stderr


def test_the_marker_is_the_string_the_hub_classifies_on():
    """Mirrored from `tensorhub/internal/builder/k8s_backend.go:96`.

    Named here so a change on that side has one place to land, rather than
    being discovered by a build that stopped being cancelled.
    """
    assert BUILD_INPUT_FAILURE_MARKER == "TENSORHUB_BUILD_INPUT_FAILURE:discovery"


@pytest.mark.parametrize("check", [
    ap.CHECK_CXX_TOOLCHAIN, ap.CHECK_CUDA_ROOT, ap.CHECK_TORCH_SINGLETON,
])
def test_every_static_row_has_a_stable_name(check):
    """The names ride out in `endpoint.lock` and into build errors."""
    assert check and check.islower() and " " not in check


# ---------------------------------------------------------------------------
# WIRING — a row nothing calls is not a gate
# ---------------------------------------------------------------------------

def test_the_new_rows_are_actually_STAMPED_by_the_gate(monkeypatch):
    """The row that closes a gap has to be reachable from the seam.

    Asserting `_cuda_root_row(...)` in isolation proves the function's opinion
    and nothing about the build: an unwired row is exactly the shape of the
    defect this issue exists to stop — a guarantee that only ever ran on one
    path. So this asks `static_mint_preconditions`, which is what discovery
    calls and `validate_endpoint_lock` reads.
    """
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
    checks = {r.check for r in rows}
    assert ap.CHECK_CUDA_ROOT in checks, "GAP A's row is not wired into the gate"
    assert ap.CHECK_TORCH_SINGLETON in checks, "GAP B's row is not wired in"
    assert ap.CHECK_CXX_TOOLCHAIN in checks

    cuda = next(r for r in rows if r.check == ap.CHECK_CUDA_ROOT)
    assert cuda.verdict == ap.REFUSED
    # And the refusal must survive the trip into endpoint.lock, since that is
    # where the hub reads it from.
    assert cuda.manifest_row()["verdict"] == "refused"


def test_a_family_that_declares_no_export_gets_no_new_rows(monkeypatch):
    """Intake-mode JIT owes the AOT lane nothing — including these two.

    The ruling this gate implements binds only endpoints that DECLARE an AOT
    export. A CPU or JIT-only image must not start failing builds because it
    has no CUDA root it never needed.
    """
    monkeypatch.setattr(ap, "has_export_declaration", lambda _f: False)
    monkeypatch.setattr(ap, "declaration_import_failures", lambda: ())
    assert ap.static_mint_preconditions({"plain": 0}, torch_available=True) == ()
