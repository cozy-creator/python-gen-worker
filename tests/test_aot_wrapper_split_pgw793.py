"""pgw#793: the AOTI wrapper constructor split — applied, declined, and
proved equivalent by RUNNING both translation units.

The transform's whole safety argument is that it is a pure statement-
preserving regrouping. These tests do not assert that from the source text
alone: :func:`test_split_is_behaviourally_identical` synthesises a wrapper
with inductor's exact constructor shape, compiles the original and the
transformed source with the real g++, RUNS both, and diffs the constants
table each produced. Everything else here is the fail-closed surface — every
wrapper shape the transform does not recognise must compile unmodified.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List

import pytest

from gen_worker import aot_wrapper_split as ws

CXX = shutil.which("g++")

#: A standalone C++ file carrying inductor's constructor shape: the same
#: out-of-class ``Cls::Cls(std::shared_ptr<ConstantMap> constants_map,``
#: signature the transform keys on, the same field types as
#: ``AOTInductorModelBase::ConstInfo``, and the same one-statement-per-line
#: assignment run. Self-contained, so it compiles and runs without torch.
_PROLOGUE = """\
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct ConstantMap {};
enum { cached_torch_device_type_cuda = 1, cached_torch_dtype_bfloat16 = 2,
       cached_torch_dtype_float32 = 3, cached_torch_layout_strided = 5 };
namespace torch { namespace aot_inductor {
enum class ConstantType { Unknown, Parameter, Buffer, TensorConstant };
}}

struct ConstInfo {
  const char* name = nullptr;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  int32_t dtype{};
  int32_t device_type{};
  int64_t offset{};
  size_t data_size{};
  int32_t layout{};
  const char* original_fqn = nullptr;
  bool from_folded{};
  int32_t type{};
};

struct AOTInductorModel {
  std::vector<ConstInfo> constants_info_;
  std::string in_spec_;
  AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                   std::shared_ptr<std::vector<int>> constants_array,
                   const std::string& device_str,
                   std::optional<std::string> cubin_dir);
};

namespace {
int unrelated_helper() { return 0; }
}  // namespace

"""

_EPILOGUE = """\

int main() {
  AOTInductorModel model(nullptr, nullptr, "cuda", std::nullopt);
  printf("%s\\n", model.in_spec_.c_str());
  for (const auto& c : model.constants_info_) {
    printf("%s|%s|%d|%d|%lld|%zu|%d|%d|%d", c.name, c.original_fqn, c.dtype,
           c.device_type, (long long)c.offset, c.data_size, c.layout,
           (int)c.from_folded, c.type);
    for (auto v : c.shape) printf("|s%lld", (long long)v);
    for (auto v : c.stride) printf("|t%lld", (long long)v);
    printf("\\n");
  }
  return unrelated_helper();
}
"""


def _statements(count: int) -> List[str]:
    """``count`` constants x the 11 fields inductor emits, one per line."""
    out: List[str] = []
    for idx in range(count):
        out.append(f'    constants_info_[{idx}].name = "block_{idx}_weight";')
        out.append(
            f"    constants_info_[{idx}].dtype = cached_torch_dtype_bfloat16;")
        out.append(
            f"    constants_info_[{idx}].device_type = "
            f"cached_torch_device_type_cuda;")
        out.append(f"    constants_info_[{idx}].offset = {idx * 8};")
        out.append(f"    constants_info_[{idx}].data_size = {23040 + idx};")
        out.append(
            f"    constants_info_[{idx}].from_folded = "
            f"{'true' if idx % 3 == 0 else 'false'};")
        out.append(
            f"    constants_info_[{idx}].type = static_cast<int32_t>("
            f"torch::aot_inductor::ConstantType::Parameter);")
        out.append(
            f"    constants_info_[{idx}].shape = {{{320 + idx}, 4, 3, 3}};")
        out.append(f"    constants_info_[{idx}].stride = {{36, 9, 3, 1}};")
        out.append(
            f"    constants_info_[{idx}].layout = static_cast<int32_t>("
            f"cached_torch_layout_strided);")
        out.append(
            f'    constants_info_[{idx}].original_fqn = "block.{idx}.weight";')
    return out


def _wrapper(constants: int) -> str:
    """A synthetic but shape-faithful wrapper translation unit."""
    body = _statements(constants)
    return "\n".join(
        [_PROLOGUE]
        + [
            "AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> "
            "constants_map,",
            "                                   std::shared_ptr<std::vector<int>> "
            "constants_array,",
            "                                   const std::string& device_str,",
            "                                   std::optional<std::string> "
            "cubin_dir)",
            f"    : constants_info_({constants}) {{",
        ]
        + body
        + [
            '    in_spec_ = R"([1, {"type": "builtins.tuple"}])";',
            "}",
            _EPILOGUE,
        ]
    )


# ---------------------------------------------------------------------------
# The transform applies to the shape it exists for
# ---------------------------------------------------------------------------


def test_applies_to_the_inductor_wrapper_shape() -> None:
    source = _wrapper(400)  # 4,400 statements — above MIN_STATEMENTS
    split, outcome = ws.split_constants_constructor(source)
    assert outcome.applied, outcome.reason
    assert outcome.statements == 4400
    assert outcome.chunks == 4400 // ws.CHUNK + (1 if 4400 % ws.CHUNK else 0)
    assert split != source
    # Every statement survives, in order, exactly once.
    original = [ln for ln in source.split("\n") if "constants_info_[" in ln]
    moved = [
        ln
        for ln in split.split("\n")
        if "constants_info_[" in ln and ws.PREFIX not in ln
    ]
    assert moved == original
    # The helpers are noinline (gcc folds called-once statics back otherwise)
    # and opt-none by default (they run once, at construction).
    assert split.count("__attribute__((noinline))") == outcome.chunks
    assert split.count('__attribute__((optimize("O0")))') == outcome.chunks


def test_everything_outside_the_statement_run_is_untouched() -> None:
    """The transform inserts helpers and rewrites one contiguous run. Both
    the text before the insertion point and the text after the run — which
    is where ``run_impl`` lives — must be byte-identical."""
    source = _wrapper(400)
    split, outcome = ws.split_constants_constructor(source)
    assert outcome.applied
    marker = "// pgw#793: the constants_info_ table"
    assert split[: split.index(marker)] == source[: source.index(
        "AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap>")]
    tail = "\n    in_spec_ = "
    assert split[split.index(tail):] == source[source.index(tail):]


def test_applies_under_a_renamed_model_class() -> None:
    """``aot_inductor.aoti_model_class_name`` renames the class, so the
    transform captures the name instead of assuming AOTInductorModel."""
    source = _wrapper(400).replace("AOTInductorModel", "SdxlDenoiserModel")
    split, outcome = ws.split_constants_constructor(source)
    assert outcome.applied, outcome.reason
    assert outcome.statements == 4400


def test_applies_with_an_extra_metadata_field() -> None:
    """``opaque_metadata`` is emitted only for some constants; an extra
    field in the run must not break contiguity."""
    lines = _wrapper(400).split("\n")
    at = lines.index('    constants_info_[7].original_fqn = "block.7.weight";')
    lines.insert(at + 1, "    constants_info_[7].opaque_metadata = {1, 2, 3};")
    split, outcome = ws.split_constants_constructor("\n".join(lines))
    assert outcome.applied, outcome.reason
    assert outcome.statements == 4401


def test_opt_none_can_be_disabled() -> None:
    split, outcome = ws.split_constants_constructor(
        _wrapper(400), opt_none=False)
    assert outcome.applied
    assert '__attribute__((optimize("O0")))' not in split


# ---------------------------------------------------------------------------
# Fail-closed: every unrecognised shape compiles unmodified
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, expect",
    [
        pytest.param(
            lambda s: s.replace(
                "AOTInductorModel::AOTInductorModel(std::shared_ptr"
                "<ConstantMap> constants_map,",
                "AOTInductorModel::AOTInductorModel(ConstantMap* constants_map,",
            ),
            "found 0",
            id="constructor-signature-changed",
        ),
        pytest.param(
            lambda s: s + "\n" + s,
            "found 2",
            id="two-constructors",
        ),
        pytest.param(
            lambda s: s.replace(
                '    constants_info_[200].name = "block_200_weight";',
                "    some_other_statement(200);",
            ),
            "not contiguous",
            id="interrupted-statement-run",
        ),
        pytest.param(
            lambda s: s.replace(
                '    constants_info_[200].shape = {520, 4, 3, 3};',
                '    constants_info_[200].shape = R"(weird)";',
            ),
            "not contiguous",
            id="raw-string-in-the-run",
        ),
        pytest.param(
            lambda s: s.replace(
                "    constants_info_[200].stride = {36, 9, 3, 1};",
                "    constants_info_[200].stride = {36, 9,",
            ),
            "not contiguous",
            id="statement-spanning-two-lines",
        ),
    ],
)
def test_declines_unrecognised_shapes(mutate, expect: str) -> None:  # type: ignore[no-untyped-def]
    source = mutate(_wrapper(400))
    split, outcome = ws.split_constants_constructor(source)
    assert not outcome.applied
    assert expect in outcome.reason
    assert split == source  # unmodified, byte for byte


def test_declines_a_small_wrapper() -> None:
    source = _wrapper(10)  # 110 statements
    split, outcome = ws.split_constants_constructor(source)
    assert not outcome.applied
    assert "not the pathological shape" in outcome.reason
    assert split == source


def test_declines_a_wrapper_with_no_constants() -> None:
    source = _PROLOGUE + _EPILOGUE
    split, outcome = ws.split_constants_constructor(source)
    assert not outcome.applied
    assert split == source


def test_self_check_rejects_a_broken_transform(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The gate is live: if re-inlining ever stops reproducing the input,
    the transform declines instead of shipping a mangled TU."""
    monkeypatch.setattr(ws, "_reinline", lambda text: text + "// drift\n")
    source = _wrapper(400)
    split, outcome = ws.split_constants_constructor(source)
    assert not outcome.applied
    assert "self-check failed" in outcome.reason
    assert split == source


# ---------------------------------------------------------------------------
# The command-line hook
# ---------------------------------------------------------------------------


def test_transform_command_rewrites_only_the_source_token(tmp_path: Path) -> None:
    source = tmp_path / "abc.wrapper.cpp"
    source.write_text(_wrapper(400))
    cmd = f"g++ {source} -O1 -std=c++20 -fPIC -c -o {tmp_path}/abc.wrapper.o"
    new_cmd, outcome = ws.transform_command(cmd)
    assert outcome is not None and outcome.applied
    before, after = shlex.split(cmd), shlex.split(new_cmd)
    assert len(before) == len(after)
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert differing == [1]  # only the source path moved
    target = Path(after[1])
    assert target.name == "abc.wrapper.pgw793.cpp"
    assert target.read_text() == ws.split_constants_constructor(
        source.read_text())[0]
    # The original is never mutated.
    assert source.read_text() == _wrapper(400)


def test_transform_command_leaves_non_compiles_alone(tmp_path: Path) -> None:
    source = tmp_path / "abc.wrapper.cpp"
    source.write_text(_wrapper(400))
    for cmd in (
        f"g++ -shared {tmp_path}/a.o {tmp_path}/b.o -o {tmp_path}/x.so",  # link
        f"g++ {source} {tmp_path}/other.cpp -c -o {tmp_path}/x.o",  # 2 sources
        f"g++ -x c++-header {tmp_path}/h.h -o {tmp_path}/h.gch",  # pch
    ):
        new_cmd, outcome = ws.transform_command(cmd)
        assert new_cmd == cmd
        assert outcome is None


def test_transform_command_declines_the_kernel_tu(tmp_path: Path) -> None:
    kernel = tmp_path / "abc.kernel.cpp"
    kernel.write_text("// Triton kernels are embedded as comments\n")
    cmd = f"g++ {kernel} -c -o {tmp_path}/abc.kernel.o"
    new_cmd, outcome = ws.transform_command(cmd)
    assert new_cmd == cmd
    assert outcome is not None and not outcome.applied


def test_transform_command_survives_an_unreadable_source(tmp_path: Path) -> None:
    cmd = f"g++ {tmp_path}/missing.cpp -c -o {tmp_path}/missing.o"
    new_cmd, outcome = ws.transform_command(cmd)
    assert new_cmd == cmd
    assert outcome is not None and not outcome.applied
    assert "unreadable source" in outcome.reason


# ---------------------------------------------------------------------------
# Identity: the transform must not re-key a single cell
# ---------------------------------------------------------------------------


def test_transform_is_outside_cell_identity() -> None:
    """The seal covers inductor config, torch flags and libraries; the key
    also covers the static code closure. This module changes none of them,
    and is not itself reachable from the closure's entrypoints."""
    from gen_worker import compile_cache

    closure = dict(compile_cache.static_code_closure())
    assert "gen_worker/aot_wrapper_split.py" not in closure
    assert "gen_worker/aot_mint.py" not in closure


def test_a_failing_transformed_tu_falls_back_to_the_original(
    tmp_path: Path, monkeypatch  # type: ignore[no-untyped-def]
) -> None:
    """If the transformed TU somehow does not build, the mint degrades to
    inductor's own source instead of failing."""
    cpp_builder = pytest.importorskip("torch._inductor.cpp_builder")
    source = tmp_path / "abc.wrapper.cpp"
    source.write_text(_wrapper(400))
    cmd = f"g++ {source} -c -o {tmp_path}/abc.wrapper.o"

    seen: List[str] = []

    def fake_run(cmd_line: str, cwd: str) -> None:
        seen.append(cmd_line)
        if "pgw793" in cmd_line:
            raise RuntimeError("synthetic compile failure")

    monkeypatch.setattr(cpp_builder, "run_compile_cmd", fake_run)
    monkeypatch.setattr(ws, "_installed", False)
    assert ws.install() is True
    try:
        cpp_builder.run_compile_cmd(cmd, str(tmp_path))
    finally:
        ws._installed = False
    assert len(seen) == 2
    assert "pgw793" in seen[0]
    assert seen[1] == cmd  # inductor's own command, byte for byte


def test_install_moves_no_seal_digest(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Installing the hook must leave every key axis where it was — a
    silent re-key is the one failure mode this transform must not have."""
    pytest.importorskip("torch")
    from gen_worker import cell_key, env_seal

    cpp_builder = pytest.importorskip("torch._inductor.cpp_builder")

    def digests() -> tuple[str, str]:
        seal = env_seal.effective_seal()
        return (
            env_seal.inductor_config_digest(),
            # loaded_libs is a live machine fact, not something install() can
            # move; everything else in the seal is compared exactly.
            cell_key.facts_digest(
                {k: v for k, v in seal.items() if k != "loaded_libs"}),
        )

    before = digests()
    monkeypatch.setattr(ws, "_installed", False)
    original = cpp_builder.run_compile_cmd
    try:
        assert ws.install() is True
        assert digests() == before
    finally:
        cpp_builder.run_compile_cmd = original
        ws._installed = False


def test_install_is_idempotent_and_killable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(ws, "_installed", False)
    monkeypatch.setenv(ws.DISABLE_ENV, "1")
    assert ws.install() is False
    monkeypatch.delenv(ws.DISABLE_ENV)
    torch_inductor = pytest.importorskip("torch._inductor.cpp_builder")
    original = torch_inductor.run_compile_cmd
    try:
        assert ws.install() is True
        assert torch_inductor.run_compile_cmd is not original
        assert ws.install() is False  # second call is a no-op
    finally:
        torch_inductor.run_compile_cmd = original
        ws._installed = False


# ---------------------------------------------------------------------------
# Equivalence, proved by running both translation units
# ---------------------------------------------------------------------------


@pytest.mark.skipif(CXX is None, reason="no g++ on this host")
def test_split_is_behaviourally_identical(tmp_path: Path) -> None:
    """Compile AND RUN the original and the transformed wrapper; the
    constants table each builds must be byte-identical."""
    source = _wrapper(400)
    split, outcome = ws.split_constants_constructor(source)
    assert outcome.applied

    outputs = []
    for name, text in (("base", source), ("split", split)):
        cpp = tmp_path / f"{name}.cpp"
        cpp.write_text(text)
        binary = tmp_path / name
        subprocess.run(
            ["nice", "-n", "10", str(CXX), "-std=c++20", "-O1", "-w",
             str(cpp), "-o", str(binary)],
            check=True, capture_output=True, timeout=600,
        )
        run = subprocess.run(
            [str(binary)], check=True, capture_output=True, timeout=120)
        outputs.append(run.stdout)

    assert outputs[0] == outputs[1]
    assert outputs[0].count(b"\n") == 401  # 400 constants + in_spec_


@pytest.mark.skipif(CXX is None, reason="no g++ on this host")
def test_opt_none_helpers_stay_position_independent(tmp_path: Path) -> None:
    """gcc's ``optimize`` attribute resets OPTIMIZATION options for the
    function it marks. AOTI's wrapper object is linked into a ``.so``, so
    ``-fPIC`` surviving the attribute is load-bearing: if it did not, the
    link would fail on an absolute relocation. Proved by linking."""
    split, outcome = ws.split_constants_constructor(_wrapper(400))
    assert outcome.applied
    cpp = tmp_path / "split.cpp"
    cpp.write_text(split.replace("int main() {", "int unused_main() {"))
    obj, lib = tmp_path / "split.o", tmp_path / "split.so"
    subprocess.run(
        ["nice", "-n", "10", str(CXX), "-std=c++20", "-O1", "-w", "-fPIC",
         "-c", str(cpp), "-o", str(obj)],
        check=True, capture_output=True, timeout=600,
    )
    subprocess.run(
        ["nice", "-n", "10", str(CXX), "-shared", str(obj), "-o", str(lib)],
        check=True, capture_output=True, timeout=600,
    )
    assert lib.stat().st_size > 0  # the table really was populated
