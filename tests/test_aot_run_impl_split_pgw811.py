"""pgw#811: the run_impl K-way split, and the gate that lets it ship.

The transform is a source rewrite of the ONE function that is 68% of an AOTI
host compile, so the whole question is "can it be wrong without anyone
noticing". These tests answer that in three layers:

* the mechanical inverse reproduces the input byte for byte (and RED-verifies
  — every mutation of a part is caught);
* the recognised shape is narrow and everything else DECLINES;
* and a shape-faithful synthetic wrapper is compiled BOTH ways, linked, and
  RUN, with byte-identical output required.

The synthetic wrapper carries every anchor inductor emits and every
declaration form the compute region uses — RAII handles, moved handles,
re-derived constant bindings, const arrays, threaded scalars — but is
self-contained, so it builds and runs without torch or a GPU.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import pytest

from gen_worker import aot_run_impl_split as v2

CXX = shutil.which("g++")

# ---------------------------------------------------------------------------
# A shape-faithful, self-contained wrapper
# ---------------------------------------------------------------------------

_RUNTIME = '''\
#define CUDA_DRIVER_CHECK(EXPR) do { (void)(EXPR); } while (0)

#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using AtenTensorHandle = int64_t*;
using DeviceStreamType = void*;

struct ConstantHandle {
  int64_t v{};
  explicit ConstantHandle(int64_t x = 0) : v(x) {}
};

namespace torch { namespace aot_inductor {

struct AOTInductorModelKernelsBase {
  int64_t hits{0};
  virtual ~AOTInductorModelKernelsBase() = default;
};

// Move-only owner, exactly the RAIIAtenTensorHandle contract the wrapper
// relies on: moving leaves the source null, reset() releases early.
struct RAIIAtenTensorHandle {
  AtenTensorHandle h{nullptr};
  RAIIAtenTensorHandle() = default;
  explicit RAIIAtenTensorHandle(AtenTensorHandle p) : h(p) {}
  RAIIAtenTensorHandle(RAIIAtenTensorHandle&& o) noexcept : h(o.h) { o.h = nullptr; }
  RAIIAtenTensorHandle& operator=(RAIIAtenTensorHandle&& o) noexcept {
    if (this != &o) { delete h; h = o.h; o.h = nullptr; }
    return *this;
  }
  RAIIAtenTensorHandle(const RAIIAtenTensorHandle&) = delete;
  RAIIAtenTensorHandle& operator=(const RAIIAtenTensorHandle&) = delete;
  ~RAIIAtenTensorHandle() { delete h; }
  void reset() { delete h; h = nullptr; }
  AtenTensorHandle release() { AtenTensorHandle p = h; h = nullptr; return p; }
  operator AtenTensorHandle() const { return h; }
};

inline std::vector<RAIIAtenTensorHandle> steal_from_raw_handles_to_raii_handles(
    AtenTensorHandle* handles, int n) {
  std::vector<RAIIAtenTensorHandle> out;
  for (int i = 0; i < n; ++i) out.emplace_back(handles[i]);
  return out;
}

inline AtenTensorHandle reinterpret_tensor_wrapper(AtenTensorHandle self,
                                                   int64_t bump) {
  return new int64_t(*self + bump);
}

inline RAIIAtenTensorHandle wrap_with_raii_handle_if_needed(AtenTensorHandle h) {
  return RAIIAtenTensorHandle(h);
}

struct AOTICudaStreamGuard {
  AOTICudaStreamGuard(DeviceStreamType, int32_t) {}
};

struct AOTInductorModel {
  std::shared_ptr<std::vector<ConstantHandle>> constants_;
  std::unique_ptr<AOTInductorModelKernelsBase> kernels_;
  int32_t device_idx_{};
  const std::optional<std::string> cubin_dir_;
  AOTInductorModel(std::shared_ptr<std::vector<ConstantHandle>> c,
                   std::unique_ptr<AOTInductorModelKernelsBase> k)
      : constants_(std::move(c)), kernels_(std::move(k)), device_idx_(7),
        cubin_dir_(std::nullopt) {}
  void run_impl(AtenTensorHandle* input_handles,
                AtenTensorHandle* output_handles, DeviceStreamType stream);
};

}}  // namespace torch::aot_inductor
'''

_KERNELS_CLASS = '''\
namespace torch::aot_inductor {
namespace {
class AOTInductorModelKernels : public AOTInductorModelKernelsBase {
  public:
    int64_t salt{11};
};
}  // namespace
}  // namespace torch::aot_inductor
'''

_TEMPLATES = '''\
using namespace torch::aot_inductor;

template <typename kernels_type_>
static __attribute__((noinline)) void call_triton_mix(
    const RAIIAtenTensorHandle& in, RAIIAtenTensorHandle& out, int64_t xnumel,
    const int64_t* shape, int32_t device_idx_, DeviceStreamType stream,
    kernels_type_& kernels, const std::optional<std::string>& cubin_dir_) {
  (void)stream; (void)cubin_dir_;
  kernels.hits += 1;
  *static_cast<AtenTensorHandle>(out) =
      (*static_cast<AtenTensorHandle>(in) * 31 + xnumel + shape[0] * 7
       + device_idx_ + kernels.salt) % 1000003;
}
'''

_CHECK_ENV = '''\
namespace torch::aot_inductor {
static bool _check_aoti_runtime_check_inputs_env() {
    return false;
}
'''


def _compute(blocks: int) -> List[str]:
    """A compute region in the emitted shape: allocate / call / reuse /
    re-derive, with every value feeding the next so that dropping,
    reordering or mis-threading ANY statement changes the output."""
    out: List[str] = []
    prev = "buf_in"
    for i in range(blocks):
        c = i % 4
        out.append(f"    const int64_t int_array_{i}[] = {{{3 + c}L, s70, s77}};")
        out.append(f"    [[maybe_unused]] auto& weight_{i} = constants_->at({c});")
        out.append(f"    AtenTensorHandle buf{i}_handle;")
        out.append(
            f"    buf{i}_handle = new int64_t(weight_{i}.v + base_w.v"
            f" - base_b.v + {i}L);")
        out.append(f"    RAIIAtenTensorHandle buf{i}(buf{i}_handle);")
        out.append(f"    int64_t call_mix_{i}_xnumel = {i}L*s70 + s77;")
        out.append(
            f"    call_triton_mix({prev}, buf{i}, call_mix_{i}_xnumel, "
            f"int_array_{i}, this->device_idx_, stream, kernels, "
            f"this->cubin_dir_);")
        out.append(f"    auto buf{i}_r = std::move(buf{i});  // reuse")
        out.append(
            f"    auto buf{i}_v = wrap_with_raii_handle_if_needed("
            f"reinterpret_tensor_wrapper(buf{i}_r, {c}L));")
        out.append(f"    auto scale_{i} = {i}L*s70*s77 + 3L;")
        out.append(
            f"    call_triton_mix(buf{i}_v, buf{i}_r, scale_{i}, int_array_{i}, "
            f"this->device_idx_, stream, kernels, this->cubin_dir_);")
        if i:
            out.append(f"    buf{i-1}_v.reset();")
        prev = f"buf{i}_r"
    out.append(f"    output_handles[0] = {prev}.release();")
    return out


def wrapper(blocks: int = 40) -> str:
    """A synthetic wrapper TU carrying every anchor and declaration form."""
    return "\n".join(
        [
            "// AOT ID: ['0_inference']",
            "#include <torch/csrc/inductor/aoti_include/cuda.h>",
            "",
            "// this region is NOT copied into the parts, exactly as the real",
            "// wrapper's extern \"C\" interface is not",
            "static int unused_interface_glue() { return 0; }",
            "",
            _RUNTIME,
            _KERNELS_CLASS,
            _TEMPLATES,
            _CHECK_ENV,
            "void AOTInductorModel::run_impl(",
            "    AtenTensorHandle* input_handles,",
            "    AtenTensorHandle* output_handles,",
            "    DeviceStreamType stream",
            ") {",
            "    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 1);",
            "    auto buf_in = std::move(inputs[0]);",
            "    int64_t s70 = *static_cast<AtenTensorHandle>(buf_in) % 5 + 2;",
            "    int64_t s77 = s70 + 3;",
            "    [[maybe_unused]] auto& base_w = constants_->at(0);",
            "    [[maybe_unused]] auto& base_b = constants_->at(1);",
            "    [[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>"
            "(*this->kernels_.get());",
            "    AOTICudaStreamGuard stream_guard(stream, this->device_idx_);",
        ]
        + _compute(blocks)
        + [
            "} // AOTInductorModel::run_impl",
            "}  // namespace torch::aot_inductor",
            "",
            _MAIN,
        ]
    )


_MAIN = '''\
int main() {
  auto constants = std::make_shared<std::vector<ConstantHandle>>();
  for (int i = 0; i < 4; ++i) constants->emplace_back(100 + i * 13);
  auto kernels = std::make_unique<torch::aot_inductor::AOTInductorModelKernels>();
  auto* raw = kernels.get();
  torch::aot_inductor::AOTInductorModel model(constants, std::move(kernels));
  AtenTensorHandle in[1] = {new int64_t(4242)};
  AtenTensorHandle out[1] = {nullptr};
  model.run_impl(in, out, nullptr);
  printf("out=%lld hits=%lld\\n", (long long)*out[0], (long long)raw->hits);
  delete out[0];
  return 0;
}
'''


@pytest.fixture(scope="module")
def source() -> str:
    return wrapper()


@pytest.fixture(autouse=True)
def _small_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real bar is 4,000 compute lines; the synthetic is smaller on
    purpose, so the threshold — not the machinery — is what is relaxed."""
    monkeypatch.setattr(v2, "MIN_COMPUTE_LINES", 100)


# ---------------------------------------------------------------------------
# It applies to the shape it exists for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("parts", [2, 4, 8])
def test_applies_and_reconstructs_exactly(source: str, parts: int) -> None:
    split = v2.split_run_impl(source, parts=parts)
    assert split.applied, split.reason
    assert len(split.parts) == parts
    assert v2.reconstruct(split.main, split.parts) == source


def test_every_statement_survives_verbatim(source: str) -> None:
    """The parts' concatenated statements ARE the original's, in order —
    the only additions being function boundaries and argument plumbing."""
    split = v2.split_run_impl(source, parts=4)
    assert split.applied
    body: List[str] = []
    for k, part in enumerate(split.parts):
        lines = part.split("\n")
        i = lines.index(v2.CHUNK_BEGIN + str(k))
        j = lines.index(v2.CHUNK_END + str(k))
        body += [ln for ln in lines[i + 1:j]
                 if not (ln.endswith(v2.MARK) or ln.endswith(v2.MARK_RD))]
    original = source.split("\n")
    start = original.index("    AOTICudaStreamGuard stream_guard(stream, this->device_idx_);")
    end = original.index("} // AOTInductorModel::run_impl")
    assert body == original[start + 1:end]


def test_threading_is_real_not_defaulted(source: str) -> None:
    """The prototype this replaces re-declared live-ins as fresh defaults.
    Every crossing name must arrive as a REFERENCE parameter instead."""
    split = v2.split_run_impl(source, parts=4)
    assert split.applied and split.threaded_max > 0
    for part in split.parts[1:]:
        signature = next(ln for ln in part.split("\n") if "_pgw811_part_" in ln
                         and ln.startswith("void "))
        assert "RAIIAtenTensorHandle&" in signature
        assert "int64_t&" in signature


def test_kernels_class_leaves_the_anonymous_namespace(source: str) -> None:
    """An anonymous-namespace type is a DIFFERENT type in every TU; a part
    casting the kernels object to its own copy would be UB — and it is the
    trap that silently voided the research lane's first measurement."""
    split = v2.split_run_impl(source, parts=2)
    assert split.applied
    for part in split.parts:
        assert "class AOTInductorModelKernels" in part
        assert "namespace {" not in part


# ---------------------------------------------------------------------------
# RED: the gate catches every corruption of a part
# ---------------------------------------------------------------------------

def _mutations(part: str) -> List[Tuple[str, str]]:
    lines = part.split("\n")
    call = next(i for i, ln in enumerate(lines)
                if ln.strip().startswith("call_triton_mix(")
                and not ln.endswith(v2.MARK))
    reuse = next(i for i, ln in enumerate(lines) if "// reuse" in ln)
    rederived = next(i for i, ln in enumerate(lines)
                     if ln.endswith(v2.MARK_RD) and "constants_->at(" in ln)
    out = []
    drop = list(lines); del drop[call]
    out.append(("a statement deleted", "\n".join(drop)))
    swap = list(lines); swap[call], swap[reuse] = swap[reuse], swap[call]
    out.append(("two statements reordered", "\n".join(swap)))
    edit = list(lines)
    edit[call] = edit[call].replace("this->device_idx_", "0")
    out.append(("an argument changed", "\n".join(edit)))
    add = list(lines); add.insert(call, "    int64_t smuggled = 1;")
    out.append(("a statement inserted", "\n".join(add)))
    hide = list(lines); hide[call] = hide[call] + v2.MARK
    out.append(("a statement disguised as generated", "\n".join(hide)))
    fake = list(lines); fake[call] = fake[call] + v2.MARK_RD
    out.append(("a statement disguised as a re-derivation", "\n".join(fake)))
    red = list(lines)
    red[rederived] = red[rederived].replace("constants_->at(", "constants_->at(9 - ")
    out.append(("a re-derived binding altered", "\n".join(red)))
    return out


def test_reconstruction_catches_every_mutation(source: str) -> None:
    split = v2.split_run_impl(source, parts=4)
    assert split.applied
    baseline = v2.reconstruct(split.main, split.parts)
    assert baseline == source

    caught = 0
    for what, mutated in _mutations(split.parts[1]):
        parts = list(split.parts)
        parts[1] = mutated
        try:
            assert v2.reconstruct(split.main, parts) != source, (
                f"the gate did NOT catch: {what}")
        except v2.Declined:
            pass                       # refusing outright is also catching it
        caught += 1
    assert caught == 7


def test_reconstruction_catches_a_mutated_main(source: str) -> None:
    split = v2.split_run_impl(source, parts=2)
    assert split.applied
    broken = split.main.replace(
        "    int64_t s77 = s70 + 3;", "    int64_t s77 = s70 + 4;")
    assert broken != split.main
    assert v2.reconstruct(broken, split.parts) != source


def test_a_faulty_emitter_declines(source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate is not decoration: break the emitter and the transform
    refuses to apply rather than shipping the damage."""
    real = v2._chunk

    def sabotage(lines, k, body, plan):  # type: ignore[no-untyped-def]
        out = real(lines, k, body, plan)
        return [ln for ln in out if "// reuse" not in ln]

    monkeypatch.setattr(v2, "_chunk", sabotage)
    split = v2.split_run_impl(source, parts=4)
    assert not split.applied
    assert "byte for byte" in split.reason


# ---------------------------------------------------------------------------
# Everything outside the recognised shape declines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda s: s.replace("#include <torch/csrc/inductor/aoti_include/cuda.h>", ""),
         "aoti_include"),
        (lambda s: s.replace("} // AOTInductorModel::run_impl", "}"),
         "end of run_impl"),
        (lambda s: s.replace("    buf0_v.reset();",
                             "    for (int i = 0; i < 2; ++i) { buf0_v.reset(); }"),
         "control flow"),
        (lambda s: s.replace("    buf0_v.reset();", "    return;"),
         "control flow"),
        (lambda s: s.replace("    int64_t call_mix_5_xnumel = 5L*s70 + s77;",
                             "    int64_t call_mix_5_xnumel = (5L*s70 + s77;"),
         "self-contained statement"),
        # s77 crosses every boundary, so it MUST be typed; an untypable
        # declaration of a crossing name declines (one that never crosses is
        # never asked, which is the point of resolving on demand).
        (lambda s: s.replace("    int64_t s77 = s70 + 3;",
                             "    auto s77 = make_something_opaque();"),
         "never declared"),
        (lambda s: s.replace("    int64_t s77 = s70 + 3;",
                             "    int64_t s77 = s70 + 3;\n    int64_t s70 = 1;"),
         "declared more than once"),
    ],
)
def test_declines_unrecognised_shapes(source: str, mutate, expect: str) -> None:  # type: ignore[no-untyped-def]
    split = v2.split_run_impl(mutate(source), parts=4)
    assert not split.applied
    assert expect in split.reason, split.reason


def test_declines_a_function_that_is_not_pathological(source: str) -> None:
    split = v2.split_run_impl(wrapper(blocks=3), parts=4)
    assert not split.applied
    assert "not the pathological shape" in split.reason


def test_declines_when_asked_not_to_split(source: str) -> None:
    assert not v2.split_run_impl(source, parts=1).applied


# ---------------------------------------------------------------------------
# BEHAVIOUR: compiled both ways, linked, run, byte-identical output
# ---------------------------------------------------------------------------

def _stub_include_root(root: Path) -> Path:
    header = root / "torch" / "csrc" / "inductor" / "aoti_include"
    header.mkdir(parents=True)
    (header / "cuda.h").write_text("#pragma once\n")
    return root


def _build(tmp: Path, name: str, sources: List[str], include: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    objects: List[str] = []
    for i, text in enumerate(sources):
        cpp = tmp / f"{name}_{i}.cpp"
        cpp.write_text(text)
        obj = tmp / f"{name}_{i}.o"
        subprocess.run(
            [CXX, str(cpp), "-std=c++20", "-O1", "-fPIC", f"-I{include}",
             "-c", "-o", str(obj)],
            check=True, capture_output=True, text=True)
        objects.append(str(obj))
    if len(objects) > 1:
        merged = tmp / f"{name}.o"
        subprocess.run([CXX, "-r", "-nostdlib", "-o", str(merged), *objects],
                       check=True, capture_output=True, text=True)
        objects = [str(merged)]
    exe = tmp / name
    subprocess.run([CXX, *objects, "-o", str(exe)],
                   check=True, capture_output=True, text=True)
    return exe


@pytest.mark.skipif(CXX is None, reason="no g++ on this host")
def test_split_is_behaviourally_identical(tmp_path: Path, source: str) -> None:
    """The whole point. Same program, split across 4 TUs and partial-linked
    exactly as the mint path does it, must produce the same bytes."""
    include = _stub_include_root(tmp_path / "inc")
    split = v2.split_run_impl(source, parts=4)
    assert split.applied, split.reason

    whole = _build(tmp_path, "whole", [source], include)
    parted = _build(tmp_path, "parted", [split.main, *split.parts], include)

    a = subprocess.run([str(whole)], capture_output=True, text=True, check=True)
    b = subprocess.run([str(parted)], capture_output=True, text=True, check=True)
    assert a.stdout == b.stdout, f"monolith {a.stdout!r} != split {b.stdout!r}"
    assert a.stdout.startswith("out=") and "hits=" in a.stdout


@pytest.mark.skipif(CXX is None, reason="no g++ on this host")
def test_a_corrupted_part_changes_the_ANSWER(tmp_path: Path, source: str) -> None:
    """Red control for the behavioural test: prove the program is sensitive
    to exactly the corruptions the gate rejects, so a passing behavioural
    test is evidence and not a tautology."""
    include = _stub_include_root(tmp_path / "inc")
    split = v2.split_run_impl(source, parts=4)
    assert split.applied
    whole = _build(tmp_path, "whole", [source], include)
    expected = subprocess.run([str(whole)], capture_output=True, text=True,
                              check=True).stdout

    for label, mutated in _mutations(split.parts[1])[:3]:
        parts = list(split.parts)
        parts[1] = mutated
        try:
            exe = _build(tmp_path / label.replace(" ", "_"), "bad",
                         [split.main, *parts], include)
        except subprocess.CalledProcessError:
            continue           # a corruption the COMPILER catches is also fine
        got = subprocess.run([str(exe)], capture_output=True, text=True).stdout
        assert got != expected, f"{label} did not change the answer"


# ---------------------------------------------------------------------------
# The real banked artifact, when this checkout has it
# ---------------------------------------------------------------------------

_BANKED = Path.home() / "cozy/cozy-creator-tracker/scripts/pgw793/results-1785426606"


@pytest.mark.skipif(not _BANKED.is_dir(), reason="banked pgw#793 wrapper not present")
@pytest.mark.parametrize("parts", [4, 8, 16])
def test_the_real_sdxl_wrapper(parts: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v2, "MIN_COMPUTE_LINES", v2.MIN_COMPUTE_LINES)
    banked = next(_BANKED.glob("*.wrapper.cpp"))
    text = banked.read_text()
    split = v2.split_run_impl(text, parts=parts)
    assert split.applied, split.reason
    assert v2.reconstruct(split.main, split.parts) == text
    assert split.compute_lines > 19_000
    assert split.declarations > 10_000


# ---------------------------------------------------------------------------
# Identity: the split must re-key NOTHING
# ---------------------------------------------------------------------------

def test_the_split_is_outside_the_static_code_closure() -> None:
    """The cell key is traced-graph x sm x toolchain x env_seal, plus the
    static code closure. Neither this module nor its driver is reachable
    from the closure's entrypoints, so no cell is re-keyed by their
    existence."""
    from gen_worker import compile_cache

    closure = dict(compile_cache.static_code_closure())
    assert "gen_worker/aot_run_impl_split.py" not in closure
    assert "gen_worker/aot_wrapper_split.py" not in closure


def test_all_three_seal_digests_are_unmoved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Installing the hook moves neither the inductor-config digest, nor the
    env seal, nor the toolchain digest. A silent re-key is the one failure
    mode this transform must not have — and the split adds a compiler
    INVOCATION, not a compiler or a flag, which is why it cannot."""
    pytest.importorskip("torch")
    from gen_worker import aot_wrapper_split as ws
    from gen_worker import cell_key, compile_cache, env_seal

    cpp_builder = pytest.importorskip("torch._inductor.cpp_builder")

    def digests() -> Tuple[str, str, str]:
        seal = env_seal.effective_seal()
        return (
            env_seal.inductor_config_digest(),
            # loaded_libs is a live machine fact no install() can move.
            cell_key.facts_digest(
                {k: v for k, v in seal.items() if k != "loaded_libs"}),
            cell_key.facts_digest(dict(compile_cache.toolchain_digest())),
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


# ---------------------------------------------------------------------------
# The mint path: K+1 compiles, one partial link, and a fail-closed fallback
# ---------------------------------------------------------------------------

def _drive(tmp_path: Path, source: str, monkeypatch: pytest.MonkeyPatch,
           fail_on: str = "") -> Tuple[List[str], object]:
    from gen_worker import aot_wrapper_split as ws

    cpp = tmp_path / "abc.wrapper.cpp"
    cpp.write_text(source)
    cmd = f"g++ {cpp} -c -o {tmp_path}/abc.wrapper.o"
    seen: List[str] = []

    def run(cmd_line: str, cwd: str) -> None:
        seen.append(cmd_line)
        if fail_on and fail_on in cmd_line:
            raise RuntimeError("synthetic compile failure")

    monkeypatch.setattr(v2, "MIN_COMPUTE_LINES", 100)
    monkeypatch.setenv(ws.JOBS_ENV, "1")
    return seen, ws.split_and_compile(cmd, str(tmp_path), run)


def test_the_mint_path_builds_k_plus_one_tus_and_partial_links(
    tmp_path: Path, source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen, outcome = _drive(tmp_path, source, monkeypatch)
    assert outcome is not None and outcome.applied  # type: ignore[attr-defined]
    assert len(seen) == v2.DEFAULT_PARTS + 2       # main + K parts + the link
    assert seen[0].count("pgw811.main.cpp") == 1
    assert seen[-1].startswith("g++ -r -nostdlib -o")
    assert seen[-1].endswith(".pgw811.part7.o")
    # The link target is the object torch asked for, untouched.
    assert f"-o {tmp_path}/abc.wrapper.o " in seen[-1] + " "


def test_a_part_that_will_not_build_falls_back_to_the_whole_wrapper(
    tmp_path: Path, source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail CLOSED: a slow mint beats a wrong artifact. The raise reaches
    the installed hook, which recompiles inductor's own command."""
    with pytest.raises(RuntimeError):
        _drive(tmp_path, source, monkeypatch, fail_on="part3")


def test_the_split_declines_quietly_on_an_unrecognised_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen, outcome = _drive(tmp_path, "int main() { return 0; }\n", monkeypatch)
    assert outcome is None
    assert seen == []          # nothing compiled; the caller runs the original


def test_the_job_count_is_a_budget_not_a_guess(monkeypatch: pytest.MonkeyPatch) -> None:
    from gen_worker import aot_wrapper_split as ws

    monkeypatch.setenv(ws.JOBS_ENV, "3")
    assert ws.host_compile_jobs(9) == 3
    assert ws.host_compile_jobs(2) == 2          # never more TUs than exist
    monkeypatch.setenv(ws.JOBS_ENV, "0")
    assert ws.host_compile_jobs(9) == 1          # never zero
    monkeypatch.setenv(ws.JOBS_ENV, "not-a-number")
    assert ws.host_compile_jobs(9) >= 1          # ignored, derived instead
    monkeypatch.delenv(ws.JOBS_ENV)
    assert 1 <= ws.host_compile_jobs(9) <= 9


def test_v2_has_its_own_kill_switch(tmp_path: Path, source: str,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """v1 has travelled much further than v2. Killing v2 must not kill it."""
    from gen_worker import aot_wrapper_split as ws

    cpp_builder = pytest.importorskip("torch._inductor.cpp_builder")
    cpp = tmp_path / "abc.wrapper.cpp"
    cpp.write_text(source)
    cmd = f"g++ {cpp} -c -o {tmp_path}/abc.wrapper.o"
    seen: List[str] = []

    monkeypatch.setenv(ws.DISABLE_V2_ENV, "1")
    monkeypatch.setattr(v2, "MIN_COMPUTE_LINES", 100)
    monkeypatch.setattr(ws, "_installed", False)
    monkeypatch.setattr(cpp_builder, "run_compile_cmd",
                        lambda c, cwd: seen.append(c))
    try:
        assert ws.install() is True
        cpp_builder.run_compile_cmd(cmd, str(tmp_path))
    finally:
        ws._installed = False
    assert len(seen) == 1 and "pgw811" not in seen[0]


# ---------------------------------------------------------------------------
# pgw#754's march clamp, asserted where it was never asserted: the ARGV
# ---------------------------------------------------------------------------

def test_the_march_clamp_is_checked_on_the_built_command_line() -> None:
    from gen_worker import host_isa

    if host_isa.mint_march() is None:
        pytest.skip("no clamp claimed on this machine")
    host_isa.assert_command_is_clamped(
        ["g++", "-march=x86-64-v3", "-c", "w.cpp", "-o", "w.o"])
    with pytest.raises(host_isa.HostIsaError, match="march=native"):
        host_isa.assert_command_is_clamped(
            ["g++", "-march=native", "-c", "w.cpp", "-o", "w.o"])


def test_an_unclamped_host_compile_is_refused_at_the_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unclamped object is not slower, it is UNPORTABLE — pgw#754 is a
    SIGILL-class defect — so this raises rather than degrading."""
    from gen_worker import aot_wrapper_split as ws
    from gen_worker import host_isa

    if host_isa.mint_march() is None:
        pytest.skip("no clamp claimed on this machine")
    cpp_builder = pytest.importorskip("torch._inductor.cpp_builder")
    monkeypatch.setattr(ws, "_installed", False)
    original = cpp_builder.run_compile_cmd
    monkeypatch.setattr(cpp_builder, "run_compile_cmd", lambda c, cwd: None)
    try:
        assert ws.install() is True
        with pytest.raises(host_isa.HostIsaError):
            cpp_builder.run_compile_cmd(
                f"g++ -march=native -c {tmp_path}/x.cpp -o {tmp_path}/x.o",
                str(tmp_path))
    finally:
        cpp_builder.run_compile_cmd = original
        ws._installed = False
