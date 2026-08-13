"""pgw#1097 — THE FOLDING FENCE.

One cell serves every fine-tune of a family because weights rebind BY NAME at
load. That is sound only while the compiled code holds no weight
VALUE. With ``constant_folding_fenced`` off — torch's default, and what
every real-weight mint ran under until this issue — ``GraphLowering.get_attr``
renders a constant's values straight into the kernel source when its SHAPE
meets either rule: 0-dim (via ``.item()``), or ``len(shape) == 1 and
shape[0] <= 8`` (``GraphLowering.can_inline_constant``). Such a weight then
appears in NO table anyone could rebind, and every other fine-tune of the
family adopts a cell carrying the minting checkpoint's tensor.

MEASURED on torch 2.13.0+cu130 (pgw#1097, CPU):

    module          weight              shape   default   fenced
    ------          ------              -----   -------   ------
    Tiny            small               (4,)    GONE      bindable
    Tiny            lin.bias            (8,)    GONE      bindable
    Tiny            scalar              ()      GONE      bindable
    ConvBias        conv_out.bias       (4,)    GONE      bindable
    Foldable        gn.weight/gn.bias   (8,)    GONE      bindable
    Foldable        logit_scale         ()      GONE      bindable
    MicroDecoder    norm.weight         (128,)  bindable  bindable

The last row is why the fleet never saw this: NO micro-family parameter is
0-dim or 1-D-with-<=8-elements, so the gauntlet's two-checkpoint sharing proof
(pgw#1073 scenario 6) passed on an architecture that cannot fold. sdxl can and
does — its one recorded eliminated constant is ``unet.conv_out.bias``, 4 floats.

These tests do not compile. They exercise the fence's own logic against
synthetic packages shaped exactly like AOTInductor's generated wrapper (the
pgw#793 fixture shape), plus the config and declared-axis halves. The
compile-side RED proof is a pod run — see pgw#1097's tracker section.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence, Tuple

import pytest

from gen_worker import aot_mint, aot_package, aot_serve

ENTRY = "denoise"


# ---------------------------------------------------------------------------
# A synthetic package, shaped like AOTInductor's own generated wrapper
# ---------------------------------------------------------------------------


def _constant_statements(rows: Sequence[Mapping[str, object]]) -> str:
    out = []
    for idx, row in enumerate(rows):
        fqn = str(row["fqn"])
        name = fqn.replace(".", "_")
        kind = str(row.get("kind", "Parameter"))
        shape = tuple(int(v) for v in row.get("shape", (16,)))  # type: ignore[arg-type]
        out.append(f'    constants_info_[{idx}].name = "{name}";')
        out.append(
            f"    constants_info_[{idx}].dtype = cached_torch_dtype_float32;")
        out.append(f"    constants_info_[{idx}].data_size = 64;")
        out.append(
            f"    constants_info_[{idx}].from_folded = "
            f"{'true' if kind == 'FoldedConstant' else 'false'};")
        out.append(
            f"    constants_info_[{idx}].type = static_cast<int32_t>("
            f"torch::aot_inductor::ConstantType::{kind});")
        out.append(
            f"    constants_info_[{idx}].shape = "
            f"{{{', '.join(str(v) for v in shape)}}};")
        out.append(f'    constants_info_[{idx}].original_fqn = "{fqn}";')
    return "\n".join(out)


def _wrapper_source(rows: Sequence[Mapping[str, object]]) -> str:
    return "\n".join([
        "// synthetic AOTInductor wrapper (pgw#1097 fixture)",
        "AOTInductorModel::AOTInductorModel(",
        "    std::shared_ptr<ConstantMap> constants_map,",
        "    std::shared_ptr<std::vector<int>> constants_array,",
        "    const std::string& device_str,",
        "    std::optional<std::string> cubin_dir)",
        "    : AOTInductorModelBase(1, 1, "
        f"{len(rows)}, device_str, std::move(cubin_dir), false) {{",
        _constant_statements(rows),
        "}",
    ])


def _package(tmp_path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    out = tmp_path / "cell.pt2"
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(
            f"data/aotinductor/{ENTRY}/c{ENTRY}.wrapper.cpp",
            _wrapper_source(rows))
    return out


def _program(*fqns: str) -> SimpleNamespace:
    """An ``ExportedProgram`` duck for :func:`program_constant_fqns`."""
    return SimpleNamespace(
        graph_signature=SimpleNamespace(
            parameters=tuple(fqns), buffers=(), lifted_tensor_constants=()),
        constants={})


# The shape the fence exists for: a graph lifting five weights, whose compiled
# artifact declares only three because two met an inline rule.
WEIGHTS: Tuple[str, ...] = (
    "proj.weight", "proj.bias", "gn.weight", "gn.bias", "logit_scale")
BOUND_ROWS = [{"fqn": n} for n in WEIGHTS]
FOLDED_ROWS = [{"fqn": n} for n in ("proj.weight", "proj.bias", "gn.weight")]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_folded_weight_is_named_and_refused(tmp_path: Path) -> None:
    """A weight the artifact will not let anyone bind is a REFUSAL, and the
    refusal names it. Nothing downstream can tell that the values in the
    kernel are one checkpoint's — only the mint can."""
    package = _package(tmp_path, FOLDED_ROWS)
    reasons = aot_package.folded_weights(
        _program(*WEIGHTS), package, WEIGHTS, ENTRY)
    assert reasons, "a folded weight must not pass the fence"
    text = reasons[0]
    assert "gn.bias" in text and "logit_scale" in text
    assert "pgw#857" in text  # the contract it enforces, named at the failure


def test_fenced_package_passes(tmp_path: Path) -> None:
    package = _package(tmp_path, BOUND_ROWS)
    assert aot_package.folded_weights(
        _program(*WEIGHTS), package, WEIGHTS, ENTRY) == []


def test_anonymous_literals_are_not_weights(tmp_path: Path) -> None:
    """An eliminated ``_tensor_constant0`` is a graph literal, not a rebindable
    weight — routine, and never this gate's business."""
    program = _program(*WEIGHTS, "_tensor_constant0")
    package = _package(tmp_path, BOUND_ROWS)
    assert aot_package.folded_weights(program, package, WEIGHTS, ENTRY) == []
    assert aot_package.eliminated_constants(program, package, ENTRY) == [
        "_tensor_constant0"]


def test_state_dict_entries_the_program_never_lifted_are_out_of_scope(
    tmp_path: Path,
) -> None:
    """``torch.export`` deduplicates TIED parameters at the program level, so
    ``b.weight`` is in the state_dict and in no graph. That is an export
    property, not an inductor fold, and this fence does not claim it."""
    package = _package(tmp_path, [{"fqn": "a.weight"}])
    assert aot_package.folded_weights(
        _program("a.weight"), package, ("a.weight", "b.weight"), ENTRY) == []


def test_no_state_dict_means_no_verdict(tmp_path: Path) -> None:
    """A caller with no resident module cannot distinguish a weight from a
    literal, so the gate abstains rather than guessing — the same discipline
    ``unbindable_constants`` uses."""
    package = _package(tmp_path, FOLDED_ROWS)
    assert aot_package.folded_weights(
        _program(*WEIGHTS), package, (), ENTRY) == []


# ---------------------------------------------------------------------------
# The setting the gate proves
# ---------------------------------------------------------------------------


FENCE_FLAG = "aot_inductor.use_runtime_constant_folding"


def test_every_mint_compiles_with_the_fence_on() -> None:
    configs = aot_mint._entry_configs(None)
    assert configs[FENCE_FLAG] is True
    assert configs["aot_inductor.package_constants_in_so"] is False


def test_weightless_and_real_weight_mints_share_one_config() -> None:
    """pgw#1080 needed this flag because a weightless mint's values are FAKE;
    pgw#1097 needs it because a real mint's values are one CHECKPOINT'S. Two
    motives, one config — so `weightless` no longer selects anything."""
    assert (aot_mint._entry_configs(None, weightless=True)
            == aot_mint._entry_configs(None, weightless=False))


def test_a_caller_cannot_turn_the_fence_off() -> None:
    configs = aot_mint._entry_configs({FENCE_FLAG: False})
    assert configs[FENCE_FLAG] is True


def test_the_fence_does_not_use_always_keep_tensor_constants() -> None:
    """The other flag restores bindability too and is NOT what ships: it also
    retains anonymous graph literals as ORDINARY constants, and a literal the
    recorded program never lifted is exactly the `program_package_drift`
    refusal. Measured red in CI on `WarmSensitive` (a plain-attribute table
    built inside `forward`). The runtime split's outputs are `FoldedConstant`
    rows, which that gate already exempts."""
    assert "always_keep_tensor_constants" not in aot_mint._entry_configs(None)


# ---------------------------------------------------------------------------
# Cells minted BEFORE the fence
# ---------------------------------------------------------------------------


def _meta(**over: object) -> dict:
    meta = aot_serve.entry_metadata(
        family="micro-diffusion", precision="bf16", cell_key="ck1-test",
        name=ENTRY, entry={
            "target": "transformer", "fork": [], "class_dims": [],
            "inputs": [{"name": "latent", "position": 0, "dtype": "float32",
                        "shape": [1, 16, "s0"], "path": []}],
            "symbols": {"s0": [8, 64]},
            "constants": [{"fqn": n, "source": aot_serve.SOURCE_STATE_DICT,
                           "dtype": "float32", "shape": [16]}
                          for n in WEIGHTS],
        },
    )
    meta.update(aot_serve.runtime_key())
    meta.update(over)
    return meta


def test_a_fenced_mint_declares_it() -> None:
    assert _meta()["constant_folding_fenced"] is True
    assert "constant_folding_fenced" in aot_serve.DECLARED_AXES


def test_a_pre_fence_cell_is_refused_before_a_byte_moves() -> None:
    """The same shape of refusal ``package_constants_in_so`` already carries:
    a cell minted without the fence may hold the minting checkpoint's copy of
    any inlined weight, so it is sound for exactly one fine-tune. Absent flag
    = a pre-fence mint."""
    stale = _meta()
    stale.pop("constant_folding_fenced")
    reason = aot_serve.verify_declared(stale)
    assert "folding fence" in reason and "pgw#1097" in reason


def test_a_fenced_cell_passes_the_declared_gate() -> None:
    assert aot_serve.verify_declared(_meta()) == ""


@pytest.mark.parametrize("value", [False, "true", None, 1])
def test_only_a_real_true_satisfies_the_gate(value: object) -> None:
    assert "folding fence" in aot_serve.verify_declared(
        _meta(constant_folding_fenced=value))


# ---------------------------------------------------------------------------
# The gauntlet's blind spot, stated as a test so it stays stated
# ---------------------------------------------------------------------------


def _inlinable(shape: Iterable[int]) -> bool:
    """``GraphLowering``'s own two rules, torch 2.13.0 (``graph.py:1523``,
    ``:1570``): 0-dim, or 1-D with at most 8 elements."""
    dims = tuple(int(v) for v in shape)
    return dims == () or (len(dims) == 1 and dims[0] <= 8)


def test_no_micro_family_weight_can_fold_which_is_why_pgw1073_passed() -> None:
    """pgw#1073 scenario 6 proved two micro checkpoints share one cell with
    exact parity. It could not have failed: no micro parameter is eligible for
    either inline rule. The property was true; it was not ENFORCED, and this
    test says so out loud so nobody reads that proof as covering the fence."""
    torch = pytest.importorskip("torch")
    from micro_diffusion.model import MicroConfig, MicroDecoder, MicroDenoiser

    cfg = MicroConfig()
    eligible = []
    for target in (MicroDenoiser(cfg), MicroDecoder(cfg)):
        for name, tensor in target.state_dict().items():
            if _inlinable(tuple(tensor.shape)):
                eligible.append(f"{type(target).__name__}.{name}")
    assert eligible == [], (
        "a micro weight became foldable — the gauntlet now COVERS the fence, "
        f"which is news either way: {eligible}")
    assert _inlinable(()) and _inlinable((8,)) and not _inlinable((9,))
    assert torch is not None


def test_the_sdxl_case_is_a_shape_rule_not_conv_fusion() -> None:
    """The one real-weight elimination the tree had recorded — sdxl's
    ``unet.conv_out.bias``, program 2423 / package 2422 — was filed as conv
    epilogue fusion. It is 4 floats in one dimension, which is
    ``can_inline_constant``, and its values were rendered into the kernel."""
    assert _inlinable((4,))


def test_a_synthetic_declaration_round_trips(tmp_path: Path) -> None:
    """The fixture is only worth what it parses like: prove the synthetic
    wrapper reads back through the same introspection the real one does."""
    package = _package(tmp_path, BOUND_ROWS)
    declared = aot_package.declared_constants(package, ENTRY)
    assert [c.fqn for c in declared] == list(WEIGHTS)
    assert all(c.source == aot_serve.SOURCE_STATE_DICT for c in declared)
    assert json.loads(json.dumps(
        aot_package.constants_manifest(package, ENTRY)))
