"""pgw#1058 — the artifact-identity gate on ADMISSION facts.

Attempt 30 published the first-ever aot-inductor compiled graph and every adopting pod
refused all 36 entries. The filed hypothesis was a lying label (entry says
H_lat=80,W_lat=192, program specialized differently). Pulling the published
compiled graph apart off-pod ($0) FALSIFIED that: the labels and the packed manifest
faithfully describe the program. The program itself was minted for a call
class real traffic never presents — sdxl's declaration omitted `dtype` on its
scalar `timestep` row and the SDK silently defaulted it to the MODULE's weight
dtype (bfloat16), while every real scheduler presents float32. 36 entries,
zero admissible calls, published.

Two defect classes die here:

1. THE SILENT DTYPE GUESS — `Input.dtype` is now required: a concrete torch
   dtype or the explicit word "model". Omission fails the declaring repo's
   import, not a rented mint pod.
2. THE UNVERIFIED LABEL (the pgw#1058 acceptance as filed, one layer down
   from pgw#1042) — the declared manifest rows are now proven against the
   artifact's OWN generated `check_input_<i>` guards, at package time (a
   divergent compiled graph is never published) and at arm time (a corrupted one is
   never served), through ONE function: `aot_package.admission_drift`.

The fixtures under tests/fixtures/pgw1058/ are extracted verbatim from the
REAL published compiled graph (compiled_graph_store `ck1-9ae7bbea…` → `sha256:82efc111…`, entry
`unet/adapter=false,cfg=false/B=1,H_lat=80,T_txt=77,W_lat=192`): its packed
manifest rows and its wrapper's generated input checks. The tests below rule
on the actual bytes the fleet refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import msgspec
import pytest

from gen_worker import aot_package
from gen_worker.api.export_contract import (
    INPUT_DTYPES,
    MODEL_DTYPE,
    DeclarationError,
    Input,
)

FIXTURES = Path(__file__).parent / "fixtures" / "pgw1058"

#: The ingress class the adopt pod's REAL warmup presented, verbatim from the
#: attempt-30 `shape_gap/no_entry_admits` event:
#:   class=unet/#0=bfloat16[1,4,80,192],#1=float32[],…
#: shaped the way the pipeline actually calls (added_cond_kwargs is ONE dict
#: argument; the contract's leaf paths replay into it, pgw#994).
def _real_warmup_call() -> dict:
    return {
        "sample": _Tensor((1, 4, 80, 192), "torch.bfloat16"),
        "timestep": _Tensor((), "torch.float32"),
        "encoder_hidden_states": _Tensor((1, 77, 2048), "torch.bfloat16"),
        "added_cond_kwargs": {
            "text_embeds": _Tensor((1, 1280), "torch.bfloat16"),
            "time_ids": _Tensor((1, 6), "torch.bfloat16"),
        },
    }


class _Tensor:
    """Just enough tensor for the ingress assertion (shape + dtype)."""

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


def _entry_meta() -> dict:
    return json.loads((FIXTURES / "attempt30_entry_meta.json").read_text())


def _guards() -> tuple:
    source = (FIXTURES / "attempt30_check_inputs.cpp").read_text()
    return aot_package.guards_from_wrapper_source(source, where="attempt30")


# ---------------------------------------------------------------------------
# 1 — the silent dtype guess is unconstructable
# ---------------------------------------------------------------------------


def test_an_input_row_without_a_dtype_is_refused_at_declaration_time():
    """The RED half of the root cause: sdxl's `Input("timestep", shape=(),
    value=1.0)` must no longer construct. The refusal names the input, the
    reason the fact cannot be inherited, and both ways to state it."""
    with pytest.raises(DeclarationError) as excinfo:
        Input("timestep", shape=(), value=1.0)
    message = str(excinfo.value)
    assert "timestep" in message
    assert MODEL_DTYPE in message
    assert "pgw#1058" in message


def test_an_unknown_dtype_word_is_refused_by_name():
    with pytest.raises(DeclarationError, match="unknown dtype 'bfloat17'"):
        Input("timestep", shape=(), dtype="bfloat17")


def test_the_explicit_inheritance_word_constructs():
    row = Input("sample", shape=("B", 4), dtype=MODEL_DTYPE)
    assert row.dtype == MODEL_DTYPE
    assert MODEL_DTYPE in INPUT_DTYPES


def test_a_row_that_dodged_validation_is_refused_at_the_feed_builder():
    """Defense in depth: `declared_inputs` refuses an empty dtype rather than
    guessing, even for a row that bypassed `Input.__post_init__`."""
    torch = pytest.importorskip("torch")
    from gen_worker.aot_contract import ExportSpec, MintRefused
    from gen_worker.aot_declaration import declared_inputs
    from gen_worker.api.decorators import Compile
    from gen_worker.api.export_contract import Dim, GraphClass

    row = Input("timestep", shape=(), dtype=MODEL_DTYPE, value=1.0)
    msgspec.structs.force_setattr(row, "dtype", "")
    decl = Compile(
        family="pgw1058", targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 1}),),
        inputs=(Input("sample", shape=("B", 4), dtype=MODEL_DTYPE), row),
        shape_strategy="static-rows",
    )
    module = torch.nn.Linear(4, 4)
    spec = ExportSpec(family="pgw1058", target="unet")
    with pytest.raises(MintRefused, match="declares no dtype"):
        declared_inputs(module, spec, decl)


# ---------------------------------------------------------------------------
# 2 — the published compiled graph's bytes, ruled on directly
# ---------------------------------------------------------------------------


def test_the_real_compiled_graphs_guards_parse_and_state_the_defect():
    """The artifact's own generated checks, read from the published wrapper:
    sample really is bfloat16[1,4,80,192] — the LABEL WAS HONEST — and
    timestep really is bfloat16, the dtype no real scheduler ever presents."""
    guards = _guards()
    assert [g.index for g in guards] == [0, 1, 2, 3, 4]
    sample, timestep = guards[0], guards[1]
    assert sample.dtype == "bfloat16"
    assert sample.dim_map() == {0: 1, 1: 4, 2: 80, 3: 192}
    assert timestep.dtype == "bfloat16"          # THE defect, held in amber
    assert timestep.dim_map() == {}              # rank-0, fully specialized


def test_the_labels_were_honest_the_filed_hypothesis_is_false():
    """pgw#1058's filed hypothesis — label vs packaged static dims diverged —
    is FALSIFIED from the bytes: the packed manifest rows agree with the
    artifact's own guards on every input, every dim, every dtype."""
    assert aot_package.input_guard_drift(_entry_meta()["inputs"], _guards()) == []


def test_the_real_compiled_graph_refuses_the_real_warmup_call_on_timestep_dtype():
    """The $0 reproduction of the field failure: the serve path's own
    admission (`assert_ingress` on the packed contract) against the exact
    warmup class the adopt pod presented. The shape-matching entry's true
    refusal is `dtype_mismatch` on timestep — the `static_dim_mismatch`
    36/36 reading came from `select()`'s reasons[:6] truncation over the
    34 wrong-shape entries."""
    from gen_worker import aot_serve

    contract = aot_serve.contract_from_meta(_entry_meta())
    with pytest.raises(aot_serve.IngressContractError) as excinfo:
        aot_serve.assert_ingress(contract, (), _real_warmup_call())
    assert excinfo.value.reason == "dtype_mismatch"
    assert "timestep" in str(excinfo.value)


def test_the_corrected_declaration_catches_the_real_compiled_graph_by_name():
    """Under the fix, sdxl declares `timestep` float32 — and the gate run
    against the PUBLISHED compiled graph's own bytes now names the defect instead of
    36 opaque admission misses: the manifest a float32 mint would carry
    drifts from this artifact's bfloat16 guard."""
    rows = [dict(r) for r in _entry_meta()["inputs"]]
    for row in rows:
        if row["name"] == "timestep":
            row["dtype"] = "float32"
    drift = aot_package.input_guard_drift(rows, _guards())
    assert len(drift) == 1
    assert "timestep" in drift[0]
    assert "float32" in drift[0] and "bfloat16" in drift[0]


# ---------------------------------------------------------------------------
# 3 — the label cross-check, against a freshly compiled package
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_package(tmp_path_factory):
    """A REAL exported+AOTI-compiled package with a static and a rank-0
    input — the smallest artifact whose generated guards carry the facts
    the gate rules on."""
    torch = pytest.importorskip("torch")

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)

        def forward(self, x, t):
            return self.lin(x) * t.float()

    x = torch.randn(2, 8)
    t = torch.full((), 100.0, dtype=torch.float32)
    program = torch.export.export(M().eval(), (x, t), strict=True)
    path = tmp_path_factory.mktemp("pgw1058") / "m.pt2"
    torch._inductor.aoti_compile_and_package(
        program, package_path=str(path),
        # pgw#1097: compile like a REAL mint does. Without the folding fence
        # this module's `lin.bias` is 1-D with 8 elements, which meets
        # `GraphLowering.can_inline_constant`, so inductor renders its values
        # into the kernel and the weight leaves the artifact's constant table
        # entirely — and the mint's folding fence then refuses this package
        # BEFORE the admission gate this fixture exists to exercise. That
        # refusal is correct (it is the pgw#1097 hazard, reproduced here by
        # accident on an unrelated fixture); it is simply not what these tests
        # are about.
        inductor_configs={"aot_inductor.package_constants_in_so": False,
                          "aot_inductor.use_runtime_constant_folding": True})
    return program, path


def _honest_rows(program) -> list:
    from gen_worker import aot_flatten

    leaves = (
        aot_flatten.Leaf(param="x", param_position=0, path=()),
        aot_flatten.Leaf(param="t", param_position=1, path=()),
    )
    rows, _symbols = aot_package.input_contract(program, leaves)
    return rows


def test_an_honest_manifest_has_no_drift(real_package):
    program, path = real_package
    assert aot_package.admission_drift(path, "", _honest_rows(program)) == []


@pytest.mark.parametrize(
    "corrupt, expect",
    [
        (lambda rows: rows[0].__setitem__("dtype", "bfloat16"),
         "declares dtype 'bfloat16'"),
        (lambda rows: rows[0]["shape"].__setitem__(0, 4),
         "declares static 4"),
        (lambda rows: rows[0]["shape"].__setitem__(0, "B"),
         "declares symbolic 'B'"),
        (lambda rows: rows.pop(),
         "cannot describe one call contract"),
    ],
    ids=["dtype", "static_dim", "symbolic_claim", "row_count"],
)
def test_a_corrupted_label_fails_closed(real_package, corrupt, expect):
    """The pgw#1058 acceptance's RED chain: corrupt any admission fact of a
    label and the SAME derivation refuses it, by name, at package time —
    which is what makes arm-time 36/36 misses impossible to publish."""
    program, path = real_package
    rows = [dict(r, shape=list(r["shape"])) for r in _honest_rows(program)]
    corrupt(rows)
    drift = aot_package.admission_drift(path, "", rows)
    assert drift and expect in drift[0]


def test_the_mint_package_gate_runs_the_check(real_package, monkeypatch):
    """`_gate_and_declare_entry` — the produce half — refuses a package whose
    guards diverge from the program-derived manifest. Wiring proof: the same
    package with a program whose placeholders disagree must raise
    MintRefused naming the drift."""
    torch = pytest.importorskip("torch")
    from gen_worker import aot_flatten, aot_mint

    class M2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)

        def forward(self, x, t):
            return self.lin(x) * t.float()

    module = M2().eval()
    # A DIFFERENT batch than the compiled package (4 vs 2): the honest
    # program for this row disagrees with the packaged artifact's guards.
    other = torch.export.export(
        module, (torch.randn(4, 8),
                 torch.full((), 100.0, dtype=torch.float32)), strict=True)
    _program, path = real_package
    row = aot_mint._MintedEntry(
        name="", spec=aot_mint.ExportSpec(family="pgw1058", target="unet"),
        module=module, owner=module, program=other,
        input_names=("x", "t"),
        flat_leaves=(
            aot_flatten.Leaf(param="x", param_position=0, path=()),
            aot_flatten.Leaf(param="t", param_position=1, path=()),
        ),
        files=[], timings={})
    with pytest.raises(aot_mint.MintRefused, match="admission drift"):
        aot_mint._gate_and_declare_entry(row, path)


def test_the_arm_gate_is_the_same_derivation(real_package):
    """The arm side calls the ONE function too — a corrupted manifest row
    arriving with published bytes is an AdoptError('admission_drift'),
    never an opaque per-call miss."""
    from gen_worker import aot_serve
    from gen_worker.compile_cache import AdoptError

    program, path = real_package
    rows = [dict(r, shape=list(r["shape"])) for r in _honest_rows(program)]
    rows[1]["dtype"] = "bfloat16"
    with pytest.raises(AdoptError) as excinfo:
        aot_serve._entry_admission_drift(path, "", rows)
    assert excinfo.value.reason == "admission_drift"
    aot_serve._entry_admission_drift(path, "", _honest_rows(program))
