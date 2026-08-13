"""pgw#1132 — the BOOT-KEY derivation owns its own branch arm, or no
``lora_bucket`` family can ever ask for its compiled graph.

pgw#822 at the mint: the child armed the branch CONTAINERS
(``compile_cache.apply_lora_execution_lane``) and handed ``torch.export`` the
BARE denoiser, whose forward never took ``lora_a``/``lora_b``. The mint's own
loop fixed it by owning the arm (``aot_mint._arm_branches``, called before the
first export in ``mint_targets``).

§4.27 step 1's loop — ``aot_mint.trace_for_key``, driven by
``boot_trace_child`` — was left with the arm only in its ``finally`` (the
re-arm after the branchless group), so the FIRST adapter-bearing row of every
bucket-bearing family met ``_export_entry``'s pgw#822 gate on a container-only
pipeline and refused. ``boot_trace_child`` turns that into ``trace_refused``,
so the derivation dies and the pod never issues a resolve — AOT adoption is
100 % unreachable for every family that declares a bucket (qwen-image and
qwen-image-edit 128, z-image 128, wan-2.2 128/64, sdxl 64, sd15 64, anima 32).

Every test here runs the REAL loop over the REAL gate: no stub stands in for
``_export_entry``, which is the thing under test.
"""

from __future__ import annotations

import types
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import aot_mint, boot_adopt, compile_cache  # noqa: E402
from gen_worker.api.decorators import Compile  # noqa: E402
from gen_worker.api.export_contract import (  # noqa: E402
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
    reset_export_declarations,
)
from gen_worker.models import lora_lifted, w8a8_lora  # noqa: E402

FAMILY = "tiny1132"
BUCKET = 16      # RANK_BUCKETS' floor — the cheapest real branch there is

ADAPTER_TRUE = "unet/adapter=true/B=2"
ADAPTER_FALSE = "unet/adapter=false/B=2"


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(BUCKET, BUCKET)

    def forward(self, sample: Any) -> Any:
        return torch.tanh(self.lin(sample))


def _declare(**changes: Any) -> Any:
    reset_export_declarations()
    fields: Dict[str, Any] = dict(
        family=FAMILY,
        targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", BUCKET), dtype="model"),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    )
    fields.update(changes)
    return register_export_declaration(Compile(**fields))


def _boot_child_pipe() -> Any:
    """EXACTLY the state ``boot_trace_child.run`` hands to ``trace_for_key``
    — its own call, on a pipeline nothing else prepared."""
    pipe = types.SimpleNamespace(unet=TinyUNet().eval())
    compile_cache.apply_lora_execution_lane(pipe, BUCKET)
    assert lora_lifted.lifted_binding(pipe.unet) is None
    return pipe


def _spec() -> aot_mint.ExportSpec:
    return aot_mint.ExportSpec(
        family=FAMILY, target="", lora_bucket=BUCKET,
        lifted_inputs=lora_lifted.LIFTED_INPUT_NAMES)


def _traced(pipe: Any, decl: Any, **shard: Any) -> List[Any]:
    rows = []
    for row in aot_mint.trace_for_key(pipe, _spec(), decl, **shard):
        row.program = None       # the child drops it; hold nothing but facts
        rows.append(row)
    return rows


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_export_declarations()
    yield
    reset_export_declarations()


# ---------------------------------------------------------------------------
# The derivation, end to end: RED at base — the first adapter-bearing row
# refuses "carries no lifted forward" and nothing downstream ever runs.
# ---------------------------------------------------------------------------


def test_a_container_only_pipeline_derives_every_declared_class() -> None:
    decl = _declare()
    rows = _traced(_boot_child_pipe(), decl)
    assert [row.name for row in rows] == [ADAPTER_TRUE, ADAPTER_FALSE]
    assert {row.declared for row in rows} == {2}
    assert all(row.nodes > 0 for row in rows)


def test_the_two_classes_are_keyed_from_DIFFERENT_graphs() -> None:
    """The pgw#790 fork must survive the arm: an adapter-bearing class that
    keyed identically to the branchless one would mean the lift never reached
    the traced graph — a green derivation over the wrong graph family."""
    decl = _declare()
    blocks = {row.name: row.block for row in _traced(_boot_child_pipe(), decl)}
    lifted = sorted(lora_lifted.LIFTED_INPUT_NAMES)
    assert blocks[ADAPTER_TRUE]["graph"]["lifted_inputs"] == lifted
    assert blocks[ADAPTER_FALSE]["graph"]["lifted_inputs"] == []
    assert blocks[ADAPTER_TRUE] != blocks[ADAPTER_FALSE]


def test_every_shard_arms_itself() -> None:
    """The derivation runs K children, each on its OWN pipeline, and a share
    may hold only branchless rows or only adapter-bearing ones. The arm is per
    LOOP, so the shares still reconstruct the whole class set."""
    decl = _declare()
    names: List[str] = []
    for index in range(2):
        rows = _traced(_boot_child_pipe(), decl, share_index=index,
                       share_count=2)
        assert {row.declared for row in rows} == {2}
        names.extend(row.name for row in rows)
    assert sorted(names) == [ADAPTER_FALSE, ADAPTER_TRUE]


def test_the_loop_leaves_the_pipeline_on_the_lifted_family() -> None:
    """``_disarm_branches`` runs for the branchless group; a loop that
    returned the pipeline branchless would leave its process on a different
    graph family than the one it just keyed."""
    decl = _declare()
    pipe = _boot_child_pipe()
    _traced(pipe, decl)
    assert lora_lifted.lifted_binding(pipe.unet) is not None
    assert w8a8_lora.branch_bucket(pipe.unet) == BUCKET


# ---------------------------------------------------------------------------
# The controls — this file's assertions can go red
# ---------------------------------------------------------------------------


def test_a_declared_input_the_module_cannot_take_still_refuses() -> None:
    """The gate under test is the mint's own and stays load-bearing: an arm
    that swallowed a real declaration mismatch would be a second silent
    derivation."""
    decl = _declare(inputs=(
        Input("sample", shape=("B", BUCKET), dtype="model"),
        Input("encoder_hidden_states", shape=("B", BUCKET), dtype="model"),
    ))
    with pytest.raises(aot_mint.MintRefused, match="are not parameters of"):
        _traced(_boot_child_pipe(), decl)


def test_the_refusal_a_boot_child_would_report_is_in_the_vocabulary() -> None:
    """``boot_trace_child`` reports any ``MintRefused`` from this loop as
    ``trace_refused``; pgw#1116's vocabulary has to carry it or the pod's
    refusal is unenumerable off-pod."""
    assert "trace_refused" in boot_adopt.REASONS
