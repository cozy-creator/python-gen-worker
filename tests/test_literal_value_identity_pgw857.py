"""pgw#857: a cell's LITERAL constants are part of the artifact, so key them.

THE BUG, in one sentence: the graph hash folded constant NAMES but never their
VALUES, so two checkpoints that need different literals shared a cell key —
and a pod could adopt one built for the other. It arms, it serves, and it is
wrong.

WHY NAMES-ONLY WAS RIGHT FOR WEIGHTS AND WRONG FOR LITERALS. A weight is
rebound from the resident ``state_dict`` at load, so two fine-tunes of one
family SHOULD share a cell — keying weight values would break exactly the
property the exclusion exists to protect. A ``SOURCE_LITERAL`` constant is
different in kind: it ships INSIDE the artifact and is never rebound
(*"nothing outside the artifact knows its value"*, ``aot_serve``), so **for a
literal the value IS the artifact**. Both were excluded; only one should have
been.

MEASURED, before building (pgw#857 measurement 1 and 2):

* **sdxl lifts ZERO literals** — five real recorded mints, ~2,420 constants
  each, every one a dotted state_dict FQN. So this fix must be a NO-OP for it,
  which matters because a forge lane is minting sdxl right now.
* **z-image and qwen-image each lift rope tables** — ~393 KB and 4.19 MB,
  both pure functions of config.

The discriminator is **assignment style, not class ancestry**:
``QwenEmbedRope`` IS an ``nn.Module`` and its ``state_dict()`` is still empty,
because its tables are plain attributes rather than ``register_buffer``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence

import pytest

from gen_worker.aot_package import (
    literal_values_digest, program_literal_fqns, program_state_dict_fqns,
)
from gen_worker.aot_serve import class_hash

torch = pytest.importorskip("torch")


class _Program:
    """The two things the digest reads off an ExportedProgram."""

    def __init__(
        self, constants: Dict[str, Any], *,
        lifted: Sequence[str] = (), params: Sequence[str] = (),
        buffers: Sequence[str] = (),
    ) -> None:
        self.constants = dict(constants)
        self.graph_signature = type("_Sig", (), {
            "parameters": tuple(params),
            "buffers": tuple(buffers),
            "lifted_tensor_constants": tuple(lifted),
        })()


def _rope(theta: float, n: int = 64) -> Any:
    """A rope table, the way both affected families build one: a pure function
    of config ints, complex64, held as a plain attribute."""
    freqs = 1.0 / (theta ** (torch.arange(0, 8, 2, dtype=torch.float64) / 8))
    grid = torch.outer(torch.arange(n, dtype=torch.float64), freqs).float()
    return torch.polar(torch.ones_like(grid), grid).to(torch.complex64)


def _graph_block(program: _Program, *, with_fix: bool) -> Dict[str, Any]:
    """The entry graph block, with and without pgw#857's field — so the
    collision can be shown rather than asserted."""
    block: Dict[str, Any] = {
        # v3 (pgw#1089): program-only. `fused_constants` is gone from the block
        # entirely — the compiler's folding decisions are a function of
        # (graph x toolchain x sm) and carry no bits the key lacks. Kept in
        # step with the real block so this collision demonstration keeps
        # describing the shape the mint actually stamps.
        "v": 3,
        "constant_fqns": sorted(program.constants),
        "lifted_inputs": [],
        "pytree": {"in": "x", "out": "y"},
        "specialization": {},
    }
    if with_fix:
        digest = literal_values_digest(program)
        if digest:
            block["literal_values"] = digest
    return block


def _entry(program: _Program, *, with_fix: bool) -> Dict[str, Any]:
    return {
        "target": "transformer",
        "fork": [],
        "class_dims": [["B", 1]],
        "range_digest": "rd",
        "graph": _graph_block(program, with_fix=with_fix),
    }


def _key(program: _Program, *, with_fix: bool) -> str:
    return class_hash(_entry(program, with_fix=with_fix), strict=True, lora_bucket=0)


# ---------------------------------------------------------------------------
# 1. THE FAILURE, PROVEN RED FIRST — two rope configs, one key
# ---------------------------------------------------------------------------


def test_RED_two_rope_configs_collided_before_the_fix() -> None:
    """The bug, demonstrated rather than described. Two checkpoints of one
    family whose rope tables differ ONLY in `theta` — same names, same shapes,
    same dtypes, different VALUES — hashed identically."""
    a = _Program({"_tensor_constant0": _rope(256.0)}, lifted=["_tensor_constant0"])
    b = _Program({"_tensor_constant0": _rope(10000.0)}, lifted=["_tensor_constant0"])

    # The values really do differ...
    assert not torch.equal(a.constants["_tensor_constant0"],
                           b.constants["_tensor_constant0"])
    # ...and the pre-fix identity could not tell them apart. THIS IS THE BUG.
    assert _key(a, with_fix=False) == _key(b, with_fix=False)


def test_GREEN_the_same_two_configs_now_separate() -> None:
    a = _Program({"_tensor_constant0": _rope(256.0)}, lifted=["_tensor_constant0"])
    b = _Program({"_tensor_constant0": _rope(10000.0)}, lifted=["_tensor_constant0"])

    assert _key(a, with_fix=True) != _key(b, with_fix=True)


def test_an_identical_literal_still_keys_identically() -> None:
    """Separation must come from the VALUE, not from re-hashing noise: two
    programs with the same literal must still share a key, or every re-mint
    would produce a new cell."""
    a = _Program({"_tensor_constant0": _rope(256.0)}, lifted=["_tensor_constant0"])
    b = _Program({"_tensor_constant0": _rope(256.0)}, lifted=["_tensor_constant0"])

    assert _key(a, with_fix=True) == _key(b, with_fix=True)


# ---------------------------------------------------------------------------
# 2. THE EXCLUSION THIS MUST NOT BREAK — fine-tunes still share a cell
# ---------------------------------------------------------------------------


def test_two_FINE_TUNES_of_one_family_still_share_a_cell() -> None:
    """The property the names-only rule exists to protect, and the one a
    careless fix would destroy. Same architecture, DIFFERENT weight values —
    they are rebound from state_dict at load, so the cell must be shared."""
    base = _Program(
        {"unet.conv_in.weight": torch.ones(8), "unet.conv_in.bias": torch.zeros(4)},
        params=["unet.conv_in.weight"], buffers=["unet.conv_in.bias"])
    tuned = _Program(
        {"unet.conv_in.weight": torch.full((8,), 7.0),
         "unet.conv_in.bias": torch.full((4,), -3.0)},
        params=["unet.conv_in.weight"], buffers=["unet.conv_in.bias"])

    assert not torch.equal(base.constants["unet.conv_in.weight"],
                           tuned.constants["unet.conv_in.weight"])
    assert _key(base, with_fix=True) == _key(tuned, with_fix=True), (
        "a fine-tune must still share its family's cell — weights are rebound "
        "from state_dict at load, and keying their values would break the "
        "premise of family-scoped cells")


def test_a_weight_is_never_digested_even_alongside_a_literal() -> None:
    """Mixed program: only the literal's value may move the digest."""
    lit = _rope(256.0)
    a = _Program({"w": torch.ones(8), "_tensor_constant0": lit},
                 params=["w"], lifted=["_tensor_constant0"])
    b = _Program({"w": torch.full((8,), 9.0), "_tensor_constant0": lit},
                 params=["w"], lifted=["_tensor_constant0"])

    assert program_state_dict_fqns(a) == ("w",)
    assert program_literal_fqns(a) == ("_tensor_constant0",)
    assert literal_values_digest(a) == literal_values_digest(b)


# ---------------------------------------------------------------------------
# 3. sdxl MUST NOT RE-KEY — verified, not inferred
# ---------------------------------------------------------------------------


def test_a_program_with_NO_literals_produces_a_byte_identical_block() -> None:
    """sdxl's measured shape: ~2,420 constants, every one a dotted state_dict
    FQN, zero lifted literals. The field must be OMITTED — not empty-valued —
    so the block serialises exactly as before and the cell key does not move.

    A forge lane is minting sdxl right now; an sdxl re-key would invalidate an
    in-flight pod.
    """
    sdxl_shaped = _Program(
        {
            "conv_in.weight": torch.ones(4),
            "down_blocks.0.resnets.0.conv1.bias": torch.zeros(4),
            "add_embedding.linear_1.weight_scale": torch.ones(1),
        },
        params=["conv_in.weight", "add_embedding.linear_1.weight_scale"],
        buffers=["down_blocks.0.resnets.0.conv1.bias"],
    )

    assert program_literal_fqns(sdxl_shaped) == ()
    assert literal_values_digest(sdxl_shaped) == ""

    before = _graph_block(sdxl_shaped, with_fix=False)
    after = _graph_block(sdxl_shaped, with_fix=True)
    assert "literal_values" not in after
    assert json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)
    assert _key(sdxl_shaped, with_fix=False) == _key(sdxl_shaped, with_fix=True)


def test_the_field_is_omitted_rather_than_emitted_empty() -> None:
    """An empty-valued field would re-key every existing cell to say
    'unchanged' — the mistake `range_digest`'s `excluded` already avoids."""
    empty = _Program({"w": torch.ones(2)}, params=["w"])
    assert "literal_values" not in _graph_block(empty, with_fix=True)


# ---------------------------------------------------------------------------
# 4. FAIL CLOSED — a literal that cannot be read must refuse, never be skipped
# ---------------------------------------------------------------------------


def test_an_unreadable_literal_refuses_rather_than_being_skipped() -> None:
    """A silently-skipped constant is the hole this function exists to close,
    so an unreadable one is an error and not a gap in the digest."""
    missing = _Program({}, lifted=["_tensor_constant0"])
    with pytest.raises(ValueError) as excinfo:
        literal_values_digest(missing)
    assert "_tensor_constant0" in str(excinfo.value)

    class _Hostile:
        dtype = "weird"
        shape = (1,)

        def detach(self):
            raise RuntimeError("no bytes here")

    with pytest.raises(ValueError):
        literal_values_digest(
            _Program({"_tensor_constant0": _Hostile()},
                     lifted=["_tensor_constant0"]))


def test_complex64_is_digested_without_a_dtype_special_case() -> None:
    """Both real instances are complex64; a raw byte view must handle it."""
    a = _Program({"c": _rope(256.0)}, lifted=["c"])
    assert len(literal_values_digest(a)) == 32
