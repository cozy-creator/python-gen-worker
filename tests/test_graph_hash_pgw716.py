"""ck6 canonical graph identity.

The load-bearing test in this file is the RANGE-COLLISION one: pgw#704 measured
three sdxl exports differing ONLY in declared dynamic range collapsing to one
node-only digest (``9dd33abbc7617d98``), which would let a worker adopt a cell
that refuses the envelope its key promised. Everything else here pins the
properties ck6 exists for — same graph shares a key, weight values never key,
different graph never shares.
"""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from gen_worker import graph_hash as gh  # noqa: E402  (after importorskip)


class _Tiny(torch.nn.Module):
    def __init__(self, width: int = 8, dtype=torch.float32) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(width, width, dtype=dtype)

    def forward(self, x):
        return torch.relu(self.lin(x))


class _TinyTwin(torch.nn.Module):
    """A DIFFERENT recipe — different class, different local names, a different
    source-level relu call — that traces to the SAME graph. This is the whole
    advantage over recipe-digest identity, and it is what a comment-only SDK
    diff looks like to the canonicalizer."""

    def __init__(self, width: int = 8, dtype=torch.float32) -> None:
        super().__init__()
        # Same FQN as _Tiny on purpose: a parameter FQN is the artifact's
        # BINDING contract (see test_module_rename_is_a_new_key), not noise.
        self.lin = torch.nn.Linear(width, width, dtype=dtype)

    def forward(self, x):
        projected = self.lin(x)  # a different way to write the same graph
        return torch.nn.functional.relu(projected)


class _TinyRenamed(torch.nn.Module):
    """Same math, different parameter FQNs — a DIFFERENT tensor-binding contract."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(width, width)

    def forward(self, x):
        return torch.relu(self.projection(x))


class _TinyPlusOne(torch.nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(width, width)

    def forward(self, x):
        return torch.relu(self.lin(x)) + 1.0


def _export(module, batch=4, width=8, lo=2, hi=16):
    dim = torch.export.Dim("batch", min=lo, max=hi)
    example = torch.randn(batch, width, dtype=next(module.parameters()).dtype)
    return torch.export.export(
        module, (example,), dynamic_shapes={"x": {0: dim}} if _first_arg(
            module) == "x" else ({0: dim},), strict=False)


def _first_arg(module) -> str:
    return list(inspect.signature(module.forward).parameters)[0]


def _capture_dynamo(module, example):
    captured = []

    def backend(gm, inputs):
        captured.append(gm)
        return gm.forward

    torch._dynamo.reset()
    torch.compile(module, backend=backend)(example)
    assert captured, "dynamo captured no graph"
    return captured[0]


def _node_lines(lines):
    return tuple(line for line in lines if line.startswith(gh.NODE_PREFIX))


def _sym_lines(lines):
    return tuple(line for line in lines if line.startswith(gh.SYM_PREFIX))


# --------------------------------------------------------------------------
# THE soundness test (pgw#704 S8, measured): declared range must key.
# --------------------------------------------------------------------------

def test_declared_range_must_not_collide_pgw704_s8():
    module = _Tiny()
    programs = [_export(module, lo=lo, hi=hi)
                for lo, hi in ((2, 16), (2, 32), (4, 8))]
    lines = [gh.canonical_lines(program) for program in programs]
    digests = [gh.digest_lines(each) for each in lines]

    # The measured finding this test exists for: the NODES are identical across
    # all three, so a node-only hash gives one digest for three artifacts
    # whose declared envelopes differ.
    assert _node_lines(lines[0]) == _node_lines(lines[1]) == _node_lines(lines[2])
    node_only = {gh.digest_lines(_node_lines(each)) for each in lines}
    assert len(node_only) == 1, "premise broken: nodes were expected to collide"

    # ...and the fix: ranges are IN the canonical form, so the three keys differ.
    assert len(set(digests)) == 3, (
        "three exports admitting different shape ranges share a graph hash — "
        "adopting one against another's key refuses the envelope the key promised")
    assert len({_sym_lines(each) for each in lines}) == 3
    for line_set in lines:
        assert _sym_lines(line_set), "no range facts recorded for a dynamic dim"


def test_declared_range_keys_the_dynamo_path_too_pgw702():
    """The same defect for dynamo cells with declared dynamic dims — one
    canonicalizer, one fix."""
    digests = set()
    for hi in (16, 64):
        module = _Tiny()
        example = torch.randn(4, 8)
        torch._dynamo.mark_dynamic(example, 0, min=2, max=hi)
        lines = gh.canonical_lines(_capture_dynamo(module, example))
        assert _sym_lines(lines), "dynamo dynamic dim recorded no range fact"
        digests.add(gh.digest_lines(lines))
    assert len(digests) == 2


# --------------------------------------------------------------------------
# The properties ck6 exists for.
# --------------------------------------------------------------------------

def test_same_graph_different_recipe_shares_the_hash():
    torch.manual_seed(0)
    one = _export(_Tiny())
    torch.manual_seed(1)
    twin = _export(_TinyTwin())
    assert gh.graph_hash(one) == gh.graph_hash(twin), (
        "two recipes that trace the same graph must share cells — that is the "
        "advantage over recipe-digest identity")


def test_module_rename_is_a_new_key():
    """A parameter FQN is what the serve path binds resident weights BY
    (pgw#721's constants-bound gate refuses by name), so renaming the module
    that holds a weight is a different tensor-binding contract, not the same graph."""
    assert gh.graph_hash(_export(_Tiny())) != gh.graph_hash(
        _export(_TinyRenamed()))


def test_weight_values_never_key():
    module = _Tiny()
    before = gh.graph_hash(_export(module))
    with torch.no_grad():
        for param in module.parameters():
            param.add_(1.5)
    assert gh.graph_hash(_export(module)) == before, (
        "a fine-tune must share the graph; hashing weight bytes would give "
        "every merge its own key space")


def test_different_graph_never_shares_the_hash():
    base = gh.graph_hash(_export(_Tiny()))
    assert gh.graph_hash(_export(_TinyPlusOne())) != base      # extra op
    assert gh.graph_hash(_export(_Tiny(dtype=torch.bfloat16))) != base  # dtype
    assert gh.graph_hash(_export(_Tiny(width=16), width=16)) != base    # shape


def test_hash_is_deterministic_across_retrace():
    module = _Tiny()
    assert gh.graph_hash(_export(module)) == gh.graph_hash(_export(module))
    dynamo = _capture_dynamo(_Tiny(), torch.randn(4, 8))
    again = _capture_dynamo(_Tiny(), torch.randn(4, 8))
    assert gh.graph_hash(dynamo) == gh.graph_hash(again)


def test_canonical_form_carries_no_provenance():
    lines = gh.canonical_lines(_export(_Tiny()))
    blob = "\n".join(lines)
    for noise in (".py", "stack_trace", "nn_module_stack", "site-packages",
                  "seq_nr", "/home/"):
        assert noise not in blob, f"provenance leaked into graph identity: {noise}"


def test_both_ir_paths_ingest():
    export_lines = gh.canonical_lines(_export(_Tiny()))
    dynamo_lines = gh.canonical_lines(_capture_dynamo(_Tiny(), torch.randn(4, 8)))
    assert export_lines[0].endswith("ir=export")
    assert dynamo_lines[0].endswith("ir=dynamo")
    assert _node_lines(export_lines) and _node_lines(dynamo_lines)


def test_non_graph_is_refused_by_name():
    with pytest.raises(gh.GraphHashError) as exc:
        gh.graph_hash({"not": "a graph"})
    assert "not a graph IR" in str(exc.value)


# --------------------------------------------------------------------------
# (pgw#1030: the combined-hash tests moved out with the symbol — the live
# formula and its coverage are `cell_key.manifest_digest`.)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Second soundness vector (#728 survey): python-int-carried SHAPES.
# Two qwen-image exports at 1328² and 1024² share the node-only digest
# b8b3995d34f42719 and separate only once tensor shapes from meta['val'] are
# folded in. The vector is only meaningful if the op/connectivity projection
# really is identical — otherwise the test proves nothing about shapes.
# --------------------------------------------------------------------------

def _shapeless_lines(lines):
    """The projection the survey called 'node-only': ops and connectivity,
    with every tensor-meta fact removed."""
    out = []
    for line in lines:
        if not line.startswith(gh.NODE_PREFIX):
            continue
        out.append(line.split(" val=")[0])
    return tuple(out)


def _export_static(module, batch=4, width=8):
    """No dynamic dims — the qwen case: the shape is a concrete python int
    carried in tensor meta, not a declared symbolic range."""
    example = torch.randn(batch, width, dtype=next(module.parameters()).dtype)
    return torch.export.export(module, (example,), strict=False)


def test_shape_only_difference_must_not_collide_pgw728():
    small = _export_static(_Tiny(width=8), batch=4)
    large = _export_static(_Tiny(width=8), batch=16)

    # The premise: ops and connectivity are IDENTICAL — a digest over those
    # alone gives one value for two artifacts serving different shapes.
    assert _shapeless_lines(gh.canonical_lines(small)) == _shapeless_lines(
        gh.canonical_lines(large))
    assert gh.digest_lines(_shapeless_lines(gh.canonical_lines(small))) == \
        gh.digest_lines(_shapeless_lines(gh.canonical_lines(large)))

    # The fix: tensor meta is part of identity, so the two keys differ.
    assert gh.graph_hash(small) != gh.graph_hash(large), (
        "two exports differing only in a shape share a graph hash — the "
        "'shapes' term of ck6's graph_hashes is load-bearing (pgw#728: two "
        "qwen rows collided on node-only digest b8b3995d34f42719)")

    # Positive control, and the reason the DECLARED-dynamic case must not be
    # confused with this one: when the dim is declared symbolic, both sizes
    # ARE one artifact and MUST share the key (pgw#704's headline — a
    # 1024x1024 export served 1152x896).
    assert gh.graph_hash(_export(_Tiny(width=8), batch=4)) == gh.graph_hash(
        _export(_Tiny(width=8), batch=16))


def test_dynamo_graphs_also_carry_shapes():
    """Dynamo keeps tensor meta under example_value, not val — reading only
    'val' silently drops every shape from the dynamo IR's identity."""
    small = _capture_dynamo(_Tiny(), torch.randn(4, 8))
    large = _capture_dynamo(_Tiny(), torch.randn(16, 8))
    assert _shapeless_lines(gh.canonical_lines(small)) == _shapeless_lines(
        gh.canonical_lines(large))
    assert gh.graph_hash(small) != gh.graph_hash(large)
