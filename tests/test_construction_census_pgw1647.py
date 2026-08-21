"""pgw#1647 — the CONSTRUCTION CENSUS and its five invariants, CPU-only.

Four members of one defect family were found on rented hardware, each costing a
pod, each fixed as a symptom: pgw#1626 (`tie_weights` never ran), pgw#1638 (the
config's quantizer never ran; and neither did `model.eval()`), pgw#1644 (the
whole-module `.to(device)` never ran, so three non-persistent RoPE buffers sat
on the CPU under an all-CUDA model and died `mat1 is on cpu` eight milliseconds
into a forward, $0.89). Every one is the same sentence — *the meta skeleton is
built from the config alone, so the `from_pretrained` machinery never runs* —
and every one was invisible to the fence that was supposed to catch it, because
that fence walked the checkpoint CONTAINER. A container names the tensors a
checkpoint carries. It cannot name a tensor the CODE creates.

So this suite asks the whole question instead of the four symptoms. Each red arm
NEUTERS one step of the prepare seam and asserts the census names it:

    I1  neuter `retie`            ⇒ the alias is not its source (identity)
    I2  neuter `preprocess_model` ⇒ the quantizer's swap does not replay
    I3  neuter `eval`             ⇒ the module comes up in TRAIN mode
    I4  neuter the device sweep   ⇒ `inv_freq` is off the target device
    I5  add an UNREGISTERED buffer ⇒ a name nobody declared is refused

I5 is the one no per-symptom fix could ever have had, and it is why this closes
the CLASS rather than the fourth instance of it: a fifth `from_pretrained` side
effect that nobody has thought of yet arrives as an unknown name and becomes a
$0 publish refusal.

The `minimax-h3` fixture is BOTH required fixtures at once and that is not a
convenience — it is the incident. Its `Qwen3VLForConditionalGeneration`
conditioner is the RoPE-buffer case pgw#1644 died on (no T5-family fixture will
ever expose it: T5 uses a learned relative-position bias, which is a real
parameter the container carries) AND the fp8-blockwise case pgw#1638 died on,
357 `weight_scale_inv` and all.

NO WEIGHTS, NO GPU, NO DOWNLOADS. Parameters come up on meta and buffers are
computed from config, so a 66 GiB DiT costs what a tiny one costs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from gen_worker.serving.streaming import census, skeleton  # noqa: E402

from test_skeleton_conformance_pgw1633 import (  # noqa: E402
    CORPUS,
    FLEET,
    QUANTIZED,
    QUANTIZED_FLEET,
    SCALE_LEAF,
    _alias_map,
)

H3 = "minimax-h3"
H3_COMPONENT = "text_encoder"

#: The three tensors pgw#1644 paid $0.89 to name, with the device census the
#: probe read on the H200's own tree: `{'meta': 1272, 'cpu': 3}`. Every one is a
#: NON-PERSISTENT buffer — in no container and in no `state_dict` — which is
#: exactly why the container-walking fence was structurally blind to them.
ROPE_BUFFERS = (
    "model.language_model.rotary_emb.inv_freq",
    "model.language_model.rotary_emb.original_inv_freq",
    "model.visual.rotary_pos_emb.inv_freq",
)


def _tree(endpoint: str) -> Path:
    return CORPUS / endpoint


def _h3() -> Path:
    return QUANTIZED / H3


# ── the census is DATA: it serializes, it round-trips, it is addressed ──────


@pytest.mark.parametrize("endpoint", sorted(FLEET))
def test_every_fleet_class_censuses_and_round_trips_pgw1647(endpoint: str) -> None:
    """The release document carries this verbatim; the hub reads no torch.

    A census that cannot survive its own wire format is not release data, it is
    a debug print. Both directions and the digest, per fleet pipeline class.
    """
    taken = census.for_tree(_tree(endpoint))
    assert taken.components, endpoint
    assert taken.tensor_count > 0, endpoint

    document = taken.as_document()
    assert document["kind"] == census.CENSUS_KIND
    assert json.loads(json.dumps(document)) == document, "not JSON-clean"

    read_back = census.Census.from_document(json.loads(taken.canonical()))
    assert read_back == taken
    assert read_back.digest == taken.digest

    # And the round-tripped census verifies the module it was taken from, which
    # is the property the serve fence depends on.
    census.verify(read_back, taken, where=endpoint)


def test_a_census_document_of_an_unknown_kind_is_refused_pgw1647() -> None:
    """Store-and-forward means the worker must check what it was forwarded."""
    document = census.for_tree(_tree("sd15")).as_document()
    document["kind"] = "gen-worker.construction-census@99"
    with pytest.raises(census.CensusError, match="construction-census@99"):
        census.Census.from_document(document)


def test_the_census_states_the_whole_tensor_set_not_the_container_s_pgw1647() -> None:
    """Parameters AND buffers, persistent AND not — the h3 conditioner.

    The census must name the three tensors no container has ever named, or it is
    the container walk again under a new name.
    """
    row = census.for_tree(_h3()).by_component()[H3_COMPONENT]
    by_name = row.by_name()

    for name in ROPE_BUFFERS:
        assert name in by_name, f"{name} is missing from the census"
        assert by_name[name].kind == census.BUFFER
        assert by_name[name].persistent is False, (
            f"{name} is recorded as persistent; a persistent buffer IS in the "
            f"container and this suite would then be measuring the wrong class"
        )

    volatile = [r.name for r in row.tensors if not r.persistent]
    assert sorted(volatile) == sorted(ROPE_BUFFERS), volatile
    assert row.module_class == "Qwen3VLForConditionalGeneration"
    assert row.eval_mode is True


def test_the_census_states_the_quantizer_s_swap_pgw1647() -> None:
    """357 rule-owned scale tensors and the class that owns them.

    The count is the incident's own number — 51 layers x 7 quantized linears,
    which is exactly what the H200 reported as orphaned — and not a shape
    chosen to pass.
    """
    _component, expected_scales, swapped_class = QUANTIZED_FLEET[H3]
    row = census.for_tree(_h3()).by_component()[H3_COMPONENT]

    assert row.quant_rule, "the census records no quant rule for a quantized tree"
    scales = [
        r for r in row.tensors if r.name.endswith(f".{SCALE_LEAF}")
    ]
    assert len(scales) == expected_scales, len(scales)
    assert all(r.rule_owned for r in scales), (
        "a scale grid is not marked rule-owned, so the lane's dtype assertion "
        "would round 357 F32 block scales to bf16 and call it a repair"
    )
    assert {name for name, _cls in row.quant_modules}, "no swapped modules recorded"
    assert {cls for _name, cls in row.quant_modules} == {swapped_class}


def test_the_census_states_the_ties_by_identity_pgw1647() -> None:
    """Every tie exposure in the corpus, as alias groups.

    Identity, never `_tied_weights_keys`: that attribute lists names a class
    MIGHT tie and whether the tie is live is a config question, so reading it as
    an answer scores a class that declares `lm_head.weight` and does not tie it
    as a defect on a perfectly good checkpoint.
    """
    seen: Dict[str, List[str]] = {}
    for endpoint in sorted(FLEET):
        for row in census.for_tree(_tree(endpoint)).components:
            if row.ties:
                seen[f"{endpoint}/{row.component}"] = [
                    "->".join(group) for group in row.ties
                ]
    assert seen, "no tie exposure in the whole corpus — the census is not reading ties"
    for groups in seen.values():
        for rendered in groups:
            assert "->" in rendered


# ── I1 TIES ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "endpoint", ["flux.1-dev", "flux.2-klein-4b", "wan-2.2", "ernie"]
)
def test_I1_red_arm_a_neutered_retie_is_a_census_mismatch_pgw1647(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete `retie` and the census stops agreeing about the tie groups.

    Not "some parameter is missing": the ALIAS and its SOURCE stop being one
    object, which is the fact a checkpoint relies on when it omits the alias.
    """
    expected = census.for_tree(_tree(endpoint))
    assert any(row.ties for row in expected.components), endpoint

    monkeypatch.setattr(skeleton, "retie", lambda module: False)
    actual = census.for_tree(_tree(endpoint))

    with pytest.raises(census.CensusMismatch) as caught:
        census.verify(expected, actual, where=endpoint)
    assert caught.value.invariant == census.I1_TIES
    assert caught.value.tensor, "the refusal names no tensor"
    tied = {
        name
        for row in expected.components
        for group in row.ties
        for name in group[1:]
    }
    assert caught.value.tensor in tied, caught.value.tensor


def test_I1_a_tie_that_became_a_COPY_is_a_mismatch_too_pgw1647() -> None:
    """The other half of the tie question, and the expensive one.

    A copy places correctly, fills correctly and leaves nothing on meta — it
    just doubles the resident bytes and serves whichever of the two the stream
    wrote last. Only identity catches it.
    """
    module = torch.nn.Module()
    source = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    module.register_parameter("source", source)
    module.register_parameter("alias", source)
    expected = census.take_component("tiny", module)
    assert expected.ties == (("alias", "source"),) or expected.ties == (
        ("source", "alias"),
    ), expected.ties

    module.register_parameter(
        "alias", torch.nn.Parameter(source.detach().clone(), requires_grad=False)
    )
    with pytest.raises(census.CensusMismatch) as caught:
        census.verify(
            census.Census((expected,)),
            census.Census((census.take_component("tiny", module),)),
        )
    assert caught.value.invariant == census.I1_TIES


# ── I2 QUANTIZER ────────────────────────────────────────────────────────────


def test_I2_red_arm_a_neutered_quantizer_is_a_census_mismatch_pgw1647(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skip `preprocess_model` and pgw#1638's 357-orphan shape reproduces.

    As a CENSUS mismatch and at $0, where the H200 reported it as `357
    tensor(s) name nothing in component 'text_encoder'` after a full 105 GB
    fetch. The refusal names the SWAP, not the missing scale tensors: the swap
    is the cause and the orphans are the symptom, and pgw#1626's post-mortem is
    that a refusal which names the symptom sends the reader to the checkpoint.
    """
    _component, expected_scales, swapped_class = QUANTIZED_FLEET[H3]
    expected = census.for_tree(_h3())

    monkeypatch.setattr(skeleton, "_prepare_quantized", lambda *a, **k: None)
    actual = census.for_tree(_h3())

    with pytest.raises(census.CensusMismatch) as caught:
        census.verify(expected, actual, where=H3)
    assert caught.value.invariant == census.I2_QUANTIZER
    assert caught.value.component == H3_COMPONENT
    assert swapped_class in str(caught.value), str(caught.value)

    # ...and the orphan arithmetic is the incident's, measured on the neutered
    # build rather than asserted from memory.
    plain = actual.by_component()[H3_COMPONENT]
    orphans = set(expected.by_component()[H3_COMPONENT].names) - set(plain.names)
    assert len(orphans) == expected_scales, len(orphans)
    assert all(name.endswith(f".{SCALE_LEAF}") for name in orphans)


# ── I3 SERVE MODE ───────────────────────────────────────────────────────────


def test_I3_red_arm_a_neutered_eval_is_a_census_mismatch_pgw1647(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skip `model.eval()` and the census says so, for every fleet component.

    44/44 weight-bearing components on the fleet were served in TRAIN mode once,
    five of them carrying a live `Dropout(p=0.1)` — every T5/UMT5 conditioner —
    randomizing their conditioning on every request with no error anywhere.
    """
    endpoint = "flux.1-dev"
    expected = census.for_tree(_tree(endpoint))

    monkeypatch.setattr(torch.nn.Module, "eval", lambda self: self)
    actual = census.for_tree(_tree(endpoint))
    assert any(not row.eval_mode for row in actual.components), (
        "with `eval` neutered every component still reports eval mode — this "
        "arm no longer measures the step it claims to"
    )

    with pytest.raises(census.CensusMismatch) as caught:
        census.verify(expected, actual, where=endpoint)
    assert caught.value.invariant == census.I3_SERVE_MODE
    assert "Dropout" in str(caught.value)


def test_I3_the_quantizer_s_replacement_modules_are_in_eval_too_pgw1647() -> None:
    """The eval step runs AFTER the swap, or it misses everything the
    quantizer built — a replacement module is constructed in train mode like
    any other, so the ORDER in `PREPARE_STEPS` is the assertion."""
    row = census.for_tree(_h3()).by_component()[H3_COMPONENT]
    assert row.eval_mode is True
    module = skeleton.build_modules(_h3()).modules[H3_COMPONENT]
    swapped = [
        sub for prefix, sub in module.named_modules()
        if type(sub).__name__ == QUANTIZED_FLEET[H3][2]
    ]
    assert swapped
    assert all(not sub.training for sub in swapped)


# ── I4 PLACEMENT ────────────────────────────────────────────────────────────


def test_I4_the_rope_buffers_come_off_the_meta_build_on_the_HOST_pgw1647() -> None:
    """The measurement pgw#1644 was root-caused with, banked as a test.

    `init_empty_weights()` redirects PARAMETERS to meta and mentions neither
    `include_buffers` nor `register_buffer`, so a RoPE module's `torch.arange`
    materializes on the default device. This is the input the sweep exists for;
    if it ever stops being true, the arm below stops proving anything.
    """
    module = skeleton.build_modules(_h3()).modules[H3_COMPONENT]
    off_meta = sorted(
        name
        for name, tensor in list(module.named_parameters(remove_duplicate=False))
        + list(module.named_buffers(remove_duplicate=False))
        if tensor.device.type != "meta"
    )
    assert off_meta == sorted(ROPE_BUFFERS), off_meta


def _fill_onto(module: Any, device: str) -> Any:
    """Fill every meta parameter with a FAKE tensor on ``device``.

    This is what `StreamingLoader` does — it allocates each destination with
    `torch.empty(..., device=target)` and rebinds `_parameters[leaf]` — except
    that the destination is a fake tensor, so a CPU-only box can hold a module
    whose parameters are on `cuda:0` and whose computed buffers are not. That
    asymmetry IS pgw#1644, and it is the only way to witness it without a card.

    Returns the fake mode, which the caller keeps open for the sweep.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    aliases = _alias_map(module)
    with mode:
        for name, tensor in list(module.named_parameters(remove_duplicate=False)):
            if tensor.device.type != "meta" or name in aliases:
                continue
            parent = (
                module.get_submodule(name.rsplit(".", 1)[0])
                if "." in name else module
            )
            leaf = name.rsplit(".", 1)[-1]
            parent._parameters[leaf] = torch.nn.Parameter(
                torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device=device),
                requires_grad=False,
            )
        skeleton.retie(module)
    return mode


def test_I4_red_arm_a_neutered_sweep_leaves_inv_freq_off_target_pgw1647(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete the sweep and the fence names `inv_freq` — offline, at $0.

    Every parameter lands on the target the way the stream lands it; the three
    computed RoPE buffers do not, because no container names them. On the H200
    that surfaced as `RuntimeError: Expected all tensors to be on the same
    device, but got mat1 is on cpu` raised inside `diffusers`, naming an
    ACTIVATION and no tensor of ours — three attempts, deterministic, after a
    full 105 GB fetch and a clean load. The whole point of walking the module
    is that the refusal names the buffers instead.
    """
    built = skeleton.build_modules(_h3())
    module = built.modules[H3_COMPONENT]
    expected = built.census()
    mode = _fill_onto(module, "cuda")

    monkeypatch.setattr(census, "place", lambda module, target: 0)
    with mode:
        skeleton.finish(module, target="cuda")

    with pytest.raises(census.CensusMismatch) as caught:
        census.verify_placement(
            H3_COMPONENT, module, "cuda",
            expected.by_component()[H3_COMPONENT], where=H3,
        )
    assert caught.value.invariant == census.I4_PLACEMENT
    assert "inv_freq" in caught.value.tensor, caught.value.tensor
    message = str(caught.value)
    for name in ROPE_BUFFERS:
        assert name in message, message
    assert "3 tensor(s) are on a device other than cuda" in message, message


def test_I4_green_arm_the_sweep_lands_every_rope_buffer_pgw1647() -> None:
    """The same tree with the seam intact: the three buffers MOVE and land."""
    built = skeleton.build_modules(_h3())
    module = built.modules[H3_COMPONENT]
    expected = built.census()
    mode = _fill_onto(module, "cuda")

    with mode:
        moved = skeleton.finish(module, target="cuda")
    assert moved == len(ROPE_BUFFERS), (
        f"the sweep moved {moved} tensor(s); the three RoPE buffers are the "
        f"ONLY tensors the container never named, so this arm no longer "
        f"measures pgw#1644"
    )

    census.verify_placement(
        H3_COMPONENT, module, "cuda", expected.by_component()[H3_COMPONENT])
    assert census.on_meta(module) == ()
    for name in ROPE_BUFFERS:
        assert module.get_buffer(name).device.type == "cuda"


def test_I4_the_sweep_walks_a_module_that_no_pipeline_registry_exposes_pgw1647() -> None:
    """pgw#1644's second defect: the sweep was PRESENT and UNREACHABLE.

    It read `pipeline.components` and fell back to the pipeline object only when
    that was an `nn.Module`. `MiniMaxH3StreamingPipeline` is a
    `ModularPipeline`/`ConfigMixin` and is neither, so root discovery returned
    `[]` and the sweep was a silent no-op for every component. `census.place`
    takes a MODULE, so there is no root discovery left to get wrong — and this
    pins that: an object with an empty `components` and no `nn.Module` nature
    cannot make the sweep skip the module inside it.
    """

    class _ModularShaped:
        components: Dict[str, Any] = {}

        def __init__(self, module: torch.nn.Module) -> None:
            self.module = module

    inner = torch.nn.Module()
    inner.register_buffer("inv_freq", torch.arange(4), persistent=False)
    pipeline = _ModularShaped(inner)
    assert not isinstance(pipeline, torch.nn.Module)
    assert pipeline.components == {}

    # The old root discovery, reproduced verbatim, to show it finding nothing.
    roots = [
        m for m in (getattr(pipeline, "components", None) or {}).values()
        if isinstance(m, torch.nn.Module)
    ]
    if not roots and isinstance(pipeline, torch.nn.Module):
        roots = [pipeline]
    assert roots == [], "the regression arm no longer reproduces pgw#1644"

    # The module walk answers about the same module regardless.
    census.verify_placement("modular", pipeline.module, "cpu")


# ── I5 TOTALITY ─────────────────────────────────────────────────────────────


def test_I5_red_arm_a_SIXTH_family_member_is_refused_by_name_pgw1647(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The arm no per-symptom fix could have had.

    Simulate a `from_pretrained` side effect nobody has written down yet — a
    construction step that registers a tensor the census never declared — and
    the release refuses at $0 naming it, instead of the fleet discovering it on
    a card the way it discovered the first four.
    """
    tree = _tree("sd15")
    expected = census.for_tree(tree)

    original = skeleton._build_on_meta

    def _with_a_sixth_member(*args: Any, **kwargs: Any) -> Any:
        built, quantization = original(*args, **kwargs)
        built.register_buffer(
            "cozy_undeclared_side_effect", torch.zeros(3), persistent=False)
        return built, quantization

    monkeypatch.setattr(skeleton, "_build_on_meta", _with_a_sixth_member)
    actual = census.for_tree(tree)

    with pytest.raises(census.CensusMismatch) as caught:
        census.verify(expected, actual, where="sd15")
    assert caught.value.invariant == census.I5_TOTALITY
    assert caught.value.tensor == "cozy_undeclared_side_effect"
    assert "never declared" in str(caught.value)


def test_I5_is_asserted_in_BOTH_directions_pgw1647(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction: a name the census has and the module lacks.

    Set equality both ways is the difference between "the module has everything
    we asked for" and "the module IS what we said it is". Only the second is a
    statement about construction.
    """
    tree = _tree("sd15")

    original = skeleton._build_on_meta

    def _with_a_sixth_member(*args: Any, **kwargs: Any) -> Any:
        built, quantization = original(*args, **kwargs)
        built.register_buffer(
            "cozy_undeclared_side_effect", torch.zeros(3), persistent=False)
        return built, quantization

    monkeypatch.setattr(skeleton, "_build_on_meta", _with_a_sixth_member)
    rich = census.for_tree(tree)
    monkeypatch.undo()
    plain = census.for_tree(tree)

    with pytest.raises(census.CensusMismatch) as caught:
        census.verify(rich, plain, where="sd15")
    assert caught.value.invariant == census.I5_TOTALITY
    assert caught.value.tensor == "cozy_undeclared_side_effect"
    assert "has nothing there" in str(caught.value)


def test_I5_a_component_appearing_or_vanishing_is_refused_pgw1647() -> None:
    """Set equality covers the COMPONENT set too, both directions."""
    full = census.for_tree(_tree("sd15"))
    fewer = census.Census(full.components[:-1])

    with pytest.raises(census.CensusMismatch) as caught:
        census.verify(full, fewer)
    assert caught.value.invariant == census.I5_TOTALITY

    with pytest.raises(census.CensusMismatch) as caught:
        census.verify(fewer, full)
    assert caught.value.invariant == census.I5_TOTALITY


def test_I5_a_shape_that_moved_is_refused_pgw1647() -> None:
    """A name present on both sides is not the same tensor if it is a
    different shape — an image bump that rebuilds a class differently."""
    row = census.for_tree(_tree("sd15")).by_component()["vae"]
    victim = row.tensors[0]
    moved = census.ComponentCensus(
        component=row.component,
        module_class=row.module_class,
        tensors=(
            census.TensorRow(
                name=victim.name, kind=victim.kind,
                shape=(victim.shape[0] + 1, *victim.shape[1:]) if victim.shape else (1,),
                dtype=victim.dtype,
            ),
            *row.tensors[1:],
        ),
        ties=row.ties, quant_rule=row.quant_rule,
        quant_modules=row.quant_modules, eval_mode=row.eval_mode,
    )
    with pytest.raises(census.CensusMismatch) as caught:
        census.verify(census.Census((moved,)), census.Census((row,)))
    assert caught.value.invariant == census.I5_TOTALITY
    assert caught.value.tensor == victim.name


def test_I5_a_rule_owned_dtype_may_move_but_its_NAME_may_not_pgw1647() -> None:
    """`postprocess_model` legitimately rewrites a scale grid's dtype.

    A `scale_fmt="ue8m0"` tree turns F32 scale grids into the exponent dtype the
    kernels read, AFTER the fill. So a rule-owned row's dtype is recorded and
    not asserted — and its name and shape are asserted like every other row, or
    the exemption would be the hole pgw#1638 came through.
    """
    row = census.for_tree(_h3()).by_component()[H3_COMPONENT]
    scale = next(r for r in row.tensors if r.name.endswith(f".{SCALE_LEAF}"))
    assert scale.rule_owned

    def _swap(replacement: census.TensorRow) -> census.ComponentCensus:
        return census.ComponentCensus(
            component=row.component, module_class=row.module_class,
            tensors=tuple(
                replacement if r.name == scale.name else r for r in row.tensors
            ),
            ties=row.ties, quant_rule=row.quant_rule,
            quant_modules=row.quant_modules, eval_mode=row.eval_mode,
        )

    recast = _swap(
        census.TensorRow(
            name=scale.name, kind=scale.kind, shape=scale.shape,
            dtype="uint8", rule_owned=True,
        )
    )
    census.verify(census.Census((row,)), census.Census((recast,)))

    reshaped = _swap(
        census.TensorRow(
            name=scale.name, kind=scale.kind, shape=(scale.shape[0] + 1,),
            dtype=scale.dtype, rule_owned=True,
        )
    )
    with pytest.raises(census.CensusMismatch) as caught:
        census.verify(census.Census((row,)), census.Census((reshaped,)))
    assert caught.value.invariant == census.I5_TOTALITY


# ── one predicate, three moments ────────────────────────────────────────────


def test_the_serve_engine_and_the_suite_call_the_SAME_predicate_pgw1647() -> None:
    """Not "the same shape of check": the same function object.

    Three moments — release build, CPU-only CI, serve — and one copy of the
    predicate. Two copies would drift, and the drift would be invisible until a
    release the CI passed refused on a card.
    """
    import inspect

    from gen_worker.serving.streaming import engine

    source = inspect.getsource(engine.StreamingLoader.build)
    assert "_census.fence(" in source, (
        "the serving engine no longer calls the census fence; whatever it does "
        "instead is a second predicate"
    )
    assert engine._census is census


def test_the_prepare_seam_is_ENUMERATED_and_lives_in_one_module_pgw1647() -> None:
    """The source-level guard: no prepare step may be added anywhere else.

    "Someone added a construction step somewhere else" is not the history of
    this defect family — the history is that steps were MISSING. But the fix is
    only stable while there is exactly one place a step can be, because a second
    place is how a census stops describing what actually gets built. The steps
    are named in `skeleton.PREPARE_STEPS`; the calls that implement them may
    appear only in `skeleton.py` and `census.py`.
    """
    import ast

    root = Path(skeleton.__file__).parent
    owned = {"skeleton.py", "census.py"}
    forbidden = {"preprocess_model", "postprocess_model", "tie_weights"}
    offenders: List[str] = []
    for path in sorted(root.glob("*.py")):
        if path.name in owned:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in forbidden or (name == "eval" and isinstance(func, ast.Attribute)):
                offenders.append(f"{path.name}:{node.lineno} {name}()")
    assert not offenders, (
        f"a prepare step is called outside the seam: {offenders}. Every step "
        f"`from_pretrained` runs belongs in skeleton.py's `build`/`finish` "
        f"pair, in `PREPARE_STEPS` order, or the census stops describing what "
        f"gets built (pgw#1647)"
    )
    assert skeleton.PREPARE_STEPS[0] == "build on meta"
    assert skeleton.PREPARE_STEPS[-1] == "device sweep"
