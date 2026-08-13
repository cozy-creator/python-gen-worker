"""pgw#967 — a graph class belongs to a TARGET, not to the whole cell.

pgw#854 recorded that no fleet family compiles anything but its denoiser and
that the SDK's own ``("transformer", "vae.decode")`` default is exercised by
nobody. This module pins the reason, which is not policy: until now the
vocabulary could not EXPRESS a second target's coordinates.

``Compile`` scopes ``Input``, ``Arg`` and ``Fork`` by target but not
``classes``, and ``validate_contract`` required every class row to state every
declared dim and fork. So a family adding its VAE decoder or text encoders had
exactly two options, both wrong: declare one flat class table and have
``mint_plans`` hand all of it to every target (sdxl: 18 identical text-encoder
graphs under 18 entry names, each compiled and paid for), or declare dims the
text encoder cannot receive.

Red-verified against the pre-change tree:

- the sdxl-shaped three-target declaration raises ``DeclarationError`` at
  construction ("graph class #18 omits declared dim(s) ['H_lat', 'W_lat']"),
  so the entry-count assertions below could not even be reached;
- with the row-scope check removed but scoping honoured nowhere, every target
  gets 18 plans and the totals below fail.

The unscoped path is pinned too, because it is what protects every published
cell: a declaration that does not scope must serialise byte-identically, or
``Compile.contract_axes`` re-keys artifacts that did not change.
"""

from __future__ import annotations

import pytest

from gen_worker import aot_declaration
from gen_worker.api.decorators import Compile
from gen_worker.api.export_contract import (
    Arg,
    DeclarationError,
    Dim,
    Fork,
    GraphClass,
    Input,
)

_TEXT_LEN = 77
_VAE_SCALE = 8

#: sdxl's declared payload bucket table, trimmed to three rows — the
#: arithmetic under test is per-row, not per-table.
_ASPECT_ROWS: tuple[tuple[int, int], ...] = ((1216, 832), (1024, 1024), (832, 1216))


def _sdxl_shaped(scoped: bool) -> Compile:
    """An sdxl-shaped declaration over three targets with unrelated call
    contracts: the UNet (aspect x CFG), the VAE decoder (aspect only — CFG has
    collapsed into one latent by decode time) and a CLIP text encoder (neither
    — 77 tokens at batch 1, whatever the request asks for)."""
    unet_scope = ("unet",) if scoped else ()
    vae_scope = ("vae.decode",) if scoped else ()
    te_scope = ("text_encoder",) if scoped else ()

    classes: list[GraphClass] = []
    for w, h in _ASPECT_ROWS:
        for cfg, batch in ((True, 2), (False, 1)):
            classes.append(GraphClass(
                dims={"B": batch, "H_lat": h // _VAE_SCALE,
                      "W_lat": w // _VAE_SCALE, "T_txt": _TEXT_LEN},
                fork={"cfg": cfg}, targets=unet_scope))
    for w, h in _ASPECT_ROWS:
        classes.append(GraphClass(
            dims={"B_img": 1, "H_lat": h // _VAE_SCALE, "W_lat": w // _VAE_SCALE},
            targets=vae_scope))
    classes.append(GraphClass(dims={"B_txt": 1, "T_txt": _TEXT_LEN}, targets=te_scope))

    return Compile(
        family="sdxl-shaped",
        targets=("unet", "vae.decode", "text_encoder"),
        text_len=_TEXT_LEN,
        shapes=_ASPECT_ROWS,
        dims=(
            Dim("B", carried_by=(("sample", 0), ("encoder_hidden_states", 0))),
            Dim("H_lat", carried_by=(("sample", 2), ("z", 2)), multiple_of=8),
            Dim("W_lat", carried_by=(("sample", 3), ("z", 3)), multiple_of=8),
            Dim("T_txt", carried_by=(("encoder_hidden_states", 1),
                                     ("input_ids", 1))),
            Dim("B_img", carried_by=(("z", 0),)),
            Dim("B_txt", carried_by=(("input_ids", 0),)),
        ),
        forks=(Fork("cfg", served=(True, False), targets=("unet",),
                    why="CFG is ONE batch-2 forward; the decoder sees one latent"),),
        classes=tuple(classes),
        inputs=(
            Input("sample", shape=("B", 4, "H_lat", "W_lat"), targets=("unet",), dtype="model"),
            Input("timestep", shape=(), value=1.0, targets=("unet",), dtype="model"),
            Input("encoder_hidden_states", shape=("B", "T_txt", 2048),
                  targets=("unet",), dtype="model"),
            Input("z", shape=("B_img", 4, "H_lat", "W_lat"), targets=("vae.decode",), dtype="model"),
            Input("input_ids", shape=("B_txt", "T_txt"), dtype="int64",
                  targets=("text_encoder",)),
        ),
        args=(Arg("return_dict", False, targets=("unet", "vae.decode")),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    )


def test_each_target_mints_only_its_own_coordinates() -> None:
    """The count IS the finding: 6 + 3 + 1, not 10 + 10 + 10."""
    decl = _sdxl_shaped(scoped=True)

    assert len(aot_declaration.mint_plans(decl, "unet")) == 6
    assert len(aot_declaration.mint_plans(decl, "vae.decode")) == 3
    assert len(aot_declaration.mint_plans(decl, "text_encoder")) == 1

    plans = aot_declaration.cell_plans(decl)
    assert len(plans) == 10
    names = {aot_declaration.plan_entry_name(p) for p in plans}
    assert "text_encoder/B_txt=1,T_txt=77" in names
    assert "vae.decode/B_img=1,H_lat=128,W_lat=128" in names
    assert "unet/cfg=true/B=2,H_lat=128,T_txt=77,W_lat=128" in names
    # No target inherits another's fork: the decoder and the encoder carry no
    # `cfg=` segment at all, so neither pays for a second arm of a flag it
    # does not take.
    assert not [n for n in names if n.startswith(("vae.", "text_")) and "cfg=" in n]


def test_unscoped_declaration_is_unchanged_and_serialises_identically() -> None:
    """The no-re-key guarantee. A declaration that does not scope keeps the
    old rule (every row states every dim and fork) and emits no `targets` key,
    so `Compile.contract_axes` — which feeds the cell key's contract digest —
    is byte-identical to what it was before scoping existed."""
    row = GraphClass(dims={"B": 2, "T_txt": 77}, fork={"cfg": True})
    assert row.as_row() == {"dims": {"B": 2, "T_txt": 77}, "fork": {"cfg": True}}
    assert row.serves("anything")

    decl = Compile(
        family="single-target",
        targets=("unet",),
        text_len=77,
        shapes=((1024, 1024),),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 1}), GraphClass(dims={"B": 2})),
        inputs=(Input("sample", shape=("B", 4, 128, 128), dtype="model"),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    )
    assert "targets" not in decl.contract_axes()["classes"][0]
    assert len(aot_declaration.mint_plans(decl, "unet")) == 2


def test_a_scoped_row_may_not_carry_a_dim_its_target_cannot_receive() -> None:
    """The check that makes scoping a contract rather than a comment: the text
    encoder takes `input_ids` only, so a latent-spatial coordinate on its row
    describes a call it can never be handed."""
    with pytest.raises(DeclarationError, match=r"carries dim\(s\) \['H_lat'\]"):
        Compile(
            family="bad-scope",
            targets=("unet", "text_encoder"),
            text_len=77,
            shapes=((1024, 1024),),
            dims=(
                Dim("B", carried_by=(("sample", 0),)),
                Dim("H_lat", carried_by=(("sample", 2),), multiple_of=8),
                Dim("B_txt", carried_by=(("input_ids", 0),)),
            ),
            classes=(
                GraphClass(dims={"B": 2, "H_lat": 128}, targets=("unet",)),
                GraphClass(dims={"B_txt": 1, "H_lat": 128},
                           targets=("text_encoder",)),
            ),
            inputs=(
                Input("sample", shape=("B", 4, "H_lat", 128), targets=("unet",), dtype="model"),
                Input("input_ids", shape=("B_txt", 77), dtype="int64",
                      targets=("text_encoder",)),
            ),
            shape_strategy="static-rows",
            warm_changes_key=False,
        )


def test_a_target_with_no_class_row_is_refused_by_name() -> None:
    """A declared target nobody gave coordinates to is a declaration defect,
    not an empty plan list that silently drops the target from the cell."""
    decl = Compile(
        family="orphan-target",
        targets=("unet", "vae.decode"),
        text_len=77,
        shapes=((1024, 1024),),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}, targets=("unet",)),),
        inputs=(Input("sample", shape=("B", 4, 128, 128), targets=("unet",), dtype="model"),
                # vae.decode needs a row of its own, or the
                # DECLARATION is refused first and this row would stop
                # measuring the missing CLASS it is named for.
                Input("z", shape=("B", 4, 128, 128), targets=("vae.decode",),
                      dtype="model"),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    )
    with pytest.raises(aot_declaration.MintRefused, match="no graph class scoped"):
        aot_declaration.cell_plans(decl)


def test_a_class_row_may_not_name_an_undeclared_target() -> None:
    with pytest.raises(DeclarationError, match=r"names target\(s\) \['vae.decode'\]"):
        Compile(
            family="unknown-target",
            targets=("unet",),
            text_len=77,
            shapes=((1024, 1024),),
            dims=(Dim("B", carried_by=(("sample", 0),)),),
            classes=(GraphClass(dims={"B": 2}, targets=("vae.decode",)),),
            inputs=(Input("sample", shape=("B", 4, 128, 128), dtype="model"),),
            shape_strategy="static-rows",
            warm_changes_key=False,
        )


# ---------------------------------------------------------------------------
# One resolver. The mint and the arm must name the same callable.
# ---------------------------------------------------------------------------


def test_mint_and_arm_resolve_a_dotted_module_target_to_the_same_callable() -> None:
    """RED on the pre-change tree: ``aot_serve._target_owner`` partitioned on
    the FIRST dot, so ``vae.decoder`` resolved to ``(vae, "decoder")`` — the
    submodule ATTRIBUTE — while ``compile_cache._resolve_target`` resolved the
    same string to ``(decoder, "forward")``. Arming would have replaced a
    submodule with a function on a graph traced from a different callable.

    ``vae.decoder`` is not hypothetical: it is the only SDXL VAE target that is
    declarable at all. ``vae.decode`` is wrapped by diffusers'
    ``apply_forward_hook``, whose ``wrapper(self, *args, **kwargs)`` carries no
    ``functools.wraps``, so its signature is irrecoverable and the traced
    callable would bake away both the tiling dispatch and the accelerate-hook
    branch.
    """
    torch = pytest.importorskip("torch")
    from gen_worker import aot_serve
    from gen_worker.compile_cache import _resolve_target

    class _Decoder(torch.nn.Module):
        def forward(self, sample):  # type: ignore[no-untyped-def]
            return sample

    class _Vae(torch.nn.Module):
        def __init__(self):  # type: ignore[no-untyped-def]
            super().__init__()
            self.decoder = _Decoder()

        def decode(self, z):  # type: ignore[no-untyped-def]
            return self.decoder(z)

    class _Pipe:
        def __init__(self):  # type: ignore[no-untyped-def]
            self.vae = _Vae()
            self.unet = _Decoder()

    pipe = _Pipe()
    for target in ("unet", "vae.decode", "vae.decoder"):
        mint = _resolve_target(pipe, target)
        assert mint is not None
        arm_module, arm_attr = aot_serve._target_owner(pipe, target)
        assert (arm_module, arm_attr) == (mint[0], mint[1]), target
        assert getattr(arm_module, arm_attr) == mint[2]

    assert aot_serve._target_owner(pipe, "vae.decoder")[0] is pipe.vae.decoder


# ---------------------------------------------------------------------------
# The proposed sdxl declaration, exactly — proven constructible here because
# the endpoint pins `gen-worker==0.91.4` and cannot carry it until this ships.
# ---------------------------------------------------------------------------

#: mirrors sdxl/src/sdxl/main.py `_SDXL_ASPECT_RATIOS` (ie#345 payload buckets)
_SDXL_ASPECTS: tuple[tuple[int, int], ...] = (
    (1536, 640), (1344, 768), (1216, 832), (1152, 896), (1024, 1024),
    (896, 1152), (832, 1216), (768, 1344), (640, 1536),
)


def _sdxl_with_decoder() -> Compile:
    """sdxl's live declaration plus its VAE DECODER — the exact proposal.

    ``vae.decoder``, not ``vae.decode``:

    * ``AutoencoderKL.decode`` is wrapped by diffusers'
      ``apply_forward_hook``, whose ``wrapper(self, *args, **kwargs)`` carries
      no ``functools.wraps``. Its signature is irrecoverably ``(*args,
      **kwargs)``, so pgw#822's pre-flight name check — the gate that exists
      so a pod is never rented to learn a sentence — is VACUOUS on it.
    * That wrapper is also where ``self._hf_hook.pre_forward(self)`` is
      called, and where ``_decode`` dispatches to ``tiled_decode``. Exporting
      through it bakes both branches away at trace time.
    * ``vae.decoder`` is the raw ``Decoder`` module: a real named signature
      (``forward(sample, latent_embeds=None)``), a plain-Tensor return, no
      hook, no dispatch. Tiling still works — ``tiled_decode`` calls
      ``self.decoder`` per tile, at tile shapes the cell does not declare, so
      an offloaded pod misses the entry and degrades to eager observably
      instead of serving a graph that skipped its own onload.

    No TEXT ENCODER target. Under the pinned ``transformers>=5.13,<6``,
    ``CLIPTextModel.forward`` is ``(input_ids, attention_mask, position_ids,
    **kwargs)`` returning ``BaseModelOutputWithPooling``. The SDXL pipeline
    needs ``output_hidden_states=True`` (it reads ``hidden_states[-2]``),
    which is expressible only as a KEYWORD — against the all-positional mint
    obligation — and there is no ``return_dict`` parameter left to escape the
    dataclass output with. ``CLIPTextTransformer`` was flattened away in v5,
    so there is no inner named-signature module to target either. Two
    independent structural blockers, for the smallest of the three prizes.
    """
    classes: list[GraphClass] = []
    for w, h in _SDXL_ASPECTS:
        for cfg, batch in ((True, 2), (False, 1)):
            classes.append(GraphClass(
                dims={"B": batch, "H_lat": h // _VAE_SCALE,
                      "W_lat": w // _VAE_SCALE, "T_txt": _TEXT_LEN},
                fork={"cfg": cfg}, targets=("unet",)))
    for w, h in _SDXL_ASPECTS:
        # CFG has collapsed by decode time: one latent, whatever the guidance.
        classes.append(GraphClass(
            dims={"B": 1, "H_lat": h // _VAE_SCALE, "W_lat": w // _VAE_SCALE},
            targets=("vae.decoder",)))

    return Compile(
        family="sdxl",
        targets=("unet", "vae.decoder"),
        text_len=_TEXT_LEN,
        shapes=_SDXL_ASPECTS,
        dims=(
            Dim("B", carried_by=(("sample", 0), ("encoder_hidden_states", 0),
                                 ("added_cond_kwargs.text_embeds", 0),
                                 ("added_cond_kwargs.time_ids", 0))),
            Dim("H_lat", carried_by=(("sample", 2),), multiple_of=8),
            Dim("W_lat", carried_by=(("sample", 3),), multiple_of=8),
            Dim("T_txt", carried_by=(("encoder_hidden_states", 1),)),
        ),
        forks=(Fork("cfg", served=(True, False), targets=("unet",),
                    why="CFG is ONE batch-2 forward (ie#345); turbo pins the "
                        "no-CFG batch-1 graph. The DECODER never forks on it"),),
        classes=tuple(classes),
        inputs=(
            Input("sample", shape=("B", ("config", "in_channels"), "H_lat", "W_lat"),
                  targets=("unet",), dtype="model"),
            Input("timestep", shape=(), value=1.0, targets=("unet",), dtype="model"),
            Input("encoder_hidden_states",
                  shape=("B", "T_txt", ("config", "cross_attention_dim")),
                  targets=("unet",), dtype="model"),
            Input("added_cond_kwargs.text_embeds", shape=("B", 1280),
                  targets=("unet",), dtype="model"),
            Input("added_cond_kwargs.time_ids", shape=("B", 6), targets=("unet",), dtype="model"),
            # Decoder.forward(sample, latent_embeds=None); `latent_embeds` is
            # the temporal-decoder argument SDXL never passes.
            Input("sample", shape=("B", ("config", "in_channels"), "H_lat", "W_lat"),
                  targets=("vae.decoder",), dtype="model"),
        ),
        # `Decoder.forward` has no `return_dict` parameter — the UNet's escape
        # is the UNet's alone.
        args=(Arg("return_dict", False, targets=("unet",)),),
        shape_strategy="static-rows",
        warm_changes_key=False,
        numerics_floor=0.995,
        numerics_warn=0.999,
    )


def test_the_proposed_sdxl_declaration_costs_nine_entries_not_eighteen() -> None:
    """A4's bill, exactly: the cell goes 36 -> 45 entries, +25%.

    36 = 18 UNet classes x 2 adapter arms (pgw#790's branchless/lora64 fork,
    synthesized by the SDK). The decoder is not lift-capable, so it forks into
    nothing and contributes its 9 rows once. Without row scoping it would
    contribute 18 — nine of them duplicate graphs under distinct entry names,
    each one a full compile paid for a second time.
    """
    decl = _sdxl_with_decoder()
    unet = aot_declaration.mint_plans(decl, "unet")
    dec = aot_declaration.mint_plans(decl, "vae.decoder")
    assert len(unet) == 18
    assert len(dec) == 9
    assert all(p.dynamic == () for p in (*unet, *dec))
    assert len(aot_declaration.cell_plans(decl)) == 27

    names = {aot_declaration.plan_entry_name(p) for p in dec}
    assert names == {
        f"vae.decoder/B=1,H_lat={h // 8},W_lat={w // 8}"
        for w, h in _SDXL_ASPECTS
    }
    # Every declared aspect the payload enum admits, and nothing else: the
    # decoder's coordinate set IS the shape bucket table.
    assert len(names) == len(_SDXL_ASPECTS)
