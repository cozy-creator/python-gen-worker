"""pgw#1245: the DECLARED DECODE-SET — what this image can decode, derived at
image build, and the typed refusal when a bound variant is outside it.

The properties proven here are the ones a claim could otherwise fake:

  1. the set is DERIVED and DETERMINISTIC — two derivations of one image agree
     byte-for-byte, including across processes, and a changed declaration
     MOVES the digest (the instrument is shown going red);
  2. it is COMPLETE — every entry carries the dimensions its decoder reads, a
     declaration cannot be written by omission, and decode paths no registered
     contract covers are recorded instead of living in a source comment;
  3. it REFUSES, typed, naming the contract and the nearest declared one —
     and the refusal is proven by RUNNING the production load dispatch, not by
     inspecting a registry;
  4. removing a decoder's declaration removes the contract and turns the same
     load into that refusal.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterator

import pytest

from gen_worker.discovery.decode_set import (
    DERIVATION,
    REFUSAL_CONTRACT_UNDECLARED,
    REFUSAL_KEY_TOPOLOGY_UNSUPPORTED,
    ContractNotDecodableError,
    DecodeSet,
    DecodeSetDriftError,
    KeyTopologyUnclassifiedError,
    KeyTopologyUnsupportedError,
    accepted_key_topologies,
    assert_matches_baked,
    derive_decode_set,
    manifest_block,
    nearest_declared,
    require_decodable,
)
from gen_worker.models.key_topology import classify_snapshot, identify_keys
from gen_worker.models.tensor_layout_contract import (
    CONTRACT_COZY_FP8_ROWWISE,
    CONTRACT_COZY_SVDQ_NVFP4_LR8,
    CONTRACT_HF_FP8_BLOCKWISE,
    CONTRACT_NUNCHAKU_V1,
    CONTRACT_PLAIN_BF16,
    DecodeDimensions,
    implements_contract,
)

# The contracts this image's decoders implement. Enumerated, not counted: the
# point of the mechanism is that `cozy.fp8-rowwise@1` and `hf.fp8-blockwise@1`
# are two contracts and not one "fp8" concept.
EXPECTED_CONTRACTS = {
    CONTRACT_COZY_FP8_ROWWISE,
    CONTRACT_COZY_SVDQ_NVFP4_LR8,
    CONTRACT_HF_FP8_BLOCKWISE,
    CONTRACT_NUNCHAKU_V1,
    CONTRACT_PLAIN_BF16,
}

_FAKE_DECODER = '''
from gen_worker.models.tensor_layout_contract import (
    DecodeDimensions, implements_contract,
)

@implements_contract(
    contract="{contract}", serves=("{body}",), composes_lora=False,
    decodes=DecodeDimensions(
        elements=("{element}",), scales=("{scale}",), key_topologies=(), bakes=()),
    why="fake decoder",
)
def decode(tensors):
    return tensors
'''


def _fake_image(root: Path, modules: dict[str, str]) -> str:
    """A package standing in for one image's decoder set."""
    pkg = root / "decode_set_fixture"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", "utf-8")
    for name, body in modules.items():
        (pkg / f"{name}.py").write_text(textwrap.dedent(body), "utf-8")
    for name in [n for n in sys.modules if n.startswith("decode_set_fixture")]:
        del sys.modules[name]
    return "decode_set_fixture"


@pytest.fixture
def fake_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in [n for n in sys.modules if n.startswith("decode_set_fixture")]:
        del sys.modules[name]


# ---------------------------------------------------------------------------
# 1. derived, and deterministic
# ---------------------------------------------------------------------------


def test_two_derivations_of_one_image_are_identical() -> None:
    first, second = derive_decode_set(), derive_decode_set()

    assert first == second
    assert first.digest == second.digest
    assert first.derivation == DERIVATION
    assert len(first.digest) == 64


def test_a_second_process_derives_the_same_digest() -> None:
    """Built twice = identical. In-process caching cannot prove this; a fresh
    interpreter with its own import order can."""
    out = subprocess.run(
        [sys.executable, "-c",
         "from gen_worker.discovery.decode_set import derive_decode_set;"
         "print(derive_decode_set().digest)"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == derive_decode_set().digest


def test_a_changed_declaration_moves_the_digest(fake_image: Path) -> None:
    """The instrument goes red: a digest that never moves proves nothing."""
    pkg = _fake_image(fake_image, {"d": _FAKE_DECODER.format(
        contract=CONTRACT_NUNCHAKU_V1, body="svdq-fp4-w4a4",
        element="nvfp4", scale="group_16")})
    before = derive_decode_set(packages=(pkg,))

    # SAME contract, SAME decoder, one dimension changed.
    (fake_image / "decode_set_fixture" / "d.py").write_text(
        textwrap.dedent(_FAKE_DECODER.format(
            contract=CONTRACT_NUNCHAKU_V1, body="svdq-fp4-w4a4",
            element="int4", scale="group_16")), "utf-8")
    for name in [n for n in sys.modules if n.startswith("decode_set_fixture")]:
        del sys.modules[name]
    after = derive_decode_set(packages=(pkg,))

    assert before.contracts() == after.contracts()  # the handle did NOT move
    assert before.digest != after.digest            # what it decodes did


def test_drift_between_the_baked_block_and_this_process_fails_closed() -> None:
    live = derive_decode_set()
    assert_matches_baked(manifest_block(live), live)  # agreement is silent

    # A lock claiming a contract this process cannot derive: the shape of a
    # decoder whose dependency is present at build and absent in the image.
    stale = manifest_block(live)
    stale["digest"] = "0" * 64
    stale["contracts"] = stale["contracts"] + [
        {"contract": "bfl.nvfp4-preswizzled@1", "decoder": "gone:decode"}]
    with pytest.raises(DecodeSetDriftError) as excinfo:
        assert_matches_baked(stale, live)
    err = excinfo.value
    assert err.code == "decode_set_drift"
    assert err.lost == ("bfl.nvfp4-preswizzled@1",)
    assert err.gained == ()
    assert "decode-set drift" in str(err)
    assert "Rebuild the image" in str(err)

    # An image that predates the block is UNPROVEN, not divergent.
    assert_matches_baked({}, live)


# ---------------------------------------------------------------------------
# 2. complete
# ---------------------------------------------------------------------------


def test_the_image_declares_exactly_these_contracts() -> None:
    ds = derive_decode_set()

    assert set(ds.contracts()) == EXPECTED_CONTRACTS
    # Registered but decoded by NOTHING here — the set is what ships, not what
    # the platform knows about (te#151: conflating the two nvfp4 layouts
    # measures LPIPS 1.11).
    assert "bfl.nvfp4-preswizzled@1" not in ds.contracts()
    assert ds.excluded_modules == ()


def test_every_entry_carries_the_dimensions_its_decoder_reads() -> None:
    ds = derive_decode_set()

    for entry in ds.entries:
        assert entry.decodes.elements, entry.contract
        assert entry.decodes.scales, entry.contract

    by_contract = {e.contract: e for e in ds.entries}
    # The two fp8 contracts differ on the axis that BRANCHES the decoder.
    rowwise = by_contract[CONTRACT_COZY_FP8_ROWWISE].decodes
    blockwise = by_contract[CONTRACT_HF_FP8_BLOCKWISE].decodes
    assert "per_channel_out" in rowwise.scales
    assert "block_128x128" not in rowwise.scales
    assert blockwise.scales == ("block_128x128",)
    # The svdq decoders CONSTRAIN no key topology, and that is a statement:
    # the nunchaku descriptor fixes the keys, so the fact lives on the handle
    # (th#1937 declined a `contract.native` synonym — one home per value).
    assert by_contract[CONTRACT_NUNCHAKU_V1].decodes.key_topologies == ()
    assert by_contract[CONTRACT_COZY_SVDQ_NVFP4_LR8].decodes.bakes == (
        "low_rank_branch",)
    # A dense decoder states `none`, which is a fact; silence would not be.
    assert by_contract[CONTRACT_PLAIN_BF16].decodes.scales == ("none",)


def test_a_declaration_cannot_be_written_by_omission() -> None:
    with pytest.raises(TypeError):
        @implements_contract(  # type: ignore[call-arg]
            contract=CONTRACT_PLAIN_BF16, serves=("bf16-w16a16",),
            composes_lora=False,
        )
        def _no_dimensions(x):
            return x

    with pytest.raises(ValueError, match="decodes.elements is empty"):
        @implements_contract(
            contract=CONTRACT_PLAIN_BF16, serves=("bf16-w16a16",),
            composes_lora=False,
            decodes=DecodeDimensions(
                elements=(), scales=("none",), key_topologies=(),
                bakes=()),
        )
        def _empty_axis(x):
            return x

    with pytest.raises(ValueError, match="is not registered"):
        @implements_contract(
            contract=CONTRACT_PLAIN_BF16, serves=("bf16-w16a16",),
            composes_lora=False,
            decodes=DecodeDimensions(
                elements=("fp6_e3m2",), scales=("none",),
                key_topologies=(), bakes=()),
        )
        def _invented_token(x):
            return x


def test_decode_paths_no_contract_covers_are_recorded_not_commented() -> None:
    ds = derive_decode_set()

    decoders = {u.decoder: u.reason for u in ds.unregistered}
    assert "gen_worker.models.w4a4:load_w4a4_denoiser" in decoders
    assert "gen_worker.models.loading:load_gguf_pipeline" in decoders
    assert "bfl.nvfp4-preswizzled@1" in decoders[
        "gen_worker.models.w4a4:load_w4a4_denoiser"]
    # Recorded is not declared: they satisfy no intersection.
    for decoder in decoders:
        assert decoder not in ds.contracts()


def test_the_manifest_block_carries_the_set_and_its_digest() -> None:
    ds = derive_decode_set()
    block = manifest_block(ds)

    assert block["derivation"] == DERIVATION
    assert block["digest"] == ds.digest
    assert {c["contract"] for c in block["contracts"]} == EXPECTED_CONTRACTS
    assert "shards" not in block["contracts"][0]
    rowwise = next(c for c in block["contracts"]
                   if c["contract"] == CONTRACT_COZY_FP8_ROWWISE)
    assert rowwise["decoder"] == "gen_worker.models.w8a8:load_w8a8_denoiser"
    assert set(rowwise["elements"]) == {"fp8_e4m3", "bf16"}
    assert {u["decoder"] for u in block["unregistered"]}


def test_the_image_build_stamps_the_block_into_endpoint_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`python -m gen_worker.discovery` is what the Dockerfile runs; this is
    the same entry point, so the set is a property of the BUILT IMAGE."""
    from gen_worker.discovery.discover import discover_manifest

    (tmp_path / "pyproject.toml").write_text(textwrap.dedent("""
        [project]
        name = "ep1245"

        [tool.gen_worker]
        main = "ep1245.main"
    """))
    src = tmp_path / "ep1245"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text(textwrap.dedent("""
        import msgspec
        from gen_worker import RequestContext, Resources, Slot, endpoint

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str = ""

        class Pipe:
            pass

        @endpoint(
            models={"pipeline": Slot(Pipe)},
            resources=Resources(gpu=True),
        )
        class Gen:
            def setup(self, pipeline: Pipe) -> None: ...

            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_()
    """))
    monkeypatch.syspath_prepend(str(tmp_path))

    manifest = discover_manifest(tmp_path)

    block = manifest["decode_set"]
    assert block["derivation"] == DERIVATION
    assert block["digest"]
    assert {c["contract"] for c in block["contracts"]} == EXPECTED_CONTRACTS
    # ONE census feeds both blocks: the lane block's contracts are the decode
    # set's, never a second walk that could disagree.
    assert {c["contract"] for c in manifest["execution_lanes"]["contracts"]} \
        == EXPECTED_CONTRACTS


# ---------------------------------------------------------------------------
# 3. the refusal
# ---------------------------------------------------------------------------


def test_the_refusal_names_the_contract_and_the_nearest_declared() -> None:
    ds = derive_decode_set()

    with pytest.raises(ContractNotDecodableError) as excinfo:
        require_decodable("bfl.nvfp4-preswizzled@1", decode_set=ds,
                          where="/snapshots/abc")
    err = excinfo.value
    assert err.code == REFUSAL_CONTRACT_UNDECLARED
    assert err.contract == "bfl.nvfp4-preswizzled@1"
    # Nearest by FORMAT, not by string: both are nvfp4, and an fp8 handle is
    # a closer string than it is a closer artifact.
    assert err.nearest == CONTRACT_COZY_SVDQ_NVFP4_LR8
    assert "bfl.nvfp4-preswizzled@1" in str(err)
    assert CONTRACT_COZY_SVDQ_NVFP4_LR8 in str(err)
    # The remedy is on either side, so the refusal says so.
    assert "ship an image whose decoder declares this one" in str(err)


@pytest.mark.parametrize("asked,nearest", [
    ("cozy.fp8-rowwise@2", CONTRACT_COZY_FP8_ROWWISE),   # wrong major
    ("hf.fp8-blockwise@2", CONTRACT_HF_FP8_BLOCKWISE),
    ("bfl.nvfp4-preswizzled@1", CONTRACT_COZY_SVDQ_NVFP4_LR8),
])
def test_nearest_declared_answers_by_format(asked: str, nearest: str) -> None:
    assert nearest_declared(asked, derive_decode_set()) == nearest


def test_an_image_that_declares_nothing_says_so() -> None:
    empty = DecodeSet(derivation=DERIVATION, entries=(), unregistered=(),
                      excluded_modules=())

    with pytest.raises(ContractNotDecodableError) as excinfo:
        require_decodable(CONTRACT_PLAIN_BF16, decode_set=empty)
    assert excinfo.value.nearest == ""
    assert "declares NO contract at all" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. RUNNING the production dispatch — a registered decoder is not a running one
# ---------------------------------------------------------------------------


def _safetensors(path: Path, tensors: dict[str, str]) -> None:
    """A header-only safetensors file. The w8a8 detector reads headers and
    nothing else, so this is the real artifact as far as it is concerned."""
    header: dict[str, object] = {}
    offset = 0
    for name, dtype in tensors.items():
        header[name] = {"dtype": dtype, "shape": [1],
                        "data_offsets": [offset, offset + 4]}
        offset += 4
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


@pytest.fixture
def fp8_rowwise_tree(tmp_path: Path) -> Path:
    """A diffusers tree whose denoiser is `cozy.fp8-rowwise@1` bytes."""
    root = tmp_path / "snapshot"
    (root / "transformer").mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "FakePipeline",
        "transformer": ["diffusers", "FakeTransformer"],
    }))
    _safetensors(root / "transformer" / "model.safetensors", {
        "transformer_blocks.0.attn.to_q.weight": "F8_E4M3",
        "transformer_blocks.0.attn.to_q.weight_scale": "F32",
    })
    return root


class _FakeCls:
    @classmethod
    def from_pretrained(cls, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise AssertionError("the guard should have refused first")


def test_the_load_dispatch_refuses_bytes_the_image_cannot_decode(
    fp8_rowwise_tree: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The execution assertion. A decode-set WITHOUT `cozy.fp8-rowwise@1` and
    a real call into the production contract dispatch — not a registry read."""
    from gen_worker.discovery import decode_set as ds_mod
    from gen_worker.models import loading

    full = derive_decode_set()
    without = ds_mod.DecodeSet(
        derivation=full.derivation,
        entries=tuple(e for e in full.entries
                      if e.contract != CONTRACT_COZY_FP8_ROWWISE),
        unregistered=full.unregistered,
        excluded_modules=full.excluded_modules,
        digest=full.digest,
    )
    monkeypatch.setattr(ds_mod, "runtime_decode_set", lambda: without)

    with pytest.raises(ContractNotDecodableError) as excinfo:
        loading.contract_loaded_component(
            fp8_rowwise_tree, "transformer", cls=_FakeCls)

    err = excinfo.value
    assert err.contract == CONTRACT_COZY_FP8_ROWWISE
    assert str(fp8_rowwise_tree) in str(err)
    assert err.nearest == CONTRACT_HF_FP8_BLOCKWISE


def test_the_same_load_passes_the_guard_when_the_contract_is_declared(
    fp8_rowwise_tree: Path,
) -> None:
    """The positive control: a guard that refuses everything would pass the
    test above and be worthless."""
    with pytest.raises(Exception) as excinfo:
        loading_module().contract_loaded_component(
            fp8_rowwise_tree, "transformer", cls=_FakeCls)

    # It got PAST the guard and died further in, on the fake class.
    assert not isinstance(excinfo.value, ContractNotDecodableError)


def loading_module() -> Any:
    from gen_worker.models import loading

    return loading


def test_removing_a_declaration_removes_the_contract_and_refuses(
    fake_image: Path,
) -> None:
    """pgw#1245's acceptance: a decoder that stops declaring stops being in
    the set, and the same request then refuses, typed."""
    both = {
        "svdq": _FAKE_DECODER.format(
            contract=CONTRACT_NUNCHAKU_V1, body="svdq-fp4-w4a4",
            element="nvfp4", scale="group_16"),
        "fp8": _FAKE_DECODER.format(
            contract=CONTRACT_COZY_FP8_ROWWISE, body="fp8-w8a8-dynamic",
            element="fp8_e4m3", scale="per_channel_out"),
    }
    pkg = _fake_image(fake_image, both)
    before = derive_decode_set(packages=(pkg,))
    assert set(before.contracts()) == {
        CONTRACT_NUNCHAKU_V1, CONTRACT_COZY_FP8_ROWWISE}
    require_decodable(CONTRACT_NUNCHAKU_V1, decode_set=before)  # decodable

    # The decoder loses its declaration — the ONLY edit.
    (fake_image / "decode_set_fixture" / "svdq.py").write_text(
        "def decode(tensors):\n    return tensors\n", "utf-8")
    for name in [n for n in sys.modules if n.startswith("decode_set_fixture")]:
        del sys.modules[name]
    after = derive_decode_set(packages=(pkg,))

    assert set(after.contracts()) == {CONTRACT_COZY_FP8_ROWWISE}
    with pytest.raises(ContractNotDecodableError) as excinfo:
        require_decodable(CONTRACT_NUNCHAKU_V1, decode_set=after)
    assert excinfo.value.code == REFUSAL_CONTRACT_UNDECLARED
    assert excinfo.value.nearest == CONTRACT_COZY_FP8_ROWWISE


def test_a_decoder_that_cannot_import_is_excluded_with_its_reason(
    fake_image: Path,
) -> None:
    """An image whose kernel extension is missing must not CLAIM the lane —
    and must be able to say why it does not."""
    pkg = _fake_image(fake_image, {
        "fp8": _FAKE_DECODER.format(
            contract=CONTRACT_COZY_FP8_ROWWISE, body="fp8-w8a8-dynamic",
            element="fp8_e4m3", scale="per_channel_out"),
        "svdq": "import a_kernel_this_image_lacks  # noqa: F401\n" +
                _FAKE_DECODER.format(
                    contract=CONTRACT_NUNCHAKU_V1, body="svdq-fp4-w4a4",
                    element="nvfp4", scale="group_16"),
    })
    ds = derive_decode_set(packages=(pkg,))

    assert set(ds.contracts()) == {CONTRACT_COZY_FP8_ROWWISE}
    excluded = {m.module: m.reason for m in ds.excluded_modules}
    assert "decode_set_fixture.svdq" in excluded
    assert "a_kernel_this_image_lacks" in excluded["decode_set_fixture.svdq"]
    with pytest.raises(ContractNotDecodableError):
        require_decodable(CONTRACT_NUNCHAKU_V1, decode_set=ds)


# ---------------------------------------------------------------------------
# 5. KEY TOPOLOGY — the axis file topology and the quant contract cannot see
# ---------------------------------------------------------------------------
#
# Measured 2026-08-14: DiffSynth's MiniMaxH3DiT accepts the minimax-NATIVE key
# set (535 keys, fused `blocks.N.attn.qkv_proj`); every minimax-h3 artifact we
# hold is the DIFFUSERS repackaging (638 keys, split `to_q/to_k/to_v`) — ONE
# key in common. It failed as `Cannot detect the model type` from an
# md5-over-key:shape lookup, after a 71 GB fetch onto a rented 4xH100, with
# 126 green tests blind to it because none exercised the load path.

_NATIVE_KEYS = (
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.attn.out_proj.weight",
    "blocks.0.mlp.fc1.weight",
)
_DIFFUSERS_KEYS = (
    "transformer_blocks.0.attn.to_q.weight",
    "transformer_blocks.0.attn.to_k.weight",
    "transformer_blocks.0.attn.to_v.weight",
)


def test_the_two_h3_repackagings_classify_differently() -> None:
    assert identify_keys(_NATIVE_KEYS) == "native.fused-qkv@1"
    assert identify_keys(_DIFFUSERS_KEYS) == "diffusers.split-qkv@1"
    assert identify_keys(("model.layers.0.self_attn.q_proj.weight",)) \
        == "transformers.split-qkv@1"
    assert identify_keys(("encoder.layer.0.attention.self.query.weight",)) \
        == "transformers.split-qkv@1"
    # The block prefix varies by family and the projection split does not, so
    # flux's `single_transformer_blocks` classifies with everything else.
    assert identify_keys(
        ("single_transformer_blocks.0.attn.to_q.weight",)) \
        == "diffusers.split-qkv@1"
    assert identify_keys(("some.tensor.weight",)) == ""
    assert identify_keys(()) == ""


def test_the_classifier_reads_headers_only(tmp_path: Path) -> None:
    """It must answer BEFORE the fetch is worth anything, so it may not
    construct a model or read tensor data."""
    root = tmp_path / "snap"
    (root / "transformer").mkdir(parents=True)
    _safetensors(root / "transformer" / "model.safetensors",
                 {k: "BF16" for k in _NATIVE_KEYS})

    whole = classify_snapshot(root)
    assert whole.topology == "native.fused-qkv@1"
    assert whole.denoiser is True
    assert classify_snapshot(root, "transformer").topology == "native.fused-qkv@1"


def test_unknown_means_REFUSE_for_a_denoiser_and_NOT_APPLICABLE_elsewhere(
    tmp_path: Path,
) -> None:
    """The exact semantics of "unclassified", because the phrase is the one a
    later reader will get wrong.

    A DENOISER whose keys match nothing registered fails closed — the model
    class is chosen from the architecture, so a hopeful pass is te#185's
    second stop. A VAE's keys are not evaluated at all: no
    architecture-specific class is selected from them, and refusing there
    would refuse the whole fleet to catch nothing."""
    root = tmp_path / "snap"
    (root / "transformer").mkdir(parents=True)
    (root / "vae").mkdir()
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "FakePipeline",
        "transformer": ["diffusers", "FakeTransformer"],
    }))
    alien = {"blocks.0.attention.wqkv.weight": "BF16",
             "blocks.0.attention.wo.weight": "BF16"}
    _safetensors(root / "transformer" / "model.safetensors", alien)
    _safetensors(root / "vae" / "model.safetensors", alien)

    denoiser = classify_snapshot(root, "transformer")
    assert denoiser.topology == ""
    assert denoiser.unclassified_denoiser is True

    vae = classify_snapshot(root, "vae")
    assert vae.topology == ""
    assert vae.unclassified_denoiser is False   # the axis does not apply

    ds = derive_decode_set()
    with pytest.raises(KeyTopologyUnclassifiedError) as excinfo:
        require_decodable(CONTRACT_PLAIN_BF16, decode_set=ds,
                          where=str(root), keys=denoiser)
    err = excinfo.value
    assert err.code == "decode_set_key_topology_unclassified"
    assert "blocks.0.attention.wqkv.weight" in str(err)
    assert "native.fused-qkv@1" in str(err)   # what IS registered
    # The same bytes under a non-denoiser component pass, by design.
    require_decodable(CONTRACT_PLAIN_BF16, decode_set=ds, where=str(root),
                      keys=vae)


def test_no_decoder_here_ingests_the_native_key_set() -> None:
    """The honest state of this image, stated rather than assumed: every
    decoder reads a repackaging, so a native tree has nowhere to land."""
    ds = derive_decode_set()

    for contract in ds.contracts():
        assert "native.fused-qkv@1" not in accepted_key_topologies(contract, ds)
    assert accepted_key_topologies(CONTRACT_PLAIN_BF16, ds) == (
        "diffusers.split-qkv@1", "transformers.split-qkv@1")


def test_an_unclassifiable_denoiser_refuses_through_the_load_dispatch(
    tmp_path: Path,
) -> None:
    """The fail-closed half, through the production dispatch rather than
    through `require_decodable` directly."""
    from gen_worker.models import loading

    root = tmp_path / "alien"
    (root / "transformer").mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "FakePipeline",
        "transformer": ["diffusers", "FakeTransformer"],
    }))
    _safetensors(root / "transformer" / "model.safetensors", {
        "blocks.0.attention.wqkv.weight": "F8_E4M3",
        "blocks.0.attention.wqkv.weight_scale": "F32",
    })

    with pytest.raises(KeyTopologyUnclassifiedError) as excinfo:
        loading.contract_loaded_component(root, "transformer", cls=_FakeCls)
    assert excinfo.value.contract == CONTRACT_COZY_FP8_ROWWISE
    assert str(root) in str(excinfo.value)


def test_a_topology_mismatch_refuses_before_the_load(tmp_path: Path) -> None:
    """The coordinator's acceptance case. Right contract, right file topology,
    WRONG key convention — and it refuses by name instead of dying as an md5
    miss inside a detection helper."""
    from gen_worker.models import loading

    root = tmp_path / "native_snapshot"
    (root / "transformer").mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "FakePipeline",
        "transformer": ["diffusers", "FakeTransformer"],
    }))
    # `cozy.fp8-rowwise@1` bytes — a DECLARED contract — written in the
    # minimax-native key convention, which no decoder here ingests.
    _safetensors(root / "transformer" / "model.safetensors", {
        "blocks.0.attn.qkv_proj.weight": "F8_E4M3",
        "blocks.0.attn.qkv_proj.weight_scale": "F32",
    })

    with pytest.raises(KeyTopologyUnsupportedError) as excinfo:
        loading.contract_loaded_component(root, "transformer", cls=_FakeCls)

    err = excinfo.value
    assert err.code == REFUSAL_KEY_TOPOLOGY_UNSUPPORTED
    assert err.observed == "native.fused-qkv@1"
    assert err.accepted == ("diffusers.split-qkv@1", "transformers.split-qkv@1")
    assert "native.fused-qkv@1" in str(err)
    assert "the KEYS are addressed differently" in str(err)
    # And the contract check did NOT fire: the format was never the problem.
    assert not isinstance(err, ContractNotDecodableError)


def test_the_accepted_topology_passes_the_same_gate(
    fp8_rowwise_tree: Path,
) -> None:
    """Positive control: the diffusers repackaging of the same contract gets
    past both halves of the guard."""
    assert classify_snapshot(fp8_rowwise_tree).topology == "diffusers.split-qkv@1"

    with pytest.raises(Exception) as excinfo:
        loading_module().contract_loaded_component(
            fp8_rowwise_tree, "transformer", cls=_FakeCls)
    assert not isinstance(excinfo.value, KeyTopologyUnsupportedError)


def test_a_late_shard_still_classifies_the_denoiser(tmp_path: Path) -> None:
    """The fail-closed path may not be defeated by shard ORDER: a tree whose
    leading shards hold only embeddings must not read as unclassified."""
    root = tmp_path / "sharded"
    denoiser = root / "transformer"
    denoiser.mkdir(parents=True)
    for i in range(1, 12):
        _safetensors(denoiser / f"model-{i:05d}-of-00012.safetensors",
                     {f"embeddings.{i}.weight": "BF16"})
    _safetensors(denoiser / "model-00012-of-00012.safetensors",
                 {"transformer_blocks.0.attn.to_q.weight": "BF16"})

    keys = classify_snapshot(root, "transformer")
    assert keys.topology == "diffusers.split-qkv@1"
    assert keys.unclassified_denoiser is False


def test_a_tree_with_no_attention_substructure_is_out_of_the_axis(
    tmp_path: Path,
) -> None:
    """CI's counter-example, kept as a test.

    The corrupt-load quarantine fixture (`test_p2_residency_reconcile.py`) is a
    root-layout snapshot holding ONE tensor named `w`. It is a legal thing to
    hand a loader and it carries no attention convention to get wrong, so the
    key-topology axis is not about it. The first cut of the fail-closed rule
    refused it — the axis is scoped to attention substructure now, derived from
    the BYTES rather than from the component name."""
    root = tmp_path / "tiny"
    root.mkdir()
    _safetensors(root / "model.safetensors", {"w": "F32"})

    keys = classify_snapshot(root)
    assert keys.denoiser is True           # root layout IS the denoiser
    assert keys.saw_tensors is True
    assert keys.attention_shaped is False
    assert keys.unclassified_denoiser is False
    require_decodable(CONTRACT_PLAIN_BF16, decode_set=derive_decode_set(),
                      where=str(root), keys=keys)   # does not raise


def test_a_FOURTH_attention_spelling_still_refuses(tmp_path: Path) -> None:
    """The case the refusal exists for survives the scoping: a denoiser whose
    attention is spelled in a way no rule here has seen is the H3 class of
    failure, and it must not be handed to a model class hopefully."""
    root = tmp_path / "fourth"
    (root / "transformer").mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "FakePipeline",
        "transformer": ["diffusers", "FakeTransformer"],
    }))
    _safetensors(root / "transformer" / "model.safetensors", {
        "blocks.0.attention.wqkv.weight": "BF16",
        "blocks.0.attention.wo.weight": "BF16",
    })

    keys = classify_snapshot(root, "transformer")
    assert keys.topology == ""
    assert keys.attention_shaped is True
    assert keys.unclassified_denoiser is True
    with pytest.raises(KeyTopologyUnclassifiedError):
        require_decodable(CONTRACT_PLAIN_BF16, decode_set=derive_decode_set(),
                          where=str(root), keys=keys)
