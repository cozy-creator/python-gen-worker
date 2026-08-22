"""pgw#1670 — the diffusers repack as a KEY-LEVEL leg of the conversion job.

se#840's mirror published an honest flat bf16 tree that the endpoint cannot
load: the hollow session intercepts only diffusers/transformers loaders, so a
component class has to be reachable through a ``model_index.json``, and a flat
root beside a ``config.json`` whose ``auto_map`` names four ``.py`` files that
do not exist is not that. The ruling put the repack on the conversion leg, and
the repack that already existed could not be it — ``singlefile_to_diffusers``
LOADS a declared diffusers pipeline class and saves it back, which needs a
class the SDK can name and a card's worth of RAM.

What these cases hold, and each is red when the thing it names is removed:

* the produced tree HAS the shape — component directories, the derived
  component config, the tokenizer split, ``model_index.json``;
* every tensor's BYTES survive the rename — digest per tensor, before and
  after, and the digest is of the tensor payload rather than of the file;
* an unknown family is a TYPED refusal that names what is declared, never a
  detection and never a pass-through;
* keys that no component claims REFUSE rather than vanish;
* a requested repack cannot be satisfied by publishing the source unchanged,
  which is the substitution shape that already cost this checkpoint two jobs.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fake_hub import _FakeHub

from gen_worker.convert.clone import (
    CAST_OUTPUT,
    NOT_POSSIBLE,
    PUBLISH_SOURCE,
    OutputSpec,
    normalize_outputs,
    run_clone,
    spec_actions,
)
from gen_worker.convert.ingest import IngestedSource, detect_snapshot_dtype
from gen_worker.convert.tree_repack import (
    apply_tree_repack,
    registered_tree_repacks,
    require_tree_repack,
)
from gen_worker.convert.tree_repack_spec import (
    ComponentConfig,
    ConfigField,
    DeclarationError,
    FileRoute,
    RepackComponent,
    TreeRepack,
    TreeRepackError,
)
from gen_worker.hubio.client import files_from_tree
from gen_worker.models.file_layout import MULTI_FILE

_WIDTH = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "I64": 8}


# ------------------------------------------------------------------ fixtures


def _safetensors(tensors: dict[str, tuple[str, int]]) -> bytes:
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, count) in tensors.items():
        end = offset + count * _WIDTH[dtype]
        header[name] = {"dtype": dtype, "shape": [count], "data_offsets": [offset, end]}
        offset = end
    blob = json.dumps(header).encode()
    body = bytes(bytearray((i * 37 + 11) % 251 for i in range(offset)))
    return struct.pack("<Q", len(blob)) + blob + body


def _tensor_digests(root: Path) -> dict[str, str]:
    """``{tensor name: sha256 of its PAYLOAD}`` over a whole tree.

    Of the payload and not of the file, because that is the claim under test:
    a repack renames tensors and moves files, and neither may touch a byte of
    any tensor.
    """

    out: dict[str, str] = {}
    for path in sorted(Path(root).rglob("*.safetensors")):
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
            data_start = 8 + n
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                start, end = info["data_offsets"]
                f.seek(data_start + start)
                payload = f.read(end - start)
                assert len(payload) == end - start
                key = f"{name}|{info['dtype']}|{tuple(info['shape'])}"
                assert key not in out, f"{name} appears twice in {root}"
                out[key] = hashlib.sha256(payload).hexdigest()
    return out


#: se#840's real key profile, at fixture scale: the MoT split is KEYS, the
#: understanding stream is `language_model.*`, the pixel tokenizers are
#: `vision_model.*` and `fm_modules.vision_model_mot_gen.*`.
def _sensenova_keys(n_layers: int = 3) -> dict[str, tuple[str, int]]:
    keys: dict[str, tuple[str, int]] = {
        "language_model.model.embed_tokens.weight": ("BF16", 1024),
        "language_model.lm_head.weight": ("BF16", 1024),
        "language_model.model.norm.weight": ("BF16", 32),
        "language_model.model.norm_mot_gen.weight": ("BF16", 32),
        "vision_model.embeddings.patch_embedding.weight": ("BF16", 96),
        "vision_model.embeddings.patch_embedding.bias": ("BF16", 8),
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight": ("BF16", 96),
        "fm_modules.timestep_embedder.mlp.0.weight": ("BF16", 128),
        "fm_modules.fm_head.conv1.weight": ("BF16", 256),
    }
    for i in range(n_layers):
        keys[f"language_model.model.layers.{i}.self_attn.q_proj.weight"] = ("BF16", 512)
        keys[f"language_model.model.layers.{i}.self_attn.q_proj_mot_gen.weight"] = ("F32", 512)
        keys[f"language_model.model.layers.{i}.mlp.gate_proj.weight"] = ("BF16", 512)
        keys[f"language_model.model.layers.{i}.mlp_mot_gen.gate_proj.weight"] = ("F32", 512)
    return keys


_UPSTREAM_CONFIG: dict[str, Any] = {
    "architectures": ["NEOChatModel"],
    "auto_map": {"AutoConfig": "configuration_neo_chat.NEOChatConfig"},
    "llm_config": {
        "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 42,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "rms_norm_eps": 1e-06, "rope_theta": 5000000.0, "rope_theta_hw": 10000.0,
        "vocab_size": 151936, "attention_bias": False,
    },
    "vision_config": {
        "hidden_size": 1024, "llm_hidden_size": 4096, "patch_size": 16,
        "num_channels": 3, "downsample_ratio": 0.5, "rope_theta_vision": 10000.0,
        "max_position_embeddings_vision": 10000,
    },
    "patch_size": 16, "downsample_ratio": 0.5, "t_eps": 0.05, "noise_scale": 1.0,
    "noise_scale_mode": "resolution", "noise_scale_base_image_seq_len": 64,
    "noise_scale_max_value": 16.0, "add_noise_scale_embedding": True,
    "timestep_shift": 1.0, "time_schedule": "standard",
}


def _flat_tree(root: Path, *, shards: int = 1, keys: dict[str, tuple[str, int]] | None = None) -> Path:
    """The shape a mirror produces: weights at the root beside the HF documents."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(_UPSTREAM_CONFIG))
    (root / "vocab.json").write_text('{"a": 0}')
    (root / "merges.txt").write_text("#version: 0.2\n")
    (root / "added_tokens.json").write_text('{"<IMG_CONTEXT>": 151667}')
    (root / "special_tokens_map.json").write_text('{"eos_token": "<|im_end|>"}')
    (root / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|im_end|>",
        "model_max_length": 12288,
    }))
    (root / "README.md").write_text("# upstream card\n")

    items = list((keys if keys is not None else _sensenova_keys()).items())
    if shards == 1:
        (root / "model.safetensors").write_bytes(_safetensors(dict(items)))
        return root
    weight_map: dict[str, str] = {}
    per = (len(items) + shards - 1) // shards
    for i in range(shards):
        chunk = dict(items[i * per:(i + 1) * per])
        assert chunk, "fixture asked for more shards than it has tensors"
        name = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        (root / name).write_bytes(_safetensors(chunk))
        for key in chunk:
            weight_map[key] = name
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0}, "weight_map": weight_map}))
    return root


# ------------------------------------------------------- the produced shape


def test_a_flat_tree_becomes_the_diffusers_shape_the_endpoint_declares(
    tmp_path: Path,
) -> None:
    """The whole deliverable, against se#840's own fixture shape.

    The endpoint's `tests/fixtures/checkpoint-configs/` is the oracle: it is
    the tree the first real `gen-worker lock --checkpoint` derive loaded to
    100% on both components, so it states what the repack must produce.
    """

    tree = _flat_tree(tmp_path / "flat")
    report = apply_tree_repack(tree, require_tree_repack("sensenova-u1.mot"))

    index = json.loads((tree / "model_index.json").read_text())
    assert index["_class_name"] == "SenseNovaU1Pipeline"
    assert index["transformer"] == ["sensenova_u1", "SenseNovaU1"]
    assert index["tokenizer"] == ["transformers", "Qwen2TokenizerFast"]

    assert (tree / "transformer" / "diffusion_pytorch_model.safetensors").is_file()
    config = json.loads((tree / "transformer" / "config.json").read_text())
    assert config["_class_name"] == "SenseNovaU1"
    # Derived from the source document, field by declared field — not copied,
    # and not defaulted.
    assert config["llm"]["num_hidden_layers"] == 42
    assert config["llm"]["rope_theta_hw"] == 10000.0
    assert config["vision"]["llm_hidden_size"] == 4096
    assert config["add_noise_scale_embedding"] is True
    assert "auto_map" not in config and "llm_config" not in config
    # The serving values that upstream's own config is WRONG about stay out of
    # it: `timestep_shift` 1.0 here against the 3.0 the reference serves.
    assert "timestep_shift" not in config

    tok = json.loads((tree / "tokenizer" / "tokenizer_config.json").read_text())
    assert tok["tokenizer_class"] == "Qwen2TokenizerFast", (
        "model_index says Fast and the tokenizer document must not say otherwise")
    assert tok["eos_token"] == "<|im_end|>", "the upstream document was replaced, not edited"
    for name in ("vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json"):
        assert (tree / "tokenizer" / name).is_file(), name
        assert not (tree / name).exists(), f"{name} was copied rather than moved"

    # The root's dead `auto_map` config is GONE: it is what makes the flat tree
    # unloadable, and leaving it would leave the tree ambiguous.
    assert not (tree / "config.json").exists()
    assert not (tree / "model.safetensors").exists()
    assert (tree / "README.md").is_file(), "keep_root is declared and must be honoured"

    assert report.file_layout == MULTI_FILE
    assert report.members == {"transformer": 1}
    assert report.rewritten_files == 0 and report.moved_files == 1, (
        "the identity key map must MOVE its members; a rewrite here would read "
        "and write 35 GB on the pod for nothing")


def test_the_shape_is_what_the_declaration_says_and_not_a_constant(
    tmp_path: Path,
) -> None:
    """The RED ARM for the case above: disable the mapping and it must fail.

    A declaration whose components are renamed produces a differently-shaped
    tree. If the engine were emitting a hardcoded `transformer/` + `tokenizer/`
    shape, this passes anyway — which is exactly the thing worth knowing.
    """

    spec = require_tree_repack("sensenova-u1.mot")
    renamed = TreeRepack(
        name="fixture-renamed",
        pipeline_class="OtherPipeline",
        requires_key_prefixes=spec.requires_key_prefixes,
        components=(
            RepackComponent(name="denoiser", library="lib_x", class_name="ClassX",
                            weight_stem="model", config=spec.components[0].config),
            RepackComponent(name="text_tokenizer", library="transformers",
                            class_name="Qwen2TokenizerFast",
                            files=spec.components[1].files),
        ),
        keep_root=spec.keep_root,
    )
    tree = _flat_tree(tmp_path / "flat")
    report = apply_tree_repack(tree, renamed)

    index = json.loads((tree / "model_index.json").read_text())
    assert index["_class_name"] == "OtherPipeline"
    assert set(index) == {"_class_name", "_diffusers_version", "denoiser", "text_tokenizer"}
    assert (tree / "denoiser" / "model.safetensors").is_file()
    assert (tree / "text_tokenizer" / "vocab.json").is_file()
    assert not (tree / "transformer").exists()
    assert report.members == {"denoiser": 1}


def test_a_sharded_source_keeps_its_members_and_gets_an_index(tmp_path: Path) -> None:
    """Members are PRESERVED — the repack neither shards nor de-shards.

    pgw#1669 is the live record of a produced layout being replaced without
    anyone saying so. This leg's answer is narrow and checkable: N members in,
    N members out, and the produced layout is read back off the tree.
    """

    tree = _flat_tree(tmp_path / "flat", shards=4)
    before = _tensor_digests(tree)
    report = apply_tree_repack(tree, require_tree_repack("sensenova-u1.mot"))

    comp = tree / "transformer"
    members = sorted(p.name for p in comp.glob("*.safetensors"))
    assert members == [
        f"diffusion_pytorch_model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]
    assert report.members == {"transformer": 4}
    assert report.moved_files == 4 and report.rewritten_files == 0

    index = json.loads((comp / "diffusion_pytorch_model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(_sensenova_keys())
    assert set(index["weight_map"].values()) == set(members)
    assert index["metadata"]["total_size"] == sum(
        (comp / m).stat().st_size for m in members)
    assert _tensor_digests(tree) == before


# ------------------------------------------------------- the bytes are safe


def test_every_tensor_is_byte_identical_under_the_rename(tmp_path: Path) -> None:
    """The load-bearing property: a repack is a rename, and a rename is free.

    Driven through a declaration that REALLY renames, because the SenseNova
    map is the identity and an identity map cannot falsify this.
    """

    spec = require_tree_repack("sensenova-u1.mot")
    from gen_worker.convert.repack_spec import RenameRule

    prefixed = TreeRepack(
        name="fixture-prefixed",
        pipeline_class="OtherPipeline",
        requires_key_prefixes=spec.requires_key_prefixes,
        components=(
            RepackComponent(
                name="transformer", library="lib_x", class_name="ClassX",
                weight_stem="diffusion_pytorch_model",
                rules=(RenameRule(kind="prefix", pairs=(("language_model.", "lm."),
                                                        ("fm_modules.", "fm."))),),
                config=spec.components[0].config,
            ),
            RepackComponent(name="tokenizer", library="transformers",
                            class_name="Qwen2TokenizerFast",
                            files=spec.components[1].files),
        ),
        keep_root=spec.keep_root,
    )

    tree = _flat_tree(tmp_path / "flat", shards=3)
    before = _tensor_digests(tree)
    report = apply_tree_repack(tree, prefixed)
    after = _tensor_digests(tree)

    assert report.rewritten_files == 3 and report.moved_files == 0, (
        "a renaming map cannot be satisfied by moving files")
    assert len(after) == len(before) == report.tensor_count

    renamed = {
        key.replace("language_model.", "lm.", 1).replace("fm_modules.", "fm.", 1): digest
        for key, digest in before.items()
    }
    assert after == renamed, "a tensor's bytes, dtype or shape moved under the rename"
    # And the names really did change, so the comparison above is not vacuous.
    assert any(k.startswith("lm.") for k in after)
    assert not any(k.startswith("language_model.") for k in after)


def test_a_member_split_across_components_copies_ranges_not_tensors(
    tmp_path: Path,
) -> None:
    """One source file, two components — the general case, byte-checked."""

    from gen_worker.convert.repack_spec import RenameRule

    split = TreeRepack(
        name="fixture-split",
        pipeline_class="SplitPipeline",
        components=(
            RepackComponent(
                name="vision", library="lib_x", class_name="Vision",
                key_prefixes=("vision_model.",), weight_stem="diffusion_pytorch_model",
                rules=(RenameRule(kind="prefix", pairs=(("vision_model.", ""),)),),
            ),
            RepackComponent(
                name="transformer", library="lib_x", class_name="ClassX",
                weight_stem="diffusion_pytorch_model",
            ),
        ),
    )
    tree = _flat_tree(tmp_path / "flat")
    before = _tensor_digests(tree)
    report = apply_tree_repack(tree, split)

    assert report.members == {"vision": 1, "transformer": 1}
    vision = _tensor_digests(tree / "vision")
    rest = _tensor_digests(tree / "transformer")
    assert len(vision) == 2 and set(vision) | set(rest) == {
        k.replace("vision_model.", "", 1) if k.startswith("vision_model.") else k
        for k in before
    }
    assert not (tree / "model.safetensors").exists(), (
        "the split source survived; it would be published twice")
    for key, digest in vision.items():
        assert before[f"vision_model.{key}"] == digest


# --------------------------------------------------------------- refusals


def test_an_unknown_family_refuses_and_names_what_is_declared() -> None:
    with pytest.raises(TreeRepackError) as exc:
        require_tree_repack("sensenova-u2.mot")
    assert "sensenova-u1.mot" in str(exc.value), (
        "a refusal that does not name the declared families is a dead end")
    assert "NAMED by the request" in str(exc.value)
    assert "sensenova-u1.mot" in registered_tree_repacks()


def test_an_unknown_family_is_refused_before_a_pod_exists() -> None:
    """`normalize_outputs` runs on the request, so a typo costs $0."""

    with pytest.raises(TreeRepackError):
        normalize_outputs([{"dtype": "bf16", "repack": "not-a-family"}])
    with pytest.raises(ValueError, match="file_type=safetensors"):
        normalize_outputs([{"dtype": "q4_k_m", "file_type": "gguf",
                            "repack": "sensenova-u1.mot"}])
    specs = normalize_outputs([
        {"dtype": "bf16", "file_layout": "multi-file", "repack": "sensenova-u1.mot"}])
    assert [s.repack for s in specs] == ["sensenova-u1.mot"]


def test_a_key_no_component_claims_refuses_instead_of_vanishing(tmp_path: Path) -> None:
    """The failure this schema exists to make impossible.

    A dropped tensor is undetectable downstream: the tree still loads, one
    weight short, and generates plausible garbage.
    """

    partial = TreeRepack(
        name="fixture-partial",
        pipeline_class="P",
        components=(
            RepackComponent(name="transformer", library="l", class_name="C",
                            key_prefixes=("language_model.",),
                            weight_stem="diffusion_pytorch_model"),
        ),
    )
    tree = _flat_tree(tmp_path / "flat")
    with pytest.raises(TreeRepackError) as exc:
        apply_tree_repack(tree, partial)
    assert "match no component" in str(exc.value)
    assert "vision_model.embeddings.patch_embedding.bias" in str(exc.value)
    assert (tree / "model.safetensors").is_file(), (
        "the refusal came after the tree had already been taken apart")


def test_a_wrong_family_refuses_on_its_declared_signature(tmp_path: Path) -> None:
    """A repack is NAMED, so the one guard left is: is this that tree?"""

    tree = _flat_tree(tmp_path / "flat", keys={"blocks.0.weight": ("BF16", 16)})
    with pytest.raises(TreeRepackError) as exc:
        apply_tree_repack(tree, require_tree_repack("sensenova-u1.mot"))
    assert "carries no key under" in str(exc.value)
    assert "language_model." in str(exc.value)


def test_a_tree_that_is_already_component_shaped_refuses(tmp_path: Path) -> None:
    tree = _flat_tree(tmp_path / "flat")
    apply_tree_repack(tree, require_tree_repack("sensenova-u1.mot"))
    with pytest.raises(TreeRepackError, match="already carries a model_index.json"):
        apply_tree_repack(tree, require_tree_repack("sensenova-u1.mot"))


def test_a_missing_declared_config_field_refuses(tmp_path: Path) -> None:
    tree = _flat_tree(tmp_path / "flat")
    doc = json.loads((tree / "config.json").read_text())
    del doc["llm_config"]["rope_theta_hw"]
    (tree / "config.json").write_text(json.dumps(doc))
    with pytest.raises(TreeRepackError) as exc:
        apply_tree_repack(tree, require_tree_repack("sensenova-u1.mot"))
    assert "rope_theta_hw" in str(exc.value)


def test_a_missing_declared_tokenizer_file_refuses(tmp_path: Path) -> None:
    tree = _flat_tree(tmp_path / "flat")
    (tree / "merges.txt").unlink()
    with pytest.raises(TreeRepackError, match="merges.txt"):
        apply_tree_repack(tree, require_tree_repack("sensenova-u1.mot"))


# ----------------------------------------------------- declaration refusals


def test_a_declaration_that_could_drop_keys_is_refused_at_declaration_time() -> None:
    def comp(name: str, prefixes: tuple[str, ...] = ()) -> RepackComponent:
        return RepackComponent(name=name, library="l", class_name="C",
                               weight_stem="w", key_prefixes=prefixes)

    with pytest.raises(DeclarationError, match="catch-all"):
        TreeRepack(name="two-catch-alls", pipeline_class="P",
                   components=(comp("a"), comp("b")))
    with pytest.raises(DeclarationError, match="LAST weight component"):
        TreeRepack(name="early-catch-all", pipeline_class="P",
                   components=(comp("a"), comp("b", ("x.",))))
    with pytest.raises(DeclarationError, match="routes no weights"):
        TreeRepack(name="no-weights", pipeline_class="P", components=(
            RepackComponent(name="a", library="l", class_name="C",
                            files=(FileRoute("vocab.json"),)),))
    with pytest.raises(DeclarationError, match="EMPTY directory"):
        RepackComponent(name="a", library="l", class_name="C")
    with pytest.raises(DeclarationError, match="exactly one of"):
        ConfigField("a", source="b", value=1)
    with pytest.raises(DeclarationError, match="declares no fields"):
        ComponentConfig(source="config.json", fields=())


def test_the_move_fast_path_is_a_property_of_the_declaration() -> None:
    """The disk preflight prices this, so it must not be a guess."""

    assert require_tree_repack("sensenova-u1.mot").is_pure_move is True


# ------------------------------------------------------ the whole clone leg


class _Ctx:
    def __init__(self, server: Any) -> None:
        self._file_api_base_url = f"http://127.0.0.1:{server.server_port}"
        self._worker_capability_token = "cap-token"
        self.owner = "tensorhub"
        self.request_id = "req-1670"
        self.destination = {"repo": "sensenova/fallback"}


def _fake_plan(source_dir: Path, strategy: str, layout: str) -> Any:
    files = [
        (p.relative_to(source_dir).as_posix(), p.stat().st_size,
         hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(source_dir.rglob("*")) if p.is_file()
    ]
    return SimpleNamespace(
        provider="huggingface",
        paths=[name for name, _, _ in files],
        source_storage_bits=32,
        classification=SimpleNamespace(
            strategy=strategy,
            attrs={"file_layout": layout, "file_type": "safetensors"},
        ),
        bank_files=lambda: list(files),
    )


def _clone(
    monkeypatch: pytest.MonkeyPatch, fake_hub: Any, tmp_path: Path, source_dir: Path,
    published: list[Any], **kwargs: Any,
) -> Any:
    plan = _fake_plan(source_dir, "transformers", "single-file")
    attrs = dict(plan.classification.attrs)
    attrs["dtype"] = detect_snapshot_dtype(source_dir)
    plan.classification.attrs.update(attrs)

    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setattr("gen_worker.convert.clone.plan_huggingface", lambda *a, **k: plan)

    def _capture(tree: Any, *a: Any, **k: Any) -> Any:
        root = Path(tree)
        published.append(SimpleNamespace(
            path=root,
            names=sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()),
            digests=_tensor_digests(root),
        ))
        return files_from_tree(tree, *a, **k)

    monkeypatch.setattr("gen_worker.convert.clone.files_from_tree", _capture)
    monkeypatch.setattr(
        "gen_worker.convert.clone.ingest_huggingface",
        lambda source_ref, dest_dir, **kw: IngestedSource(
            provider="huggingface", source_ref=source_ref,
            source_revision="13a8d0f3", dir=source_dir, layout="single-file",
            model_family="fake", model_family_variant="fake1",
            classification=plan.classification, attrs=attrs,
            metadata={"source_provider": "huggingface"},
            repo_spec={"kind": "model", "library_name": "transformers"},
        ))
    return run_clone(
        _Ctx(fake_hub), provider="huggingface",
        source_ref="sensenova/SenseNova-U1.5-8B-MoT-Preview",
        destination_repo="sensenova/sensenova-u1-5-8b-mot-preview",
        destination_release="r1", **kwargs,
    )


def test_the_cast_and_the_repack_are_ONE_submission(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """se#840's manifest, end to end: cast F32->BF16 AND land the shape.

    The ruling's reason for putting the repack on the conversion leg is that
    the 50 GB read is paid once. This asserts both halves came out of one
    `run_clone`, and that the hub was told the layout the tree actually is.
    """

    source_dir = _flat_tree(tmp_path / "source", shards=2)
    before = _tensor_digests(source_dir)
    published: list[Any] = []

    result = _clone(
        monkeypatch, fake_hub, tmp_path, source_dir, published,
        outputs=[{"dtype": "bf16", "file_layout": "multi-file",
                  "file_type": "safetensors", "repack": "sensenova-u1.mot"}],
    )

    assert not result.failed_flavors, result.failed_flavors
    assert len(published) == 1
    tree = published[0]
    assert "model_index.json" in tree.names
    assert "transformer/config.json" in tree.names
    assert "tokenizer/tokenizer_config.json" in tree.names
    assert any(n.startswith("transformer/diffusion_pytorch_model") for n in tree.names)
    assert not any(n == "config.json" for n in tree.names)

    # The cast really ran: every F32 tensor is bf16 now, and the ones that
    # were already bf16 are byte-identical through both legs.
    assert not any("|F32|" in k for k in tree.digests), sorted(tree.digests)[:3]
    untouched = {k: v for k, v in before.items() if "|BF16|" in k}
    assert untouched and all(tree.digests[k] == v for k, v in untouched.items()), (
        "a tensor the cast had no reason to touch changed in the repack")

    row = list(_FakeHub.state["publishes"].values())[0]
    assert row["dtype"] == "bf16"
    assert row["file_layout"] == MULTI_FILE, (
        "the produced tree IS a diffusers component tree; the row must say so")
    meta = row.get("metadata") or {}
    assert meta.get("attr_tree_repack") == "sensenova-u1.mot"
    assert meta.get("component_dtypes") == {"transformer": "bf16"}, (
        "the hub's own per-component header read must see the repacked component")

    # ⚠️ ONE MEMBER OUT OF A TWO-SHARD SOURCE, AND THE REPACK IS NOT WHO DID
    # THAT. `stream_reencode` writes one output file per weight SET, so the
    # cast has already collapsed the shards by the time the repack runs; the
    # repack then preserves what it was handed (the sharded case above proves
    # it preserves 4 of 4 when nothing collapsed them first). Asserted rather
    # than glossed, because "the produced member count" is exactly the fact
    # `sensenova-u1.mot@2` is banked from, and pgw#1669 is the open issue for
    # the axis that collapses it.
    assert meta.get("attr_repack_members") == "transformer:1"


def test_a_repack_request_can_never_be_satisfied_by_publishing_the_source() -> None:
    """The substitution shape that cost this checkpoint pgw#1668 and pgw#1669.

    A source whose dtype already matches takes the PUBLISH_SOURCE arm, which
    hands the INGEST tree to the publisher and runs no transform at all. With
    a repack requested that arm would publish a flat tree carrying a
    `tree_repack` attribute — a lie with a receipt.
    """

    plain = OutputSpec(dtype="bf16", file_layout="multi-file", file_type="safetensors")
    repacked = OutputSpec(dtype="bf16", file_layout="multi-file",
                          file_type="safetensors", repack="sensenova-u1.mot")

    assert spec_actions([plain], publish_as_is=True, source_dtype="bf16",
                        explicit_outputs=True, cast_eligible=True) == [PUBLISH_SOURCE]
    assert spec_actions([repacked], publish_as_is=True, source_dtype="bf16",
                        explicit_outputs=True, cast_eligible=True) == [CAST_OUTPUT]
    # And a source that cannot be transformed at all REFUSES rather than
    # falling back to the untransformed tree.
    assert spec_actions([repacked], publish_as_is=True, source_dtype="bf16",
                        explicit_outputs=True, cast_eligible=False) == [NOT_POSSIBLE]


def test_a_bf16_source_asked_only_for_a_repack_is_still_repacked(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The arm above, driven for real: no cast to do, and the shape still lands."""

    keys = {k: ("BF16", n) for k, (_d, n) in _sensenova_keys().items()}
    source_dir = _flat_tree(tmp_path / "source", keys=keys)
    published: list[Any] = []

    result = _clone(
        monkeypatch, fake_hub, tmp_path, source_dir, published,
        outputs=[{"dtype": "bf16", "file_layout": "multi-file",
                  "file_type": "safetensors", "repack": "sensenova-u1.mot"}],
    )
    assert not result.failed_flavors, result.failed_flavors
    assert "model_index.json" in published[0].names
    assert published[0].path.resolve() != source_dir.resolve(), (
        "the SOURCE tree was handed to the publisher — the repack was skipped")
    assert _tensor_digests(source_dir), "the ingest tree was consumed rather than read"


# ------------------------------------ the REAL key set and the REAL config


#: `sensenova/SenseNova-U1.5-8B-MoT-Preview` rev `13a8d0f3`'s own
#: `config.json`, verbatim and entire (2,631 bytes, fetched at $0 — the
#: document, not one weight byte). Inline rather than as a fixture file so the
#: diff shows exactly which document the field map below was proven against.
_REAL_UPSTREAM_CONFIG: dict[str, Any] = json.loads("""{
    "architectures": [
        "NEOChatModel"
    ],
    "auto_map": {
        "AutoConfig": "configuration_neo_chat.NEOChatConfig",
        "AutoModel": "modeling_neo_chat.NEOChatModel",
        "AutoModelForCausalLM": "modeling_neo_chat.NEOChatModel"
    },
    "downsample_ratio": 0.5,
    "eos_token_id": 151645,
    "llm_config": {
        "_name_or_path": null,
        "architectures": [
            "Qwen3ForCausalLM"
        ],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "max_position_embeddings": 262144,
        "max_position_embeddings_hw": 10000,
        "max_window_layers": 42,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 42,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": null,
        "rope_theta": 5000000.0,
        "rope_theta_hw": 10000.0,
        "sliding_window": null,
        "torch_dtype": "bfloat16",
        "use_cache": false,
        "use_deepep": false,
        "use_sliding_window": false,
        "vocab_size": 151936,
        "pure_llm": false
    },
    "model_type": "neo_chat",
    "pad_token_id": 151643,
    "template": "neo1_0",
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.37.2",
    "use_backbone_lora": 0,
    "use_llm_lora": 0,
    "min_pixels": 65536,
    "max_pixels": 16777216,
    "patch_size": 16,
    "timestep_shift": 1.0,
    "time_schedule": "standard",
    "time_shift_type": "exponential",
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 64,
    "max_image_seq_len": 4096,
    "noise_scale_mode": "resolution",
    "noise_scale_base_image_seq_len": 64,
    "add_noise_scale_embedding": true,
    "noise_scale_max_value": 16.0,
    "noise_scale": 1.0,
    "P_mean": -0.8,
    "P_std": 0.8,
    "t_eps": 0.05,
    "fm_head_dim": 1536,
    "fm_head_layers": 2,
    "fm_head_mlp_ratio": 1,
    "extra_num_layers_post": 0,
    "concat_time_token_num": 0,
    "use_pixel_head": true,
    "use_adaLN": false,
    "vision_config": {
        "architectures": [
            "NEOVisionModel"
        ],
        "attention_dropout": 0.0,
        "auto_map": {
            "AutoConfig": "configuration_neo_vit.NEOVisionConfig",
            "AutoModel": "modeling_neo_vit.NEOVisionModel"
        },
        "llm_hidden_size": 4096,
        "downsample_ratio": 0.5,
        "hidden_size": 1024,
        "model_type": "neo_vision",
        "rope_theta_vision": 10000.0,
        "max_position_embeddings_vision": 10000,
        "num_channels": 3,
        "patch_size": 16,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.37.2",
        "min_pixels": 65536,
        "max_pixels": 16777216
    }
}""")

#: What the endpoint's `transformer/config.json` MUST be: the tree the first
#: real `gen-worker lock --checkpoint` derive hollow-loaded to 100% on both
#: components (se#840). `SenseNovaConfig(llm=LLMConfig(**llm), ...)` is built
#: from these kwargs, so an extra key is a `TypeError` at load and a missing one
#: is a silently wrong default.
_ENDPOINT_TRANSFORMER_CONFIG: dict[str, Any] = {
    "_class_name": "SenseNovaU1",
    "_diffusers_version": "0.39.0",
    "llm": {
        "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 42,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "rms_norm_eps": 1e-06, "rope_theta": 5000000.0, "rope_theta_hw": 10000.0,
        "vocab_size": 151936, "attention_bias": False,
    },
    "vision": {
        "hidden_size": 1024, "llm_hidden_size": 4096, "patch_size": 16,
        "num_channels": 3, "downsample_ratio": 0.5, "rope_theta_vision": 10000.0,
        "max_position_embeddings_vision": 10000,
    },
    "patch_size": 16, "downsample_ratio": 0.5, "t_eps": 0.05, "noise_scale": 1.0,
    "noise_scale_mode": "resolution", "noise_scale_base_image_seq_len": 64,
    "noise_scale_max_value": 16.0, "add_noise_scale_embedding": True,
}


def _real_key_names() -> list[str]:
    """All 1116 upstream key names, out of the VENDORED tensorfs corpus.

    `sensenova-u1.mot@1` (tensorfs#161) was extracted from the real published
    headers by HTTP range, so its tensor table IS the upstream key set — and it
    is already in this repo. Copying the names into a fixture would be a second
    producer of a fact the corpus already carries.
    """

    root = Path(__file__).resolve().parents[2] / "src" / "gen_worker" / "_vendor"
    doc = json.loads(
        (root / "tensorfs" / "spec" / "v2" / "topologies"
         / "sensenova-u1.mot.v1.json").read_text("utf-8"))
    names = [n for comp in doc["components"] for n in comp["tensors"]]
    assert len(names) == 1116, f"the vendored corpus moved: {len(names)} names"
    return names


def test_the_real_key_set_and_the_real_config_produce_the_endpoints_tree(
    tmp_path: Path,
) -> None:
    """The deliverable, against upstream's OWN documents rather than a fixture.

    A field map that resolves on a hand-written fixture and not on the real
    `config.json` is the exact bug this case exists to catch — the real one
    nests under `llm_config`/`vision_config`, carries a dead `auto_map`, and
    states a `timestep_shift` the reference implementation does not serve.
    """

    tree = tmp_path / "flat"
    tree.mkdir()
    (tree / "config.json").write_text(json.dumps(_REAL_UPSTREAM_CONFIG))
    (tree / "vocab.json").write_text("{}")
    (tree / "merges.txt").write_text("#version: 0.2\n")
    (tree / "added_tokens.json").write_text("{}")
    (tree / "special_tokens_map.json").write_text("{}")
    (tree / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"}))
    names = _real_key_names()
    (tree / "model.safetensors").write_bytes(
        _safetensors({name: ("BF16", 1) for name in names}))

    report = apply_tree_repack(tree, require_tree_repack("sensenova-u1.mot"))

    assert report.tensor_count == 1116, "a real key fell out of the routing"
    assert json.loads((tree / "transformer" / "config.json").read_text()) == (
        _ENDPOINT_TRANSFORMER_CONFIG), (
        "the derived component config is not the one the endpoint's SenseNovaU1 "
        "kwargs accept — an extra key is a TypeError at load and a missing one "
        "is a silently wrong default")

    member = tree / "transformer" / "diffusion_pytorch_model.safetensors"
    raw = member.read_bytes()
    header = json.loads(raw[8:8 + int.from_bytes(raw[:8], "little")])
    assert set(header) - {"__metadata__"} == set(names), (
        "the produced component does not carry exactly the upstream key set")
