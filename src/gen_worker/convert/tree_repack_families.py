"""The platform's declared tree repacks — one table, read as data.

Each entry is a KEY MAP plus the config derivation that goes with it. Adding a
family is adding a row here; nothing else in ``convert/`` learns a model's
name.
"""

from __future__ import annotations

from .tree_repack import register_tree_repack
from .tree_repack_spec import (
    ComponentConfig,
    ConfigField,
    FileRoute,
    RepackComponent,
    TreeRepack,
)

__all__ = ["SENSENOVA_U1_MOT"]


def _llm(target: str, source: str | None = None) -> ConfigField:
    return ConfigField(f"llm.{target}", f"llm_config.{source or target}")


def _vision(target: str, source: str | None = None) -> ConfigField:
    return ConfigField(f"vision.{target}", f"vision_config.{source or target}")


def _root(name: str) -> ConfigField:
    return ConfigField(name, name)


#: SenseNova-U1.5-8B-MoT — a NATIVELY unified model that is ONE component.
#:
#: There is no VAE and no text encoder to split out: the understanding stream IS
#: the text encoder and the pixel head IS the decoder (se#840's porting map). So
#: the key map is deliberately the identity — ``language_model.*``,
#: ``vision_model.*`` and ``fm_modules.*`` are already the module path the
#: endpoint's ``SenseNovaU1`` carries, and the component DIRECTORY is what
#: supplies the "transformer" scope in a diffusers tree. What the repack
#: actually produces is the directory shape, the two config documents and the
#: ``model_index.json`` that make ``ctx.load`` hollow-legal — the flat upstream
#: root is not loadable by our code at all, because the hollow session
#: intercepts only diffusers/transformers loaders and a root ``config.json``
#: whose ``auto_map`` names four ``.py`` files that do not exist reaches none of
#: them.
#:
#: The three declared ``requires_key_prefixes`` are what stops this map running
#: over a tree that merely happens to be flat.
SENSENOVA_U1_MOT = register_tree_repack(TreeRepack(
    name="sensenova-u1.mot",
    pipeline_class="SenseNovaU1Pipeline",
    requires_key_prefixes=("language_model.", "vision_model.", "fm_modules."),
    components=(
        RepackComponent(
            name="transformer",
            library="sensenova_u1",
            class_name="SenseNovaU1",
            weight_stem="diffusion_pytorch_model",
            # No key_prefixes: the model is one component, so this is the
            # catch-all and nothing can fall out of the tree unnoticed.
            config=ComponentConfig(source="config.json", fields=(
                _llm("hidden_size"),
                _llm("intermediate_size"),
                _llm("num_hidden_layers"),
                _llm("num_attention_heads"),
                _llm("num_key_value_heads"),
                _llm("head_dim"),
                _llm("rms_norm_eps"),
                _llm("rope_theta"),
                _llm("rope_theta_hw"),
                _llm("vocab_size"),
                _llm("attention_bias"),
                _vision("hidden_size"),
                _vision("llm_hidden_size"),
                _vision("patch_size"),
                _vision("num_channels"),
                _vision("downsample_ratio"),
                _vision("rope_theta_vision"),
                _vision("max_position_embeddings_vision"),
                _root("patch_size"),
                _root("downsample_ratio"),
                _root("t_eps"),
                _root("noise_scale"),
                _root("noise_scale_mode"),
                _root("noise_scale_base_image_seq_len"),
                _root("noise_scale_max_value"),
                _root("add_noise_scale_embedding"),
                # NOT transcribed, deliberately: `timestep_shift`, the
                # `dynamic` schedule and the flow-matching-head geometry. The
                # config's `timestep_shift` is 1.0 and the reference
                # implementation SERVES 3.0, `time_schedule` is hard-forced to
                # "standard" by upstream's own code, and `use_pixel_head` turns
                # the fm-head fields off. A field nobody reads is a field that
                # drifts silently — the serving values live in
                # `SenseNovaU1Defaults` (pgw#1664) where they are one producer.
            )),
        ),
        RepackComponent(
            name="tokenizer",
            library="transformers",
            class_name="Qwen2TokenizerFast",
            files=(
                FileRoute("vocab.json"),
                FileRoute("merges.txt"),
                FileRoute("added_tokens.json"),
                FileRoute("special_tokens_map.json"),
                # The upstream repo ships NO `tokenizer.json`; the fast
                # tokenizer is built from vocab+merges on first load. Declared
                # optional so a source that does ship one carries it through.
                FileRoute("tokenizer.json", required=False),
                FileRoute("tokenizer_config.json", json_overrides=(
                    # `model_index.json` says Qwen2TokenizerFast and upstream's
                    # own document says Qwen2Tokenizer. Two spellings of one
                    # fact is how an AutoTokenizer load lands on the slow class
                    # while the pipeline believes it has the fast one.
                    ConfigField("tokenizer_class", value="Qwen2TokenizerFast"),
                )),
            ),
        ),
    ),
    keep_root=("README.md", "README_CN.md", "LICENSE", "LICENSE.txt", ".gitattributes"),
))
