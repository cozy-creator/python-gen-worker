#!/usr/bin/env python3
"""
FP8 Quantization Script using torchao
Converts diffusion models from fp16/bf16 to fp8 for faster inference
Best for GPUs with compute capability >= 8.9 (RTX 4090, H100, etc.)
"""

import argparse
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import TorchAoConfig as DiffusersTorchAoConfig
from transformers import TorchAoConfig as TransformersTorchAoConfig
from torchao.quantization import Float8WeightOnlyConfig


def quantize_to_fp8(
    model_path: str,
    output_path: str,
    quantize_transformer: bool = True,
    quantize_text_encoders: bool = False,
    compile_model: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16
):
    """
    Quantize a diffusion model to FP8 using torchao
    
    Args:
        model_path: Path to the model (local or HuggingFace repo)
        output_path: Where to save the quantized model
        quantize_transformer: Quantize the main transformer/unet (recommended)
        quantize_text_encoders: Also quantize text encoders (optional, less impact)
        compile_model: Apply torch.compile for extra speedup (takes time on first run)
        torch_dtype: Base dtype (bfloat16 recommended for newer GPUs)
    """
    print(f"Loading model from {model_path}...")
    print(f"Using dtype: {torch_dtype}")
    
    # Build quantization config
    quant_mapping = {}
    
    if quantize_transformer:
        # For SD1.5/SDXL, this quantizes the UNet
        # For Flux/SD3, this quantizes the Transformer
        quant_mapping["transformer"] = DiffusersTorchAoConfig("float8_weight_only")
        quant_mapping["unet"] = DiffusersTorchAoConfig("float8_weight_only")
        print("✓ Will quantize transformer/unet to FP8")
    
    if quantize_text_encoders:
        # Quantize text encoders (smaller impact on speed, may help with VRAM)
        quant_mapping["text_encoder"] = TransformersTorchAoConfig(Float8WeightOnlyConfig())
        quant_mapping["text_encoder_2"] = TransformersTorchAoConfig(Float8WeightOnlyConfig())
        quant_mapping["text_encoder_3"] = TransformersTorchAoConfig(Float8WeightOnlyConfig())
        print("✓ Will quantize text encoders to FP8")
    
    if not quant_mapping:
        raise ValueError("Must quantize at least one component!")
    
    pipeline_quant_config = PipelineQuantizationConfig(quant_mapping=quant_mapping)
    
    # Load and quantize
    print("\nLoading and quantizing model...")
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        quantization_config=pipeline_quant_config,
        torch_dtype=torch_dtype,
        device_map="balanced"
    )
    
    print(f"Loaded pipeline: {pipe.__class__.__name__}")
    
    # Optional: compile for extra speed (takes time on first run)
    if compile_model:
        print("\nCompiling model with torch.compile...")
        print("(This will take a few minutes on first run, but speeds up inference)")
        
        # Compile the main model component
        if hasattr(pipe, 'transformer'):
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        elif hasattr(pipe, 'unet'):
            pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
        
        print("✓ Model compiled")
    
    # Save
    print(f"\nSaving quantized model to {output_path}...")
    # Note: safe_serialization=False because torchao quantized models can't use safetensors yet
    pipe.save_pretrained(output_path, safe_serialization=False)
    
    print(f"\n✓ Quantization complete!")
    print(f"✓ Saved to: {output_path}")
    
    # Print expected benefits
    print("\n📊 Expected Benefits:")
    print("  • ~50% VRAM reduction")
    print("  • 1.3-1.5x faster inference (with torch.compile)")
    print("  • Minimal quality loss")
    print("\n⚠️  Note: First inference will be slow due to compilation.")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize diffusion models to FP8 using torchao"
    )
    
    parser.add_argument(
        "model",
        help="Model path (local directory or HuggingFace repo)"
    )
    
    parser.add_argument(
        "output",
        help="Output directory for quantized model"
    )
    
    parser.add_argument(
        "--quantize-text-encoders",
        action="store_true",
        help="Also quantize text encoders (optional, smaller impact)"
    )
    
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Apply torch.compile for extra speedup (slower first run)"
    )
    
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use float32 as base dtype instead of bfloat16"
    )
    
    args = parser.parse_args()
    
    torch_dtype = torch.float32 if args.fp32 else torch.bfloat16
    
    quantize_to_fp8(
        args.model,
        args.output,
        quantize_transformer=True,
        quantize_text_encoders=args.quantize_text_encoders,
        compile_model=args.compile,
        torch_dtype=torch_dtype
    )


if __name__ == "__main__":
    main()
