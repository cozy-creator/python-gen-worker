#!/usr/bin/env python3
"""
Safetensors <-> Diffusers Converter Script
Converts between single safetensors files and diffusers format
Works with any model architecture (SD1.5, SDXL, SD3, Flux, etc.)
"""

import argparse
import os
from pathlib import Path
from diffusers import DiffusionPipeline
import torch


def safetensors_to_diffusers(
    safetensors_path: str,
    output_path: str,
    torch_dtype: torch.dtype = torch.float16
):
    """
    Convert safetensors file to diffusers format
    Automatically detects model type
    
    Args:
        safetensors_path: Path to the .safetensors file
        output_path: Directory to save diffusers format
        torch_dtype: Torch dtype to use (default: float16)
    """
    print(f"Converting {safetensors_path} to diffusers format...")
    
    # Use DiffusionPipeline to automatically detect model type
    pipe = DiffusionPipeline.from_single_file(
        safetensors_path,
        torch_dtype=torch_dtype
    )
    
    print(f"Detected pipeline: {pipe.__class__.__name__}")
    
    # Save in diffusers format
    pipe.save_pretrained(
        output_path,
        safe_serialization=True
    )
    
    print(f"✓ Conversion complete! Saved to: {output_path}")


def diffusers_to_safetensors(
    diffusers_path: str,
    output_path: str,
    torch_dtype: torch.dtype = torch.float16
):
    """
    Convert diffusers format to single safetensors file
    Works with any model architecture by dynamically detecting components
    
    Args:
        diffusers_path: Path to diffusers model directory
        output_path: Path for output .safetensors file
        torch_dtype: Torch dtype to use (default: float16)
    """
    print(f"Converting {diffusers_path} to safetensors format...")
    
    # Load the diffusers model (automatically detects type)
    pipe = DiffusionPipeline.from_pretrained(
        diffusers_path,
        torch_dtype=torch_dtype
    )
    
    print(f"Loaded pipeline: {pipe.__class__.__name__}")
    
    # Ensure output path has .safetensors extension
    if not output_path.endswith('.safetensors'):
        output_path += '.safetensors'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    from safetensors.torch import save_file
    
    # Dynamically collect all model components
    state_dict = {}
    
    # Get all components from the pipeline configuration
    for component_name in pipe.config.keys():
        if component_name.startswith('_'):  # Skip private attributes
            continue
            
        component = getattr(pipe, component_name, None)
        
        # Check if component has state_dict (is a model/module)
        if component is not None and hasattr(component, 'state_dict'):
            try:
                component_state = component.state_dict()
                for key, value in component_state.items():
                    state_dict[f"{component_name}.{key}"] = value
                print(f"  ✓ Added {component_name} ({len(component_state)} tensors)")
            except Exception as e:
                print(f"  ⚠ Skipped {component_name}: {e}")
    
    if not state_dict:
        raise ValueError("No model components found to save")
    
    # Save to single file
    print(f"Saving {len(state_dict)} total tensors...")
    save_file(state_dict, output_path)
    
    print(f"✓ Conversion complete! Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert between safetensors and diffusers formats (auto-detects model type)"
    )
    
    parser.add_argument(
        "mode",
        choices=["to-diffusers", "to-safetensors"],
        help="Conversion mode"
    )
    
    parser.add_argument(
        "input",
        help="Input path (safetensors file or diffusers directory)"
    )
    
    parser.add_argument(
        "output",
        help="Output path (directory for diffusers, file for safetensors)"
    )
    
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use float32 instead of float16 (larger but more precise)"
    )
    
    args = parser.parse_args()
    
    torch_dtype = torch.float32 if args.fp32 else torch.float16
    
    if args.mode == "to-diffusers":
        safetensors_to_diffusers(
            args.input,
            args.output,
            torch_dtype
        )
    else:
        diffusers_to_safetensors(
            args.input,
            args.output,
            torch_dtype
        )


if __name__ == "__main__":
    main()
