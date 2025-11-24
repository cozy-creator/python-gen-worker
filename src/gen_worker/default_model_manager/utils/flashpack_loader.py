"""
FlashPack Loading Integration for DefaultModelManager

This module provides FlashPack loading capability to the model manager.

Integration:
   from .utils.flashpack_loader import FlashPackLoader
   self.flashpack_loader = FlashPackLoader()
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Type, Union
import hashlib

import torch
from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)

# FlashPack suffix for directories
FLASHPACK_SUFFIX = ".flashpack"

# Components that can be loaded from FlashPack
FLASHPACK_COMPONENTS = ["unet", "vae", "text_encoder", "text_encoder_2", "transformer"]


class FlashPackLoader:
    """
    Handles loading models from FlashPack format.
    
    FlashPack provides 2-4x faster loading compared to safetensors.
    This loader transparently checks for FlashPack versions and falls
    back to standard loading if not available.
    """
    
    def __init__(
        self,
        cozy_models_dir: str = "/workspace/.cozy-creator/models",
        hf_cache_dir: str = "/workspace/.cache/huggingface/hub"
    ):
        self.cozy_models_dir = Path(cozy_models_dir)
        self.hf_cache_dir = Path(hf_cache_dir)
        self._flashpack_available = self._check_flashpack_installed()
    
    def _check_flashpack_installed(self) -> bool:
        """Check if flashpack library is available"""
        try:
            from flashpack import unpack_from_file
            return True
        except ImportError:
            logger.warning("FlashPack not installed. Using standard loading.")
            return False
    
    def get_flashpack_path(self, model_id: str, source: str) -> Optional[Path]:
        """
        Get the FlashPack directory path for a model if it exists.
        
        Args:
            model_id: Model identifier (e.g., "pony.realism")
            source: Source string from pipeline_defs
            
        Returns:
            Path to FlashPack directory or None if not found
        """
        if not self._flashpack_available:
            return None
        
        # Determine base path based on source type
        if source.startswith("hf:"):
            base_path = self._get_hf_flashpack_path(source[3:])
        else:
            base_path = self._get_civitai_flashpack_path(model_id, source)
        
        if base_path and base_path.exists():
            # Verify it has the required files
            if (base_path / "pipeline").exists():
                logger.info(f"⚡ FlashPack found for {model_id}: {base_path}")
                return base_path
        
        return None
    
    def _get_hf_flashpack_path(self, repo_id: str) -> Optional[Path]:
        """Get FlashPack path for HuggingFace model"""
        folder_name = f"models--{repo_id.replace('/', '--')}"
        flashpack_path = self.hf_cache_dir / (folder_name + FLASHPACK_SUFFIX)
        return flashpack_path
    
    def _get_civitai_flashpack_path(self, model_id: str, source: str) -> Optional[Path]:
        """Get FlashPack path for Civitai model"""
        safe_name = model_id.replace("/", "-")
        
        # Find the original model directory
        matching_dirs = list(self.cozy_models_dir.glob(f"{safe_name}--*"))
        if not matching_dirs:
            # Try finding by URL hash
            url_hash = hashlib.md5(source.encode()).hexdigest()[:8]
            matching_dirs = list(self.cozy_models_dir.glob(f"{safe_name}--{url_hash}"))
        
        if not matching_dirs:
            return None
        
        # Get the FlashPack sibling directory
        original_dir = matching_dirs[0]
        flashpack_path = original_dir.parent / (original_dir.name + FLASHPACK_SUFFIX)
        return flashpack_path
    
    async def load_from_flashpack(
        self,
        model_id: str,
        flashpack_path: Path,
        pipeline_class: Type[DiffusionPipeline],
    ) -> Optional[DiffusionPipeline]:
        """
        Load a model from FlashPack format.
        
        Args:
            model_id: Model identifier
            flashpack_path: Path to FlashPack directory
            pipeline_class: Pipeline class to instantiate
            
        Returns:
            Loaded pipeline or None if loading failed
        """
        try:
            from flashpack import unpack_from_file
            import asyncio
            
            logger.info(f"⚡ Loading {model_id} from FlashPack...")
            
            # Determine dtype based on model type
            torch_dtype = torch.bfloat16 if "flux" in model_id.lower() else torch.float16
            
            # Load pipeline config (scheduler, tokenizer, etc.)
            pipeline_config_dir = flashpack_path / "pipeline"
            
            # Load base pipeline from config
            pipeline = await asyncio.to_thread(
                pipeline_class.from_pretrained,
                str(pipeline_config_dir),
                torch_dtype=torch_dtype,
                local_files_only=True
            )
            
            # Load FlashPack components
            for component_name in FLASHPACK_COMPONENTS:
                fp_file = flashpack_path / f"{component_name}.flashpack"
                if fp_file.exists():
                    logger.info(f"   Loading {component_name} from FlashPack...")
                    component = await asyncio.to_thread(
                        unpack_from_file,
                        str(fp_file)
                    )
                    # Move to appropriate dtype
                    if hasattr(component, 'to'):
                        component = component.to(dtype=torch_dtype)
                    setattr(pipeline, component_name, component)
            
            logger.info(f"✅ Successfully loaded {model_id} from FlashPack")
            return pipeline
            
        except Exception as e:
            logger.error(f"❌ FlashPack loading failed for {model_id}: {e}")
            logger.exception("Full traceback:")
            return None
    
    def has_flashpack(self, model_id: str, source: str) -> bool:
        """Check if FlashPack version exists for a model"""
        return self.get_flashpack_path(model_id, source) is not None

