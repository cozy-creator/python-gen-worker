"""
FlashPack Loading Integration for DefaultModelManager

This module provides FlashPack loading capability to the model manager.
It checks if a FlashPack version of a model exists and loads from it
for faster loading times (2-4s vs 8-12s for safetensors).

Now with local cache support - copies models from NFS to local NVMe first.

Integration:
1. Add this import to manager.py:
   from .utils.flashpack_loader import FlashPackLoader

2. Initialize in DefaultModelManager.__init__():
   self.flashpack_loader = FlashPackLoader()

3. Modify _load_model_by_source() to try FlashPack first (see integration code below)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Type, Union, Dict, List
import hashlib
import asyncio

import torch
from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)

# FlashPack suffix for directories
FLASHPACK_SUFFIX = ".flashpack"

# Components that can be loaded from FlashPack
FLASHPACK_COMPONENTS = ["unet", "vae", "text_encoder", "text_encoder_2", "transformer"]

# NFS paths
NFS_COZY_MODELS = "/workspace/.cozy-creator/models"
NFS_HF_CACHE = "/workspace/.cache/huggingface/hub"


class FlashPackLoader:
    """
    Handles loading models from FlashPack format with local cache support.
    
    FlashPack provides 2-4x faster loading compared to safetensors.
    Local cache copies models from NFS to local NVMe for additional speedup.
    """
    
    def __init__(
        self,
        cozy_models_dir: str = NFS_COZY_MODELS,
        hf_cache_dir: str = NFS_HF_CACHE,
        use_local_cache: bool = True,
    ):
        self.cozy_models_dir = Path(cozy_models_dir)
        self.hf_cache_dir = Path(hf_cache_dir)
        self._flashpack_available = self._check_flashpack_installed()
        
        # Initialize local cache if enabled
        self.local_cache = None
        if use_local_cache:
            try:
                from .local_cache import LocalModelCache
                self.local_cache = LocalModelCache()
                logger.info("✓ Local NVMe cache enabled")
            except ImportError:
                logger.warning("LocalModelCache not available, using NFS directly")
    
    def _check_flashpack_installed(self) -> bool:
        """Check if flashpack library is available"""
        try:
            from flashpack import assign_from_file
            return True
        except ImportError:
            logger.warning("FlashPack not installed. Using standard loading.")
            return False
    
    def get_flashpack_path(self, model_id: str, source: str) -> Optional[Path]:
        """
        Get the FlashPack directory path for a model if it exists.
        Checks local cache first, then NFS.
        
        Args:
            model_id: Model identifier (e.g., "pony.realism")
            source: Source string from pipeline_defs
            
        Returns:
            Path to FlashPack directory or None if not found
        """
        if not self._flashpack_available:
            return None
        
        # Check local cache first
        if self.local_cache:
            local_path = self.local_cache.get_local_path_if_cached(model_id, source)
            if local_path and local_path.exists() and FLASHPACK_SUFFIX in local_path.name:
                logger.info(f"⚡ FlashPack found in local cache for {model_id}")
                return local_path
        
        # Check NFS
        if source.startswith("hf:"):
            base_path = self._get_hf_flashpack_path(source[3:])
        else:
            base_path = self._get_civitai_flashpack_path(model_id, source)
        
        if base_path and base_path.exists():
            if (base_path / "pipeline").exists():
                logger.info(f"⚡ FlashPack found on NFS for {model_id}: {base_path}")
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
        Copies to local cache first if enabled.
        
        Args:
            model_id: Model identifier
            flashpack_path: Path to FlashPack directory (on NFS)
            pipeline_class: Pipeline class to instantiate
            
        Returns:
            Loaded pipeline or None if loading failed
        """
        try:
            from flashpack import assign_from_file
            
            # Copy to local cache first if enabled
            load_path = flashpack_path
            if self.local_cache:
                # Get source for cache lookup
                source = self._infer_source_from_path(flashpack_path)
                local_path = await self.local_cache.ensure_local(
                    model_id, source, priority=True
                )
                if local_path:
                    load_path = local_path
                    logger.info(f"⚡ Loading {model_id} from local cache")
                else:
                    logger.warning(f"Local cache failed, loading from NFS")
            
            logger.info(f"⚡ Loading {model_id} from FlashPack at {load_path}...")
            
            # Determine dtype based on model type
            torch_dtype = torch.bfloat16 if "flux" in model_id.lower() else torch.float16
            
            # Load pipeline config (scheduler, tokenizer, etc.)
            pipeline_config_dir = load_path / "pipeline"
            
            # Load base pipeline from config (this creates the model structure)
            pipeline = await asyncio.to_thread(
                pipeline_class.from_pretrained,
                str(pipeline_config_dir),
            )
            
            # Assign FlashPack weights to each component
            for component_name in FLASHPACK_COMPONENTS:
                fp_file = load_path / f"{component_name}.flashpack"
                if fp_file.exists() and hasattr(pipeline, component_name):
                    component = getattr(pipeline, component_name)
                    if component is not None:
                        logger.info(f"   Assigning {component_name} from FlashPack...")
                        await asyncio.to_thread(
                            assign_from_file,
                            component,
                            str(fp_file)
                        )
            
            # Move to cuda with correct dtype
            pipeline.to("cuda", dtype=torch_dtype)
            
            logger.info(f"✅ Successfully loaded {model_id} from FlashPack")
            return pipeline
            
        except Exception as e:
            logger.error(f"❌ FlashPack loading failed for {model_id}: {e}")
            logger.exception("Full traceback:")
            return None
    
    def _infer_source_from_path(self, flashpack_path: Path) -> str:
        """Infer source string from FlashPack path for cache lookup"""
        path_str = str(flashpack_path)
        
        if "models--" in path_str:
            # HuggingFace model
            # Extract repo_id from models--org--name.flashpack
            name = flashpack_path.name.replace(FLASHPACK_SUFFIX, "")
            repo_id = name.replace("models--", "").replace("--", "/")
            return f"hf:{repo_id}"
        else:
            # Civitai model - return path as source
            return path_str
    
    def has_flashpack(self, model_id: str, source: str) -> bool:
        """Check if FlashPack version exists for a model"""
        return self.get_flashpack_path(model_id, source) is not None
    
    async def prefetch_deployment_models(
        self,
        model_ids: List[str],
        sources: Dict[str, str],
        exclude_model_id: Optional[str] = None
    ):
        """
        Background prefetch models for a deployment to local cache.
        
        Args:
            model_ids: List of model IDs from deployment
            sources: Dict mapping model_id → source string
            exclude_model_id: Model to skip (already being loaded)
        """
        if not self.local_cache:
            return
        
        # Filter out the model already being loaded
        models_to_prefetch = [
            mid for mid in model_ids 
            if mid != exclude_model_id
        ]
        
        if models_to_prefetch:
            logger.info(f"🔄 Starting background prefetch for {len(models_to_prefetch)} models")
            await self.local_cache.prefetch_models(models_to_prefetch, sources)
    
    def get_cache_stats(self) -> Optional[Dict]:
        """Get local cache statistics"""
        if self.local_cache:
            return self.local_cache.get_cache_stats()
        return None