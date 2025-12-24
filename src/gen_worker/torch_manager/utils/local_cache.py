"""
Local Model Cache - Copies models from NFS to local NVMe for faster loading.

Flow:
1. First job arrives with model_id
2. Copy that model from NFS → local (prioritized)
3. Load from local (fast!)
4. Background: Copy remaining deployment models

Usage:
    from .utils.local_cache import LocalModelCache
    
    # In worker init
    self.local_cache = LocalModelCache()
    
    # Before loading a model
    local_path = await self.local_cache.ensure_local(model_id, source, priority=True)
    
    # Start background copying for other models
    asyncio.create_task(self.local_cache.prefetch_models(other_model_ids, sources))
"""

import os
import shutil
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, List, Set
import hashlib
import time

logger = logging.getLogger(__name__)

# Local cache location (container disk - NVMe)
LOCAL_CACHE_DIR = "/root/.local-model-cache"

# NFS locations
NFS_COZY_MODELS = "/workspace/.cozy-creator/models"
NFS_HF_CACHE = "/workspace/.cache/huggingface/hub"

# FlashPack suffix
FLASHPACK_SUFFIX = ".flashpack"


class LocalModelCache:
    """
    Manages local NVMe cache of models for faster loading.
    
    Models are copied from NFS to local disk on-demand, with the
    currently requested model getting priority.
    """
    
    def __init__(
        self,
        local_cache_dir: str = LOCAL_CACHE_DIR,
        nfs_cozy_dir: str = NFS_COZY_MODELS,
        nfs_hf_dir: str = NFS_HF_CACHE,
    ):
        self.local_cache_dir = Path(local_cache_dir)
        self.nfs_cozy_dir = Path(nfs_cozy_dir)
        self.nfs_hf_dir = Path(nfs_hf_dir)
        
        # Track what's cached and what's being copied
        self.cached_models: Set[str] = set()
        self.copying_models: Set[str] = set()
        self._copy_lock = asyncio.Lock()
        
        # Create local cache directory
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan for already cached models
        self._scan_existing_cache()
        
        logger.info(f"LocalModelCache initialized at {self.local_cache_dir}")
        logger.info(f"Already cached: {len(self.cached_models)} models")
    
    def _scan_existing_cache(self):
        """Scan local cache for already copied models"""
        if not self.local_cache_dir.exists():
            return
        
        for item in self.local_cache_dir.iterdir():
            if item.is_dir():
                # Extract model_id from directory name
                # Format: {model_id}--{hash} or {model_id}--{hash}.flashpack
                name = item.name
                if FLASHPACK_SUFFIX in name:
                    name = name.replace(FLASHPACK_SUFFIX, "")
                if "--" in name:
                    model_id = name.rsplit("--", 1)[0]
                    self.cached_models.add(model_id)
    
    async def ensure_local(
        self,
        model_id: str,
        source: str,
        priority: bool = False
    ) -> Optional[Path]:
        """
        Ensure model is in local cache, copying from NFS if needed.
        
        Args:
            model_id: Model identifier
            source: Source string from pipeline_defs
            priority: If True, this is the active job's model (copy immediately)
            
        Returns:
            Path to local model (FlashPack dir or safetensors file), or None if failed
        """
        # Check if already cached
        local_path = self._get_local_path(model_id, source)
        if local_path and local_path.exists():
            logger.info(f"✓ Model {model_id} already in local cache")
            self.cached_models.add(model_id)
            return local_path
        
        # Check if currently being copied
        if model_id in self.copying_models:
            if priority:
                # Wait for copy to complete
                logger.info(f"⏳ Waiting for {model_id} copy to complete...")
                while model_id in self.copying_models:
                    await asyncio.sleep(0.5)
                return self._get_local_path(model_id, source)
            else:
                # Non-priority, just return None (will use NFS)
                return None
        
        # Need to copy
        return await self._copy_to_local(model_id, source, priority)
    
    async def _copy_to_local(
        self,
        model_id: str,
        source: str,
        priority: bool
    ) -> Optional[Path]:
        """Copy model from NFS to local cache"""
        
        async with self._copy_lock:
            # Double-check after acquiring lock
            local_path = self._get_local_path(model_id, source)
            if local_path and local_path.exists():
                return local_path
            
            if model_id in self.copying_models:
                return None
            
            self.copying_models.add(model_id)
        
        try:
            # Find source on NFS
            nfs_path = self._find_nfs_path(model_id, source)
            if not nfs_path:
                logger.error(f"❌ Model {model_id} not found on NFS")
                return None
            
            # Determine local destination
            local_dest = self._get_local_dest(model_id, source, nfs_path)
            
            logger.info(f"📦 Copying {model_id} to local cache...")
            logger.info(f"   From: {nfs_path}")
            logger.info(f"   To:   {local_dest}")
            
            start_time = time.time()
            
            # Copy the model
            if nfs_path.is_dir():
                await asyncio.to_thread(
                    shutil.copytree,
                    str(nfs_path),
                    str(local_dest),
                    dirs_exist_ok=True
                )
            else:
                local_dest.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(
                    shutil.copy2,
                    str(nfs_path),
                    str(local_dest)
                )
            
            elapsed = time.time() - start_time
            size_gb = self._get_size_gb(local_dest)
            speed = size_gb / elapsed if elapsed > 0 else 0
            
            logger.info(f"✅ Copied {model_id} in {elapsed:.1f}s ({size_gb:.2f}GB @ {speed:.1f}GB/s)")
            
            self.cached_models.add(model_id)
            return local_dest
            
        except Exception as e:
            logger.error(f"❌ Failed to copy {model_id}: {e}")
            return None
            
        finally:
            self.copying_models.discard(model_id)
    
    def _find_nfs_path(self, model_id: str, source: str) -> Optional[Path]:
        """Find model path on NFS, preferring FlashPack"""
        
        if source.startswith("hf:"):
            return self._find_hf_nfs_path(source[3:])
        else:
            return self._find_civitai_nfs_path(model_id, source)
    
    def _find_hf_nfs_path(self, repo_id: str) -> Optional[Path]:
        """Find HuggingFace model on NFS"""
        folder_name = f"models--{repo_id.replace('/', '--')}"
        
        # Check for FlashPack version first
        flashpack_path = self.nfs_hf_dir / (folder_name + FLASHPACK_SUFFIX)
        if flashpack_path.exists():
            return flashpack_path
        
        # Fall back to regular HF cache
        hf_path = self.nfs_hf_dir / folder_name
        if hf_path.exists():
            return hf_path
        
        return None
    
    def _find_civitai_nfs_path(self, model_id: str, source: str) -> Optional[Path]:
        """Find Civitai model on NFS"""
        safe_name = model_id.replace("/", "-")
        
        # Find the model directory
        matching_dirs = list(self.nfs_cozy_dir.glob(f"{safe_name}--*"))
        if not matching_dirs:
            return None
        
        original_dir = matching_dirs[0]
        
        # Check for FlashPack version first
        flashpack_path = original_dir.parent / (original_dir.name + FLASHPACK_SUFFIX)
        if flashpack_path.exists():
            return flashpack_path
        
        # Fall back to safetensors
        safetensors_files = list(original_dir.glob("*.safetensors"))
        if safetensors_files:
            return safetensors_files[0]
        
        return original_dir
    
    def _get_local_path(self, model_id: str, source: str) -> Optional[Path]:
        """Get expected local path for a model"""
        nfs_path = self._find_nfs_path(model_id, source)
        if not nfs_path:
            return None
        
        return self._get_local_dest(model_id, source, nfs_path)
    
    def _get_local_dest(self, model_id: str, source: str, nfs_path: Path) -> Path:
        """Get local destination path matching NFS structure"""
        # Keep the same directory/file name, just change base path
        return self.local_cache_dir / nfs_path.name
    
    def _get_size_gb(self, path: Path) -> float:
        """Get size of path in GB"""
        if path.is_file():
            return path.stat().st_size / (1024 ** 3)
        
        total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return total / (1024 ** 3)
    
    async def prefetch_models(
        self,
        model_ids: List[str],
        sources: Dict[str, str]
    ):
        """
        Background task to prefetch models to local cache.
        
        Args:
            model_ids: List of model IDs to prefetch
            sources: Dict mapping model_id → source string
        """
        logger.info(f"🔄 Starting background prefetch for {len(model_ids)} models")
        
        for model_id in model_ids:
            if model_id in self.cached_models:
                continue
            
            source = sources.get(model_id)
            if not source:
                logger.warning(f"No source found for {model_id}, skipping prefetch")
                continue
            
            await self.ensure_local(model_id, source, priority=False)
            
            # Small delay between copies to avoid overwhelming I/O
            await asyncio.sleep(0.1)
        
        logger.info(f"✅ Background prefetch complete. Cached: {len(self.cached_models)} models")
    
    def get_local_path_if_cached(self, model_id: str, source: str) -> Optional[Path]:
        """
        Get local path only if model is already cached.
        Does not trigger a copy.
        
        Args:
            model_id: Model identifier
            source: Source string
            
        Returns:
            Local path if cached, None otherwise
        """
        if model_id not in self.cached_models:
            return None
        
        local_path = self._get_local_path(model_id, source)
        if local_path and local_path.exists():
            return local_path
        
        # Was in set but not on disk - remove from set
        self.cached_models.discard(model_id)
        return None
    
    def is_cached(self, model_id: str) -> bool:
        """Check if model is in local cache"""
        return model_id in self.cached_models
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_size = 0
        if self.local_cache_dir.exists():
            total_size = sum(
                f.stat().st_size 
                for f in self.local_cache_dir.rglob('*') 
                if f.is_file()
            )
        
        return {
            "cached_models": len(self.cached_models),
            "models": list(self.cached_models),
            "total_size_gb": total_size / (1024 ** 3),
            "cache_dir": str(self.local_cache_dir),
            "currently_copying": list(self.copying_models),
        }