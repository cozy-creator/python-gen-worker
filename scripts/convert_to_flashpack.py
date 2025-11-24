#!/usr/bin/env python3
"""
FlashPack Model Conversion Script

Run this on a GPU pod with NFS mounted to /workspace/.cozy-creator/models/

Usage:
    # Phase 2: Test with single model
    python convert_to_flashpack.py --model-id pony.realism --update-db
    
    # Phase 3: Convert all models
    python convert_to_flashpack.py --all --update-db
    
    # Dry run (no actual conversion)
    python convert_to_flashpack.py --all --dry-run
    
    # Force reconversion
    python convert_to_flashpack.py --all --force --update-db
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from flashpack import pack_to_file
import psycopg2
from datetime import datetime
import aiohttp
import asyncio
import hashlib
from urllib.parse import urlparse, parse_qs, unquote
import re
from tqdm import tqdm
import warnings

# Suppress fp16 CPU warnings
warnings.filterwarnings("ignore", message=".*float16.*cpu.*")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===============================================
# CONFIGURATION
# ===============================================

class Config:
    """Configuration for FlashPack conversion"""
    
    # Storage paths
    COZY_MODELS_DIR = "/workspace/.cozy-creator/models"
    HF_CACHE_DIR = os.getenv("HF_HOME", "/workspace/.cache/huggingface/hub")
    
    # FlashPack settings
    FLASHPACK_SUFFIX = ".flashpack"
    DTYPE = torch.float16
    COMPONENT_NAMES = ["unet", "vae", "text_encoder", "text_encoder_2", "transformer"]
    
    # Database
    DB_DSN = os.getenv("DB_DSN")
    
    @staticmethod
    def get_flashpack_dir(base_path: str) -> str:
        """
        Convert a model path to its FlashPack equivalent.
        
        Examples:
        /workspace/.cozy-creator/models/pony.realism--a3f8c12d/
            → /workspace/.cozy-creator/models/pony.realism--a3f8c12d.flashpack/
        
        ~/.cache/huggingface/hub/models--org--model/snapshots/abc/
            → ~/.cache/huggingface/hub/models--org--model.flashpack/
        """
        path = Path(base_path)
        
        # If it's a file, use parent directory
        if path.is_file():
            path = path.parent
        
        # For HF snapshots, go up to repo folder
        if "snapshots" in path.parts:
            # Find the models--org--name part
            parts = list(path.parts)
            repo_idx = None
            for i, part in enumerate(parts):
                if part.startswith("models--"):
                    repo_idx = i
                    break
            
            if repo_idx is not None:
                # Reconstruct path up to repo
                path = Path(*parts[:repo_idx + 1])
        
        # Create FlashPack sibling directory
        flashpack_name = path.name + Config.FLASHPACK_SUFFIX
        flashpack_path = path.parent / flashpack_name
        
        return str(flashpack_path)


# ===============================================
# MODEL DOWNLOADER
# ===============================================

class ModelDownloader:
    """Downloads models from HuggingFace or Civitai"""
    
    def __init__(self):
        self.cozy_dir = Path(Config.COZY_MODELS_DIR)
        self.hf_dir = Path(Config.HF_CACHE_DIR)
        self.civitai_api_key = os.getenv("CIVITAI_API_KEY")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def download_if_needed(self, model_id: str, source: str) -> Optional[str]:
        """Download model if not present, return path"""
        
        # Check if already downloaded
        path = self._find_existing_model(model_id, source)
        if path:
            logger.info(f"✓ Model already downloaded: {model_id}")
            return path
        
        # Download based on source type
        if source.startswith("hf:"):
            return await self._download_hf(source[3:])
        elif "civitai.com" in source:
            return await self._download_civitai(model_id, source)
        else:
            logger.error(f"Unknown source type for {model_id}: {source}")
            return None
    
    def _find_existing_model(self, model_id: str, source: str) -> Optional[str]:
        """Check if model is already downloaded"""
        if source.startswith("hf:"):
            return self._find_hf_model(source[3:])
        else:
            return self._find_civitai_model(model_id)
    
    def _find_hf_model(self, repo_id: str) -> Optional[str]:
        """Find HuggingFace model on disk"""
        folder_name = f"models--{repo_id.replace('/', '--')}"
        repo_path = self.hf_dir / "hub" / folder_name
        
        snapshots_dir = repo_path / "snapshots"
        if not snapshots_dir.exists():
            return None
        
        snapshots = sorted(snapshots_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if not snapshots:
            return None
        
        return str(snapshots[0])
    
    def _find_civitai_model(self, model_id: str) -> Optional[str]:
        """Find Civitai model on disk"""
        safe_name = model_id.replace("/", "-")
        matching_dirs = list(self.cozy_dir.glob(f"{safe_name}--*"))
        
        if not matching_dirs:
            return None
        
        model_dir = matching_dirs[0]
        safetensors_files = list(model_dir.glob("*.safetensors"))
        if safetensors_files:
            return str(safetensors_files[0])
        
        return str(model_dir)
    
    async def _download_hf(self, repo_id: str) -> Optional[str]:
        """Download from HuggingFace"""
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"📥 Downloading from HuggingFace: {repo_id}")
            
            # Download the model
            path = await asyncio.to_thread(
                snapshot_download,
                repo_id,
                cache_dir=str(self.hf_dir.parent),  # HF adds 'hub' subfolder
            )
            
            logger.info(f"✓ Downloaded HF model to: {path}")
            return path
            
        except Exception as e:
            logger.error(f"Failed to download HF model {repo_id}: {e}")
            return None
    
    async def _download_civitai(self, model_id: str, url: str) -> Optional[str]:
        """Download from Civitai"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Get download URL if needed
            download_url = await self._get_civitai_download_url(url)
            if not download_url:
                return None
            
            # Get filename from redirect
            filename = await self._get_civitai_filename(download_url)
            if not filename:
                filename = f"{model_id}.safetensors"
            
            # Create destination path
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            safe_name = model_id.replace("/", "-")
            dest_dir = self.cozy_dir / f"{safe_name}--{url_hash}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / filename
            
            logger.info(f"📥 Downloading from Civitai: {model_id}")
            logger.info(f"   Destination: {dest_path}")
            
            # Download with progress
            await self._download_with_progress(download_url, str(dest_path))
            
            logger.info(f"✓ Downloaded Civitai model to: {dest_path}")
            return str(dest_path)
            
        except Exception as e:
            logger.error(f"Failed to download Civitai model {model_id}: {e}")
            return None
    
    async def _get_civitai_download_url(self, url: str) -> Optional[str]:
        """Convert Civitai URL to download URL"""
        if "/api/download/" in url:
            return url
        
        try:
            parsed = urlparse(url)
            path_parts = parsed.path.split("/")
            
            # Extract model ID from URL
            model_idx = None
            for i, part in enumerate(path_parts):
                if part == "models" and i + 1 < len(path_parts):
                    model_idx = i + 1
                    break
            
            if model_idx is None:
                logger.error(f"Could not parse Civitai URL: {url}")
                return None
            
            model_number = path_parts[model_idx].split("?")[0]
            api_url = f"https://civitai.com/api/v1/models/{model_number}"
            
            headers = {}
            if self.civitai_api_key:
                headers["Authorization"] = f"Bearer {self.civitai_api_key}"
            
            async with self.session.get(api_url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Civitai API error: {response.status}")
                    return None
                
                data = await response.json()
                if "modelVersions" in data and len(data["modelVersions"]) > 0:
                    return data["modelVersions"][0]["downloadUrl"]
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting Civitai download URL: {e}")
            return None
    
    async def _get_civitai_filename(self, url: str) -> Optional[str]:
        """Get original filename from Civitai redirect"""
        try:
            headers = {}
            if self.civitai_api_key:
                headers["Authorization"] = f"Bearer {self.civitai_api_key}"
            
            async with self.session.get(url, headers=headers, allow_redirects=False) as response:
                if response.status in (301, 302, 307):
                    location = response.headers.get("location")
                    if location:
                        parsed = urlparse(location)
                        query_params = parse_qs(parsed.query)
                        
                        content_disp = query_params.get("response-content-disposition", [None])[0]
                        if content_disp:
                            match = re.search(r'filename="([^"]+)"', content_disp)
                            if match:
                                return unquote(match.group(1))
                        
                        if parsed.path:
                            return os.path.basename(parsed.path)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting Civitai filename: {e}")
            return None
    
    async def _download_with_progress(self, url: str, dest_path: str):
        """Download file with progress bar"""
        headers = {}
        if self.civitai_api_key:
            headers["Authorization"] = f"Bearer {self.civitai_api_key}"
        
        temp_path = dest_path + ".tmp"
        
        # Check for partial download
        initial_size = 0
        if os.path.exists(temp_path):
            initial_size = os.path.getsize(temp_path)
            if initial_size > 0:
                headers["Range"] = f"bytes={initial_size}-"
                logger.info(f"Resuming download from byte {initial_size}")
        
        timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=60)
        
        async with self.session.get(url, headers=headers, timeout=timeout) as response:
            if initial_size > 0 and response.status == 206:
                total_size = initial_size + int(response.headers.get("content-length", 0))
            elif response.status == 200:
                total_size = int(response.headers.get("content-length", 0))
                initial_size = 0
            else:
                raise Exception(f"Download failed with status {response.status}")
            
            mode = "ab" if initial_size > 0 else "wb"
            
            with tqdm(total=total_size, initial=initial_size, unit="iB", unit_scale=True) as pbar:
                with open(temp_path, mode) as f:
                    async for chunk in response.content.iter_chunked(8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Move to final destination
            os.rename(temp_path, dest_path)


# ===============================================
# MODEL PATH RESOLVER
# ===============================================

class ModelPathResolver:
    """Resolves model paths from pipeline_defs config"""
    
    def __init__(self, downloader: Optional[ModelDownloader] = None):
        self.cozy_dir = Path(Config.COZY_MODELS_DIR)
        self.hf_dir = Path(Config.HF_CACHE_DIR)
        self.downloader = downloader
    
    async def get_model_path(self, model_id: str, source: str) -> Optional[str]:
        """
        Get the source path for a model, downloading if needed.
        
        Args:
            model_id: Model identifier (e.g., "pony.realism")
            source: Source URL/identifier from pipeline_defs
            
        Returns:
            Path to model or None if not found
        """
        # First check if model exists on disk
        if source.startswith("hf:"):
            path = self._get_hf_model_path(source[3:])
        else:
            path = self._get_civitai_model_path(model_id)
        
        # If found, return it
        if path:
            return path
        
        # If not found and we have a downloader, try downloading
        if self.downloader:
            logger.info(f"Model not found locally, attempting download...")
            path = await self.downloader.download_if_needed(model_id, source)
            return path
        
        return None
    
    def get_model_path_sync(self, model_id: str, source: str) -> Optional[str]:
        """Synchronous version - just checks disk, no download"""
        if source.startswith("hf:"):
            return self._get_hf_model_path(source[3:])
        else:
            return self._get_civitai_model_path(model_id)
    
    def _get_hf_model_path(self, repo_id: str) -> Optional[str]:
        """Get HuggingFace model path"""
        # Handle different huggingface_hub versions
        try:
            from huggingface_hub.file_download import repo_folder_name
            # Try new API first (no args, returns function)
            try:
                folder_name = repo_folder_name(repo_id=repo_id, repo_type="model")
            except TypeError:
                # Old API: repo_folder_name(repo_id, repo_type)
                folder_name = repo_folder_name(repo_id, "model")
        except ImportError:
            # Fallback: construct manually
            folder_name = f"models--{repo_id.replace('/', '--')}"
        
        repo_path = self.hf_dir / "hub" / folder_name
        
        # Find the snapshot directory
        snapshots_dir = repo_path / "snapshots"
        if not snapshots_dir.exists():
            logger.warning(f"HF model not found: {repo_id} (looked in {repo_path})")
            return None
        
        # Get the latest snapshot
        snapshots = sorted(snapshots_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if not snapshots:
            logger.warning(f"No snapshots found for HF model: {repo_id}")
            return None
        
        return str(snapshots[0])
    
    def _get_civitai_model_path(self, model_id: str) -> Optional[str]:
        """Get Civitai/direct download model path"""
        safe_name = model_id.replace("/", "-")
        
        # Look for directories matching pattern: {safe_name}--{hash}
        matching_dirs = list(self.cozy_dir.glob(f"{safe_name}--*"))
        
        if not matching_dirs:
            logger.warning(f"Civitai model not found: {model_id}")
            return None
        
        model_dir = matching_dirs[0]
        
        # Look for safetensors file
        safetensors_files = list(model_dir.glob("*.safetensors"))
        if safetensors_files:
            return str(safetensors_files[0])
        
        # Or return the directory itself
        return str(model_dir)


# ===============================================
# FLASHPACK CONVERTER
# ===============================================

class FlashPackConverter:
    """Converts models to FlashPack format"""
    
    def __init__(self, dtype: torch.dtype = torch.float16, downloader: Optional[ModelDownloader] = None):
        self.dtype = dtype
        self.downloader = downloader
        self.path_resolver = ModelPathResolver(downloader)
    
    async def convert_model(
        self,
        model_id: str,
        source: str,
        force: bool = False,
        dry_run: bool = False
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Convert a model to FlashPack format.
        
        Args:
            model_id: Model identifier
            source: Source from pipeline_defs
            force: Overwrite existing FlashPack
            dry_run: Don't actually convert, just check
            
        Returns:
            (success, flashpack_path, stats)
        """
        stats = {
            "load_time": 0.0,
            "convert_time": 0.0,
            "original_size_gb": 0.0,
            "flashpack_size_gb": 0.0
        }
        
        try:
            # Get source path (may download if needed)
            source_path = await self.path_resolver.get_model_path(model_id, source)
            if not source_path:
                logger.error(f"❌ Source not found for {model_id}")
                return False, "", stats
            
            # Determine output path
            output_path = Config.get_flashpack_dir(source_path)
            output_dir = Path(output_path)
            
            # Check if already exists
            if output_dir.exists() and not force:
                logger.info(f"⚡ FlashPack already exists for {model_id}: {output_dir}")
                # Calculate size
                stats["flashpack_size_gb"] = self._get_dir_size_gb(output_dir)
                return True, str(output_dir), stats
            
            if dry_run:
                logger.info(f"[DRY RUN] Would convert {model_id}")
                logger.info(f"           Source: {source_path}")
                logger.info(f"           Output: {output_dir}")
                return True, str(output_dir), stats
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 Converting {model_id} to FlashPack")
            logger.info(f"{'='*80}")
            logger.info(f"Source: {source_path}")
            logger.info(f"Output: {output_dir}")
            
            # Calculate original size
            stats["original_size_gb"] = self._get_path_size_gb(source_path)
            logger.info(f"Original size: {stats['original_size_gb']:.2f} GB")
            
            # Load the pipeline
            start_time = datetime.now()
            pipe = self._load_pipeline(source_path)
            if not pipe:
                raise ValueError(f"Failed to load pipeline from {source_path}")
            
            stats["load_time"] = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ Loaded in {stats['load_time']:.2f} seconds")
            
            # Move to CPU to save GPU memory during conversion
            pipe.to("cpu")
            
            # Start conversion
            start_time = datetime.now()
            
            # Save pipeline config (scheduler, tokenizer, etc.)
            pipeline_config_dir = output_dir / "pipeline"
            logger.info(f"📦 Saving pipeline config → {pipeline_config_dir}")
            pipe.save_pretrained(str(pipeline_config_dir))
            
            # Convert each component
            converted_components = []
            for component_name in Config.COMPONENT_NAMES:
                if hasattr(pipe, component_name):
                    component = getattr(pipe, component_name)
                    if component is not None:
                        success = self._pack_component(component, component_name, output_dir)
                        if success:
                            converted_components.append(component_name)
            
            stats["convert_time"] = (datetime.now() - start_time).total_seconds()
            stats["flashpack_size_gb"] = self._get_dir_size_gb(output_dir)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ Conversion Complete: {model_id}")
            logger.info(f"{'='*80}")
            logger.info(f"Components: {converted_components}")
            logger.info(f"Load time: {stats['load_time']:.2f}s")
            logger.info(f"Convert time: {stats['convert_time']:.2f}s")
            logger.info(f"Original size: {stats['original_size_gb']:.2f} GB")
            logger.info(f"FlashPack size: {stats['flashpack_size_gb']:.2f} GB")
            logger.info(f"Savings: {(1 - stats['flashpack_size_gb']/stats['original_size_gb'])*100:.1f}%")
            logger.info(f"Location: {output_dir}\n")
            
            return True, str(output_dir), stats
            
        except Exception as e:
            logger.error(f"❌ Conversion failed for {model_id}: {e}")
            logger.exception("Full traceback:")
            return False, "", stats
    
    def _load_pipeline(self, source_path: str) -> Optional[DiffusionPipeline]:
        """Load pipeline from source"""
        path = Path(source_path)
        
        try:
            if path.is_file() and path.suffix == ".safetensors":
                # Single file (safetensors)
                logger.info(f"Loading from single file: {path.name}")
                return StableDiffusionXLPipeline.from_single_file(
                    str(path),
                    torch_dtype=self.dtype
                )
            else:
                # Directory (HF format)
                logger.info(f"Loading from directory: {path}")
                
                # Try loading with different approaches
                try:
                    # Approach 1: Auto-detect pipeline type
                    pipe = DiffusionPipeline.from_pretrained(
                        str(path),
                        torch_dtype=self.dtype,
                        local_files_only=True
                    )
                    return pipe
                except Exception as e1:
                    logger.warning(f"Auto-detect failed: {e1}, trying SDXL pipeline...")
                    
                    try:
                        # Approach 2: Force SDXL pipeline
                        pipe = StableDiffusionXLPipeline.from_pretrained(
                            str(path),
                            torch_dtype=self.dtype,
                            local_files_only=True
                        )
                        return pipe
                    except Exception as e2:
                        logger.warning(f"SDXL pipeline failed: {e2}, trying with variant...")
                        
                        try:
                            # Approach 3: Try with fp16 variant
                            pipe = DiffusionPipeline.from_pretrained(
                                str(path),
                                torch_dtype=self.dtype,
                                variant="fp16",
                                local_files_only=True
                            )
                            return pipe
                        except Exception as e3:
                            logger.error(f"All loading approaches failed:")
                            logger.error(f"  Auto-detect: {e1}")
                            logger.error(f"  SDXL: {e2}")
                            logger.error(f"  With variant: {e3}")
                            return None
                            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _pack_component(self, component, name: str, output_dir: Path) -> bool:
        """Pack a single component to FlashPack format"""
        try:
            output_file = output_dir / f"{name}.flashpack"
            logger.info(f"🧱 Packing {name} → {output_file.name}")
            
            pack_to_file(component, str(output_file), target_dtype=self.dtype)
            
            size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"   ✓ {name} packed ({size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"   ✗ Failed to pack {name}: {e}")
            return False
    
    def _get_path_size_gb(self, path: str) -> float:
        """Get size of path in GB"""
        path_obj = Path(path)
        if path_obj.is_file():
            return path_obj.stat().st_size / (1024 ** 3)
        elif path_obj.is_dir():
            return self._get_dir_size_gb(path_obj)
        return 0.0
    
    def _get_dir_size_gb(self, directory: Path) -> float:
        """Get total size of directory in GB"""
        total = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
        return total / (1024 ** 3)


# ===============================================
# DATABASE UPDATER
# ===============================================

class DatabaseUpdater:
    """Updates PostgreSQL with FlashPack availability"""
    
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.conn = None
    
    def connect(self):
        """Connect to database"""
        if not self.dsn:
            logger.warning("No DB_DSN provided, skipping database updates")
            return False
        
        try:
            self.conn = psycopg2.connect(self.dsn)
            logger.info("✓ Connected to PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def update_model(self, model_id: str, flashpack_path: str):
        """Mark a model as having FlashPack"""
        if not self.conn:
            return
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE pipeline_defs 
                    SET has_flashpack = TRUE,
                        updated_at = NOW()
                    WHERE name = %s
                """, (model_id,))
            
            self.conn.commit()
            logger.info(f"✓ Database updated: {model_id} → has_flashpack = TRUE")
            
        except Exception as e:
            logger.error(f"Failed to update database for {model_id}: {e}")
            self.conn.rollback()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("✓ Database connection closed")


# ===============================================
# BATCH CONVERTER
# ===============================================

class BatchConverter:
    """Batch convert multiple models"""
    
    def __init__(self, update_db: bool = False, download: bool = False):
        self.update_db = update_db
        self.download = download
        self.db_updater = DatabaseUpdater(Config.DB_DSN) if update_db else None
        self.results: Dict[str, Tuple[bool, str, Dict]] = {}
    
    async def convert_all(
        self,
        models: Dict[str, dict],
        force: bool = False,
        dry_run: bool = False
    ):
        """Convert all models"""
        
        if self.db_updater:
            if not self.db_updater.connect():
                logger.warning("Proceeding without database updates")
                self.db_updater = None
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 Starting FlashPack Batch Conversion")
        logger.info(f"{'='*80}")
        logger.info(f"Models to convert: {len(models)}")
        logger.info(f"Force overwrite: {force}")
        logger.info(f"Dry run: {dry_run}")
        logger.info(f"Download missing: {self.download}")
        logger.info(f"Database updates: {self.db_updater is not None}")
        logger.info(f"{'='*80}\n")
        
        # Create downloader context if needed
        if self.download:
            async with ModelDownloader() as downloader:
                converter = FlashPackConverter(downloader=downloader)
                await self._process_models(converter, models, force, dry_run)
        else:
            converter = FlashPackConverter()
            await self._process_models(converter, models, force, dry_run)
        
        if self.db_updater:
            self.db_updater.close()
        
        self._print_summary()
    
    async def _process_models(
        self,
        converter: FlashPackConverter,
        models: Dict[str, dict],
        force: bool,
        dry_run: bool
    ):
        """Process each model"""
        for i, (model_id, config) in enumerate(models.items(), 1):
            source = config.get("source", "")
            
            if not source:
                logger.warning(f"⚠️  [{i}/{len(models)}] Skipping {model_id}: No source defined")
                self.results[model_id] = (False, "no_source", {})
                continue
            
            logger.info(f"\n[{i}/{len(models)}] Processing: {model_id}")
            
            success, path, stats = await converter.convert_model(
                model_id,
                source,
                force=force,
                dry_run=dry_run
            )
            
            self.results[model_id] = (success, path, stats)
            
            if success and self.db_updater and not dry_run:
                self.db_updater.update_model(model_id, path)
    
    def _print_summary(self):
        """Print conversion summary"""
        successful = []
        failed = []
        total_original_gb = 0.0
        total_flashpack_gb = 0.0
        
        for model_id, (success, path, stats) in self.results.items():
            if success:
                successful.append((model_id, path, stats))
                total_original_gb += stats.get("original_size_gb", 0)
                total_flashpack_gb += stats.get("flashpack_size_gb", 0)
            else:
                failed.append((model_id, path))
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 CONVERSION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total models: {len(self.results)}")
        logger.info(f"✅ Successful: {len(successful)}")
        logger.info(f"❌ Failed: {len(failed)}")
        
        if total_original_gb > 0:
            savings_pct = (1 - total_flashpack_gb / total_original_gb) * 100
            logger.info(f"\n💾 Storage:")
            logger.info(f"Original total: {total_original_gb:.2f} GB")
            logger.info(f"FlashPack total: {total_flashpack_gb:.2f} GB")
            logger.info(f"Savings: {total_original_gb - total_flashpack_gb:.2f} GB ({savings_pct:.1f}%)")
        
        if successful:
            logger.info(f"\n✅ Successfully Converted:")
            for model_id, path, stats in successful:
                logger.info(f"   • {model_id}")
                if stats.get("flashpack_size_gb", 0) > 0:
                    logger.info(f"     → {stats['flashpack_size_gb']:.2f} GB")
        
        if failed:
            logger.info(f"\n❌ Failed:")
            for model_id, reason in failed:
                logger.info(f"   • {model_id}: {reason}")
        
        logger.info(f"\n{'='*80}\n")


# ===============================================
# MAIN CLI
# ===============================================

def load_models_from_db() -> Dict[str, dict]:
    """Load model configurations from PostgreSQL"""
    dsn = Config.DB_DSN
    if not dsn:
        logger.error("DB_DSN environment variable not set!")
        return {}
    
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT name, source, class_name, custom_pipeline, 
                   default_args, metadata, components, estimated_size_bytes
            FROM pipeline_defs
            WHERE source IS NOT NULL AND source != ''
        """)
        
        models = {}
        for row in cur.fetchall():
            name, source, class_name, custom_pipeline, default_args, metadata, components, size = row
            models[name] = {
                "source": source,
                "class_name": class_name,
                "custom_pipeline": custom_pipeline,
                "default_args": default_args or {},
                "metadata": metadata or {},
                "components": components or {},
                "estimated_size_gb": float(size) if size else 0
            }
        
        cur.close()
        conn.close()
        
        logger.info(f"✓ Loaded {len(models)} models from database")
        return models
        
    except Exception as e:
        logger.error(f"Failed to load models from database: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Convert models to FlashPack format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model-id",
        help="Convert single model by ID"
    )
    
    parser.add_argument(
        "--models",
        help="Convert specific models (comma-separated list, e.g., 'model1,model2,model3')"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all models from database"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing FlashPack files"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without actually converting"
    )
    
    parser.add_argument(
        "--update-db",
        action="store_true",
        help="Update PostgreSQL has_flashpack column"
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing models before conversion"
    )
    
    args = parser.parse_args()
    
    if not args.model_id and not args.models and not args.all:
        parser.print_help()
        sys.exit(1)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️  No GPU detected, conversion may be slower")
    
    # Load models from database
    all_models = load_models_from_db()
    if not all_models:
        logger.error("No models found in database!")
        sys.exit(1)
    
    if args.model_id:
        # Single model conversion
        if args.model_id not in all_models:
            logger.error(f"Model '{args.model_id}' not found in database!")
            logger.info(f"Available models: {list(all_models.keys())}")
            sys.exit(1)
        
        models_to_convert = {args.model_id: all_models[args.model_id]}
    
    elif args.models:
        # Multiple specific models
        model_list = [m.strip() for m in args.models.split(',')]
        models_to_convert = {}
        
        for model_id in model_list:
            if model_id not in all_models:
                logger.warning(f"Model '{model_id}' not found in database, skipping")
            else:
                models_to_convert[model_id] = all_models[model_id]
        
        if not models_to_convert:
            logger.error("None of the specified models found in database!")
            sys.exit(1)
        
        logger.info(f"Will convert {len(models_to_convert)} models: {list(models_to_convert.keys())}")
    
    else:
        # All models
        models_to_convert = all_models
    
    # Run conversion
    batch_converter = BatchConverter(update_db=args.update_db, download=args.download)
    asyncio.run(batch_converter.convert_all(
        models_to_convert,
        force=args.force,
        dry_run=args.dry_run
    ))


if __name__ == "__main__":
    main()