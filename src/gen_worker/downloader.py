import asyncio
import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import aiohttp
import backoff
from tqdm import tqdm


class ModelDownloader(ABC):
    @abstractmethod
    async def download(self, model_ref: str, dest_dir: str, filename: Optional[str] = None) -> str:
        """Download a model artifact and return the local file path."""
        raise NotImplementedError


@dataclass
class DownloadResult:
    path: str
    sha256: Optional[str] = None
    bytes_written: int = 0


class CozyHubDownloader(ModelDownloader):
    """
    Simple async downloader for Cozy hub model artifacts.

    If model_ref is a full URL (http/https), it is fetched directly.
    Otherwise, it is appended to base_url as a path segment.
    """

    def __init__(self, base_url: str, token: Optional[str] = None, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def _resolve_url(self, model_ref: str) -> str:
        ref = model_ref.strip()
        if ref.startswith("http://") or ref.startswith("https://"):
            return ref
        if not self.base_url:
            raise ValueError("COZY_HUB_URL is required for non-URL model_ref")
        return f"{self.base_url}/{ref.lstrip('/')}"

    def _default_filename(self, url: str) -> str:
        path = urlparse(url).path
        name = os.path.basename(path)
        return name or "model.bin"

    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=5)
    async def download(self, model_ref: str, dest_dir: str, filename: Optional[str] = None) -> str:
        url = self._resolve_url(model_ref)
        os.makedirs(dest_dir, exist_ok=True)
        target_name = filename or self._default_filename(url)
        target_path = os.path.join(dest_dir, target_name)

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        sha256 = hashlib.sha256()
        bytes_written = 0

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                total = resp.content_length or 0
                progress = tqdm(total=total, unit="B", unit_scale=True, desc=f"download {target_name}")
                with open(target_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(1 << 20):
                        if not chunk:
                            continue
                        f.write(chunk)
                        sha256.update(chunk)
                        bytes_written += len(chunk)
                        progress.update(len(chunk))
                progress.close()

        return target_path
