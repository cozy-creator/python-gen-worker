from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from .cozy_cas import CozyHubClient, CozySnapshotDownloader
from .downloader import ModelDownloader
from .hf_downloader import HuggingFaceHubDownloader
from .model_refs import ParsedModelRef, parse_model_ref


class ModelRefDownloader(ModelDownloader):
    """
    Composite downloader for phase-1 model refs:
      - Cozy Hub snapshots (default): org/repo[:tag] or org/repo@sha256:<digest>
      - Hugging Face repos: hf:org/repo[@revision]

    Returns a local directory path for both schemes.
    """

    def __init__(
        self,
        cozy_base_url: Optional[str] = None,
        cozy_token: Optional[str] = None,
        hf_home: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self._cozy_base_url = (cozy_base_url or "").strip() or None
        self._cozy_token = (cozy_token or "").strip() or None
        self._hf = HuggingFaceHubDownloader(hf_home=hf_home, hf_token=hf_token)
        self._cozy: Optional[CozySnapshotDownloader] = None
        if self._cozy_base_url:
            client = CozyHubClient(self._cozy_base_url, token=self._cozy_token)
            self._cozy = CozySnapshotDownloader(client)

    async def _download_async(self, parsed: ParsedModelRef, dest_dir: Path) -> Path:
        if parsed.scheme == "hf" and parsed.hf is not None:
            return self._hf.download(parsed.hf).local_dir

        if parsed.scheme == "cozy" and parsed.cozy is not None:
            if self._cozy is None:
                raise RuntimeError("cozy downloads require COZY_HUB_URL")
            return await self._cozy.ensure_snapshot(dest_dir, parsed.cozy)

        raise ValueError("invalid parsed model ref")

    def download(self, model_ref: str, dest_dir: str, filename: Optional[str] = None) -> str:
        parsed = parse_model_ref(model_ref)
        # dest_dir is the worker/model cache base directory
        base = Path(dest_dir)
        # We ignore filename; snapshots/refs already define structure.
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Nested loop scenario; run in a new loop in a thread.
                return _run_in_thread(self._download_async(parsed, base))
        except RuntimeError:
            pass
        return asyncio.run(self._download_async(parsed, base)).as_posix()


def _run_in_thread(coro) -> str:
    out: dict[str, str] = {}
    err: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            out["v"] = asyncio.run(coro).as_posix()
        except BaseException as e:
            err["e"] = e

    import threading

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "e" in err:
        raise err["e"]
    return out["v"]
