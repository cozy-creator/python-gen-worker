"""pgw#904: the rig's second-process adoption fetches the EXACT compiled graph the
publish leg produced — by checkpoint id, never by listing. Catalog discovery
(`aot_compiled_graphs`) is deleted; a rig that re-listed the family repo and ranked
rows would be re-growing the resolver the cutover removed. The rig driver
KNOWS what it published, so the adopting process is told, exactly as a
serving pod is told by ``Arm.artifact``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import requests


def fetch_named_compiled_graph(
    base: str, family: str, checkpoint_id: str, cache_dir: Path,
) -> Path:
    """Resolve ONE named checkpoint and download its digest-verified tarball."""
    from gen_worker import compile_cache as cc

    repo = cc.system_repo(family)
    resp = requests.get(
        f"{base}/api/v1/repos/{repo}/resolve",
        params={"digest": checkpoint_id}, timeout=30)
    resp.raise_for_status()
    files = (resp.json() or {}).get("files") or []
    entry: Any = next(
        f for f in files if str(f.get("path") or "").endswith(".tar.gz"))
    dest_dir = Path(cache_dir) / "graphs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw = requests.get(str(entry["url"]), timeout=120).content
    want = str(entry.get("digest") or "")
    got = "sha256:" + hashlib.sha256(raw).hexdigest()
    if want and got != want:
        raise RuntimeError(
            f"named compiled graph {checkpoint_id} bytes refused: {got} != {want}")
    dest = dest_dir / f"{checkpoint_id.split(':', 1)[-1]}.tar.gz"
    dest.write_bytes(raw)
    return dest


__all__ = ["fetch_named_compiled_graph"]
