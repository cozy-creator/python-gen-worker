"""``python -m gen_worker.serving`` — boot an endpoint the ship-code-as-is way.

The production entry for the catalog-free serve path, and the cozy-local /
CI shape of the pgw#1372 boot: load the author's module from its endpoint
directory, ``setup(ctx)`` against a local checkpoint tree (EAGER, always),
optionally adopt compiled graphs from a store — a local CAS
(``--graph-store``) or the hub's th#2133 adopt route (``--hub-base-url`` +
``--release``) — then serve the requested invocations and print each result
envelope as JSON. Holes are reported, never fatal; the mint is background
machinery this runner does not own (pgw#1371).

Examples::

    python -m gen_worker.serving ./sdxl \
        --checkpoint /ckpts/dreamshaper --checkpoint-ref ckpt:dreamshaper@2 \
        --invoke generate --payload '{"prompt": "a lighthouse"}'

    python -m gen_worker.serving ./sdxl --lane fp8 --sm sm_89 \
        --graph-store ~/.cache/cozy/compiled-graphs ... --invoke generate ...
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import msgspec

from .context import DeployBinding
from .host import EndpointHost
from .loader import load_endpoint

#: One explicit budget per hub hop; a hung boot pull must fail visibly.
HTTP_TIMEOUT_S = 60.0


class HttpReleaseGraphTransport:
    """The thin th#2133 wire: one GET for the answer, one per presigned URL."""

    def __init__(self, base_url: str, timeout_s: float = HTTP_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def release_compiled_graphs(
        self, release_id: str, lane: str, sm: str
    ) -> Mapping[str, Any]:
        query = urllib.parse.urlencode({"lane": lane, "sm": sm})
        url = (
            f"{self.base_url}/v1/worker/releases/"
            f"{urllib.parse.quote(release_id, safe='')}/compiled-graphs?{query}"
        )
        with urllib.request.urlopen(url, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit(f"adopt route answered non-object JSON from {url}")
        return payload

    def fetch_blob(self, url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=self.timeout_s) as response:
            return bytes(response.read())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gen_worker.serving",
        description="Boot an author endpoint eagerly; adopt compiled graphs when offered.",
    )
    parser.add_argument("endpoint_dir", help="directory holding endpoint.toml")
    parser.add_argument("--checkpoint", required=True, help="local checkpoint tree")
    parser.add_argument("--checkpoint-ref", default="local/checkpoint")
    parser.add_argument("--defaults", default="{}",
                        help="per-checkpoint override JSON (the hub's deploy state, locally)")
    parser.add_argument("--lane", default="",
                        help="active lane contract handle (the deploy's pick)")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--invoke", action="append", default=[],
                        help="handler name to invoke (repeatable, pairs with --payload)")
    parser.add_argument("--payload", action="append", default=[],
                        help="JSON payload for the matching --invoke")
    # Adoption sources, mutually exclusive; absent = the eager bridge.
    parser.add_argument("--graph-store", default="",
                        help="local compiled-graph CAS root (cozy-local shape)")
    parser.add_argument("--hub-base-url", default="",
                        help="hub base URL for the th#2133 adopt route")
    parser.add_argument("--release", default="", help="endpoint-release id (hub adopt)")
    parser.add_argument("--sm", default="", help="this GPU's sm (e.g. sm_89)")
    parser.add_argument("--artifacts-dir", default=".compiled-graphs")
    return parser


def _adoption_source(
    args: argparse.Namespace, module_name: str
) -> tuple[Any, Any]:
    """(store, document) for the boot, or (None, None) — the eager bridge."""
    if not (args.graph_store or args.hub_base_url):
        return None, None
    if not args.sm:
        raise SystemExit("--sm is required to adopt (artifacts are per-sm)")
    if args.hub_base_url:
        from .hub_store import HubGraphStore

        if not args.release:
            raise SystemExit("--release is required with --hub-base-url")
        store: Any = HubGraphStore(
            HttpReleaseGraphTransport(args.hub_base_url), args.release,
            args.lane, args.sm,
        )
        return store, store.get_graphs(args.release)
    from .._vendor.tensorfs import LocalCAS
    from .._vendor.torchcg.store import LocalGraphStore

    store = LocalGraphStore(LocalCAS(Path(args.graph_store)))
    return store, store.get_graphs(module_name)


def _aoti_loader(path: Path, record: Any) -> Any:
    # The AOTInductor runtime load: the packaged compiled graph becomes the
    # module forward for its graph class. Exact-env by construction — the
    # audit already ran before any author code touched an artifact.
    import torch._inductor

    return torch._inductor.aoti_load_package(str(path))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.invoke) != len(args.payload):
        raise SystemExit("--invoke and --payload must pair up")
    loaded = load_endpoint(Path(args.endpoint_dir))
    binding = DeployBinding(
        checkpoint_ref=args.checkpoint_ref,
        checkpoint_dir=Path(args.checkpoint),
        defaults=json.loads(args.defaults),
    )
    host = EndpointHost(
        loaded, binding, lane_contract=args.lane, output_dir=Path(args.output_dir)
    )
    store, document = _adoption_source(args, loaded.module_name)
    host.setup(
        store=store, document=document, sm=args.sm,
        loader=_aoti_loader, artifacts_dir=Path(args.artifacts_dir),
    )
    if host.adoption is not None:
        print(
            json.dumps({
                "adopted": [record.graph for record in host.adoption.adopted],
                "holes": [
                    {"graph": hole.record.graph, "reason": hole.reason}
                    for hole in host.holes
                ],
            }),
            file=sys.stderr,
        )
    for index, (function, raw) in enumerate(zip(args.invoke, args.payload)):
        result = host.dispatch(function, json.loads(raw), request_id=f"local-{index}")
        sys.stdout.buffer.write(msgspec.json.encode(result))
        sys.stdout.buffer.write(b"\n")
    host.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
