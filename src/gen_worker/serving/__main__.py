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
import urllib.error
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
    """The thin th#2133 wire: one GET for the answer, one per presigned URL.

    The cozy-local / CI transport. In production the ask is parent-mediated
    (``serving.hub_store.BrokerReleaseGraphTransport``) because the compute
    child holds no credential — this one names a host and a bearer, which is
    exactly what the split exists to take away from a serving process.
    """

    def __init__(
        self, base_url: str, bearer: str = "", timeout_s: float = HTTP_TIMEOUT_S
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.bearer = bearer
        self.timeout_s = timeout_s

    def release_compiled_graphs(
        self, release_id: str, lane: str, sm: str
    ) -> Mapping[str, Any]:
        query = urllib.parse.urlencode({"lane": lane, "sm": sm})
        url = (
            f"{self.base_url}/v1/worker/releases/"
            f"{urllib.parse.quote(release_id, safe='')}/compiled-graphs?{query}"
        )
        request = urllib.request.Request(url)
        if self.bearer:
            request.add_header("Authorization", f"Bearer {self.bearer}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                from .hub_store import ReleaseNotStamped

                raise ReleaseNotStamped(
                    f"release {release_id} carries no stamped compiled-graph "
                    f"document; serving eager"
                ) from exc
            raise SystemExit(
                f"adopt route answered {exc.code} from {url}: "
                f"{exc.read().decode('utf-8', 'replace')[:400]}"
            ) from exc
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
    parser.add_argument("--model", default="",
                        help="the hub row's `model` classification (e.g. sdxl); "
                             "empty = unclassified, serves platform fallbacks")
    parser.add_argument("--defaults", default="{}",
                        help="per-checkpoint override JSON (the hub's deploy state, locally)")
    parser.add_argument("--lane", default="",
                        help="active lane contract handle (the deploy's pick)")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--invoke", action="append", default=[],
                        help="entrypoint name to invoke (repeatable, pairs with --payload)")
    parser.add_argument("--payload", action="append", default=[],
                        help="JSON payload for the matching --invoke")
    parser.add_argument("--envelope", action="append", default=[],
                        help="full request envelope JSON for the matching --invoke "
                             "(signature-derived: model/models, adapters, input); "
                             "runs the ServeLoop with residency leases instead of "
                             "the direct-args host dispatch")
    parser.add_argument("--vram-budget-gb", type=float, default=0.0,
                        help="envelope mode: the residency VRAM budget; 0 = size "
                             "the budget to the checkpoint (fit exactly one)")
    parser.add_argument("--weight-bytes", type=int, default=0,
                        help="envelope mode: this checkpoint's manifest weight "
                             "bytes (the tensorfs manifest states this in "
                             "production; 0 = stat the local tree)")
    # Adoption sources, mutually exclusive; absent = the eager bridge.
    parser.add_argument("--graph-store", default="",
                        help="local compiled-graph CAS root (cozy-local shape)")
    parser.add_argument("--hub-base-url", default="",
                        help="hub base URL for the th#2133 adopt route")
    parser.add_argument("--release", default="", help="endpoint-release id (hub adopt)")
    parser.add_argument("--hub-token", default="",
                        help="worker bearer for the adopt route (local/CI only; "
                             "in production the parent holds the credential)")
    parser.add_argument("--via-broker", action="store_true",
                        help="make the adopt ask through procsplit's action "
                             "broker — the PRODUCTION path, allowlisted as "
                             "`release.compiled_graphs`")
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
        from .hub_store import (
            BrokerReleaseGraphTransport,
            HubGraphStore,
            ReleaseNotStamped,
        )

        if not args.release:
            raise SystemExit("--release is required with --hub-base-url")
        transport: Any = (
            BrokerReleaseGraphTransport(
                base_url=args.hub_base_url, bearer=args.hub_token
            )
            if args.via_broker
            else HttpReleaseGraphTransport(args.hub_base_url, args.hub_token)
        )
        store: Any = HubGraphStore(transport, args.release, args.lane, args.sm)
        try:
            document = store.get_graphs(args.release)
        except ReleaseNotStamped as exc:
            # NOT a boot failure: an un-stamped release serves eager, which is
            # the whole eager bridge. Say it, then take it.
            print(f"adopt: {exc}", file=sys.stderr)
            return None, None
        return store, document
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


def _serve_envelopes(args: argparse.Namespace, loaded: Any) -> int:
    """Envelope mode: the production dispatch loop, locally — the
    signature-derived envelope through ServeLoop with residency leases
    (admission before allocation) around every invocation."""
    from .residency import ResidencyManager
    from .serve_loop import ServeLoop, manifest_sizer

    tree = Path(args.checkpoint)
    weight = args.weight_bytes or sum(
        f.stat().st_size for f in tree.rglob("*") if f.is_file()
    ) or 1
    headroom = max(weight // 4, 1)
    budget = int(args.vram_budget_gb * (1024**3)) or (weight + headroom)

    class _LocalResolver:
        def resolve(self, model_cls: type, checkpoint_ref: str) -> DeployBinding:
            return DeployBinding(
                checkpoint_ref=checkpoint_ref,
                checkpoint_dir=tree,
                model=args.model or None,
                defaults=json.loads(args.defaults),
            )

        def default_pick(self, model_cls: type, slot_name: str) -> str:
            return str(args.checkpoint_ref)

    loop = ServeLoop(
        loaded,
        residency=ResidencyManager(
            budget,
            manifest_sizer({args.checkpoint_ref: weight}, headroom_bytes=headroom),
        ),
        resolver=_LocalResolver(),
        lane_contract=args.lane,
        output_dir=Path(args.output_dir),
    )
    for index, (function, raw) in enumerate(zip(args.invoke, args.envelope)):
        outcome = loop.invoke(function, json.loads(raw), request_id=f"local-{index}")
        for warning in outcome.warnings:
            print(f"warn: {warning}", file=sys.stderr)
        sys.stdout.buffer.write(msgspec.json.encode(outcome.result))
        sys.stdout.buffer.write(b"\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.envelope and args.payload:
        raise SystemExit("--payload and --envelope are two modes; pick one")
    if args.envelope:
        if len(args.invoke) != len(args.envelope):
            raise SystemExit("--invoke and --envelope must pair up")
        return _serve_envelopes(args, load_endpoint(Path(args.endpoint_dir)))
    if len(args.invoke) != len(args.payload):
        raise SystemExit("--invoke and --payload must pair up")
    loaded = load_endpoint(Path(args.endpoint_dir))
    binding = DeployBinding(
        checkpoint_ref=args.checkpoint_ref,
        checkpoint_dir=Path(args.checkpoint),
        model=args.model or None,
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
        report: dict[str, Any] = {
            "adopted": [record.graph for record in host.adoption.adopted],
            "holes": [
                {"graph": hole.record.graph, "reason": hole.reason}
                for hole in host.holes
            ],
        }
        # The hub's own miss list, when the store is the hub: the ORDERED
        # holes pgw#1371's background mint takes as `enable_compiled(holes=)`.
        answered = getattr(store, "misses", None)
        if answered is not None:
            report["answered_misses"] = list(answered)
        print(json.dumps(report), file=sys.stderr)
    for index, (function, raw) in enumerate(zip(args.invoke, args.payload)):
        result = host.dispatch(function, json.loads(raw), request_id=f"local-{index}")
        sys.stdout.buffer.write(msgspec.json.encode(result))
        sys.stdout.buffer.write(b"\n")
    host.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
