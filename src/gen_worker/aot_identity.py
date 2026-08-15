"""The identity the hub named for one compiled-graph adoption.

Artifact admission, graph-class identity, toolchain compatibility, and graph
witness verification belong to TCG. The worker retains only this typed
projection of the hub's resolve/arm decision so it can require the exact
``compiled_graph_key`` before handing bytes to TCG and preserve receipt
publisher context in its ordering layer.
"""

from __future__ import annotations

from typing import Protocol

import msgspec


class NamesAnArtifact(Protocol):
    """A source that names one compiled graph by already-authorized facts."""

    @property
    def compiled_graph_key(self) -> str: ...

    @property
    def toolchain_digest(self) -> str: ...

    @property
    def env_seal_digest(self) -> str: ...

    @property
    def publisher_org(self) -> str: ...


class ExpectedIdentity(msgspec.Struct, frozen=True):
    """The immutable hub projection threaded to the exact-key TCG arm."""

    compiled_graph_key: str
    toolchain_digest: str
    env_seal_digest: str
    graph_contract_digest: str
    publisher_org: str

    @classmethod
    def named_by(
        cls, named: NamesAnArtifact, graph_contract_digest: str,
    ) -> ExpectedIdentity:
        """Build the projection in one place from the authorized answer."""
        return ExpectedIdentity(
            compiled_graph_key=named.compiled_graph_key,
            toolchain_digest=named.toolchain_digest,
            env_seal_digest=named.env_seal_digest,
            graph_contract_digest=graph_contract_digest,
            publisher_org=named.publisher_org,
        )


__all__ = ["ExpectedIdentity", "NamesAnArtifact"]
