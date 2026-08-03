"""The endpoint half of a package whose declaration submodule refuses."""

from __future__ import annotations

import msgspec

from gen_worker import RequestContext, endpoint, import_export_declaration

_AOT_DECLARATION_IMPORTED = import_export_declaration(
    ".aot_declaration", package=__package__)


class PkgIn(msgspec.Struct):
    text: str = ""


class PkgOut(msgspec.Struct):
    response: str


@endpoint
class BlockedPkgEndpoint:
    def pkg_echo(self, ctx: RequestContext, data: PkgIn) -> PkgOut:
        return PkgOut(response=f"pkg-served:{data.text}")
