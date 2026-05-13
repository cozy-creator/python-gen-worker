"""Compatibility shim: ConversionContext lives in ``gen_worker.request_context``.

Issue #1 (slim-request-context) replaced the standalone wrapper class with a
proper ``RequestContext`` subclass. This module re-exports the new class so
``from gen_worker.conversion import ConversionContext`` keeps working for
existing conversion endpoints; the wrapper methods (``mktemp``,
``checkpoint_dir``, ``open_output_writer``, ``copy_unconverted_components``,
``cancelled``) are now methods on the subclass.
"""

from __future__ import annotations

from ..request_context import ConversionContext

__all__ = ["ConversionContext"]
