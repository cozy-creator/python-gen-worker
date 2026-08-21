"""Model bindings for ``@endpoint(models={...})``."""

from __future__ import annotations

from typing import Any, Literal

import msgspec
from msgspec import structs

from ..models.refs import (
    HuggingFaceRef,
    WireRef,
    fold_ref,
    normalize_model_ref,
    parse_model_ref,
    refuse_ref_fragment,
)

ModelSource = Literal["tensorhub", "huggingface", "civitai", "modelscope"]


def _clean(s: object) -> str:
    return str(s or "").strip()


STORAGE_DTYPES: tuple[str, ...] = ("fp8", "fp8+te")


def _clean_storage_dtype(v: object) -> str:
    q = _clean(v).lower()
    if q and q not in STORAGE_DTYPES:
        raise ValueError(
            f"unknown storage_dtype {q!r}; expected one of {STORAGE_DTYPES}"
        )
    return q


class ModelRef(msgspec.Struct, frozen=True):
    """ONE structured model reference: ``source`` is explicit, never inferred from shape or which factory built the value."""

    source: ModelSource
    path: str
    release: str = ""
    revision: str = ""
    subfolder: str = ""
    dtype: str = ""
    storage_dtype: str = ""
    version: str = ""
    files: tuple[str, ...] = ()
    components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        force(self, "path", _clean(self.path))
        force(self, "release", _clean(self.release))
        force(self, "revision", _clean(self.revision))
        force(self, "subfolder", _clean(self.subfolder))
        force(self, "dtype", _clean(self.dtype))
        force(self, "version", _clean(self.version))
        force(self, "files", tuple(_clean(p) for p in self.files if _clean(p)))
        force(self, "components", tuple(_clean(p) for p in self.components if _clean(p)))
        force(self, "storage_dtype", _clean_storage_dtype(self.storage_dtype))

        refuse_ref_fragment(self.path, where=f"{self.source} binding")
        if self.source != "tensorhub" and self.release:
            raise ValueError(
                f"{self.source} bindings have no release axis (release="
                f"{self.release!r}); it is a tensorhub release identifier. Use "
                "revision= for a huggingface/modelscope commit or version= for "
                "a civitai model-version."
            )
        if self.source == "tensorhub":
            if not self.path:
                raise ValueError("Hub requires a non-empty ref")
        elif self.source == "huggingface":
            if "/" not in self.path:
                raise ValueError(f"HF(id=) must be 'owner/repo', got {self.path!r}")
        elif self.source == "civitai":
            if not self.path:
                raise ValueError("Civitai requires a non-empty model id")
            if self.components:
                raise ValueError(
                    "Civitai bindings do not support components= "
                    "(civitai artifacts aren't component-structured)"
                )
        elif self.source == "modelscope":
            if "/" not in self.path:
                raise ValueError(f"ModelScope(id=) must be 'owner/repo', got {self.path!r}")
            if self.components:
                raise ValueError(
                    "ModelScope bindings do not support components= (use files= instead)"
                )
        else:
            raise ValueError(f"unknown ModelRef source {self.source!r}")

    @property
    def label(self) -> str:
        """Human-readable label for ``model_used`` metadata / logging — mirrors the wire form each source's registry keys on."""
        if self.source == "tensorhub":
            return str(fold_ref(self.path, release=self.release))
        if self.source == "civitai":
            return f"{self.path}@{self.version}" if self.version else self.path
        return f"{self.path}@{self.revision}" if self.revision else self.path


def Hub(
    ref: str,
    *,
    release: str = "",
    storage_dtype: str = "",
    components: tuple[str, ...] = (),
) -> ModelRef:
    """Tensorhub-backed binding: ``Hub("owner/repo@<release>", storage_dtype=, components=)``."""
    return ModelRef(
        source="tensorhub", path=ref, release=release,
        storage_dtype=storage_dtype, components=components,
    )


def HF(
    ref: str,
    *,
    revision: str = "",
    dtype: str = "",
    subfolder: str = "",
    files: tuple[str, ...] = (),
    components: tuple[str, ...] = (),
    storage_dtype: str = "",
) -> ModelRef:
    """HuggingFace-backed binding: ``HF(id, revision=, dtype=, subfolder=, files=, components=, storage_dtype=)``."""
    return ModelRef(
        source="huggingface", path=ref, revision=revision, dtype=dtype,
        subfolder=subfolder, files=files, components=components,
        storage_dtype=storage_dtype,
    )


def Civitai(ref: str, *, version: str = "") -> ModelRef:
    """Civitai-backed binding: ``Civitai(model_id, version=)``."""
    return ModelRef(source="civitai", path=ref, version=version)


def ModelScope(
    ref: str, *, revision: str = "", files: tuple[str, ...] = (),
) -> ModelRef:
    """ModelScope-backed binding: ``ModelScope(id, revision=, files=)``."""
    return ModelRef(source="modelscope", path=ref, revision=revision, files=files)


Binding = ModelRef
BINDING_TYPES: tuple[type, ...] = (ModelRef,)


def wire_ref(binding: Binding) -> WireRef:
    """Normal-form ref string for the wire / cache key — delegates to the ONE grammar module (``gen_worker.models.refs``)."""

    if binding.source == "tensorhub":
        return fold_ref(binding.path, release=binding.release)
    if binding.source == "huggingface":
        return HuggingFaceRef(binding.path, binding.revision or None).canonical()
    return WireRef(binding.path)


def rebind_pick(
    binding: Binding,
    *,
    resolved_ref: str = "",
    cast: str = "",
) -> Binding:
    """THE fold of a hub pick into a binding: the HelloAck path (``resolved_ref`` + ``cast``)."""

    rebound: Any = binding
    if resolved_ref:
        parsed = parse_model_ref(resolved_ref)
        if parsed.tensorhub is None:
            raise ValueError(f"resolution {resolved_ref!r} is not a tensorhub ref")
        refuse_ref_fragment(resolved_ref, where="hub resolution")
    try:
        if cast:
            rebound = structs.replace(rebound, storage_dtype=cast)
    except TypeError as exc:
        raise ValueError(f"pick does not fit binding {binding!r}: {exc}") from exc
    if resolved_ref:
        expected = normalize_model_ref(resolved_ref)
        if wire_ref(rebound) != expected:
            raise ValueError(
                f"pick {expected!r} does not round-trip through the binding "
                f"(got {wire_ref(rebound)!r})"
            )
    return rebound


__all__ = [
    "Binding", "BINDING_TYPES", "Civitai", "HF", "Hub", "ModelRef",
    "ModelScope", "STORAGE_DTYPES", "rebind_pick", "wire_ref",
]
