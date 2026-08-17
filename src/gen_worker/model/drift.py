"""The drift assertion: does the MINT still describe the DECLARATION?

torchcg G16 puts the declaration first and the recipe second, which leaves
exactly one question a mint can answer that a declaration cannot: did the thing
that actually compiled match the thing the bindings were generated against?

That question has a real failure behind it. A declaration changes in a PR; the
bindings regenerate; the artifacts on the fleet were minted from the previous
declaration and are still keyed, entitled and downloadable. Nothing about the
artifact says it is stale, because a compiled-graph key folds the graph, the SM
and the toolchain — and the graph axis moved for a reason the KEY cannot
distinguish from any other re-mint. So the check is: same runners, same ingress
digests, or refuse.

Two things this module deliberately does not do. It does not compare class
hashes, because the declaration-time export has none to compare with (a class
hash folds ``target`` and device placement; a CPU fake trace cannot produce the
one an ``sm_86`` mint will). And it does not RANK: a recipe either restates the
declaration or it does not.
"""

from __future__ import annotations

from .._vendor.torchcg.recipe import Recipe, RecipeError, RecipeReference
from .errors import ModelError, ModelRefusal
from .snapshot import ModelExport


def assert_recipe(export: ModelExport, recipe: Recipe) -> None:
    """Refuse a mint-emitted recipe that drifted from the binding source.

    Raises :class:`ModelError` with :attr:`ModelRefusal.DECLARATION_DRIFT` on
    a missing runner, an extra one, or a class minted against a different call.
    Equal ingress digests imply equal signatures, because the signature is a
    projection of the ingress — so this is the whole comparison, not a sample
    of it.
    """

    if str(recipe.family) != str(export.family):
        raise ModelError(
            ModelRefusal.DECLARATION_DRIFT,
            f"recipe names family {str(recipe.family)!r}; these bindings were generated "
            f"from {str(export.family)!r}",
        )
    try:
        recipe.assert_declaration(export.declared_runners())
    except RecipeError as exc:
        raise ModelError(ModelRefusal.DECLARATION_DRIFT, str(exc)) from exc


def assert_reference(recipe: Recipe, reference: RecipeReference) -> None:
    """Refuse a digest-pinned reference that does not name this exact recipe."""

    try:
        recipe.verify(reference)
    except RecipeError as exc:
        raise ModelError(ModelRefusal.DECLARATION_DRIFT, str(exc)) from exc


__all__ = ["assert_recipe", "assert_reference"]
