"""pgw#1333: the facts stanza a harness resolution carries.

``MintSlot.facts`` has no default — a slot that cannot say whether the catalog
answered or was never asked is unconstructable. Harness endpoints resolve
against no catalog, so they say exactly that, once, here. Every one of them
declares no ``@worker_function(objectives=...)``, so nothing CHECKS these
facts; a harness that starts declaring a serving contract must pass a real
:class:`~gen_worker.serving_facts.ServingFacts` instead (see
``mint_catalog_slot_pgw969``, which does).
"""

from __future__ import annotations

from gen_worker.serving_facts import FactsUnavailable

TEST_FACTS = FactsUnavailable(owed_by="a test harness (no catalog is involved)")

__all__ = ["TEST_FACTS"]
