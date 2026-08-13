"""Hub I/O — the worker-wide tensorhub control-plane and CAS write client.

``client`` is the CAS write client (``HubClient``): chunked-CAS publishes,
commit declares, capability-token auth. ``journal`` is the resumable publish
journal ("never redo the cast to redo the upload"). Extracted from ``convert/``
(pgw#1206 A1): the client's consumers — executor, fleet_compiled_graphs,
request_context — were never conversion code. Phase B folds the remaining
upload transports into this package.

No re-exports here: the single public facade is ``gen_worker`` itself; deep
imports name the module they mean.
"""
