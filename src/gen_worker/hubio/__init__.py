"""Hub I/O — the worker-wide tensorhub control-plane and CAS write client.

``client`` is the CAS write client (``HubClient``): chunked-CAS publishes,
commit declares, capability-token auth. ``journal`` is the resumable publish
journal ("never redo the cast to redo the upload"). It lives outside
``convert/`` because the client's consumers — executor, fleet_cells,
request_context — are not conversion code.

No re-exports here: the single public facade is ``gen_worker`` itself; deep
imports name the module they mean.
"""
