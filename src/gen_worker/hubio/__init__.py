"""Hub I/O — the worker-wide Tensorhub control-plane adapter.

``client`` translates worker product operations into Tensorhub HTTP requests;
HashRepo owns CAS manifests, objects, transfer sessions, and transfer recovery.
``publish_state`` keeps only the producer-output fact needed to avoid repeating
an expensive cast after an interrupted publish. These modules live outside
``convert/`` because executor, fleet_cells, and request_context also use them.

No re-exports here: the single public facade is ``gen_worker`` itself; deep
imports name the module they mean.
"""
