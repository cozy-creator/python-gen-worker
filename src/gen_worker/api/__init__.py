"""Declaration-layer modules: the @endpoint decorator, bindings, types, errors.

The single public facade is ``gen_worker`` itself — import public names from
there::

    from gen_worker import endpoint, Resources, Slot, ValidationError

This package deliberately re-exports nothing; code that needs a name the
facade does not carry imports the defining module (``gen_worker.api.types``,
``gen_worker.api.errors``, ...).
"""
