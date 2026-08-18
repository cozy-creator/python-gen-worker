### A successful v2 build warned that it had found nothing (pgw#1389)

`validate_endpoint_lock` learned `entrypoints[]` (th#2146) and folds it into
`functions` at its one site. `discover.py`'s second, standalone emptiness check
did not, so every SUCCESSFUL v2 build printed

    warning: no @endpoint or @job objects found

over a manifest that declares its entrypoints perfectly well. The wording was
wrong twice over: `@endpoint` is precisely the surface v2 replaced, so the line
named the OLD decorator as missing while the NEW one was present and discovered.

Measured on the standing builder: a v2 sdxl publish emitted it, and the same
build admitted. A false alarm on the success path is not free — the pgw#1387
investigation started from exactly these two warning lines.
