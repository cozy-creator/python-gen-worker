import msgspec
from gen_worker import RequestContext, worker_function


class MarcoPoloInput(msgspec.Struct):
    text: str = ""


class MarcoPoloOutput(msgspec.Struct):
    response: str


@worker_function()
def marco_polo(ctx: RequestContext, data: MarcoPoloInput) -> MarcoPoloOutput:
    """Returns 'polo' when input is 'marco'; otherwise a fallback response."""
    if ctx.is_canceled():
        raise InterruptedError("Request cancelled")

    if str(data.text or "").strip().lower() == "marco":
        return MarcoPoloOutput(response="polo")
    
    return MarcoPoloOutput(response="Bro you're supposed to say 'marco'!")

