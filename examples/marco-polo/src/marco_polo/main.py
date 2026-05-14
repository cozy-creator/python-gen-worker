import msgspec
from gen_worker import RequestContext, inference


class MarcoPoloInput(msgspec.Struct):
    text: str = ""


class MarcoPoloOutput(msgspec.Struct):
    response: str


@inference()
class MarcoPolo:
    def setup(self) -> None:
        pass

    @inference.function(name="marco_polo")
    def marco_polo(self, ctx: RequestContext, data: MarcoPoloInput) -> MarcoPoloOutput:
        """Returns 'polo' when input is 'marco'; otherwise a fallback response."""
        # Deterministic minimal handler used for latency tests.
        ctx.raise_if_canceled()

        if str(data.text or "").strip().lower() == "marco":
            return MarcoPoloOutput(response="polo")

        return MarcoPoloOutput(response="Bro you're supposed to say 'marco'!")

    def shutdown(self) -> None:
        pass
