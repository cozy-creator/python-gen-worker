# source-hash bust: 2026-06-03T00:00Z (raise on non-marco input for failure-branch e2e)
import msgspec
from gen_worker import RequestContext, ValidationError, inference, invocable


class MarcoPoloInput(msgspec.Struct):
    text: str = ""


class MarcoPoloOutput(msgspec.Struct):
    response: str


@inference()
class MarcoPolo:
    def setup(self) -> None:
        pass

    @invocable(name="marco_polo")
    def marco_polo(self, ctx: RequestContext, data: MarcoPoloInput) -> MarcoPoloOutput:
        """Returns 'polo' when input is 'marco'; otherwise raises so the request fails."""
        # Deterministic minimal handler used for latency tests and for
        # exercising both billing branches: the marco->polo path succeeds
        # (capture), any other input raises ValidationError so the worker
        # reports a non-retryable FAILED job (hold release).
        ctx.raise_if_canceled()

        if str(data.text or "").strip().lower() == "marco":
            return MarcoPoloOutput(response="polo")

        raise ValidationError(f"expected 'marco', got {data.text!r}")
# bust 1780012800
