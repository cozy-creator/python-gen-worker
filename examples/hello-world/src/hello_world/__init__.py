import msgspec

from gen_worker import ActionContext, ResourceRequirements, worker_function


class HelloInput(msgspec.Struct):
    name: str = "world"


class HelloOutput(msgspec.Struct):
    message: str


@worker_function(ResourceRequirements())
def hello(ctx: ActionContext, payload: HelloInput) -> HelloOutput:
    _ = ctx
    return HelloOutput(message=f"hello, {payload.name}")
