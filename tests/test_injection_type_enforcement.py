import unittest

from gen_worker.injection import InjectionSpec, ModelRef, ModelRefSource
from gen_worker.worker import ActionContext, Worker


class _StubModelManager:
    async def get_active_pipeline(self, model_id: str):
        return _ActualPipeline()


class _ExpectedPipeline:
    pass


class _ActualPipeline:
    pass


class TestInjectionTypeEnforcement(unittest.TestCase):
    def test_rejects_model_manager_type_mismatch(self) -> None:
        w = Worker(user_module_names=[], model_manager=_StubModelManager(), worker_jwt="dummy-worker-jwt")
        ctx = ActionContext("run-1")
        inj = InjectionSpec(
            param_name="pipeline",
            param_type=_ExpectedPipeline,
            model_ref=ModelRef(ModelRefSource.FIXED, "foo"),
        )

        with self.assertRaises(ValueError):
            w._resolve_injected_value(ctx, _ExpectedPipeline, "model-id", inj)
