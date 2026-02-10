import unittest

import msgspec

from gen_worker.injection import InjectionSpec, ModelRef, ModelRefSource
from gen_worker.worker import Worker


class _Payload(msgspec.Struct):
    model: str


class TestPayloadModelSelection(unittest.TestCase):
    def test_payload_key_must_exist_in_mapping(self) -> None:
        w = Worker(
            user_module_names=[],
            worker_jwt="dummy-worker-jwt",
            manifest={
                "models": {
                    "sd15": "cozy:demo/sd15:latest",
                    "flux": "cozy:demo/flux:latest",
                }
            },
        )
        inj = InjectionSpec(
            param_name="pipeline",
            param_type=object,
            model_ref=ModelRef(ModelRefSource.PAYLOAD, "model"),
        )
        payload = _Payload(model="does-not-exist")
        with self.assertRaises(ValueError) as ctx:
            w._resolve_model_id_for_injection("generate", inj, payload)
        self.assertIn("unknown model key", str(ctx.exception).lower())
        self.assertIn("sd15", str(ctx.exception))

    def test_payload_key_resolves_to_repo_ref(self) -> None:
        w = Worker(
            user_module_names=[],
            worker_jwt="dummy-worker-jwt",
            manifest={"models": {"sd15": "cozy:demo/sd15:latest"}},
        )
        inj = InjectionSpec(
            param_name="pipeline",
            param_type=object,
            model_ref=ModelRef(ModelRefSource.PAYLOAD, "model"),
        )
        payload = _Payload(model="sd15")
        out = w._resolve_model_id_for_injection("generate", inj, payload)
        self.assertEqual(out, "cozy:demo/sd15:latest")
