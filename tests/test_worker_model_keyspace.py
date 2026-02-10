import unittest

from gen_worker.injection import InjectionSpec, ModelRef, ModelRefSource
from gen_worker.worker import Worker


class _Payload:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestWorkerModelKeyspace(unittest.TestCase):
    def test_payload_model_selection_uses_per_function_mapping(self) -> None:
        w = Worker.__new__(Worker)
        w._model_id_by_key_by_function = {"generate": {"sdxl": "hf:stabilityai/stable-diffusion-xl-base-1.0"}}
        w._release_model_id_by_key = {}
        w._release_allowed_model_ids = None

        inj = InjectionSpec(
            param_name="pipe",
            param_type=object,
            model_ref=ModelRef(ModelRefSource.PAYLOAD, "model_key"),
        )
        payload = _Payload(model_key="sdxl")

        got = Worker._resolve_model_id_for_injection(w, "generate", inj, payload)  # type: ignore[arg-type]
        self.assertEqual(got, "hf:stabilityai/stable-diffusion-xl-base-1.0")

    def test_payload_model_selection_rejects_unknown_key(self) -> None:
        w = Worker.__new__(Worker)
        w._model_id_by_key_by_function = {"generate": {"sdxl": "hf:stabilityai/stable-diffusion-xl-base-1.0"}}
        w._release_model_id_by_key = {}
        w._release_allowed_model_ids = None

        inj = InjectionSpec(
            param_name="pipe",
            param_type=object,
            model_ref=ModelRef(ModelRefSource.PAYLOAD, "model_key"),
        )
        payload = _Payload(model_key="nope")

        with self.assertRaises(ValueError):
            Worker._resolve_model_id_for_injection(w, "generate", inj, payload)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

