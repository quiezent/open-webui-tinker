import asyncio
import types
import unittest

from tinker_native_pipeline import Pipe


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "|".join([f"{m['role']}:{m['content']}" for m in messages]) + "|assistant:"

    def encode(self, text, add_special_tokens=True):
        return [1, 2, 3]

    def decode(self, tokens):
        return "decoded output"


class FakeSequence:
    tokens = [10, 11]


class FakeResult:
    sequences = [FakeSequence()]


class FakeFuture:
    def result(self, timeout=None):
        return FakeResult()


class FakeSampler:
    def __init__(self):
        self.last_kwargs = None

    def get_tokenizer(self):
        return FakeTokenizer()

    def sample(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeFuture()


class FakeServiceClient:
    def __init__(self):
        self.sampler = FakeSampler()
        self.model_paths = []

    def create_sampling_client(self, model_path):
        self.model_paths.append(model_path)
        return self.sampler


class PipelineTests(unittest.TestCase):
    def build_pipe(self):
        pipe = Pipe()
        fake_service = FakeServiceClient()
        fake_tinker = types.SimpleNamespace(
            ServiceClient=lambda: fake_service,
            types=types.SimpleNamespace(
                ModelInput=types.SimpleNamespace(from_ints=lambda ints: ints),
                SamplingParams=lambda **kwargs: kwargs,
            ),
        )
        pipe._tinker_module = fake_tinker
        return pipe, fake_service

    def test_pipes_exposes_multiple_models_and_names(self):
        pipe, _ = self.build_pipe()
        pipe.valves.CHECKPOINTS_JSON = (
            '{"a":{"name":"Model A","checkpoint":"tinker://job/sampler_weights/final"},'
            '"b":{"checkpoint":"tinker://job2/sampler_weights/final","temperature":0.2}}'
        )

        models = pipe.pipes()
        self.assertEqual(2, len(models))
        self.assertEqual("a", models[0]["id"])
        self.assertEqual("Model A", models[0]["name"])
        self.assertEqual("b", models[1]["id"])

    def test_pipe_happy_path(self):
        pipe, service = self.build_pipe()
        pipe.valves.CHECKPOINTS_JSON = '{"m1":"tinker://abc/sampler_weights/final"}'

        result = asyncio.run(
            pipe.pipe(
                {
                    "model": "m1",
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.5,
                },
                __user__={"id": "u1"},
            )
        )

        self.assertEqual("decoded output", result)
        self.assertEqual(["tinker://abc/sampler_weights/final"], service.model_paths)
        self.assertIsNotNone(service.sampler.last_kwargs)

    def test_pipe_handles_prefixed_model_id(self):
        pipe, _ = self.build_pipe()
        pipe.valves.CHECKPOINTS_JSON = '{"m1":"tinker://abc/sampler_weights/final"}'
        result = asyncio.run(pipe.pipe({"model": "tinker_native.m1", "messages": []}))
        self.assertEqual("decoded output", result)

    def test_pipe_unknown_model(self):
        pipe, _ = self.build_pipe()
        pipe.valves.CHECKPOINTS_JSON = '{"m1":"tinker://abc/sampler_weights/final"}'

        result = asyncio.run(pipe.pipe({"model": "missing", "messages": []}))
        self.assertIn("Unknown model", result)


if __name__ == "__main__":
    unittest.main()
