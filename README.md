# Open WebUI + Tinker Native Pipeline (for sharing post-trained checkpoints)

This repository contains an Open WebUI **Pipe Function** that talks to Tinker checkpoints through the **native Tinker Python SDK** (not OpenAI-compatible endpoints).

If you post-train models like **GPT-OSS-20B** and **GPT-OSS-120B** in Tinker, this pipe lets you expose each checkpoint as a selectable model in Open WebUI so teammates can use a ChatGPT-like interface for side-by-side comparison.

## Why this is useful for model sharing

- Register multiple post-trained checkpoints and expose them as separate models.
- Give each model a user-friendly name (e.g., “GPT-OSS-20B Support Assistant v3”).
- Let non-technical users test model quality in Open WebUI without needing direct Tinker tooling.
- Keep using native checkpoint URIs (`tinker://.../sampler_weights/final`).

## Requirements

Inside the Open WebUI runtime environment:

```bash
pip install tinker
```

Then set your API key (or set it in the function valves):

```bash
export TINKER_API_KEY=...
```

## Add to Open WebUI

1. Go to **Admin → Functions**.
2. Create a new **Pipe** function.
3. Paste `tinker_native_pipeline.py`.
4. Configure `CHECKPOINTS_JSON` and save.
5. Ask users to select one of the exposed models in new chats.

## Recommended `CHECKPOINTS_JSON` (GPT-OSS examples)

```json
{
  "gpt-oss-20b-ft": {
    "name": "GPT-OSS-20B (Post-trained)",
    "description": "General assistant checkpoint",
    "checkpoint": "tinker://JOB_ID_20B:train:0/sampler_weights/final",
    "temperature": 0.3,
    "top_p": 0.95,
    "max_tokens": 1024
  },
  "gpt-oss-120b-ft": {
    "name": "GPT-OSS-120B (Post-trained)",
    "description": "Higher quality but higher cost",
    "checkpoint": "tinker://JOB_ID_120B:train:0/sampler_weights/final",
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 1024
  }
}
```

## Also supported: simple mapping

```json
{
  "team-default": "tinker://JOB_ID:train:0/sampler_weights/final"
}
```

## Operational notes

- The pipe accepts both `model: "my-id"` and prefixed `model: "tinker_native.my-id"` requests.
- If tokenizer chat templates are available, they are used automatically.
- If not, it falls back to a plain role-tagged prompt format.
- Current behavior is non-streaming (`num_samples=1`).

## Local tests

```bash
python -m unittest -v tests/test_tinker_native_pipeline.py
```

## Sources used to shape this implementation

- Tinker docs (`SamplingClient`, installation, training/sampling guide).
- Open WebUI docs (0.5 migration notes for function signatures).
