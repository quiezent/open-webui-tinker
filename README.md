# Open WebUI + Tinker Native Pipeline (SDK-first)

This repository contains an Open WebUI **Pipe Function** that talks to Tinker checkpoints through the **native Tinker Python SDK** (not OpenAI-compatible endpoints).

## Why this version is better

- Uses Tinker's native SDK flow (`ServiceClient` + `SamplingClient`) for `tinker://...` checkpoints.
- Supports **multiple checkpoints** so each appears as a separate model in Open WebUI.
- Lets you configure **per-checkpoint defaults** (temperature/top_p/max_tokens/stop) for easy A/B model comparison.
- Compatible with newer Open WebUI function signatures (`async pipe(..., __request__)`).

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
4. Configure valves, especially `CHECKPOINTS_JSON`.

## `CHECKPOINTS_JSON` formats

### Simple mapping

```json
{
  "tinker-final-1": "tinker://05a8613d-3de1-5206-a321-ddc55d231ee3:train:0/sampler_weights/final",
  "tinker-exp-2": "tinker://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:train:0/sampler_weights/final"
}
```

### Advanced mapping (per-checkpoint defaults)

```json
{
  "tinker-final-1": {
    "checkpoint": "tinker://05a8613d-3de1-5206-a321-ddc55d231ee3:train:0/sampler_weights/final",
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 1024,
    "stop": ["<|im_end|>"]
  },
  "tinker-exp-2": {
    "checkpoint": "tinker://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:train:0/sampler_weights/final",
    "temperature": 0.7
  }
}
```

## Notes

- If tokenizer chat templates are available, the pipe will use them automatically.
- If not, it falls back to a plain role-tagged prompt format.
- The code currently returns non-streamed responses (`num_samples=1`).

## Sources used to update this implementation

- Tinker docs (`SamplingClient`, installation, training/sampling guide).
- Open WebUI docs (0.5 migration notes for updated function signatures).
