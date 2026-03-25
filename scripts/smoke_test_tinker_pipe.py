#!/usr/bin/env python3
"""Quick smoke test for tinker_native_pipeline.py.

Example:
  python scripts/smoke_test_tinker_pipe.py \
    --checkpoint tinker://.../sampler_weights/final \
    --api-key tml-... \
    --prompt "Say hello"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinker_native_pipeline import Pipe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Smoke test for Open WebUI Tinker native pipe")
    p.add_argument("--checkpoint", required=True, help="Tinker checkpoint URI")
    p.add_argument("--api-key", default="", help="Tinker API key (tml-...)")
    p.add_argument("--prompt", default="Reply with: smoke_ok", help="User prompt")
    p.add_argument("--model-id", default="smoke", help="Temporary model id")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.2)
    return p


async def run(args: argparse.Namespace) -> str:
    pipe = Pipe()

    if args.api_key:
        pipe.valves.TINKER_API_KEY = args.api_key

    pipe.valves.CHECKPOINTS_JSON = json.dumps(
        {
            args.model_id: {
                "name": "Smoke Test Model",
                "checkpoint": args.checkpoint,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
        }
    )

    return await pipe.pipe(
        {
            "model": args.model_id,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output = asyncio.run(run(args))
    print(output)


if __name__ == "__main__":
    main()
