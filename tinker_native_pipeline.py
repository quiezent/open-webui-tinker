"""
Open WebUI Pipe function for native Tinker checkpoints.

This implementation uses Tinker Python SDK primitives (ServiceClient + SamplingClient)
so you can chat directly with checkpoint URIs like:
`tinker://.../sampler_weights/final`

No OpenAI-compatibility layer is required.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


@dataclass
class CheckpointConfig:
    checkpoint: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None


class Pipe:
    class Valves(BaseModel):
        TINKER_API_KEY: str = Field(
            default="",
            description="Optional override for TINKER_API_KEY environment variable.",
        )
        CHECKPOINTS_JSON: str = Field(
            default=(
                '{"tinker-final-1": '
                '"tinker://05a8613d-3de1-5206-a321-ddc55d231ee3:train:0/sampler_weights/final"}'
            ),
            description=(
                "JSON object that maps Open WebUI model IDs to checkpoint configs. "
                "Value can be a string URI or object: "
                "{\"checkpoint\": \"tinker://...\", \"temperature\": 0.7, \"max_tokens\": 1024, "
                "\"top_p\": 1.0, \"stop\": [\"<eos>\"]}."
            ),
        )
        DEFAULT_SYSTEM_PROMPT: str = Field(
            default="You are a helpful assistant.",
            description="Fallback system prompt if no system message is present.",
        )
        DEFAULT_TEMPERATURE: float = Field(default=0.7)
        DEFAULT_TOP_P: float = Field(default=1.0)
        DEFAULT_MAX_TOKENS: int = Field(default=1024)
        REQUEST_TIMEOUT_SECONDS: int = Field(
            default=120,
            description="Timeout waiting on SDK sample future.",
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "tinker_native"
        self.name = "Tinker Native"
        self.valves = self.Valves()

        # Lazy initialized to avoid import failures at module import time.
        self._tinker_module = None
        self._service_client = None
        self._sampler_cache: Dict[str, Any] = {}
        self._tokenizer_cache: Dict[str, Any] = {}

    def _load_tinker(self):
        if self._tinker_module is not None:
            return self._tinker_module

        try:
            import tinker  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Tinker SDK is not installed. Install it in Open WebUI environment with `pip install tinker`."
            ) from exc

        if self.valves.TINKER_API_KEY:
            os.environ["TINKER_API_KEY"] = self.valves.TINKER_API_KEY

        self._tinker_module = tinker
        return tinker

    def _get_service_client(self):
        if self._service_client is not None:
            return self._service_client

        tinker = self._load_tinker()
        self._service_client = tinker.ServiceClient()
        return self._service_client

    @staticmethod
    def _normalize_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join([p for p in parts if p])
        return str(content)

    def _ensure_system_message(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if any(m.get("role") == "system" for m in messages):
            return messages
        return [{"role": "system", "content": self.valves.DEFAULT_SYSTEM_PROMPT}, *messages]

    @staticmethod
    def _as_chat_template_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for m in messages:
            role = str(m.get("role", "user"))
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            out.append({"role": role, "content": Pipe._normalize_message_content(m.get("content", ""))})
        return out

    def _render_prompt(self, tokenizer: Any, messages: List[Dict[str, Any]]) -> str:
        chat_messages = self._as_chat_template_messages(messages)

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception:
                pass

        lines: List[str] = []
        for m in chat_messages:
            role = m["role"].upper()
            lines.append(f"{role}: {m['content']}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)

    def _parse_checkpoints(self) -> Dict[str, CheckpointConfig]:
        try:
            raw = json.loads(self.valves.CHECKPOINTS_JSON)
        except Exception:
            return {}

        if not isinstance(raw, dict):
            return {}

        parsed: Dict[str, CheckpointConfig] = {}
        for model_id, value in raw.items():
            if isinstance(value, str):
                parsed[str(model_id)] = CheckpointConfig(checkpoint=value)
                continue

            if isinstance(value, dict) and isinstance(value.get("checkpoint"), str):
                stop_val = value.get("stop")
                stop = [str(x) for x in stop_val] if isinstance(stop_val, list) else None
                parsed[str(model_id)] = CheckpointConfig(
                    checkpoint=value["checkpoint"],
                    temperature=float(value["temperature"]) if value.get("temperature") is not None else None,
                    top_p=float(value["top_p"]) if value.get("top_p") is not None else None,
                    max_tokens=int(value["max_tokens"]) if value.get("max_tokens") is not None else None,
                    stop=stop,
                )

        return parsed

    def pipes(self) -> List[Dict[str, str]]:
        models: List[Dict[str, str]] = []
        for model_id, cfg in self._parse_checkpoints().items():
            tail = cfg.checkpoint.split("/")[-1] if "/" in cfg.checkpoint else "checkpoint"
            models.append({"id": model_id, "name": f"{self.name}: {model_id} ({tail})"})
        return models

    def _get_sampler_and_tokenizer(self, checkpoint_uri: str) -> Tuple[Any, Any]:
        if checkpoint_uri in self._sampler_cache and checkpoint_uri in self._tokenizer_cache:
            return self._sampler_cache[checkpoint_uri], self._tokenizer_cache[checkpoint_uri]

        service_client = self._get_service_client()
        sampler = service_client.create_sampling_client(model_path=checkpoint_uri)
        tokenizer = sampler.get_tokenizer()

        self._sampler_cache[checkpoint_uri] = sampler
        self._tokenizer_cache[checkpoint_uri] = tokenizer
        return sampler, tokenizer

    @staticmethod
    def _decode_sequence(result: Any, tokenizer: Any) -> str:
        # Tinker examples expose `result.sequences`; guard for mild schema variance.
        sequences = getattr(result, "sequences", None) or getattr(result, "samples", None)
        if sequences and len(sequences) > 0:
            seq = sequences[0]
            tokens = getattr(seq, "tokens", None)
            if tokens is not None:
                try:
                    return tokenizer.decode(tokens).strip()
                except Exception:
                    return str(tokens)
        return str(result)

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Any = None,
    ) -> str:
        checkpoints = self._parse_checkpoints()
        model_id = str(body.get("model", "")).strip()

        if model_id not in checkpoints:
            available = ", ".join(sorted(checkpoints.keys())) or "<none configured>"
            return f"[Tinker Native] Unknown model '{model_id}'. Available: {available}"

        cfg = checkpoints[model_id]
        messages = body.get("messages", [])
        if not isinstance(messages, list):
            return "[Tinker Native] Invalid request: 'messages' must be a list."

        messages = self._ensure_system_message(messages)

        try:
            tinker = self._load_tinker()
            sampler, tokenizer = self._get_sampler_and_tokenizer(cfg.checkpoint)

            prompt_text = self._render_prompt(tokenizer, messages)
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            model_input = tinker.types.ModelInput.from_ints(prompt_tokens)

            sampling_params = tinker.types.SamplingParams(
                max_tokens=int(body.get("max_tokens", cfg.max_tokens or self.valves.DEFAULT_MAX_TOKENS)),
                temperature=float(body.get("temperature", cfg.temperature if cfg.temperature is not None else self.valves.DEFAULT_TEMPERATURE)),
                top_p=float(body.get("top_p", cfg.top_p if cfg.top_p is not None else self.valves.DEFAULT_TOP_P)),
                stop=body.get("stop", cfg.stop),
            )

            future = sampler.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            result = future.result(timeout=self.valves.REQUEST_TIMEOUT_SECONDS)
            return self._decode_sequence(result, tokenizer)

        except Exception as exc:
            user_hint = (__user__ or {}).get("email") or (__user__ or {}).get("id") or "unknown-user"
            return f"[Tinker Native] Sampling failed for {user_hint}: {exc}"
