"""
Open WebUI Pipe function for native Tinker checkpoints.

This implementation uses Tinker Python SDK primitives (ServiceClient + SamplingClient)
so you can chat directly with checkpoint URIs like:
`tinker://.../sampler_weights/final`
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CheckpointConfig:
    checkpoint: str
    name: Optional[str] = None
    description: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None


class Pipe:
    class Valves:
        def __init__(self):
            self.TINKER_API_KEY: str = ""
            self.CHECKPOINTS_JSON: str = json.dumps(
                {
                    "gpt-oss-20b-ft": {
                        "name": "GPT-OSS-20B (Post-trained)",
                        "checkpoint": "tinker://REPLACE_ME:train:0/sampler_weights/final",
                        "temperature": 0.3,
                        "top_p": 0.95,
                        "max_tokens": 1024,
                    },
                    "gpt-oss-120b-ft": {
                        "name": "GPT-OSS-120B (Post-trained)",
                        "checkpoint": "tinker://REPLACE_ME:train:0/sampler_weights/final",
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "max_tokens": 1024,
                    },
                }
            )
            self.DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."
            self.DEFAULT_TEMPERATURE: float = 0.7
            self.DEFAULT_TOP_P: float = 1.0
            self.DEFAULT_MAX_TOKENS: int = 1024
            self.REQUEST_TIMEOUT_SECONDS: int = 120

    def __init__(self):
        self.type = "pipe"
        self.id = "tinker_native"
        self.name = "Tinker Native"
        self.valves = self.Valves()

        self._tinker_module = None
        self._service_client = None
        self._sampler_cache: Dict[str, Any] = {}
        self._tokenizer_cache: Dict[str, Any] = {}

    def _load_tinker(self):
        if self._tinker_module is not None:
            return self._tinker_module

        import tinker  # type: ignore

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
                elif isinstance(item, dict) and item.get("type") == "text":
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
            lines.append(f"{m['role'].upper()}: {m['content']}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)

    def _coerce_checkpoint_config(self, value: Any) -> Optional[CheckpointConfig]:
        if isinstance(value, str):
            return CheckpointConfig(checkpoint=value)

        if isinstance(value, dict) and isinstance(value.get("checkpoint"), str):
            stop_val = value.get("stop")
            return CheckpointConfig(
                checkpoint=value["checkpoint"],
                name=str(value.get("name")) if value.get("name") is not None else None,
                description=str(value.get("description")) if value.get("description") is not None else None,
                temperature=float(value["temperature"]) if value.get("temperature") is not None else None,
                top_p=float(value["top_p"]) if value.get("top_p") is not None else None,
                max_tokens=int(value["max_tokens"]) if value.get("max_tokens") is not None else None,
                stop=[str(x) for x in stop_val] if isinstance(stop_val, list) else None,
            )

        return None

    def _parse_checkpoints(self) -> Dict[str, CheckpointConfig]:
        try:
            raw = json.loads(self.valves.CHECKPOINTS_JSON)
        except Exception:
            return {}

        parsed: Dict[str, CheckpointConfig] = {}
        if isinstance(raw, dict):
            for model_id, value in raw.items():
                cfg = self._coerce_checkpoint_config(value)
                if cfg:
                    parsed[str(model_id)] = cfg
            return parsed

        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    continue
                cfg = self._coerce_checkpoint_config(item)
                if cfg:
                    parsed[item["id"]] = cfg

        return parsed

    def _resolve_model_id(self, requested_model: str, checkpoint_map: Dict[str, CheckpointConfig]) -> str:
        requested = (requested_model or "").strip()
        if requested in checkpoint_map:
            return requested

        prefix = f"{self.id}."
        if requested.startswith(prefix):
            candidate = requested[len(prefix) :]
            if candidate in checkpoint_map:
                return candidate

        return requested

    def pipes(self) -> List[Dict[str, str]]:
        models: List[Dict[str, str]] = []
        for model_id, cfg in self._parse_checkpoints().items():
            tail = cfg.checkpoint.split("/")[-1] if "/" in cfg.checkpoint else "checkpoint"
            name = cfg.name or f"{self.name}: {model_id} ({tail})"
            entry = {"id": model_id, "name": name}
            if cfg.description:
                entry["description"] = cfg.description
            models.append(entry)
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

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Any = None,
    ) -> str:
        checkpoints = self._parse_checkpoints()
        model_id = self._resolve_model_id(str(body.get("model", "")), checkpoints)

        if model_id not in checkpoints:
            available = ", ".join(sorted(checkpoints.keys())) or "<none configured>"
            return f"[Tinker Native] Unknown model '{body.get('model', '')}'. Available: {available}"

        messages = body.get("messages", [])
        if not isinstance(messages, list):
            return "[Tinker Native] Invalid request: 'messages' must be a list."

        cfg = checkpoints[model_id]
        messages = self._ensure_system_message(messages)

        try:
            tinker = self._load_tinker()
            sampler, tokenizer = self._get_sampler_and_tokenizer(cfg.checkpoint)

            prompt_text = self._render_prompt(tokenizer, messages)
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            model_input = tinker.types.ModelInput.from_ints(prompt_tokens)

            temperature = self._clamp(
                float(
                    body.get(
                        "temperature",
                        cfg.temperature if cfg.temperature is not None else self.valves.DEFAULT_TEMPERATURE,
                    )
                ),
                0.0,
                2.0,
            )
            top_p = self._clamp(
                float(body.get("top_p", cfg.top_p if cfg.top_p is not None else self.valves.DEFAULT_TOP_P)),
                0.0,
                1.0,
            )
            max_tokens = max(1, int(body.get("max_tokens", cfg.max_tokens or self.valves.DEFAULT_MAX_TOKENS)))

            sampling_params = tinker.types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=body.get("stop", cfg.stop),
            )

            future = sampler.sample(prompt=model_input, num_samples=1, sampling_params=sampling_params)
            result = future.result(timeout=self.valves.REQUEST_TIMEOUT_SECONDS)
            return self._decode_sequence(result, tokenizer)
        except Exception as exc:
            user_hint = (__user__ or {}).get("email") or (__user__ or {}).get("id") or "unknown-user"
            return f"[Tinker Native] Sampling failed for {user_hint} on {model_id}: {exc}"
