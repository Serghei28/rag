"""Клиент локальной LLM через HTTP API Ollama."""

from dataclasses import dataclass

import httpx

from rag.config import GenerationConfig


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OllamaClient:
    def __init__(self, host: str, config: GenerationConfig):
        self.host = host.rstrip("/")
        self.config = config

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def parameters(self) -> dict:
        config = self.config
        return {
            "temperature": config.temperature,
            "num_ctx": config.num_ctx,
            "num_predict": config.max_tokens,
            "seed": config.seed,
        }

    def chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> LLMResponse:
        """`json_mode` — Ollama гарантирует, что ответ будет валидным JSON (для LLM-судьи)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self.parameters,
            "keep_alive": "10m",
        }
        if json_mode:
            payload["format"] = "json"
        try:
            response = httpx.post(
                f"{self.host}/api/chat", json=payload, timeout=self.config.timeout
            )
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Ollama недоступна ({self.host}): запустите Ollama или `ollama serve`"
            ) from exc
        if response.status_code == 404:
            raise RuntimeError(f"В Ollama нет модели {self.model}: `ollama pull {self.model}`")
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            text=data["message"]["content"].strip(),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )
