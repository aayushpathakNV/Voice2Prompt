"""
Emit layer — wraps the compressed prompt in OpenAI or Anthropic message format
and dispatches to a cloud LLM API endpoint.

Supports:
  - OpenAI Chat Completions  (format: openai)
  - Anthropic Messages API   (format: anthropic)
  - Any OpenAI-compatible endpoint via base_url (NVIDIA NIM, vLLM, Ollama)

Pipeline metadata (token savings, compression ratio) is attached as a system
prompt header when emit.attach_metadata is true.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from voice2prompt.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmitResult:
    api_response: dict
    prompt_sent: str
    metadata_header: str
    model: str
    format: str


class ApiClient:
    def __init__(self, config: dict):
        self._format = config.get("format", "openai")
        self._base_url = config.get("base_url", "https://api.openai.com/v1")
        self._model = config.get("model", "gpt-4o")
        self._attach_metadata = config.get("attach_metadata", True)
        self._api_key = config.get("api_key") or os.environ.get("VOICE2PROMPT_API_KEY", "")

    def _build_metadata_header(self, original_tokens: int, compressed_tokens: int, ratio: str) -> str:
        saved = original_tokens - compressed_tokens
        pct = round(saved / max(original_tokens, 1) * 100)
        return f"[voice2prompt] saved {saved} tokens ({pct}%) | ratio {ratio}"

    async def send(
        self,
        compressed_prompt: str,
        original_tokens: int = 0,
        compressed_tokens: int = 0,
        ratio: str = "",
        **kwargs,
    ) -> EmitResult:
        metadata_header = ""
        if self._attach_metadata and original_tokens:
            metadata_header = self._build_metadata_header(original_tokens, compressed_tokens, ratio)

        if self._format == "anthropic":
            return await self._send_anthropic(compressed_prompt, metadata_header, **kwargs)
        return await self._send_openai(compressed_prompt, metadata_header, **kwargs)

    async def _send_openai(self, prompt: str, metadata_header: str, **kwargs) -> EmitResult:
        import httpx  # type: ignore

        messages = []
        if metadata_header:
            messages.append({"role": "system", "content": metadata_header})
        messages.append({"role": "user", "content": prompt})

        payload = {"model": self._model, "messages": messages, **kwargs}
        logger.info("emit_openai", model=self._model, base_url=self._base_url)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=30.0,
            )
            response.raise_for_status()
            return EmitResult(
                api_response=response.json(),
                prompt_sent=prompt,
                metadata_header=metadata_header,
                model=self._model,
                format="openai",
            )

    async def _send_anthropic(self, prompt: str, metadata_header: str, **kwargs) -> EmitResult:
        import httpx  # type: ignore

        payload = {
            "model": self._model,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        if metadata_header:
            payload["system"] = metadata_header

        logger.info("emit_anthropic", model=self._model)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return EmitResult(
                api_response=response.json(),
                prompt_sent=prompt,
                metadata_header=metadata_header,
                model=self._model,
                format="anthropic",
            )
