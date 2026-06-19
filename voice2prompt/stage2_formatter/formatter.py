"""
Stage 2 — Local LLM Formatter.

Transforms a pre-processed transcript into clean, structured markdown using a
sub-4B model with 4-bit quantization and full GPU offload.

Default model: Phi-3.5 Mini Instruct Q4_K_M (~2.5 GB VRAM, ~180-240 tok/s on RTX 1000 Ada)
Budget: <= 550 ms for 300-word input on RTX 1000 Ada (50 W TGP).

IMPORTANT: Do NOT use Llama 3.1 8B as the default. At ~70-90 tok/s on RTX 1000 Ada,
a 100-token output takes ~1.2 s — exceeding the 550 ms budget. 8B is a quality-mode
option only, selectable via config.

Tokens are streamed sentence-by-sentence into an asyncio.Queue consumed by Stage 3.
This overlap saves ~120 ms vs. waiting for full Stage 2 output.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from voice2prompt.utils.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a prompt structuring assistant. Filler words are already removed.
Your task: group ideas under ## headings; express each as a concise bullet (-).
Rules:
- One bullet per idea. Max 15 words per bullet.
- Preserve all technical terms, numbers, names, constraints exactly.
- Do NOT add information absent from the transcript.
- Output ONLY the structured markdown. No preamble or commentary.\
"""

# Sentence boundary: ends with .  !  ?  or a completed markdown bullet line
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=\n)")


class Formatter:
    """
    Streams structured markdown from a sub-4B GGUF model via llama-cpp-python.
    Completed sentences are put into the provided asyncio.Queue for Stage 3.
    """

    def __init__(self, config: dict):
        self._config = config
        self._model_path = config.get("model", "Phi-3.5-mini-instruct-Q4_K_M.gguf")
        self._n_gpu_layers = config.get("n_gpu_layers", -1)
        self._max_tokens = config.get("max_tokens", 512)
        self._temperature = config.get("temperature", 0.0)
        self._context_size = config.get("context_size", 2048)
        self._llm = None  # lazy-loaded on first call

    def _load_model(self):
        if self._llm is not None:
            return

        from llama_cpp import Llama  # type: ignore

        model_path = self._model_path
        if not Path(model_path).is_absolute():
            # Look for the model in a local models/ directory
            model_path = str(Path(__file__).parents[3] / "models" / model_path)

        logger.info("loading_formatter_model", model_path=model_path, n_gpu_layers=self._n_gpu_layers)
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._context_size,
            verbose=False,
        )

    async def stream(self, transcript: str, queue: asyncio.Queue) -> None:
        """
        Stream formatted markdown tokens into queue.
        Puts None sentinel when generation is complete.

        Args:
            transcript: Pre-processed transcript from Stage 1.
            queue:      asyncio.Queue shared with Stage 3 compressor.
        """
        self._load_model()
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(None, self._stream_sync, transcript, queue, loop)
        finally:
            await queue.put(None)  # sentinel — Stage 3 exits its consumer loop

    def _stream_sync(self, transcript: str, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        t0 = time.perf_counter()
        buffer = ""
        total_tokens = 0

        stream = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        )

        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if not delta:
                continue

            buffer += delta
            total_tokens += 1

            # Emit completed sentences/bullets to Stage 3 as they arrive
            parts = _SENTENCE_END.split(buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        asyncio.run_coroutine_threadsafe(queue.put(sentence), loop)
                buffer = parts[-1]

        # Flush remainder
        if buffer.strip():
            asyncio.run_coroutine_threadsafe(queue.put(buffer.strip()), loop)

        latency_ms = (time.perf_counter() - t0) * 1000
        tok_per_s = total_tokens / max(latency_ms / 1000, 1e-6)
        logger.info(
            "formatter_complete",
            model_id=self._model_path,
            latency_ms=latency_ms,
            tokens_out=total_tokens,
            tok_per_s=round(tok_per_s, 1),
        )
