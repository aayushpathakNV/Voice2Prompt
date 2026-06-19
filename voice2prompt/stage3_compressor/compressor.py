"""
Stage 3 — Token Compressor.

Consumes Stage 2's sentence stream from an asyncio.Queue and applies
LLMLingua-2 token compression, targeting >= 60% token reduction while
preserving >= 95% semantic accuracy (ROUGE-L >= 0.85).

Model:   microsoft/llmlingua-2-bert-large-multilingual-cased-meetingbank
VRAM:    ~1.5 GB
Budget:  <= 300 ms on RTX 1000 Ada (overlapped with Stage 2 tail).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from voice2prompt.utils.logging import get_logger

logger = get_logger(__name__)

_STREAM_TIMEOUT_S = 2.0


@dataclass
class CompressResult:
    text: str
    formatted_text: str = ""
    original_tokens: int = 0
    compressed_tokens: int = 0
    ratio: str = ""
    rouge_l_estimate: float = 0.0
    latency_ms: float = 0.0


class Compressor:
    """
    Wraps LLMLingua-2. Subscribes to Stage 2's asyncio.Queue and accumulates
    sentences into a compression buffer, flushing when the sentinel arrives.

    force_tokens ensures structure markers (##, -, **) are never dropped.
    """

    def __init__(self, config: dict):
        self._config = config
        self._model_name = config.get(
            "model",
            "microsoft/llmlingua-2-bert-large-multilingual-cased-meetingbank",
        )
        self._device = config.get("device", "cuda")
        self._rate = config.get("rate", 0.4)
        self._force_tokens: list[str] = config.get("force_tokens", ["##", "-", "**"])
        self._drop_consecutive = config.get("drop_consecutive", True)
        self._compressor: Any = None  # lazy-loaded on first call

    def _load_model(self):
        if self._compressor is not None:
            return

        from llmlingua import PromptCompressor  # type: ignore

        logger.info("loading_compressor_model", model=self._model_name, device=self._device)
        self._compressor = PromptCompressor(
            model_name=self._model_name,
            use_llmlingua2=True,
            device_map=self._device,
        )

    async def compress_stream(self, queue: asyncio.Queue) -> CompressResult:
        """
        Consume sentences from queue until None sentinel, then compress the
        accumulated buffer.

        Args:
            queue: asyncio.Queue fed by Stage 2 formatter.

        Returns:
            CompressResult with compressed text and metadata.
        """
        self._load_model()
        t0 = time.perf_counter()

        sentences: list[str] = []
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=_STREAM_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning("compressor_stream_timeout", buffered_sentences=len(sentences))
                break

            if item is None:
                break
            sentences.append(item)

        formatted_text = "\n".join(sentences)
        result = await asyncio.get_running_loop().run_in_executor(
            None, self._compress_sync, formatted_text
        )
        result.latency_ms = (time.perf_counter() - t0) * 1000
        result.formatted_text = formatted_text

        logger.info(
            "compressor_complete",
            model_id=self._model_name,
            latency_ms=result.latency_ms,
            tokens_in=result.original_tokens,
            tokens_out=result.compressed_tokens,
            ratio=result.ratio,
        )
        return result

    def _compress_sync(self, text: str) -> CompressResult:
        if not text.strip():
            return CompressResult(text=text)

        output = self._compressor.compress_prompt(
            text,
            rate=self._rate,
            force_tokens=self._force_tokens,
            drop_consecutive=self._drop_consecutive,
        )

        compressed = output["compressed_prompt"]
        ratio_str = output.get("ratio", "")

        # Token counts from LLMLingua-2 output
        original_tokens = output.get("origin_tokens", 0)
        compressed_tokens = output.get("compressed_tokens", 0)

        # Estimate ROUGE-L (placeholder — replace with actual evaluation in tests)
        rouge_l = self._estimate_rouge_l(text, compressed)

        return CompressResult(
            text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            ratio=ratio_str,
            rouge_l_estimate=rouge_l,
        )

    def _estimate_rouge_l(self, reference: str, hypothesis: str) -> float:
        """
        Lightweight ROUGE-L estimate using LCS token overlap.
        Full ROUGE-L computation is done in the test suite via rouge-score.
        """
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not ref_tokens or not hyp_tokens:
            return 0.0

        # LCS length via DP
        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(2)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
                else:
                    dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
        lcs = dp[m % 2][n]

        precision = lcs / n
        recall = lcs / m
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    async def compress(self, prompt: str, dry_run: bool = False) -> CompressResult:
        """
        Single-shot compression (non-streaming). Useful for testing.

        Args:
            prompt:  Already-formatted markdown string.
            dry_run: If True, return stats without modifying the prompt.
        """
        self._load_model()
        t0 = time.perf_counter()
        result = await asyncio.get_running_loop().run_in_executor(
            None, self._compress_sync, prompt
        )
        result.latency_ms = (time.perf_counter() - t0) * 1000
        if dry_run:
            result.text = prompt
        return result
