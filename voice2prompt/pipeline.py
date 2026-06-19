"""
Async orchestrator for the three-stage Voice2Prompt pipeline.

Stages execute in a streaming pipeline:
  Stage 1 (STT) → Stage 2 (Formatter, streaming) → Stage 3 (Compressor, overlapped)

Stage 3 subscribes to Stage 2's sentence queue via asyncio.Queue, saving ~120 ms
vs. sequential execution on the RTX 1000 Ada target.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from voice2prompt.stage1_stt.transcriber import Transcriber
from voice2prompt.stage1_stt.filler_pass import filler_pass
from voice2prompt.stage2_formatter.formatter import Formatter
from voice2prompt.stage3_compressor.compressor import Compressor
from voice2prompt.emit.api_client import ApiClient
from voice2prompt.utils.logging import get_logger

logger = get_logger(__name__)

_QUEUE_SENTINEL = None
_STREAM_TIMEOUT_S = 2.0


@dataclass
class PipelineMetadata:
    stage1_latency_ms: float = 0.0
    stage2_latency_ms: float = 0.0
    stage3_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: str = ""
    rouge_l_estimate: float = 0.0


@dataclass
class PipelineResult:
    prompt: str
    metadata: PipelineMetadata = field(default_factory=PipelineMetadata)
    raw_transcript: str = ""
    formatted_prompt: str = ""


class Pipeline:
    def __init__(self, config: dict):
        self._config = config
        self._stt = Transcriber(config["stage1"])
        self._formatter = Formatter(config["stage2"])
        self._compressor = Compressor(config["stage3"])
        self._emitter = ApiClient(config["emit"])

    @classmethod
    def from_config(cls, config_path: str | Path) -> "Pipeline":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    async def run(self, audio: bytes | Path) -> PipelineResult:
        import time

        t0 = time.perf_counter()
        meta = PipelineMetadata()

        # Stage 1 — STT + filler pre-pass
        transcript = await self._stt.transcribe(audio)
        meta.stage1_latency_ms = (time.perf_counter() - t0) * 1000
        clean_text = filler_pass(transcript.text)
        logger.info("stage1_complete", latency_ms=meta.stage1_latency_ms, text_len=len(clean_text))

        # Stages 2 + 3 — overlapped via asyncio.Queue
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=32)

        t2 = time.perf_counter()
        fmt_task = asyncio.create_task(self._formatter.stream(clean_text, queue))
        cmp_task = asyncio.create_task(self._compressor.compress_stream(queue))

        await asyncio.gather(fmt_task, cmp_task)
        compress_result = cmp_task.result()

        meta.stage2_latency_ms = (time.perf_counter() - t2) * 1000
        meta.stage3_latency_ms = compress_result.latency_ms
        meta.total_latency_ms = (time.perf_counter() - t0) * 1000
        meta.original_tokens = compress_result.original_tokens
        meta.compressed_tokens = compress_result.compressed_tokens
        meta.compression_ratio = compress_result.ratio
        meta.rouge_l_estimate = compress_result.rouge_l_estimate

        logger.info(
            "pipeline_complete",
            total_latency_ms=meta.total_latency_ms,
            compression_ratio=meta.compression_ratio,
        )

        return PipelineResult(
            prompt=compress_result.text,
            metadata=meta,
            raw_transcript=transcript.text,
            formatted_prompt=compress_result.formatted_text,
        )
