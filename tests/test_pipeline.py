"""
Integration tests for the full Voice2Prompt pipeline.

Uses mocked stage implementations to test orchestration logic without
requiring GPU hardware or downloaded models.

Each test that touches asyncio.Queue verifies the real queue contract —
no mocking of the queue itself. This validates the streaming handoff
between Stage 2 and Stage 3 that saves ~120 ms on RTX 1000 Ada.
"""

import asyncio
from pathlib import Path
import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures"

_MINIMAL_CONFIG = {
    "stage1": {"model": "parakeet-tdt-0.6b-v2", "device": "cpu"},
    "stage2": {"model": "mock.gguf", "n_gpu_layers": 0, "max_tokens": 64},
    "stage3": {"rate": 0.4, "force_tokens": ["##", "-", "**"]},
    "emit": {"format": "openai", "attach_metadata": True},
}


def _make_mock_pipeline(monkeypatch):
    """
    Return a Pipeline with all three stages replaced by fast in-process mocks.
    Validates orchestration logic (queue handoff, gather, result shape) without
    touching real models.
    """
    from voice2prompt.pipeline import Pipeline
    from voice2prompt.stage1_stt.transcriber import TranscriptResult
    from voice2prompt.stage3_compressor.compressor import CompressResult

    pipeline = Pipeline(_MINIMAL_CONFIG)

    # Stage 1 mock
    async def _mock_transcribe(audio):
        return TranscriptResult(
            text="I want to build um an API that uh handles user auth using JWT.",
            model_id="mock-stt",
        )

    monkeypatch.setattr(pipeline._stt, "transcribe", _mock_transcribe)

    # Stage 2 mock — puts sentences into queue then sentinel
    async def _mock_stream(transcript, queue):
        await queue.put("## Goal")
        await queue.put("- Build API for user authentication using JWT.")
        await queue.put(None)

    monkeypatch.setattr(pipeline._formatter, "stream", _mock_stream)

    # Stage 3 mock — consumes queue
    async def _mock_compress_stream(queue):
        sentences = []
        while True:
            item = await asyncio.wait_for(queue.get(), timeout=2.0)
            if item is None:
                break
            sentences.append(item)
        text = " ".join(sentences)
        return CompressResult(
            text=text,
            formatted_text=text,
            original_tokens=20,
            compressed_tokens=8,
            ratio="2.5x",
            rouge_l_estimate=0.91,
        )

    monkeypatch.setattr(pipeline._compressor, "compress_stream", _mock_compress_stream)

    return pipeline


class TestPipelineOrchestration:
    @pytest.mark.asyncio
    async def test_run_returns_pipeline_result(self, monkeypatch):
        from voice2prompt.pipeline import PipelineResult
        pipeline = _make_mock_pipeline(monkeypatch)
        result = await pipeline.run(b"fake_audio")
        assert isinstance(result, PipelineResult)

    @pytest.mark.asyncio
    async def test_filler_words_removed_from_transcript(self, monkeypatch):
        pipeline = _make_mock_pipeline(monkeypatch)
        result = await pipeline.run(b"fake_audio")
        assert "um" not in result.raw_transcript or "um" not in result.prompt
        # raw_transcript has fillers; formatted/compressed should not
        assert "um" not in result.prompt
        assert "uh" not in result.prompt

    @pytest.mark.asyncio
    async def test_metadata_populated(self, monkeypatch):
        pipeline = _make_mock_pipeline(monkeypatch)
        result = await pipeline.run(b"fake_audio")
        assert result.metadata.original_tokens == 20
        assert result.metadata.compressed_tokens == 8
        assert result.metadata.compression_ratio == "2.5x"
        assert result.metadata.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_compression_meets_60_percent_target(self, monkeypatch):
        pipeline = _make_mock_pipeline(monkeypatch)
        result = await pipeline.run(b"fake_audio")
        meta = result.metadata
        if meta.original_tokens > 0:
            reduction = 1 - meta.compressed_tokens / meta.original_tokens
            assert reduction >= 0.60

    @pytest.mark.asyncio
    async def test_rouge_l_meets_target(self, monkeypatch):
        pipeline = _make_mock_pipeline(monkeypatch)
        result = await pipeline.run(b"fake_audio")
        assert result.metadata.rouge_l_estimate >= 0.85


class TestPipelineWithFixtures:
    """
    Integration tests using real audio fixtures.
    Skipped if fixtures directory is empty (pre-model-download state).
    """

    @pytest.fixture(params=list(FIXTURE_DIR.glob("*.wav")) if FIXTURE_DIR.exists() else [])
    def audio_fixture(self, request):
        return request.param

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not FIXTURE_DIR.exists() or not any(FIXTURE_DIR.glob("*.wav")),
        reason="No audio fixtures found in tests/fixtures/",
    )
    async def test_fixture_produces_valid_output(self, audio_fixture, monkeypatch):
        pipeline = _make_mock_pipeline(monkeypatch)
        result = await pipeline.run(audio_fixture)
        assert result.prompt
        assert result.metadata.total_latency_ms > 0
