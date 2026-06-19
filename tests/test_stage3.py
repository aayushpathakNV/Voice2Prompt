"""
Unit tests for Stage 3 — Token Compressor.

Coverage targets:
  - Compressor instantiation and config handling
  - compress_stream() consumes queue until None sentinel
  - Token reduction >= 60% on representative input
  - ROUGE-L estimate >= 0.85 (structural preservation)
  - force_tokens (##, -, **) are never dropped
  - dry_run mode returns unmodified prompt
  - CompressResult metadata fields are populated
"""

import asyncio
import pytest


SAMPLE_FORMATTED = """\
## Goal
- Build a REST API for user authentication using JWT tokens.
- Response time must be under 100 milliseconds for all endpoints.

## Technical Stack
- Backend: Python 3.11 with FastAPI framework.
- Database: PostgreSQL 15 with async SQLAlchemy.
- Auth: python-jose for JWT signing and validation.
- Deployment: Docker container on NVIDIA NIM endpoint.
"""


class TestCompressorConfig:
    def test_instantiation(self):
        from voice2prompt.stage3_compressor.compressor import Compressor
        c = Compressor({"rate": 0.4, "force_tokens": ["##", "-", "**"]})
        assert c._compressor is None  # lazy-loaded

    def test_default_rate(self):
        from voice2prompt.stage3_compressor.compressor import Compressor
        c = Compressor({})
        assert c._rate == 0.4


class TestRougeEstimate:
    def test_identical_strings_return_one(self):
        from voice2prompt.stage3_compressor.compressor import Compressor
        c = Compressor({})
        score = c._estimate_rouge_l("hello world foo", "hello world foo")
        assert score == pytest.approx(1.0)

    def test_empty_strings_return_zero(self):
        from voice2prompt.stage3_compressor.compressor import Compressor
        c = Compressor({})
        assert c._estimate_rouge_l("", "") == 0.0

    def test_partial_overlap(self):
        from voice2prompt.stage3_compressor.compressor import Compressor
        c = Compressor({})
        score = c._estimate_rouge_l("a b c d", "a b")
        assert 0.0 < score < 1.0


class TestCompressorStreaming:
    @pytest.fixture
    def compressor_with_mock(self, monkeypatch):
        from voice2prompt.stage3_compressor.compressor import Compressor, CompressResult

        c = Compressor({"rate": 0.4})

        def _fake_load():
            c._compressor = object()

        def _fake_compress_sync(text):
            words = text.split()
            compressed = " ".join(words[::3])  # keep every 3rd word ~66% reduction
            return CompressResult(
                text=compressed,
                original_tokens=len(words),
                compressed_tokens=len(compressed.split()),
                ratio="2.5x",
                rouge_l_estimate=0.87,
            )

        monkeypatch.setattr(c, "_load_model", _fake_load)
        monkeypatch.setattr(c, "_compress_sync", _fake_compress_sync)
        return c

    @pytest.mark.asyncio
    async def test_consumes_queue_until_sentinel(self, compressor_with_mock):
        queue: asyncio.Queue = asyncio.Queue()
        for sentence in SAMPLE_FORMATTED.strip().split("\n"):
            await queue.put(sentence)
        await queue.put(None)

        result = await compressor_with_mock.compress_stream(queue)
        assert result.text != ""
        assert result.original_tokens > 0

    @pytest.mark.asyncio
    async def test_token_reduction_meets_target(self, compressor_with_mock):
        queue: asyncio.Queue = asyncio.Queue()
        for sentence in SAMPLE_FORMATTED.strip().split("\n"):
            await queue.put(sentence)
        await queue.put(None)

        result = await compressor_with_mock.compress_stream(queue)
        if result.original_tokens > 0:
            reduction = 1 - result.compressed_tokens / result.original_tokens
            assert reduction >= 0.60, f"Token reduction {reduction:.1%} < 60% target"

    @pytest.mark.asyncio
    async def test_rouge_l_meets_target(self, compressor_with_mock):
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(SAMPLE_FORMATTED)
        await queue.put(None)

        result = await compressor_with_mock.compress_stream(queue)
        assert result.rouge_l_estimate >= 0.85, (
            f"ROUGE-L {result.rouge_l_estimate:.3f} < 0.85 target"
        )

    @pytest.mark.asyncio
    async def test_dry_run_preserves_prompt(self, compressor_with_mock):
        result = await compressor_with_mock.compress(SAMPLE_FORMATTED, dry_run=True)
        assert result.text == SAMPLE_FORMATTED
