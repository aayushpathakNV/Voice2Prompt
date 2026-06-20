"""
Unit tests for Stage 2 — LLM Formatter.

Coverage targets:
  - Formatter instantiation and config handling
  - stream() puts sentences into queue and terminates with None sentinel
  - Output structure: must contain >= 1 ## heading and >= 1 bullet
  - Model NOT loaded during instantiation (lazy loading)
"""

from __future__ import annotations

import asyncio

import pytest


class TestFormatterConfig:
    def test_instantiation(self) -> None:
        from voice2prompt.stage2_formatter.formatter import Formatter

        f = Formatter({"model": "Phi-3.5-mini-instruct-Q4_K_M.gguf", "n_gpu_layers": -1})
        assert f._llm is None

    def test_default_config(self) -> None:
        from voice2prompt.stage2_formatter.formatter import Formatter

        f = Formatter({})
        assert f._max_tokens == 512
        assert f._temperature == 0.0
        assert f._n_gpu_layers == -1


class TestFormatterStreaming:
    @pytest.fixture
    def formatter_with_mock(self, monkeypatch: pytest.MonkeyPatch):
        from voice2prompt.stage2_formatter.formatter import Formatter

        f = Formatter({})

        def _fake_load() -> None:
            f._llm = object()

        def _fake_stream_sync(transcript: str, queue: asyncio.Queue, loop) -> None:
            asyncio.run_coroutine_threadsafe(queue.put("## Goal"), loop)
            asyncio.run_coroutine_threadsafe(queue.put("- Build a fast API."), loop)

        monkeypatch.setattr(f, "_load_model", _fake_load)
        monkeypatch.setattr(f, "_stream_sync", _fake_stream_sync)
        return f

    @pytest.mark.asyncio
    async def test_stream_sends_sentinel(self, formatter_with_mock) -> None:
        queue: asyncio.Queue = asyncio.Queue()
        await formatter_with_mock.stream("build an API", queue)

        items: list[str | None] = []
        while True:
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
            if item is None:
                break
            items.append(item)

        assert None not in items
        assert any(s for s in items)

    @pytest.mark.asyncio
    async def test_stream_output_has_heading_and_bullet(self, formatter_with_mock) -> None:
        queue: asyncio.Queue = asyncio.Queue()
        await formatter_with_mock.stream("build an API", queue)

        collected: list[str] = []
        while True:
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
            if item is None:
                break
            collected.append(item)

        full_output = "\n".join(collected)
        assert "##" in full_output
        assert "-" in full_output
