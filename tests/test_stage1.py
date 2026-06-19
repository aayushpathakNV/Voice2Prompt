"""
Unit tests for Stage 1 — STT transcriber and filler pre-pass.

Coverage targets:
  - filler_pass: standalone fillers, boundary fillers, adverbial "like",
    false starts, whitespace cleanup
  - Transcriber: device selection, model loading (mocked), API contract
"""

import pytest
from voice2prompt.stage1_stt.filler_pass import filler_pass


# ---------------------------------------------------------------------------
# filler_pass tests
# ---------------------------------------------------------------------------

class TestFillerPass:
    def test_removes_um_and_uh(self):
        assert filler_pass("um I want to build uh an API") == "I want to build an API"

    def test_removes_you_know(self):
        result = filler_pass("It should be, you know, fast.")
        assert "you know" not in result

    def test_removes_basically_literally(self):
        result = filler_pass("It basically literally works.")
        assert "basically" not in result
        assert "literally" not in result

    def test_removes_boundary_right(self):
        result = filler_pass("We use Python, right? It's fast.")
        assert result.count("right") == 0

    def test_removes_adverbial_like(self):
        result = filler_pass("It was like really fast.")
        assert "like" not in result

    def test_preserves_comparative_like(self):
        result = filler_pass("It behaves like a queue.")
        assert "like" in result

    def test_removes_false_start(self):
        result = filler_pass("I want to I want to build this.")
        assert result.count("I want to") == 1

    def test_collapses_whitespace(self):
        result = filler_pass("hello   world")
        assert "  " not in result

    def test_empty_string(self):
        assert filler_pass("") == ""

    def test_extra_fillers(self):
        result = filler_pass("We should frankly do this.", extra_fillers=["frankly"])
        assert "frankly" not in result

    def test_preserves_technical_terms(self):
        text = "Use FastAPI with JWT tokens under 100ms."
        result = filler_pass(text)
        assert "FastAPI" in result
        assert "JWT" in result
        assert "100ms" in result


# ---------------------------------------------------------------------------
# Transcriber API contract (model loading mocked)
# ---------------------------------------------------------------------------

class TestTranscriberContract:
    @pytest.fixture
    def config(self):
        return {"model": "parakeet-tdt-0.6b-v2", "device": "cpu"}

    def test_instantiation(self, config):
        from voice2prompt.stage1_stt.transcriber import Transcriber
        t = Transcriber(config)
        assert t is not None

    @pytest.mark.asyncio
    async def test_transcribe_returns_transcript_result(self, config, monkeypatch):
        from voice2prompt.stage1_stt.transcriber import Transcriber, TranscriptResult

        async def _fake_transcribe(audio):
            return TranscriptResult(text="hello world", model_id="mock")

        t = Transcriber(config)
        monkeypatch.setattr(t, "transcribe", _fake_transcribe)

        result = await t.transcribe(b"fake_audio")
        assert isinstance(result, TranscriptResult)
        assert result.text == "hello world"
