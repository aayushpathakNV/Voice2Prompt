"""
Unit tests for Stage 1 — STT transcriber and filler pre-pass.

No GPU or downloaded model required.  The Transcriber tests mock
_load_model / _transcribe_sync so they run on any machine.
A synthetic 1-second 16 kHz sine-wave WAV is generated in-memory
to exercise the audio I/O path of the Transcriber.
"""

from __future__ import annotations

import io
import math
import struct
import wave

import pytest

from voice2prompt.stage1_stt.filler_pass import filler_pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16_000,
                    freq: float = 440.0) -> bytes:
    """Generate a mono 16-bit PCM sine-wave WAV in memory."""
    n_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            sample = int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
            wf.writeframes(struct.pack("<h", sample))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# filler_pass — standalone fillers
# ---------------------------------------------------------------------------

class TestFillerPassStandalone:
    def test_removes_um_and_uh(self):
        assert filler_pass("um I want to build uh an API") == "I want to build an API"

    def test_removes_umm_uhh(self):
        result = filler_pass("umm so uhh let me think")
        assert "umm" not in result
        assert "uhh" not in result

    def test_removes_you_know(self):
        result = filler_pass("It should be, you know, fast.")
        assert "you know" not in result

    def test_removes_you_know_what(self):
        result = filler_pass("You know what, let's use JWT.")
        assert "you know what" not in result
        assert "JWT" in result

    def test_removes_basically_literally(self):
        result = filler_pass("It basically literally works.")
        assert "basically" not in result
        assert "literally" not in result

    def test_removes_i_mean(self):
        result = filler_pass("Use Redis, i mean, it's just faster.")
        assert "i mean" not in result
        assert "Redis" in result

    def test_removes_sort_of(self):
        result = filler_pass("It's sort of a cache layer.")
        assert "sort of" not in result

    def test_removes_kind_of(self):
        result = filler_pass("It kind of works like a queue.")
        assert "kind of" not in result

    def test_removes_essentially(self):
        result = filler_pass("It's essentially a wrapper.")
        assert "essentially" not in result


# ---------------------------------------------------------------------------
# filler_pass — boundary fillers
# ---------------------------------------------------------------------------

class TestFillerPassBoundary:
    def test_removes_right_at_sentence_end(self):
        result = filler_pass("We use Python, right?")
        assert "right" not in result

    def test_removes_okay_at_sentence_start(self):
        result = filler_pass("Okay, let's deploy on Friday.")
        assert result.lower().startswith("let")

    def test_removes_ok_before_period(self):
        import re
        result = filler_pass("That looks good ok.")
        assert not re.search(r"\bok\b", result)

    def test_preserves_right_mid_sentence(self):
        # "right" as an adjective mid-sentence must not be stripped
        result = filler_pass("The right approach is dependency injection.")
        assert "right" in result

    def test_preserves_so_as_sentence_starter(self):
        # "so" was previously (incorrectly) a boundary filler — must survive
        result = filler_pass("So the plan is to use FastAPI.")
        assert "so" in result.lower()

    def test_preserves_well_as_sentence_starter(self):
        result = filler_pass("Well, actually it depends on the context.")
        assert "well" in result.lower()


# ---------------------------------------------------------------------------
# filler_pass — adverbial "like"
# ---------------------------------------------------------------------------

class TestFillerPassLike:
    def test_removes_comma_bounded_like(self):
        result = filler_pass("It was, like, really fast.")
        assert ", like," not in result

    def test_preserves_verb_like(self):
        # "I like Python" — "like" is the main verb, must not be stripped
        result = filler_pass("I like Python for this.")
        assert "like" in result

    def test_preserves_comparative_like(self):
        result = filler_pass("It behaves like a queue.")
        assert "like" in result

    def test_preserves_like_before_determiner(self):
        result = filler_pass("Something like a cache works here.")
        assert "like" in result


# ---------------------------------------------------------------------------
# filler_pass — false starts
# ---------------------------------------------------------------------------

class TestFillerPassFalseStarts:
    def test_removes_simple_false_start(self):
        result = filler_pass("I want to I want to build this.")
        assert result.count("I want to") == 1

    def test_removes_two_word_false_start(self):
        result = filler_pass("We should we should deploy it.")
        assert result.count("we should") <= 1

    def test_no_infinite_loop_on_repeated_words(self):
        # Should terminate in <= 3 passes
        result = filler_pass("the the the the the thing")
        assert result  # just must not hang

    def test_cascade_false_start(self):
        result = filler_pass("I need I need I need to fix this.")
        assert result.count("I need") == 1


# ---------------------------------------------------------------------------
# filler_pass — technical term preservation
# ---------------------------------------------------------------------------

class TestFillerPassPreservation:
    def test_preserves_technical_terms(self):
        text = "Use FastAPI with JWT tokens under 100ms."
        result = filler_pass(text)
        assert "FastAPI" in result
        assert "JWT" in result
        assert "100ms" in result

    def test_preserves_numbers(self):
        result = filler_pass("It should handle um 10000 requests per second.")
        assert "10000" in result

    def test_preserves_proper_nouns(self):
        result = filler_pass("Use basically PostgreSQL and Redis.")
        assert "PostgreSQL" in result
        assert "Redis" in result

    def test_preserves_urls_and_paths(self):
        result = filler_pass("The endpoint is basically /api/v1/users.")
        assert "/api/v1/users" in result

    def test_empty_string(self):
        assert filler_pass("") == ""

    def test_only_fillers(self):
        result = filler_pass("um uh basically")
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# filler_pass — extra_fillers from config
# ---------------------------------------------------------------------------

class TestFillerPassExtraFillers:
    def test_extra_single_word(self):
        result = filler_pass("We should frankly do this.", extra_fillers=["frankly"])
        assert "frankly" not in result

    def test_extra_multi_word(self):
        result = filler_pass("As I was saying, use gRPC.", extra_fillers=["as i was saying"])
        assert "as i was saying" not in result.lower()
        assert "gRPC" in result

    def test_extra_fillers_case_insensitive(self):
        result = filler_pass("FRANKLY this is faster.", extra_fillers=["frankly"])
        assert "frankly" not in result.lower()


# ---------------------------------------------------------------------------
# filler_pass — whitespace and punctuation cleanup
# ---------------------------------------------------------------------------

class TestFillerPassCleanup:
    def test_collapses_whitespace(self):
        result = filler_pass("hello   world")
        assert "  " not in result

    def test_no_leading_comma(self):
        # Removing a boundary filler at the start must not leave ", word"
        result = filler_pass("okay, let's start.")
        assert not result.startswith(",")

    def test_no_double_comma(self):
        result = filler_pass("We use, you know, Redis, okay.")
        assert ",," not in result


# ---------------------------------------------------------------------------
# Transcriber — instantiation and API contract (no model download required)
# ---------------------------------------------------------------------------

class TestTranscriberInstantiation:
    def test_model_not_loaded_at_init(self):
        from voice2prompt.stage1_stt.transcriber import Transcriber
        t = Transcriber({"model": "parakeet-tdt-0.6b-v2", "device": "cpu"})
        assert t._model is None

    def test_defaults(self):
        from voice2prompt.stage1_stt.transcriber import Transcriber
        t = Transcriber({})
        assert "parakeet" in t._model_name
        assert t._device in ("cuda", "mps", "cpu")

    def test_faster_whisper_config(self):
        from voice2prompt.stage1_stt.transcriber import Transcriber
        t = Transcriber({"model": "faster-whisper", "whisper_size": "tiny"})
        assert t._whisper_size == "tiny"


class TestTranscriberMocked:
    """Tests that mock _load_model and _transcribe_sync — no GPU needed."""

    @pytest.fixture
    def transcriber(self):
        from voice2prompt.stage1_stt.transcriber import Transcriber
        return Transcriber({"model": "parakeet-tdt-0.6b-v2", "device": "cpu"})

    @pytest.fixture
    def fake_result(self):
        from voice2prompt.stage1_stt.transcriber import TranscriptResult, WordTimestamp
        return TranscriptResult(
            text="I want to build uh an API that handles user auth.",
            language="en",
            duration_s=3.2,
            word_timestamps=[
                WordTimestamp("I", 0.0, 0.1),
                WordTimestamp("want", 0.1, 0.3),
            ],
            model_id="parakeet-tdt-0.6b-v2",
            latency_ms=42.0,
        )

    @pytest.mark.asyncio
    async def test_transcribe_returns_transcript_result(self, transcriber, fake_result, monkeypatch):
        from voice2prompt.stage1_stt.transcriber import TranscriptResult

        monkeypatch.setattr(transcriber, "_load_model", lambda: None)
        monkeypatch.setattr(transcriber, "_transcribe_sync", lambda _audio: fake_result)
        transcriber._model = object()  # mark as loaded

        result = await transcriber.transcribe(b"fake_audio")
        assert isinstance(result, TranscriptResult)
        assert result.text == fake_result.text
        assert result.model_id == "parakeet-tdt-0.6b-v2"

    @pytest.mark.asyncio
    async def test_transcribe_accepts_path(self, transcriber, fake_result, tmp_path, monkeypatch):
        from pathlib import Path

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(_make_wav_bytes())

        monkeypatch.setattr(transcriber, "_load_model", lambda: None)
        monkeypatch.setattr(transcriber, "_transcribe_sync", lambda _audio: fake_result)
        transcriber._model = object()

        result = await transcriber.transcribe(audio_file)
        assert result.text == fake_result.text

    @pytest.mark.asyncio
    async def test_transcribe_accepts_bytes(self, transcriber, fake_result, monkeypatch):
        monkeypatch.setattr(transcriber, "_load_model", lambda: None)
        monkeypatch.setattr(transcriber, "_transcribe_sync", lambda _audio: fake_result)
        transcriber._model = object()

        result = await transcriber.transcribe(_make_wav_bytes())
        assert result.text == fake_result.text

    def test_latency_is_set_by_transcribe_sync(self, transcriber, monkeypatch):
        """_transcribe_sync must stamp latency_ms on the result."""
        from voice2prompt.stage1_stt.transcriber import TranscriptResult

        raw = TranscriptResult(text="hello", model_id="mock", latency_ms=0.0)

        def _fake_backend(_audio):
            return raw  # latency not yet set

        transcriber._model = ("nemo", object())
        transcriber._backend = "nemo"
        monkeypatch.setattr(transcriber, "_run_nemo", _fake_backend)

        result = transcriber._transcribe_sync(b"fake")
        assert result.latency_ms > 0


class TestNemoOutputParsing:
    """Unit-test the NeMo Hypothesis parser directly without NeMo installed."""

    def _make_transcriber(self):
        from voice2prompt.stage1_stt.transcriber import Transcriber
        t = Transcriber({"model": "parakeet-tdt-0.6b-v2", "device": "cpu"})
        t._model_name = "parakeet-tdt-0.6b-v2"
        return t

    def test_parses_string_output(self):
        t = self._make_transcriber()
        result = t._parse_nemo_output(["hello world"])
        assert result.text == "hello world"

    def test_parses_hypothesis_with_text_attr(self):
        t = self._make_transcriber()

        class FakeHyp:
            text = "build an API"
            timestep = {}

        result = t._parse_nemo_output([FakeHyp()])
        assert result.text == "build an API"

    def test_parses_hypothesis_with_word_timestamps(self):
        t = self._make_transcriber()

        class FakeHyp:
            text = "build an API"
            timestep = {"word": [("build", 0.0, 0.3), ("an", 0.3, 0.4), ("API", 0.4, 0.7)]}

        result = t._parse_nemo_output([FakeHyp()])
        assert len(result.word_timestamps) == 3
        assert result.word_timestamps[0].word == "build"
        assert result.word_timestamps[2].end_s == pytest.approx(0.7)

    def test_handles_empty_output(self):
        t = self._make_transcriber()
        result = t._parse_nemo_output([])
        assert result.text == ""

    def test_handles_missing_timestep(self):
        t = self._make_transcriber()

        class FakeHyp:
            text = "no timestamps"

        result = t._parse_nemo_output([FakeHyp()])
        assert result.text == "no timestamps"
        assert result.word_timestamps == []


class TestFasterWhisperDeviceMapping:
    """Verify the MPS→CPU fallback in the faster-whisper loader."""

    def test_mps_maps_to_cpu(self, monkeypatch):
        from voice2prompt.stage1_stt.transcriber import Transcriber

        loaded = {}

        def _fake_whisper(size, device, compute_type):
            loaded["device"] = device
            loaded["compute_type"] = compute_type
            return object()

        import voice2prompt.stage1_stt.transcriber as mod
        monkeypatch.setattr(
            "voice2prompt.stage1_stt.transcriber.WhisperModel",
            _fake_whisper,
            raising=False,
        )

        # Patch the import inside the method
        import sys
        import types
        fake_fw = types.ModuleType("faster_whisper")
        fake_fw.WhisperModel = _fake_whisper
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

        t = Transcriber({"model": "faster-whisper", "device": "cpu"})
        # Simulate MPS device having been selected
        t._device = "mps"
        t._load_faster_whisper()

        assert loaded.get("device") == "cpu"
        assert loaded.get("compute_type") == "int8"
