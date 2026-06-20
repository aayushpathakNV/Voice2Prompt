"""
Unit tests for Stage 1 — STT transcriber and filler pre-pass.

No GPU or downloaded model required.  The Transcriber tests mock
_load_model / _transcribe_sync so they run on any machine.
A synthetic 1-second 16 kHz sine-wave WAV is generated in-memory
to exercise the audio I/O path of the Transcriber.
"""

from __future__ import annotations

import io
import struct
import wave

import pytest

from voice2prompt.stage1_stt.filler_pass import filler_pass
from voice2prompt.stage1_stt.transcriber import Transcriber, TranscriptResult


def _make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000, freq: float = 440.0) -> bytes:
    n_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            sample = int(32767 * 0.3 * __import__("math").sin(2 * 3.14159 * freq * i / sample_rate))
            wf.writeframes(struct.pack("<h", sample))
    return buf.getvalue()


class TestFillerPassStandalone:
    def test_removes_um_and_uh(self) -> None:
        assert filler_pass("um I want to build uh an API") == "I want to build an API"

    def test_removes_umm_uhh(self) -> None:
        result = filler_pass("umm so uhh let me think")
        assert "umm" not in result
        assert "uhh" not in result

    def test_removes_you_know(self) -> None:
        assert filler_pass("you know we need caching") == "we need caching"

    def test_removes_you_know_what(self) -> None:
        assert filler_pass("you know what I mean") == ""

    def test_removes_basically_literally(self) -> None:
        assert filler_pass("basically we need Redis literally for caching") == "we need Redis for caching"

    def test_removes_i_mean(self) -> None:
        assert filler_pass("i mean it should be fast") == "it should be fast"

    def test_removes_sort_of(self) -> None:
        result = filler_pass("It's sort of a cache layer.")
        assert "sort of" not in result

    def test_removes_kind_of(self) -> None:
        assert filler_pass("kind of important feature") == "important feature"

    def test_removes_essentially(self) -> None:
        assert filler_pass("essentially a rewrite") == "a rewrite"


class TestFillerPassBoundary:
    def test_removes_right_at_sentence_end(self) -> None:
        assert filler_pass("Sounds good, right.") == "Sounds good."

    def test_removes_okay_at_sentence_start(self) -> None:
        assert filler_pass("Okay let's start.") == "let's start."

    def test_removes_ok_before_period(self) -> None:
        assert filler_pass("That works ok.") == "That works."

    def test_preserves_right_mid_sentence(self) -> None:
        assert "right" in filler_pass("turn right at the corner")

    def test_preserves_so_as_sentence_starter(self) -> None:
        assert filler_pass("So we need Redis").startswith("So")

    def test_preserves_well_as_sentence_starter(self) -> None:
        assert filler_pass("Well that depends").startswith("Well")


class TestFillerPassLike:
    def test_removes_comma_bounded_like(self) -> None:
        result = filler_pass("It was, like, really fast.")
        assert ", like," not in result

    def test_preserves_verb_like(self) -> None:
        assert filler_pass("I like Python") == "I like Python"

    def test_preserves_comparative_like(self) -> None:
        assert filler_pass("looks like rain") == "looks like rain"

    def test_preserves_like_before_determiner(self) -> None:
        assert filler_pass("like a charm") == "like a charm"


class TestFillerPassFalseStarts:
    def test_removes_simple_false_start(self) -> None:
        assert filler_pass("I want I want to build an API") == "I want to build an API"

    def test_removes_two_word_false_start(self) -> None:
        assert filler_pass("we need we need caching") == "we need caching"

    def test_no_infinite_loop_on_repeated_words(self) -> None:
        result = filler_pass("the the the the the")
        assert result.count("the") <= 5

    def test_cascade_false_start(self) -> None:
        result = filler_pass("I want to I want to build")
        assert "I want to I want to" not in result


class TestFillerPassPreservation:
    def test_preserves_technical_terms(self) -> None:
        text = "use CUDA 12.4 and TensorRT 10.0"
        assert filler_pass(text) == text

    def test_preserves_numbers(self) -> None:
        text = "latency under 850 ms for 30 s audio"
        assert filler_pass(text) == text

    def test_preserves_proper_nouns(self) -> None:
        text = "deploy on NVIDIA NIM"
        assert filler_pass(text) == text

    def test_preserves_urls_and_paths(self) -> None:
        text = "see https://api.openai.com/v1 and /etc/config.yaml"
        assert filler_pass(text) == text

    def test_empty_string(self) -> None:
        assert filler_pass("") == ""

    def test_only_fillers(self) -> None:
        assert filler_pass("um uh you know") == ""


class TestFillerPassExtraFillers:
    def test_extra_single_word(self) -> None:
        assert filler_pass("so actually we need Redis", extra_fillers=["actually"]) == "so we need Redis"

    def test_extra_multi_word(self) -> None:
        assert filler_pass("at the end of the day it works", extra_fillers=["at the end of the day"]) == "it works"

    def test_extra_fillers_case_insensitive(self) -> None:
        assert filler_pass("ACTUALLY it works", extra_fillers=["actually"]) == "it works"


class TestFillerPassCleanup:
    def test_collapses_whitespace(self) -> None:
        assert filler_pass("too   many    spaces") == "too many spaces"

    def test_no_leading_comma(self) -> None:
        assert not filler_pass(", um hello").startswith(",")

    def test_no_double_comma(self) -> None:
        assert ",," not in filler_pass("hello, um, world")


class TestTranscriberInstantiation:
    def test_model_not_loaded_at_init(self) -> None:
        t = Transcriber({})
        assert t._model is None

    def test_defaults(self) -> None:
        t = Transcriber({})
        assert t._model_name == "parakeet-tdt-0.6b-v2"
        assert t._whisper_size == "large-v3-turbo"

    def test_faster_whisper_config(self) -> None:
        t = Transcriber({"model": "faster-whisper", "whisper_size": "base"})
        assert t._model_name == "faster-whisper"
        assert t._whisper_size == "base"


class TestTranscriberMocked:
    @pytest.fixture
    def transcriber(self) -> Transcriber:
        return Transcriber({})

    @pytest.fixture
    def fake_result(self) -> TranscriptResult:
        return TranscriptResult(text="hello world", model_id="mock", latency_ms=0.0)

    @pytest.mark.asyncio
    async def test_transcribe_returns_transcript_result(
        self, transcriber: Transcriber, fake_result: TranscriptResult, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(transcriber, "_load_model", lambda: None)
        monkeypatch.setattr(transcriber, "_transcribe_sync", lambda audio: fake_result)
        transcriber._model = object()
        result = await transcriber.transcribe(b"fake")
        assert isinstance(result, TranscriptResult)
        assert result.text == fake_result.text

    @pytest.mark.asyncio
    async def test_transcribe_accepts_path(
        self, transcriber: Transcriber, fake_result: TranscriptResult, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(transcriber, "_load_model", lambda: None)
        monkeypatch.setattr(transcriber, "_transcribe_sync", lambda audio: fake_result)
        transcriber._model = object()
        result = await transcriber.transcribe(__import__("pathlib").Path("test.wav"))
        assert result.text == fake_result.text

    @pytest.mark.asyncio
    async def test_transcribe_accepts_bytes(
        self, transcriber: Transcriber, fake_result: TranscriptResult, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(transcriber, "_load_model", lambda: None)
        monkeypatch.setattr(transcriber, "_transcribe_sync", lambda audio: fake_result)
        transcriber._model = object()
        result = await transcriber.transcribe(_make_wav_bytes())
        assert result.text == fake_result.text

    def test_latency_is_set_by_transcribe_sync(
        self, transcriber: Transcriber, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = TranscriptResult(text="hello", model_id="mock", latency_ms=0.0)

        def _fake_backend(audio) -> TranscriptResult:
            return raw

        transcriber._model = object()
        transcriber._backend = "nemo"
        monkeypatch.setattr(transcriber, "_run_nemo", _fake_backend)
        result = transcriber._transcribe_sync(b"fake")
        assert result.latency_ms > 0


class TestNemoOutputParsing:
    def _make_transcriber(self) -> Transcriber:
        t = Transcriber({"model": "parakeet-tdt-0.6b-v2"})
        return t

    def test_parses_string_output(self) -> None:
        t = self._make_transcriber()
        result = t._parse_nemo_output(["  hello world  "])
        assert result.text == "hello world"

    def test_parses_hypothesis_with_text_attr(self) -> None:
        class Hyp:
            text = "  from attr  "

        t = self._make_transcriber()
        result = t._parse_nemo_output([Hyp()])
        assert result.text == "from attr"

    def test_parses_hypothesis_with_word_timestamps(self) -> None:
        class Hyp:
            text = "hello"
            timestep = {"word": [["hello", 0.0, 0.5]]}

        t = self._make_transcriber()
        result = t._parse_nemo_output([Hyp()])
        assert len(result.word_timestamps) == 1
        assert result.word_timestamps[0].word == "hello"

    def test_handles_empty_output(self) -> None:
        t = self._make_transcriber()
        result = t._parse_nemo_output([])
        assert result.text == ""

    def test_handles_missing_timestep(self) -> None:
        class Hyp:
            text = "no timestamps"

        t = self._make_transcriber()
        result = t._parse_nemo_output([Hyp()])
        assert result.word_timestamps == []


class TestFasterWhisperDeviceMapping:
    def test_mps_maps_to_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types

        class FakeWhisperModel:
            def __init__(self, size, device, compute_type):
                self.size = size
                self.device = device
                self.compute_type = compute_type

        fake_fw = types.ModuleType("faster_whisper")
        fake_fw.WhisperModel = FakeWhisperModel
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

        t = Transcriber({"model": "faster-whisper"})
        t._device = "mps"

        backend, model = t._load_faster_whisper()
        assert backend == "faster_whisper"
        assert model.device == "cpu"
        assert model.compute_type == "int8"
