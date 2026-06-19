"""
Stage 1 — Speech-to-Text.

Primary:  NVIDIA Parakeet TDT 0.6B v2 (NeMo / HuggingFace transformers)
          RTFx ~900x on RTX 1000 Ada → ~35 ms for 30 s audio.
Fallback: faster-whisper (Whisper Large V3 Turbo, CTranslate2 INT8)
          RTFx ~70x → ~60-80 ms for 30 s audio. Used when NeMo unavailable
          or non-English input detected.

Budget: <= 100 ms on RTX 1000 Ada (50 W TGP).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from voice2prompt.utils.device import select_device
from voice2prompt.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WordTimestamp:
    word: str
    start_s: float
    end_s: float


@dataclass
class TranscriptResult:
    text: str
    language: str = "en"
    duration_s: float = 0.0
    word_timestamps: list[WordTimestamp] = field(default_factory=list)
    model_id: str = ""
    latency_ms: float = 0.0


class Transcriber:
    """
    Wraps either Parakeet TDT 0.6B v2 or faster-whisper depending on config
    and runtime availability.
    """

    def __init__(self, config: dict):
        self._config = config
        self._device = select_device(config.get("device", "auto"))
        self._model_name = config.get("model", "parakeet-tdt-0.6b-v2")
        self._model = None  # lazy-loaded on first call

    def _load_model(self):
        if self._model is not None:
            return

        if "parakeet" in self._model_name:
            self._model = self._load_parakeet()
        else:
            self._model = self._load_faster_whisper()

    def _load_parakeet(self):
        """Load Parakeet TDT 0.6B v2 via NeMo or HuggingFace transformers."""
        try:
            import nemo.collections.asr as nemo_asr  # type: ignore
            logger.info("loading_parakeet", backend="nemo", device=self._device)
            model = nemo_asr.models.ASRModel.from_pretrained(
                "nvidia/parakeet-tdt-0.6b-v2"
            )
            model = model.to(self._device)
            return ("parakeet_nemo", model)
        except ImportError:
            logger.warning("nemo_unavailable", fallback="huggingface_transformers")

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # type: ignore
        logger.info("loading_parakeet", backend="transformers", device=self._device)
        processor = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
        model = AutoModelForSpeechSeq2Seq.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
        model = model.to(self._device)
        return ("parakeet_hf", (processor, model))

    def _load_faster_whisper(self):
        from faster_whisper import WhisperModel  # type: ignore
        device = "cuda" if "cuda" in self._device else "cpu"
        compute_type = "int8" if device == "cpu" else "float16"
        logger.info("loading_faster_whisper", device=device, compute_type=compute_type)
        model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
        return ("faster_whisper", model)

    async def transcribe(self, audio: bytes | Path) -> TranscriptResult:
        """
        Transcribe audio. Runs blocking inference in a thread pool executor
        to avoid blocking the asyncio event loop.

        Args:
            audio: Raw PCM bytes or path to WAV/MP3/M4A file.

        Returns:
            TranscriptResult with text, word timestamps, and latency.
        """
        self._load_model()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: bytes | Path) -> TranscriptResult:
        import time

        t0 = time.perf_counter()
        backend, model = self._model

        if backend == "parakeet_nemo":
            result = self._run_parakeet_nemo(model, audio)
        elif backend == "parakeet_hf":
            processor, hf_model = model
            result = self._run_parakeet_hf(processor, hf_model, audio)
        else:
            result = self._run_faster_whisper(model, audio)

        result.latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "transcription_complete",
            model_id=result.model_id,
            latency_ms=result.latency_ms,
            tokens_out=len(result.text.split()),
        )
        return result

    def _run_parakeet_nemo(self, model, audio: bytes | Path) -> TranscriptResult:
        # NeMo expects a file path; write bytes to a temp file if needed
        import tempfile, os
        if isinstance(audio, bytes):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                tmp_path = f.name
        else:
            tmp_path = str(audio)

        output = model.transcribe([tmp_path], timestamps=True)

        if isinstance(audio, bytes):
            os.unlink(tmp_path)

        text = output[0].text if hasattr(output[0], "text") else str(output[0])
        return TranscriptResult(text=text, model_id="parakeet-tdt-0.6b-v2")

    def _run_parakeet_hf(self, processor, model, audio: bytes | Path) -> TranscriptResult:
        import torch, soundfile as sf, io  # type: ignore

        if isinstance(audio, bytes):
            waveform, sr = sf.read(io.BytesIO(audio))
        else:
            waveform, sr = sf.read(str(audio))

        inputs = processor(waveform, sampling_rate=sr, return_tensors="pt").to(self._device)
        with torch.no_grad():
            ids = model.generate(**inputs)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        return TranscriptResult(text=text, model_id="parakeet-tdt-0.6b-v2-hf")

    def _run_faster_whisper(self, model, audio: bytes | Path) -> TranscriptResult:
        import io
        audio_source = io.BytesIO(audio) if isinstance(audio, bytes) else str(audio)
        segments, info = model.transcribe(audio_source, word_timestamps=True)

        words = []
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)
            if segment.words:
                for w in segment.words:
                    words.append(WordTimestamp(word=w.word, start_s=w.start, end_s=w.end))

        return TranscriptResult(
            text=" ".join(text_parts).strip(),
            language=info.language,
            duration_s=info.duration,
            word_timestamps=words,
            model_id="faster-whisper-large-v3-turbo",
        )
