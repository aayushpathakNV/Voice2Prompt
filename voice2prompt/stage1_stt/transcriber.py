"""
Stage 1 — Speech-to-Text.

Primary:  NVIDIA Parakeet TDT 0.6B v2 via NeMo.
          RTFx ~900x on RTX 1000 Ada → ~35 ms for 30 s audio.
          Model: nvidia/parakeet-tdt-0.6b-v2

Fallback: faster-whisper (Whisper Large V3 Turbo, CTranslate2 INT8/float16).
          RTFx ~70x on RTX 1000 Ada → ~60-80 ms for 30 s audio.
          Activated when NeMo is not installed or config model is "faster-whisper".

Fallback chain: NeMo Parakeet → faster-whisper
  (HuggingFace transformers does not expose a usable Parakeet TDT decoder
   without NeMo, so we skip that path entirely.)

Budget: <= 100 ms on RTX 1000 Ada (50 W TGP).
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voice2prompt.utils.device import select_device
from voice2prompt.utils.logging import get_logger

logger = get_logger(__name__)

_PARAKEET_V2 = "nvidia/parakeet-tdt-0.6b-v2"
_PARAKEET_V3 = "nvidia/parakeet-tdt-0.6b-v3"
_WHISPER_DEFAULT = "large-v3-turbo"


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
    Wraps Parakeet TDT 0.6B v2 (NeMo) or faster-whisper depending on config
    and runtime availability. Models are lazy-loaded on first transcribe() call.

    Config keys (all optional):
        model:         "parakeet-tdt-0.6b-v2" | "parakeet-tdt-0.6b-v3" | "faster-whisper"
        device:        "auto" | "cuda" | "mps" | "cpu"
        whisper_size:  faster-whisper model size, default "large-v3-turbo"
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._device = select_device(config.get("device", "auto"))
        self._model_name = config.get("model", "parakeet-tdt-0.6b-v2")
        self._whisper_size = config.get("whisper_size", _WHISPER_DEFAULT)
        self._backend: str | None = None
        self._model: Any = None

    async def transcribe(self, audio: bytes | Path) -> TranscriptResult:
        self._load_model()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _load_model(self) -> None:
        if self._model is not None:
            return

        use_parakeet = "parakeet" in self._model_name.lower()

        if use_parakeet:
            loaded = self._try_load_nemo()
            if loaded is not None:
                self._backend, self._model = loaded
                return
            logger.warning("nemo_unavailable: falling back to faster-whisper")

        self._backend, self._model = self._load_faster_whisper()

    def _try_load_nemo(self) -> tuple[str, Any] | None:
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            return None

        nemo_id = _PARAKEET_V3 if "v3" in self._model_name else _PARAKEET_V2
        logger.info(
            "loading_model",
            backend="nemo",
            model=nemo_id,
            device=self._device,
        )
        model = nemo_asr.models.ASRModel.from_pretrained(nemo_id)
        model = model.to(self._device)
        model.eval()
        return ("nemo", model)

    def _load_faster_whisper(self) -> tuple[str, Any]:
        from faster_whisper import WhisperModel

        if self._device == "mps":
            fw_device = "cpu"
            compute_type = "int8"
            logger.warning("faster_whisper_mps_unsupported", fallback="cpu+int8")
        elif self._device == "cuda":
            fw_device = "cuda"
            compute_type = "float16"
        else:
            fw_device = "cpu"
            compute_type = "int8"

        logger.info(
            "loading_model",
            backend="faster-whisper",
            model=self._whisper_size,
            device=fw_device,
            compute_type=compute_type,
        )
        model = WhisperModel(
            self._whisper_size,
            device=fw_device,
            compute_type=compute_type,
        )
        return ("faster_whisper", model)

    def _transcribe_sync(self, audio: bytes | Path) -> TranscriptResult:
        t0 = time.perf_counter()

        if self._backend == "nemo":
            result = self._run_nemo(audio)
        else:
            result = self._run_faster_whisper(audio)

        result.latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "transcription_complete",
            model_id=result.model_id,
            backend=self._backend,
            latency_ms=round(result.latency_ms, 1),
            word_count=len(result.text.split()),
        )
        return result

    def _run_nemo(self, audio: bytes | Path) -> TranscriptResult:
        tmp_path: str | None = None
        try:
            if isinstance(audio, bytes):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio)
                    tmp_path = f.name
                input_path = tmp_path
            else:
                input_path = str(audio)

            output = self._model.transcribe([input_path], timestamps=True)
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        return self._parse_nemo_output(output)

    def _parse_nemo_output(self, output: list) -> TranscriptResult:
        if not output:
            return TranscriptResult(text="", model_id=self._model_name)

        hyp = output[0]

        if isinstance(hyp, str):
            return TranscriptResult(text=hyp.strip(), model_id=self._model_name)

        if hasattr(hyp, "text"):
            text = hyp.text.strip()
        else:
            text = str(hyp).strip()

        words: list[WordTimestamp] = []
        if hasattr(hyp, "timestep") and isinstance(hyp.timestep, dict):
            raw_words = hyp.timestep.get("word", [])
            for entry in raw_words:
                if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                    continue
                words.append(
                    WordTimestamp(
                        word=entry[0],
                        start_s=entry[1],
                        end_s=entry[2],
                    )
                )

        return TranscriptResult(
            text=text,
            model_id=self._model_name,
            word_timestamps=words,
        )

    def _run_faster_whisper(self, audio: bytes | Path) -> TranscriptResult:
        if isinstance(audio, bytes):
            audio_source = io.BytesIO(audio)
        else:
            audio_source = str(audio)

        segments, info = self._model.transcribe(
            audio_source,
            word_timestamps=True,
            language=None,
        )

        text_parts: list[str] = []
        words: list[WordTimestamp] = []

        for segment in segments:
            text_parts.append(segment.text)
            if segment.words:
                for w in segment.words:
                    words.append(
                        WordTimestamp(
                            word=w.word.strip(),
                            start_s=w.start,
                            end_s=w.end,
                        )
                    )

        return TranscriptResult(
            text=" ".join(text_parts).strip(),
            language=info.language,
            duration_s=info.duration,
            word_timestamps=words,
            model_id=f"faster-whisper-{self._whisper_size}",
        )
