"""
Cross-platform audio capture via sounddevice.

Supports:
  - Windows: WASAPI (default) or DirectSound
  - macOS:   CoreAudio
  - Linux:   ALSA / PulseAudio / PipeWire (auto-detected by sounddevice)

Captures mono PCM at 16 kHz (required by Parakeet and Whisper).
Silence detection auto-stops recording when the mic goes quiet for
`silence_duration_s` seconds (configurable, default 1.5 s).
"""

from __future__ import annotations

import io
import platform
import queue
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
BLOCK_FRAMES = 1024           # ~64 ms per block at 16 kHz
SILENCE_THRESHOLD_RMS = 300   # int16 RMS below this = silence
MAX_RECORD_SECONDS = 60


@dataclass
class AudioCaptureConfig:
    sample_rate: int = SAMPLE_RATE
    silence_duration_s: float = 1.5
    max_duration_s: float = MAX_RECORD_SECONDS
    device: int | str | None = None   # None = system default
    silence_threshold: int = SILENCE_THRESHOLD_RMS


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def list_input_devices() -> list[dict]:
    """Return all available input devices with index, name, and platform info."""
    import sounddevice as sd  # type: ignore
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def record_until_silence(
    config: AudioCaptureConfig | None = None,
    on_speech_start: Callable[[], None] | None = None,
    on_silence_detected: Callable[[], None] | None = None,
) -> bytes:
    """
    Record from the default microphone until silence is detected or max duration
    is reached. Returns raw WAV bytes (16 kHz, mono, int16).

    Args:
        config:               Capture configuration. Uses defaults if None.
        on_speech_start:      Optional callback fired when first non-silent block arrives.
        on_silence_detected:  Optional callback fired when silence threshold triggers stop.

    Returns:
        WAV-encoded bytes ready to pass to Transcriber.transcribe().

    Raises:
        RuntimeError: If no input device is available.
        ImportError:  If sounddevice is not installed.
    """
    import sounddevice as sd  # type: ignore

    cfg = config or AudioCaptureConfig()
    audio_q: queue.Queue[np.ndarray | None] = queue.Queue()
    silence_blocks_needed = int(
        cfg.silence_duration_s * cfg.sample_rate / BLOCK_FRAMES
    )
    max_blocks = int(cfg.max_duration_s * cfg.sample_rate / BLOCK_FRAMES)

    speech_started = False
    silence_count = 0

    def _callback(indata: np.ndarray, frames: int, time_info, status):
        nonlocal speech_started, silence_count
        block = indata[:, 0].copy()  # mono
        rms = _rms(block)

        if rms > cfg.silence_threshold:
            if not speech_started and on_speech_start:
                on_speech_start()
            speech_started = True
            silence_count = 0
        elif speech_started:
            silence_count += 1
            if silence_count >= silence_blocks_needed:
                if on_silence_detected:
                    on_silence_detected()
                audio_q.put(None)  # sentinel
                raise sd.CallbackStop

        if speech_started:
            audio_q.put(block)

    # Select host API per platform for lowest latency
    extra_kwargs: dict = {}
    system = platform.system()
    if system == "Windows":
        try:
            next(i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"])
            extra_kwargs["extra_settings"] = sd.WasapiSettings(exclusive=False)
        except StopIteration:
            pass  # fall back to default API
    # macOS and Linux: sounddevice picks CoreAudio / ALSA automatically

    blocks: list[np.ndarray] = []
    stop_event = threading.Event()

    def _drain():
        block_count = 0
        while block_count < max_blocks:
            try:
                item = audio_q.get(timeout=cfg.max_duration_s + 1)
            except queue.Empty:
                break
            if item is None:
                break
            blocks.append(item)
            block_count += 1
        stop_event.set()

    drain_thread = threading.Thread(target=_drain, daemon=True)
    drain_thread.start()

    with sd.InputStream(
        samplerate=cfg.sample_rate,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCK_FRAMES,
        device=cfg.device,
        callback=_callback,
        **extra_kwargs,
    ):
        stop_event.wait(timeout=cfg.max_duration_s + 2)

    drain_thread.join(timeout=1.0)

    if not blocks:
        raise RuntimeError("No speech detected. Check microphone permissions and input device.")

    audio_np = np.concatenate(blocks)
    return _to_wav_bytes(audio_np, cfg.sample_rate)


def load_audio_file(path: str | Path) -> bytes:
    """
    Load a WAV, MP3, or M4A file and return WAV bytes resampled to 16 kHz mono.
    Uses soundfile for WAV/FLAC, ffmpeg (via subprocess) for MP3/M4A.
    """
    import soundfile as sf  # type: ignore
    from pathlib import Path as _Path

    p = _Path(path)
    suffix = p.suffix.lower()

    if suffix in (".wav", ".flac", ".ogg"):
        data, sr = sf.read(str(p), dtype="int16", always_2d=True)
        mono = data[:, 0]
        if sr != SAMPLE_RATE:
            mono = _resample(mono, sr, SAMPLE_RATE)
        return _to_wav_bytes(mono, SAMPLE_RATE)

    # MP3 / M4A / AAC — decode via ffmpeg
    return _decode_via_ffmpeg(str(p))


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        import resampy  # type: ignore
        return resampy.resample(audio.astype(np.float32), orig_sr, target_sr).astype(np.int16)
    except ImportError:
        pass
    # Naive linear interpolation fallback (low quality but dependency-free)
    ratio = target_sr / orig_sr
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.int16)


def _decode_via_ffmpeg(path: str) -> bytes:
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install it to load MP3/M4A files:\n"
            "  Windows: winget install ffmpeg\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg"
        )
    result = subprocess.run(
        [
            ffmpeg, "-i", path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "wav",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return result.stdout


def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()
