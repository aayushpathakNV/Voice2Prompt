"""
Microphone capture for Voice2Prompt.

Push-to-talk: HOLD the hotkey (default Alt+P) to record, RELEASE to stop.
  - Windows: uses Win32 RegisterHotKey + GetAsyncKeyState (no admin needed).
  - Other:   falls back to Enter-to-start / Enter-to-stop.

Returns WAV bytes (16-bit mono 16 kHz) ready for faster-whisper / Parakeet.

Dependencies: sounddevice, numpy  (both in the [ml] extra)
"""

from __future__ import annotations

import io
import queue
import struct
import time
import wave
from typing import Callable

import numpy as np

from voice2prompt.utils.logging import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
BLOCKSIZE = 1024  # ~64 ms per block


def _frames_to_wav(frames: list[np.ndarray]) -> bytes:
    """Pack raw PCM frames into an in-memory WAV file."""
    pcm = np.concatenate(frames, axis=0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def record_push_to_talk(
    hotkey: str = "ctrl+shift+r",
    on_start: Callable[[], None] | None = None,
    on_stop: Callable[[], None] | None = None,
    _use_enter_fallback: bool = False,
) -> bytes:
    """
    Hold `hotkey` to record; release to stop — just like WhisperFlow.

    On Windows this uses Win32 RegisterHotKey + GetAsyncKeyState so it
    works even when another window has focus.  No admin rights required.
    Pass _use_enter_fallback=True to force the Enter-to-start/stop mode.

    on_start: optional callback fired when recording begins.
    on_stop:  optional callback fired when recording ends.
    Returns WAV bytes, or b"" if nothing was captured.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Audio capture requires 'sounddevice'. "
            "Install with:  pip install sounddevice"
        ) from exc

    import sys
    import threading

    frames: list[np.ndarray] = []
    audio_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def _sd_callback(indata: np.ndarray, frames_count: int, time_info, status) -> None:
        if status:
            logger.warning("sounddevice_status", status=str(status))
        audio_q.put(indata.copy())

    def _drain_queue() -> None:
        while not stop_event.is_set():
            try:
                frames.append(audio_q.get(timeout=0.05))
            except queue.Empty:
                pass
        while not audio_q.empty():
            try:
                frames.append(audio_q.get_nowait())
            except queue.Empty:
                break

    # ── Choose trigger mechanism ─────────────────────────────────────────────
    use_win32 = sys.platform == "win32" and not _use_enter_fallback

    if use_win32:
        try:
            from voice2prompt.hotkey import is_hotkey_held, wait_for_hotkey_press
            print(f"\nHold [{hotkey.upper()}] to record — release to process...", flush=True)
            wait_for_hotkey_press(hotkey)      # blocks until key-down

            logger.info("recording_started", hotkey=hotkey)
            if on_start:
                on_start()

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCKSIZE,
                callback=_sd_callback,
            ):
                drain_thread = threading.Thread(target=_drain_queue, daemon=True)
                drain_thread.start()
                while is_hotkey_held(hotkey):  # loop until key-up
                    time.sleep(0.01)
                stop_event.set()
                drain_thread.join(timeout=1.0)

        except OSError as exc:
            print(f"\n  [hotkey] {exc}", flush=True)
            print("  Falling back to Enter-to-start / Enter-to-stop.\n", flush=True)
            use_win32 = False  # handled below

    if not use_win32:
        # Fallback: Enter to start, Enter to stop
        con = "CON" if sys.platform == "win32" else "/dev/tty"
        print("\nPress [Enter] to START recording...", flush=True)
        try:
            input()
        except EOFError:
            with open(con) as tty:
                tty.readline()

        logger.info("recording_started")
        if on_start:
            on_start()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCKSIZE,
            callback=_sd_callback,
        ):
            drain_thread = threading.Thread(target=_drain_queue, daemon=True)
            drain_thread.start()
            print("Press [Enter] to STOP recording...", flush=True)
            try:
                input()
            except EOFError:
                with open(con) as tty:
                    tty.readline()
            stop_event.set()
            drain_thread.join(timeout=1.0)

    logger.info("recording_stopped", frames=len(frames))
    if on_stop:
        on_stop()

    return _frames_to_wav(frames) if frames else b""


def record_fixed(seconds: float = 5.0) -> bytes:
    """Record for a fixed duration and return WAV bytes. Useful for testing."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Audio capture requires 'sounddevice'. "
            "Install with:  pip install sounddevice"
        ) from exc

    logger.info("recording_fixed", seconds=seconds)
    pcm = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    sd.wait()
    return _frames_to_wav([pcm])
