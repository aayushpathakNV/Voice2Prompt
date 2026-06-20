"""
Interactive local test for Stage 1 + Stage 2.

Usage:
    py -3.13 scripts/test_stages.py
    py -3.13 scripts/test_stages.py --transcript "um I want to build uh a REST API"
    py -3.13 scripts/test_stages.py --audio path/to/file.wav   # requires faster-whisper
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def run_stage1_filler(transcript: str) -> str:
    from voice2prompt.stage1_stt.filler_pass import filler_pass

    print("\n=== Stage 1: Filler pre-pass ===")
    print(f"Input:  {transcript!r}")
    cleaned = filler_pass(transcript)
    print(f"Output: {cleaned!r}")
    return cleaned


async def run_stage1_stt(audio_path: Path) -> str:
    from voice2prompt.stage1_stt import filler_pass
    from voice2prompt.stage1_stt.transcriber import Transcriber

    print("\n=== Stage 1: STT (faster-whisper) ===")
    print(f"Audio: {audio_path}")
    transcriber = Transcriber({"model": "faster-whisper", "device": "auto"})
    result = await transcriber.transcribe(audio_path)
    print(f"Model:   {result.model_id}")
    print(f"Latency: {result.latency_ms:.1f} ms")
    print(f"Raw:     {result.text!r}")
    cleaned = filler_pass(result.text)
    print(f"Cleaned: {cleaned!r}")
    return cleaned


async def run_stage2(transcript: str, mock: bool) -> None:
    from voice2prompt.stage2_formatter.formatter import Formatter

    print("\n=== Stage 2: Formatter ===")
    formatter = Formatter(
        {
            "model": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
            "n_gpu_layers": -1,
            "max_tokens": 256,
        }
    )

    queue: asyncio.Queue = asyncio.Queue()

    if mock:
        print("(mock mode — reflects your actual transcript, no GGUF needed)")

        def _fake_stream_sync(text: str, q: asyncio.Queue, loop) -> None:
            # Split the cleaned transcript into sentences by period/newline,
            # then emit a ## heading and one bullet per sentence so the
            # output actually reflects the caller's input.
            import re
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
            if not sentences:
                sentences = [text.strip()]

            asyncio.run_coroutine_threadsafe(q.put("## Request"), loop)
            for sentence in sentences:
                # Capitalise and trim to ≤15 words like the real model would
                words = sentence.split()
                bullet = " ".join(words[:15])
                bullet = bullet[0].upper() + bullet[1:] if bullet else bullet
                asyncio.run_coroutine_threadsafe(q.put(f"- {bullet}."), loop)

        formatter._llm = object()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, _fake_stream_sync, transcript, queue, loop
        )
        await queue.put(None)
    else:
        print("(live mode — requires models/Phi-3.5-mini-instruct-Q4_K_M.gguf)")
        await formatter.stream(transcript, queue)

    sentences: list[str] = []
    while True:
        item = await asyncio.wait_for(queue.get(), timeout=120.0)
        if item is None:
            break
        sentences.append(item)
        print(f"  chunk: {item!r}")

    print("\n--- Structured output ---")
    print("\n".join(sentences))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test Voice2Prompt stages 1 and 2")
    parser.add_argument(
        "--transcript",
        default="um so I want to build uh a REST API with JWT auth and deploy it on Kubernetes",
        help="Raw transcript text (skips STT)",
    )
    parser.add_argument("--audio", type=Path, help="WAV file for STT (overrides --transcript)")
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock Stage 2 (default, no GGUF needed)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real Phi-3.5 GGUF model for Stage 2",
    )
    args = parser.parse_args()

    if args.audio:
        transcript = await run_stage1_stt(args.audio)
    else:
        transcript = await run_stage1_filler(args.transcript)

    await run_stage2(transcript, mock=not args.live)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
