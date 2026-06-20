"""
voice2prompt CLI entry point.

Usage
-----
# Push-to-talk (hold Ctrl+Shift+R, release to process):
    py -m voice2prompt record

# Push-to-talk with custom hotkey:
    py -m voice2prompt record --hotkey alt+shift+p

# Process a WAV file directly (no mic):
    py -m voice2prompt run audio.wav

# Run Stage 1 only (filler-pass on text):
    py -m voice2prompt clean "um so I want to build uh a REST API"

Stage 2 requires the GGUF model at:
    models/Phi-3.5-mini-instruct-Q4_K_M.gguf

Download from: https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice2prompt",
        description="Local voice → structured prompt pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # record subcommand
    rec = sub.add_parser("record", help="Push-to-talk mic recording (hold hotkey)")
    rec.add_argument(
        "--hotkey",
        default="ctrl+shift+r",
        help="Global hotkey to hold while speaking, e.g. ctrl+shift+r, alt+shift+p (default: ctrl+shift+r)",
    )
    rec.add_argument(
        "--stage2",
        action="store_true",
        default=True,
        help="Run Stage 2 formatter after transcription (default: on)",
    )
    rec.add_argument("--no-stage2", dest="stage2", action="store_false")
    rec.add_argument(
        "--model",
        default="Phi-3.5-mini-instruct-Q4_K_M.gguf",
        help="GGUF model filename inside models/ dir",
    )
    rec.add_argument(
        "--whisper-size",
        default="base",
        dest="whisper_size",
        help="faster-whisper model size: tiny|base|small|medium|large-v3-turbo (default: base)",
    )

    # run subcommand (from file)
    run = sub.add_parser("run", help="Process an existing audio file")
    run.add_argument("audio", type=Path, help="Path to WAV file")
    run.add_argument("--no-stage2", dest="stage2", action="store_false", default=True)
    run.add_argument(
        "--model",
        default="Phi-3.5-mini-instruct-Q4_K_M.gguf",
        help="GGUF model filename inside models/ dir",
    )

    # clean subcommand (text only, no audio)
    clean = sub.add_parser("clean", help="Run Stage 1 filler-pass on raw text")
    clean.add_argument("text", help="Raw transcript text to clean")

    return parser


async def _run_pipeline(
    audio: bytes | Path,
    run_stage2: bool,
    model: str,
    whisper_size: str = "base",
) -> None:
    from voice2prompt.stage1_stt.filler_pass import filler_pass
    from voice2prompt.stage1_stt.transcriber import Transcriber

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    print(f"\n[Stage 1] Loading STT model (faster-whisper '{whisper_size}')…", flush=True)
    print("          (first run downloads the model — may take a moment)", flush=True)
    transcriber = Transcriber({
        "model": "faster-whisper",
        "device": "auto",
        "whisper_size": whisper_size,
    })
    print("[Stage 1] Transcribing audio…", flush=True)
    result = await transcriber.transcribe(audio)

    print(f"  Raw transcript ({result.latency_ms:.0f} ms):  {result.text!r}")
    cleaned = filler_pass(result.text)
    print(f"  After filler-pass:              {cleaned!r}")

    if not run_stage2:
        print("\nResult:\n" + cleaned)
        return

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    from voice2prompt.stage2_formatter.formatter import Formatter

    print("\n[Stage 2] Formatting…")
    formatter = Formatter({"model": model, "n_gpu_layers": -1, "max_tokens": 512})

    queue: asyncio.Queue = asyncio.Queue()
    sentences: list[str] = []

    async def _collect() -> None:
        while True:
            item = await queue.get()
            if item is None:
                break
            sentences.append(item)
            print(f"  {item}")

    await asyncio.gather(
        formatter.stream(cleaned, queue),
        _collect(),
    )

    print("\n── Structured prompt ──────────────────────────────")
    print("\n".join(sentences))
    print("───────────────────────────────────────────────────")


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    if args.command == "clean":
        from voice2prompt.stage1_stt.filler_pass import filler_pass
        print(filler_pass(args.text))
        return

    if args.command == "record":
        from voice2prompt.audio import record_push_to_talk

        print(f"voice2prompt ready  |  hotkey: {args.hotkey.upper()}  |  Ctrl+C to quit")
        print("─" * 55, flush=True)

        try:
            while True:
                audio_bytes = record_push_to_talk(
                    hotkey=args.hotkey,
                    on_start=lambda: print("  ● Recording…", flush=True),
                    on_stop=lambda: print("  ■ Stopped.\n", flush=True),
                )
                if not audio_bytes:
                    print("No audio captured (held too briefly — try again).\n")
                    continue

                asyncio.run(_run_pipeline(
                    audio_bytes, args.stage2, args.model,
                    whisper_size=args.whisper_size,
                ))
                print("\n─" * 28, flush=True)

        except KeyboardInterrupt:
            print("\nGoodbye.")

    elif args.command == "run":
        if not args.audio.exists():
            print(f"File not found: {args.audio}", file=sys.stderr)
            sys.exit(1)
        asyncio.run(_run_pipeline(args.audio, args.stage2, args.model))


if __name__ == "__main__":
    main()
