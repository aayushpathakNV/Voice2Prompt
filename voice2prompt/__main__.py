"""
CLI entry point for Voice2Prompt.

Usage:
    voice2prompt record              # capture from microphone until silence
    voice2prompt hotkey              # hold Ctrl+Shift+R to record (push-to-talk loop)
    voice2prompt run <audio_file>    # process a WAV/MP3/M4A file
    voice2prompt devices             # list available input devices
    voice2prompt --help

All commands write the compressed prompt to stdout so it can be piped:
    voice2prompt run clip.wav | pbcopy        (macOS)
    voice2prompt run clip.wav | clip          (Windows)
    voice2prompt run clip.wav | xclip -sel c  (Linux)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _find_config() -> Path:
    """Locate config/pipeline.yaml relative to CWD or package root."""
    candidates = [
        Path("config/pipeline.yaml"),
        Path(__file__).parents[1] / "config" / "pipeline.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "config/pipeline.yaml not found. Run from the project root or pass --config."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice2prompt",
        description="Local voice → compressed prompt pipeline",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to pipeline.yaml")
    parser.add_argument(
        "--json", action="store_true", help="Output full JSON result instead of prompt text"
    )
    parser.add_argument(
        "--no-emit", action="store_true", help="Skip cloud API call; print compressed prompt only"
    )

    sub = parser.add_subparsers(dest="command")

    # record
    rec = sub.add_parser("record", help="Capture from microphone until silence")
    rec.add_argument("--device", default=None, help="Input device index or name")
    rec.add_argument(
        "--silence", type=float, default=1.5, help="Seconds of silence before auto-stop (default 1.5)"
    )
    rec.add_argument(
        "--max-duration", type=float, default=60.0, help="Max recording duration in seconds"
    )

    # run
    run = sub.add_parser("run", help="Process an audio file")
    run.add_argument("file", type=Path, help="WAV, MP3, or M4A file to process")

    # devices
    sub.add_parser("devices", help="List available audio input devices")

    # hotkey  (push-to-talk, Windows Win32 global hotkey)
    hk = sub.add_parser(
        "hotkey",
        help="Push-to-talk: hold hotkey to record, release to process (loops until Ctrl+C)",
    )
    hk.add_argument(
        "--hotkey",
        default="ctrl+shift+r",
        help="Key combo to hold while speaking (default: ctrl+shift+r)",
    )
    hk.add_argument("--device", default=None, help="Input device index or name")

    return parser


async def _run_pipeline(audio_bytes: bytes, config_path: Path, no_emit: bool) -> dict:
    from voice2prompt.pipeline import Pipeline

    pipeline = Pipeline.from_config(config_path)
    result = await pipeline.run(audio_bytes)

    output = {
        "prompt": result.prompt,
        "metadata": {
            "total_latency_ms": round(result.metadata.total_latency_ms, 1),
            "stage1_latency_ms": round(result.metadata.stage1_latency_ms, 1),
            "stage2_latency_ms": round(result.metadata.stage2_latency_ms, 1),
            "stage3_latency_ms": round(result.metadata.stage3_latency_ms, 1),
            "original_tokens": result.metadata.original_tokens,
            "compressed_tokens": result.metadata.compressed_tokens,
            "compression_ratio": result.metadata.compression_ratio,
            "rouge_l_estimate": round(result.metadata.rouge_l_estimate, 3),
        },
        "raw_transcript": result.raw_transcript,
    }

    if not no_emit and result.prompt:
        try:
            emit_result = await pipeline._emitter.send(
                result.prompt,
                original_tokens=result.metadata.original_tokens,
                compressed_tokens=result.metadata.compressed_tokens,
                ratio=result.metadata.compression_ratio,
            )
            output["api_response"] = emit_result.api_response
        except Exception as e:
            output["emit_error"] = str(e)

    return output


def _cmd_devices() -> None:
    try:
        from voice2prompt.audio import list_input_devices
        devices = list_input_devices()
        if not devices:
            print("No input devices found.", file=sys.stderr)
            return
        print(f"{'Index':<6} {'Name'}")
        print("-" * 50)
        for d in devices:
            print(f"{d['index']:<6} {d['name']}")
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice", file=sys.stderr)
        sys.exit(1)


def _cmd_record(args: argparse.Namespace, config_path: Path, output_json: bool, no_emit: bool) -> None:
    try:
        from voice2prompt.audio import AudioCaptureConfig, record_until_silence
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice", file=sys.stderr)
        sys.exit(1)

    cfg = AudioCaptureConfig(
        silence_duration_s=args.silence,
        max_duration_s=args.max_duration,
        device=args.device,
    )

    print("Listening… speak now. Recording stops after silence.", file=sys.stderr)

    def _on_start():
        print("  [recording]", file=sys.stderr)

    def _on_silence():
        print("  [silence detected, stopping]", file=sys.stderr)

    try:
        audio_bytes = record_until_silence(cfg, on_speech_start=_on_start, on_silence_detected=_on_silence)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(_run_pipeline(audio_bytes, config_path, no_emit))
    _print_result(result, output_json)


def _cmd_run(args: argparse.Namespace, config_path: Path, output_json: bool, no_emit: bool) -> None:
    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        from voice2prompt.audio import load_audio_file
        audio_bytes = load_audio_file(args.file)
    except RuntimeError as e:
        print(f"Error loading audio: {e}", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(_run_pipeline(audio_bytes, config_path, no_emit))
    _print_result(result, output_json)


def _cmd_hotkey(args: argparse.Namespace, config_path: Path, output_json: bool, no_emit: bool) -> None:
    """Push-to-talk loop: hold hotkey → record → pipeline → repeat until Ctrl+C."""
    try:
        from voice2prompt.audio import record_push_to_talk
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice", file=sys.stderr)
        sys.exit(1)

    print(f"voice2prompt  |  hotkey: {args.hotkey.upper()}  |  Ctrl+C to quit", file=sys.stderr)
    print("-" * 55, file=sys.stderr)

    try:
        while True:
            audio_bytes = record_push_to_talk(
                hotkey=args.hotkey,
                on_start=lambda: print("  ● Recording…", file=sys.stderr, flush=True),
                on_stop=lambda: print("  ■ Stopped.\n", file=sys.stderr, flush=True),
            )
            if not audio_bytes:
                print("No audio captured (held too briefly — try again).\n", file=sys.stderr)
                continue

            result = asyncio.run(_run_pipeline(audio_bytes, config_path, no_emit))
            _print_result(result, output_json)
            print("-" * 55, file=sys.stderr)

    except KeyboardInterrupt:
        print("\nGoodbye.", file=sys.stderr)


def _print_result(result: dict, output_json: bool) -> None:
    if output_json:
        print(json.dumps(result, indent=2))
    else:
        print(result["prompt"])
        meta = result["metadata"]
        print(
            f"\n[voice2prompt] {meta['total_latency_ms']:.0f} ms | "
            f"{meta['original_tokens']} → {meta['compressed_tokens']} tokens "
            f"({meta['compression_ratio']}) | ROUGE-L {meta['rouge_l_estimate']}",
            file=sys.stderr,
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "devices":
        _cmd_devices()
        return

    config_path = args.config or _find_config()

    if args.command == "record":
        _cmd_record(args, config_path, args.json, args.no_emit)
    elif args.command == "run":
        _cmd_run(args, config_path, args.json, args.no_emit)
    elif args.command == "hotkey":
        _cmd_hotkey(args, config_path, args.json, args.no_emit)


if __name__ == "__main__":
    main()
