"""
Latency and compression benchmarking for Voice2Prompt.

Runs the full pipeline over audio fixtures and reports per-stage latency,
token counts, compression ratio, and ROUGE-L. Results are printed as a
Markdown table and optionally saved to a JSON report file.

Usage:
    python scripts/benchmark.py --audio tests/fixtures/ --report
    python scripts/benchmark.py --audio my_clip.wav --runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Voice2Prompt pipeline benchmark")
    p.add_argument(
        "--audio",
        type=Path,
        default=Path("tests/fixtures"),
        help="Path to a WAV file or directory of WAV files.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
        help="Pipeline config YAML.",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per file (for averaging).",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="Save results to benchmark_report.json.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs before timing (not included in results).",
    )
    return p.parse_args()


def collect_files(audio_path: Path) -> list[Path]:
    if audio_path.is_file():
        return [audio_path]
    files = list(audio_path.glob("*.wav")) + list(audio_path.glob("*.mp3"))
    if not files:
        print(f"No WAV/MP3 files found in {audio_path}", file=sys.stderr)
        sys.exit(1)
    return sorted(files)


async def run_benchmark(args: argparse.Namespace) -> None:
    from voice2prompt.pipeline import Pipeline

    pipeline = Pipeline.from_config(args.config)
    files = collect_files(args.audio)

    print(f"\nBenchmarking {len(files)} file(s), {args.runs} run(s) each, {args.warmup} warmup(s)\n")

    all_results = []

    for audio_file in files:
        audio_bytes = audio_file.read_bytes()

        # Warmup
        for _ in range(args.warmup):
            await pipeline.run(audio_bytes)

        run_results = []
        for run_idx in range(args.runs):
            t0 = time.perf_counter()
            result = await pipeline.run(audio_bytes)
            wall_ms = (time.perf_counter() - t0) * 1000
            m = result.metadata

            run_results.append({
                "file": audio_file.name,
                "run": run_idx + 1,
                "wall_ms": round(wall_ms, 1),
                "stage1_ms": round(m.stage1_latency_ms, 1),
                "stage2_ms": round(m.stage2_latency_ms, 1),
                "stage3_ms": round(m.stage3_latency_ms, 1),
                "total_ms": round(m.total_latency_ms, 1),
                "original_tokens": m.original_tokens,
                "compressed_tokens": m.compressed_tokens,
                "ratio": m.compression_ratio,
                "rouge_l": round(m.rouge_l_estimate, 3),
            })

        all_results.extend(run_results)

        # Per-file summary
        wall_times = [r["wall_ms"] for r in run_results]
        print(f"  {audio_file.name}")
        print(f"    wall P50={statistics.median(wall_times):.0f} ms  "
              f"P95={sorted(wall_times)[int(len(wall_times)*0.95)]:.0f} ms  "
              f"mean={statistics.mean(wall_times):.0f} ms")
        last = run_results[-1]
        print(f"    stage1={last['stage1_ms']} ms  stage2={last['stage2_ms']} ms  "
              f"stage3={last['stage3_ms']} ms")
        if last["original_tokens"]:
            reduction = (1 - last["compressed_tokens"] / last["original_tokens"]) * 100
            print(f"    tokens {last['original_tokens']} → {last['compressed_tokens']} "
                  f"({reduction:.0f}% reduction, ratio {last['ratio']}, ROUGE-L {last['rouge_l']})")
        print()

    # Overall summary
    all_wall = [r["wall_ms"] for r in all_results]
    if all_wall:
        print("=" * 60)
        print(f"OVERALL  n={len(all_wall)}  "
              f"P50={statistics.median(all_wall):.0f} ms  "
              f"P95={sorted(all_wall)[int(len(all_wall)*0.95)]:.0f} ms  "
              f"mean={statistics.mean(all_wall):.0f} ms")
        sla_pass = sum(1 for t in all_wall if t < 1000)
        print(f"< 1 s SLA: {sla_pass}/{len(all_wall)} ({sla_pass/len(all_wall)*100:.0f}%)")

    if args.report:
        report_path = Path("benchmark_report.json")
        report_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nReport saved to {report_path}")


def main():
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
