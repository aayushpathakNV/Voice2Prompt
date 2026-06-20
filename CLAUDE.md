# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_stage1.py -v

# Run a single test by name
pytest tests/test_stage1.py::TestFillerPass::test_removes_um_and_uh -v

# Lint
ruff check voice2prompt/ tests/

# Type check
mypy voice2prompt/

# Benchmark (requires audio fixtures in tests/fixtures/)
python scripts/benchmark.py --audio tests/fixtures/ --report
python scripts/benchmark.py --audio my_clip.wav --runs 5 --warmup 2

# CLI (after install)
voice2prompt record            # live microphone
voice2prompt run audio.wav     # from file
```

`pytest-asyncio` is configured with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` decorator needed on individual tests (though it's present on some for clarity).

## Architecture

Three stages execute as a **streaming pipeline**, not sequentially. The key design decision: Stage 3 starts processing before Stage 2 finishes, saving ~120 ms on the primary target (RTX 1000 Ada, 50W, 8 GB VRAM).

```
audio → [Stage 1: STT + filler pre-pass] → clean_text
                                               ↓
                              [Stage 2: LLM Formatter]  ──sentences──→  [Stage 3: Compressor]
                                    (llama-cpp-python)    asyncio.Queue   (LLMLingua-2)
                                                                              ↓
                                                                     compressed prompt → API
```

**The queue is load-bearing.** `pipeline.py` creates an `asyncio.Queue(maxsize=32)`, passes it to both `formatter.stream()` and `compressor.compress_stream()`, then runs both with `asyncio.gather`. Stage 2 puts completed sentences into the queue; Stage 3 consumes them. Stage 2 puts `None` as a sentinel when done. Do not replace this with sequential awaits.

### Stage responsibilities

- **Stage 1** (`stage1_stt/`): `Transcriber` wraps Parakeet TDT 0.6B v2 (NeMo → HF transformers fallback) or faster-whisper. Runs blocking inference in a thread pool executor so it doesn't block the event loop. After transcription, `filler_pass()` runs synchronously on CPU (< 5 ms) to strip fillers and false starts before Stage 2 sees the text.

- **Stage 2** (`stage2_formatter/`): `Formatter` uses llama-cpp-python with `stream=True`. Tokens are buffered into sentences and each completed sentence is put onto the queue via `asyncio.run_coroutine_threadsafe` (because llama.cpp runs in a thread pool executor). **Must use a sub-4B model** — 8B at ~70–90 tok/s on RTX 1000 Ada takes ~1.2 s for 100 tokens, exceeding the 550 ms budget.

- **Stage 3** (`stage3_compressor/`): `Compressor` consumes the sentence queue, accumulates a buffer, then calls LLMLingua-2 in a thread pool executor when the sentinel arrives. The ROUGE-L method (`_estimate_rouge_l`) is a lightweight LCS approximation used at runtime; full evaluation uses `rouge-score` in tests.

- **Emit** (`emit/api_client.py`): Wraps compressed prompt in OpenAI Chat Completions or Anthropic Messages format. Accepts any `base_url` for OpenAI-compatible endpoints (NVIDIA NIM, vLLM, Ollama). API key is read from `VOICE2PROMPT_API_KEY` env var.

### Model loading

All three models are **lazy-loaded** on the first call to avoid paying startup cost unless the stage is used. Calling `_load_model()` a second time is a no-op. Models stay resident in VRAM for the lifetime of the process — there is no model swapping between requests.

VRAM budget at steady state (RTX 1000 Ada, 8 GB): Parakeet ~0.8 GB + Phi-3.5 Mini Q4 ~2.5 GB + LLMLingua-2 ~1.5 GB + OS/drivers ~1.2 GB ≈ 6 GB used, ~2 GB headroom.

### Config

`config/pipeline.yaml` is the only place to change models or tuning knobs. Stage classes read their sub-dict at `__init__` time. Never hardcode model paths in Python.

### Cross-platform device selection

`utils/device.py:select_device()` resolves `"auto"` → CUDA → MPS → CPU in that order, logging a warning on each fallback. All stage constructors call this at init. CPU fallback is supported but degrades P95 latency to ~5–7 s.

## Public API contracts

Do not change these signatures without updating all call sites and tests:

```python
# pipeline.py
Pipeline.from_config(config_path: str | Path) -> Pipeline
await pipeline.run(audio: bytes | Path) -> PipelineResult

# stage1_stt/transcriber.py
await transcriber.transcribe(audio: bytes | Path) -> TranscriptResult

# stage1_stt/filler_pass.py  (sync, CPU-only)
filler_pass(text: str, extra_fillers: list[str] | None) -> str

# stage2_formatter/formatter.py
await formatter.stream(transcript: str, queue: asyncio.Queue) -> None

# stage3_compressor/compressor.py
await compressor.compress_stream(queue: asyncio.Queue) -> CompressResult
await compressor.compress(prompt: str, dry_run: bool) -> CompressResult  # single-shot, for tests
```

## Testing approach

- Unit tests mock the model (`_load_model` + `_compress_sync` / `_stream_sync`) but use real `asyncio.Queue` instances — never mock the queue.
- `test_pipeline.py` validates orchestration logic (queue handoff, gather, result shape) without GPU by monkeypatching all three stages.
- `tests/fixtures/` holds `.wav` files for integration tests; the fixture-based tests in `test_pipeline.py` auto-skip when the directory is empty.
- Latency assertions do not belong in tests — use `scripts/benchmark.py` for those.

## Out of scope for v1

No GUI, no real-time cloud streaming, no multi-speaker diarization, no non-English languages, no model fine-tuning.
