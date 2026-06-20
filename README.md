# Voice2Prompt

A three-stage, fully local pipeline that transforms spoken input into compressed, structured prompts for cloud LLM APIs.

**Version:** 0.2 (Draft) | **Target:** NVIDIA RTX 1000 Ada Laptop (8 GB GDDR6, 50 W TGP)

```
Microphone/File
      │
      ▼
┌─────────────────┐
│  Stage 1 - STT  │  Parakeet TDT 0.6B v2   ≤ 100 ms
│  + Filler Pass  │  Regex pre-pass          ≤   5 ms
└────────┬────────┘
         │ clean transcript
         ▼
┌─────────────────┐
│ Stage 2 - LLM   │  Phi-3.5 Mini Q4_K_M    ≤ 550 ms  ─── streams sentences ──►┐
│   Formatter     │  (llama-cpp-python)                                          │
└─────────────────┘                                                              │
                                                                                 ▼
                                                                    ┌─────────────────┐
                                                                    │ Stage 3 -       │  LLMLingua-2   ≤ 300 ms
                                                                    │ Compressor      │  (overlapped)
                                                                    └────────┬────────┘
                                                                             │ ≥60% token reduction
                                                                             ▼
                                                                    Cloud LLM API
                                                              (OpenAI / Anthropic / NIM)
```

**End-to-end P95 latency: ~850 ms on RTX 1000 Ada** (< 1 s target)

---

## Features

- **Privacy-first:** Audio and transcripts never leave the local machine during Stages 1–3
- **Token compression:** ≥ 60% token reduction vs. raw transcript, saving API cost and latency
- **Streaming pipeline:** Stage 3 overlaps with Stage 2 tail, saving ~120 ms vs. sequential execution
- **VRAM-efficient:** All three models fit concurrently in 8 GB VRAM
- **Multi-platform:** Windows 11 (CUDA 12.x), macOS 14+ (Apple Silicon MPS), Linux CUDA
- **Configurable:** Swap models per stage via `config/pipeline.yaml` without code changes
- **OpenAI-compatible output:** Works with any OpenAI-compatible endpoint (NIM, vLLM, Ollama)

---

## Quickstart

### Prerequisites

- Python 3.10+
- CUDA 12.x (Windows/Linux) or macOS 14+ with Apple Silicon
- ~6 GB VRAM free at startup

### Install

```bash
pip install -e ".[dev]"
```

### Run

```bash
# From microphone
voice2prompt record

# From audio file
voice2prompt run audio.wav

# Python API
from voice2prompt import Pipeline
pipeline = Pipeline.from_config("config/pipeline.yaml")
result = await pipeline.run(audio_bytes)
print(result.prompt)
print(result.metadata)  # token savings, latency breakdown
```

---

## Project Structure

```
voice2prompt/
├── voice2prompt/            # Main package
│   ├── pipeline.py          # Async orchestrator (asyncio.Queue between stages)
│   ├── stage1_stt/
│   │   ├── transcriber.py   # Parakeet TDT 0.6B v2 + faster-whisper fallback
│   │   └── filler_pass.py   # Regex/trie filler removal (< 5 ms)
│   ├── stage2_formatter/
│   │   └── formatter.py     # llama-cpp-python streaming (Phi-3.5 Mini Q4_K_M)
│   ├── stage3_compressor/
│   │   └── compressor.py    # LLMLingua-2 streaming consumer
│   ├── emit/
│   │   └── api_client.py    # OpenAI / Anthropic emit layer
│   └── utils/
│       ├── device.py        # CUDA → MPS → CPU auto-detection
│       └── logging.py       # Structured JSON logging
├── config/
│   └── pipeline.yaml        # Stage model selection and tuning knobs
├── tests/
│   ├── fixtures/            # 10 audio fixtures for integration tests
│   ├── test_stage1.py
│   ├── test_stage2.py
│   ├── test_stage3.py
│   └── test_pipeline.py
├── scripts/
│   └── benchmark.py         # Latency and compression benchmarking
├── docs/
│   └── architecture.md      # Deep-dive: streaming design, VRAM budget, latency trace
├── CLAUDE.md                # AI assistant context for this repo
└── pyproject.toml
```

---

## VRAM Budget (RTX 1000 Ada, 8 GB)

| Component          | VRAM     |
|--------------------|----------|
| Parakeet 0.6B      | ~0.8 GB  |
| Phi-3.5 Mini Q4    | ~2.5 GB  |
| LLMLingua-2 BERT   | ~1.5 GB  |
| OS + drivers       | ~1.2 GB  |
| Headroom           | ~2.0 GB  |
| **Total**          | **~8.0 GB** |

---

## Latency Budget (RTX 1000 Ada, 30 s audio)

| Event                            | t (ms)  |
|----------------------------------|---------|
| Audio capture complete           | 0       |
| Parakeet transcription done      | ~35     |
| Filler pre-pass done; Stage 2 starts | ~40 |
| Stage 2 first sentence → Stage 3 starts | ~165 |
| Stage 2 complete                 | ~590    |
| Stage 3 flush complete           | ~710    |
| Compressed prompt ready          | < 800   |

---

## Configuration

Edit `config/pipeline.yaml` to swap models or tune compression:

```yaml
stage1:
  model: parakeet-tdt-0.6b-v2   # or: faster-whisper
  device: auto                   # cuda | mps | cpu

stage2:
  model: Phi-3.5-mini-instruct-Q4_K_M.gguf
  n_gpu_layers: -1               # full GPU offload
  max_tokens: 512

stage3:
  model: microsoft/llmlingua-2-bert-large-multilingual-cased-meetingbank
  rate: 0.4                      # retain 40% of tokens
  force_tokens: ["##", "-", "**"]

emit:
  format: openai                 # openai | anthropic
  base_url: https://api.openai.com/v1
  attach_metadata: true
```

---

## Development

```bash
# Run tests
pytest tests/ -v

# Run benchmarks
python scripts/benchmark.py --audio tests/fixtures/ --report

# Type checking
mypy voice2prompt/
```

Test coverage target: ≥ 80% per stage. Integration tests use the 10 audio fixtures in `tests/fixtures/`.

---

## Hardware Tiers

| Tier | GPU | VRAM | P95 E2E |
|------|-----|------|---------|
| Primary target | RTX 1000 Ada Laptop (50W) | 8 GB | ~850 ms |
| Tier 1+ | RTX 3060/4060 Laptop (80–100W) | 8 GB | ~700 ms |
| Tier 2 | RTX 3080/4070 Desktop | 10–12 GB | ~600 ms |
| Tier 3 | Apple M3 Pro / M4 (MPS) | Unified 18–36 GB | ~900 ms |
| CPU fallback | No discrete GPU | — | ~5–7 s |

---

## License

Models used are open-weight with commercial licenses:
- Parakeet TDT 0.6B v2: CC BY 4.0
- Phi-3.5 Mini Instruct: MIT
- LLMLingua-2: MIT

See `docs/architecture.md` for full design rationale.
