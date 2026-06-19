# Voice2Prompt — AI Assistant Context

## What This Project Does

Three-stage local pipeline: voice → structured markdown → compressed prompt → cloud LLM API.

- **Stage 1:** Parakeet TDT 0.6B v2 (NeMo/HF) for STT + regex filler pre-pass
- **Stage 2:** Phi-3.5 Mini Instruct Q4_K_M via llama-cpp-python (streaming output)
- **Stage 3:** LLMLingua-2 (BERT-large) consuming Stage 2's token stream concurrently
- **Target:** NVIDIA RTX 1000 Ada Laptop (8 GB GDDR6, 50 W TGP) — P95 E2E < 1 s

## Architecture Constraints to Always Respect

1. **Stages 1–3 are fully local.** No audio or transcript ever leaves the machine before Stage 3 output. Do not suggest cloud STT or cloud formatting.

2. **Stage 2 must use a sub-4B model.** Llama 3.1 8B cannot meet the 550 ms budget on RTX 1000 Ada (~70–90 tok/s → ~1.2 s for 100 tokens). The default is Phi-3.5 Mini Q4_K_M. 8B is only valid as an explicit "quality mode" option.

3. **Streaming pipeline is load-bearing.** Stage 3 subscribes to Stage 2's sentence queue via `asyncio.Queue`. The ~120 ms overlap is required to hit < 1 s total. Do not replace with sequential await.

4. **VRAM budget is 8 GB total.** All three models must fit concurrently: Parakeet (~0.8 GB) + Phi-3.5 Mini Q4 (~2.5 GB) + LLMLingua-2 (~1.5 GB) + OS/drivers (~1.2 GB) = ~6 GB used, ~2 GB headroom. Any model swap must be checked against this budget.

5. **Stage model swaps happen via `config/pipeline.yaml` only.** No hardcoded model paths in Python. Stage classes read config at init.

## Key Latency Budgets (RTX 1000 Ada)

| Stage | Budget |
|-------|--------|
| Stage 1 STT | ≤ 100 ms |
| Stage 1 filler pre-pass | ≤ 5 ms (CPU, regex) |
| Stage 2 Formatter | ≤ 550 ms |
| Stage 3 Compressor | ≤ 300 ms (overlapped, not sequential) |
| Emit + overhead | ≤ 45 ms |
| **Total P95** | **~850 ms** |

## Python API Contracts

Each stage exposes exactly one public async method:

```python
# Stage 1
transcribe(audio: bytes | Path) -> TranscriptResult

# Stage 2
format_transcript(raw: str) -> AsyncIterator[str]  # streams sentences

# Stage 3
compress(prompt: AsyncIterator[str]) -> CompressResult

# Orchestrator
pipeline.run(audio: bytes) -> PipelineResult
```

Do not change these signatures without updating all call sites and tests.

## Tech Stack Choices (Do Not Swap Without Explicit Approval)

| Component | Choice | Why |
|-----------|--------|-----|
| STT runtime | NeMo / HF transformers | Parakeet TDT only available here |
| LLM runtime | llama-cpp-python | Needed for GGUF Q4, full GPU offload, streaming |
| Compression | llmlingua (pip) | LLMLingua-2 BERT, 3–6× faster than v1 |
| Async | Python asyncio | asyncio.Queue for streaming pipeline |
| Config | YAML (PyYAML) | Stage swaps without code changes |
| Logging | structlog JSON | Per-request latency_ms, tokens_in, tokens_out, model_id |

## Testing Requirements

- ≥ 80% unit test coverage per stage
- Integration tests in `tests/test_pipeline.py` use the 10 audio fixtures in `tests/fixtures/`
- Compression tests must assert ROUGE-L ≥ 0.85 and token reduction ≥ 60%
- Latency tests are benchmarks (not assertions) — they live in `scripts/benchmark.py`

## What NOT to Do

- Do not mock the asyncio.Queue in integration tests — use real async iteration
- Do not add a GUI in v1 (CLI + Python library API is the explicit v1 surface)
- Do not implement real-time streaming to the cloud API (post-MVP)
- Do not add multi-speaker diarization (post-MVP)
- Do not add languages other than English (v1.0 scope)
- Do not fine-tune any model on user data

## Emit Layer

Supports two formats configured via `emit.format`:
- `openai`: OpenAI Chat Completions `messages: [{role, content}]`
- `anthropic`: Anthropic Messages API

Accepts any `base_url` for OpenAI-compatible endpoints (NVIDIA NIM, vLLM, Ollama).
Pipeline metadata (token savings, compression ratio) attached as system prompt header when `attach_metadata: true`.

## Owner

Aayush Pathak (aapathak@nvidia.com)
PRD: `../Voice_to_Prompt_Pipeline_PRD.docx` — Version 0.2, June 2026
