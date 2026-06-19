# Voice2Prompt — Architecture

## Streaming Pipeline Design

The key latency optimization in v0.2 is that Stage 3 does **not** wait for Stage 2 to finish. An `asyncio.Queue` connects them:

```
Stage 2 (formatter)                   Stage 3 (compressor)
──────────────────────────────────    ─────────────────────────────────
t=40ms:  starts generating            (idle)
t=165ms: emits sentence 1 → queue →  t=165ms: starts compressing sent. 1
t=300ms: emits sentence 2 → queue →  continues...
t=590ms: generation complete          t=710ms: final flush complete
```

This overlap saves ~120 ms vs. sequential execution — the primary mechanism for hitting < 1 s on the RTX 1000 Ada (50W).

### asyncio.Queue Sentinel

Stage 2 puts a `None` sentinel into the queue when generation is complete. Stage 3's consumer loop exits on `None` and flushes its final buffer. A 2 s timeout sentinel prevents deadlocks if Stage 2 crashes mid-stream.

---

## Stage 1 — Speech-to-Text

### Primary: Parakeet TDT 0.6B v2

- FastConformer + TDT decoder, single-pass up to 24 min
- RTFx ~900x on RTX 1000 Ada (50W): a 30 s clip transcribes in ~35 ms
- Word-level timestamps, auto punctuation and capitalization
- WER ~6.3% on diverse test sets
- License: CC BY 4.0

### Fallback: faster-whisper (Whisper Large V3 Turbo)

- CTranslate2 backend, INT8 quantization
- RTFx ~70x on RTX 1000 Ada → ~60–80 ms for 30 s clip (within 100 ms budget)
- Used when NeMo unavailable or non-English input detected
- License: MIT

### Filler Pre-Pass

Runs on CPU in < 5 ms after transcription. Reduces Stage 2 input by ~25%, saving ~60–90 ms of LLM generation on the Ada GPU.

Strips:
- Exact-match fillers: `um`, `uh`, `like` (adverbial), `you know`, `basically`, `literally`, `sort of`, `kind of`, `right`/`okay` at sentence boundaries
- False starts: repeated partial phrases within 5-word window
- STT punctuation artifacts and extra whitespace

Does **not** restructure content — that is Stage 2's responsibility.

---

## Stage 2 — LLM Formatter

### Model Selection (v0.2)

v0.1 used Llama 3.1 8B. On RTX 1000 Ada (8 GB, 50W), 8B achieves ~70–90 tok/s → ~1.2 s for a 100-token output. This exceeds the 550 ms budget.

v0.2 mandates a sub-4B model with Q4_K_M quantization and full GPU offload (`n_gpu_layers=-1`).

| Model | Params | Tok/s (Ada est.) | 100-tok latency | Notes |
|-------|--------|-----------------|-----------------|-------|
| **Phi-3.5 Mini Instruct** | 3.8B | ~180–240 | ~450 ms | Default |
| Gemma 2 2B Instruct | 2B | ~280–360 | ~300 ms | Fastest; less structured |
| Llama 3.2 3B Instruct | 3B | ~240–310 | ~350 ms | Strong instruction-following |
| Llama 3.1 8B Instruct | 8B | ~70–90 | ~1200 ms | Quality mode only; SLA not met |

### System Prompt

```
You are a prompt structuring assistant. Filler words are already removed.
Your task: group ideas under ## headings; express each as a concise bullet (-).
Rules:
- One bullet per idea. Max 15 words per bullet.
- Preserve all technical terms, numbers, names, constraints exactly.
- Do NOT add information absent from the transcript.
- Output ONLY the structured markdown. No preamble or commentary.
```

### Example

Input (after filler pre-pass):
> "I want to build an API that handles user auth. It should use JWT tokens. It needs to be fast — under 100ms. We're using Python, FastAPI, and Postgres."

Output:
```markdown
## Goal
- Build an API that handles user authentication.

## Technical Constraints
- Auth: JWT tokens.
- Response time: < 100 ms.
- Stack: Python, FastAPI, PostgreSQL.
```

---

## Stage 3 — Token Compressor

### LLMLingua-2

- BERT-large token classifier (not a generative model)
- 3–6× faster than LLMLingua-1 with equivalent faithfulness
- VRAM: ~1.5 GB
- Target: `rate=0.4` (retain 40% of tokens → ≥ 60% reduction)

### Retention Policy

| Content type | Retention |
|--------------|-----------|
| Structure markers (`##`, `-`, `**`) | 100% (force_tokens) |
| Instructions and constraints | 80–90% (conservative) |
| Descriptive context | 30–40% (aggressive) |
| Technical terms, numbers, proper nouns | 100% |

### Accuracy Target

ROUGE-L of compressed output vs. uncompressed formatted prompt ≥ 0.85.

---

## VRAM Budget (RTX 1000 Ada, 8 GB)

```
Parakeet 0.6B           ~0.8 GB  ████░░░░░░░░░░░░
Phi-3.5 Mini Q4_K_M     ~2.5 GB  ████████████░░░░
LLMLingua-2 BERT-large  ~1.5 GB  ███████░░░░░░░░░
OS + CUDA drivers       ~1.2 GB  ██████░░░░░░░░░░
─────────────────────────────────────────────────
Used                    ~6.0 GB
Headroom                ~2.0 GB
Total                    8.0 GB
```

All three models are loaded at startup and kept resident. There is no model swapping between requests.

---

## Emit Layer

Wraps the compressed prompt in the target API's message format:

**OpenAI Chat Completions:**
```json
{
  "messages": [
    {"role": "system", "content": "[voice2prompt] saved 247 tokens (63%)"},
    {"role": "user", "content": "<compressed prompt>"}
  ]
}
```

**Anthropic Messages API:**
```json
{
  "system": "[voice2prompt] saved 247 tokens (63%)",
  "messages": [{"role": "user", "content": "<compressed prompt>"}]
}
```

The metadata header is toggled by `emit.attach_metadata` in config. Any `base_url` is accepted for OpenAI-compatible endpoints (NVIDIA NIM, vLLM, Ollama).

---

## Device Selection

Auto-detected at startup in order: CUDA → MPS → CPU. Each stage logs a warning if falling back. CPU fallback P95 latency: ~5–7 s (graceful degradation, not a supported SLA).

---

## Hardware Tiers

| Tier | GPU | VRAM | P95 E2E |
|------|-----|------|---------|
| Primary | RTX 1000 Ada Laptop (50W) | 8 GB | ~850 ms |
| 1+ | RTX 3060/4060 Laptop (80–100W) | 8 GB | ~700 ms |
| 2 | RTX 3080/4070 Desktop | 10–12 GB | ~600 ms |
| 3 | Apple M3 Pro / M4 (MPS) | 18–36 GB unified | ~900 ms |
| Fallback | CPU only | — | ~5–7 s |
