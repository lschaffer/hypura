# Modern 14B–30B Model Optimization & OpenAI Interface Integration Guide

## 1. Executive Summary

This document details:
1. **Target Architecture & Hardware**: Performance analysis and placement strategies for 14B–30B class models on Apple Silicon Mac mini Pro (M4 Pro / M5 / M6 with 24GB and 32GB Unified Memory).
2. **2026 Modern Model Landscape (14B–30B)**: Evaluation of dense, hybrid SSM/linear attention, and fine-grained Mixture-of-Experts (MoE) models for agentic reasoning and tool calling.
3. **OpenAI Compatibility Layer**: Architectural design and implementation specifications for exposing standard OpenAI endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) with Server-Sent Events (SSE) streaming alongside existing Ollama endpoints.

---

## 2. Hardware Profiles: Mac mini Pro 24GB vs 32GB

On Apple Silicon with Unified Memory Architecture (UMA), both CPU and GPU share the same memory pool. However, macOS imposes strict Metal working set limits (`recommendedMaxWorkingSetSize`):
* **24GB Mac mini (M4 Pro / M5 Pro / M6 Pro)**:
  * OS & Desktop Overhead: ~2.0 GB - 3.0 GB
  * Metal Working Set Limit: ~17.8 GB safe GPU buffer
  * Fast NVMe Read: ~5.0 GB/s - 7.5 GB/s sequential
* **32GB Mac mini (M4 Pro / M5 Pro / M6 Pro / M2 Max / M3 Max)**:
  * OS & Desktop Overhead: ~2.5 GB - 3.5 GB
  * Metal Working Set Limit: ~24.5 GB - 26.0 GB safe GPU buffer
  * Fast NVMe Read: ~5.0 GB/s - 7.5 GB/s sequential

### The 14B–30B "Memory Wall" Challenge
* A **14B Q4_K_M** model weighs ~8.4 GB. It fits entirely in GPU memory on both 24GB and 32GB machines, achieving 20–35+ tok/s.
* A **24B–27B Q4_K_M** model weighs ~14.0–16.5 GB. On a 24GB Mac, adding an 8k–16k context window pushes total allocation beyond 18 GB, threatening OOM crash under vanilla llama.cpp. Hypura's storage-tier placement solves this by offloading non-critical tensors.
* A **30B–32B Q4_K_M** model weighs ~18.5–22.5 GB. On a 24GB Mac, this strictly requires Hypura's **Dense FFN streaming** or **Sparse MoE / Expert Streaming**. On a 32GB Mac, it fits with a compact context, but large context windows require Hypura's dynamic tiering.

---

## 3. Evaluated 2026 Models in the 14B–30B Range

| Model | Architecture | Parameter Count | Quantization & Size | Best Mode on 24GB Mac | Best Mode on 32GB Mac | Expected tok/s & Strengths |
|---|---|---|---|---|---|---|
| **Qwen 2.5 / 3.x 14B** | Dense Transformer | 14.7B | Q4_K_M (~8.5 GB) | Full-Resident (GPU) | Full-Resident (GPU) | **~25–38 tok/s**<br>Extremely fast, high instruction following, agent tool calling |
| **Mistral Small 3 / Devstral 24B** | Dense Transformer | 23.6B | Q4_K_M (~14.2 GB) | Hybrid Resident / FFN Streaming with >8k ctx | Full-Resident (GPU) | **~18–24 tok/s (32GB)**<br>**~8–14 tok/s (24GB)**<br>Excellent native `[TOOL_CALLS]`, coding, low latency |
| **Gemma 4 26B / 31B** | Gated Delta Net / Hybrid SSM | 26.2B / 31B | Q4_K_M (~15.8 GB / ~19.1 GB) | Sparse State GPU + FFN streaming | Full-Resident (GPU) for 26B | **~14–22 tok/s**<br>Linear attention scaling, minimal KV cache footprint, strong reasoning |
| **Qwen 2.5-Coder / 3.x 32B** | Dense Transformer | 32.5B | Q4_K_M (~19.8 GB) | Dense FFN Streaming | Hybrid / Full-Resident (compact ctx) | **~3–6 tok/s (24GB streaming)**<br>**~12–16 tok/s (32GB)**<br>State-of-the-art coding and agentic planning |
| **Granite 3.x / 4.x 20B–30B** | Dense Transformer | 21B / 30B | Q4_K_M (~12.8 GB / ~18.2 GB) | Full-Resident (20B) / FFN Streaming (30B) | Full-Resident (GPU) | **~15–25 tok/s**<br>Enterprise safety, strict JSON/tool formatting |
| **Phi-3.5-MoE (16x3.8B)** | Sparse MoE (2/16 active) | 41.9B total (~6.6B active) | Q4_K_M (~22.5 GB) | Expert-Streaming (NVMe + Neuron Cache) | Expert-Streaming | **~12–18 tok/s**<br>Only 6.6B active per token. 98%+ neuron cache hit rate delivers near-resident speeds |
| **DeepSeek-MoE / Distill 16B–32B** | Fine-Grained MoE + Shared | 16B–32B (~2.5–5B active) | Q4_K_M (~10–19 GB) | Shared on GPU, Routed on NVMe | Shared on GPU, Routed on NVMe | **~16–26 tok/s**<br>Granular expert routing maximizes cache hit rate |

---

## 4. Architectural Improvements for 14B–30B Support

### A. Dynamic FFN Tier Splitting
Currently, Hypura either puts all FFN tensors on NVMe or keeps them all in RAM/GPU.
* For 24B–32B models on 24GB machines, only **2–4 GB** exceeds the working set limit.
* **Solution**: Implement progressive layer splitting:
  - First $N$ layers and last $M$ layers remain GPU-resident.
  - Only middle layers (or layers with high computation-to-I/O overlap) stream FFN weights through the pool buffer.
  - This increases generation throughput from ~3 tok/s to ~10–14 tok/s.

### B. Fine-Grained MoE Stride & Shared Expert Handling
* Support architectures where shared experts (`ffn_gate_shexp`) are always GPU-resident, while granular routed experts (`ffn_gate_exps`) stream from NVMe.
* Dynamic stride alignment for fused expert tensors with arbitrary expert counts (e.g. 16, 64, or 128 experts).

### C. Universal Chat Template Engine
* Migrate from hardcoded prompt string formatting in `src/server/chat.rs` to GGUF-embedded Jinja2 chat template parsing (`tokenizer.chat_template`).
* Automatically supports any prompt convention (ChatML, Harmony, Gemma turn tokens, Tekken, Llama-3 headers) without code modification.

---

## 5. OpenAI Interface Support (`/v1/*`)

### Why Support OpenAI Interface?
1. **Tool Ecosystem**: Direct compatibility with tools like Cursor, Cline, Continue.dev, LangChain, LlamaIndex, LiteLLM, AutoGen, and the official OpenAI Python/Node SDKs.
2. **Standard SSE Streaming**: Server-Sent Events (`text/event-stream`) standard with `data: {...}` chunks and `data: [DONE]`.
3. **First-Class Function Calling**: Native `tool_calls` structured schemas without vendor-specific wrappers.

### Implemented Endpoints
* `GET /v1/models`: Returns list of available models formatted according to OpenAI Model object specifications.
* `POST /v1/chat/completions`: Chat completions supporting both synchronous JSON responses and SSE streaming (`stream: true`), message histories (`system`, `user`, `assistant`, `tool`), and tool calling.
* `POST /v1/completions`: Raw text prompt completion.
