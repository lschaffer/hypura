# Changelog

All notable changes to the **Hypura** project are documented in this file.

---

## [0.2.2] - 2026-08-20

### ✨ Features & Bug Fixes

#### 1. Sampler State Synchronization (`llama_sampler_accept`)
* Integrated `llama_sampler_accept` into `LlamaSampler::sample()` to ensure newly sampled tokens are registered in the sampler chain.
* **Fixes Repetition Loops:** Resolves infinite word repeating loops (`type, type, type...`) during tool call generation on models like Qwen 2.5 / 3.8.

#### 2. Max 90% Unified Memory Limit Guard
* Added a hard 90% physical system memory limit (`(hw.memory.total_bytes * 0.90)`) for Metal GPU offloading (`compute_gpu_budget`) and RAM keep-resident mode (`load_model`).
* Prevents total memory exhaustion and system instability on Apple Silicon Macs (e.g. 24GB M-series).

#### 3. Expanded Native Tool Call Format Support (Mistral 24B & Qwen)
* Added support for `[TOOL_CALLS] [...]` array structures (standard Mistral v3 / Mistral 24B tool calling format) in `parse_tool_calls`.
* Reverted experimental `muse-glimmer` additions to maintain clean macOS compatibility.

---

## [0.2.1] - 2026-08-16

### 🚀 Major Highlights & Real-World Agent Demo
* **Demonstration & Verification:** Tested and validated with the **[Tealkit Agentic App](https://github.com/lschaffer/tealkit)** (Windows native application running on Windows 11) connected over local LAN to the **Hypura engine running on a Mac mini**.
* **Hardware & Engine:** Ran 100% locally on **Apple Silicon Mac mini M4 Pro (24 GB Unified RAM)** with **`devstral-small-2:24b`** (Mistral 3 architecture) with a **12k context window** under full Metal GPU acceleration.
* 🎥 **Video Demo:** Watch the cross-platform agentic workflow in action on YouTube: **[https://youtu.be/i28xrFum3KM](https://youtu.be/i28xrFum3KM)**

---

### ✨ Features & Enhancements

#### 1. Dynamic On-Demand Multi-Model Serving (`hypura serve`)
* `hypura serve` can now start as a dynamic background daemon without specifying a model upfront.
* Models are loaded on-demand when client requests arrive at `/api/generate` or `/api/chat`.
* Automatically hot-swaps models in GPU/Unified memory when the client selects a different model.
* Caches active models for instant response times on subsequent requests.

#### 2. Zero-Copy Ollama Model Sharing (`src/server/registry.rs`)
* Hypura automatically discovers all models installed in Ollama (`~/.ollama/models` or `$OLLAMA_MODELS`) by parsing manifests and mapping directly to `blobs/sha256-*` GGUF files.
* **Zero disk duplication:** Access all your downloaded Ollama models seamlessly without copying or manual symlinking.

#### 3. Native Ollama & OpenAI Tool Calling Support (`src/server/chat.rs`)
* Added full native tool calling parser and JSON schema prompt formatting.
* Supports Gemma 4 channel syntax (`<|tool_call>call:func{...}<tool_call|>`), standard ChatML JSON/XML tool calls, and OpenAI function calling structures.
* Automatically filters internal thought tokens (`<|channel>thought...<channel|>`) to ensure clean responses for client agents (Cline, Roo Code, Tealkit, Open WebUI).

#### 4. Dynamic Context Sizing & Auto-Expanding KV Cache
* Added `num_ctx` support in `GenerateOptions` and client request payloads.
* Context size is dynamically tokenized and allocated per-request (`effective_ctx = (prompt_len + max_tokens).max(config.n_ctx)`).
* Prevents `KV cache full / failed to find a memory slot` errors when prompts contain large tool schemas or long multi-turn conversations.

#### 5. New CLI Commands & Monitoring APIs
* **`hypura list` (`src/cli/list.rs`):** Displays all available local and Ollama models with parameter counts, quantization levels, architectures, and sizes.
* **`hypura ps` (`src/cli/ps.rs`):** CLI tool to inspect active models loaded in memory, context sizes, and GPU offload status.
* **`GET /api/ps` (`src/server/routes.rs`):** Standard Ollama-compatible process status endpoint.

#### 6. Multimodal GGUF LLM Loading Support
* Patched `llama.cpp` loader (`llama_model_loader::done_getting_tensors`) to tolerate unmapped multimodal vision/audio encoder tensors.
* Enables single-file loading of unified multimodal GGUF models (e.g. Mistral-Small 3.2 24B / Pixtral, Gemma 4) directly for text generation and tool calling.

#### 7. Tailscale Funnel & Public HTTPS Script (`start_hypura_funnel.sh`)
* Added automated startup script with background Tailscale Funnel configuration (`tailscale funnel --bg --yes 6000`).
* Displays public HTTPS endpoints (`https://<mac>.<tailnet>.ts.net`) for remote agent access over SSL.

---

### 📚 Documentation
* Added **[docs/MEMORY_SIZING.md](docs/MEMORY_SIZING.md)**: Detailed memory breakdown tables, KV cache calculation formulas, and context window sizing for Apple Silicon Unified RAM architectures.
