# Real-World Agent Tests & Demos

This document records real-world agentic benchmarks and end-to-end multi-step tool-calling workflows evaluated on **Hypura** running on Apple Silicon.

---

## 🎬 Video Demonstrations

### 1. CumulusAI + Hypura: Gemma 4 26B Checks Datalogger Flash Space
* **Model:** `gemma4-26b-it-text` (Gemma 4 architecture, Gated Delta Net)
* **Application:** [CumulusAI](https://thiescloud.com) connected to Hypura on Apple Silicon Mac mini (M4 Pro 24GB Unified RAM).
* **Task:** Autonomous diagnostic check of remote meteorological station datalogger flash storage and telemetry health via cloud APIs.
* 🎥 **Watch on YouTube:** [https://youtu.be/6BRQYrkONqg](https://youtu.be/6BRQYrkONqg)

---

### 2. CumulusAI + Hypura: Run an Oversized Qwen 3.8 27B on Mac mini M4 Pro – Stations, Reasoning & Charts
* **Model:** `qwen3.8-27b`
* **Application:** [CumulusAI](https://thiescloud.com) connected to Hypura.
* **Task:** Multi-turn query requesting online weather stations within a specific radius, filtering by temperature ranges, executing multi-step reasoning, and generating formatted markdown tables and chart summaries.
* 🎥 **Watch on YouTube:** [https://youtu.be/brzBlL2LutQ](https://youtu.be/brzBlL2LutQ)

---

### 3. Tealkit + Hypura: Cross-Platform Native Agent Workflow
* **Model:** `devstral-small-2:24b` (Mistral 3 architecture) with 12k context window.
* **Application:** [Tealkit](https://github.com/lschaffer/tealkit) running on Windows 11 connected over local LAN to Hypura running on macOS.
* **Task:** End-to-end multi-step agent reasoning and local tool calling.
* 🎥 **Watch on YouTube:** [https://youtu.be/i28xrFum3KM](https://youtu.be/i28xrFum3KM)

---

## 🎯 Evaluated Models for Practical Agentic Workflows

When deploying real-world agentic apps (e.g. sensor data analysis from cloud APIs), models must balance **accurate tool calling**, **multi-step reasoning**, and **interactive latency** on unified memory hardware.

| Model | Parameters | Architecture | Latency / Behavior on 24GB Mac mini | Recommended Use Case |
|---|---|---|---|---|
| **Qwen 3.8 / 2.5** | 27B | ChatML / XML Tools | Fast prompt evaluation, reliable parameter extraction | Multi-station telemetry queries, complex data formatting |
| **Gemma 4** | 26B | Gated Delta Net / Hybrid SSM | Low per-token latency, robust tool syntax compliance | Real-time diagnostic checks, hardware status monitoring |
| **Mistral Small** | 24B | `[INST]` / `[AVAILABLE_TOOLS]` | High throughput, concise structured answers | Fast single/multi-tool API orchestration |
| **Granite 4.1 / Gemma 4** | 30B / 31B | Dense Transformer / SSM | Strong reasoning, but higher latency on 24GB memory | Deep analytical reasoning (non-interactive batch workflows) |
