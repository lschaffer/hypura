# Hypura Memory Sizing & Context Window Guide

This guide details the relationship between **Model Weights**, **KV-Cache**, and **Unified Memory Allocation** on Apple Silicon (M-series) Macs running Hypura.

---

## 1. Unified RAM Architecture Overview

On macOS Apple Silicon Macs, the CPU and Metal GPU share a single pool of unified RAM. Memory consumption during inference is divided into:

1. **Model Weights:** The quantized layer tensors offloaded to the GPU/RAM (e.g. 12.3 GB for Gemma 4 19.9B/26B in Q4K).
2. **KV-Cache:** Attention key-value state allocated per token in the active context window.
3. **Compute Buffers & Metal Runtime:** Temporary scratch buffers for forward passes (~500 MB – 800 MB).
4. **macOS System & Background Apps:** Base OS reservation (~2.0 GB – 2.5 GB).

---

## 2. KV-Cache Sizing Formula

For a model with Grouped-Query Attention (GQA):

$$\text{KV Bytes per Token} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times \text{Bytes per Element}$$

For **Gemma 4 19.9B / 26B** ($N_{\text{layers}} = 30$, $N_{\text{kv\_heads}} = 8$, $D_{\text{head}} = 176$, FP16 $\text{bytes} = 2$):

$$\text{KV Bytes per Token} = 2 \times 30 \times 8 \times 176 \times 2 = 168{,}960\text{ bytes} \approx 165\text{ KB / token}$$

---

## 3. Context Window Memory Sizing (24 GB Unified RAM)

The table below illustrates memory allocations for **Gemma 4 19.9B/26B (Q4K, 12.3 GB weights)** on a **24 GB Unified RAM** Mac:

| Context Window ($n_{\text{ctx}}$) | KV-Cache Memory | Model + Compute Overhead | Total System + Hypura RAM | Free RAM Remaining | Hardware Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4k (4,096 tokens)** | ~0.68 GB | ~12.9 GB | **~15.4 GB** | **~8.6 GB Free** | 100% GPU / Fast |
| **8k (8,192 tokens)** | ~1.35 GB | ~12.9 GB | **~16.8 GB** | **~7.2 GB Free** | 100% GPU / Fast |
| **16k (16,384 tokens)** | ~2.70 GB | ~12.9 GB | **~18.1 GB** | **~5.9 GB Free** | 100% GPU / Fast |
| **32k (32,768 tokens)** | ~5.40 GB | ~12.9 GB | **~20.8 GB** | **~3.2 GB Free** | 100% GPU / Fast |

*(Includes ~2.5 GB baseline for macOS and running background processes)*

---

## 4. Serving Commands

### Recommended Default (16k Context)
```bash
hypura serve <MODEL_PATH> --host 0.0.0.0 --port 6000 --context 16384
```

### Extended Context (32k Context)
```bash
hypura serve <MODEL_PATH> --host 0.0.0.0 --port 6000 --context 32768
```

### Resource-Constrained / Low-Memory (4k Context)
```bash
hypura serve <MODEL_PATH> --host 0.0.0.0 --port 6000 --context 4096
```
