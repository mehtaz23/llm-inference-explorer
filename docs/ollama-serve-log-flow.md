# Ollama Serve Log Flow

A walkthrough of what `ollama serve` logs mean on first run, using a real startup on an Apple M3 Pro as the reference trace.

---

## Phase 1 — Initialization

```
time=2024-... level=INFO source=routes.go msg="server config"
time=2024-... level=INFO source=server.go msg="starting ollama" version=0.22.1
```

Ollama generates an asymmetric key pair on first run. This is **not** SSH — the keys are used for Ollama's own API authentication and request signing mechanism. Communication itself is plain HTTP on localhost.

The server loads environment variables (`OLLAMA_HOST`, `OLLAMA_MODELS`, etc.) and begins listening on `127.0.0.1:11434`.

---

## Phase 2 — GPU Discovery (Apple Silicon)

```
level=INFO source=gpu.go msg="apple silicon" library=metal
level=INFO source=gpu.go msg="detected GPU" gpu=0 driver=Metal
  available=13.3 GiB
```

On Apple Silicon, Ollama uses Metal as its inference backend (via llama.cpp's Metal implementation). The reported VRAM is the portion of **unified memory** currently available — not a dedicated GPU memory chip.

The default context size (`num_ctx`) is set to **4096 tokens**. This is a hardcoded Ollama default, not a value derived from available VRAM. With an M3 Pro you have headroom to increase it significantly:

```bash
ollama run llama3.2 --option num_ctx 8192
```

---

## Phase 3 — Model Pull

```
pulling manifest
pulling <sha>... 100% ▕███████████████▏ 1.87 GiB
pulling <sha>... 100% ▕███████████████▏  ...
```

`ollama pull` downloads a model in content-addressed blobs (by SHA digest). The first blob is the quantized weights file — the largest download. The remaining small files (~126 MB total) are metadata: tokenizer config, chat templates, and the model manifest.

If you re-pull the same model, Ollama checks local blob hashes and skips anything already cached — only changed blobs are re-downloaded.

---

## Phase 4 — Model Loading

```
level=INFO source=ggml.go msg="ggml.metal: found device" name="Apple M3 Pro" ...
level=INFO source=server.go msg="loading model" model=llama3.2
  arch=llama quantization=Q4_K file_size=1.87 GiB
  layers.gpu=29
```

What's happening here:

| What | Detail |
|---|---|
| Format | GGUF (llama.cpp's model file format) |
| Quantization | Q4_K — 4-bit weights with K-quant grouping, a balance of speed and quality |
| Layers offloaded | 29 of 29 → all layers on GPU (Metal) |
| Model weights | ~1.9 GiB → GPU |
| KV cache | ~448 MiB → GPU (stores attention key/value pairs for the 4096-token context window) |
| Compute graph | ~256 MiB → GPU |
| Total allocation | ~2.6 GiB |

All 29 layers on GPU means no CPU fallback during the forward pass — inference runs entirely on Metal.

**Why 29 layers for a "3B" model?** Llama 3.2 3B has 28 transformer blocks. llama.cpp counts the output projection (lm_head) as an additional layer, giving 29 total.

---

## Phase 5 — Runner Start

```
level=INFO source=runner.go msg="llama runner started" duration=8.9s
```

The llama.cpp runner process is initialized. The 8–9 second startup time reflects Metal library initialization and shader compilation. This is a one-time cost per server start — the runner stays resident as long as the model is loaded.

---

## Phase 6 — First Inference

```
level=INFO source=server.go msg="request served" method=POST path=/api/generate
  status=200 duration=19.8s
```

The first inference call is slower than subsequent ones. Metal JIT-compiles compute shaders on the first forward pass, adding roughly 8–12 seconds of one-time overhead. From the second request onward, generation runs at full throughput — typically 30–60 tokens/sec for a 3B model on an M3 Pro.

---

## Quick Reference — Key Ollama Commands

```bash
ollama list               # show downloaded models
ollama pull llama3.2      # download a model
ollama rm llama3.2        # remove a model
ollama ps                 # show currently loaded models and VRAM usage
ollama run llama3.2       # interactive chat (starts serve if not running)
ollama serve              # start the API server manually
```

Models are stored in `~/.ollama/models/`. To remove everything:

```bash
rm -rf ~/.ollama/models
```
