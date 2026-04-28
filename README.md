# LLM Inference Explorer

A minimal Streamlit app for running local LLMs via [Ollama](https://ollama.com) and observing inference mechanics in real time — prefill, decode loop, token streaming, and performance metrics.

## What it does

- Connects to a local Ollama instance and lists your downloaded models
- Streams generated tokens one at a time as they arrive via SSE
- Displays **tokens/sec**, **time to first token**, and **total duration** after each generation
- Sidebar explains the prefill → decode → SSE pipeline as it happens

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com) running locally with at least one model pulled

## Quickstart

```bash
# Pull a model if you haven't already
ollama pull llama3.2

# Install dependencies and start the app
make dev
```

The app will be available at `http://localhost:8501`.

## Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| HTTP client | httpx (streaming) |
| Inference runtime | Ollama → llama.cpp |
| Dependency management | uv |

## Docker (CPU-only)

A `compose.yml` is included to run Ollama in a container. Note that on macOS this disables Metal GPU acceleration — use `ollama serve` natively for M-series GPU inference.

```bash
docker compose up
```

## Project structure

```
app/
  streamlit_app.py   # UI and streaming display logic
  ollama_client.py   # Ollama REST API wrapper (list models, streaming generate)
  benchmark.py       # (in progress)
compose.yml          # Ollama container (CPU-only)
Makefile             # `make dev` to run the app
pyproject.toml       # Python project config and dependencies
```
