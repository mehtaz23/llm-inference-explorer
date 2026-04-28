import httpx
import json
import time
from dataclasses import dataclass

OLLAMA_BASE = "http://localhost:11434"


@dataclass
class GenerateResult:
    text: str
    tokens_per_second: float
    total_duration_ms: float
    time_to_first_token_ms: float


def list_models() -> list[str]:
    r = httpx.get(f"{OLLAMA_BASE}/api/tags")
    r.raise_for_status()
    return [m["name"] for m in r.json()["models"]]


def generate_streaming(
    model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 512
):
    """Yields (token_text, is_done, stats_dict) tuples."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    t_start = time.perf_counter()
    first_token = True
    ttft_ms = 0.0

    with httpx.stream(
        "POST", f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("response", "")

            if first_token and token:
                ttft_ms = (time.perf_counter() - t_start) * 1000
                first_token = False

            if data.get("done"):
                # Ollama returns eval_count (tokens generated) and eval_duration (ns)
                eval_count = data.get("eval_count", 0)
                eval_duration_ns = data.get("eval_duration", 1)
                tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0
                total_ms = data.get("total_duration", 0) / 1e6
                yield (
                    token,
                    True,
                    {
                        "tokens_per_second": round(tps, 1),
                        "total_duration_ms": round(total_ms),
                        "time_to_first_token_ms": round(ttft_ms),
                        "tokens_generated": eval_count,
                    },
                )
                return

            yield token, False, {}
