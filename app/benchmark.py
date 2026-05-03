import statistics

from ollama_client import generate_streaming

DEFAULT_PROMPT = (
    "Explain what a KV cache is and why it matters for LLM inference. "
    "Be concise but technically precise."
)


def run_benchmark(
    model: str,
    prompt: str = DEFAULT_PROMPT,
    runs: int = 5,
    temperature: float = 0.0,  # 0 = deterministic, fairer comparison
    max_tokens: int = 256,
) -> dict:
    results = []

    for i in range(runs):
        full_text = ""
        stats = {}
        for token, done, s in generate_streaming(
            model, prompt, temperature, max_tokens
        ):
            full_text += token
            if done:
                stats = s

        if stats:
            results.append(stats)

    if not results:
        return {}

    return {
        "model": model,
        "runs": runs,
        "avg_tokens_per_second": round(
            statistics.mean(r["tokens_per_second"] for r in results), 1
        ),
        "avg_ttft_ms": round(
            statistics.mean(r["time_to_first_token_ms"] for r in results)
        ),
        "avg_total_ms": round(statistics.mean(r["total_duration_ms"] for r in results)),
        "min_tps": round(min(r["tokens_per_second"] for r in results), 1),
        "max_tps": round(max(r["tokens_per_second"] for r in results), 1),
        "raw": results,
    }


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"
    print(f"Benchmarking {model} over 5 runs...")
    result = run_benchmark(model)
    print(f"  avg tokens/sec : {result['avg_tokens_per_second']}")
    print(f"  avg TTFT       : {result['avg_ttft_ms']} ms")
    print(f"  avg total time : {result['avg_total_ms']} ms")
    print(f"  tps range      : {result['min_tps']} – {result['max_tps']}")
