import streamlit as st
from ollama_client import list_models, generate_streaming

st.set_page_config(page_title="LLM Inference Explorer", layout="wide")
st.title("LLM Inference Explorer")
st.caption("Local inference via Ollama (llama.cpp under the hood)")

# ── Sidebar: controls ──────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    try:
        models = list_models()
    except Exception:
        st.error("Ollama not reachable — is `ollama serve` running?")
        st.stop()

    model = st.selectbox("Model", models)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    max_tokens = st.slider("Max tokens", 64, 2048, 512, 64)

    st.divider()
    st.markdown("""
    **What's happening under the hood**

    1. Your prompt hits Ollama's REST API
    2. Ollama passes it to llama.cpp
    3. llama.cpp runs the **prefill** (processes all your tokens at once)
    4. Then the **decode loop** starts — one token sampled at a time
    5. Each token streams back here via SSE
    """)

# ── Main: prompt + output ─────────────────────────────────────────
prompt = st.text_area(
    "Prompt", "Explain what a KV cache is in one paragraph.", height=120
)

if st.button("Generate", type="primary"):
    output_box = st.empty()
    stats_box = st.empty()
    full_text = ""

    with st.spinner("Prefilling prompt..."):
        gen = generate_streaming(model, prompt, temperature, max_tokens)
        # grab first token (marks end of prefill / start of decode)
        first = next(gen)

    token, done, stats = first
    full_text += token

    # stream the rest
    for token, done, stats in gen:
        full_text += token
        output_box.markdown(full_text + "▌")
        if done:
            break

    output_box.markdown(full_text)

    if stats:
        col1, col2, col3 = stats_box.columns(3)
        col1.metric("Tokens / sec", stats["tokens_per_second"])
        col2.metric("Time to first token", f"{stats['time_to_first_token_ms']} ms")
        col3.metric("Total time", f"{stats['total_duration_ms']} ms")
