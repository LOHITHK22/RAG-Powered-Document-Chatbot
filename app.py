import os
import numpy as np
import streamlit as st

from utils import extract_text_from_pdf, chunk_text
from embedder import embed_chunks, save_faiss_index
from retriever import retrieve_top_chunks  # (query, k=3, index_path="index/faiss.index")

# ---------- Page setup ----------
st.set_page_config(
    page_title="RAG Doc Assistant",
    page_icon=None,  # keep simple; emoji can be flaky on some setups
    layout="wide",
)

# ---------- Styles (dark-friendly) ----------
st.markdown("""
<style>
/* Layout width */
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

/* Headings */
h1, h2, h3 { font-weight: 700; }

/* Cards for chunks — adapt to light/dark */
:root {
  --card-bg: #fafafa;
  --card-border: #e6e6e6;
  --card-text: #0f1115;
  --badge-bg: #eef;
  --badge-border: #ccd;
}
@media (prefers-color-scheme: dark) {
  :root {
    --card-bg: #14161b;
    --card-border: #2a2e36;
    --card-text: #e6e8ee;
    --badge-bg: #28324a;
    --badge-border: #3a4a6a;
  }
}
html[data-theme="dark"] :root {
  --card-bg: #14161b;
  --card-border: #2a2e36;
  --card-text: #e6e8ee;
  --badge-bg: #28324a;
  --badge-border: #3a4a6a;
}

/* Chunk card */
.chunk-card {
  border: 1px solid var(--card-border);
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 10px;
  background: var(--card-bg);
  box-shadow: 0 1px 2px rgba(0,0,0,0.08);
  color: var(--card-text);
  overflow-x: auto;
  white-space: pre-wrap;
}
.score-badge {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--badge-bg);
  border: 1px solid var(--badge-border);
  display: inline-block;
  margin-left: 8px;
}

/* Expander content */
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] { color: var(--card-text); }
[data-testid="stExpander"] .streamlit-expanderContent { background: transparent; }

/* Code blocks */
pre, code, kbd, samp {
  background: transparent !important;
  color: var(--card-text) !important;
}

/* Divider line inside cards */
hr { margin: 0.5rem 0 0.75rem 0; border: none; border-top: 1px solid var(--card-border); }

/* Hero title + subtitle */
.hero-title {
  font-size: 2.6rem;
  font-weight: 800;
  text-align: center;
  background: linear-gradient(90deg, #7aa2f7, #c084fc);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: 0 0 14px rgba(122,162,247,0.35);
  margin: 0 0 0.35rem 0;
}
.hero-subtitle {
  text-align: center;
  font-size: 0.98rem;
  color: #9ca3af;
  margin: 0 0 1.6rem 0;
}
</style>

<h1 class="hero-title">RAG-Powered Document Chatbot</h1>
<p class="hero-subtitle">Upload your PDFs, ask questions, get grounded answers with citations.</p>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("LLM model (Ollama)", ["mistral", "llama3"], index=0)
    top_k = st.slider("Results (k)", min_value=1, max_value=8, value=3)
    st.caption("Lower FAISS distance = more similar.")

    st.divider()
    st.header("📦 Index")
    st.caption("Upload one or more PDFs. New uploads overwrite the current FAISS index.")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    build_index = st.button("Build / Rebuild Index", type="primary", use_container_width=True)
    clear_chat = st.button("Clear Chat History", use_container_width=True)

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if clear_chat:
    st.session_state.messages = []
    st.toast("Chat cleared", icon="🧹")

# ---------- Index building ----------
def process_and_index(files):
    all_chunks = []
    for f in files:
        save_path = os.path.join("data", "uploaded_files", f.name)
        with open(save_path, "wb") as out:
            out.write(f.read())

        text = extract_text_from_pdf(save_path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    vectors = embed_chunks(all_chunks)
    if vectors is None or len(vectors) == 0:
        st.error("Failed to generate embeddings. Are the PDFs scanned images with no text?")
        return False, 0, (0,)

    save_faiss_index(np.array(vectors), all_chunks)
    return True, len(all_chunks), np.array(vectors).shape

if build_index:
    if not uploaded_files:
        st.warning("Upload at least one PDF first.")
    else:
        with st.status("Indexing documents…", expanded=True) as status:
            ok, chunk_count, vec_shape = process_and_index(uploaded_files)
            if ok:
                st.write(f"✅ Chunks created: **{chunk_count}**")
                st.write(f"✅ Vector shape: **{vec_shape}**")
                status.update(label="Index ready", state="complete")
                st.toast("Index built successfully", icon="✅")
            else:
                status.update(label="Index failed", state="error")

st.markdown("### 💬 Ask a question")
st.caption("Tip: build the index first. Then ask anything grounded in your docs.")

# ---------- Chat UI ----------
user_query = st.chat_input("Type your question…")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Handle latest user message
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    q = st.session_state.messages[-1]["content"]

    # Retrieve
    try:
        results = retrieve_top_chunks(q, k=top_k)
        context = "\n\n".join([r["chunk"] for r in results])
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Retrieval failed. Make sure you built the index. Details: {e}")
        st.stop()

    # Generate (Ollama)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            import requests
            prompt = f"""You are a helpful assistant. Use only the context to answer.
If the answer is not in the context, say you don’t have enough information.

Context:
{context}

Question: {q}
Answer:"""
            try:
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": False},
                    timeout=120
                )
                resp.raise_for_status()
                answer = resp.json().get("response", "").strip()
            except Exception as e:
                answer = f"LLM error. Is Ollama running and the `{model_name}` model pulled? Details: {e}"

        st.markdown(answer)

        # Sources
        with st.expander("📚 Sources & Scores"):
            if results:
                for i, r in enumerate(results, 1):
                    st.markdown(
                        f"""
<div class="chunk-card">
  <b>Chunk {i}</b>
  <span class="score-badge">distance: {r['score']:.2f}</span>
  <hr/>
  <div>{r['chunk']}</div>
</div>
""",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No chunks returned by retriever.")
