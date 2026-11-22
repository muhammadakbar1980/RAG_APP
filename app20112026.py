"""
PDF Chat - RAG application
Upload PDFs -> extract text -> split into token chunks -> create embeddings
-> store in FAISS -> ask questions -> Groq answers from the retrieved chunks.

Run locally:  streamlit run app.py
Secrets:      GROQ_API_KEY in Streamlit Cloud secrets (or type it in the sidebar)
"""

import io
import os

import numpy as np
import streamlit as st

st.set_page_config(page_title="PDF Chat (RAG)", page_icon="📚", layout="wide")

# ---------------------------------------------------------------- settings
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]   # free Groq models
EMBED_MODEL = "BAAI/bge-small-en-v1.5"                        # free, small, runs on CPU
MAX_FILE_MB = 20


# ---------------------------------------------------------------- helpers
def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("GROQ_API_KEY", "")


@st.cache_resource(show_spinner="Loading tokenizer...")
def load_tokenizer():
    """tiktoken tokenizer. Returns None if it cannot load (word-based fallback is used)."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading embedding model (first time takes ~30 seconds)...")
def load_embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=EMBED_MODEL)


# ---------------------------------------------------------------- STEP 1: extract text
def extract_pdf_pages(file_bytes, file_name):
    """Returns a list of {"source", "page", "text"} for each page with text."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", " ")
        text = " ".join(text.split())          # clean extra spaces/new lines
        if text:
            pages.append({"source": file_name, "page": number, "text": text})
    return pages


# ---------------------------------------------------------------- STEP 2: tokens + chunks
def split_into_chunks(pages, chunk_size, overlap, tokenizer):
    """Split each page into chunks of about chunk_size tokens with overlap."""
    chunks = []
    step = max(1, chunk_size - overlap)
    for page in pages:
        if tokenizer is not None:
            tokens = tokenizer.encode(page["text"])
            for start in range(0, len(tokens), step):
                piece = tokens[start:start + chunk_size]
                chunks.append({**page, "text": tokenizer.decode(piece), "tokens": len(piece)})
                if start + chunk_size >= len(tokens):
                    break
        else:  # fallback: 1 word is about 1.3 tokens
            words = page["text"].split()
            size = max(1, int(chunk_size / 1.3))
            word_step = max(1, int(step / 1.3))
            for start in range(0, len(words), word_step):
                piece = words[start:start + size]
                chunks.append({**page, "text": " ".join(piece), "tokens": int(len(piece) * 1.3)})
                if start + size >= len(words):
                    break
    return chunks


# ---------------------------------------------------------------- STEP 3: embeddings
def create_embeddings(texts, embedder):
    vectors = np.array(list(embedder.embed(texts)), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms          # normalised -> inner product = cosine similarity


# ---------------------------------------------------------------- STEP 4: FAISS store
def build_faiss_index(vectors):
    import faiss
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search(question, index, chunks, embedder, top_k):
    query_vector = create_embeddings([question], embedder)
    scores, ids = index.search(query_vector, min(top_k, len(chunks)))
    results = []
    for score, i in zip(scores[0], ids[0]):
        if i != -1:
            results.append({**chunks[i], "score": float(score)})
    return results


# ---------------------------------------------------------------- STEP 5: Groq answer
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the context
taken from the user's PDF documents.
Rules:
- If the answer is not in the context, say: "I could not find this in the uploaded documents."
- Do not make up information.
- Keep answers clear and well organised. Use bullet points when helpful.
- At the end, mention the sources you used as (file name, page number)."""


def ask_groq(question, context_chunks, history, api_key, model):
    from groq import Groq
    client = Groq(api_key=api_key)

    context = "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" for c in context_chunks
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-4:]:                          # short memory of the chat
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
    })

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.2,
    )
    return chat_completion.choices[0].message.content


# ---------------------------------------------------------------- session state
for key, default in {"index": None, "chunks": [], "messages": [], "stats": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = get_api_key()
    if api_key:
        st.success("Groq API key loaded")
    else:
        api_key = st.text_input("Groq API key", type="password",
                                help="Free key from console.groq.com/keys")
    model = st.selectbox("Groq model", GROQ_MODELS)

    st.subheader("Chunk settings")
    chunk_size = st.slider("Chunk size (tokens)", 200, 1000, 500, 50)
    overlap = st.slider("Overlap (tokens)", 0, 200, 50, 10)
    top_k = st.slider("Chunks used per answer", 2, 10, 4)

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("♻️ Remove documents"):
        st.session_state.update(index=None, chunks=[], messages=[], stats=None)
        st.rerun()
    st.caption("🔒 Files are processed in memory and not stored.")

# ---------------------------------------------------------------- upload + process
st.title("📚 Chat with your PDFs")
st.write("Upload PDF files, click **Process**, then ask questions about them.")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button("⚙️ Process documents", type="primary", disabled=not uploaded_files):
    all_pages, skipped = [], []
    with st.status("Processing documents...", expanded=True) as status:
        # Step 1: extract text
        st.write("1️⃣ Extracting text from PDF...")
        for file in uploaded_files:
            size_mb = file.size / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                skipped.append(f"{file.name} (larger than {MAX_FILE_MB} MB)")
                continue
            try:
                pages = extract_pdf_pages(file.getvalue(), file.name)
            except Exception as error:
                skipped.append(f"{file.name} (could not read: {error})")
                continue
            if not pages:
                skipped.append(f"{file.name} (no text found - may be a scanned PDF)")
            all_pages.extend(pages)

        for item in skipped:
            st.warning("Skipped: " + item)
        if not all_pages:
            status.update(label="No readable text found", state="error")
            st.stop()

        # Step 2: tokens + chunks
        st.write("2️⃣ Splitting text into token chunks...")
        if overlap >= chunk_size:
            st.error("Overlap must be smaller than chunk size.")
            st.stop()
        tokenizer = load_tokenizer()
        chunks = split_into_chunks(all_pages, chunk_size, overlap, tokenizer)

        # Step 3: embeddings
        st.write(f"3️⃣ Creating embeddings for {len(chunks)} chunks...")
        try:
            embedder = load_embedder()
            vectors = create_embeddings([c["text"] for c in chunks], embedder)
        except Exception as error:
            status.update(label="Embedding failed", state="error")
            st.error(f"Could not create embeddings: {error}")
            st.stop()

        # Step 4: FAISS
        st.write("4️⃣ Storing vectors in FAISS...")
        st.session_state.index = build_faiss_index(vectors)
        st.session_state.chunks = chunks
        st.session_state.messages = []
        st.session_state.stats = {
            "files": len({p["source"] for p in all_pages}),
            "pages": len(all_pages),
            "chunks": len(chunks),
            "tokens": sum(c["tokens"] for c in chunks),
            "dimension": int(vectors.shape[1]),
        }
        status.update(label="Documents ready ✅", state="complete", expanded=False)

# ---------------------------------------------------------------- stats
stats = st.session_state.stats
if stats:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Files", stats["files"])
    c2.metric("Pages", stats["pages"])
    c3.metric("Chunks", stats["chunks"])
    c4.metric("Tokens", f"{stats['tokens']:,}")
    c5.metric("Vector size", stats["dimension"])
    with st.expander("👀 Preview chunks"):
        for c in st.session_state.chunks[:5]:
            st.markdown(f"**{c['source']} - page {c['page']} ({c['tokens']} tokens)**")
            st.caption(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))

# ---------------------------------------------------------------- chat
st.divider()
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sources"):
                for s in message["sources"]:
                    st.markdown(f"**{s['source']} - page {s['page']}** (similarity {s['score']:.2f})")
                    st.caption(s["text"][:400] + "...")

question = st.chat_input("Ask a question about your documents",
                         disabled=st.session_state.index is None)
if st.session_state.index is None:
    st.info("Upload and process PDF files to start chatting.")

if question:
    if not api_key:
        st.error("Please add your Groq API key in the sidebar.")
        st.stop()

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and thinking..."):
            try:
                embedder = load_embedder()
                found = search(question, st.session_state.index,
                               st.session_state.chunks, embedder, top_k)
                answer = ask_groq(question, found, history, api_key, model)
            except Exception as error:
                found = []
                answer = f"⚠️ Something went wrong: {error}"
        st.markdown(answer)
        if found:
            with st.expander("📎 Sources"):
                for s in found:
                    st.markdown(f"**{s['source']} - page {s['page']}** (similarity {s['score']:.2f})")
                    st.caption(s["text"][:400] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": found})
