import os
import sys
import re
import tempfile
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi

# Add src directory to path so we can import existing modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator import (
    _load_env,
    load_llm,
    generate_answer,
    chunk_text,
    classify_and_answer,
    hybrid_retrieval,
    score_confidence,
    extract_name,
    extract_age,
    extract_gender,
    extract_diagnosis,
    extract_medications,
    extract_symptoms,
    extract_allergies,
    extract_dob,
    extract_visit_date,
    FIELD_REGISTRY,
)
from ingest import extract_text_from_pdf
from embedding import load_embedding_model, generate_embeddings

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .patient-card {
        background: #f8faff;
        border: 1px solid #d0e2ff;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .patient-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.5rem;
    }
    .field-row {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.3rem;
        font-size: 0.9rem;
    }
    .field-label {
        font-weight: 600;
        color: #444;
        min-width: 110px;
    }
    .field-value {
        color: #222;
    }
    .conf-high   { color: #1e8a3e; font-weight: 600; }
    .conf-medium { color: #b86e00; font-weight: 600; }
    .conf-low    { color: #c0392b; font-weight: 600; }
    .source-badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .stChatMessage { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# =================================================
# SESSION STATE INIT
# =================================================
def init_state():
    defaults = {
        "documents":    [],
        "all_chunks":   [],
        "embeddings":   None,
        "bm25":         None,
        "embed_model":  None,
        "llm_client":   None,
        "chat_history": [],
        "indexed":      False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# =================================================
# CACHED RESOURCE LOADERS
# =================================================
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embed_model():
    return load_embedding_model()


@st.cache_resource(show_spinner="Connecting to Groq...")
def get_llm_client():
    _load_env()
    return load_llm()


# =================================================
# DOCUMENT PROCESSING
# =================================================
def process_uploaded_files(uploaded_files) -> tuple[list, list]:
    raw_documents, all_chunks = [], []
    seen_names = set()
    documents = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for uf in uploaded_files:
            path = os.path.join(tmpdir, uf.name)
            with open(path, "wb") as f:
                f.write(uf.read())
            raw_text = extract_text_from_pdf(path)
            raw_documents.append({"doc_id": uf.name, "raw_text": raw_text})
            for chunk in chunk_text(raw_text):
                all_chunks.append({"doc_id": uf.name, "content": chunk})

    for doc in raw_documents:
        name = extract_name(doc["raw_text"])
        key  = name.strip().lower() if name else doc["doc_id"]
        if key not in seen_names:
            seen_names.add(key)
            documents.append(doc)

    return documents, all_chunks


def build_index(all_chunks, embed_model):
    texts = [c["content"] for c in all_chunks]
    embeddings = generate_embeddings(embed_model, texts)
    tokenized  = [re.findall(r"\w+", t.lower()) for t in texts]
    bm25       = BM25Okapi(tokenized)
    return embeddings, bm25


# =================================================
# PATIENT REPORT VIEW
# =================================================
def render_patient_card(doc: dict):
    t     = doc["raw_text"]
    name  = extract_name(t)       or "Unknown"
    age   = extract_age(t)        or "N/A"
    gender= extract_gender(t)     or "N/A"
    dob   = extract_dob(t)        or "N/A"
    vdate = extract_visit_date(t) or "N/A"
    diag  = extract_diagnosis(t)  or "N/A"
    meds  = extract_medications(t)
    syms  = extract_symptoms(t)
    allg  = extract_allergies(t)

    st.markdown(f"""
    <div class="patient-card">
        <div class="patient-name">👤 {name}</div>
        <div class="field-row"><span class="field-label">Age</span><span class="field-value">{age}</span></div>
        <div class="field-row"><span class="field-label">Gender</span><span class="field-value">{gender}</span></div>
        <div class="field-row"><span class="field-label">DOB</span><span class="field-value">{dob}</span></div>
        <div class="field-row"><span class="field-label">Visit Date</span><span class="field-value">{vdate}</span></div>
        <div class="field-row"><span class="field-label">Diagnosis</span><span class="field-value">{diag}</span></div>
        <div class="field-row"><span class="field-label">Medications</span><span class="field-value">{', '.join(meds) if meds else 'N/A'}</span></div>
        <div class="field-row"><span class="field-label">Symptoms</span><span class="field-value">{', '.join(syms) if syms else 'N/A'}</span></div>
        <div class="field-row"><span class="field-label">Allergies</span><span class="field-value">{', '.join(allg) if allg else 'N/A'}</span></div>
        <div class="field-row"><span class="field-label">File</span><span class="field-value">{doc['doc_id']}</span></div>
    </div>
    """, unsafe_allow_html=True)


# =================================================
# CONFIDENCE DISPLAY
# =================================================
def render_confidence(conf: dict):
    label  = conf["label"]
    score  = conf["score"]
    css    = {"HIGH": "conf-high", "MEDIUM": "conf-medium", "LOW": "conf-low"}[label]
    filled = int(score / 5)
    bar    = "█" * filled + "░" * (20 - filled)

    st.markdown(f"""
    <div style="background:#f5f5f5; border-radius:8px; padding:0.7rem 1rem; margin-top:0.5rem; font-size:0.85rem;">
        <span class="{css}">Confidence: {score}% ({label})</span>
        &nbsp; <code style="font-size:0.8rem;">[{bar}]</code><br>
        <span style="color:#666;">Retrieval: {conf['retrieval']}% &nbsp;|&nbsp;
        Coverage: {conf['coverage']}% &nbsp;|&nbsp;
        Quality: {conf['quality']}%</span>
    </div>
    """, unsafe_allow_html=True)


# =================================================
# SIDEBAR
# =================================================
with st.sidebar:
    st.markdown("## 🏥 Medical RAG")
    st.markdown("---")

    st.markdown("### 📂 Upload Patient PDFs")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("🔄 Index Documents", use_container_width=True, type="primary"):
            with st.spinner("Processing PDFs..."):
                embed_model = get_embed_model()
                docs, chunks = process_uploaded_files(uploaded_files)
                embeddings, bm25 = build_index(chunks, embed_model)

                st.session_state.documents   = docs
                st.session_state.all_chunks  = chunks
                st.session_state.embeddings  = embeddings
                st.session_state.bm25        = bm25
                st.session_state.embed_model = embed_model
                st.session_state.llm_client  = get_llm_client()
                st.session_state.indexed     = True
                st.session_state.chat_history = []

            st.success(f"✅ Indexed {len(docs)} patient(s) from {len(uploaded_files)} file(s)")

    if st.session_state.indexed:
        st.markdown("---")
        st.markdown("### 👥 Loaded Patients")
        for doc in st.session_state.documents:
            name = extract_name(doc["raw_text"]) or doc["doc_id"]
            st.markdown(f"- **{name}**")

        st.markdown("---")
        if st.button("🗑️ Clear All", use_container_width=True):
            for k in ["documents", "all_chunks", "embeddings", "bm25",
                      "embed_model", "llm_client", "chat_history", "indexed"]:
                st.session_state[k] = [] if k in ("documents", "all_chunks", "chat_history") else None
            st.session_state.indexed = False
            st.rerun()


# =================================================
# MAIN CONTENT
# =================================================
st.markdown('<div class="main-header">🏥 Medical RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload patient PDFs, view structured reports, and ask questions.</div>', unsafe_allow_html=True)

if not st.session_state.indexed:
    st.info("👈 Upload PDF files in the sidebar and click **Index Documents** to get started.")
    st.stop()

# Tabs
tab_chat, tab_report = st.tabs(["💬 Chat", "📋 Patient Reports"])

# =================================================
# TAB 1: CHAT
# =================================================
with tab_chat:
    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("source"):
                st.markdown(f'<span class="source-badge">{msg["source"]}</span>', unsafe_allow_html=True)
            if msg.get("confidence"):
                render_confidence(msg["confidence"])

    # Chat input
    query = st.chat_input("Ask a medical question...")
    if query:
        # Show user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                structured = classify_and_answer(query, st.session_state.documents)

            if structured:
                st.markdown(f"```\n{structured}\n```")
                st.markdown('<span class="source-badge">⚡ Structured Layer</span>', unsafe_allow_html=True)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"```\n{structured}\n```",
                    "source": "⚡ Structured Layer",
                })
            else:
                with st.spinner("Querying Llama 3.3 70B..."):
                    top_indices = hybrid_retrieval(
                        query,
                        st.session_state.all_chunks,
                        st.session_state.embed_model,
                        st.session_state.embeddings,
                        st.session_state.bm25,
                    )
                    context = "".join(
                        f"[{st.session_state.all_chunks[i]['doc_id']}]\n"
                        f"{st.session_state.all_chunks[i]['content']}\n\n"
                        for i in top_indices
                    )
                    answer = generate_answer(st.session_state.llm_client, query, context)

                    # Confidence
                    query_emb  = generate_embeddings(st.session_state.embed_model, [query])[0]
                    raw_scores = np.dot(st.session_state.embeddings, query_emb)
                    top_scores = raw_scores[top_indices]
                    top_scores = (top_scores - np.min(raw_scores)) / (np.max(raw_scores) - np.min(raw_scores) + 1e-8)
                    conf = score_confidence(query, answer, context, top_scores)

                st.markdown(answer)
                st.markdown('<span class="source-badge">🤖 Llama 3.3 70B</span>', unsafe_allow_html=True)
                render_confidence(conf)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "source": "🤖 Llama 3.3 70B",
                    "confidence": conf,
                })

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# =================================================
# TAB 2: PATIENT REPORTS
# =================================================
with tab_report:
    st.markdown("### Patient Records")

    if not st.session_state.documents:
        st.info("No documents loaded.")
    else:
        cols = st.columns(min(len(st.session_state.documents), 2))
        for i, doc in enumerate(st.session_state.documents):
            with cols[i % 2]:
                render_patient_card(doc)