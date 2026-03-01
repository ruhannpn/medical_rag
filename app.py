import os
import sys
import re
import tempfile
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi

# Add src directory to path
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, _ROOT)

from generator import (
    _load_env, load_llm, generate_answer, chunk_text,
    classify_and_answer, hybrid_retrieval, score_confidence,
    extract_name, extract_age, extract_gender, extract_diagnosis,
    extract_medications, extract_symptoms, extract_allergies,
    extract_dob, extract_visit_date, FIELD_REGISTRY,
)
from ingest import extract_text_from_pdf
from embedding import load_embedding_model, generate_embeddings

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="MedRAG — AI Medical Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c14 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: #0b1120 !important;
    border-right: 1px solid #1a2740 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0b1120; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }

/* ── Hero ── */
.hero {
    position: relative;
    padding: 2.5rem 0 1.5rem;
    overflow: hidden;
}
.hero-grid {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,163,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,163,255,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    mask-image: radial-gradient(ellipse 80% 80% at 50% 0%, black, transparent);
}
.hero-glow {
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 500px; height: 200px;
    background: radial-gradient(ellipse, rgba(0,163,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,163,255,0.08);
    border: 1px solid rgba(0,163,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 500;
    color: #00a3ff;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-badge::before {
    content: '';
    width: 6px; height: 6px;
    background: #00a3ff;
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1.1;
    color: #fff;
    margin-bottom: 0.6rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    background: linear-gradient(135deg, #00a3ff 0%, #0066ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 0.95rem;
    color: #4a6080;
    font-weight: 400;
    max-width: 500px;
    line-height: 1.6;
}

/* ── Stat chips ── */
.stats-row {
    display: flex;
    gap: 12px;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}
.stat-chip {
    background: #0d1929;
    border: 1px solid #1a2e47;
    border-radius: 8px;
    padding: 8px 16px;
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.stat-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #00a3ff;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label {
    font-size: 0.7rem;
    color: #3a5070;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Sidebar ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.5rem 0 1rem;
}
.sidebar-logo-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #00a3ff, #0044cc);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}
.sidebar-logo-text {
    font-size: 1.1rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.01em;
}
.sidebar-logo-text span { color: #00a3ff; }
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    color: #2a4060;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.2rem 0 0.5rem;
}
.patient-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #0d1929;
    border: 1px solid #1a2e47;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 0.85rem;
    color: #c0d4f0;
    font-weight: 500;
}
.patient-pill-dot {
    width: 7px; height: 7px;
    background: #00a3ff;
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 6px rgba(0,163,255,0.6);
}

/* ── Tab styling ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #0b1120;
    border-bottom: 1px solid #1a2740;
    gap: 0;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #3a5070 !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #00a3ff !important;
    border-bottom: 2px solid #00a3ff !important;
    background: transparent !important;
}

/* ── Patient card ── */
.pcard {
    background: linear-gradient(135deg, #0d1929 0%, #0a1520 100%);
    border: 1px solid #1a2e47;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.pcard:hover {
    border-color: #00a3ff44;
    box-shadow: 0 0 30px rgba(0,163,255,0.06);
}
.pcard::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00a3ff, #0044cc);
}
.pcard-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8f4ff;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.pcard-avatar {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #00a3ff22, #0044cc22);
    border: 1px solid #00a3ff33;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
}
.pcard-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}
.pcard-field {
    background: #080c14;
    border: 1px solid #121e30;
    border-radius: 8px;
    padding: 8px 10px;
}
.pcard-field-label {
    font-size: 0.65rem;
    color: #2a4060;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 3px;
}
.pcard-field-value {
    font-size: 0.82rem;
    color: #8ab4d4;
    font-weight: 500;
    line-height: 1.3;
}
.pcard-diag {
    grid-column: 1 / -1;
    background: rgba(0,163,255,0.05);
    border: 1px solid rgba(0,163,255,0.15);
}
.pcard-diag .pcard-field-value { color: #00a3ff; }
.pcard-meds {
    grid-column: 1 / -1;
}
.med-tag {
    display: inline-block;
    background: rgba(0,100,255,0.1);
    border: 1px solid rgba(0,100,255,0.2);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    color: #6090d0;
    margin: 2px 3px 2px 0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    flex-direction: row-reverse !important;
}
.answer-block {
    background: #0d1929;
    border: 1px solid #1a2e47;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    color: #b0c8e8;
    line-height: 1.7;
}
.source-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 6px;
    letter-spacing: 0.04em;
}
.source-structured {
    background: rgba(0,200,100,0.08);
    border: 1px solid rgba(0,200,100,0.2);
    color: #00c864;
}
.source-llm {
    background: rgba(0,163,255,0.08);
    border: 1px solid rgba(0,163,255,0.2);
    color: #00a3ff;
}

/* ── Confidence ── */
.conf-block {
    background: #080c14;
    border: 1px solid #1a2740;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-top: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}
.conf-bar-track {
    height: 4px;
    background: #1a2740;
    border-radius: 2px;
    margin: 6px 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
}
.conf-high-fill  { background: linear-gradient(90deg, #00c864, #00a050); }
.conf-med-fill   { background: linear-gradient(90deg, #f59e0b, #d97706); }
.conf-low-fill   { background: linear-gradient(90deg, #ef4444, #b91c1c); }
.conf-high-text  { color: #00c864; }
.conf-med-text   { color: #f59e0b; }
.conf-low-text   { color: #ef4444; }
.conf-metrics {
    display: flex;
    gap: 16px;
    color: #2a4060;
    font-size: 0.68rem;
    margin-top: 4px;
}
.conf-metric-val { color: #4a7090; }

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem;
    text-align: center;
}
.empty-icon {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, #0d1929, #111d2e);
    border: 1px solid #1a2e47;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    margin-bottom: 1rem;
}
.empty-title {
    font-size: 1rem;
    font-weight: 600;
    color: #c0d4f0;
    margin-bottom: 0.4rem;
}
.empty-sub {
    font-size: 0.82rem;
    color: #2a4060;
    line-height: 1.5;
}

/* ── Streamlit overrides ── */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
    border: 1px solid #1a2e47 !important;
    background: #0d1929 !important;
    color: #8ab4d4 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #00a3ff55 !important;
    color: #00a3ff !important;
    background: #0d1929 !important;
}
[data-testid="stFileUploader"] {
    background: #0d1929 !important;
    border: 1px dashed #1a2e47 !important;
    border-radius: 10px !important;
}
.stChatInput > div {
    background: #0d1929 !important;
    border: 1px solid #1a2e47 !important;
    border-radius: 12px !important;
}
.stChatInput input {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #c0d4f0 !important;
    font-size: 0.9rem !important;
}
div[data-testid="stMarkdownContainer"] p {
    color: #8ab4d4;
    font-size: 0.9rem;
}
.stSuccess {
    background: rgba(0,200,100,0.08) !important;
    border: 1px solid rgba(0,200,100,0.2) !important;
    color: #00c864 !important;
    border-radius: 8px !important;
}
.stInfo {
    background: rgba(0,163,255,0.06) !important;
    border: 1px solid rgba(0,163,255,0.15) !important;
    border-radius: 8px !important;
}
pre, code {
    font-family: 'JetBrains Mono', monospace !important;
    background: #080c14 !important;
    border: 1px solid #1a2740 !important;
    border-radius: 6px !important;
    color: #8ab4d4 !important;
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)


# =================================================
# SESSION STATE
# =================================================
def init_state():
    defaults = {
        "documents": [], "all_chunks": [], "embeddings": None,
        "bm25": None, "embed_model": None, "llm_client": None,
        "chat_history": [], "indexed": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# =================================================
# RESOURCE LOADERS
# =================================================
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embed_model():
    return load_embedding_model()


@st.cache_resource(show_spinner="Connecting to Groq...")
def get_llm_client():
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass  # no secrets file locally — fall through to .env
    _load_env()  # always try .env as fallback
    return load_llm()


# =================================================
# DOCUMENT PROCESSING
# =================================================
def process_uploaded_files(uploaded_files):
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
        key = name.strip().lower() if name else doc["doc_id"]
        if key not in seen_names:
            seen_names.add(key)
            documents.append(doc)

    return documents, all_chunks


def build_index(all_chunks, embed_model):
    texts = [c["content"] for c in all_chunks]
    embeddings = generate_embeddings(embed_model, texts)
    bm25 = BM25Okapi([re.findall(r"\w+", t.lower()) for t in texts])
    return embeddings, bm25


# =================================================
# COMPONENTS
# =================================================
def render_patient_card(doc: dict):
    t = doc["raw_text"]
    name  = extract_name(t)       or "Unknown"
    age   = extract_age(t)        or "N/A"
    gender= extract_gender(t)     or "N/A"
    dob   = extract_dob(t)        or "N/A"
    vdate = extract_visit_date(t) or "N/A"
    diag  = extract_diagnosis(t)  or "N/A"
    meds  = extract_medications(t)
    syms  = extract_symptoms(t)
    allg  = extract_allergies(t)

    med_tags = "".join(f'<span class="med-tag">{m}</span>' for m in meds) if meds else '<span style="color:#2a4060;font-size:0.78rem;">None on record</span>'
    sym_val  = ", ".join(syms) if syms else "N/A"
    allg_val = ", ".join(allg) if allg else "None"

    st.markdown(f"""
    <div class="pcard">
        <div class="pcard-name">
            <div class="pcard-avatar">👤</div>
            {name}
        </div>
        <div class="pcard-grid">
            <div class="pcard-field pcard-diag">
                <div class="pcard-field-label">Diagnosis</div>
                <div class="pcard-field-value">{diag}</div>
            </div>
            <div class="pcard-field">
                <div class="pcard-field-label">Age</div>
                <div class="pcard-field-value">{age}</div>
            </div>
            <div class="pcard-field">
                <div class="pcard-field-label">Gender</div>
                <div class="pcard-field-value">{gender}</div>
            </div>
            <div class="pcard-field">
                <div class="pcard-field-label">DOB</div>
                <div class="pcard-field-value">{dob}</div>
            </div>
            <div class="pcard-field">
                <div class="pcard-field-label">Visit Date</div>
                <div class="pcard-field-value">{vdate}</div>
            </div>
            <div class="pcard-field pcard-meds">
                <div class="pcard-field-label">Medications</div>
                <div style="margin-top:4px;">{med_tags}</div>
            </div>
            <div class="pcard-field">
                <div class="pcard-field-label">Symptoms</div>
                <div class="pcard-field-value">{sym_val}</div>
            </div>
            <div class="pcard-field">
                <div class="pcard-field-label">Allergies</div>
                <div class="pcard-field-value">{allg_val}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_confidence(conf: dict):
    label = conf["label"]
    score = conf["score"]
    css   = {"HIGH": "conf-high", "MEDIUM": "conf-med", "LOW": "conf-low"}[label]
    fill  = {"HIGH": "conf-high-fill", "MEDIUM": "conf-med-fill", "LOW": "conf-low-fill"}[label]

    st.markdown(f"""
    <div class="conf-block">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#2a4060; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em;">Confidence</span>
            <span class="{css}-text" style="font-weight:600;">{score}% — {label}</span>
        </div>
        <div class="conf-bar-track">
            <div class="conf-bar-fill {fill}" style="width:{score}%;"></div>
        </div>
        <div class="conf-metrics">
            <span>Retrieval <span class="conf-metric-val">{conf['retrieval']}%</span></span>
            <span>Coverage <span class="conf-metric-val">{conf['coverage']}%</span></span>
            <span>Quality <span class="conf-metric-val">{conf['quality']}%</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)



def render_structured_response(text: str):
    """Render structured layer output as clean patient cards."""
    lines = text.split("\n")
    patients = []
    current = None

    for line in lines:
        line = line.strip()
        if not line or set(line) <= {"=", "-", " "}:
            continue
        if line == "Patient Report":
            continue
        if ":" not in line:
            continue

        label, _, value = line.partition(":")
        label = label.strip()
        value = value.strip()

        if label == "Name":
            if current is not None:
                patients.append(current)
            current = {"Name": value, "fields": []}
        elif current is not None and value and value != "N/A":
            current["fields"].append((label, value))

    if current is not None:
        patients.append(current)

    import streamlit.components.v1 as components
    # Render all cards in one call via components.html — bypasses Streamlit HTML sanitization
    cards_html = []
    for p in patients:
        name = p["Name"]
        fields_html = []
        for label, value in p["fields"]:
            label_l = label.lower()
            if label_l == "diagnosis":
                fields_html.append(f"""
                <div style="grid-column:1/-1;background:rgba(0,163,255,0.06);border:1px solid rgba(0,163,255,0.15);border-radius:8px;padding:8px 10px;margin-bottom:2px;">
                    <div style="font-size:0.65rem;color:#2a4060;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">{label}</div>
                    <div style="font-size:0.85rem;color:#00a3ff;font-weight:500;">{value}</div>
                </div>""")
            elif label_l == "medications":
                tags = "".join(
                    f'<span style="display:inline-block;background:rgba(0,100,255,0.1);border:1px solid rgba(0,100,255,0.2);border-radius:4px;padding:2px 8px;font-size:0.72rem;color:#6090d0;margin:2px 3px 2px 0;">{m.strip()}</span>'
                    for m in value.split(",")
                )
                fields_html.append(f"""
                <div style="grid-column:1/-1;background:#080c14;border:1px solid #121e30;border-radius:8px;padding:8px 10px;margin-bottom:2px;">
                    <div style="font-size:0.65rem;color:#2a4060;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:5px;">{label}</div>
                    <div>{tags}</div>
                </div>""")
            else:
                fields_html.append(f"""
                <div style="background:#080c14;border:1px solid #121e30;border-radius:8px;padding:8px 10px;margin-bottom:2px;">
                    <div style="font-size:0.65rem;color:#2a4060;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">{label}</div>
                    <div style="font-size:0.82rem;color:#8ab4d4;font-weight:500;">{value}</div>
                </div>""")

        cards_html.append(f"""
        <div style="background:#0d1929;border:1px solid #1a2e47;border-radius:12px;padding:1rem 1.2rem;margin-bottom:10px;position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00a3ff,#0044cc);"></div>
            <div style="font-size:1rem;font-weight:700;color:#e8f4ff;margin-bottom:10px;display:flex;align-items:center;gap:8px;">
                <span style="background:linear-gradient(135deg,#00a3ff22,#0044cc22);border:1px solid #00a3ff33;border-radius:6px;padding:2px 8px;font-size:0.8rem;">👤</span>
                {name}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
                {"".join(fields_html)}
            </div>
        </div>""")

    full_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Space Grotesk', sans-serif; }}
    body {{ background: transparent; padding: 4px; }}
    </style>
    {"".join(cards_html)}
    """
    height = len(patients) * 220
    components.html(full_html, height=height, scrolling=False)


# =================================================
# SIDEBAR
# =================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🧬</div>
        <div class="sidebar-logo-text">Med<span>RAG</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Upload Records</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⟳  Index Documents", use_container_width=True):
            with st.spinner("Processing..."):
                embed_model = get_embed_model()
                docs, chunks = process_uploaded_files(uploaded_files)
                embeddings, bm25 = build_index(chunks, embed_model)
                st.session_state.update({
                    "documents": docs, "all_chunks": chunks,
                    "embeddings": embeddings, "bm25": bm25,
                    "embed_model": embed_model,
                    "llm_client": get_llm_client(),
                    "indexed": True, "chat_history": [],
                })
            st.success(f"Indexed {len(docs)} patient(s)")

    if st.session_state.indexed:
        st.markdown('<div class="sidebar-section">Patients</div>', unsafe_allow_html=True)
        for doc in st.session_state.documents:
            name = extract_name(doc["raw_text"]) or doc["doc_id"]
            st.markdown(f"""
            <div class="patient-pill">
                <div class="patient-pill-dot"></div>
                {name}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✕  Clear All", use_container_width=True):
            for k in ["documents", "all_chunks", "embeddings", "bm25",
                      "embed_model", "llm_client", "chat_history"]:
                st.session_state[k] = [] if isinstance(st.session_state[k], list) else None
            st.session_state.indexed = False
            st.rerun()


# =================================================
# MAIN
# =================================================
n_patients = len(st.session_state.documents)
n_chunks   = len(st.session_state.all_chunks)
n_msgs     = len([m for m in st.session_state.chat_history if m["role"] == "user"])

st.markdown(f"""
<div class="hero">
    <div class="hero-grid"></div>
    <div class="hero-glow"></div>
    <div class="hero-badge">AI · Medical Intelligence</div>
    <div class="hero-title">Clinical Records<br><span>Intelligence</span></div>
    <div class="hero-sub">Upload patient PDFs, query records with natural language, and get AI-powered insights backed by hybrid retrieval.</div>
    <div class="stats-row">
        <div class="stat-chip">
            <div class="stat-value">{n_patients}</div>
            <div class="stat-label">Patients</div>
        </div>
        <div class="stat-chip">
            <div class="stat-value">{n_chunks}</div>
            <div class="stat-label">Chunks indexed</div>
        </div>
        <div class="stat-chip">
            <div class="stat-value">{n_msgs}</div>
            <div class="stat-label">Queries run</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.indexed:
    # Mobile-friendly upload — show uploader directly in main area
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📂</div>
        <div class="empty-title">Upload Patient Records</div>
        <div class="empty-sub">Drop your PDF files below to get started.</div>
    </div>
    """, unsafe_allow_html=True)

    mobile_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if mobile_files:
        if st.button("⟳  Index Documents", use_container_width=True, type="primary"):
            with st.spinner("Processing..."):
                embed_model = get_embed_model()
                docs, chunks = process_uploaded_files(mobile_files)
                embeddings, bm25 = build_index(chunks, embed_model)
                st.session_state.update({
                    "documents": docs, "all_chunks": chunks,
                    "embeddings": embeddings, "bm25": bm25,
                    "embed_model": embed_model,
                    "llm_client": get_llm_client(),
                    "indexed": True, "chat_history": [],
                })
            st.rerun()
    st.stop()

tab_chat, tab_report = st.tabs(["  💬  Chat  ", "  📋  Patient Reports  "])

# ── CHAT TAB ──
with tab_chat:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(f'<div class="answer-block">{msg["content"]}</div>', unsafe_allow_html=True)
                if msg.get("source") == "structured":
                    st.markdown('<span class="source-tag source-structured">⚡ Structured Layer</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="source-tag source-llm">🤖 Llama 3.3 70B</span>', unsafe_allow_html=True)
                if msg.get("confidence"):
                    render_confidence(msg["confidence"])
            else:
                st.markdown(msg["content"])

    query = st.chat_input("Ask anything about your patients...")
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner(""):
                structured = classify_and_answer(query, st.session_state.documents)

            if structured:
                render_structured_response(structured)
                st.markdown('<span class="source-tag source-structured">⚡ Structured Layer</span>', unsafe_allow_html=True)
                st.session_state.chat_history.append({
                    "role": "assistant", "content": structured, "source": "structured",
                })
            else:
                with st.spinner("Querying Llama 3.3 70B..."):
                    top_indices = hybrid_retrieval(
                        query, st.session_state.all_chunks,
                        st.session_state.embed_model,
                        st.session_state.embeddings,
                        st.session_state.bm25,
                    )
                    context = "".join(
                        f"[{st.session_state.all_chunks[i]['doc_id']}]\n{st.session_state.all_chunks[i]['content']}\n\n"
                        for i in top_indices
                    )
                    answer = generate_answer(st.session_state.llm_client, query, context)
                    query_emb  = generate_embeddings(st.session_state.embed_model, [query])[0]
                    raw_scores = np.dot(st.session_state.embeddings, query_emb)
                    top_scores = raw_scores[top_indices]
                    top_scores = (top_scores - np.min(raw_scores)) / (np.max(raw_scores) - np.min(raw_scores) + 1e-8)
                    conf = score_confidence(query, answer, context, top_scores)

                st.markdown(f'<div class="answer-block">{answer}</div>', unsafe_allow_html=True)
                st.markdown('<span class="source-tag source-llm">🤖 Llama 3.3 70B</span>', unsafe_allow_html=True)
                render_confidence(conf)
                st.session_state.chat_history.append({
                    "role": "assistant", "content": answer,
                    "source": "llm", "confidence": conf,
                })

    if st.session_state.chat_history:
        if st.button("✕  Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

# ── REPORTS TAB ──
with tab_report:
    if not st.session_state.documents:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📋</div>
            <div class="empty-title">No patient records</div>
            <div class="empty-sub">Upload and index PDFs to view structured patient cards.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cols = st.columns(min(len(st.session_state.documents), 2))
        for i, doc in enumerate(st.session_state.documents):
            with cols[i % 2]:
                render_patient_card(doc)