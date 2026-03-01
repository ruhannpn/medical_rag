# Medical RAG Assistant

A layered Retrieval-Augmented Generation (RAG) system for querying patient medical reports using hybrid retrieval and a large language model — with a full Streamlit UI.

---

## 🚀 Live Demo

 https://medicalrag-a49jzxrqf5afsqynfmclpk.streamlit.app/

---

## 📌 Features

- **PDF Upload** — upload one or more patient medical reports at runtime
- **Chat Interface** — ask natural language questions about patients
- **Patient Report View** — structured card view of all extracted patient data
- **Layered Retrieval** — structured regex layer for direct lookups, hybrid RAG for complex questions
- **Hybrid Search** — BM25 sparse retrieval + dense vector embeddings fused with weighted scoring
- **Confidence Scoring** — every LLM answer is scored across retrieval quality, keyword coverage, and answer certainty
- **Llama 3.3 70B via Groq** — fast, free LLM inference for open-ended medical questions

---

## 🧠 How It Works

```
User Query
    │
    ▼
Intent Classifier
    │
    ├── Structured Layer (regex)   →  direct field extraction, no LLM
    │       report / summary
    │       give me ages / medications
    │       condition lookups
    │
    └── Hybrid RAG + LLM
            │
            ├── BM25 (keyword match)
            ├── Vector Embeddings (semantic match)
            ├── Weighted fusion (60% vector / 40% BM25)
            └── Llama 3.3 70B via Groq API
                    └── Confidence Score
```

### Retrieval Strategies

| Layer | Method | When Used |
|---|---|---|
| Structured | Regex extraction | Explicit field queries (age, diagnosis, medications) |
| Dense | Sentence embeddings + cosine similarity | Semantic questions |
| Sparse | BM25Okapi | Keyword-heavy queries |
| Hybrid | Weighted fusion of dense + sparse | All LLM fallback queries |

---

## 🗂️ Project Structure

```
medical_report/
├── app.py                  # Streamlit UI
├── requirements.txt
├── README.md
└── src/
    ├── generator.py        # Core RAG logic, LLM, structured layer, confidence scoring
    ├── ingest.py           # PDF text extraction
    ├── embedding.py        # Sentence embedding model
    ├── retriever.py        # Retrieval utilities
    └── chunking.py         # Text chunking
```

---

## ⚙️ Local Setup

**1. Clone the repo**
```bash
git clone "https://github.com/ruhannpn/medical_rag"
cd medical_report
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a `.env` file in the `src/` folder:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```
Get a free key at [console.groq.com](https://console.groq.com)

**5. Run the app**
```bash
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **Create app** → select repo → branch `main` → main file `app.py`
4. Under **Advanced settings → Secrets**, add:
```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
```
5. Click **Deploy**

---

## 📦 Requirements

```
streamlit
groq
rank-bm25
transformers
sentence-transformers
numpy
pymupdf
python-dotenv
```

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for Llama 3.3 70B inference |

---

## 💡 Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Llama 3.3 70B via Groq |
| Embeddings | Sentence Transformers |
| Sparse Retrieval | BM25Okapi (rank-bm25) |
| PDF Parsing | PyMuPDF |
| Language | Python 3.11+ |

---

## 👤 Author

Built by **Ruhan** —  https://github.com/ruhannpn