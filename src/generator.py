import os
import re
import numpy as np
from groq import Groq
from rank_bm25 import BM25Okapi

from ingest import extract_text_from_pdf
from embedding import load_embedding_model, generate_embeddings

# =================================================
# ENV LOADING  — tries every likely location
# =================================================
def _load_env():
    here   = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    candidates = [
        os.path.join(here,   ".env"),   # src/.env  ← your file
        os.path.join(here,   "env"),
        os.path.join(parent, ".env"),
        os.path.join(parent, "env"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        print(f"[Info] Reading env from: {path}")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), val)
        return
    print(f"[Warning] No env file found. Searched: {candidates}")

_load_env()


# =================================================
# LLM  (Groq — Llama 3.3 70B)
# =================================================
def load_llm() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\n"
            "Make sure your 'env' file (in src/ or project root) contains:\n"
            "  GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx"
        )
    return Groq(api_key=api_key)


def generate_answer(client: Groq, query: str, context: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical records assistant. "
                    "Answer questions using ONLY the patient records provided. "
                    "Be concise and accurate. If information is not in the records, say so clearly. "
                    "When multiple patients are present, address each one separately."
                ),
            },
            {
                "role": "user",
                "content": f"Patient Records:\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# =================================================
# CHUNKING
# =================================================
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start: start + chunk_size]))
        start += chunk_size - overlap
    return chunks


# =================================================
# LOAD DOCUMENTS  (deduplicated by patient name)
# =================================================
def load_all_documents(data_folder: str) -> tuple[list, list]:
    raw_documents: list[dict] = []
    all_chunks: list[dict] = []

    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"[Warning] No PDF files found in '{data_folder}'.")
        return [], []

    for filename in pdf_files:
        path = os.path.join(data_folder, filename)
        raw_text = extract_text_from_pdf(path)
        raw_documents.append({"doc_id": filename, "raw_text": raw_text})
        for chunk in chunk_text(raw_text):
            all_chunks.append({"doc_id": filename, "content": chunk})

    documents: list[dict] = []
    seen_names: set[str] = set()
    for doc in raw_documents:
        name = extract_name(doc["raw_text"])
        key = name.strip().lower() if name else doc["doc_id"]
        if key not in seen_names:
            seen_names.add(key)
            documents.append(doc)

    return documents, all_chunks


# =================================================
# STRUCTURED EXTRACTION HELPERS
# =================================================
def _find(pattern: str, text: str, group: int = 1) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(group).strip() if m else None


def extract_name(text: str):       return _find(r"Name:\s*(.+)", text)
def extract_age(text: str):        return _find(r"Age:\s*(\d+)", text)
def extract_diagnosis(text: str):  return _find(r"Diagnosis[:\s]+(.+)", text)
def extract_gender(text: str):     return _find(r"(?:Gender|Sex)[:\s]+(.+)", text)
def extract_dob(text: str):        return _find(r"(?:DOB|Date of Birth)[:\s]+(.+)", text)
def extract_visit_date(text: str): return _find(r"(?:Visit Date|Date of Visit|Appointment Date)[:\s]+(.+)", text)


_MED_PATTERN = re.compile(
    r"\b(metformin|lisinopril|amlodipine|atorvastatin|aspirin|salbutamol|albuterol|"
    r"montelukast|fluticasone|budesonide|salmeterol|omeprazole|losartan|"
    r"hydrochlorothiazide|glipizide|sitagliptin|insulin|prednisone|ibuprofen|"
    r"paracetamol|acetaminophen|cetirizine|loratadine|amoxicillin|azithromycin)\b",
    re.IGNORECASE,
)


def extract_medications(text: str) -> list[str]:
    candidates: list[str] = []

    block = re.search(
        r"(?:Medications?|Prescribed Medications?|Current Medications?)[:\s]*\n(.*?)"
        r"(?=\n[A-Z][^\n]{0,50}:|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if block:
        for line in block.group(1).splitlines():
            line = re.sub(r"^[-•*\d.\s]+", "", line).strip()
            if line:
                candidates.append(line)

    if not candidates:
        inline = _find(r"(?:Medications?|Prescribed)[:\s]+(.+)", text)
        if inline:
            candidates = [s.strip() for s in re.split(r"[,;]", inline) if s.strip()]

    if not candidates:
        found = _MED_PATTERN.findall(text)
        candidates = list(dict.fromkeys(m.capitalize() for m in found))

    return [
        c for c in candidates
        if 1 <= len(c.split()) <= 5
        and re.match(r"[A-Z]", c)
        and not re.search(r"\b(after|follow|month|week|day|return|visit|per|as needed)\b", c, re.I)
    ]


def extract_symptoms(text: str) -> list[str]:
    block = re.search(
        r"(?:Symptoms?|Chief Complaint|Presenting Complaints?)[:\s]*\n(.*?)"
        r"(?=\n[A-Z][^\n]{0,50}:|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if block:
        items = re.findall(r"[-•*\d.]*\s*([A-Za-z][^\n,;]{2,})", block.group(1))
        if items:
            return [i.strip() for i in items]

    inline = _find(r"(?:Symptoms?|Chief Complaint)[:\s]+(.+)", text)
    if inline:
        return [s.strip() for s in re.split(r"[,;]", inline) if s.strip()]
    return []


def extract_allergies(text: str) -> list[str]:
    inline = _find(r"Allergies?[:\s]+(.+)", text)
    if inline and inline.lower() not in ("none", "nkda", "none known", "n/a"):
        return [a.strip() for a in re.split(r"[,;]", inline) if a.strip()]
    return []


# =================================================
# FIELD REGISTRY
# =================================================
FIELD_REGISTRY: dict[str, dict] = {
    "name":        {"keywords": ["name"],                                         "extractor": extract_name,        "label": "Name"},
    "age":         {"keywords": ["age"],                                          "extractor": extract_age,         "label": "Age"},
    "gender":      {"keywords": ["gender", "sex"],                                "extractor": extract_gender,      "label": "Gender"},
    "dob":         {"keywords": ["dob", "date of birth", "birth"],                "extractor": extract_dob,         "label": "DOB"},
    "visit_date":  {"keywords": ["visit date", "visit", "appointment"],           "extractor": extract_visit_date,  "label": "Visit Date"},
    "diagnosis":   {"keywords": ["diagnosis", "diagnos", "condition", "disease"], "extractor": extract_diagnosis,   "label": "Diagnosis"},
    "medications": {"keywords": ["medication", "medicine", "drug", "prescribed", "prescription"],
                                                                                  "extractor": extract_medications, "label": "Medications"},
    "symptoms":    {"keywords": ["symptom", "complaint", "presenting"],           "extractor": extract_symptoms,    "label": "Symptoms"},
    "allergies":   {"keywords": ["allerg"],                                       "extractor": extract_allergies,   "label": "Allergies"},
}

DEFAULT_SUMMARY_FIELDS = list(FIELD_REGISTRY.keys())


def detect_requested_fields(query: str) -> list[str]:
    q = query.lower()
    matched = [
        field_key for field_key, meta in FIELD_REGISTRY.items()
        if any(kw in q for kw in meta["keywords"])
    ]
    if matched and "name" not in matched:
        matched.insert(0, "name")
    return matched if matched else DEFAULT_SUMMARY_FIELDS


def build_custom_report(documents: list, fields: list[str]) -> str:
    lines = ["Patient Report", "=" * 50]
    for doc in documents:
        t = doc["raw_text"]
        for field_key in fields:
            meta = FIELD_REGISTRY[field_key]
            value = meta["extractor"](t)
            if isinstance(value, list):
                value = ", ".join(value) if value else "N/A"
            lines.append(f"  {meta['label']:<14}: {value or 'N/A'}")
        lines.append("-" * 50)
    return "\n".join(lines)


# =================================================
# STRUCTURED QUERY HANDLERS
# =================================================
def structured_asthma(documents: list) -> list:
    return list({
        n for doc in documents
        if "asthma" in doc["raw_text"].lower()
        and (n := extract_name(doc["raw_text"]))
    })


# =================================================
# HYBRID RETRIEVAL
# =================================================
def _normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)


def hybrid_retrieval(
    query: str,
    all_chunks: list,
    embed_model,
    embeddings: np.ndarray,
    bm25: BM25Okapi,
    top_k: int = 5,
    vector_weight: float = 0.6,
) -> np.ndarray:
    query_emb = generate_embeddings(embed_model, [query])[0]
    vector_scores = _normalize(np.dot(embeddings, query_emb))
    tokenized_query = re.findall(r"\w+", query.lower())
    bm25_scores = _normalize(bm25.get_scores(tokenized_query))
    combined = vector_weight * vector_scores + (1 - vector_weight) * bm25_scores
    return np.argsort(combined)[::-1][:top_k]



# =================================================
# CONFIDENCE SCORING
# =================================================
_UNCERTAINTY_PHRASES = re.compile(
    r"\b(not found|not mentioned|no information|unclear|unknown|not specified|"
    r"not available|not provided|cannot determine|not in (the )?(records?|documents?|context))\b",
    re.IGNORECASE,
)


def score_confidence(query: str, answer: str, context: str, top_scores: np.ndarray) -> dict:
    """
    Score LLM answer confidence across three signals:
    1. Retrieval score  — how strongly the top chunks matched the query (0-1)
    2. Answer coverage  — how many query keywords appear in the retrieved context (0-1)
    3. Answer quality   — penalise if the LLM expressed uncertainty (0 or 1)
    """
    retrieval_score = float(np.mean(top_scores)) if len(top_scores) > 0 else 0.0

    query_words = set(re.findall(r"\w+", query.lower())) - {
        "a", "an", "the", "is", "are", "was", "were", "of", "in",
        "for", "to", "and", "or", "with", "give", "me", "their",
        "what", "who", "how", "tell", "show", "about",
    }
    context_words = set(re.findall(r"\w+", context.lower()))
    coverage = len(query_words & context_words) / len(query_words) if query_words else 0.0

    uncertainty = 1.0 if _UNCERTAINTY_PHRASES.search(answer) else 0.0
    quality_score = 1.0 - uncertainty

    final = (0.4 * retrieval_score) + (0.4 * coverage) + (0.2 * quality_score)
    final_pct = round(final * 100, 1)

    if final_pct >= 75:
        label = "HIGH"
    elif final_pct >= 45:
        label = "MEDIUM"
    else:
        label = "LOW"

    return {
        "score": final_pct,
        "label": label,
        "retrieval": round(retrieval_score * 100, 1),
        "coverage":  round(coverage * 100, 1),
        "quality":   round(quality_score * 100, 1),
    }


def format_confidence(conf: dict) -> str:
    bar_len = int(conf["score"] / 5)
    bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
    return (
        f"Confidence : [{bar}] {conf['score']}% ({conf['label']})"
        f"  Retrieval: {conf['retrieval']}%  Coverage: {conf['coverage']}%  Quality: {conf['quality']}%"
    )

# =================================================
# QUERY INTENT CLASSIFIER
# =================================================
# Strict report triggers — only explicit data/listing requests go to structured layer.
# Everything else (questions about treatment, history, advice, etc.) falls to LLM.
_REPORT_TRIGGERS = re.compile(
    r"\b(report|summary|overview|profile|list all|all patients)\b",
    re.IGNORECASE,
)

# Explicit field-only requests: "give me the ages", "show me medications"
# Must pair a listing verb WITH a known field keyword to qualify.
_LISTING_VERBS = re.compile(
    r"\b(give me|show me|what are|tell me|get me)\b",
    re.IGNORECASE,
)

_SINGLE_FIELD_TRIGGERS = {
    "asthma": (lambda q: "asthma" in q, structured_asthma, "Patients with Asthma"),
}


def _is_field_only_query(query: str) -> bool:
    """
    Returns True only when the query is clearly a data listing request —
    a listing verb paired with a known field keyword, with no open-ended
    question words like 'why', 'how', 'explain', 'describe', 'what is'.
    """
    q = query.lower()

    # If it contains open-ended question words, send to LLM
    if re.search(r"\b(why|how|explain|describe|what is|what was|who is|tell me about|detail|common|occur|typical|usually|generally|often|cause|treat|prevent|risk|recommend)\b", q):
        return False

    # Must have a listing verb AND a field keyword
    has_verb  = bool(_LISTING_VERBS.search(q))
    has_field = any(
        kw in q
        for meta in FIELD_REGISTRY.values()
        for kw in meta["keywords"]
    )
    return has_verb and has_field


def classify_and_answer(query: str, documents: list) -> str | None:
    q = query.lower()

    # Intent 1: Explicit report/summary request
    if _REPORT_TRIGGERS.search(q):
        fields = detect_requested_fields(query)
        return build_custom_report(documents, fields)

    # Intent 2: Specific condition lookups
    for _key, (predicate, handler, label) in _SINGLE_FIELD_TRIGGERS.items():
        if predicate(q):
            result = handler(documents)
            if result:
                return f"[{label}]\n" + "\n".join(f"  {v}" for v in result)

    # Intent 3: Field-only listing query ("give me the ages", "show me medications")
    if _is_field_only_query(query):
        fields = detect_requested_fields(query)
        return build_custom_report(documents, fields)

    # Everything else → LLM
    return None


# =================================================
# MAIN
# =================================================
def main():
    data_folder = "../data"

    print("Loading documents...")
    documents, all_chunks = load_all_documents(data_folder)
    if not all_chunks:
        print("No content to index. Exiting.")
        return

    texts = [c["content"] for c in all_chunks]

    print("Loading embedding model...")
    embed_model = load_embedding_model()

    print("Generating embeddings...")
    embeddings = generate_embeddings(embed_model, texts)

    print("Initializing BM25...")
    tokenized_corpus = [re.findall(r"\w+", t.lower()) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    print("Connecting to Groq (Llama 3.3 70B)...")
    client = load_llm()

    print("\nLayered Hybrid RAG Ready. Type 'exit' to quit.\n")

    while True:
        query = input("Ask a medical question: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        answer = classify_and_answer(query, documents)
        if answer:
            print("\n=== Final Answer (Structured Layer) ===")
            print(answer)
            print("-" * 60)
            continue

        top_indices = hybrid_retrieval(query, all_chunks, embed_model, embeddings, bm25)
        context = "".join(
            f"[{all_chunks[i]['doc_id']}]\n{all_chunks[i]['content']}\n\n"
            for i in top_indices
        )

        # Retrieval scores for confidence
        query_emb = generate_embeddings(embed_model, [query])[0]
        raw_scores = np.dot(embeddings, query_emb)
        top_raw_scores = raw_scores[top_indices]
        top_raw_scores = (top_raw_scores - np.min(raw_scores)) / (np.max(raw_scores) - np.min(raw_scores) + 1e-8)

        print("\nGenerating answer...\n")
        answer = generate_answer(client, query, context)

        conf = score_confidence(query, answer, context, top_raw_scores)

        print("=== Final Answer (Llama 3.3 70B) ===")
        print(answer)
        print()
        print(format_confidence(conf))
        print("-" * 60)


if __name__ == "__main__":
    main()