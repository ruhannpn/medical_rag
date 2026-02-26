import os
import re
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ingest import extract_text_from_pdf
from embedding import load_embedding_model, generate_embeddings


# =================================================
# LLM
# =================================================
def load_local_llm(model_name: str = "google/flan-t5-large"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def generate_answer(tokenizer, model, prompt: str, max_tokens: int = 300) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# =================================================
# CHUNKING
# =================================================
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start: start + chunk_size]))
        start += chunk_size - overlap
    return chunks


# =================================================
# LOAD DOCUMENTS
# =================================================
def load_all_documents(data_folder: str) -> tuple:
    documents, all_chunks = [], []

    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"[Warning] No PDF files found in '{data_folder}'.")
        return documents, all_chunks

    for filename in sorted(pdf_files):
        path = os.path.join(data_folder, filename)
        raw_text = extract_text_from_pdf(path)
        documents.append({"doc_id": filename, "raw_text": raw_text})
        for chunk in chunk_text(raw_text):
            all_chunks.append({"doc_id": filename, "content": chunk})

    return documents, all_chunks


# =================================================
# STRUCTURED EXTRACTION HELPERS
# =================================================
def _find(pattern: str, text: str, flags=re.IGNORECASE):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def extract_name(text: str):
    return _find(r"Name:\s*(.+)", text)

def extract_age(text: str):
    return _find(r"Age:\s*(\d+)", text)

def extract_diagnosis(text: str):
    return _find(r"Diagnosis[:\s]+(.+)", text)

def extract_medications(text: str):
    """Extract medications from a 'Medications:' section or bullet list."""
    section = _find(r"Medications?[:\s]+(.+?)(?=\n[A-Z]|\Z)", text)
    if section:
        meds = [m.strip() for m in re.split(r"[,;\n]+", section) if m.strip()]
        return meds if meds else None
    # Fallback: bullet/dash list
    meds = re.findall(r"[-•]\s*([A-Za-z][^\n,;]+)", text)
    return [m.strip() for m in meds] if meds else None


# =================================================
# PER-DOCUMENT FULL PROFILE
# =================================================
def build_patient_profile(doc: dict) -> dict:
    text = doc["raw_text"]
    meds = extract_medications(text)
    return {
        "doc_id":      doc["doc_id"],
        "name":        extract_name(text)      or "Unknown",
        "age":         extract_age(text)        or "N/A",
        "diagnosis":   extract_diagnosis(text)  or "N/A",
        "medications": ", ".join(meds) if meds else "N/A",
    }


def get_all_profiles(documents: list) -> list:
    return [build_patient_profile(d) for d in documents]


# =================================================
# STRUCTURED QUERY HANDLERS
# =================================================
def structured_age(documents: list) -> list:
    return [f"  {p['name']}: {p['age']}" for p in get_all_profiles(documents)]


def structured_asthma(documents: list) -> list:
    return [
        f"  {extract_name(d['raw_text']) or 'Unknown'}"
        for d in documents
        if "asthma" in d["raw_text"].lower()
    ]


def structured_diagnosis(documents: list) -> list:
    return [f"  {p['name']}: {p['diagnosis']}" for p in get_all_profiles(documents)]


def structured_medications(documents: list) -> list:
    return [f"  {p['name']}: {p['medications']}" for p in get_all_profiles(documents)]


def structured_summary(documents: list) -> list:
    lines = ["Patient Summary Report", "=" * 50]
    for p in get_all_profiles(documents):
        lines += [
            f"\nPatient     : {p['name']}",
            f"  File        : {p['doc_id']}",
            f"  Age         : {p['age']}",
            f"  Diagnosis   : {p['diagnosis']}",
            f"  Medications : {p['medications']}",
        ]
    lines.append("\n" + "=" * 50)
    return lines


# =================================================
# MULTI-KEYWORD QUERY ROUTER
# =================================================
# Each entry: (keyword_to_match, section_label, handler_function)
# Handlers are deduplicated — each fires at most once per query.
ROUTES = [
    ("summary",    "Summary",     structured_summary),
    ("report",     "Summary",     structured_summary),
    ("age",        "Ages",        structured_age),
    ("medication", "Medications", structured_medications),
    ("drug",       "Medications", structured_medications),
    ("diagnosis",  "Diagnoses",   structured_diagnosis),
    ("asthma",     "Asthma",      structured_asthma),
]


def try_structured(query: str, documents: list):
    """
    Scan the query for ALL known keywords and run every matching handler.
    Returns combined output lines, or None if no keywords matched.
    """
    q = query.lower()
    output = []
    seen_handlers = set()

    for keyword, label, handler in ROUTES:
        if keyword in q and id(handler) not in seen_handlers:
            seen_handlers.add(id(handler))
            result = handler(documents)
            if result:
                # Summary is self-labelled; others get a section header
                if handler is not structured_summary:
                    output.append(f"\n[{label}]")
                output.extend(result)

    return output if output else None


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
    top_k: int = 4,
    vector_weight: float = 0.6,
) -> np.ndarray:
    query_emb = generate_embeddings(embed_model, [query])[0]
    vector_scores = _normalize(np.dot(embeddings, query_emb))

    tokenized_query = re.findall(r"\w+", query.lower())
    bm25_scores = _normalize(bm25.get_scores(tokenized_query))

    combined = vector_weight * vector_scores + (1 - vector_weight) * bm25_scores
    return np.argsort(combined)[::-1][:top_k]


def build_prompt(query: str, context: str) -> str:
    return (
        "You are a medical assistant. Answer using ONLY the context below.\n"
        "Do NOT mix information from different documents.\n"
        "If the answer is not in the context, reply: 'Not found in documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


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

    print("Loading LLM...")
    tokenizer, model = load_local_llm()

    print("\nLayered Hybrid RAG Ready. Type 'exit' to quit.\n")

    while True:
        query = input("Ask a medical question: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        # --- Structured layer (all matching keywords fire) ---
        structured_answers = try_structured(query, documents)
        if structured_answers:
            print("\n=== Final Answer (Structured Layer) ===")
            print("\n".join(structured_answers))
            print("-" * 60)
            continue

        # --- Hybrid RAG fallback ---
        top_indices = hybrid_retrieval(query, all_chunks, embed_model, embeddings, bm25)
        context = "".join(
            f"Document: {all_chunks[i]['doc_id']}\n{all_chunks[i]['content']}\n\n"
            for i in top_indices
        )

        print("\nGenerating answer...\n")
        answer = generate_answer(tokenizer, model, build_prompt(query, context))

        print("=== Final Answer ===")
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
