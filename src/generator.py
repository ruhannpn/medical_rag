from __future__ import annotations
import os
import re
import numpy as np
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from extractor import ClinicalExtractor
from database import DatabaseManager

# Import extraction functions from ClinicalExtractor for backward compatibility
extract_name = ClinicalExtractor.extract_name
extract_age = ClinicalExtractor.extract_age
extract_gender = ClinicalExtractor.extract_gender
extract_dob = ClinicalExtractor.extract_dob
extract_visit_date = ClinicalExtractor.extract_visit_date
extract_diagnosis = ClinicalExtractor.extract_diagnosis
extract_medications = ClinicalExtractor.extract_medications
extract_symptoms = ClinicalExtractor.extract_symptoms
extract_allergies = ClinicalExtractor.extract_allergies

# =================================================
# ENV LOADING
# =================================================
def _load_env():
    here   = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    candidates = [
        os.path.join(here,   ".env"),
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
# LLM Loader
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
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

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

def build_custom_report(patients: list[dict], fields: list[str]) -> str:
    """
    Builds custom reports using database patients dictionary list.
    """
    lines = ["Patient Report", "=" * 50]
    for pat in patients:
        for field_key in fields:
            meta = FIELD_REGISTRY[field_key]
            value = pat.get(field_key)
            if isinstance(value, list):
                value = ", ".join(value) if value else "N/A"
            lines.append(f"  {meta['label']:<14}: {value or 'N/A'}")
        lines.append("-" * 50)
    return "\n".join(lines)

# =================================================
# STRUCTURED QUERY HANDLERS
# =================================================
def structured_asthma(patients: list[dict]) -> list[str]:
    """
    Returns list of names of patients with Asthma.
    """
    return list({
        pat["name"] for pat in patients
        if "asthma" in pat["raw_text"].lower()
        and pat["name"]
    })

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
    Score LLM answer confidence.
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
_REPORT_TRIGGERS = re.compile(
    r"\b(report|summary|overview|profile|list all|all patients)\b",
    re.IGNORECASE,
)

_LISTING_VERBS = re.compile(
    r"\b(give me|show me|what are|tell me|get me)\b",
    re.IGNORECASE,
)

_SINGLE_FIELD_TRIGGERS = {
    "asthma": (lambda q: "asthma" in q, structured_asthma, "Patients with Asthma"),
}

def _is_field_only_query(query: str) -> bool:
    q = query.lower()
    if re.search(r"\b(why|how|explain|describe|what is|what was|who is|tell me about|detail|common|occur|typical|usually|generally|often|cause|treat|prevent|risk|recommend)\b", q):
        return False

    has_verb  = bool(_LISTING_VERBS.search(q))
    has_field = any(
        kw in q
        for meta in FIELD_REGISTRY.values()
        for kw in meta["keywords"]
    )
    return has_verb and has_field

def classify_and_answer(query: str, patients: list[dict]) -> str | None:
    """
    Classifies intent and generates a deterministic report if matched.
    patients: list of dict representing patients from the database.
    """
    q = query.lower()

    # Intent 1: Explicit report/summary request
    if _REPORT_TRIGGERS.search(q):
        if re.search(r"\b(detail|history|explain|view|insight|about|overview of|breakdown)\b", q):
            return None
        fields = detect_requested_fields(query)
        return build_custom_report(patients, fields)

    # Intent 2: Specific condition lookups
    for _key, (predicate, handler, label) in _SINGLE_FIELD_TRIGGERS.items():
        if predicate(q):
            result = handler(patients)
            if result:
                return f"[{label}]\n" + "\n".join(f"  {v}" for v in result)

    # Intent 3: Field-only listing query
    if _is_field_only_query(query):
        fields = detect_requested_fields(query)
        return build_custom_report(patients, fields)

    return None