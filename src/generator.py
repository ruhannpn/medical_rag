from ingest import extract_text_from_pdf
from chunking import chunk_text_by_sections
from embedding import load_embedding_model, generate_embeddings
from retriever import build_faiss_index, search_index
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_local_llm():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def generate_answer(tokenizer, model, prompt, max_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()


def compute_confidence(score):
    if score > 0.30:
        return "High"
    elif score > 0.15:
        return "Medium"
    else:
        return "Low"


def detect_relevant_section(query):
    query_lower = query.lower()

    if "prescribe" in query_lower or "medication" in query_lower:
        return "Prescription"
    elif "diagnosis" in query_lower:
        return "Diagnosis"
    elif "age" in query_lower:
        return "Patient Information"
    elif "lab" in query_lower or "blood" in query_lower:
        return "Laboratory Results"
    elif "recommend" in query_lower:
        return "Recommendations"
    else:
        return None


def is_summary_query(query):
    query_lower = query.lower()
    summary_keywords = ["summary", "summarize", "overview", "brief"]
    return any(keyword in query_lower for keyword in summary_keywords)


def main():
    pdf_path = "../data/sample_medical_report.pdf"

    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)

    print("Splitting document by sections...")
    section_chunks = chunk_text_by_sections(raw_text)
    print(f"Total sections found: {len(section_chunks)}")

    texts = [chunk["content"] for chunk in section_chunks]

    print("Loading embedding model...")
    embed_model = load_embedding_model()

    print("Generating embeddings...")
    embeddings = generate_embeddings(embed_model, texts)

    print("Building FAISS index (Cosine Similarity)...")
    index = build_faiss_index(embeddings)

    print("Loading local LLM...")
    tokenizer, model = load_local_llm()

    print("\nRAG system ready. Type 'exit' to quit.\n")

    while True:
        query = input("Ask a medical question: ")

        if query.lower() == "exit":
            print("Exiting...")
            break

        # ===============================
        # 🔥 STRUCTURED SUMMARY MODE
        # ===============================
        if is_summary_query(query):
            print("\nDetected Summary Query — Using hierarchical summarization.\n")

            section_summaries = []

            for chunk in section_chunks:
                section_name = chunk["section"]
                content = chunk["content"]

                summary_prompt = f"""
Summarize this medical section in 1 concise sentence.

Section: {section_name}

Content:
{content}

Summary:
"""
                section_summary = generate_answer(
                    tokenizer, model, summary_prompt, max_tokens=60
                )

                section_summaries.append(
                    f"{section_name}: {section_summary}"
                )

            combined_prompt = f"""
Combine the following section summaries into a clear overall
medical report summary in 4-5 sentences.

Section Summaries:
{chr(10).join(section_summaries)}

Final Summary:
"""

            final_summary = generate_answer(
                tokenizer, model, combined_prompt, max_tokens=200
            )

            print("=== Structured Summary Output ===")
            print(final_summary)
            print("-" * 60)
            continue

        # ===============================
        # 🔥 FACT-BASED RETRIEVAL MODE
        # ===============================
        detected_section = detect_relevant_section(query)

        if detected_section:
            filtered_chunks = [
                chunk for chunk in section_chunks
                if chunk["section"] == detected_section
            ]

            filtered_texts = [chunk["content"] for chunk in filtered_chunks]
            filtered_embeddings = generate_embeddings(embed_model, filtered_texts)

            temp_index = build_faiss_index(filtered_embeddings)
            query_embedding = generate_embeddings(embed_model, [query])
            scores, indices = search_index(temp_index, query_embedding, k=1)

            retrieved_section = filtered_chunks[indices[0][0]]["section"]
            retrieved_context = filtered_chunks[indices[0][0]]["content"]
            top_score = scores[0][0]

        else:
            query_embedding = generate_embeddings(embed_model, [query])
            scores, indices = search_index(index, query_embedding, k=1)

            retrieved_section = section_chunks[indices[0][0]]["section"]
            retrieved_context = section_chunks[indices[0][0]]["content"]
            top_score = scores[0][0]

        confidence = compute_confidence(top_score)

        print("\n--- Retrieval Result ---")
        print(f"Source Section: {retrieved_section}")
        print(f"Similarity Score: {round(top_score, 3)}")
        print(f"Confidence Level: {confidence}")

        if confidence == "Low":
            print("\n⚠️ Insufficient confidence to generate reliable answer.")
            print("Please rephrase your question.\n")
            print("-" * 60)
            continue

        prompt = f"""
Answer using ONLY the provided context.
If the answer is not explicitly stated, reply:
"Not found in document."

Context:
{retrieved_context}

Question:
{query}

Answer:
"""

        print("\nGenerating answer...\n")
        final_answer = generate_answer(tokenizer, model, prompt)

        print("=== Final Structured Output ===")
        print(f"Answer: {final_answer}")
        print(f"Confidence: {confidence}")
        print(f"Source Section: {retrieved_section}")
        print("-" * 60)


if __name__ == "__main__":
    main()