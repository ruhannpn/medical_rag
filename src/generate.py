import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------
# Load embedding model
# -----------------------------
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Sample medical sentences
# -----------------------------
sentences = [
    "Patient diagnosed with Type 2 Diabetes.",
    "Prescribed Metformin 500mg twice daily.",
    "Blood pressure recorded at 140/90.",
    "No signs of cardiovascular disease.",
    "Recommended lifestyle modifications."
]

# -----------------------------
# Generate embeddings
# -----------------------------
embeddings = embed_model.encode(sentences)
embeddings = np.array(embeddings).astype("float32")

# -----------------------------
# Create FAISS index
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"Stored {index.ntotal} sentence vectors in FAISS index.")

# -----------------------------
# User Query
# -----------------------------
query = "What medication was prescribed?"
print("\nUser Query:", query)

query_embedding = embed_model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

# -----------------------------
# Retrieve top-k similar chunks
# -----------------------------
k = 2
distances, indices = index.search(query_embedding, k)

retrieved_context = ""
print("\nRetrieved Context:")
for i in indices[0]:
    print("-", sentences[i])
    retrieved_context += sentences[i] + "\n"

# -----------------------------
# AUGMENTATION
# -----------------------------
prompt = f"""
Use the context below to answer the question.

Context:
{retrieved_context}

Question:
{query}
"""

# -----------------------------
# LOCAL LLM (FLAN-T5)
# -----------------------------
print("\nLoading local LLM...")
generator = pipeline("text-generation", model="google/flan-t5-base")

print("\nGenerating answer...")
response = generator(prompt, max_length=100)

print("\nFinal Answer:")
print(response[0]["generated_text"])