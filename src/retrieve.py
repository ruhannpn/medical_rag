import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample medical sentences
sentences = [
    "Patient diagnosed with Type 2 Diabetes.",
    "Prescribed Metformin 500mg twice daily.",
    "Blood pressure recorded at 140/90.",
    "No signs of cardiovascular disease.",
    "Recommended lifestyle modifications."
]

print("Generating embeddings...")
embeddings = model.encode(sentences)

# Convert to numpy array (FAISS requires float32)
embeddings = np.array(embeddings).astype("float32")

# Step 1: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Step 2: Add embeddings to index
index.add(embeddings)

print(f"Stored {index.ntotal} sentence vectors in FAISS index.")

# Step 3: Ask a query
query = "What medication was prescribed?"
print("\nQuery:", query)

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

# Step 4: Search top 2 similar sentences
k = 2
distances, indices = index.search(query_embedding, k)

print("\nTop matching sentences:")

for i in indices[0]:
    print("-", sentences[i])