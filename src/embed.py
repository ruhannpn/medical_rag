from sentence_transformers import SentenceTransformer
import numpy as np

print("Script started...")

# Step 1: Create fake medical sentences
sentences = [
    "Patient diagnosed with Type 2 Diabetes.",
    "Prescribed Metformin 500mg twice daily.",
    "Blood pressure recorded at 140/90.",
    "No signs of cardiovascular disease.",
    "Recommended lifestyle modifications."
]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(sentences)

print("Embeddings generated!")

print("\nEmbedding shape:", embeddings.shape)

print("\nFirst sentence:")
print(sentences[0])

print("\nFirst embedding vector (first 10 values):")
print(embeddings[0][:10])