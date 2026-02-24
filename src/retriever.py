import faiss
import numpy as np


def normalize_vectors(vectors):
    """
    Normalize vectors to unit length.
    Required for cosine similarity.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def build_faiss_index(embeddings):
    """
    Build FAISS index using cosine similarity.
    """
    # Normalize embeddings
    embeddings = normalize_vectors(embeddings)

    dimension = embeddings.shape[1]

    # Inner Product index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def search_index(index, query_embedding, k=3):
    """
    Search top-k most similar vectors using cosine similarity.
    """
    # Normalize query embedding
    query_embedding = normalize_vectors(query_embedding)

    scores, indices = index.search(query_embedding, k)
    return scores, indices