from sentence_transformers import SentenceTransformer
import numpy as np


def load_embedding_model():
    """
    Load sentence transformer model.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def generate_embeddings(model, texts):
    """
    Convert list of texts into float32 embeddings.
    """
    embeddings = model.encode(texts)
    return np.array(embeddings).astype("float32")