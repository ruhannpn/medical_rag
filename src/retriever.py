from __future__ import annotations
import os
import re
import shutil
import numpy as np
from rank_bm25 import BM25Okapi

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class RAGIndex:
    def __init__(self, index_dir="data/indices"):
        self.index_dir = index_dir
        self.vectorstore = None
        self.bm25 = None
        self.chunks = []  # list of {"content": str, "doc_id": str, "patient_id": int}
        self._embeddings_model = None
        os.makedirs(self.index_dir, exist_ok=True)

    @property
    def embeddings_model(self):
        if self._embeddings_model is None:
            # HuggingFaceEmbeddings wraps sentence-transformers all-MiniLM-L6-v2
            self._embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return self._embeddings_model

    def is_empty(self) -> bool:
        return self.vectorstore is None or len(self.chunks) == 0

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        m_min = np.min(arr)
        m_max = np.max(arr)
        diff = m_max - m_min
        if diff < 1e-8:
            return np.zeros_like(arr)
        return (arr - m_min) / (diff + 1e-8)

    def build_or_update(self, new_chunks: list[dict], embed_model=None):
        """
        Builds or appends new chunks to the FAISS and BM25 index using LangChain.
        new_chunks: list of dicts with keys "content", "doc_id", "patient_id"
        """
        if not new_chunks:
            return
            
        # Append new chunks to our local state list
        start_idx = len(self.chunks)
        self.chunks.extend(new_chunks)
        
        # Build/Update BM25 from all chunks
        texts = [c["content"] for c in self.chunks]
        tokenized_corpus = [re.findall(r"\w+", t.lower()) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Convert new chunks to LangChain Document objects
        lc_docs = []
        for i, c in enumerate(new_chunks):
            lc_docs.append(Document(
                page_content=c["content"],
                metadata={
                    "doc_id": c["doc_id"],
                    "patient_id": c["patient_id"],
                    "global_idx": start_idx + i
                }
            ))

        # Build or add to LangChain FAISS store
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(
                lc_docs,
                self.embeddings_model,
                distance_strategy="COSINE"
            )
        else:
            self.vectorstore.add_documents(lc_docs)

    def search(self, query: str, embed_model=None, top_k: int = 5, vector_weight: float = 0.6) -> list[int]:
        """
        Performs hybrid dense (LangChain FAISS) + sparse (BM25) search.
        Returns a list of global indices matching chunks in self.chunks.
        """
        if self.is_empty():
            return []

        n_chunks = len(self.chunks)
        effective_k = min(top_k, n_chunks)

        # 1. Dense Search: retrieve scores for all documents for normalization
        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=n_chunks)
        
        vector_scores = np.zeros(n_chunks)
        for doc, score in results_with_scores:
            idx = doc.metadata.get("global_idx")
            if idx is not None and idx < n_chunks:
                # Convert cosine distance (1 - cos_sim) to cosine similarity
                vector_scores[idx] = 1.0 - score

        # 2. Sparse Search
        tokenized_query = re.findall(r"\w+", query.lower())
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # 3. Normalization and Fusion
        norm_vector = self._normalize(vector_scores)
        norm_bm25 = self._normalize(bm25_scores)
        
        combined = vector_weight * norm_vector + (1 - vector_weight) * norm_bm25
        
        # Sort and return top indices
        top_indices = np.argsort(combined)[::-1][:effective_k]
        return top_indices.tolist()

    def get_chunk_scores(self, query: str, embed_model=None, indices: list[int] = None) -> np.ndarray:
        """
        Retrieves normalized vector similarity scores for given chunk indices.
        """
        if self.is_empty() or not indices:
            return np.zeros(len(indices or []))

        n_chunks = len(self.chunks)
        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=n_chunks)

        vector_scores = np.zeros(n_chunks)
        for doc, score in results_with_scores:
            idx = doc.metadata.get("global_idx")
            if idx is not None and idx < n_chunks:
                vector_scores[idx] = 1.0 - score

        norm_vector = self._normalize(vector_scores)
        return norm_vector[indices]

    def save(self, user_id: int):
        """
        Serializes the LangChain FAISS index to data/indices/{user_id}/.
        """
        if self.vectorstore is not None:
            folder_path = os.path.join(self.index_dir, str(user_id))
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path, ignore_errors=True)
            self.vectorstore.save_local(folder_path)

    def load(self, user_id: int, db_chunks: list[dict]):
        """
        Loads user index. Reconstructs BM25 index on the fly.
        """
        self.chunks = db_chunks
        
        if not self.chunks:
            self.vectorstore = None
            self.bm25 = None
            return

        # Reconstruct BM25 Okapi
        texts = [c["content"] for c in self.chunks]
        tokenized_corpus = [re.findall(r"\w+", t.lower()) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Load LangChain FAISS index
        folder_path = os.path.join(self.index_dir, str(user_id))
        if os.path.exists(folder_path):
            self.vectorstore = FAISS.load_local(
                folder_path,
                self.embeddings_model,
                allow_dangerous_deserialization=True
            )
        else:
            self.vectorstore = None

    def clear(self, user_id: int):
        """
        Clears memory state and deletes local files.
        """
        self.vectorstore = None
        self.bm25 = None
        self.chunks = []
        folder_path = os.path.join(self.index_dir, str(user_id))
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path, ignore_errors=True)
            except Exception as e:
                print(f"[Warning] Failed to delete FAISS index folder for user {user_id}: {e}")