"""
FAISS Vector Store
==================
A lightweight wrapper around Facebook AI's FAISS library for storing
and searching 512-D ArcFace identity embeddings.

Why FAISS?
    - A brute-force Python loop over 100K vectors takes seconds.
    - FAISS uses optimised C++ / BLAS kernels and can search 100K vectors
      in < 1 millisecond on CPU.
    - For even larger databases (millions), FAISS supports approximate
      nearest-neighbour (ANN) indices like IVF and HNSW.

This module handles:
    1. Adding vectors + metadata to the index.
    2. Searching the index for the Top-K nearest neighbours.
    3. Saving / loading the index to / from disk for persistence.
"""

import json
import faiss
import numpy as np
from pathlib import Path

from . import config


class FaissVectorStore:
    """
    A FAISS-backed vector database for face identity embeddings.

    Internally uses IndexFlatIP (Inner Product), which computes the
    dot product between L2-normalised vectors. Because our ArcFace
    embeddings are already L2-normalised, dot product == cosine similarity.

    Metadata (identity label, image path, etc.) is stored in a parallel
    Python list, indexed by the same integer position as the FAISS index.
    """

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        """
        Create an empty vector store.

        Args:
            dim: Dimensionality of the vectors (512 for ArcFace).
        """
        self.dim = dim
        # IndexFlatIP = exact inner-product (cosine similarity for normalised vectors)
        # For millions of vectors, swap to faiss.IndexIVFFlat for speed.
        self.index = faiss.IndexFlatIP(dim)
        # Parallel metadata list: metadata[i] corresponds to the i-th vector in the index.
        self.metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, metadata_list: list[dict]) -> None:
        """
        Add vectors and their metadata to the store.

        Args:
            embeddings:    np.ndarray of shape (N, 512), float32, L2-normalised.
            metadata_list: list of N dicts, each containing at least:
                           {"identity": "n000123", "image_path": "path/to/img.jpg"}
        """
        assert embeddings.shape[0] == len(metadata_list), \
            f"Got {embeddings.shape[0]} vectors but {len(metadata_list)} metadata entries."
        assert embeddings.shape[1] == self.dim, \
            f"Expected {self.dim}-D vectors, got {embeddings.shape[1]}-D."

        # FAISS requires float32 and contiguous memory layout
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

    def search(self, query_embedding: np.ndarray, top_k: int = config.TOP_K) -> list[dict]:
        """
        Search the store for the Top-K nearest neighbours.

        Args:
            query_embedding: np.ndarray of shape (512,), L2-normalised.
            top_k:           Number of nearest neighbours to return.

        Returns:
            A list of dicts, each containing:
            {
                "identity":   "n000123",
                "image_path": "path/to/img.jpg",
                "similarity": 0.87,         # cosine similarity score
                "rank":       1              # 1-indexed rank
            }
        """
        # FAISS expects a 2-D query: (1, 512)
        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)

        # D = distances (cosine similarities), I = indices into the store
        D, I = self.index.search(query, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
            if idx == -1:
                continue  # FAISS returns -1 for unfilled slots
            entry = self.metadata[idx].copy()
            entry["similarity"] = float(score)
            entry["rank"] = rank
            results.append(entry)

        return results

    @property
    def total_vectors(self) -> int:
        """Number of vectors currently stored."""
        return self.index.ntotal

    # ------------------------------------------------------------------
    # Persistence: Save and Load the index + metadata to/from disk
    # ------------------------------------------------------------------
    def save(self, directory: str | Path = config.FAISS_INDEX_DIR) -> None:
        """
        Persist the FAISS index and metadata to disk.

        Saves two files:
            - index.faiss    : The binary FAISS index.
            - metadata.json  : The metadata list as JSON.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"[VectorStore] Saved {self.total_vectors} vectors to {directory}")

    def load(self, directory: str | Path = config.FAISS_INDEX_DIR) -> None:
        """
        Load a previously saved FAISS index and metadata from disk.
        """
        directory = Path(directory)
        index_path = directory / "index.faiss"
        meta_path = directory / "metadata.json"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No saved index found in {directory}")

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        print(f"[VectorStore] Loaded {self.total_vectors} vectors from {directory}")
