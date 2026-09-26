import json
import logging
import faiss
import numpy as np
from pathlib import Path

from ..core import config
from ..interfaces.base_vector_store import BaseVectorStore
from ..core.exceptions import (
    VectorDimensionMismatchError,
    VectorCountMismatchError,
    IndexNotFoundError,
    EmptyIndexError,
)

logger = logging.getLogger(__name__)

class FaissVectorStore(BaseVectorStore):
    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._metadata = []
        logger.info(f"Initialized FAISS vector store (dimension: {dim})")

    def add(self, embeddings: np.ndarray, metadata_list: list[dict]) -> None:
        logger.info(f"Adding {embeddings.shape[0]} vectors to FAISS index...")
        if embeddings.shape[0] != len(metadata_list):
            raise VectorCountMismatchError(f"Mismatch: {embeddings.shape[0]} vectors, {len(metadata_list)} metadata entries.")
        if embeddings.shape[1] != self._dim:
            raise VectorDimensionMismatchError(f"Expected {self._dim}-D, got {embeddings.shape[1]}-D.")

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(embeddings)
        self._metadata.extend(metadata_list)
        logger.info(f"Index now contains {self.total_vectors} vectors.")

    def search(self, query_embedding: np.ndarray, top_k: int = config.TOP_K) -> list[dict]:
        logger.debug(f"Searching FAISS index for top {top_k} matches...")
        if self.total_vectors == 0:
            raise EmptyIndexError("FAISS index is empty.")

        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        distances, indices = self._index.search(query, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
            if idx != -1:
                entry = self._metadata[idx].copy()
                entry.update({"similarity": float(score), "rank": rank})
                results.append(entry)

        logger.debug(f"Search returned {len(results)} matches.")
        return results

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    def save(self, directory: str | Path = config.FAISS_INDEX_DIR) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving {self.total_vectors} vectors to {directory}...")
        faiss.write_index(self._index, str(directory / "index.faiss"))
        with open(directory / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2)

    def load(self, directory: str | Path = config.FAISS_INDEX_DIR) -> None:
        directory = Path(directory)
        logger.info(f"Loading FAISS index from {directory}...")
        index_path = directory / "index.faiss"
        meta_path = directory / "metadata.json"
        
        if not index_path.exists() or not meta_path.exists():
            raise IndexNotFoundError("Index or metadata file missing.")

        self._index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)
        logger.info(f"Loaded {self.total_vectors} vectors successfully.")